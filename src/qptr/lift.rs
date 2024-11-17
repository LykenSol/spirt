//! [`QPtr`](crate::TypeKind::QPtr) lifting to typed memory (e.g. to SPIR-V).
//
// FIXME(eddyb) the `legalize`-vs-`analyze`+`lift` split can be confusing,
// and may need more than documentation (but for now, see `qptr::legalize` docs).

// HACK(eddyb) sharing layout code with other modules.
use super::layout::*;

use crate::func_at::{FuncAt, FuncAtMut};
use crate::qptr::{
    QPtrAttr, QPtrMemUsage, QPtrMemUsageFlags, QPtrMemUsageKind, QPtrOp, QPtrUsage, shapes,
};
use crate::transform::{InnerInPlaceTransform, InnerTransform, Transformed, Transformer};
use crate::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstKind, DeclDef, Diag, DiagLevel, EntityDefs, EntityOrientedDenseMap, Func,
    FuncDecl, FxIndexMap, GlobalVar, GlobalVarDecl, Module, Node, NodeKind, NodeOutputDecl, Region,
    Type, TypeDef, TypeKind, TypeOrConst, Value, scalar, spv,
};
use smallvec::SmallVec;
use std::mem;
use std::num::NonZeroU32;
use std::rc::Rc;

struct LiftError(Diag);

/// Context for lifting `QPtr`s to SPIR-V `OpTypePointer`s.
///
/// See also `passes::qptr::lift_to_spv_ptrs` (which drives this).
pub struct LiftToSpvPtrs<'a> {
    cx: Rc<Context>,
    wk: &'static spv::spec::WellKnown,
    layout_cache: LayoutCache<'a>,
}

impl<'a> LiftToSpvPtrs<'a> {
    pub fn new(cx: Rc<Context>, layout_config: &'a LayoutConfig) -> Self {
        Self {
            cx: cx.clone(),
            wk: &spv::spec::Spec::get().well_known,
            layout_cache: LayoutCache::new(cx, layout_config),
        }
    }

    pub fn lift_global_var(&self, global_var_decl: &mut GlobalVarDecl) {
        match self.spv_ptr_type_and_addr_space_for_global_var(global_var_decl) {
            Ok((spv_ptr_type, addr_space)) => {
                global_var_decl.attrs = self.strip_qptr_usage_attr(global_var_decl.attrs);
                global_var_decl.type_of_ptr_to = spv_ptr_type;
                global_var_decl.addr_space = addr_space;
                global_var_decl.shape = None;
            }
            Err(LiftError(e)) => {
                global_var_decl.attrs.push_diag(&self.cx, e);
            }
        }
        // FIXME(eddyb) if globals have initializers pointing at other globals,
        // here is where they might get fixed up, but that usage is illegal so
        // likely needs to get legalized on `qptr`s, before here.
    }

    pub fn lift_all_funcs(&self, module: &mut Module, funcs: impl IntoIterator<Item = Func>) {
        for func in funcs {
            LiftToSpvPtrInstsInFunc {
                lifter: self,
                global_vars: &module.global_vars,

                parent_region: None,

                deferred_ptr_noops: Default::default(),
                data_inst_use_counts: Default::default(),

                func_has_qptr_analysis_bug_diags: false,
            }
            .in_place_transform_func_decl(&mut module.funcs[func]);
        }
    }

    fn find_qptr_usage_attr(&self, attrs: AttrSet) -> Option<&QPtrUsage> {
        self.cx[attrs].attrs.iter().find_map(|attr| match attr {
            Attr::QPtr(QPtrAttr::Usage(usage)) => Some(&usage.0),
            _ => None,
        })
    }

    fn require_qptr_usage_attr(&self, attrs: AttrSet) -> Result<&QPtrUsage, LiftError> {
        self.find_qptr_usage_attr(attrs)
            .ok_or_else(|| LiftError(Diag::bug(["missing `qptr.usage` attribute".into()])))
    }

    fn strip_qptr_usage_attr(&self, attrs: AttrSet) -> AttrSet {
        self.cx.intern(AttrSetDef {
            attrs: self.cx[attrs]
                .attrs
                .iter()
                .filter(|attr| !matches!(attr, Attr::QPtr(QPtrAttr::Usage(_))))
                .cloned()
                .collect(),
        })
    }

    fn spv_ptr_type_and_addr_space_for_global_var(
        &self,
        global_var_decl: &GlobalVarDecl,
    ) -> Result<(Type, AddrSpace), LiftError> {
        let wk = self.wk;

        let qptr_usage = self.require_qptr_usage_attr(global_var_decl.attrs)?;

        let shape =
            global_var_decl.shape.ok_or_else(|| LiftError(Diag::bug(["missing shape".into()])))?;
        let (storage_class, pointee_type) = match (global_var_decl.addr_space, shape) {
            (AddrSpace::Handles, shapes::GlobalVarShape::Handles { handle, fixed_count }) => {
                let (storage_class, handle_type) = match handle {
                    shapes::Handle::Opaque(ty) => {
                        if self.pointee_type_for_usage(qptr_usage)? != ty {
                            return Err(LiftError(Diag::bug([
                                "mismatched opaque handle types in `qptr.usage` vs `shape`".into(),
                            ])));
                        }
                        (wk.UniformConstant, ty)
                    }
                    // FIXME(eddyb) validate usage against `buf` and/or expand
                    // the type to make sure it has the right size.
                    shapes::Handle::Buffer(AddrSpace::SpvStorageClass(storage_class), _buf) => {
                        (storage_class, self.pointee_type_for_usage(qptr_usage)?)
                    }
                    shapes::Handle::Buffer(AddrSpace::Handles, _) => {
                        return Err(LiftError(Diag::bug([
                            "invalid `AddrSpace::Handles` in `Handle::Buffer`".into(),
                        ])));
                    }
                };
                (
                    storage_class,
                    if fixed_count == Some(NonZeroU32::new(1).unwrap()) {
                        handle_type
                    } else {
                        self.spv_op_type_array(handle_type, fixed_count.map(|c| c.get()), None)?
                    },
                )
            }
            // FIXME(eddyb) validate usage against `layout` and/or expand
            // the type to make sure it has the right size.
            (
                AddrSpace::SpvStorageClass(storage_class),
                shapes::GlobalVarShape::UntypedData(_layout),
            ) => (storage_class, self.pointee_type_for_usage(qptr_usage)?),
            (
                AddrSpace::SpvStorageClass(storage_class),
                shapes::GlobalVarShape::TypedInterface(ty),
            ) => (storage_class, ty),

            (
                AddrSpace::Handles,
                shapes::GlobalVarShape::UntypedData(_) | shapes::GlobalVarShape::TypedInterface(_),
            )
            | (AddrSpace::SpvStorageClass(_), shapes::GlobalVarShape::Handles { .. }) => {
                return Err(LiftError(Diag::bug(["mismatched `addr_space` and `shape`".into()])));
            }
        };
        let addr_space = AddrSpace::SpvStorageClass(storage_class);
        Ok((self.spv_ptr_type(addr_space, pointee_type), addr_space))
    }

    /// Returns `Some` iff `ty` is a SPIR-V `OpTypePointer`.
    //
    // FIXME(eddyb) deduplicate with `qptr::lower`.
    fn as_spv_ptr_type(&self, ty: Type) -> Option<(AddrSpace, Type)> {
        match &self.cx[ty].kind {
            TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
                if spv_inst.opcode == self.wk.OpTypePointer =>
            {
                let sc = match spv_inst.imms[..] {
                    [spv::Imm::Short(_, sc)] => sc,
                    _ => unreachable!(),
                };
                let pointee = match type_and_const_inputs[..] {
                    [TypeOrConst::Type(elem_type)] => elem_type,
                    _ => unreachable!(),
                };
                Some((AddrSpace::SpvStorageClass(sc), pointee))
            }
            _ => None,
        }
    }

    fn spv_ptr_type(&self, addr_space: AddrSpace, pointee_type: Type) -> Type {
        let wk = self.wk;

        let storage_class = match addr_space {
            AddrSpace::Handles => unreachable!(),
            AddrSpace::SpvStorageClass(storage_class) => storage_class,
        };
        self.cx.intern(
            spv::Inst {
                opcode: wk.OpTypePointer,
                imms: [spv::Imm::Short(wk.StorageClass, storage_class)].into_iter().collect(),
            }
            .into_canonical_type_with(
                &self.cx,
                [TypeOrConst::Type(pointee_type)].into_iter().collect(),
            ),
        )
    }

    fn pointee_type_for_usage(&self, usage: &QPtrUsage) -> Result<Type, LiftError> {
        let wk = self.wk;

        match usage {
            &QPtrUsage::Handles(shapes::Handle::Opaque(ty)) => Ok(ty),
            QPtrUsage::Handles(shapes::Handle::Buffer(_, data_usage)) => {
                let attr_spv_decorate_block = Attr::SpvAnnotation(spv::Inst {
                    opcode: wk.OpDecorate,
                    imms: [spv::Imm::Short(wk.Decoration, wk.Block)].into_iter().collect(),
                });
                match &data_usage.kind {
                    QPtrMemUsageKind::Unused => {
                        self.spv_op_type_struct([], [attr_spv_decorate_block])
                    }
                    QPtrMemUsageKind::OffsetBase(fields) => self.spv_op_type_struct(
                        fields.iter().map(|(&field_offset, field_usage)| {
                            Ok((
                                field_offset,
                                self.pointee_type_for_mem_usage(field_usage, data_usage.flags)?,
                            ))
                        }),
                        [attr_spv_decorate_block],
                    ),
                    QPtrMemUsageKind::StrictlyTyped(_)
                    | QPtrMemUsageKind::DirectAccess(_)
                    | QPtrMemUsageKind::DynOffsetBase { .. } => self.spv_op_type_struct(
                        [Ok((
                            0,
                            self.pointee_type_for_mem_usage(
                                data_usage,
                                QPtrMemUsageFlags::empty(),
                            )?,
                        ))],
                        [attr_spv_decorate_block],
                    ),
                }
            }
            QPtrUsage::Memory(usage) => {
                self.pointee_type_for_mem_usage(usage, QPtrMemUsageFlags::empty())
            }
        }
    }

    fn pointee_type_for_mem_usage(
        &self,
        usage: &QPtrMemUsage,
        outer_effective_flags: QPtrMemUsageFlags,
    ) -> Result<Type, LiftError> {
        // FIXME(eddyb) does this make sense across all flags?
        let effective_flags = outer_effective_flags | usage.flags;

        // TODO(eddyb) implement (or at least validate that there are no gaps).
        if effective_flags.contains(QPtrMemUsageFlags::COPY_SRC_AND_DST) {
            return Err(LiftError(Diag::bug([
                "unimplemented `qptr.copy` src+dst (gap filling)".into()
            ])));
        }

        match &usage.kind {
            QPtrMemUsageKind::Unused => self.spv_op_type_struct([], []),
            &QPtrMemUsageKind::StrictlyTyped(ty) | &QPtrMemUsageKind::DirectAccess(ty) => Ok(ty),
            QPtrMemUsageKind::OffsetBase(fields) => {
                // HACK(eddyb) force the size of `OpTypeStruct`s that would be
                // otherwise undersized (as e.g. `qptr.copy` src/dst).
                let size_forcing_zst_tail_field = usage
                    .max_size
                    .filter(|&size| {
                        let inherent_unaligned_size =
                            fields.last_key_value().map_or(0, |(&field_offset, field_usage)| {
                                field_offset.checked_add(field_usage.max_size.unwrap()).unwrap()
                            });
                        size > inherent_unaligned_size
                    })
                    .map(|size| Ok((size, self.spv_op_type_struct([], [])?)));

                self.spv_op_type_struct(
                    fields
                        .iter()
                        .map(|(&field_offset, field_usage)| {
                            Ok((
                                field_offset,
                                self.pointee_type_for_mem_usage(field_usage, effective_flags)?,
                            ))
                        })
                        .chain(size_forcing_zst_tail_field),
                    [],
                )
            }
            QPtrMemUsageKind::DynOffsetBase { element, stride } => {
                let element_type = self.pointee_type_for_mem_usage(element, effective_flags)?;

                let fixed_len = usage
                    .max_size
                    .map(|size| {
                        if size % stride.get() != 0 {
                            return Err(LiftError(Diag::bug([format!(
                                "DynOffsetBase: size ({size}) not a multiple of stride ({stride})"
                            )
                            .into()])));
                        }
                        Ok(size / stride.get())
                    })
                    .transpose()?;

                self.spv_op_type_array(element_type, fixed_len, Some(*stride))
            }
        }
    }

    fn spv_op_type_array(
        &self,
        element_type: Type,
        fixed_len: Option<u32>,
        stride: Option<NonZeroU32>,
    ) -> Result<Type, LiftError> {
        let wk = self.wk;

        let stride_attrs = stride.map(|stride| {
            self.cx.intern(AttrSetDef {
                attrs: [Attr::SpvAnnotation(spv::Inst {
                    opcode: wk.OpDecorate,
                    imms: [
                        spv::Imm::Short(wk.Decoration, wk.ArrayStride),
                        spv::Imm::Short(wk.LiteralInteger, stride.get()),
                    ]
                    .into_iter()
                    .collect(),
                })]
                .into(),
            })
        });

        let spv_opcode = if fixed_len.is_some() { wk.OpTypeArray } else { wk.OpTypeRuntimeArray };

        Ok(self.cx.intern(TypeDef {
            attrs: stride_attrs.unwrap_or_default(),
            kind: spv::Inst::from(spv_opcode).into_canonical_type_with(
                &self.cx,
                [
                    Some(TypeOrConst::Type(element_type)),
                    fixed_len.map(|len| {
                        TypeOrConst::Const(self.cx.intern(scalar::Const::from_u32(len)))
                    }),
                ]
                .into_iter()
                .flatten()
                .collect(),
            ),
        }))
    }

    fn spv_op_type_struct(
        &self,
        field_offsets_and_types: impl IntoIterator<Item = Result<(u32, Type), LiftError>>,
        extra_attrs: impl IntoIterator<Item = Attr>,
    ) -> Result<Type, LiftError> {
        let wk = self.wk;

        let field_offsets_and_types = field_offsets_and_types.into_iter();
        let mut attrs = AttrSetDef::default();
        let mut type_and_const_inputs =
            SmallVec::with_capacity(field_offsets_and_types.size_hint().0);
        for (i, field_offset_and_type) in field_offsets_and_types.enumerate() {
            let (offset, field_type) = field_offset_and_type?;
            attrs.attrs.insert(Attr::SpvAnnotation(spv::Inst {
                opcode: wk.OpMemberDecorate,
                imms: [
                    spv::Imm::Short(wk.LiteralInteger, i.try_into().unwrap()),
                    spv::Imm::Short(wk.Decoration, wk.Offset),
                    spv::Imm::Short(wk.LiteralInteger, offset),
                ]
                .into_iter()
                .collect(),
            }));
            type_and_const_inputs.push(TypeOrConst::Type(field_type));
        }
        attrs.attrs.extend(extra_attrs);
        Ok(self.cx.intern(TypeDef {
            attrs: self.cx.intern(attrs),
            kind: spv::Inst::from(wk.OpTypeStruct)
                .into_canonical_type_with(&self.cx, type_and_const_inputs),
        }))
    }

    /// Attempt to compute a `TypeLayout` for a given (SPIR-V) `Type`.
    fn layout_of(&self, ty: Type) -> Result<TypeLayout, LiftError> {
        self.layout_cache.layout_of(ty).map_err(|LayoutError(err)| LiftError(err))
    }
}

struct LiftToSpvPtrInstsInFunc<'a> {
    lifter: &'a LiftToSpvPtrs<'a>,
    global_vars: &'a EntityDefs<GlobalVar>,

    parent_region: Option<Region>,

    /// Some `QPtr`->`QPtr` `QPtrOp`s must be noops in SPIR-V, but because some
    /// of them have meaningful semantic differences in SPIR-T, replacement of
    /// their uses must be deferred until after `try_lift_data_inst_def` has had
    /// a chance to observe the distinction.
    ///
    /// E.g. `QPtrOp::BufferData`s cannot adjust the SPIR-V pointer type, due to
    /// interactions between the `Block` annotation and any potential trailing
    /// `OpTypeRuntimeArray`s (which cannot be nested in non-`Block` structs).
    ///
    /// The `QPtrOp` itself is only removed after the entire function is lifted,
    /// (using `data_inst_use_counts` to determine whether they're truly unused).
    deferred_ptr_noops: FxIndexMap<DataInst, DeferredPtrNoop>,

    // FIXME(eddyb) consider removing this and just do a full second traversal.
    data_inst_use_counts: EntityOrientedDenseMap<DataInst, NonZeroU32>,

    // HACK(eddyb) this is used to avoid noise when `qptr::analyze` failed.
    func_has_qptr_analysis_bug_diags: bool,
}

struct DeferredPtrNoop {
    output_pointer: Value,

    output_pointer_addr_space: AddrSpace,

    /// Should be equivalent to `layout_of` on `output_pointer`'s pointee type,
    /// except in the case of `QPtrOp::BufferData`.
    output_pointee_layout: TypeLayout,

    parent_region: Region,
}

impl LiftToSpvPtrInstsInFunc<'_> {
    // FIXME(eddyb) maybe all this data should be packaged up together in a
    // type with fields like those of `DeferredPtrNoop` (or even more).
    fn type_of_val_as_spv_ptr_with_layout(
        &self,
        func_at_value: FuncAt<'_, Value>,
    ) -> Result<(AddrSpace, TypeLayout), LiftError> {
        let v = func_at_value.position;

        if let Value::NodeOutput { node: v_data_inst, output_idx: 0 } = v {
            if let Some(ptr_noop) = self.deferred_ptr_noops.get(&v_data_inst) {
                return Ok((
                    ptr_noop.output_pointer_addr_space,
                    ptr_noop.output_pointee_layout.clone(),
                ));
            }
        }

        let (addr_space, pointee_type) = self
            .lifter
            .as_spv_ptr_type(func_at_value.type_of(&self.lifter.cx))
            .ok_or_else(|| LiftError(Diag::bug(["pointer input not an `OpTypePointer`".into()])))?;

        Ok((addr_space, self.lifter.layout_of(pointee_type)?))
    }

    fn try_lift_data_inst_def(
        &mut self,
        mut func_at_data_inst: FuncAtMut<'_, DataInst>,
    ) -> Result<Transformed<DataInstDef>, LiftError> {
        let wk = self.lifter.wk;
        let cx = &self.lifter.cx;

        let func_at_data_inst_frozen = func_at_data_inst.reborrow().freeze();
        let data_inst = func_at_data_inst_frozen.position;
        let data_inst_def = func_at_data_inst_frozen.def();
        let func = func_at_data_inst_frozen.at(());
        let type_of_val = |v: Value| func.at(v).type_of(cx);

        // FIXME(eddyb) this should be a method on some sort of "cursor" type.
        let insert_aux_data_inst =
            |this: &mut Self, func: FuncAtMut<'_, ()>, mut aux_data_inst_def: DataInstDef| {
                // HACK(eddyb) account for `deferred_ptr_noops` interactions.
                this.resolve_deferred_ptr_noop_uses(&mut aux_data_inst_def.inputs);
                this.add_value_uses(&aux_data_inst_def.inputs);

                let aux_data_inst = func.nodes.define(cx, aux_data_inst_def.into());

                // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                // due to the need to borrow `regions` and `nodes`
                // at the same time - perhaps some kind of `FuncAtMut` position
                // types for "where a list is in a parent entity" could be used
                // to make this more ergonomic, although the potential need for
                // an actual list entity of its own, should be considered.
                func.regions[this.parent_region.unwrap()].children.insert_before(
                    aux_data_inst,
                    data_inst,
                    func.nodes,
                );

                aux_data_inst
            };

        let replacement_data_inst_def = match &data_inst_def.kind {
            NodeKind::FuncCall { .. }
            | NodeKind::Select(_)
            | NodeKind::Loop { .. }
            | NodeKind::ExitInvocation(_) => {
                return Ok(Transformed::Unchanged);
            }

            DataInstKind::Scalar(_) | DataInstKind::Vector(_) => return Ok(Transformed::Unchanged),

            DataInstKind::QPtr(QPtrOp::FuncLocalVar(_mem_layout)) => {
                let output_qptr_usage =
                    self.lifter.require_qptr_usage_attr(data_inst_def.outputs[0].attrs)?;

                // FIXME(eddyb) validate against `mem_layout`!
                let pointee_type = self.lifter.pointee_type_for_usage(output_qptr_usage)?;

                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.kind = DataInstKind::SpvInst(
                    spv::Inst {
                        opcode: wk.OpVariable,
                        imms: [spv::Imm::Short(wk.StorageClass, wk.Function)].into_iter().collect(),
                    },
                    spv::InstLowering::default(),
                );
                data_inst_def.outputs[0].attrs =
                    self.lifter.strip_qptr_usage_attr(data_inst_def.outputs[0].attrs);
                data_inst_def.outputs[0].ty =
                    self.lifter.spv_ptr_type(AddrSpace::SpvStorageClass(wk.Function), pointee_type);
                data_inst_def
            }
            DataInstKind::QPtr(QPtrOp::HandleArrayIndex) => {
                let (addr_space, layout) =
                    self.type_of_val_as_spv_ptr_with_layout(func.at(data_inst_def.inputs[0]))?;
                let handle = match layout {
                    // FIXME(eddyb) standardize variant order in enum/match.
                    TypeLayout::HandleArray(handle, _) => handle,
                    TypeLayout::Handle(_) => {
                        return Err(LiftError(Diag::bug(["cannot index single Handle".into()])));
                    }
                    TypeLayout::Concrete(_) => {
                        return Err(LiftError(Diag::bug(
                            ["cannot index memory as handles".into()],
                        )));
                    }
                };
                let handle_type = match handle {
                    shapes::Handle::Opaque(ty) => ty,
                    shapes::Handle::Buffer(_, buf) => buf.original_type,
                };

                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.kind =
                    DataInstKind::SpvInst(wk.OpAccessChain.into(), spv::InstLowering::default());
                data_inst_def.outputs[0].attrs =
                    self.lifter.strip_qptr_usage_attr(data_inst_def.outputs[0].attrs);
                data_inst_def.outputs[0].ty = self.lifter.spv_ptr_type(addr_space, handle_type);
                data_inst_def
            }
            DataInstKind::QPtr(QPtrOp::BufferData) => {
                let buf_ptr = data_inst_def.inputs[0];
                let (addr_space, buf_layout) =
                    self.type_of_val_as_spv_ptr_with_layout(func.at(buf_ptr))?;

                let buf_data_layout = match buf_layout {
                    TypeLayout::Handle(shapes::Handle::Buffer(_, buf)) => TypeLayout::Concrete(buf),
                    _ => return Err(LiftError(Diag::bug(["non-Buffer pointee".into()]))),
                };

                self.deferred_ptr_noops.insert(data_inst, DeferredPtrNoop {
                    output_pointer: buf_ptr,
                    output_pointer_addr_space: addr_space,
                    output_pointee_layout: buf_data_layout,
                    parent_region: self.parent_region.unwrap(),
                });

                // FIXME(eddyb) avoid the repeated call to `type_of_val`,
                // maybe don't even replace the `QPtrOp::BufferData` instruction?
                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.outputs[0].ty = type_of_val(buf_ptr);
                data_inst_def
            }
            &DataInstKind::QPtr(QPtrOp::BufferDynLen { fixed_base_size, dyn_unit_stride }) => {
                let buf_ptr = data_inst_def.inputs[0];
                let (_, buf_layout) = self.type_of_val_as_spv_ptr_with_layout(func.at(buf_ptr))?;

                let buf_data_layout = match buf_layout {
                    TypeLayout::Handle(shapes::Handle::Buffer(_, buf)) => buf,
                    _ => return Err(LiftError(Diag::bug(["non-Buffer pointee".into()]))),
                };

                let field_idx = match &buf_data_layout.components {
                    Components::Fields { offsets, layouts }
                        if offsets.last() == Some(&fixed_base_size)
                            && layouts.last().map_or(false, |last_field| {
                                last_field.mem_layout.fixed_base.size == 0
                                    && last_field.mem_layout.dyn_unit_stride
                                        == Some(dyn_unit_stride)
                                    && matches!(last_field.components, Components::Elements {
                                        fixed_len: None,
                                        ..
                                    })
                            }) =>
                    {
                        u32::try_from(offsets.len() - 1).unwrap()
                    }
                    // FIXME(eddyb) support/diagnose more cases.
                    _ => {
                        return Err(LiftError(Diag::bug([
                            "buffer data type shape mismatch".into()
                        ])));
                    }
                };

                DataInstDef {
                    kind: DataInstKind::SpvInst(
                        spv::Inst {
                            opcode: wk.OpArrayLength,
                            imms: [spv::Imm::Short(wk.LiteralInteger, field_idx)]
                                .into_iter()
                                .collect(),
                        },
                        spv::InstLowering::default(),
                    ),
                    ..data_inst_def.clone()
                }
            }
            &DataInstKind::QPtr(QPtrOp::Offset(offset)) => {
                let mut data_inst_def = data_inst_def.clone();

                let output_qptr_usage = self
                    .lifter
                    .find_qptr_usage_attr(data_inst_def.outputs[0].attrs)
                    .unwrap_or(&QPtrUsage::Memory(QPtrMemUsage::UNUSED));

                let mut func = func_at_data_inst.reborrow().at(());
                let (output_pointer, (output_pointer_addr_space, output_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_usage(
                        data_inst_def.inputs[0],
                        offset,
                        output_qptr_usage,
                        func.reborrow(),
                        insert_aux_data_inst,
                    )?;
                // FIXME(eddyb) not being able to reuse the original `DataInst`
                // is a bit ridiculous, but correctly doing that would complicate
                // `adjust_pointer_for_offset_and_usage` in general.
                self.deferred_ptr_noops.insert(data_inst, DeferredPtrNoop {
                    output_pointer,
                    output_pointer_addr_space,
                    output_pointee_layout,
                    parent_region: self.parent_region.unwrap(),
                });
                // FIXME(eddyb) avoid the repeated call to `type_of_val`,
                // maybe don't even replace the `QPtrOp::Offset` instruction?
                data_inst_def.kind = QPtrOp::Offset(0).into();
                data_inst_def.outputs[0].ty = func.freeze().at(output_pointer).type_of(cx);
                data_inst_def
            }
            DataInstKind::QPtr(QPtrOp::DynOffset { stride, index_bounds }) => {
                let (stride, index_bounds) = (*stride, index_bounds.clone());
                let data_inst_def = data_inst_def.clone();

                let output_qptr_usage = self
                    .lifter
                    .find_qptr_usage_attr(data_inst_def.outputs[0].attrs)
                    .unwrap_or(&QPtrUsage::Memory(QPtrMemUsage::UNUSED));

                let strided_qptr_usage = QPtrUsage::Memory(QPtrMemUsage {
                    // FIXME(eddyb) there might be a better way to estimate the
                    // relevant extent for the array, maybe assume length >= 1
                    // so the minimum range is always `0..stride`?
                    max_size: index_bounds.clone().map(|index_bounds| {
                        u32::try_from(index_bounds.end)
                            .ok()
                            .unwrap_or(0)
                            .checked_mul(stride.get())
                            .unwrap_or(0)
                    }),
                    flags: QPtrMemUsageFlags::empty(),
                    kind: QPtrMemUsageKind::DynOffsetBase {
                        // FIXME(eddyb) allocating `Rc` a bit wasteful here.
                        element: Rc::new(QPtrMemUsage::UNUSED),

                        stride,
                    },
                });

                let (array_ptr, (array_addr_space, array_layout)) = self
                    .adjust_pointer_for_offset_and_usage(
                        data_inst_def.inputs[0],
                        0,
                        &strided_qptr_usage,
                        func_at_data_inst.reborrow().at(()),
                        insert_aux_data_inst,
                    )?;

                let (elem_layout, array_index_multiplier) = match array_layout {
                    TypeLayout::Concrete(array_layout) => match &array_layout.components {
                        Components::Elements { stride: layout_stride, elem, fixed_len }
                            if stride.get() % layout_stride.get() == 0
                                && Ok(index_bounds.clone())
                                    == fixed_len
                                        .map(|len| i32::try_from(len.get()).map(|len| 0..len))
                                        .transpose() =>
                        {
                            (elem.clone(), stride.get() / layout_stride.get())
                        }
                        _ => {
                            return Err(LiftError(Diag::bug([
                                "matching array not found in pointee type layout".into(),
                            ])));
                        }
                    },
                    _ => unreachable!(),
                };

                let array_index = if array_index_multiplier == 1 {
                    data_inst_def.inputs[1]
                } else {
                    // FIXME(eddyb) implement
                    return Err(LiftError(Diag::bug([
                        "unimplemented stride factor (index multiplier)".into(),
                    ])));
                };

                let usage_bounded_intra_elem = match output_qptr_usage {
                    &QPtrUsage::Memory(QPtrMemUsage { max_size: Some(max_size), .. }) => {
                        max_size <= elem_layout.mem_layout.fixed_base.size
                    }
                    _ => false,
                };
                if !usage_bounded_intra_elem {
                    // FIXME(eddyb) should this change the choice of pointer
                    // representation, or at least leave `QPtrOp::DynIndex`
                    // behind unchanged?
                }

                let mut data_inst_def = data_inst_def;
                data_inst_def.kind =
                    DataInstKind::SpvInst(wk.OpAccessChain.into(), spv::InstLowering::default());
                data_inst_def.inputs = [array_ptr, array_index].into_iter().collect();
                data_inst_def.outputs[0].attrs =
                    self.lifter.strip_qptr_usage_attr(data_inst_def.outputs[0].attrs);
                data_inst_def.outputs[0].ty =
                    self.lifter.spv_ptr_type(array_addr_space, elem_layout.original_type);
                data_inst_def
            }
            DataInstKind::QPtr(op @ (QPtrOp::Load { offset } | QPtrOp::Store { offset })) => {
                let mut data_inst_def = data_inst_def.clone();

                let (spv_opcode, access_type) = match op {
                    QPtrOp::Load { .. } => (wk.OpLoad, data_inst_def.outputs[0].ty),
                    QPtrOp::Store { .. } => (wk.OpStore, type_of_val(data_inst_def.inputs[1])),
                    _ => unreachable!(),
                };

                // FIXME(eddyb) this is awkward (or at least its needs DRY-ing)
                // because only an approximation is needed, most checks are
                // done by `adjust_pointer_for_offset_and_usage`.
                let access_qptr_usage = match self.lifter.layout_of(access_type)? {
                    TypeLayout::HandleArray(..) => {
                        return Err(LiftError(Diag::bug([
                            "cannot access whole HandleArray".into()
                        ])));
                    }
                    TypeLayout::Handle(shapes::Handle::Opaque(ty)) => {
                        QPtrUsage::Handles(shapes::Handle::Opaque(ty))
                    }
                    TypeLayout::Handle(shapes::Handle::Buffer(as_, _)) => {
                        QPtrUsage::Handles(shapes::Handle::Buffer(as_, QPtrMemUsage::UNUSED))
                    }
                    TypeLayout::Concrete(concrete) => QPtrUsage::Memory(QPtrMemUsage {
                        max_size: (concrete.mem_layout.dyn_unit_stride.is_none())
                            .then_some(concrete.mem_layout.fixed_base.size),
                        flags: QPtrMemUsageFlags::empty(),
                        kind: QPtrMemUsageKind::DirectAccess(concrete.original_type),
                    }),
                };

                let (adjusted_ptr, (_, adjusted_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_usage(
                        data_inst_def.inputs[0],
                        *offset,
                        &access_qptr_usage,
                        func_at_data_inst.reborrow().at(()),
                        insert_aux_data_inst,
                    )?;

                // FIXME(eddyb) implement at least same-size bitcasting
                // (more generally, accesses should be {de,re}composed).
                match adjusted_pointee_layout {
                    TypeLayout::Handle(shapes::Handle::Opaque(ty)) if ty == access_type => {}
                    TypeLayout::Concrete(concrete) if concrete.original_type == access_type => {}

                    _ => {
                        return Err(LiftError(Diag::bug([
                            "expected access type not found in pointee type layout".into(),
                        ])));
                    }
                }

                data_inst_def.kind =
                    DataInstKind::SpvInst(spv_opcode.into(), spv::InstLowering::default());
                data_inst_def.inputs[0] = adjusted_ptr;

                data_inst_def
            }
            &DataInstKind::QPtr(QPtrOp::Copy { size }) => {
                let mut data_inst_def = data_inst_def.clone();

                let max_size = Some(size.get());
                // FIXME(eddyb) `QPtrMemUsageKind::Unused` might
                // make more sense data-structure wise, but it
                // risks potentially losing the `flags`.
                let kind = QPtrMemUsageKind::OffsetBase(Default::default());

                let (dst_adjusted_ptr, (_, dst_adjusted_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_usage(
                        data_inst_def.inputs[0],
                        0,
                        &QPtrUsage::Memory(QPtrMemUsage {
                            max_size,
                            flags: QPtrMemUsageFlags::COPY_DST,
                            kind: kind.clone(),
                        }),
                        func_at_data_inst.reborrow().at(()),
                        insert_aux_data_inst,
                    )?;

                let (src_adjusted_ptr, (_, src_adjusted_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_usage(
                        data_inst_def.inputs[1],
                        0,
                        &QPtrUsage::Memory(QPtrMemUsage {
                            max_size,
                            flags: QPtrMemUsageFlags::COPY_SRC,
                            kind: kind.clone(),
                        }),
                        func_at_data_inst.reborrow().at(()),
                        insert_aux_data_inst,
                    )?;

                data_inst_def.inputs[0] = dst_adjusted_ptr;
                data_inst_def.inputs[1] = src_adjusted_ptr;

                let spv_opcode = match (dst_adjusted_pointee_layout, src_adjusted_pointee_layout) {
                    (TypeLayout::Concrete(dst_concrete), TypeLayout::Concrete(src_concrete))
                        if dst_concrete.original_type == src_concrete.original_type
                            && dst_concrete.mem_layout.fixed_base.size == size.get()
                            && dst_concrete.mem_layout.dyn_unit_stride.is_none() =>
                    {
                        wk.OpCopyMemory
                    }

                    // TODO(eddyb) implement by expanding pointee leaves in `0..size`.
                    _ => {
                        data_inst_def
                            .inputs
                            .push(Value::Const(cx.intern(scalar::Const::from_u32(size.get()))));
                        wk.OpCopyMemorySized
                    }
                };

                data_inst_def.kind =
                    DataInstKind::SpvInst(spv_opcode.into(), spv::InstLowering::default());
                data_inst_def
            }

            DataInstKind::SpvInst(_, lowering) | DataInstKind::SpvExtInst { lowering, .. } => {
                let lowering_disaggregated_output = lowering.disaggregated_output;

                let mut changed_data_inst_def = None;

                for attr in &cx[data_inst_def.attrs].attrs {
                    let attr = match attr {
                        Attr::QPtr(attr) => attr,
                        _ => continue,
                    };

                    let data_inst_def = changed_data_inst_def
                        .get_or_insert_with(|| func_at_data_inst.reborrow().def().clone());

                    match *attr {
                        QPtrAttr::ToSpvPtrInput { input_idx, pointee: expected_pointee_type } => {
                            let input_idx = usize::try_from(input_idx).unwrap();
                            let expected_pointee_type = expected_pointee_type.0;

                            let input_ptr = data_inst_def.inputs[input_idx];

                            // FIXME(eddyb) this is awkward (or at least its needs DRY-ing)
                            // because only an approximation is needed, most checks are
                            // done by `adjust_pointer_for_offset_and_usage`.
                            let expected_pointee_layout =
                                self.lifter.layout_of(expected_pointee_type)?;
                            let expected_qptr_usage = match &expected_pointee_layout {
                                TypeLayout::HandleArray(..) => {
                                    return Err(LiftError(Diag::bug([
                                        "cannot access whole HandleArray".into(),
                                    ])));
                                }
                                &TypeLayout::Handle(shapes::Handle::Opaque(ty)) => {
                                    QPtrUsage::Handles(shapes::Handle::Opaque(ty))
                                }
                                &TypeLayout::Handle(shapes::Handle::Buffer(as_, _)) => {
                                    QPtrUsage::Handles(shapes::Handle::Buffer(
                                        as_,
                                        QPtrMemUsage::UNUSED,
                                    ))
                                }
                                TypeLayout::Concrete(concrete) => QPtrUsage::Memory(QPtrMemUsage {
                                    max_size: (concrete.mem_layout.dyn_unit_stride.is_none())
                                        .then_some(concrete.mem_layout.fixed_base.size),
                                    flags: QPtrMemUsageFlags::empty(),
                                    kind: QPtrMemUsageKind::StrictlyTyped(concrete.original_type),
                                }),
                            };

                            let (adjusted_ptr, (_, adjusted_pointee_layout)) = self
                                .adjust_pointer_for_offset_and_usage(
                                    input_ptr,
                                    0,
                                    &expected_qptr_usage,
                                    func_at_data_inst.reborrow().at(()),
                                    insert_aux_data_inst,
                                )?;
                            match (adjusted_pointee_layout, expected_pointee_layout) {
                                (
                                    TypeLayout::Handle(shapes::Handle::Opaque(a)),
                                    TypeLayout::Handle(shapes::Handle::Opaque(b)),
                                ) if a == b => {}
                                (TypeLayout::Concrete(a), TypeLayout::Concrete(b))
                                    if a.original_type == b.original_type => {}

                                _ => {
                                    return Err(LiftError(Diag::bug([
                                        "ToSpvPtrInput: expected type not found in pointee type layout"
                                            .into(),
                                    ])));
                                }
                            }
                            data_inst_def.inputs[input_idx] = adjusted_ptr;
                        }
                        QPtrAttr::FromSpvPtrOutput { addr_space, pointee } => {
                            assert!(lowering_disaggregated_output.is_none());

                            assert_eq!(data_inst_def.outputs.len(), 1);
                            data_inst_def.outputs[0].ty =
                                self.lifter.spv_ptr_type(addr_space.0, pointee.0);
                        }
                        QPtrAttr::Usage(_) => {}
                    }
                }

                return Ok(
                    changed_data_inst_def.map_or(Transformed::Unchanged, Transformed::Changed)
                );
            }
        };
        Ok(Transformed::Changed(replacement_data_inst_def))
    }

    /// Derive a pointer from `ptr` which simultaneously accounts for `offset`
    /// and compatibility with `target_usage`, by introducing new instructions
    /// (e.g. `OpAccessChain`) if needed (via `insert_aux_data_inst`).
    //
    // FIXME(eddyb) customize errors, to tell apart Offset/Load/Store/ToSpvPtrInput.
    // FIXME(eddyb) the returned `(AddrSpace, TypeLayout)` describes the returned
    // pointer, i.e. it's a cached copy of `as_spv_ptr_type(type_of(final_ptr))`,
    // ideally it would be wrapped in some `struct` that disambiguates it.
    fn adjust_pointer_for_offset_and_usage(
        &mut self,
        ptr: Value,
        offset: i32,
        target_usage: &QPtrUsage,

        // FIXME(eddyb) bundle these into some kind of "cursor" type.
        mut func: FuncAtMut<'_, ()>,
        mut insert_aux_data_inst: impl FnMut(&mut Self, FuncAtMut<'_, ()>, DataInstDef) -> DataInst,
    ) -> Result<(Value, (AddrSpace, TypeLayout)), LiftError> {
        let wk = self.lifter.wk;
        let cx = &self.lifter.cx;

        let (addr_space, mut pointee_layout) =
            self.type_of_val_as_spv_ptr_with_layout(func.reborrow().freeze().at(ptr))?;

        let mk_access_chain = |access_chain_inputs: SmallVec<_>, final_pointee_type| {
            if access_chain_inputs.len() > 1 {
                Value::NodeOutput {
                    node: insert_aux_data_inst(self, func, DataInstDef {
                        attrs: Default::default(),
                        kind: DataInstKind::SpvInst(
                            wk.OpAccessChain.into(),
                            spv::InstLowering::default(),
                        ),
                        inputs: access_chain_inputs,
                        child_regions: [].into_iter().collect(),
                        outputs: [NodeOutputDecl {
                            attrs: Default::default(),
                            ty: self.lifter.spv_ptr_type(addr_space, final_pointee_type),
                        }]
                        .into_iter()
                        .collect(),
                    }),
                    output_idx: 0,
                }
            } else {
                ptr
            }
        };

        let mut access_chain_inputs: SmallVec<_> = [ptr].into_iter().collect();

        if let TypeLayout::HandleArray(handle, _) = pointee_layout {
            access_chain_inputs.push(Value::Const(cx.intern(scalar::Const::from_u32(0))));
            pointee_layout = TypeLayout::Handle(handle);
        }
        let (mut pointee_layout, target_usage) = match (pointee_layout, target_usage) {
            (TypeLayout::HandleArray(..), _) => unreachable!(),

            // All the illegal cases are here to keep the rest tidier.
            (_, QPtrUsage::Handles(shapes::Handle::Buffer(..))) => {
                return Err(LiftError(Diag::bug(["cannot access whole Buffer".into()])));
            }
            (TypeLayout::Handle(_), _) if offset != 0 => {
                return Err(LiftError(Diag::bug(["cannot offset Handles".into()])));
            }
            (TypeLayout::Handle(shapes::Handle::Buffer(..)), _) => {
                return Err(LiftError(Diag::bug(["cannot offset/access into Buffer".into()])));
            }
            (TypeLayout::Handle(_), QPtrUsage::Memory(_)) => {
                return Err(LiftError(Diag::bug(["cannot access Handle as memory".into()])));
            }
            (TypeLayout::Concrete(_), QPtrUsage::Handles(_)) => {
                return Err(LiftError(Diag::bug(["cannot access memory as Handle".into()])));
            }

            (
                TypeLayout::Handle(shapes::Handle::Opaque(pointee_handle_type)),
                &QPtrUsage::Handles(shapes::Handle::Opaque(access_handle_type)),
            ) => {
                assert_eq!(offset, 0);

                if pointee_handle_type != access_handle_type {
                    return Err(LiftError(Diag::bug([
                        "(opaque handle) pointer vs access type mismatch (".into(),
                        pointee_handle_type.into(),
                        " vs ".into(),
                        access_handle_type.into(),
                        ")".into(),
                    ])));
                }

                return Ok((
                    mk_access_chain(access_chain_inputs, pointee_handle_type),
                    (addr_space, TypeLayout::Handle(shapes::Handle::Opaque(pointee_handle_type))),
                ));
            }

            (TypeLayout::Concrete(pointee_layout), QPtrUsage::Memory(mem_usage)) => {
                (pointee_layout, mem_usage)
            }
        };

        let mut offset = u32::try_from(offset)
            .ok()
            .ok_or_else(|| LiftError(Diag::bug(["negative offset".into()])))?;

        loop {
            // FIXME(eddyb) should `QPtrMemUsage` have have an `.extent()` method?
            let target_extent =
                Extent { start: 0, end: target_usage.max_size }.saturating_add(offset);

            // FIXME(eddyb) should `MemTypeLayout` have have an `.extent()` method?
            let pointee_extent = Extent {
                start: 0,
                end: (pointee_layout.mem_layout.dyn_unit_stride.is_none())
                    .then_some(pointee_layout.mem_layout.fixed_base.size),
            };

            // FIXME(eddyb) how much of this is actually useful given the
            // `if offset == 0 { break; }` tied to running out of leaves?
            let is_compatible = offset == 0 && pointee_extent.includes(&target_extent) && {
                match target_usage.kind {
                    QPtrMemUsageKind::Unused | QPtrMemUsageKind::OffsetBase(_) => true,
                    QPtrMemUsageKind::StrictlyTyped(target_ty) => {
                        pointee_layout.original_type == target_ty
                    }
                    QPtrMemUsageKind::DirectAccess(target_ty) => {
                        // NOTE(eddyb) in theory, non-atomic accesses understood
                        // by SPIR-T natively (mostly `qptr.{load,store}`) only
                        // need to cover the extent of the access, as long as
                        // the types involved are plain bits (scalars/vectors).
                        //
                        // FIXME(eddyb) take advantage of this by implementing
                        // scalar merge/auto-bitcast in `qptr::{analyze,lift}`.
                        let can_bitcast = |ty: Type| {
                            matches!(cx[ty].kind, TypeKind::Scalar(_) | TypeKind::Vector(_))
                        };
                        pointee_layout.original_type == target_ty
                            || can_bitcast(pointee_layout.original_type) && can_bitcast(target_ty)
                    }
                    QPtrMemUsageKind::DynOffsetBase { stride: target_stride, .. } => {
                        match pointee_layout.components {
                            Components::Elements { stride, .. } => {
                                // FIXME(eddyb) take advantage of this by implementing
                                // stride factoring in `qptr::{analyze,lift}`.
                                target_stride.get() % stride.get() == 0
                            }
                            _ => false,
                        }
                    }
                }
            };

            // Only stop descending into the pointee type when it already fits
            // `target_usage` exactly (i.e. can only get worse, not better).
            if is_compatible && pointee_extent == target_extent {
                break;
            }

            let mut component_indices =
                pointee_layout.components.find_components_containing(target_extent);
            let idx = match (component_indices.next(), component_indices.next()) {
                (None, _) => {
                    // While none of the components fully contain `target_extent`,
                    // there's a good chance the pointer is already compatible
                    // with `target_usage` (and the only reason to keep going
                    // would be to find smaller types that remain compatible).
                    if is_compatible {
                        break;
                    }

                    // FIXME(eddyb) this could include the chosen indices,
                    // and maybe the current type and/or layout.
                    return Err(LiftError(Diag::bug([format!(
                        "offsets {:?}..{:?} not found in pointee type layout, \
                         after {} access chain indices",
                        target_extent.start,
                        target_extent.end,
                        access_chain_inputs.len() - 1
                    )
                    .into()])));
                }
                (Some(_), Some(_)) => {
                    return Err(LiftError(Diag::bug([
                        "ambiguity due to ZSTs in pointee type layout".into(),
                    ])));
                }
                (Some(idx), None) => idx,
            };
            drop(component_indices);

            let idx_as_i32 = i32::try_from(idx).ok().ok_or_else(|| {
                LiftError(Diag::bug([format!("{idx} not representable as a positive s32").into()]))
            })?;
            access_chain_inputs
                .push(Value::Const(cx.intern(scalar::Const::from_u32(idx_as_i32 as u32))));

            match &pointee_layout.components {
                Components::Scalar => unreachable!(),
                Components::Elements { stride, elem, .. } => {
                    offset %= stride.get();
                    pointee_layout = elem.clone();
                }
                Components::Fields { offsets, layouts } => {
                    offset -= offsets[idx];
                    pointee_layout = layouts[idx].clone();
                }
            }
        }

        Ok((
            mk_access_chain(access_chain_inputs, pointee_layout.original_type),
            (addr_space, TypeLayout::Concrete(pointee_layout)),
        ))
    }

    /// Apply rewrites implied by `deferred_ptr_noops` to `values`.
    ///
    /// This **does not** update `data_inst_use_counts` - in order to do that,
    /// you must call `self.remove_value_uses(values)` beforehand, and then also
    /// call `self.after_value_uses(values)` afterwards.
    fn resolve_deferred_ptr_noop_uses(&self, values: &mut [Value]) {
        for v in values {
            // FIXME(eddyb) the loop could theoretically be avoided, but that'd
            // make tracking use counts harder.
            while let Value::NodeOutput { node: inst, output_idx: 0 } = *v {
                match self.deferred_ptr_noops.get(&inst) {
                    Some(ptr_noop) => {
                        *v = ptr_noop.output_pointer;
                    }
                    None => break,
                }
            }
        }
    }

    // FIXME(eddyb) these are only this whacky because an `u32` is being
    // encoded as `Option<NonZeroU32>` for (dense) map entry reasons.
    fn add_value_uses(&mut self, values: &[Value]) {
        for &v in values {
            if let Value::NodeOutput { node: inst, .. } = v {
                let count = self.data_inst_use_counts.entry(inst);
                *count = Some(
                    NonZeroU32::new(count.map_or(0, |c| c.get()).checked_add(1).unwrap()).unwrap(),
                );
            }
        }
    }
    fn remove_value_uses(&mut self, values: &[Value]) {
        for &v in values {
            if let Value::NodeOutput { node: inst, .. } = v {
                let count = self.data_inst_use_counts.entry(inst);
                *count = NonZeroU32::new(count.unwrap().get() - 1);
            }
        }
    }
}

impl Transformer for LiftToSpvPtrInstsInFunc<'_> {
    // FIXME(eddyb) this is intentionally *shallow* and will not handle pointers
    // "hidden" in composites (which should be handled in SPIR-T explicitly).
    fn transform_const_use(&mut self, ct: Const) -> Transformed<Const> {
        // FIXME(eddyb) maybe cache this remap (in `LiftToSpvPtrs`, globally).
        let ct_def = &self.lifter.cx[ct];
        if let ConstKind::PtrToGlobalVar(gv) = ct_def.kind {
            Transformed::Changed(self.lifter.cx.intern(ConstDef {
                attrs: ct_def.attrs,
                ty: self.global_vars[gv].type_of_ptr_to,
                kind: ct_def.kind.clone(),
            }))
        } else {
            Transformed::Unchanged
        }
    }

    fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
        self.add_value_uses(&[*v]);

        v.inner_transform_with(self)
    }

    fn in_place_transform_region_def(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        let outer_region = mem::replace(&mut self.parent_region, Some(func_at_region.position));
        func_at_region.inner_in_place_transform_with(self);
        self.parent_region = outer_region;
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        func_at_node.reborrow().inner_in_place_transform_with(self);

        let record_lift_error = |this: &mut Self, attrs: &mut AttrSet, LiftError(e)| {
            // HACK(eddyb) do not add redundant errors to `qptr::analyze` bugs.
            this.func_has_qptr_analysis_bug_diags = this.func_has_qptr_analysis_bug_diags
                || this.lifter.cx[*attrs].attrs.iter().any(|attr| match attr {
                    Attr::Diagnostics(diags) => diags.0.iter().any(|diag| match diag.level {
                        DiagLevel::Bug(loc) => {
                            loc.file().ends_with("qptr/analyze.rs")
                                || loc.file().ends_with("qptr\\analyze.rs")
                        }
                        _ => false,
                    }),
                    _ => false,
                });

            if !this.func_has_qptr_analysis_bug_diags {
                attrs.push_diag(&this.lifter.cx, e);
            }
        };

        let mut lifted = self.try_lift_data_inst_def(func_at_node.reborrow());
        if let Ok(Transformed::Unchanged) = lifted {
            let data_inst_def = func_at_node.reborrow().def();
            if let DataInstKind::QPtr(_) = data_inst_def.kind {
                lifted = Err(LiftError(Diag::bug(["unimplemented qptr instruction".into()])));
            } else {
                for output in &data_inst_def.outputs {
                    if matches!(self.lifter.cx[output.ty].kind, TypeKind::QPtr) {
                        lifted = Err(LiftError(Diag::bug([
                            "unimplemented qptr-producing instruction".into(),
                        ])));
                        break;
                    }
                }
            }
        }
        match lifted {
            Ok(Transformed::Unchanged) => {}
            Ok(Transformed::Changed(new_def)) => {
                // HACK(eddyb) this whole dance ensures that use counts
                // remain accurate, no matter what rewrites occur.
                let data_inst_def = func_at_node.reborrow().def();
                self.remove_value_uses(&data_inst_def.inputs);
                *data_inst_def = new_def;
                self.resolve_deferred_ptr_noop_uses(&mut data_inst_def.inputs);
                self.add_value_uses(&data_inst_def.inputs);
            }
            Err(e) => {
                record_lift_error(self, &mut func_at_node.reborrow().def().attrs, e);
            }
        }

        let func_at_node_frozen = func_at_node.reborrow().freeze();
        let node_def = func_at_node_frozen.def();
        if let NodeKind::FuncCall { .. } = node_def.kind {
            let func = func_at_node_frozen.at(());
            if node_def.inputs.iter().any(|&v| {
                self.lifter.as_spv_ptr_type(func.at(v).type_of(&self.lifter.cx)).is_some()
            }) {
                record_lift_error(
                    self,
                    &mut func_at_node.def().attrs,
                    LiftError(Diag::bug(["unimplemented calls with pointer args".into()])),
                );
            }
        }
    }

    fn in_place_transform_func_decl(&mut self, func_decl: &mut FuncDecl) {
        func_decl.inner_in_place_transform_with(self);

        // Remove all `deferred_ptr_noops` instructions that are truly unused.
        if let DeclDef::Present(func_def_body) = &mut func_decl.def {
            let deferred_ptr_noops = mem::take(&mut self.deferred_ptr_noops);
            // NOTE(eddyb) reverse order is important, as each removal can reduce
            // use counts of an earlier definition, allowing further removal.
            for (inst, ptr_noop) in deferred_ptr_noops.into_iter().rev() {
                if self.data_inst_use_counts.get(inst).is_none() {
                    // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                    // due to the need to borrow `regions` and `nodes`
                    // at the same time - perhaps some kind of `FuncAtMut` position
                    // types for "where a list is in a parent entity" could be used
                    // to make this more ergonomic, although the potential need for
                    // an actual list entity of its own, should be considered.
                    func_def_body.regions[ptr_noop.parent_region]
                        .children
                        .remove(inst, &mut func_def_body.nodes);

                    self.remove_value_uses(&func_def_body.at(inst).def().inputs);
                }
            }
        }
    }
}
