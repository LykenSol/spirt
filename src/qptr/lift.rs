//! [`QPtr`](crate::TypeKind::QPtr) lifting to typed memory (e.g. to SPIR-V).
//
// FIXME(eddyb) the `legalize`-vs-`analyze`+`lift` split can be confusing,
// and may need more than documentation (but for now, see `qptr::legalize` docs).

use crate::func_at::{FuncAt, FuncAtMut};
use crate::mem::{
    DataHapp, DataHappFlags, DataHappKind, MemAccesses, MemAttr, MemOp, const_data, shapes,
};
use crate::qptr::{QPtrAttr, QPtrOp};
use crate::transform::{InnerInPlaceTransform, InnerTransform, Transformed, Transformer};
use crate::{
    AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstDef, ConstKind, Context, DataInst,
    DataInstDef, DataInstKind, DeclDef, Diag, DiagLevel, EntityDefs, EntityOrientedDenseMap, Func,
    FuncDecl, FxIndexMap, GlobalVar, GlobalVarDecl, GlobalVarInit, Module, Node, NodeKind, Region,
    Type, TypeDef, TypeKind, TypeOrConst, Value, Var, VarDecl, VarKind, scalar, spv, vector,
};
use itertools::Either;
use smallvec::SmallVec;
use std::mem;
use std::num::NonZeroU32;
use std::rc::Rc;

// HACK(eddyb) sharing layout code with other modules.
// FIXME(eddyb) can this just be a non-glob import?
use crate::mem::layout::*;

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
        // HACK(eddyb) only change any fields of `global_var_decl` on success.
        let lift_result = self
            .spv_pointee_type_and_addr_space_for_global_var(global_var_decl)
            .and_then(|(spv_pointee_type, addr_space)| {
                let maybe_init = match &mut global_var_decl.def {
                    DeclDef::Imported(_) => None,
                    DeclDef::Present(global_var_def_body) => {
                        global_var_def_body.initializer.as_mut()
                    }
                };

                let maybe_init_and_lifted_init = maybe_init
                    .map(|init| {
                        let lifted_init = self.try_lift_global_var_init(init, spv_pointee_type)?;
                        Ok((init, lifted_init))
                    })
                    .transpose()?;

                global_var_decl.attrs = self.strip_mem_accesses_attr(global_var_decl.attrs);
                global_var_decl.type_of_ptr_to = self.spv_ptr_type(addr_space, spv_pointee_type);
                global_var_decl.addr_space = addr_space;
                global_var_decl.shape = None;

                if let Some((init, lifted_init)) = maybe_init_and_lifted_init {
                    *init = lifted_init;
                }

                Ok(())
            });
        match lift_result {
            Ok(()) => {}
            Err(LiftError(e)) => {
                global_var_decl.attrs.push_diag(&self.cx, e);
            }
        }
    }
    fn try_lift_global_var_init(
        &self,
        global_var_init: &GlobalVarInit,
        ty: Type,
    ) -> Result<GlobalVarInit, LiftError> {
        let data = match global_var_init {
            &GlobalVarInit::Direct(ct) => return Ok(GlobalVarInit::Direct(ct)),

            // FIXME(eddyb) there is no need for this to clone, but also this
            // should be rare (only an error case?).
            GlobalVarInit::SpvAggregate { .. } => {
                return Ok(global_var_init.clone());
            }

            GlobalVarInit::Data(data) => data,
        };
        let layout = match self.layout_of(ty)? {
            // FIXME(eddyb) consider bad interactions with "interface blocks"?
            TypeLayout::Handle(_) | TypeLayout::HandleArray(..) => {
                return Err(LiftError(Diag::bug(["handles should not have initializers".into()])));
            }
            TypeLayout::Concrete(layout) => layout,
        };

        // Whether `candidate_layout` is an aggregate (to recurse into).
        let is_aggregate = |candidate_layout: &MemTypeLayout| {
            matches!(
                &self.cx[candidate_layout.original_type].kind,
                TypeKind::SpvInst { value_lowering: spv::ValueLowering::Disaggregate(_), .. }
            )
        };

        let mut leaf_values = SmallVec::new();
        let result = layout.deeply_flatten_if(0, &is_aggregate, &mut |leaf_offset, leaf| {
            let leaf_offset = u32::try_from(leaf_offset).ok().ok_or_else(|| {
                LayoutError(Diag::bug(
                    [format!("negative layout leaf offset {leaf_offset}").into()],
                ))
            })?;

            let leaf_size = NonZeroU32::new(leaf.mem_layout.fixed_base.size).ok_or_else(|| {
                LayoutError(Diag::bug([
                    format!("zero-sized initializer leaf at offset {leaf_offset}, with type `")
                        .into(),
                    leaf.original_type.into(),
                    "`".into(),
                ]))
            })?;

            // FIXME(eddyb) avoid out-of-bounds panics with malformed layouts
            // (and/or guarantee certain invariants in layouts that didn't error).
            let mut leaf_parts = data.read(leaf_offset..(leaf_offset + leaf_size.get()));

            let leaf_part = leaf_parts.next().unwrap();
            let is_single_whole_part = leaf_parts.next().is_none()
                && match &leaf_part {
                    const_data::Part::Uninit { .. } | const_data::Part::Bytes(_) => true,
                    const_data::Part::Symbolic { size, maybe_partial_slice, value: _ } => {
                        maybe_partial_slice == &(0..size.get())
                    }
                };
            if !is_single_whole_part {
                // FIXME(eddyb) needs a better error (or even partial support?).
                return Err(LayoutError(Diag::bug([
                    format!("NYI: initializer leaf at offset {leaf_offset}, with type `").into(),
                    leaf.original_type.into(),
                    "`, straddles an undef and/or symbolic boundary".into(),
                ])));
            }

            let leaf_value = match leaf_part {
                const_data::Part::Uninit { .. } => self.cx.intern(ConstDef {
                    attrs: Default::default(),
                    ty: leaf.original_type,
                    kind: ConstKind::Undef,
                }),
                const_data::Part::Bytes(bytes) => {
                    let mut total_read_scalar_size = 0;
                    let mut read_next_scalar = |leaf_scalar_type: scalar::Type| {
                        let byte_len = match leaf_scalar_type {
                            scalar::Type::Bool => {
                                self.layout_cache.config.abstract_bool_size_align.0
                            }
                            scalar::Type::SInt(_)
                            | scalar::Type::UInt(_)
                            | scalar::Type::Float(_) => {
                                let bit_width = leaf_scalar_type.bit_width();
                                assert_eq!(bit_width % 8, 0);
                                bit_width / 8
                            }
                        } as usize;

                        let mut copied_bytes = [0; 16];
                        copied_bytes[..byte_len]
                            .copy_from_slice(&bytes[total_read_scalar_size..][..byte_len]);
                        if self.layout_cache.config.is_big_endian {
                            copied_bytes[..byte_len].reverse();
                        }
                        let bits = u128::from_le_bytes(copied_bytes);

                        let leaf_scalar = scalar::Const::try_from_bits(leaf_scalar_type, bits)
                            .ok_or_else(|| {
                                // HACK(eddyb) only `bool` should be able to fail this,
                                // everything else uses whole bytes (enforced above).
                                assert!(matches!(leaf_scalar_type, scalar::Type::Bool));
                                // FIXME(eddyb) needs a better error, esp. for `bool`.
                                LayoutError(Diag::bug([
                                    format!(
                                        "initializer leaf at offset {}, with type `",
                                        leaf_offset + (total_read_scalar_size as u32)
                                    )
                                    .into(),
                                    leaf.original_type.into(),
                                    format!("`, has invalid value {bits}").into(),
                                ]))
                            })?;

                        total_read_scalar_size += byte_len;

                        Ok(leaf_scalar)
                    };

                    let leaf_const_kind = match self.cx[leaf.original_type].kind {
                        TypeKind::Scalar(ty) => read_next_scalar(ty)?.into(),
                        TypeKind::Vector(ty) => {
                            // HACK(eddyb) buffering elems due to `Result`.
                            let elems: SmallVec<[_; 4]> = (0..ty.elem_count.get())
                                .map(|_| read_next_scalar(ty.elem))
                                .collect::<Result<_, _>>()?;
                            vector::Const::from_elems(ty, elems).into()
                        }
                        _ => {
                            return Err(LayoutError(Diag::bug([
                                format!(
                                    "NYI: initializer leaf at offset {leaf_offset}, with type `"
                                )
                                .into(),
                                leaf.original_type.into(),
                                format!("`, made of bytes ({bytes:?})").into(),
                            ])));
                        }
                    };

                    assert_eq!(total_read_scalar_size, bytes.len());

                    self.cx.intern(ConstDef {
                        attrs: Default::default(),
                        ty: leaf.original_type,
                        kind: leaf_const_kind,
                    })
                }
                const_data::Part::Symbolic { value, .. } => value,
            };

            let expected_ty = leaf.original_type;
            let found_ty = self.cx[leaf_value].ty;
            if expected_ty != found_ty {
                return Err(LayoutError(Diag::bug([
                    "initializer leaf type mismatch: expected `".into(),
                    expected_ty.into(),
                    "`, found `".into(),
                    found_ty.into(),
                    "` typed value `".into(),
                    leaf_value.into(),
                    "`".into(),
                ])));
            }

            leaf_values.push(leaf_value);

            Ok(())
        });
        result.map_err(|LayoutError(e)| LiftError(e))?;

        let expected_leaf_count = self.cx[layout.original_type].disaggregated_leaf_count();
        let found_leaf_count = leaf_values.len();
        if expected_leaf_count != found_leaf_count {
            return Err(LiftError(Diag::bug([format!(
                "initializer leaf count mismatch: expected {expected_leaf_count} leaves, \
                 found {found_leaf_count} leaves"
            )
            .into()])));
        }

        Ok(if is_aggregate(&layout) {
            GlobalVarInit::SpvAggregate { ty, leaves: leaf_values }
        } else {
            assert_eq!(leaf_values.len(), 1);
            GlobalVarInit::Direct(leaf_values.pop().unwrap())
        })
    }

    pub fn lift_all_funcs(&self, module: &mut Module, funcs: impl IntoIterator<Item = Func>) {
        for func in funcs {
            LiftToSpvPtrInstsInFunc {
                lifter: self,
                global_vars: &module.global_vars,

                parent_region: None,

                deferred_ptr_noops: Default::default(),
                var_use_counts: Default::default(),

                func_has_mem_or_qptr_bug_diags: false,
            }
            .in_place_transform_func_decl(&mut module.funcs[func]);
        }
    }

    fn find_mem_accesses_attr(&self, attrs: AttrSet) -> Option<&MemAccesses> {
        self.cx[attrs].attrs.iter().find_map(|attr| match attr {
            Attr::Mem(MemAttr::Accesses(accesses)) => Some(&accesses.0),
            _ => None,
        })
    }

    fn require_mem_accesses_attr(&self, attrs: AttrSet) -> Result<&MemAccesses, LiftError> {
        self.find_mem_accesses_attr(attrs)
            .ok_or_else(|| LiftError(Diag::bug(["missing `mem.accesses` attribute".into()])))
    }

    fn strip_mem_accesses_attr(&self, attrs: AttrSet) -> AttrSet {
        self.cx.intern(AttrSetDef {
            attrs: self.cx[attrs]
                .attrs
                .iter()
                .filter(|attr| !matches!(attr, Attr::Mem(MemAttr::Accesses(_))))
                .cloned()
                .collect(),
        })
    }

    fn spv_pointee_type_and_addr_space_for_global_var(
        &self,
        global_var_decl: &GlobalVarDecl,
    ) -> Result<(Type, AddrSpace), LiftError> {
        let wk = self.wk;

        let mem_accesses = self.require_mem_accesses_attr(global_var_decl.attrs)?;

        let shape =
            global_var_decl.shape.ok_or_else(|| LiftError(Diag::bug(["missing shape".into()])))?;
        let pointee_type = self.pointee_type_for_shape_and_accesses(shape, mem_accesses)?;
        let storage_class = match (global_var_decl.addr_space, shape) {
            (AddrSpace::Handles, shapes::GlobalVarShape::Handles { handle, fixed_count: _ }) => {
                match handle {
                    shapes::Handle::Opaque(_) => wk.UniformConstant,
                    shapes::Handle::Buffer(AddrSpace::SpvStorageClass(storage_class), _) => {
                        storage_class
                    }
                    shapes::Handle::Buffer(AddrSpace::Handles, _) => {
                        return Err(LiftError(Diag::bug([
                            "invalid `AddrSpace::Handles` in `Handle::Buffer`".into(),
                        ])));
                    }
                }
            }
            (
                AddrSpace::SpvStorageClass(storage_class),
                shapes::GlobalVarShape::UntypedData(_) | shapes::GlobalVarShape::TypedInterface(_),
            ) => storage_class,

            (
                AddrSpace::Handles,
                shapes::GlobalVarShape::UntypedData(_) | shapes::GlobalVarShape::TypedInterface(_),
            )
            | (AddrSpace::SpvStorageClass(_), shapes::GlobalVarShape::Handles { .. }) => {
                return Err(LiftError(Diag::bug(["mismatched `addr_space` and `shape`".into()])));
            }
        };
        let addr_space = AddrSpace::SpvStorageClass(storage_class);
        Ok((pointee_type, addr_space))
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

    fn pointee_type_for_shape_and_accesses(
        &self,
        shape: shapes::GlobalVarShape,
        accesses: &MemAccesses,
    ) -> Result<Type, LiftError> {
        let wk = self.wk;

        match (shape, accesses) {
            (
                shapes::GlobalVarShape::Handles { handle, fixed_count },
                MemAccesses::Handles(handle_accesses),
            ) => {
                let handle_type = match (handle, handle_accesses) {
                    (shapes::Handle::Opaque(ty), &shapes::Handle::Opaque(access_ty)) => {
                        if access_ty != ty {
                            return Err(LiftError(Diag::bug([
                                "mismatched opaque handle types in `mem.accesses` vs `shape`"
                                    .into(),
                            ])));
                        }
                        ty
                    }
                    (shapes::Handle::Buffer(_, buf), shapes::Handle::Buffer(_, data_happ)) => {
                        let max_size_allowed_by_shape =
                            buf.dyn_unit_stride.is_none().then_some(buf.fixed_base.size);
                        let attr_spv_decorate_block = Attr::SpvAnnotation(spv::Inst {
                            opcode: wk.OpDecorate,
                            imms: [spv::Imm::Short(wk.Decoration, wk.Block)].into_iter().collect(),
                        });
                        match &data_happ.kind {
                            DataHappKind::Dead => {
                                self.spv_op_type_struct([], [attr_spv_decorate_block])?
                            }
                            DataHappKind::Disjoint(fields) => self.spv_op_type_struct(
                                fields.iter().map(|(&field_offset, field_happ)| {
                                    Ok((
                                        field_offset,
                                        self.pointee_type_for_data_happ(
                                            field_happ,
                                            data_happ.flags,
                                            max_size_allowed_by_shape
                                                .and_then(|max| max.checked_sub(field_offset)),
                                        )?,
                                    ))
                                }),
                                [attr_spv_decorate_block],
                            )?,
                            DataHappKind::StrictlyTyped(_)
                            | DataHappKind::Direct(_)
                            | DataHappKind::Repeated { .. } => self.spv_op_type_struct(
                                [Ok((
                                    0,
                                    self.pointee_type_for_data_happ(
                                        data_happ,
                                        DataHappFlags::empty(),
                                        max_size_allowed_by_shape,
                                    )?,
                                ))],
                                [attr_spv_decorate_block],
                            )?,
                        }
                    }
                    _ => {
                        return Err(LiftError(Diag::bug([
                            "mismatched `mem.accesses` and `shape`".into(),
                        ])));
                    }
                };
                if fixed_count == Some(NonZeroU32::new(1).unwrap()) {
                    Ok(handle_type)
                } else {
                    self.spv_op_type_array(handle_type, fixed_count.map(|c| c.get()), None)
                }
            }
            (shapes::GlobalVarShape::UntypedData(layout), MemAccesses::Data(happ)) => {
                self.pointee_type_for_data_happ(happ, DataHappFlags::empty(), Some(layout.size))
            }

            // FIXME(eddyb) validate against accesses? (maybe in `mem::analyze`?)
            (shapes::GlobalVarShape::TypedInterface(ty), _) => Ok(ty),

            _ => Err(LiftError(Diag::bug(["mismatched `mem.accesses` and `shape`".into()]))),
        }
    }

    fn pointee_type_for_data_happ(
        &self,
        happ: &DataHapp,
        outer_effective_flags: DataHappFlags,
        // HACK(eddyb) `mem::analyze` should be merging shape and accesses itself.
        // FIXME(eddyb) this isn't actually used to validate anything, only as
        // a fallback for now (i.e. to avoid spurious `OpTypeRuntimeArray`s).
        max_size_allowed_by_shape: Option<u32>,
    ) -> Result<Type, LiftError> {
        // FIXME(eddyb) does this make sense across all flags?
        let effective_flags = outer_effective_flags | happ.flags;

        // TODO(eddyb) implement (or at least validate that there are no gaps).
        if effective_flags.contains(DataHappFlags::COPY_SRC_AND_DST) {
            let already_valid = match &happ.kind {
                // FIXME(eddyb) support more cases.
                &DataHappKind::StrictlyTyped(ty) | &DataHappKind::Direct(ty) => ty
                    .as_scalar(&self.cx)
                    .is_some_and(|ty| happ.max_size == Some(ty.bit_width() / 8)),
                _ => false,
            };

            if !already_valid {
                return Err(LiftError(Diag::bug([
                    "unimplemented `mem.copy` src+dst (gap filling) for ".into(),
                    MemAccesses::Data(happ.clone()).into(),
                ])));
            }
        }

        match &happ.kind {
            DataHappKind::Dead => self.spv_op_type_struct([], []),
            &DataHappKind::StrictlyTyped(ty) | &DataHappKind::Direct(ty) => Ok(ty),
            DataHappKind::Disjoint(fields) => {
                // HACK(eddyb) force the size of `OpTypeStruct`s that would be
                // otherwise undersized (as e.g. `mem.copy` src/dst).
                let size_forcing_zst_tail_field = happ
                    .max_size
                    .filter(|&size| {
                        let inherent_unaligned_size =
                            fields.last_key_value().map_or(0, |(&field_offset, field_happ)| {
                                field_offset.checked_add(field_happ.max_size.unwrap()).unwrap()
                            });
                        size > inherent_unaligned_size
                    })
                    .map(|size| Ok((size, self.spv_op_type_struct([], [])?)));

                self.spv_op_type_struct(
                    fields
                        .iter()
                        .map(|(&field_offset, field_happ)| {
                            Ok((
                                field_offset,
                                self.pointee_type_for_data_happ(
                                    field_happ,
                                    effective_flags,
                                    max_size_allowed_by_shape
                                        .and_then(|max| max.checked_sub(field_offset)),
                                )?,
                            ))
                        })
                        .chain(size_forcing_zst_tail_field),
                    [],
                )
            }
            DataHappKind::Repeated { element, stride } => {
                let element_type =
                    self.pointee_type_for_data_happ(element, effective_flags, None)?;

                let fixed_len = happ
                    .max_size
                    .or(max_size_allowed_by_shape)
                    .map(|size| {
                        if !size.is_multiple_of(stride.get()) {
                            return Err(LiftError(Diag::bug([format!(
                                "Repeated: size ({size}) not a multiple of stride ({stride})"
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
    /// (using `var_use_counts` to determine whether they're truly unused).
    deferred_ptr_noops: FxIndexMap<Var, DeferredPtrNoop>,

    var_use_counts: EntityOrientedDenseMap<Var, NonZeroU32>,

    // HACK(eddyb) this is used to avoid noise on top of `mem`/`qptr` diagnostics.
    func_has_mem_or_qptr_bug_diags: bool,
}

struct DeferredPtrNoop {
    output_pointer: Value,

    output_pointer_addr_space: AddrSpace,

    /// Should be equivalent to `layout_of` on `output_pointer`'s pointee type,
    /// except in the case of `QPtrOp::BufferData`.
    output_pointee_layout: TypeLayout,

    parent_region: Region,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum MaybeDynOffset {
    Const(i32),
    Dyn { index: Value, stride: NonZeroU32, array_max_size: Option<u32> },
}

impl LiftToSpvPtrInstsInFunc<'_> {
    // FIXME(eddyb) maybe all this data should be packaged up together in a
    // type with fields like those of `DeferredPtrNoop` (or even more).
    fn type_of_val_as_spv_ptr_with_layout(
        &self,
        func_at_value: FuncAt<'_, Value>,
    ) -> Result<(AddrSpace, TypeLayout), LiftError> {
        let v = func_at_value.position;

        if let Value::Var(v) = v
            && let Some(ptr_noop) = self.deferred_ptr_noops.get(&v)
        {
            return Ok((
                ptr_noop.output_pointer_addr_space,
                ptr_noop.output_pointee_layout.clone(),
            ));
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
            NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::ExitInvocation(_) => {
                return Ok(Transformed::Unchanged);
            }

            DataInstKind::Scalar(_) | DataInstKind::Vector(_) => return Ok(Transformed::Unchanged),

            &DataInstKind::FuncCall(_callee) => {
                for &v in &data_inst_def.inputs {
                    if self.lifter.as_spv_ptr_type(type_of_val(v)).is_some() {
                        return Err(LiftError(Diag::bug([
                            "unimplemented calls with pointer args".into(),
                        ])));
                    }
                }
                return Ok(Transformed::Unchanged);
            }

            &DataInstKind::Mem(MemOp::FuncLocalVar(mem_layout)) => {
                let output_mem_accesses = self
                    .lifter
                    .require_mem_accesses_attr(func.at(data_inst_def.outputs[0]).decl().attrs)?;

                // HACK(eddyb) reusing the same functionality meant for globals.
                let pointee_type = self.lifter.pointee_type_for_shape_and_accesses(
                    shapes::GlobalVarShape::UntypedData(mem_layout),
                    output_mem_accesses,
                )?;

                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.kind = DataInstKind::SpvInst(
                    spv::Inst {
                        opcode: wk.OpVariable,
                        imms: [spv::Imm::Short(wk.StorageClass, wk.Function)].into_iter().collect(),
                    },
                    spv::InstLowering::default(),
                );
                let output_decl = func_at_data_inst.reborrow().at(data_inst_def.outputs[0]).decl();
                output_decl.attrs = self.lifter.strip_mem_accesses_attr(output_decl.attrs);
                output_decl.ty =
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
                let output_decl = func_at_data_inst.reborrow().at(data_inst_def.outputs[0]).decl();
                output_decl.attrs = self.lifter.strip_mem_accesses_attr(output_decl.attrs);
                output_decl.ty = self.lifter.spv_ptr_type(addr_space, handle_type);
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

                self.deferred_ptr_noops.insert(
                    data_inst_def.outputs[0],
                    DeferredPtrNoop {
                        output_pointer: buf_ptr,
                        output_pointer_addr_space: addr_space,
                        output_pointee_layout: buf_data_layout,
                        parent_region: self.parent_region.unwrap(),
                    },
                );

                // FIXME(eddyb) avoid the repeated call to `type_of_val`,
                // maybe don't even replace the `QPtrOp::BufferData` instruction?
                let data_inst_def = data_inst_def.clone();

                let new_output_ty = type_of_val(buf_ptr);
                let output_decl = func_at_data_inst.reborrow().at(data_inst_def.outputs[0]).decl();
                output_decl.ty = new_output_ty;

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
                            && layouts.last().is_some_and(|last_field| {
                                last_field.mem_layout.fixed_base.size == 0
                                    && last_field.mem_layout.dyn_unit_stride
                                        == Some(dyn_unit_stride)
                                    && matches!(
                                        last_field.components,
                                        Components::Elements { fixed_len: None, .. }
                                    )
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
            DataInstKind::QPtr(offset_op @ (QPtrOp::Offset(_) | QPtrOp::DynOffset { .. })) => {
                let mut data_inst_def = data_inst_def.clone();

                let maybe_dyn_offset = match offset_op {
                    &QPtrOp::Offset(offset) => MaybeDynOffset::Const(offset),
                    QPtrOp::DynOffset { stride, index_bounds } => MaybeDynOffset::Dyn {
                        index: data_inst_def.inputs[1],
                        stride: *stride,
                        array_max_size: index_bounds.clone().map(|index_bounds| {
                            u32::try_from(index_bounds.end)
                                .ok()
                                .unwrap_or(0)
                                .checked_mul(stride.get())
                                .unwrap_or(0)
                        }),
                    },
                    _ => unreachable!(),
                };

                let output_mem_accesses = self
                    .lifter
                    .find_mem_accesses_attr(func.at(data_inst_def.outputs[0]).decl().attrs)
                    .unwrap_or(&MemAccesses::Data(DataHapp::DEAD));

                let mut func = func_at_data_inst.reborrow().at(());
                let (output_pointer, (output_pointer_addr_space, output_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_accesses(
                        data_inst_def.inputs[0],
                        maybe_dyn_offset,
                        output_mem_accesses,
                        None,
                        func.reborrow(),
                        insert_aux_data_inst,
                    )?;
                // FIXME(eddyb) not being able to reuse the original `DataInst`
                // is a bit ridiculous, but correctly doing that would complicate
                // `adjust_pointer_for_offset_and_accesses` in general.
                self.deferred_ptr_noops.insert(
                    data_inst_def.outputs[0],
                    DeferredPtrNoop {
                        output_pointer,
                        output_pointer_addr_space,
                        output_pointee_layout,
                        parent_region: self.parent_region.unwrap(),
                    },
                );
                // FIXME(eddyb) avoid the repeated call to `type_of_val`,
                // maybe don't even replace the original instruction?
                data_inst_def.kind = QPtrOp::Offset(0).into();
                data_inst_def.inputs = [output_pointer].into_iter().collect();
                let new_output_ty = func.reborrow().freeze().at(output_pointer).type_of(cx);
                let output_decl = func.at(data_inst_def.outputs[0]).decl();
                output_decl.ty = new_output_ty;
                data_inst_def
            }
            DataInstKind::Mem(op @ (MemOp::Load { offset } | MemOp::Store { offset })) => {
                let mut data_inst_def = data_inst_def.clone();

                // HACK(eddyb) `_` will match multiple variants soon.
                #[allow(clippy::match_wildcard_for_single_variants)]
                let (access_op, stored_value) = match op {
                    MemOp::Load { .. } => (MemOp::Load { offset: None }, None),
                    MemOp::Store { .. } => {
                        (MemOp::Store { offset: None }, Some(data_inst_def.inputs[1]))
                    }
                    _ => unreachable!(),
                };
                let access_type = stored_value
                    .map_or_else(|| func.at(data_inst_def.outputs[0]).decl().ty, type_of_val);
                let offset = *offset;

                // FIXME(eddyb) this is awkward (or at least its needs DRY-ing)
                // because only an approximation is needed, most checks are
                // done by `adjust_pointer_for_offset_and_accesses`.
                let access_mem_accesses = match self.lifter.layout_of(access_type)? {
                    TypeLayout::HandleArray(..) => {
                        return Err(LiftError(Diag::bug([
                            "cannot access whole HandleArray".into()
                        ])));
                    }
                    TypeLayout::Handle(shapes::Handle::Opaque(ty)) => {
                        MemAccesses::Handles(shapes::Handle::Opaque(ty))
                    }
                    TypeLayout::Handle(shapes::Handle::Buffer(as_, _)) => {
                        MemAccesses::Handles(shapes::Handle::Buffer(as_, DataHapp::DEAD))
                    }
                    TypeLayout::Concrete(concrete) => MemAccesses::Data(DataHapp {
                        max_size: (concrete.mem_layout.dyn_unit_stride.is_none())
                            .then_some(concrete.mem_layout.fixed_base.size),
                        flags: DataHappFlags::empty(),
                        kind: DataHappKind::Direct(concrete.original_type),
                    }),
                };

                let mut func = func_at_data_inst.reborrow().at(());
                let mut partial_offset = 0;
                let (adjusted_ptr, (_, adjusted_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_accesses(
                        data_inst_def.inputs[0],
                        MaybeDynOffset::Const(offset.map_or(0, |o| o.get())),
                        &access_mem_accesses,
                        Some(&mut partial_offset),
                        func.reborrow(),
                        insert_aux_data_inst,
                    )?;

                // FIXME(eddyb) implement at least same-size bitcasting
                // (more generally, accesses should be {de,re}composed).
                let pointee_type = match adjusted_pointee_layout {
                    TypeLayout::Handle(shapes::Handle::Opaque(ty)) => ty,
                    TypeLayout::Concrete(concrete) => concrete.original_type,
                    _ => unreachable!(),
                };

                // HACK(eddyb) avoids using too many tuples.
                #[derive(Copy, Clone)]
                struct BitWrangle {
                    pointee_scalar_type: scalar::Type,
                    pointee_uint_width: scalar::IntWidth,

                    access_scalar_type: scalar::Type,
                    access_scalar_width: scalar::IntWidth,

                    access_bit_offset_in_pointee: u32,
                }
                let valid_bitwrangling_access =
                    (pointee_type != access_type).then_some(()).and_then(|()| {
                        let pointee_scalar_type = pointee_type.as_scalar(cx)?;
                        let access_scalar_type = access_type.as_scalar(cx)?;

                        // HACK(eddyb) the simplest case to support requires
                        // unsigned integers (to allow zext w/o a bitcast).
                        let pointee_uint_width = match pointee_scalar_type {
                            scalar::Type::UInt(w) => w,
                            _ => return None,
                        };

                        let access_scalar_width =
                            scalar::IntWidth::try_from_bits(match access_scalar_type {
                                // HACK(eddyb) this treats booleans as integers,
                                // sized by the `LayoutConfig`, at the cost of
                                // introducing conversion complications later.
                                scalar::Type::Bool => {
                                    self.lifter.layout_cache.config.abstract_bool_size_align.1 * 8
                                }
                                _ => access_scalar_type.bit_width(),
                            })?;

                        let le_bit_offset = partial_offset * 8;
                        let access_bit_offset_in_pointee =
                            if self.lifter.layout_cache.config.is_big_endian {
                                pointee_uint_width.bits()
                                    - access_scalar_width.bits()
                                    - le_bit_offset
                            } else {
                                le_bit_offset
                            };

                        Some(BitWrangle {
                            pointee_scalar_type,
                            pointee_uint_width,
                            access_scalar_type,
                            access_scalar_width,
                            access_bit_offset_in_pointee,
                        })
                    });

                if pointee_type != access_type && valid_bitwrangling_access.is_none() {
                    return Err(LiftError(Diag::bug([
                        "expected access type `".into(),
                        access_type.into(),
                        "` not found in pointee type layout (found leaf: `".into(),
                        pointee_type.into(),
                        "`)".into(),
                    ])));
                }

                let Some(bw) = valid_bitwrangling_access else {
                    data_inst_def.kind = access_op.into();
                    data_inst_def.inputs[0] = adjusted_ptr;

                    return Ok(Transformed::Changed(data_inst_def));
                };

                let is_partial = bw.pointee_uint_width > bw.access_scalar_width;

                let inst_for_access_type_to_uint = |v, uint_ty| {
                    match bw.access_scalar_type {
                        // FIXME(eddyb) consider using a SPIR-T `Select` node.
                        scalar::Type::Bool => (
                            DataInstKind::SpvInst(wk.OpSelect.into(), spv::InstLowering::default()),
                            [v].into_iter()
                                .chain([true, false].map(|b| {
                                    Value::Const(
                                        cx.intern(scalar::Const::from_bits(uint_ty, b as u128)),
                                    )
                                }))
                                .collect(),
                        ),
                        // FIXME(eddyb) SPIR-T should have its own bitcast.
                        // FIXME(eddyb) try avoiding noop `OpBitcast`s.
                        _ => (
                            DataInstKind::SpvInst(
                                wk.OpBitcast.into(),
                                spv::InstLowering::default(),
                            ),
                            [v].into_iter().collect(),
                        ),
                    }
                };
                let inst_for_int_to_access_type = |v, int_ty| {
                    match bw.access_scalar_type {
                        scalar::Type::Bool => (
                            scalar::Op::IntBinary(scalar::IntBinOp::Ne).into(),
                            [v, Value::Const(cx.intern(scalar::Const::from_bits(int_ty, 0)))]
                                .into_iter()
                                .collect(),
                        ),
                        // FIXME(eddyb) SPIR-T should have its own bitcast.
                        // FIXME(eddyb) try avoiding noop `OpBitcast`s.
                        _ => (
                            DataInstKind::SpvInst(
                                wk.OpBitcast.into(),
                                spv::InstLowering::default(),
                            ),
                            [v].into_iter().collect(),
                        ),
                    }
                };

                // NOTE(eddyb) all partial accesses require loading the whole
                // pointee value, even when storing (as the store must combine
                // both the old pointee value and the newly stored value).
                let loaded_pointee = (stored_value.is_none() || is_partial).then(|| {
                    let load_pointee_inst = insert_aux_data_inst(
                        self,
                        func.reborrow(),
                        DataInstDef {
                            attrs: Default::default(),
                            kind: DataInstKind::Mem(MemOp::Load { offset: None }),
                            inputs: [adjusted_ptr].into_iter().collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        },
                    );
                    let loaded_pointee = func.vars.define(
                        cx,
                        VarDecl {
                            attrs: Default::default(),
                            ty: pointee_type,
                            def_parent: Either::Right(load_pointee_inst),
                            def_idx: 0,
                        },
                    );
                    func.nodes[load_pointee_inst].outputs.push(loaded_pointee);
                    Value::Var(loaded_pointee)
                });

                let loaded_pointee = match (loaded_pointee, stored_value) {
                    // HACK(eddyb) handled below (unify)?
                    (Some(loaded_pointee), _) if is_partial => loaded_pointee,

                    (Some(loaded_pointee), None) => {
                        assert!(!is_partial);

                        // FIXME(eddyb) SPIR-T should have its own bitcast.
                        (data_inst_def.kind, data_inst_def.inputs) =
                            inst_for_int_to_access_type(loaded_pointee, bw.pointee_scalar_type);
                        return Ok(Transformed::Changed(data_inst_def));
                    }
                    (None, Some(stored_value)) => {
                        assert!(!is_partial);

                        let (cast_kind, cast_inputs) =
                            inst_for_access_type_to_uint(stored_value, bw.pointee_scalar_type);
                        let cast_stored_value_inst = insert_aux_data_inst(
                            self,
                            func.reborrow(),
                            DataInstDef {
                                attrs: Default::default(),
                                kind: cast_kind,
                                inputs: cast_inputs,
                                child_regions: [].into_iter().collect(),
                                outputs: [].into_iter().collect(),
                            },
                        );
                        let cast_stored_value = func.vars.define(
                            cx,
                            VarDecl {
                                attrs: Default::default(),
                                ty: pointee_type,
                                def_parent: Either::Right(cast_stored_value_inst),
                                def_idx: 0,
                            },
                        );
                        func.nodes[cast_stored_value_inst].outputs.push(cast_stored_value);

                        data_inst_def.kind = access_op.into();
                        data_inst_def.inputs[0] = adjusted_ptr;
                        data_inst_def.inputs[1] = Value::Var(cast_stored_value);

                        return Ok(Transformed::Changed(data_inst_def));
                    }
                    (None, None) | (Some(_), Some(_)) => unreachable!(),
                };

                // FIXME(eddyb) unify this with the logic above?
                assert!(is_partial);
                let new_kind_and_inputs = if let Some(stored_value) = stored_value {
                    let shl_amount = bw.access_bit_offset_in_pointee;

                    // FIXME(eddyb) consider adding a method for this on `IntWidth`.
                    let mask = |w: scalar::IntWidth| !0u128 >> (128 - w.bits());

                    let mask_loaded_pointee_inst = insert_aux_data_inst(
                        self,
                        func.reborrow(),
                        DataInstDef {
                            attrs: Default::default(),
                            kind: scalar::Op::IntBinary(scalar::IntBinOp::And).into(),
                            inputs: [
                                loaded_pointee,
                                Value::Const(cx.intern(scalar::Const::from_bits(
                                    bw.pointee_scalar_type,
                                    mask(bw.pointee_uint_width)
                                        & !(mask(bw.access_scalar_width) << shl_amount),
                                ))),
                            ]
                            .into_iter()
                            .collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        },
                    );
                    let masked_loaded_pointee = func.vars.define(
                        cx,
                        VarDecl {
                            attrs: Default::default(),
                            ty: pointee_type,
                            def_parent: Either::Right(mask_loaded_pointee_inst),
                            def_idx: 0,
                        },
                    );
                    func.nodes[mask_loaded_pointee_inst].outputs.push(masked_loaded_pointee);

                    let zext_input_scalar_ty = match bw.access_scalar_type {
                        scalar::Type::SInt(_) => scalar::Type::SInt(bw.access_scalar_width),
                        scalar::Type::Bool | scalar::Type::UInt(_) | scalar::Type::Float(_) => {
                            scalar::Type::UInt(bw.access_scalar_width)
                        }
                    };
                    let zext_input_ty = cx.intern(zext_input_scalar_ty);
                    let zext_input = if zext_input_ty == access_type {
                        stored_value
                    } else {
                        let (kind, inputs) =
                            inst_for_access_type_to_uint(stored_value, zext_input_scalar_ty);
                        let cast_stored_value_inst = insert_aux_data_inst(
                            self,
                            func.reborrow(),
                            DataInstDef {
                                attrs: Default::default(),
                                kind,
                                inputs,
                                child_regions: [].into_iter().collect(),
                                outputs: [].into_iter().collect(),
                            },
                        );
                        let cast_stored_value = func.vars.define(
                            cx,
                            VarDecl {
                                attrs: Default::default(),
                                ty: zext_input_ty,
                                def_parent: Either::Right(cast_stored_value_inst),
                                def_idx: 0,
                            },
                        );
                        func.nodes[cast_stored_value_inst].outputs.push(cast_stored_value);
                        Value::Var(cast_stored_value)
                    };
                    let zext_stored_value_inst = insert_aux_data_inst(
                        self,
                        func.reborrow(),
                        DataInstDef {
                            attrs: Default::default(),
                            kind: scalar::Op::IntUnary(scalar::IntUnOp::TruncOrZeroExtend).into(),
                            inputs: [zext_input].into_iter().collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        },
                    );
                    let zext_stored_value = func.vars.define(
                        cx,
                        VarDecl {
                            attrs: Default::default(),
                            ty: pointee_type,
                            def_parent: Either::Right(zext_stored_value_inst),
                            def_idx: 0,
                        },
                    );
                    func.nodes[zext_stored_value_inst].outputs.push(zext_stored_value);

                    let shifted_left_stored_value = if shl_amount == 0 {
                        zext_stored_value
                    } else {
                        let shl_stored_value_inst = insert_aux_data_inst(
                            self,
                            func.reborrow(),
                            DataInstDef {
                                attrs: Default::default(),
                                kind: scalar::Op::IntBinary(scalar::IntBinOp::Shl).into(),
                                inputs: [
                                    Value::Var(zext_stored_value),
                                    Value::Const(cx.intern(scalar::Const::from_u32(shl_amount))),
                                ]
                                .into_iter()
                                .collect(),
                                child_regions: [].into_iter().collect(),
                                outputs: [].into_iter().collect(),
                            },
                        );
                        let shifted_left_stored_value = func.vars.define(
                            cx,
                            VarDecl {
                                attrs: Default::default(),
                                ty: pointee_type,
                                def_parent: Either::Right(shl_stored_value_inst),
                                def_idx: 0,
                            },
                        );
                        func.nodes[shl_stored_value_inst].outputs.push(shifted_left_stored_value);
                        shifted_left_stored_value
                    };

                    let merge_values_inst = insert_aux_data_inst(
                        self,
                        func.reborrow(),
                        DataInstDef {
                            attrs: Default::default(),
                            kind: scalar::Op::IntBinary(scalar::IntBinOp::Or).into(),
                            inputs: [
                                Value::Var(masked_loaded_pointee),
                                Value::Var(shifted_left_stored_value),
                            ]
                            .into_iter()
                            .collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        },
                    );
                    let merged_value = func.vars.define(
                        cx,
                        VarDecl {
                            attrs: Default::default(),
                            ty: pointee_type,
                            def_parent: Either::Right(merge_values_inst),
                            def_idx: 0,
                        },
                    );
                    func.nodes[merge_values_inst].outputs.push(merged_value);

                    (
                        access_op.into(),
                        [adjusted_ptr, Value::Var(merged_value)].into_iter().collect(),
                    )
                } else {
                    let shr_amount = bw.access_bit_offset_in_pointee;
                    let shifted_right_pointee = if shr_amount == 0 {
                        loaded_pointee
                    } else {
                        let shr_pointee_inst = insert_aux_data_inst(
                            self,
                            func.reborrow(),
                            DataInstDef {
                                attrs: Default::default(),
                                kind: scalar::Op::IntBinary(scalar::IntBinOp::ShrU).into(),
                                inputs: [
                                    loaded_pointee,
                                    Value::Const(cx.intern(scalar::Const::from_u32(shr_amount))),
                                ]
                                .into_iter()
                                .collect(),
                                child_regions: [].into_iter().collect(),
                                outputs: [].into_iter().collect(),
                            },
                        );
                        let shifted_pointee = func.vars.define(
                            cx,
                            VarDecl {
                                attrs: Default::default(),
                                ty: pointee_type,
                                def_parent: Either::Right(shr_pointee_inst),
                                def_idx: 0,
                            },
                        );
                        func.nodes[shr_pointee_inst].outputs.push(shifted_pointee);
                        Value::Var(shifted_pointee)
                    };

                    // FIXME(eddyb) some of this seems extra roundabout.
                    let (trunc_scalar_ty, trunc_op) = match bw.access_scalar_type {
                        scalar::Type::SInt(_) => (
                            scalar::Type::SInt(bw.access_scalar_width),
                            scalar::Op::IntUnary(scalar::IntUnOp::TruncOrSignExtend),
                        ),
                        scalar::Type::Bool | scalar::Type::UInt(_) | scalar::Type::Float(_) => (
                            scalar::Type::UInt(bw.access_scalar_width),
                            scalar::Op::IntUnary(scalar::IntUnOp::TruncOrZeroExtend),
                        ),
                    };
                    let (trunc_kind, trunc_inputs) =
                        (trunc_op.into(), [shifted_right_pointee].into_iter().collect());
                    let trunc_ty = cx.intern(trunc_scalar_ty);

                    if trunc_ty == access_type {
                        (trunc_kind, trunc_inputs)
                    } else {
                        let trunc_inst = insert_aux_data_inst(
                            self,
                            func.reborrow(),
                            DataInstDef {
                                attrs: Default::default(),
                                kind: trunc_kind,
                                inputs: trunc_inputs,
                                child_regions: [].into_iter().collect(),
                                outputs: [].into_iter().collect(),
                            },
                        );
                        let trunc_output = func.vars.define(
                            cx,
                            VarDecl {
                                attrs: Default::default(),
                                ty: trunc_ty,
                                def_parent: Either::Right(trunc_inst),
                                def_idx: 0,
                            },
                        );
                        func.nodes[trunc_inst].outputs.push(trunc_output);

                        inst_for_int_to_access_type(Value::Var(trunc_output), trunc_scalar_ty)
                    }
                };
                (data_inst_def.kind, data_inst_def.inputs) = new_kind_and_inputs;
                data_inst_def
            }
            &DataInstKind::Mem(MemOp::Copy { size }) => {
                let mut data_inst_def = data_inst_def.clone();

                let max_size = Some(size.get());
                // FIXME(eddyb) `DataHappKind::Dead` might
                // make more sense data-structure wise, but it
                // risks potentially losing the `flags`.
                let kind = DataHappKind::Disjoint(Default::default());

                let (dst_adjusted_ptr, (_, dst_adjusted_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_accesses(
                        data_inst_def.inputs[0],
                        MaybeDynOffset::Const(0),
                        &MemAccesses::Data(DataHapp {
                            max_size,
                            flags: DataHappFlags::COPY_DST,
                            kind: kind.clone(),
                        }),
                        None,
                        func_at_data_inst.reborrow().at(()),
                        insert_aux_data_inst,
                    )?;

                let (src_adjusted_ptr, (_, src_adjusted_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_accesses(
                        data_inst_def.inputs[1],
                        MaybeDynOffset::Const(0),
                        &MemAccesses::Data(DataHapp {
                            max_size,
                            flags: DataHappFlags::COPY_SRC,
                            kind: kind.clone(),
                        }),
                        None,
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
                            // done by `adjust_pointer_for_offset_and_accesses`.
                            let expected_pointee_layout =
                                self.lifter.layout_of(expected_pointee_type)?;
                            let expected_mem_accesses = match &expected_pointee_layout {
                                TypeLayout::HandleArray(..) => {
                                    return Err(LiftError(Diag::bug([
                                        "cannot access whole HandleArray".into(),
                                    ])));
                                }
                                &TypeLayout::Handle(shapes::Handle::Opaque(ty)) => {
                                    MemAccesses::Handles(shapes::Handle::Opaque(ty))
                                }
                                &TypeLayout::Handle(shapes::Handle::Buffer(as_, _)) => {
                                    MemAccesses::Handles(shapes::Handle::Buffer(
                                        as_,
                                        DataHapp::DEAD,
                                    ))
                                }
                                TypeLayout::Concrete(concrete) => MemAccesses::Data(DataHapp {
                                    max_size: (concrete.mem_layout.dyn_unit_stride.is_none())
                                        .then_some(concrete.mem_layout.fixed_base.size),
                                    flags: DataHappFlags::empty(),
                                    kind: DataHappKind::StrictlyTyped(concrete.original_type),
                                }),
                            };

                            let (adjusted_ptr, (_, adjusted_pointee_layout)) = self
                                .adjust_pointer_for_offset_and_accesses(
                                    input_ptr,
                                    MaybeDynOffset::Const(0),
                                    &expected_mem_accesses,
                                    None,
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
                                        "ToSpvPtrInput: expected type not found \
                                         in pointee type layout"
                                            .into(),
                                    ])));
                                }
                            }
                            data_inst_def.inputs[input_idx] = adjusted_ptr;
                        }
                        QPtrAttr::FromSpvPtrOutput { addr_space, pointee } => {
                            assert!(lowering_disaggregated_output.is_none());

                            assert_eq!(data_inst_def.outputs.len(), 1);
                            let output_decl =
                                func_at_data_inst.reborrow().at(data_inst_def.outputs[0]).decl();
                            output_decl.ty = self.lifter.spv_ptr_type(addr_space.0, pointee.0);
                        }
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
    /// and compatibility with `target_accesses`, by introducing new instructions
    /// (e.g. `OpAccessChain`) if needed (via `insert_aux_data_inst`).
    //
    // FIXME(eddyb) customize errors, to tell apart Offset/Load/Store/ToSpvPtrInput.
    // FIXME(eddyb) the returned `(AddrSpace, TypeLayout)` describes the returned
    // pointer, i.e. it's a cached copy of `as_spv_ptr_type(type_of(final_ptr))`,
    // ideally it would be wrapped in some `struct` that disambiguates it.
    //
    // FIXME(eddyb) consider undoing all of this work, and relying on a more
    // flexible pointer representation, instead.
    fn adjust_pointer_for_offset_and_accesses(
        &mut self,
        ptr: Value,
        mut offset: MaybeDynOffset,
        target_accesses: &MemAccesses,

        // HACK(eddyb) find a better API, maybe wrap inputs/outputs of this
        // whole "adjustment" process into `struct`s etc.
        allow_partial_offsets_and_write_them_back_into: Option<&mut u32>,

        // FIXME(eddyb) bundle these into some kind of "cursor" type.
        mut func: FuncAtMut<'_, ()>,
        insert_aux_data_inst: impl Fn(&mut Self, FuncAtMut<'_, ()>, DataInstDef) -> DataInst,
    ) -> Result<(Value, (AddrSpace, TypeLayout)), LiftError> {
        let wk = self.lifter.wk;
        let cx = &self.lifter.cx;

        let (addr_space, mut pointee_layout) =
            self.type_of_val_as_spv_ptr_with_layout(func.reborrow().freeze().at(ptr))?;

        let mut access_chain_inputs: SmallVec<_> = [ptr].into_iter().collect();

        // HACK(eddyb) account for `deferred_ptr_noops` interactions.
        self.resolve_deferred_ptr_noop_uses(&mut access_chain_inputs);

        // HACK(eddyb) disallowing naming the original `ptr` again.
        #[allow(unused)]
        let ptr = ();

        let access_chain_data_inst_kind =
            DataInstKind::SpvInst(wk.OpAccessChain.into(), spv::InstLowering::default());

        let mk_access_chain = |this: &mut Self,
                               mut func: FuncAtMut<'_, ()>,
                               access_chain_inputs: SmallVec<_>,
                               final_pointee_type| {
            if access_chain_inputs.len() > 1 {
                let node = insert_aux_data_inst(
                    this,
                    func.reborrow(),
                    DataInstDef {
                        attrs: Default::default(),
                        kind: access_chain_data_inst_kind.clone(),
                        inputs: access_chain_inputs,
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    },
                );

                let output_var = func.vars.define(
                    cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: this.lifter.spv_ptr_type(addr_space, final_pointee_type),
                        def_parent: Either::Right(node),
                        def_idx: 0,
                    },
                );
                func.nodes[node].outputs.push(output_var);

                Value::Var(output_var)
            } else {
                access_chain_inputs[0]
            }
        };

        if let TypeLayout::HandleArray(handle, _) = pointee_layout {
            access_chain_inputs.push(Value::Const(cx.intern(scalar::Const::from_u32(0))));
            pointee_layout = TypeLayout::Handle(handle);
        }
        let (mut pointee_layout, target_happ) = match (pointee_layout, target_accesses) {
            (TypeLayout::HandleArray(..), _) => unreachable!(),

            // All the illegal cases are here to keep the rest tidier.
            (_, MemAccesses::Handles(shapes::Handle::Buffer(..))) => {
                return Err(LiftError(Diag::bug(["cannot access whole Buffer".into()])));
            }
            (TypeLayout::Handle(_), _) if offset != MaybeDynOffset::Const(0) => {
                return Err(LiftError(Diag::bug(["cannot offset Handles".into()])));
            }
            (TypeLayout::Handle(shapes::Handle::Buffer(..)), _) => {
                return Err(LiftError(Diag::bug(["cannot offset/access into Buffer".into()])));
            }
            (TypeLayout::Handle(_), MemAccesses::Data(_)) => {
                return Err(LiftError(Diag::bug(["cannot access Handle as memory".into()])));
            }
            (TypeLayout::Concrete(_), MemAccesses::Handles(_)) => {
                return Err(LiftError(Diag::bug(["cannot access memory as Handle".into()])));
            }

            (
                TypeLayout::Handle(shapes::Handle::Opaque(pointee_handle_type)),
                &MemAccesses::Handles(shapes::Handle::Opaque(access_handle_type)),
            ) => {
                assert!(offset == MaybeDynOffset::Const(0));

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
                    mk_access_chain(self, func, access_chain_inputs, pointee_handle_type),
                    (addr_space, TypeLayout::Handle(shapes::Handle::Opaque(pointee_handle_type))),
                ));
            }

            (TypeLayout::Concrete(pointee_layout), MemAccesses::Data(data_happ)) => {
                (pointee_layout, data_happ)
            }
        };

        // HACK(eddyb) helper for `if !target_fits_in_pointee` (see below).
        let decompose_array_indexing = |this: &Self, func_at_ptr: FuncAt<'_, Value>| {
            let func = func_at_ptr.at(());
            let inst = match func_at_ptr.position {
                Value::Var(v) => match func.at(v).decl().kind() {
                    VarKind::NodeOutput { node: inst, output_idx: 0 } => inst,
                    _ => return None,
                },
                Value::Const(_) => return None,
            };
            let inst_def = func.at(inst).def();
            if inst_def.inputs.len() != 2 || inst_def.kind != access_chain_data_inst_kind {
                return None;
            }

            let array_ptr = inst_def.inputs[0];
            let array_index = inst_def.inputs[1];
            let (array_address_space, array_layout) =
                this.type_of_val_as_spv_ptr_with_layout(func.at(array_ptr)).ok()?;
            if addr_space != array_address_space {
                return None;
            }

            match array_layout {
                TypeLayout::Concrete(array_layout) => match &array_layout.components {
                    Components::Elements { stride, elem, .. } => {
                        Some((array_ptr, array_index, *stride, elem.clone()))
                    }
                    _ => None,
                },
                _ => None,
            }
        };

        loop {
            // FIXME(eddyb) should `DataHapp` have have an `.extent()` method?
            let target_extent = match offset {
                MaybeDynOffset::Const(offset) => {
                    // FIXME(eddyb) allow `target_extent` to represent negatives,
                    // or special-case it as overlapping no components. and thus
                    // requiring walking up `ptr` and/or a special representation.
                    let offset = u32::try_from(offset)
                        .ok()
                        .ok_or_else(|| LiftError(Diag::bug(["negative offset".into()])))?;
                    Extent { start: 0, end: target_happ.max_size }.saturating_add(offset)
                }
                MaybeDynOffset::Dyn { array_max_size, .. } => {
                    Extent { start: 0, end: array_max_size }
                }
            };

            // FIXME(eddyb) should `MemTypeLayout` have have an `.extent()` method?
            let pointee_extent = Extent {
                start: 0,
                end: (pointee_layout.mem_layout.dyn_unit_stride.is_none())
                    .then_some(pointee_layout.mem_layout.fixed_base.size),
            };

            let target_fits_in_pointee = pointee_extent.includes(&target_extent);

            // HACK(eddyb) escaping the logical pointer bounds is illegal,
            // but can be made to work by walking up the pointer definition.
            // FIXME(eddyb) consider tracking representations of `qptr`s
            // that deviate from "`Value` of SPIR-V logical pointer type".
            // FIXME(eddyb) obsolete this by making `qptr::legalize` handle more
            // dynamic offsets than those it needs to for dataflow/escaping reasons
            // (tho could it easily do that w/o the pre-lift accesses analysis?).
            if !target_fits_in_pointee && access_chain_inputs.len() == 1 {
                // HACK(eddyb) approximating a `try {...}` block.
                let mut maybe_recompose_dyn_indexing = || {
                    let (array_ptr, array_index, array_stride, array_elem) =
                        decompose_array_indexing(
                            self,
                            func.reborrow().freeze().at(access_chain_inputs[0]),
                        )?;

                    let array_index_ty = func.reborrow().freeze().at(array_index).type_of(cx);
                    let array_index_scalar_ty = array_index_ty.as_scalar(cx)?;
                    let (index_addend, remainder_offset) = match offset {
                        MaybeDynOffset::Const(offset) => {
                            let offset = u32::try_from(offset).ok()?;
                            (
                                Value::Const(cx.intern(scalar::Const::int_try_from_i128(
                                    array_index_scalar_ty,
                                    (offset / array_stride.get()).into(),
                                )?)),
                                offset % array_stride.get(),
                            )
                        }
                        MaybeDynOffset::Dyn { index, stride, .. } => {
                            // FIXME(eddyb) implement stride factoring.
                            if stride != array_stride {
                                return None;
                            }

                            // FIMXE(eddyb) cast mismatched types.
                            let index_scalar_ty =
                                func.reborrow().freeze().at(index).type_of(cx).as_scalar(cx)?;
                            if index_scalar_ty != array_index_scalar_ty {
                                return None;
                            }

                            (index, 0)
                        }
                    };

                    let combined_index_inst = insert_aux_data_inst(
                        self,
                        func.reborrow(),
                        DataInstDef {
                            attrs: Default::default(),
                            kind: DataInstKind::Scalar(scalar::IntBinOp::Add.into()),
                            inputs: [array_index, index_addend].into_iter().collect(),
                            child_regions: [].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        },
                    );

                    let combined_index_output_var = func.vars.define(
                        cx,
                        VarDecl {
                            attrs: Default::default(),
                            ty: array_index_ty,
                            def_parent: Either::Right(combined_index_inst),
                            def_idx: 0,
                        },
                    );
                    func.nodes[combined_index_inst].outputs.push(combined_index_output_var);

                    access_chain_inputs =
                        [array_ptr, Value::Var(combined_index_output_var)].into_iter().collect();
                    offset = MaybeDynOffset::Const(remainder_offset.try_into().unwrap());
                    pointee_layout = array_elem;

                    Some(())
                };
                if let Some(()) = maybe_recompose_dyn_indexing() {
                    continue;
                }
            }

            let positive_const_offset = match offset {
                MaybeDynOffset::Const(offset) => u32::try_from(offset).ok(),
                MaybeDynOffset::Dyn { .. } => None,
            };
            let has_compatible_offset = target_fits_in_pointee
                && positive_const_offset.is_some_and(|offset| {
                    offset == 0 || allow_partial_offsets_and_write_them_back_into.is_some()
                });
            let is_compatible = has_compatible_offset && {
                match target_happ.kind {
                    DataHappKind::Dead
                    | DataHappKind::Disjoint(_)
                    | DataHappKind::Repeated { .. } => true,

                    DataHappKind::StrictlyTyped(target_ty) => {
                        pointee_layout.original_type == target_ty
                    }
                    DataHappKind::Direct(target_ty) => {
                        // NOTE(eddyb) in theory, non-atomic accesses understood
                        // by SPIR-T natively (mostly `mem.{load,store}`) only
                        // need to cover the extent of the access, as long as
                        // the types involved are plain bits (scalars/vectors).
                        //
                        // FIXME(eddyb) take advantage of this by implementing
                        // scalar merge/auto-bitcast in `mem::analyze`+`qptr::lift`.
                        let can_bitwrangle = |ty: Type| {
                            matches!(cx[ty].kind, TypeKind::Scalar(_) | TypeKind::Vector(_))
                        };
                        pointee_layout.original_type == target_ty
                            || can_bitwrangle(pointee_layout.original_type)
                                && can_bitwrangle(target_ty)
                    }
                }
            };

            // Only stop descending into the pointee type when it already fits
            // `target_happ` exactly (i.e. can only get worse, not better).
            if is_compatible && pointee_extent == target_extent {
                break;
            }

            // Handle dynamic indexing without using `find_components_containing`,
            // which has can only express constant offsets, not symbolic ones.
            match (&pointee_layout.components, offset) {
                (
                    Components::Elements { stride: array_stride, elem, .. },
                    MaybeDynOffset::Dyn { index, stride: index_stride, .. },
                ) if index_stride.get().is_multiple_of(array_stride.get()) => {
                    let index_multiplier = index_stride.get() / array_stride.get();

                    let index = if index_multiplier == 1 {
                        index
                    } else {
                        // FIXME(eddyb) implement stride factoring here, and
                        // take advantage of it in `mem::analyze`.
                        return Err(LiftError(Diag::bug([format!(
                            "unimplemented stride factor (index multiplier) of {index_multiplier}"
                        )
                        .into()])));
                    };

                    // HACK(eddyb) separate the `OpAccessChain`s into one for
                    // obtaining the array pointer itself, and one for indexing
                    // the array, to allow folding the latter in subsequent calls
                    // to `adjust_pointer_for_offset_and_accesses`.
                    // FIXME(eddyb) consider tracking representations of `qptr`s
                    // that deviate from "`Value` of SPIR-V logical pointer type".
                    let array_ptr = mk_access_chain(
                        self,
                        func.reborrow(),
                        access_chain_inputs,
                        pointee_layout.original_type,
                    );
                    access_chain_inputs = [array_ptr, index].into_iter().collect();

                    offset = MaybeDynOffset::Const(0);
                    pointee_layout = elem.clone();

                    continue;
                }

                _ => {}
            }

            let mut component_indices =
                pointee_layout.components.find_components_containing(target_extent);
            let idx = match (component_indices.next(), component_indices.next()) {
                (None, _) => {
                    // While none of the components fully contain `target_extent`,
                    // there's a good chance the pointer is already compatible
                    // with `target_happ` (and the only reason to keep going
                    // would be to find smaller types that remain compatible).
                    if is_compatible || offset == MaybeDynOffset::Const(0) {
                        break;
                    }

                    // FIXME(eddyb) this could include the chosen indices,
                    // and/or maybe the original type as well?
                    return Err(LiftError(Diag::bug([
                        format!("offsets {target_extent} not found, in the layout of ").into(),
                        pointee_layout.original_type.into(),
                    ])));
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

            match &mut offset {
                MaybeDynOffset::Const(offset) => {
                    let mut offset_u32 = u32::try_from(*offset).unwrap();
                    match &pointee_layout.components {
                        Components::Scalar => unreachable!(),
                        Components::Elements { stride, .. } => {
                            offset_u32 %= stride.get();
                        }
                        Components::Fields { offsets, .. } => {
                            offset_u32 -= offsets[idx];
                        }
                    };
                    *offset = offset_u32.try_into().unwrap();
                }

                // HACK(eddyb) `target_extent.start` should be `0` for `Dyn`,
                // so no matching components should ever have an offset.
                MaybeDynOffset::Dyn { .. } => assert_eq!(target_extent.start, 0),
            }

            // FIXME(eddyb) `find_components_containing` should probably
            // return some of this information for free.
            pointee_layout = match &pointee_layout.components {
                Components::Scalar => unreachable!(),
                Components::Elements { elem, .. } => elem.clone(),
                Components::Fields { layouts, .. } => layouts[idx].clone(),
            };
        }

        if let Some(writeback_offset) = allow_partial_offsets_and_write_them_back_into {
            match offset {
                // HACK(eddyb) `offset` is ensured positive even w/ partial offsets.
                MaybeDynOffset::Const(offset) => *writeback_offset = offset.try_into().unwrap(),
                MaybeDynOffset::Dyn { .. } => unreachable!(),
            }
        }

        Ok((
            mk_access_chain(self, func, access_chain_inputs, pointee_layout.original_type),
            (addr_space, TypeLayout::Concrete(pointee_layout)),
        ))
    }

    /// Apply rewrites implied by `deferred_ptr_noops` to `values`.
    ///
    /// This **does not** update `var_use_counts` - in order to do that,
    /// you must call `self.remove_value_uses(values)` beforehand, and then also
    /// call `self.after_value_uses(values)` afterwards.
    fn resolve_deferred_ptr_noop_uses(&self, values: &mut [Value]) {
        for v in values {
            // FIXME(eddyb) the loop could theoretically be avoided, but that'd
            // make tracking use counts harder.
            while let Value::Var(var) = *v {
                match self.deferred_ptr_noops.get(&var) {
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
            if let Value::Var(v) = v {
                let count = self.var_use_counts.entry(v);
                *count = Some(
                    NonZeroU32::new(count.map_or(0, |c| c.get()).checked_add(1).unwrap()).unwrap(),
                );
            }
        }
    }
    fn remove_value_uses(&mut self, values: &[Value]) {
        for &v in values {
            if let Value::Var(v) = v {
                let count = self.var_use_counts.entry(v);
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
        let outer_region = self.parent_region.replace(func_at_region.position);
        func_at_region.inner_in_place_transform_with(self);
        self.parent_region = outer_region;
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        func_at_node.reborrow().inner_in_place_transform_with(self);

        let mut lifted = self.try_lift_data_inst_def(func_at_node.reborrow());
        if let Ok(Transformed::Unchanged) = lifted {
            let func_at_node = func_at_node.reborrow().freeze();
            let data_inst_def = func_at_node.def();
            if let DataInstKind::QPtr(_) = data_inst_def.kind {
                lifted = Err(LiftError(Diag::bug(["unimplemented qptr instruction".into()])));
            } else {
                for &output_var in &data_inst_def.outputs {
                    if matches!(
                        self.lifter.cx[func_at_node.at(output_var).decl().ty].kind,
                        TypeKind::QPtr
                    ) {
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
                let data_inst_def = func_at_node.def();
                self.remove_value_uses(&data_inst_def.inputs);
                *data_inst_def = new_def;
                self.resolve_deferred_ptr_noop_uses(&mut data_inst_def.inputs);
                self.add_value_uses(&data_inst_def.inputs);
            }
            Err(LiftError(e)) => {
                let node = func_at_node.position;
                let func = func_at_node.at(());
                let data_inst_def = &mut func.nodes[node];

                // HACK(eddyb) do not add redundant errors to `mem`/`qptr` bugs.
                self.func_has_mem_or_qptr_bug_diags = self.func_has_mem_or_qptr_bug_diags
                    || Diag::bug_src_path_prefix()
                        .and_then(|src_path_prefix| {
                            let qptr_lift_suffix = std::panic::Location::caller()
                                .file()
                                .strip_prefix(src_path_prefix)?;
                            (qptr_lift_suffix.starts_with("qptr")
                                && qptr_lift_suffix.ends_with("lift.rs"))
                            .then_some((src_path_prefix, qptr_lift_suffix))
                        })
                        .is_some_and(|(src_path_prefix, qptr_lift_suffix)| {
                            let all_attrs = [data_inst_def.attrs].into_iter().chain(
                                data_inst_def
                                    .outputs
                                    .iter()
                                    .map(|&output_var| func.vars[output_var].attrs),
                            );
                            all_attrs.flat_map(|attrs| attrs.diags(&self.lifter.cx)).any(|diag| {
                                match diag.level {
                                    DiagLevel::Bug(loc) => loc
                                        .file()
                                        .strip_prefix(src_path_prefix)
                                        .is_some_and(|suffix| {
                                            (suffix.starts_with("mem")
                                                || suffix.starts_with("qptr"))
                                                && suffix != qptr_lift_suffix
                                        }),
                                    _ => false,
                                }
                            })
                        });

                if !self.func_has_mem_or_qptr_bug_diags {
                    data_inst_def.attrs.push_diag(&self.lifter.cx, e);
                }
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
            for (output_var, ptr_noop) in deferred_ptr_noops.into_iter().rev() {
                let is_used = self.var_use_counts.get(output_var).is_some();
                if !is_used {
                    let inst = func_def_body.at(output_var).decl().def_parent.right().unwrap();

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
