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
    FuncDecl, FxIndexMap, GlobalVar, GlobalVarDecl, GlobalVarInit, Module, Node, NodeDef, NodeKind,
    Region, Type, TypeDef, TypeKind, TypeOrConst, Value, Var, VarDecl, VarKind, scalar, spv,
    vector,
};
use itertools::{Either, Itertools as _};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::mem;
use std::num::{NonZeroI32, NonZeroU32};
use std::ops::RangeInclusive;
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
        let mut had_mem_accesses = false;
        let mut new_attrs = AttrSetDef {
            attrs: self.cx[attrs]
                .attrs
                .iter()
                .filter(|attr| {
                    let is_mem_accesses = matches!(attr, Attr::Mem(MemAttr::Accesses(_)));
                    had_mem_accesses |= is_mem_accesses;
                    !is_mem_accesses
                })
                .cloned()
                .collect(),
        };

        // HACK(eddyb) if the attribute wasn't found in the first place, but
        // there wasn't an error preventing `strip_mem_accesses_attr` from
        // being called, that means the fallback kicked in, and all of the BUGs
        // that `mem::analyze` had emitted, can be discarded.
        // FIXME(eddyb) figure out a better way to negocitate this, maybe move
        // the fallback logic into `mem::analyze` itself, auto-degrading as-needed?
        if !had_mem_accesses && !new_attrs.diags().is_empty() {
            new_attrs.mutate_diags(|diags| {
                let Some(src_path_prefix) = Diag::bug_src_path_prefix().filter(|src_path_prefix| {
                    std::panic::Location::caller().file().strip_prefix(src_path_prefix).is_some_and(
                        |qptr_lift_suffix| {
                            qptr_lift_suffix.starts_with("qptr")
                                && qptr_lift_suffix.ends_with("lift.rs")
                        },
                    )
                }) else {
                    return;
                };
                diags.retain(|diag| {
                    let remove_diag = match diag.level {
                        DiagLevel::Bug(loc) => {
                            loc.file().strip_prefix(src_path_prefix).is_some_and(|suffix| {
                                suffix.starts_with("mem") && suffix.ends_with("analyze.rs")
                            })
                        }
                        _ => false,
                    };
                    !remove_diag
                });
            });
        }

        self.cx.intern(new_attrs)
    }

    // HACK(eddyb) try to deduce an array-like fallback, from alignment.
    fn fallback_accesses_from_shape(&self, shape: shapes::GlobalVarShape) -> Option<MemAccesses> {
        let fallback_data_happ_from_layout = |mem_layout: shapes::MaybeDynMemLayout| {
            let align = mem_layout.fixed_base.align;
            (align.is_power_of_two()
                && align <= 8
                && mem_layout.fixed_base.size.is_multiple_of(align)
                && mem_layout
                    .dyn_unit_stride
                    .is_none_or(|stride| stride.get().is_multiple_of(align)))
            .then(|| {
                // TODO(eddyb) remove temporary hack of using 4 where possible.
                let element_size =
                    if mem_layout.dyn_unit_stride.is_some() && false { 4 } else { align };
                let element = DataHapp {
                    max_size: Some(element_size),
                    flags: DataHappFlags::empty(),
                    kind: DataHappKind::Direct(self.cx.intern(scalar::Type::UInt(
                        scalar::IntWidth::try_from_bits(element_size * 8).unwrap(),
                    ))),
                };
                DataHapp {
                    max_size: mem_layout
                        .dyn_unit_stride
                        .is_none()
                        .then_some(mem_layout.fixed_base.size),
                    flags: DataHappFlags::empty(),
                    kind: DataHappKind::Repeated {
                        element: Rc::new(element),
                        stride: NonZeroU32::new(element_size).unwrap(),
                    },
                }
            })
        };
        match shape {
            shapes::GlobalVarShape::Handles {
                handle: shapes::Handle::Buffer(addr_space, buf),
                fixed_count: _,
            } => fallback_data_happ_from_layout(buf)
                .map(|happ| MemAccesses::Handles(shapes::Handle::Buffer(addr_space, happ))),
            shapes::GlobalVarShape::UntypedData(mem_layout) => {
                fallback_data_happ_from_layout(shapes::MaybeDynMemLayout {
                    fixed_base: mem_layout,
                    dyn_unit_stride: None,
                })
                .map(MemAccesses::Data)
            }
            _ => None,
        }
    }

    fn spv_pointee_type_and_addr_space_for_global_var(
        &self,
        global_var_decl: &GlobalVarDecl,
    ) -> Result<(Type, AddrSpace), LiftError> {
        let wk = self.wk;

        let shape =
            global_var_decl.shape.ok_or_else(|| LiftError(Diag::bug(["missing shape".into()])))?;

        let mem_accesses;
        let mem_accesses = match self.require_mem_accesses_attr(global_var_decl.attrs) {
            Ok(mem_accesses) => mem_accesses,
            Err(e) => {
                mem_accesses = self.fallback_accesses_from_shape(shape).ok_or(e)?;
                &mem_accesses
            }
        };

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
                        // FIXME(eddyb) this doesn't handle flags!
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

        // Memory used as a destination for some copies, and a source for others,
        // must not have any padding bytes in its type, as they make it impossible
        // to fully preserve (all bytes of) the value being copied through it.
        let disallow_padding = effective_flags.contains(DataHappFlags::COPY_SRC_AND_DST);

        // FIXME(eddyb) the naive expansion of copies (to uint loads and stores)
        // lacks any kind of pointee type awareness, so it can't skip over padding
        // in either the source or the destination - easier to make this stricter.
        let disallow_padding =
            disallow_padding || effective_flags.intersects(DataHappFlags::COPY_SRC_AND_DST);

        let size_of = |ty| match self.layout_of(ty).ok()? {
            TypeLayout::HandleArray(..) | TypeLayout::Handle(_) => None,
            TypeLayout::Concrete(concrete) => (concrete.mem_layout.dyn_unit_stride.is_none())
                .then_some(concrete.mem_layout.fixed_base.size),
        };
        let mk_padding_err = || {
            LiftError(Diag::bug([
                "failed to guarantee a padding-free type for ".into(),
                MemAccesses::Data(happ.clone()).into(),
            ]))
        };

        match &happ.kind {
            &DataHappKind::StrictlyTyped(ty) | &DataHappKind::Direct(ty) => {
                let is_strict = matches!(happ.kind, DataHappKind::StrictlyTyped(_));

                // HACK(eddyb) in order to support loads and stores that copies
                // might need to generate, the scalar leaves have to all be
                // unsigned integers, even without `disallow_padding`.
                if effective_flags.intersects(DataHappFlags::COPY_SRC_AND_DST) {
                    // FIXME(eddyb) the boolean silliness here results in a few
                    // `&&`/`||` uses that maybe should be `Option`/`Result`.
                    let already_valid = match self.cx[ty].kind {
                        // FIXME(eddyb) consider supporting more types here.
                        TypeKind::Scalar(ty) => {
                            let bit_width = ty.bit_width();
                            let mem_size = match ty {
                                scalar::Type::Bool => {
                                    self.layout_cache.config.abstract_bool_size_align.0
                                }
                                _ => bit_width / 8,
                            };

                            // FIXME(eddyb) should this consider increasing the
                            // width of type and/or adding extra filler?
                            // (is this even possible?)
                            happ.max_size == Some(mem_size) && {
                                let mem_uint = scalar::Type::UInt(
                                    scalar::IntWidth::try_from_bits(mem_size * 8).unwrap(),
                                );

                                ty == mem_uint || {
                                    if !is_strict {
                                        return Ok(self.cx.intern(mem_uint));
                                    }

                                    // FIXME(eddyb) what can be done here?
                                    false
                                }
                            }
                        }
                        TypeKind::Vector(ty) => {
                            // FIXME(eddyb) implement rewriting non-uint vectors.
                            match ty.elem {
                                scalar::Type::UInt(elem_width) => {
                                    let mem_size = (elem_width.bits() / 8)
                                        .checked_mul(ty.elem_count.get().into())
                                        .unwrap();

                                    happ.max_size == Some(mem_size)
                                }
                                _ => false,
                            }
                        }
                        _ => false,
                    };
                    if !already_valid {
                        return Err(mk_padding_err());
                    }
                }

                Ok(ty)
            }
            DataHappKind::Dead | DataHappKind::Disjoint(_) => {
                let no_fields = BTreeMap::new();
                let fields = match &happ.kind {
                    DataHappKind::Disjoint(fields) => &**fields,
                    _ => &no_fields,
                };

                // HACK(eddyb) in order to be able to detect gaps both between
                // fields, but also before/after the first/last field, extra
                // iterator entries are used, which have `None` in the second
                // component (instead of a `Some(field_happ)`), with the actual
                // gaps being observed through the use of `tuple_windows`.
                let mut field_offsets_and_types = [(Some(0), None)]
                    .into_iter()
                    .chain(
                        fields.iter().map(|(&field_offset, field_happ)| {
                            (Some(field_offset), Some(field_happ))
                        }),
                    )
                    .chain([((happ.max_size).or(max_size_allowed_by_shape), None)])
                    .tuple_windows()
                    .flat_map(|((field_offset, field_happ), (next_offset, next_happ))| {
                        let field_offset = field_offset.unwrap();
                        let is_last = next_happ.is_none();

                        // FIXME(eddyb) the use of `Option` (instead of `Result`)
                        // in some of these cases is suboptimal and/or confusing.
                        let field_type = field_happ.map(|field_happ| {
                            self.pointee_type_for_data_happ(
                                field_happ,
                                effective_flags,
                                max_size_allowed_by_shape
                                    .map(|max| max.saturating_sub(field_offset)),
                            )
                        });
                        let field_size = field_type
                            .as_ref()
                            .map_or(Ok(0), |ty| size_of(*ty.as_ref().map_err(|_e| ())?).ok_or(()))
                            .ok();
                        let field_range = field_size.and_then(|field_size| {
                            Some(field_offset..field_offset.checked_add(field_size)?)
                        });

                        let extra_field = if disallow_padding {
                            let maybe_gap = field_range
                                .and_then(|field_range| {
                                    Some((
                                        field_range.end,
                                        next_offset?.checked_sub(field_range.end)?,
                                    ))
                                })
                                .ok_or_else(mk_padding_err)
                                .map(|(gap_offset, gap_size)| {
                                    Some((gap_offset, NonZeroU32::new(gap_size)?))
                                })
                                .transpose();
                            maybe_gap.map(|gap| {
                                let (gap_offset, gap_size) = gap?;

                                // HACK(eddyb) pick `u32`, `u16` or `u8`,
                                // preferring the largest one of them,
                                // that `gap_size` is a multiple of.
                                let filler_unit_size = 1 << gap_size.trailing_zeros().clamp(0, 2);
                                let filler_unit = self.cx.intern(scalar::Type::UInt(
                                    scalar::IntWidth::try_from_bits(filler_unit_size * 8).unwrap(),
                                ));
                                let filler_count = gap_size.get() / filler_unit_size;
                                let filler = if filler_count == 1 {
                                    filler_unit
                                } else {
                                    self.spv_op_type_array(
                                        filler_unit,
                                        Some(filler_count),
                                        Some(NonZeroU32::new(filler_unit_size).unwrap()),
                                    )?
                                };
                                Ok((gap_offset, filler))
                            })
                        } else if is_last {
                            // HACK(eddyb) force the size of `OpTypeStruct`s that would be
                            // otherwise undersized (as e.g. `mem.copy` src/dst).
                            next_offset
                                .filter(|&size| size > field_range.unwrap_or(0..0).end)
                                .map(|size| Ok((size, self.spv_op_type_struct([], [])?)))
                        } else {
                            None
                        };

                        [field_type.map(|ty| ty.map(|ty| (field_offset, ty))), extra_field]
                            .into_iter()
                            .flatten()
                    });

                // HACK(eddyb) avoid creating redundant `OpTypeStruct`s.
                match [field_offsets_and_types.next(), field_offsets_and_types.next()] {
                    [Some(Ok((0, field_type))), None] => Ok(field_type),
                    first_fields => self.spv_op_type_struct(
                        first_fields.into_iter().flatten().chain(field_offsets_and_types),
                        [],
                    ),
                }
            }
            DataHappKind::Repeated { element, stride } => {
                let element_type =
                    self.pointee_type_for_data_happ(element, effective_flags, None)?;

                // FIXME(eddyb) can this occur legitimately, does it need handling?
                if disallow_padding && size_of(element_type) != Some(stride.get()) {
                    return Err(mk_padding_err());
                }

                let fixed_size = happ.max_size.or(max_size_allowed_by_shape);

                // HACK(eddyb) if the index can only be `0`, there's no reason
                // to keep an arbitrarily large stride.
                let stride = if let Some(size) = fixed_size.and_then(NonZeroU32::new)
                    && size < *stride
                {
                    size
                } else {
                    *stride
                };

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

                self.spv_op_type_array(element_type, fixed_len, Some(stride))
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

    // HACK(eddyb) `RefCell` to avoid `&mut` complications around `Builder` usage.
    var_use_counts: RefCell<EntityOrientedDenseMap<Var, NonZeroU32>>,

    // HACK(eddyb) this is used to avoid noise on top of `mem`/`qptr` diagnostics.
    func_has_mem_or_qptr_bug_diags: bool,
}

struct DeferredPtrNoop {
    // TODO(eddyb) replace mechanism!
    actually_noop: bool,

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
        func_at_data_inst: FuncAtMut<'_, DataInst>,
    ) -> Result<Transformed<DataInstDef>, LiftError> {
        let wk = self.lifter.wk;
        let cx = &self.lifter.cx;

        let data_inst = func_at_data_inst.position;

        // FIXME(eddyb) are there better names for this?
        let mut bld = Builder {
            cx,
            wk,
            func: func_at_data_inst.at(()),
            insert_aux_node: |func: FuncAtMut<'_, ()>, mut aux_node_def: DataInstDef| {
                // HACK(eddyb) account for `deferred_ptr_noops` interactions.
                self.resolve_deferred_ptr_noop_uses(&mut aux_node_def.inputs);
                self.add_value_uses(&aux_node_def.inputs);

                let aux_data_inst = func.nodes.define(cx, aux_node_def.into());

                // HACK(eddyb) can't really use helpers like `FuncAtMut::def`,
                // due to the need to borrow `regions` and `nodes`
                // at the same time - perhaps some kind of `FuncAtMut` position
                // types for "where a list is in a parent entity" could be used
                // to make this more ergonomic, although the potential need for
                // an actual list entity of its own, should be considered.
                func.regions[self.parent_region.unwrap()].children.insert_before(
                    aux_data_inst,
                    data_inst,
                    func.nodes,
                );

                aux_data_inst
            },
        };

        let data_inst_def = bld.func_at(data_inst).def();

        let replacement_data_inst_def = match &data_inst_def.kind {
            NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::ExitInvocation(_) => {
                return Ok(Transformed::Unchanged);
            }

            DataInstKind::Scalar(_) | DataInstKind::Vector(_) => return Ok(Transformed::Unchanged),

            &DataInstKind::FuncCall(_callee) => {
                for &v in &data_inst_def.inputs {
                    if self.lifter.as_spv_ptr_type(bld.type_of(v)).is_some() {
                        return Err(LiftError(Diag::bug([
                            "unimplemented calls with pointer args".into(),
                        ])));
                    }
                }
                return Ok(Transformed::Unchanged);
            }

            &DataInstKind::Mem(MemOp::FuncLocalVar(mem_layout)) => {
                // HACK(eddyb) reusing the same functionality meant for globals.
                let shape = shapes::GlobalVarShape::UntypedData(mem_layout);

                let output_attrs = bld.func_at(data_inst_def.outputs[0]).decl().attrs;

                let mem_accesses;
                let mem_accesses = match self.lifter.require_mem_accesses_attr(output_attrs) {
                    Ok(mem_accesses) => mem_accesses,
                    Err(e) => {
                        mem_accesses = self.lifter.fallback_accesses_from_shape(shape).ok_or(e)?;
                        &mem_accesses
                    }
                };

                let pointee_type =
                    self.lifter.pointee_type_for_shape_and_accesses(shape, mem_accesses)?;

                let mut data_inst_def = data_inst_def.clone();
                data_inst_def.kind = DataInstKind::SpvInst(
                    spv::Inst {
                        opcode: wk.OpVariable,
                        imms: [spv::Imm::Short(wk.StorageClass, wk.Function)].into_iter().collect(),
                    },
                    spv::InstLowering::default(),
                );
                let output_decl = bld.func.reborrow().at(data_inst_def.outputs[0]).decl();
                output_decl.attrs = self.lifter.strip_mem_accesses_attr(output_decl.attrs);
                output_decl.ty =
                    self.lifter.spv_ptr_type(AddrSpace::SpvStorageClass(wk.Function), pointee_type);
                data_inst_def
            }
            DataInstKind::QPtr(QPtrOp::HandleArrayIndex) => {
                let (addr_space, layout) =
                    self.type_of_val_as_spv_ptr_with_layout(bld.func_at(data_inst_def.inputs[0]))?;
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
                let output_decl = bld.func.reborrow().at(data_inst_def.outputs[0]).decl();
                output_decl.attrs = self.lifter.strip_mem_accesses_attr(output_decl.attrs);
                output_decl.ty = self.lifter.spv_ptr_type(addr_space, handle_type);
                data_inst_def
            }
            DataInstKind::QPtr(QPtrOp::BufferData) => {
                let buf_ptr = data_inst_def.inputs[0];
                let (addr_space, buf_layout) =
                    self.type_of_val_as_spv_ptr_with_layout(bld.func_at(buf_ptr))?;

                let buf_data_layout = match buf_layout {
                    TypeLayout::Handle(shapes::Handle::Buffer(_, buf)) => TypeLayout::Concrete(buf),
                    _ => return Err(LiftError(Diag::bug(["non-Buffer pointee".into()]))),
                };

                // FIXME(eddyb) avoid the repeated call to `type_of`,
                // maybe don't even replace the `QPtrOp::BufferData` instruction?
                let data_inst_def = data_inst_def.clone();

                let new_output_ty = bld.type_of(buf_ptr);
                let output_decl = bld.func.reborrow().at(data_inst_def.outputs[0]).decl();
                output_decl.ty = new_output_ty;

                self.deferred_ptr_noops.insert(
                    data_inst_def.outputs[0],
                    DeferredPtrNoop {
                        actually_noop: true,
                        output_pointer: buf_ptr,
                        output_pointer_addr_space: addr_space,
                        output_pointee_layout: buf_data_layout,
                        parent_region: self.parent_region.unwrap(),
                    },
                );

                data_inst_def
            }
            &DataInstKind::QPtr(QPtrOp::BufferDynLen { fixed_base_size, dyn_unit_stride }) => {
                let buf_ptr = data_inst_def.inputs[0];
                let (_, buf_layout) =
                    self.type_of_val_as_spv_ptr_with_layout(bld.func_at(buf_ptr))?;

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
                    .find_mem_accesses_attr(bld.func_at(data_inst_def.outputs[0]).decl().attrs)
                    .unwrap_or(&MemAccesses::Data(DataHapp::DEAD));

                let mut partial_offset = MaybeDynOffset::Const(0);
                let (output_pointer, (output_pointer_addr_space, output_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_accesses(
                        data_inst_def.inputs[0],
                        maybe_dyn_offset,
                        output_mem_accesses,
                        Some(&mut partial_offset),
                        &mut bld,
                    )?;

                // FIXME(eddyb) avoid the repeated call to `type_of_val`,
                // maybe don't even replace the original instruction?
                (data_inst_def.kind, data_inst_def.inputs) = match partial_offset {
                    MaybeDynOffset::Const(offset) => {
                        (QPtrOp::Offset(offset).into(), [output_pointer].into_iter().collect())
                    }
                    MaybeDynOffset::Dyn { index, stride, array_max_size } => {
                        (
                            QPtrOp::DynOffset {
                                stride,
                                index_bounds: array_max_size
                                    .and_then(|array_size| {
                                        // FIXME(eddyb) replace this when the `std` method stabilizes.
                                        let checked_exact_div =
                                            |a: u32, b: u32| a.is_multiple_of(b).then(|| a / b);
                                        checked_exact_div(array_size, stride.get())?.try_into().ok()
                                    })
                                    .map(|array_len| 0..array_len),
                            }
                            .into(),
                            [output_pointer, index].into_iter().collect(),
                        )
                    }
                };

                if partial_offset == MaybeDynOffset::Const(0) || true {
                    if partial_offset == MaybeDynOffset::Const(0) {
                        let new_output_ty = bld.type_of(output_pointer);
                        let output_decl = bld.func.reborrow().at(data_inst_def.outputs[0]).decl();
                        output_decl.ty = new_output_ty;
                    }

                    // FIXME(eddyb) not being able to reuse the original `DataInst`
                    // is a bit ridiculous, but correctly doing that would complicate
                    // `adjust_pointer_for_offset_and_accesses` in general.
                    self.deferred_ptr_noops.insert(
                        data_inst_def.outputs[0],
                        DeferredPtrNoop {
                            actually_noop: partial_offset == MaybeDynOffset::Const(0),
                            output_pointer,
                            output_pointer_addr_space,
                            output_pointee_layout,
                            parent_region: self.parent_region.unwrap(),
                        },
                    );
                }

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
                let access_type = stored_value.map_or_else(
                    || bld.func_at(data_inst_def.outputs[0]).decl().ty,
                    |v| bld.type_of(v),
                );
                let offset = *offset;

                // This access may need to be split into two or more accesses,
                // if it's not statically guaranteed to fit entirely within
                // one single scalar (found in the pointee type, at the offset).
                //
                // HACK(eddyb) the tracking for "2+ sub-accesses" consists of:
                // - `maybe_access_scalar`, indicating the next sub-access position
                //   within the overall loaded/stored (`access_type`-typed) value
                // - `loaded_partial_uint` (only for load `op`s), accumulating
                //   an unsigned integer value (of the same width as `access_type`)
                //   from previous sub-loads (i.e. before `maybe_access_scalar`)
                //   (store `op`s are simpler, as they can instead repeatedly slice
                //    the original `stored_value`, for every sub-store, as needed)
                let mut maybe_access_scalar = access_type.as_scalar(cx).and_then(|ty| {
                    Some(ScalarBitSlice {
                        ty,
                        width: scalar::IntWidth::try_from_bits(match ty {
                            // HACK(eddyb) this treats booleans as integers,
                            // sized by the `LayoutConfig`, at the cost of
                            // introducing conversion complications later.
                            scalar::Type::Bool => {
                                self.lifter.layout_cache.config.abstract_bool_size_align.0 * 8
                            }
                            _ => ty.bit_width(),
                        })?,
                        bit_offset: Value::Const(cx.intern(scalar::Const::from_u32(0))),
                        bit_offset_bounds: 0..=0,
                    })
                });
                let mut partially_loaded_uint = None;

                // FIXME(eddyb) this is awkward (or at least its needs DRY-ing)
                // because only an approximation is needed, most checks are
                // done by `adjust_pointer_for_offset_and_accesses`.
                let approx_mem_accesses = match self.lifter.layout_of(access_type)? {
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
                    TypeLayout::Concrete(concrete) => {
                        let ty = concrete.original_type;
                        // HACK(eddyb) shrink scalar accesses to mere bytes.
                        let (approx_ty, approx_size) = if maybe_access_scalar.is_some() {
                            (cx.intern(scalar::Type::UInt(scalar::IntWidth::I8)), Some(1))
                        } else {
                            (
                                ty,
                                (concrete.mem_layout.dyn_unit_stride.is_none())
                                    .then_some(concrete.mem_layout.fixed_base.size),
                            )
                        };
                        MemAccesses::Data(DataHapp {
                            max_size: approx_size,
                            flags: DataHappFlags::empty(),
                            kind: DataHappKind::Direct(approx_ty),
                        })
                    }
                };

                loop {
                    let offset = {
                        // FIXME(eddyb) implement big-endian accesses.
                        assert!(!self.lifter.layout_cache.config.is_big_endian);

                        let initial_offset = MaybeDynOffset::Const(offset.map_or(0, |o| o.get()));
                        let extra_offset = maybe_access_scalar.as_mut().map(|access_scalar| {
                            if let Value::Const(bit_offset) = access_scalar.bit_offset
                                && let Some(bit_offset) =
                                    bit_offset.as_scalar(cx).and_then(|ct| ct.int_as_u32())
                            {
                                assert!(
                                    bit_offset.is_multiple_of(8)
                                        && bit_offset < access_scalar.width.bits()
                                );
                                return MaybeDynOffset::Const((bit_offset / 8).try_into().unwrap());
                            }

                            // HACK(eddyb) avoid potentially out-of-bounds "0 bit"
                            // accesses from `bit_offset` being (dynamically)
                            // equal to the bit width of the whole access,
                            // by wrapping the (otherwise useless) last sub-access
                            // back around the overlap with the first sub-access.
                            let bit_offset_ty = bld.type_of(access_scalar.bit_offset);
                            let bit_offset_scalar_ty = bit_offset_ty.as_scalar(cx).unwrap();
                            let wrapped_bit_offset = bld.scalar_op(
                                scalar::IntBinOp::And,
                                [
                                    access_scalar.bit_offset,
                                    Value::Const(cx.intern(scalar::Const::from_bits(
                                        bit_offset_scalar_ty,
                                        (access_scalar.width.bits() - 1).into(),
                                    ))),
                                ],
                                bit_offset_ty,
                            );
                            access_scalar.bit_offset = wrapped_bit_offset;
                            MaybeDynOffset::Dyn {
                                index: bld.scalar_op(
                                    scalar::IntBinOp::DivU,
                                    [
                                        wrapped_bit_offset,
                                        Value::Const(cx.intern(scalar::Const::from_bits(
                                            bit_offset_scalar_ty,
                                            8,
                                        ))),
                                    ],
                                    bit_offset_ty,
                                ),
                                stride: NonZeroU32::new(1).unwrap(),
                                array_max_size: Some(access_scalar.bit_offset_bounds.end() / 8),
                            }
                        });
                        bld.offset_add(
                            initial_offset,
                            extra_offset.unwrap_or(MaybeDynOffset::Const(0)),
                        )?
                    };

                    let mut partial_offset = MaybeDynOffset::Const(0);
                    let (adjusted_ptr, (_, adjusted_pointee_layout)) = self
                        .adjust_pointer_for_offset_and_accesses(
                            data_inst_def.inputs[0],
                            offset,
                            &approx_mem_accesses,
                            Some(&mut partial_offset),
                            &mut bld,
                        )?;

                    // FIXME(eddyb) consider zeroing out `partial_offset` when
                    // `maybe_access_scalar` has a non-zero `bit_offset` itself,
                    // as even a dynamic `partial_offset` will end up being
                    // zero due to `offset` having been already rounded up to
                    // just after the memory scalar of the previous sub-access.

                    let (partial_bit_offset, partial_bit_offset_bounds) = match partial_offset {
                        // FIXME(eddyb) can negative offsets make sense here?
                        MaybeDynOffset::Const(offset) => {
                            let bit_offset = offset
                                .checked_mul(8)
                                .and_then(|bit_offset| bit_offset.try_into().ok())
                                .ok_or_else(|| {
                                    LiftError(Diag::bug([format!(
                                        "unsupported negative offset `{offset}`"
                                    )
                                    .into()]))
                                })?;
                            (
                                Value::Const(cx.intern(scalar::Const::from_u32(bit_offset))),
                                bit_offset..=bit_offset,
                            )
                        }
                        MaybeDynOffset::Dyn { index, stride, array_max_size } => {
                            let index_ty = bld.type_of(index);
                            let stride_in_bits =
                                cx.intern(
                                    index_ty
                                        .as_scalar(cx)
                                        .and_then(|index_ty| {
                                            scalar::Const::int_try_from_i128(
                                                index_ty,
                                                i128::from(stride.get()).checked_mul(8).unwrap(),
                                            )
                                        })
                                        .ok_or_else(|| {
                                            LiftError(Diag::bug([
                                        format!("`{stride} * 8` not representable in index type `")
                                            .into(),
                                        index_ty.into(),
                                        "`".into(),
                                    ]))
                                        })?,
                                );

                            let bit_offset = bld.scalar_op(
                                scalar::IntBinOp::Mul,
                                [index, Value::Const(stride_in_bits)],
                                index_ty,
                            );

                            let pointee_layout = match &adjusted_pointee_layout {
                                TypeLayout::Concrete(concrete) => concrete.mem_layout,
                                _ => unreachable!(),
                            };
                            let pointee_size = pointee_layout
                                .dyn_unit_stride
                                .is_none()
                                .then_some(pointee_layout.fixed_base.size);

                            // HACK(eddyb) intersect partial offset's 3 constraints:
                            // - must be less than `array_max_size`
                            // - must be less than the pointee's own size
                            //   (i.e. the access overlaps *at least* one pointee byte)
                            // - must be a multiple of `stride`
                            let max_partial_offset = [pointee_size, array_max_size]
                                .into_iter()
                                .flatten()
                                .reduce(|a, b| a.min(b))
                                .map(|size| size.saturating_sub(1) / stride * stride.get());

                            (
                                bit_offset,
                                0..=max_partial_offset
                                    .and_then(|o| o.checked_mul(8))
                                    .unwrap_or(u32::MAX),
                            )
                        }
                    };

                    let mem_type = match adjusted_pointee_layout {
                        TypeLayout::Handle(shapes::Handle::Opaque(ty)) => ty,
                        TypeLayout::Concrete(concrete) => concrete.original_type,
                        _ => unreachable!(),
                    };

                    let simple_direct_access = mem_type == access_type
                        && maybe_access_scalar
                            .as_ref()
                            .is_none_or(|access_scalar| access_scalar.bit_offset_bounds == (0..=0))
                        && partial_bit_offset_bounds == (0..=0);

                    if simple_direct_access {
                        data_inst_def.kind = access_op.into();
                        data_inst_def.inputs[0] = adjusted_ptr;

                        return Ok(Transformed::Changed(data_inst_def));
                    }

                    let (mem_uint, access_scalar) = (!simple_direct_access)
                        .then_some(())
                        .and_then(|()| {
                            let mem_scalar_type = mem_type.as_scalar(cx)?;

                            // HACK(eddyb) the simplest case to support requires
                            // unsigned integers (to allow zext w/o a bitcast).
                            let mem_uint_width = match mem_scalar_type {
                                scalar::Type::UInt(w) => w,
                                _ => return None,
                            };

                            let access_scalar = maybe_access_scalar.as_mut()?;

                            let (le_bit_offset, le_bit_offset_bounds) =
                                (partial_bit_offset, partial_bit_offset_bounds.clone());
                            let (mem_uint_bit_offset, mem_uint_bit_offset_bounds) =
                                if self.lifter.layout_cache.config.is_big_endian {
                                    // FIXME(eddyb) support big-endian w/ dynamic offset.
                                    // FIXME(eddyb) the logic below doesn't even
                                    // seem to support larger accesses.
                                    let le_bit_offset = match le_bit_offset {
                                        Value::Const(ct) => ct.as_scalar(cx)?.int_as_u32()?,
                                        Value::Var(_) => return None,
                                    };
                                    let be_bit_offset = mem_uint_width
                                        .bits()
                                        .checked_sub(access_scalar.width.bits())?
                                        .checked_sub(le_bit_offset)?;
                                    (
                                        Value::Const(
                                            cx.intern(scalar::Const::from_u32(be_bit_offset)),
                                        ),
                                        be_bit_offset..=be_bit_offset,
                                    )
                                } else {
                                    (le_bit_offset, le_bit_offset_bounds)
                                };

                            Some((
                                ScalarBitSlice {
                                    ty: mem_scalar_type,
                                    width: mem_uint_width,
                                    bit_offset: mem_uint_bit_offset,
                                    bit_offset_bounds: mem_uint_bit_offset_bounds,
                                },
                                access_scalar,
                            ))
                        })
                        .ok_or_else(|| {
                            LiftError(Diag::bug([
                                "expected access type `".into(),
                                access_type.into(),
                                "` not found in pointee type layout (found leaf: `".into(),
                                mem_type.into(),
                                "`)".into(),
                            ]))
                        })?;

                    // FIXME(eddyb) fix the naming/terminology here.
                    let access_scalar_for_next_subaccess = {
                        // FIXME(eddyb) implement big-endian accesses.
                        assert!(!self.lifter.layout_cache.config.is_big_endian);

                        // FIXME(eddyb) consider zeroing out `mem_uint.bit_offset`
                        // when `access_scalar.bit_offset` itself isn't zero.
                        let mem_bit_offset_bounds = if access_scalar.bit_offset_bounds != (0..=0) {
                            0..=0
                        } else {
                            mem_uint.bit_offset_bounds.clone()
                        };
                        let next_bit_offset_bounds = {
                            let prev_bounds = &access_scalar.bit_offset_bounds;
                            let [start, end] = [
                                (prev_bounds.start(), mem_bit_offset_bounds.end()),
                                (prev_bounds.end(), mem_bit_offset_bounds.start()),
                            ]
                            .map(|(prev_bound, &bit_offset_bound)| {
                                let extra =
                                    mem_uint.width.bits().checked_sub(bit_offset_bound).unwrap();
                                prev_bound
                                    .checked_add(extra)
                                    .unwrap()
                                    .min(access_scalar.width.bits())
                            });
                            start..=end
                        };

                        if *next_bit_offset_bounds.start() == access_scalar.width.bits() {
                            None
                        } else {
                            // FIXME(eddyb) use const-folding to avoid this special-case.
                            let subaccess_width = if let Value::Const(mem_bit_offset) =
                                mem_uint.bit_offset
                                && let Some(mem_bit_offset) =
                                    mem_bit_offset.as_scalar(cx).and_then(|ct| ct.int_as_u32())
                            {
                                Value::Const(cx.intern(scalar::Const::from_u32(
                                    mem_uint.width.bits().checked_sub(mem_bit_offset).unwrap(),
                                )))
                            } else {
                                let mem_bit_offset_ty = bld.type_of(mem_uint.bit_offset);
                                let mem_bit_offset_scalar_ty =
                                    mem_bit_offset_ty.as_scalar(cx).unwrap();
                                bld.scalar_op(
                                    scalar::IntBinOp::Sub,
                                    [
                                        Value::Const(cx.intern(scalar::Const::from_bits(
                                            mem_bit_offset_scalar_ty,
                                            mem_uint.width.bits().into(),
                                        ))),
                                        mem_uint.bit_offset,
                                    ],
                                    mem_bit_offset_ty,
                                )
                            };
                            // FIXME(eddyb) use const-folding to avoid this special-case.
                            let next_bit_offset =
                                if let (Value::Const(bit_offset), Value::Const(subaccess_width)) =
                                    (access_scalar.bit_offset, subaccess_width)
                                    && let Some(bit_offset) =
                                        bit_offset.as_scalar(cx).and_then(|ct| ct.int_as_u32())
                                    && let Some(subaccess_width) =
                                        subaccess_width.as_scalar(cx).and_then(|ct| ct.int_as_u32())
                                {
                                    Value::Const(cx.intern(scalar::Const::from_u32(
                                        bit_offset.checked_add(subaccess_width).unwrap(),
                                    )))
                                } else {
                                    let bit_offset_ty = bld.type_of(access_scalar.bit_offset);
                                    bld.scalar_op(
                                        scalar::IntBinOp::Add,
                                        [access_scalar.bit_offset, subaccess_width],
                                        bit_offset_ty,
                                    )
                                };
                            Some(ScalarBitSlice {
                                bit_offset: next_bit_offset,
                                bit_offset_bounds: next_bit_offset_bounds,
                                ..access_scalar.clone()
                            })
                        }
                    };

                    let new_kind_and_inputs = if let Some(stored_value) = stored_value {
                        let merged_value = bld.bit_sliced_update(
                            |rax| rax.op(MemOp::Load { offset: None }, [adjusted_ptr], mem_type),
                            mem_uint.clone(),
                            stored_value,
                            access_scalar.clone(),
                        );
                        let (store_op, store_inputs) =
                            (MemOp::Store { offset: None }, [adjusted_ptr, merged_value]);

                        // FIXME(eddyb) dedup with the load side?
                        if let Some(access_scalar_for_next_subaccess) =
                            access_scalar_for_next_subaccess
                        {
                            bld.op_without_output(store_op, store_inputs);
                            *access_scalar = access_scalar_for_next_subaccess;
                            continue;
                        }

                        (store_op.into(), store_inputs.into_iter().collect())
                    } else {
                        let raw_load =
                            bld.op(MemOp::Load { offset: None }, [adjusted_ptr], mem_type);

                        let sliced_load = bld.scalar_op(
                            scalar::IntBinOp::ShrU,
                            [raw_load, mem_uint.bit_offset],
                            mem_type,
                        );

                        let needs_multiple_loads = partially_loaded_uint.is_some()
                            || access_scalar_for_next_subaccess.is_some();

                        let complete_loaded_uint = if needs_multiple_loads {
                            let access_uint_ty = cx.intern(scalar::Type::UInt(access_scalar.width));
                            let sliced_load_as_uint = bld.scalar_op(
                                scalar::IntUnOp::TruncOrZeroExtend,
                                [sliced_load],
                                access_uint_ty,
                            );
                            // FIXME(eddyb) consider using `bit_sliced_update`?
                            let sliced_load_at_access_offset = bld.scalar_op(
                                scalar::IntBinOp::Shl,
                                [sliced_load_as_uint, access_scalar.bit_offset],
                                access_uint_ty,
                            );
                            // FIXME(eddyb) use auto-simplification to avoid this special-case.
                            // HACK(eddyb) using `Iterator::reduce` to keep this short.
                            let merged_loaded_uint =
                                [partially_loaded_uint, Some(sliced_load_at_access_offset)]
                                    .into_iter()
                                    .flatten()
                                    .reduce(|a, b| {
                                        bld.scalar_op(scalar::IntBinOp::Or, [a, b], access_uint_ty)
                                    })
                                    .unwrap();

                            // FIXME(eddyb) dedup with the store side?
                            if let Some(access_scalar_for_next_subaccess) =
                                access_scalar_for_next_subaccess
                            {
                                partially_loaded_uint = Some(merged_loaded_uint);
                                *access_scalar = access_scalar_for_next_subaccess;
                                continue;
                            }

                            merged_loaded_uint
                        } else {
                            sliced_load
                        };

                        let complete_loaded_uint_ty = bld.type_of(complete_loaded_uint);
                        let complete_loaded_uint_scalar_ty =
                            complete_loaded_uint_ty.as_scalar(cx).unwrap();
                        match access_scalar.ty {
                            scalar::Type::Bool => {
                                let bool_mask = Value::Const(cx.intern(scalar::Const::from_bits(
                                    complete_loaded_uint_scalar_ty,
                                    !0u128 >> (128 - access_scalar.width.bits()),
                                )));
                                let masked_loaded_uint = bld.scalar_op(
                                    scalar::IntBinOp::And,
                                    [complete_loaded_uint, bool_mask],
                                    complete_loaded_uint_ty,
                                );

                                (
                                    scalar::Op::IntBinary(scalar::IntBinOp::Ne).into(),
                                    [
                                        masked_loaded_uint,
                                        Value::Const(cx.intern(scalar::Const::from_bits(
                                            complete_loaded_uint_scalar_ty,
                                            0,
                                        ))),
                                    ]
                                    .into_iter()
                                    .collect(),
                                )
                            }
                            scalar::Type::SInt(_)
                                if access_scalar.width.bits()
                                    != complete_loaded_uint_scalar_ty.bit_width() =>
                            {
                                (
                                    scalar::Op::IntUnary(scalar::IntUnOp::TruncOrSignExtend).into(),
                                    [complete_loaded_uint].into_iter().collect(),
                                )
                            }
                            scalar::Type::UInt(_)
                                if access_scalar.width.bits()
                                    != complete_loaded_uint_scalar_ty.bit_width() =>
                            {
                                (
                                    scalar::Op::IntUnary(scalar::IntUnOp::TruncOrZeroExtend).into(),
                                    [complete_loaded_uint].into_iter().collect(),
                                )
                            }
                            // FIXME(eddyb) don't resort to a bitcast for non-floats,
                            // when the last `Or` of `complete_loaded_uint` could
                            // instead take over the original load instruction.
                            _ => {
                                let trunc_output = bld.scalar_op(
                                    scalar::IntUnOp::TruncOrZeroExtend,
                                    [complete_loaded_uint],
                                    cx.intern(scalar::Type::UInt(access_scalar.width)),
                                );

                                // FIXME(eddyb) SPIR-T should have its own bitcast.
                                (
                                    DataInstKind::SpvInst(
                                        wk.OpBitcast.into(),
                                        spv::InstLowering::default(),
                                    ),
                                    [trunc_output].into_iter().collect(),
                                )
                            }
                        }
                    };
                    (data_inst_def.kind, data_inst_def.inputs) = new_kind_and_inputs;
                    break data_inst_def;
                }
            }
            &DataInstKind::Mem(MemOp::Copy { size }) => {
                let mut data_inst_def = data_inst_def.clone();

                let max_size = Some(size.get());
                // FIXME(eddyb) `DataHappKind::Dead` might
                // make more sense data-structure wise, but it
                // risks potentially losing the `flags`.
                let kind = DataHappKind::Disjoint(Default::default());

                let mut dst_partial_offset = MaybeDynOffset::Const(0);
                let (dst_adjusted_ptr, (_, dst_adjusted_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_accesses(
                        data_inst_def.inputs[0],
                        MaybeDynOffset::Const(0),
                        &MemAccesses::Data(DataHapp {
                            max_size,
                            flags: DataHappFlags::COPY_DST,
                            kind: kind.clone(),
                        }),
                        Some(&mut dst_partial_offset),
                        &mut bld,
                    )?;

                let mut src_partial_offset = MaybeDynOffset::Const(0);
                let (src_adjusted_ptr, (_, src_adjusted_pointee_layout)) = self
                    .adjust_pointer_for_offset_and_accesses(
                        data_inst_def.inputs[1],
                        MaybeDynOffset::Const(0),
                        &MemAccesses::Data(DataHapp {
                            max_size,
                            flags: DataHappFlags::COPY_SRC,
                            kind: kind.clone(),
                        }),
                        Some(&mut src_partial_offset),
                        &mut bld,
                    )?;

                // FIXME(eddyb) this feels inefficient but there's not really
                // any way to maximize `adjust_pointer_for_offset_and_accesses`
                // reuse of intermediaries (also, can dedup later, in theory).
                let [(dst_ptr, dst_offset), (src_ptr, src_offset)] = [
                    (data_inst_def.inputs[0], dst_adjusted_ptr, dst_partial_offset),
                    (data_inst_def.inputs[1], src_adjusted_ptr, src_partial_offset),
                ]
                .map(|(original_ptr, adjusted_ptr, partial_offset)| {
                    let supported_offset = match partial_offset {
                        MaybeDynOffset::Const(offset) => Some(offset),
                        MaybeDynOffset::Dyn { .. } => None,
                    };
                    // HACK(eddyb) guarantee that all necessary additions succed.
                    let supported_offset = supported_offset.filter(|offset| {
                        i32::try_from(size.get())
                            .ok()
                            .and_then(|size| offset.checked_add(size))
                            .is_some()
                    });
                    supported_offset.map_or((original_ptr, 0), |offset| (adjusted_ptr, offset))
                });

                match (dst_adjusted_pointee_layout, src_adjusted_pointee_layout) {
                    (TypeLayout::Concrete(dst_concrete), TypeLayout::Concrete(src_concrete))
                        if dst_concrete.original_type == src_concrete.original_type
                            && dst_concrete.mem_layout.fixed_base.size == size.get()
                            && dst_concrete.mem_layout.dyn_unit_stride.is_none()
                            && (dst_ptr, dst_offset) == (dst_adjusted_ptr, 0)
                            && (src_ptr, src_offset) == (src_adjusted_ptr, 0) =>
                    {
                        (data_inst_def.kind, data_inst_def.inputs) = (
                            DataInstKind::SpvInst(
                                wk.OpCopyMemory.into(),
                                spv::InstLowering::default(),
                            ),
                            [dst_adjusted_ptr, src_adjusted_ptr].into_iter().collect(),
                        );
                        data_inst_def
                    }

                    // FIXME(eddyb) consider a loop (e.g. above a certain size),
                    // but there's a chance the pointee isn't (entirely) an array,
                    // so `mem::analyze` would have to be also adjusted.
                    _ => {
                        // HACK(eddyb) this is only needed because of `bld`
                        // borrowing `self` immutably, and it's easier to use
                        // `bld` to create the instructions in the first place.
                        let mut new_nodes = crate::EntityListIter { first: None, last: None };
                        let mut offset = 0;
                        while let Some(remaining_bytes @ 1..) = size.get().checked_sub(offset) {
                            // HACK(eddyb) prefer `u32` over `u16` over `u8`.
                            let copy_unit_size = 1 << remaining_bytes.trailing_zeros().clamp(0, 2);
                            let copy_unit = cx.intern(scalar::Type::UInt(
                                scalar::IntWidth::try_from_bits(copy_unit_size * 8).unwrap(),
                            ));

                            let (load_node, loaded_value) = bld.maybe_define_node(
                                MemOp::Load {
                                    offset: NonZeroI32::new(
                                        src_offset
                                            .checked_add(i32::try_from(offset).unwrap())
                                            .unwrap(),
                                    ),
                                },
                                [src_ptr],
                                Some(copy_unit),
                            );
                            let (store_node, _) = bld.maybe_define_node(
                                MemOp::Store {
                                    offset: NonZeroI32::new(
                                        dst_offset
                                            .checked_add(i32::try_from(offset).unwrap())
                                            .unwrap(),
                                    ),
                                },
                                [dst_ptr, loaded_value.unwrap()],
                                None,
                            );

                            new_nodes.first.get_or_insert(load_node.unwrap());
                            new_nodes.last = Some(store_node.unwrap());

                            offset += copy_unit_size;
                        }

                        // HACK(eddyb) this is responsible for transforming the
                        // above `load`+`store` pairs into the appropriate
                        // (sub)accesses that are supported by the pointee.
                        let mut iter = bld.func.at(new_nodes);
                        while let Some(mut func_at_node) = iter.next() {
                            // HACK(eddyb) originally defining the nodes will
                            // have invoked `add_value_uses`, so that has to
                            // be undone before `in_place_transform_node_def`
                            // re-adds them (to avoid them being double-counted).
                            self.remove_value_uses(&func_at_node.reborrow().def().inputs);

                            self.in_place_transform_node_def(func_at_node);
                        }

                        // FIXME(eddyb) it might be possible to avoid this kind
                        // of "tombstone", but it would complicate the above.
                        (data_inst_def.kind, data_inst_def.inputs) = (
                            DataInstKind::SpvInst(wk.OpNop.into(), spv::InstLowering::default()),
                            [].into_iter().collect(),
                        );
                        data_inst_def
                    }
                }
            }

            &DataInstKind::ThunkBind(_) => {
                for &v in &data_inst_def.inputs {
                    if self.lifter.as_spv_ptr_type(bld.type_of(v)).is_some() {
                        return Err(LiftError(Diag::bug([
                            "unsupported `thunk.bind` with pointer inputs".into(),
                        ])));
                    }
                }
                return Ok(Transformed::Unchanged);
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
                        .get_or_insert_with(|| bld.func_at(data_inst).def().clone());

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
                                    &mut bld,
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
                                bld.func.reborrow().at(data_inst_def.outputs[0]).decl();
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
        &self,
        mut ptr: Value,
        mut offset: MaybeDynOffset,
        target_accesses: &MemAccesses,

        // HACK(eddyb) find a better API, maybe wrap inputs/outputs of this
        // whole "adjustment" process into `struct`s etc.
        //
        // TODO(eddyb) also return the start of the pointer, somehow, to make
        // it easier to find the pointer *after* the pointee, without ending
        // up with redunandant offset math (or just do the redundant math).
        allow_partial_offsets_and_write_them_back_into: Option<&mut MaybeDynOffset>,

        bld: &mut Builder<'_, impl FnMut(FuncAtMut<'_, ()>, NodeDef) -> Node>,
    ) -> Result<(Value, (AddrSpace, TypeLayout)), LiftError> {
        let wk = self.lifter.wk;
        let cx = &self.lifter.cx;

        // FIXME(eddyb) this effectively duplicates parts of `qptr::legalize`,
        // but the choice for typed memory are provided by `mem::analyze`, which
        // must run on the output of `qptr::legalize`, so the best thing to do
        // would be to share some of the offset recombination logic.
        loop {
            let (base_ptr, base_maybe_dyn_offset) = match ptr {
                Value::Const(ct) => {
                    let ConstKind::PtrToGlobalVar { global_var, offset: Some(offset) } =
                        cx[ct].kind
                    else {
                        break;
                    };
                    (
                        Value::Const(cx.intern(ConstDef {
                            attrs: Default::default(),
                            ty: self.global_vars[global_var].type_of_ptr_to,
                            kind: ConstKind::PtrToGlobalVar { global_var, offset: None },
                        })),
                        MaybeDynOffset::Const(i32::try_from(offset.get()).ok().ok_or_else(
                            || {
                                LiftError(Diag::bug([format!(
                                    "{offset} not representable as a positive s32"
                                )
                                .into()]))
                            },
                        )?),
                    )
                }
                Value::Var(var) => {
                    let Either::Right(node) = bld.func.vars[var].def_parent else {
                        break;
                    };
                    let node_def = &bld.func.nodes[node];
                    let NodeKind::QPtr(offset_op @ (QPtrOp::Offset(_) | QPtrOp::DynOffset { .. })) =
                        &node_def.kind
                    else {
                        break;
                    };
                    (
                        node_def.inputs[0],
                        match offset_op {
                            &QPtrOp::Offset(offset) => MaybeDynOffset::Const(offset),
                            QPtrOp::DynOffset { stride, index_bounds } => MaybeDynOffset::Dyn {
                                index: node_def.inputs[1],
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
                        },
                    )
                }
            };

            offset = bld.offset_add(offset, base_maybe_dyn_offset)?;

            ptr = base_ptr;
        }

        let (addr_space, mut pointee_layout) =
            self.type_of_val_as_spv_ptr_with_layout(bld.func_at(ptr))?;

        let mut access_chain_inputs: SmallVec<[_; 8]> = [ptr].into_iter().collect();

        // HACK(eddyb) this is only used to work around a ridiculous edge case,
        // where `qptr.dyn_offset`, when used with a `qptr.buffer_data`, and if
        // fully deferred, would replace the `qptr.buffer_data` output with its
        // input (i.e. the whole buffer itself), breaking downstream uses.
        let original_ptr_before_misguided_resolve = access_chain_inputs[0];

        // HACK(eddyb) account for `deferred_ptr_noops` interactions.
        self.resolve_deferred_ptr_noop_uses(&mut access_chain_inputs);

        // HACK(eddyb) see the earlier comment on matching variable.
        let original_ptr_after_misguided_resolve = access_chain_inputs[0];

        // HACK(eddyb) disallowing naming the original `ptr` again.
        #[allow(unused)]
        let ptr = ();

        let access_chain_data_inst_kind =
            DataInstKind::SpvInst(wk.OpAccessChain.into(), spv::InstLowering::default());

        let mk_access_chain =
            |bld: &mut Builder<'_, _>, access_chain_inputs: SmallVec<_>, final_pointee_type| {
                if access_chain_inputs.len() > 1 {
                    bld.op(
                        access_chain_data_inst_kind.clone(),
                        access_chain_inputs,
                        self.lifter.spv_ptr_type(addr_space, final_pointee_type),
                    )
                } else {
                    let ptr = access_chain_inputs[0];

                    // HACK(eddyb) see comments on these variables.
                    if ptr == original_ptr_after_misguided_resolve {
                        original_ptr_before_misguided_resolve
                    } else {
                        ptr
                    }
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
                    mk_access_chain(bld, access_chain_inputs, pointee_handle_type),
                    (addr_space, TypeLayout::Handle(shapes::Handle::Opaque(pointee_handle_type))),
                ));
            }

            (TypeLayout::Concrete(pointee_layout), MemAccesses::Data(data_happ)) => {
                (pointee_layout, data_happ)
            }
        };

        // HACK(eddyb) helper for `if !target_fits_in_pointee` (see below).
        // TODO(eddyb) try disabling this by returning `None`, it might not be
        // needed anymore for anything.
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
            // FIXME(eddyb) should `MemTypeLayout` have have an `.extent()` method?
            let pointee_extent = Extent {
                start: 0,
                end: (pointee_layout.mem_layout.dyn_unit_stride.is_none())
                    .then_some(pointee_layout.mem_layout.fixed_base.size),
            };

            // HACK(eddyb) if a dynamic index could only be `0`, without going
            // outside of the bounds of the pointee, ignore the actual dynamic
            // value and replace it with just the constant offset of `0` bytes.
            // FIXME(eddyb) this could also generate an `assume index == 0`,
            // if SPIR-T had such a concept.
            // FIXME(eddyb) this assumes an "inbounds"-style offsetting operation.
            if let MaybeDynOffset::Dyn { stride, .. } = offset
                && pointee_extent.end.is_some_and(|size| stride.get() > size)
            {
                offset = MaybeDynOffset::Const(0);
            }

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
                        decompose_array_indexing(self, bld.func_at(access_chain_inputs[0]))?;

                    let array_index_ty = bld.type_of(array_index);
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
                            let index_scalar_ty = bld.type_of(index).as_scalar(cx)?;
                            if index_scalar_ty != array_index_scalar_ty {
                                return None;
                            }

                            (index, 0)
                        }
                    };

                    let combined_index = bld.scalar_op(
                        scalar::IntBinOp::Add,
                        [array_index, index_addend],
                        array_index_ty,
                    );

                    access_chain_inputs = [array_ptr, combined_index].into_iter().collect();
                    offset = MaybeDynOffset::Const(remainder_offset.try_into().unwrap());
                    pointee_layout = array_elem;

                    Some(())
                };
                if let Some(()) = maybe_recompose_dyn_indexing() {
                    continue;
                }
            }

            let has_compatible_offset = target_fits_in_pointee
                && (offset == MaybeDynOffset::Const(0)
                    || allow_partial_offsets_and_write_them_back_into.is_some());
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
            if let (
                Components::Elements { stride: array_stride, elem, .. },
                MaybeDynOffset::Dyn { index, stride: index_stride, .. },
            ) = (&pointee_layout.components, offset)
                && target_happ.max_size.is_some_and(|target_size| target_size <= index_stride.get())
            {
                // FIXME(eddyb) replace this when the `std` method stabilizes.
                let checked_exact_div = |a: u32, b: u32| a.is_multiple_of(b).then(|| a / b);

                let index_ty = bld.type_of(index);
                let index_typed_const = |x: u32| {
                    let ct = index_ty
                        .as_scalar(cx)
                        .and_then(|index_ty| scalar::Const::int_try_from_i128(index_ty, x.into()))
                        .ok_or_else(|| {
                            LiftError(Diag::bug([
                                format!("{x} not representable in index type `").into(),
                                index_ty.into(),
                                "`".into(),
                            ]))
                        })?;
                    Ok(Value::Const(cx.intern(ct)))
                };

                // HACK(eddyb) in the worst-case scenario, where neither of the
                // two strides (`array_stride` and `index_stride`) is a multiple
                // of the other, the `index` must be first be multiplied, such
                // that it's in terms of a smaller `common_stride`, that is also
                // a divisor of `array_stride`, allowing later divison+remainder.
                // FIXME(eddyb) dedup some of this with `offset_add`.
                let (index, index_stride) = if !array_stride
                    .get()
                    .is_multiple_of(index_stride.get())
                {
                    let common_stride = if index_stride.get().is_multiple_of(array_stride.get()) {
                        *array_stride
                    } else {
                        // HACK(eddyb) instead of computing GCD, just use
                        // the largest power of 2 they have in common.
                        NonZeroU32::new(
                            1 << array_stride.trailing_zeros().min(index_stride.trailing_zeros()),
                        )
                        .unwrap()
                    };
                    let index_multiplier =
                        checked_exact_div(index_stride.get(), common_stride.get()).unwrap();
                    (
                        bld.scalar_op(
                            scalar::IntBinOp::Mul,
                            [index, index_typed_const(index_multiplier)?],
                            index_ty,
                        ),
                        common_stride,
                    )
                } else {
                    (index, index_stride)
                };

                let index_divisor =
                    checked_exact_div(array_stride.get(), index_stride.get()).unwrap();

                // FIXME(eddyb) use auto-simplification to avoid this special-case.
                let (index, leftover_offset) = if index_divisor == 1 {
                    (index, MaybeDynOffset::Const(0))
                } else {
                    let index_divisor_value = index_typed_const(index_divisor)?;
                    (
                        bld.scalar_op(
                            scalar::IntBinOp::DivU,
                            [index, index_divisor_value],
                            index_ty,
                        ),
                        MaybeDynOffset::Dyn {
                            index: bld.scalar_op(
                                scalar::IntBinOp::RemS,
                                [index, index_divisor_value],
                                index_ty,
                            ),
                            stride: index_stride,
                            array_max_size: Some(array_stride.get()),
                        },
                    )
                };

                // HACK(eddyb) separate the `OpAccessChain`s into one for
                // obtaining the array pointer itself, and one for indexing
                // the array, to allow folding the latter in subsequent calls
                // to `adjust_pointer_for_offset_and_accesses`.
                // FIXME(eddyb) consider tracking representations of `qptr`s
                // that deviate from "`Value` of SPIR-V logical pointer type".
                // TODO(eddyb) try replacing this with `access_chain_inputs.push(index)`,
                // it might not be needed anymore for anything.
                let array_ptr =
                    mk_access_chain(bld, access_chain_inputs, pointee_layout.original_type);
                access_chain_inputs = [array_ptr, index].into_iter().collect();

                offset = leftover_offset;
                pointee_layout = elem.clone();

                continue;
            }

            let mut component_indices =
                pointee_layout.components.find_components_containing(target_extent);
            let idx = match (component_indices.next(), component_indices.next()) {
                (None, _) => {
                    // While none of the components fully contain `target_extent`,
                    // there's a good chance the pointer is already compatible
                    // with `target_happ` (and the only reason to keep going
                    // would be to find smaller types that remain compatible).
                    //
                    // TODO(eddyb) is the partial offset escape hatch correct??
                    if is_compatible
                        || offset == MaybeDynOffset::Const(0)
                        || allow_partial_offsets_and_write_them_back_into.is_some()
                    {
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
                        format!(
                            "ambiguity for offsets {target_extent} (due to ZSTs?), \
                             in the layout of "
                        )
                        .into(),
                        pointee_layout.original_type.into(),
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
            *writeback_offset = offset;
        }

        Ok((
            mk_access_chain(bld, access_chain_inputs, pointee_layout.original_type),
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
                    Some(ptr_noop) if ptr_noop.actually_noop => {
                        *v = ptr_noop.output_pointer;
                    }
                    _ => break,
                }
            }
        }
    }

    // FIXME(eddyb) these are only this whacky because an `u32` is being
    // encoded as `Option<NonZeroU32>` for (dense) map entry reasons.
    // HACK(eddyb) `&self` instead of `&mut self` to avoid complications around
    // `Builder` usage.
    fn add_value_uses(&self, values: &[Value]) {
        for &v in values {
            if let Value::Var(v) = v {
                let mut var_use_counts = self.var_use_counts.borrow_mut();
                let count = var_use_counts.entry(v);
                *count = Some(
                    NonZeroU32::new(count.map_or(0, |c| c.get()).checked_add(1).unwrap()).unwrap(),
                );
            }
        }
    }
    fn remove_value_uses(&self, values: &[Value]) {
        for &v in values {
            if let Value::Var(v) = v {
                let mut var_use_counts = self.var_use_counts.borrow_mut();
                let count = var_use_counts.entry(v);
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
        if let ConstKind::PtrToGlobalVar { global_var, offset } = ct_def.kind {
            let mut attrs = ct_def.attrs;
            let mut ty = ct_def.ty;

            // FIXME(eddyb) remove the cost of adding `Diag`s that will just be
            // ignored by `adjust_pointer_for_offset_and_accesses`.
            if let Some(offset) = offset {
                attrs.push_diag(
                    &self.lifter.cx,
                    Diag::bug([format!("NYI: global var immediate offset ({offset})").into()]),
                );
            } else {
                ty = self.global_vars[global_var].type_of_ptr_to;
            }
            Transformed::Changed(self.lifter.cx.intern(ConstDef {
                attrs,
                ty,
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
                // HACK(eddyb) doing this here can cause extra unwanted rewrites.
                // self.resolve_deferred_ptr_noop_uses(&mut data_inst_def.inputs);
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
                let is_used = self.var_use_counts.borrow().get(output_var).is_some();
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

// FIXME(eddyb) reconsider the name and placement of this.
struct Builder<'a, IAN: FnMut(FuncAtMut<'_, ()>, NodeDef) -> Node> {
    cx: &'a Context,
    wk: &'static spv::spec::WellKnown,

    func: FuncAtMut<'a, ()>,
    insert_aux_node: IAN,
}

#[derive(Clone)]
struct ScalarBitSlice {
    ty: scalar::Type,
    // FIXME(eddyb) is keeping this separate even needed?
    // NOTE(eddyb) this is the width of `ty` *not* the slice!
    width: scalar::IntWidth,

    bit_offset: Value,
    bit_offset_bounds: RangeInclusive<u32>,
}

impl<IAN: FnMut(FuncAtMut<'_, ()>, NodeDef) -> Node> Builder<'_, IAN> {
    // HACK(eddyb) `self.func.reborrow().freeze().at(position)` w/o `&mut self`.
    fn func_at<P: Copy>(&self, position: P) -> FuncAt<'_, P> {
        let FuncAtMut { regions, nodes, vars, position: () } = &self.func;
        FuncAt { regions, nodes, vars, position }
    }

    fn type_of(&self, v: Value) -> Type {
        self.func_at(v).type_of(self.cx)
    }

    fn op(
        &mut self,
        kind: impl Into<NodeKind>,
        inputs: impl IntoIterator<Item = Value>,
        output_type: Type,
    ) -> Value {
        self.maybe_define_node(kind, inputs, Some(output_type)).1.unwrap()
    }

    fn op_without_output(
        &mut self,
        kind: impl Into<NodeKind>,
        inputs: impl IntoIterator<Item = Value>,
    ) {
        self.maybe_define_node(kind, inputs, None);
    }

    fn maybe_define_node(
        &mut self,
        kind: impl Into<NodeKind>,
        inputs: impl IntoIterator<Item = Value>,
        output_type: Option<Type>,
    ) -> (Option<Node>, Option<Value>) {
        let node_def = NodeDef {
            // FIXME(eddyb) strongly consider copying debuginfo from the initial
            // instigating node (that resulted in the `Builder` being created).
            attrs: Default::default(),
            kind: kind.into(),
            inputs: inputs.into_iter().collect(),
            child_regions: [].into_iter().collect(),
            outputs: [].into_iter().collect(),
        };

        // HACK(eddyb) constant-folding here should be "free" (in the sense of
        // never defining any `Node`s or `Var`s when it succeeds) and it allows
        // simplifying the calling side.
        if let NodeKind::Scalar(scalar::Op::IntBinary(
            scalar::IntBinOp::Shl | scalar::IntBinOp::ShrS | scalar::IntBinOp::ShrU,
        )) = node_def.kind
            && let Value::Const(shift_amount) = node_def.inputs[1]
            && shift_amount.as_scalar(self.cx).and_then(|ct| ct.int_as_u32()) == Some(0)
        {
            return (None, Some(node_def.inputs[0]));
        }
        // FIXME(eddyb) this situation is technically illegal according to SPIR-V.
        if let NodeKind::Scalar(scalar::Op::IntUnary(
            scalar::IntUnOp::TruncOrSignExtend | scalar::IntUnOp::TruncOrZeroExtend,
        )) = node_def.kind
            && Some(self.type_of(node_def.inputs[0])) == output_type
        {
            return (None, Some(node_def.inputs[0]));
        }

        let node = (self.insert_aux_node)(self.func.reborrow(), node_def);

        let output_var = output_type.map(|output_type| {
            let output_var = self.func.vars.define(
                self.cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: output_type,
                    def_parent: Either::Right(node),
                    def_idx: 0,
                },
            );
            self.func.nodes[node].outputs.push(output_var);

            Value::Var(output_var)
        });
        (Some(node), output_var)
    }

    // FIXME(eddyb) this needs a more compact name.
    fn scalar_op(
        &mut self,
        op: impl Into<scalar::Op>,
        inputs: impl IntoIterator<Item = Value>,
        output_type: Type,
    ) -> Value {
        self.op(op.into(), inputs, output_type)
    }

    fn bit_sliced_update(
        &mut self,
        fetch_old_dst: impl FnOnce(&mut Self) -> Value,
        dst_slice: ScalarBitSlice,
        src: Value,
        src_slice: ScalarBitSlice,
    ) -> Value {
        let cx = self.cx;
        let wk = self.wk;

        // FIXME(eddyb) figure out if this restriction can/should be lifted,
        // and/or if it should be enforced at the type level.
        assert!(matches!(dst_slice.ty, scalar::Type::UInt(_)));

        let dst_type = self.cx.intern(dst_slice.ty);

        // FIXME(eddyb) consider adding a method for this on `IntWidth`.
        let mask = |w: scalar::IntWidth| !0u128 >> (128 - w.bits());

        // FIXME(eddyb) use const-folding to avoid this special-case.
        let dst_minus_src_mask = if let Value::Const(bit_offset) = dst_slice.bit_offset
            && let Some(bit_offset) = bit_offset.as_scalar(cx).and_then(|ct| ct.int_as_u32())
        {
            Value::Const(cx.intern(scalar::Const::from_bits(
                dst_slice.ty,
                mask(dst_slice.width) & !(mask(src_slice.width) << bit_offset),
            )))
        } else {
            let src_mask_in_dst = self.scalar_op(
                scalar::IntBinOp::Shl,
                [
                    Value::Const(cx.intern(scalar::Const::from_bits(
                        dst_slice.ty,
                        mask(dst_slice.width) & mask(src_slice.width),
                    ))),
                    dst_slice.bit_offset,
                ],
                dst_type,
            );
            self.scalar_op(scalar::IntUnOp::Not, [src_mask_in_dst], dst_type)
        };

        let masked_old_dst = if let Value::Const(dst_minus_src_mask) = dst_minus_src_mask
            && dst_minus_src_mask.as_scalar(cx).and_then(|ct| ct.int_as_u32()) == Some(0)
        {
            Value::Const(cx.intern(scalar::Const::from_bits(dst_slice.ty, 0)))
        } else {
            let old_dst = fetch_old_dst(self);
            self.scalar_op(scalar::IntBinOp::And, [old_dst, dst_minus_src_mask], dst_type)
        };

        let src_as_int = match src_slice.ty {
            // FIXME(eddyb) consider using a SPIR-T `Select` node.
            scalar::Type::Bool => self.op(
                DataInstKind::SpvInst(wk.OpSelect.into(), spv::InstLowering::default()),
                [src].into_iter().chain([true, false].map(|b| {
                    Value::Const(cx.intern(scalar::Const::from_bits(dst_slice.ty, b as u128)))
                })),
                dst_type,
            ),
            // FIXME(eddyb) try to remove some of the signedness edge cases.
            scalar::Type::UInt(_) => src,
            scalar::Type::SInt(_) if src_slice.width != dst_slice.width => src,
            // FIXME(eddyb) SPIR-T should have its own bitcast.
            _ => self.op(
                DataInstKind::SpvInst(wk.OpBitcast.into(), spv::InstLowering::default()),
                [src],
                cx.intern(scalar::Type::UInt(src_slice.width)),
            ),
        };

        let sliced_src_as_int = self.scalar_op(
            scalar::IntBinOp::ShrU,
            [src_as_int, src_slice.bit_offset],
            // FIXME(eddyb) maybe allow deducing types automatically instead?
            self.type_of(src_as_int),
        );

        let sliced_src_as_dst_uint =
            self.scalar_op(scalar::IntUnOp::TruncOrZeroExtend, [sliced_src_as_int], dst_type);

        let sliced_src_at_dst_offset = self.scalar_op(
            scalar::IntBinOp::Shl,
            [sliced_src_as_dst_uint, dst_slice.bit_offset],
            dst_type,
        );

        // FIXME(eddyb) use auto-simplification to avoid this special-case.
        if let Value::Const(masked_old_dst) = masked_old_dst
            && masked_old_dst.as_scalar(cx).and_then(|ct| ct.int_as_u32()) == Some(0)
        {
            sliced_src_at_dst_offset
        } else {
            self.scalar_op(
                scalar::IntBinOp::Or,
                [masked_old_dst, sliced_src_at_dst_offset],
                dst_type,
            )
        }
    }

    fn offset_add(
        &mut self,
        a: MaybeDynOffset,
        b: MaybeDynOffset,
    ) -> Result<MaybeDynOffset, LiftError> {
        // FIXME(eddyb) implement merging of arbitrary offsets.
        Ok(match (a, b) {
            (MaybeDynOffset::Const(0), offset) | (offset, MaybeDynOffset::Const(0)) => offset,
            (MaybeDynOffset::Const(a), MaybeDynOffset::Const(b)) => {
                MaybeDynOffset::Const(a.checked_add(b).ok_or_else(|| {
                    LiftError(Diag::bug([format!("{a} + {b} overflowed").into()]))
                })?)
            }

            _ => {
                // HACK(eddyb) approximating a `try {...}` block.
                let mut maybe_combine_offsets = || {
                    // HACK(eddyb) this is mostly copied from `qptr::legalize`.
                    #[derive(Copy, Clone)]
                    enum IndexValue {
                        Dyn(Value),
                        One,
                    }

                    let [a, b] = [a, b].map(|offset| {
                        Some(match offset {
                            MaybeDynOffset::Const(offset) => {
                                // FIXME(eddyb) allow negative offsets here.
                                let offset = u32::try_from(offset).ok()?;
                                (IndexValue::One, NonZeroU32::new(offset).unwrap(), Some(offset))
                            }
                            MaybeDynOffset::Dyn { index, stride, array_max_size } => {
                                (IndexValue::Dyn(index), stride, array_max_size)
                            }
                        })
                    });
                    let (a_index, a_stride, a_array_max_size) = a?;
                    let (b_index, b_stride, b_array_max_size) = b?;

                    let common_stride = {
                        let (min_stride, max_stride) =
                            (a_stride.min(b_stride), a_stride.max(b_stride));
                        if max_stride.get().is_multiple_of(min_stride.get()) {
                            min_stride
                        } else {
                            // HACK(eddyb) instead of computing GCD, just use
                            // the largest power of 2 they have in common.
                            NonZeroU32::new(
                                1 << a_stride.trailing_zeros().min(b_stride.trailing_zeros()),
                            )
                            .unwrap()
                        }
                    };

                    let index_ty = [a_index, b_index]
                        .into_iter()
                        .filter_map(|index| match index {
                            IndexValue::Dyn(index) => Some(self.type_of(index)),
                            IndexValue::One => None,
                        })
                        .dedup()
                        .exactly_one()
                        .ok()?;
                    let index_scalar_ty = index_ty.as_scalar(self.cx)?;

                    // HACK(eddyb) this overlaps with parts of `qptr::legalize`.
                    let [scaled_a_index, scaled_b_index] =
                        [(a_index, a_stride), (b_index, b_stride)].map(|(index, stride)| {
                            let scale = stride.get() / common_stride;
                            let scale_value = Value::Const(self.cx.intern(
                                scalar::Const::int_try_from_i128(index_scalar_ty, scale.into())?,
                            ));

                            Some(match index {
                                IndexValue::Dyn(index) => {
                                    // FIXME(eddyb) use auto-simplification to avoid this special-case.
                                    if scale == 1 {
                                        index
                                    } else {
                                        self.scalar_op(
                                            scalar::IntBinOp::Mul,
                                            [index, scale_value],
                                            index_ty,
                                        )
                                    }
                                }
                                IndexValue::One => scale_value,
                            })
                        });

                    Some(MaybeDynOffset::Dyn {
                        index: self.scalar_op(
                            scalar::IntBinOp::Add,
                            [scaled_a_index?, scaled_b_index?],
                            index_ty,
                        ),
                        stride: common_stride,
                        array_max_size: a_array_max_size
                            .and_then(|size| size.checked_add(b_array_max_size?)),
                    })
                };
                maybe_combine_offsets().ok_or_else(|| {
                    let fmt_offset = |offset| match offset {
                        MaybeDynOffset::Const(offset) => vec![format!("const (`{offset}`)").into()],
                        MaybeDynOffset::Dyn { index, stride, .. } => {
                            vec![
                                "dyn (`(N: ".into(),
                                self.type_of(index).into(),
                                format!(")×{stride}`)").into(),
                            ]
                        }
                    };
                    LiftError(Diag::bug(
                        (["failed to merge offsets: ".into()].into_iter())
                            .chain(fmt_offset(a))
                            .chain([" + ".into()])
                            .chain(fmt_offset(b)),
                    ))
                })?
            }
        })
    }
}
