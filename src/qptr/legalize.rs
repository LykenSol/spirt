//! [`QPtr`](crate::TypeKind::QPtr) legalization (e.g. for SPIR-V "logical addressing").
//!
//! # Memory vs pointers: semantics and limitations
//!
//! Memory semantics can usually be placed into one of two groups:
//! - "typed memory": accesses must strictly fit allocation ("object") types
//!   - often associated with GC-oriented languages/runtimes (including wasm GC),
//!     where the types tend to be limited to `struct`s and arrays (no `union`s),
//!     and serve to enforce memory safety, aid in precise GC, etc.
//!   - **SPIR-V "logical addressing"** is an unusual entry in this category,
//!     as other than resource handles (image/sampler/etc. "descriptors"),
//!     memory types are limited to plain data, buffers even using explicit layout
//!     (may eventually be rectified by extensions like `SPV_KHR_untyped_pointers`)
//! - "untyped memory": free-form accesses without any predeclared structure
//!   - underpins languages like C and Rust, IRs like LLVM and wasm, etc.
//!   - sometimes described as "array of bytes", though `[u8]` isn't accurate
//!     (see <https://www.ralfj.de/blog/2018/07/24/pointers-and-bytes.html>)
//!   - can be emulated on top of typed memory, if all values written to memory
//!     are representable through common types (i.e. integers, `u8` or larger),
//!     optionally combined with best-effort static analysis aka "type recovery"
//!
//! Pointer *values*, on top of accessing memory, can also be categorized by:
//! - granularity: from whole objects (GC refs) down to byte-level offsetting
//! - bitwise encoding: from hidden/read-only (GC refs) to mutably exposed bits
//!   - the actual choice of encoding is often a memory address, even if hidden
//!     (or e.g. a 32-bit offset from common base, aka "pointer compression")
//! - dynamic dataflow (`if`-`else`/loops/etc.): typically supported if types match
//!   (and even GC refs often allow type mixing through e.g. subtyping casts)
//! - indirect storage: like dynamic dataflow, but writing to compatible memory
//!   (at least for GC runtimes, this is *the* reason memory is typed at all)
//!
//! **SPIR-V "logical addressing"** is also an outlier here, disallowing what it
//! calls "variable pointers" (i.e. dynamic choice that is not array indexing),
//! even between same-address-space pointers *or pointing into the same variable*,
//! and offering no memory storage of pointers, not even relying on typed memory.
//!
//! # Legalization vs type recovery
//!
//! The worst-case semantic mismatch arises in SPIR-T between `qptr`-style
//! untyped memory and pointers (where even address spaces are erased), and
//! **SPIR-V "logical addressing"**, and that is tackled in two steps:
//! - [`legalize`](super::legalize) handles dynamic dataflow/indirect storage
//!   - chooses representations for both dynamic dataflow (locally, very flexible)
//!     and indirect storage (global encoding space, limited by pointer width)
//!   - the result is static usage: all pointers are global/local vars + offsets
//! - [`analyze`](super::analyze) and [`lift`](super::lift) handle untyped memory
//!   - may become (partially) unnecessary with e.g. `SPV_KHR_untyped_pointers`
//!   - greatly limited in scope/difficulty by complete legalization
// FIXME(eddyb) ^^ refactor `analyze`+`lift` to take advantage of this?
//!   - "type recovery" done on a best-effort basis, as most variables could use
//!     e.g. flat `[u32; N]` arrays (at some potential/unknown performance cost)
//!   - the result is typed memory accessed through SPIR-V typed pointers
//
// FIXME(eddyb) should the above docs go somewhere more general than `legalize`?

use super::QPtrOp;
use crate::func_at::{FuncAt, FuncAtMut};
use crate::transform::{InnerInPlaceTransform as _, Transformed, Transformer};
use crate::visit::Visitor;
use crate::{
    AttrSet, Const, ConstDef, ConstKind, Context, DeclDef, Diag, EntityOrientedDenseMap, Func,
    FuncDefBody, FxIndexMap, FxIndexSet, GlobalVar, Module, Node, NodeDef, NodeKind,
    NodeOutputDecl, Region, RegionInputDecl, Type, TypeKind, Value, cfg, scalar, spv,
};
use itertools::{Either, EitherOrBoth, Itertools};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::cell::Cell;
use std::collections::VecDeque;
use std::hash::Hash;
use std::num::{NonZeroI32, NonZeroU32, NonZeroU64};
use std::rc::Rc;
use std::{iter, mem};

// FIXME(eddyb) move this into some common utilities (or even obsolete its need).
#[derive(Default)]
struct ParentMap {
    node_parent: EntityOrientedDenseMap<Node, Region>,
    region_parent: EntityOrientedDenseMap<Region, Node>,
}

impl ParentMap {
    fn new(func_def_body: &FuncDefBody) -> Self {
        let mut parent_map = Self::default();

        // FIXME(eddyb) adopt this style of queue-based visiting in more places.
        let mut queue = VecDeque::new();
        queue.push_back(func_def_body.body);
        while let Some(region) = queue.pop_front() {
            for func_at_node in func_def_body.at(region).at_children() {
                parent_map.node_parent.insert(func_at_node.position, region);
                for &child_region in &func_at_node.def().child_regions {
                    parent_map.region_parent.insert(child_region, func_at_node.position);
                    queue.push_back(child_region);
                }
            }
        }

        parent_map
    }
}

pub struct LegalizePtrs {
    cx: Rc<Context>,
    wk: &'static spv::spec::WellKnown,

    cached_qptr_type: Cell<Option<Type>>,
    cached_null_qptr_const: Cell<Option<Const>>,
}

impl LegalizePtrs {
    pub fn new(cx: Rc<Context>) -> Self {
        Self { cx, wk: &spv::spec::Spec::get().well_known }
    }

    // TODO(eddyb) also handle `GlobalVar` initializers!
    pub fn legalize_all_funcs(
        &self,
        module: &mut Module,
        funcs: impl IntoIterator<Item = Func> + Clone,
    ) {
        let mut base_maps = BaseMaps::default();

        // See `ScanBasesInFunc` description of the same field.
        let mut write_back_escaped_ptr_offset_shape = None;

        // FIXME(eddyb) this might be too much state to keep around at once.
        struct FuncScanResults {
            // HACK(eddyb) is it worth storing vs re-computing?
            parent_map: ParentMap,

            uses_escaped_ptrs: bool,

            // HACK(eddyb) `base_maps.per_func[_].loop_body_input_base_choices_and_offset_shape`
            // is only fully finalized late due to needing to handle escaped cases.
            loop_body_inputs_using_escaped_ptrs: Vec<((Region, u32), (AnyEscapedBase, Offset<()>))>,
        }

        let mut per_func_scan_results = EntityOrientedDenseMap::new();

        for func in funcs.clone() {
            if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
                let parent_map = ParentMap::new(func_def_body);
                let mut scanner = ScanBasesInFunc {
                    legalizer: self,
                    parent_map: &parent_map,

                    escaped_base_map: &mut base_maps.escaped,
                    func_base_map: FuncLocalBaseMap::default(),

                    write_back_escaped_ptr_offset_shape: None,
                    summarized_ptrs: FxIndexMap::default(),
                };
                scanner.scan_func(func_def_body);

                if let Some((AnyEscapedBase, offset_shape)) =
                    scanner.write_back_escaped_ptr_offset_shape
                {
                    write_back_escaped_ptr_offset_shape
                        .get_or_insert((AnyEscapedBase, Offset::Zero))
                        .1
                        .merge_in_place(offset_shape);
                }

                let summarized_ptrs = scanner.summarized_ptrs;
                let mut func_base_map = scanner.func_base_map;
                let mut results = FuncScanResults {
                    parent_map,
                    uses_escaped_ptrs: false,
                    loop_body_inputs_using_escaped_ptrs: vec![],
                };

                // TODO(eddyb) skip canonicalization on functions where scanning
                // resulted in an empty `summarized_ptrs`!

                // FIXME(eddyb) move this into a separate method, maybe even a
                // new type that's a whole-module `Scanner`?
                for (ptr, PtrSummary { bases, offset_shape }) in summarized_ptrs {
                    let any_escaped = bases.as_ref().right().and_then(|u| u.any_escaped);
                    if let Some(AnyEscapedBase) = any_escaped {
                        results.uses_escaped_ptrs = true;

                        // Account for this pointer needing `BaseChoice::Dyn` once
                        // escaped bases are also included.
                        if let Some(&BaseChoice::Fixed(base)) = bases.as_ref().left() {
                            func_base_map.dyn_choice_bases.insert(base);
                        }
                    }

                    let Value::RegionInput { region, input_idx } = ptr else {
                        continue;
                    };
                    if results.parent_map.region_parent.get(region).is_none() {
                        continue;
                    }
                    func_base_map.loop_body_input_base_choices_and_offset_shape.extend(
                        bases.left().map(|bases| ((region, input_idx), (bases, offset_shape))),
                    );
                    results.loop_body_inputs_using_escaped_ptrs.extend(
                        any_escaped
                            .map(|any_escaped| ((region, input_idx), (any_escaped, offset_shape))),
                    );
                }

                base_maps.per_func.insert(func, func_base_map);
                per_func_scan_results.insert(func, results);
            }
        }

        if let Some((AnyEscapedBase, offset_shape)) = write_back_escaped_ptr_offset_shape {
            for escaped_offset_shape in base_maps.escaped.bases.values_mut() {
                escaped_offset_shape.merge_in_place(offset_shape);
            }
        }

        // FIXME(eddyb) is this step really necessary?
        let mut any_escaped_offset_shape = Offset::Zero;
        for &escaped_offset_shape in base_maps.escaped.bases.values() {
            any_escaped_offset_shape.merge_in_place(escaped_offset_shape);
        }

        // After scanning all functions, `base_maps.escaped` should be complete,
        // and can be added back to all functions that need it.
        for func in funcs.clone() {
            let Some(FuncScanResults {
                parent_map: _,
                uses_escaped_ptrs,
                loop_body_inputs_using_escaped_ptrs,
            }) = per_func_scan_results.get_mut(func)
            else {
                continue;
            };

            if !*uses_escaped_ptrs {
                assert!(loop_body_inputs_using_escaped_ptrs.is_empty());
                continue;
            }

            // TODO(eddyb) track `qptr.load`s of pointers (from `summarized_ptrs`),
            // and generate replacement integer loads+`switch`es for each of them,
            // so that it looks just like any other pointer-outputting `Select` node,
            // by the time it... well, I guess it could also be done on-demand,
            // then transformed as normal (esp. since it doesn't require rewrites).

            let escaped_bases = base_maps.escaped.bases.keys().copied().map(Base::Global);
            let func_base_map = &mut base_maps.per_func[func];
            func_base_map.dyn_choice_bases.extend(escaped_bases.clone());
            let base_to_base_idx =
                |base| func_base_map.dyn_choice_bases.get_index_of(&base).unwrap();

            if loop_body_inputs_using_escaped_ptrs.is_empty() || escaped_bases.len() == 0 {
                continue;
            }

            // Precompute a `BaseChoice` that already includes all escaped bases,
            // to cheaply combine with each loop body input's own `BaseChoice`.
            let escaped_base_choices = escaped_bases
                .map(BaseChoice::Fixed)
                .reduce(|a, b| a.merge(b, base_to_base_idx))
                .unwrap();
            for (loop_body_input, (AnyEscapedBase, mut offset_shape)) in
                loop_body_inputs_using_escaped_ptrs.drain(..)
            {
                // FIXME(eddyb) is this step really necessary?
                offset_shape.merge_in_place(any_escaped_offset_shape);

                func_base_map
                    .loop_body_input_base_choices_and_offset_shape
                    .entry(loop_body_input)
                    .and_modify(|(base_choices, merged_offset_shape)| {
                        // HACK(eddyb) stealing the existing set using a placeholder.
                        // FIXME(eddyb) ideally this could avoid cloning.
                        *base_choices = mem::replace(base_choices, BaseChoice::Dyn {
                            base_index_bitset: SmallVec::new(),
                            base_index_selector: (),
                        })
                        .merge(escaped_base_choices.clone(), base_to_base_idx);
                        merged_offset_shape.merge_in_place(offset_shape);
                    })
                    .or_insert_with(|| (escaped_base_choices.clone(), offset_shape));
            }
        }

        // TODO(eddyb) in theory, everything in `base_maps.per_func` could be
        // attached as attributes on the function/loop body inputs/etc.

        for func in funcs {
            if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
                let FuncScanResults {
                    mut parent_map,
                    uses_escaped_ptrs: _,
                    loop_body_inputs_using_escaped_ptrs,
                } = per_func_scan_results.remove(func).unwrap();
                assert!(loop_body_inputs_using_escaped_ptrs.is_empty());

                let func_base_map = &base_maps.per_func[func];

                // FIXME(eddyb) consider fusing local variables that show up in
                // `func_base_map.bases` into one big single variable, to reduce
                // the need for control-flow (using `summarized_ptrs` would allow
                // grouping them together for more granularity).

                CanonicalizePtrsInFunc {
                    legalizer: self,
                    parent_map: &mut parent_map,
                    canonicalized_ptrs: FxHashMap::default(),
                    value_replacements: FxHashMap::default(),
                }
                .in_place_transform_region_def(func_def_body.at_mut_body());
            }
        }
    }

    /// Get the (likely cached) `QPtr` type.
    fn qptr_type(&self) -> Type {
        if let Some(cached) = self.cached_qptr_type.get() {
            return cached;
        }
        let ty = self.cx.intern(TypeKind::QPtr);
        self.cached_qptr_type.set(Some(ty));
        ty
    }

    fn null_qptr_const(&self) -> Const {
        if let Some(cached) = self.cached_null_qptr_const.get() {
            return cached;
        }
        let ct = self.cx.intern(ConstDef {
            attrs: Default::default(),
            ty: self.qptr_type(),
            // FIXME(eddyb) maybe `qptr` should
            // have its own null constant?
            kind: ConstKind::SpvInst {
                spv_inst_and_const_inputs: Rc::new((
                    self.wk.OpConstantNull.into(),
                    [].into_iter().collect(),
                )),
            },
        });
        self.cached_null_qptr_const.set(Some(ct));
        ct
    }

    fn ptr_as_base(&self, func_at_ptr: FuncAt<'_, Value>) -> Option<Base> {
        match func_at_ptr.position {
            // FIXME(eddyb) implement more constant address pointers.
            Value::Const(ct) => match &self.cx[ct].kind {
                ConstKind::PtrToGlobalVar(_) => Some(GlobalBase::GlobalVar { ptr_to_var: ct }),
                ConstKind::SpvInst { spv_inst_and_const_inputs } => {
                    let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                    (spv_inst.opcode == self.wk.OpConstantNull && const_inputs.is_empty())
                        .then_some(GlobalBase::Null)
                }
                _ => None,
            }
            .map(Base::Global),

            Value::NodeOutput { node, output_idx: 0 } => {
                let node_def = func_at_ptr.at(node).def();
                match node_def.kind {
                    NodeKind::QPtr(QPtrOp::FuncLocalVar(_)) => Some(Base::FuncLocalVar(node)),
                    // FIXME(eddyb) implement.
                    NodeKind::QPtr(QPtrOp::HandleArrayIndex) => None,
                    NodeKind::QPtr(QPtrOp::BufferData) => match node_def.inputs[..] {
                        [Value::Const(ct)]
                            if matches!(self.cx[ct].kind, ConstKind::PtrToGlobalVar(_)) =>
                        {
                            Some(Base::Global(GlobalBase::BufferData { ptr_to_buffer: ct }))
                        }
                        _ => None,
                    },
                    _ => None,
                }
            }

            _ => None,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum GlobalBase {
    // FIXME(eddyb) support other (invalid) constant integer address pointers.
    Null,

    GlobalVar {
        // HACK(eddyb) `ConstKind::PtrToGlobalVar(gv)` is a more convenient key,
        // and just as cheap wrt hashing, than the `gv: GlobalVar` it references.
        ptr_to_var: Const,
    },

    // FIXME(eddyb) support `HandleArrayIndex`.
    BufferData {
        // HACK(eddyb) `ConstKind::PtrToGlobalVar(gv)` is a more convenient key,
        // and just as cheap wrt hashing, than the `gv: GlobalVar` it references.
        ptr_to_buffer: Const,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Base {
    Global(GlobalBase),
    FuncLocalVar(Node),
    // FIXME(eddyb) support `OpImageTexelPointer` and any other exotic pointers.
}

// FIXME(eddyb) this type doesn't do very much by itself.
#[derive(Default)]
struct BaseMaps {
    per_func: EntityOrientedDenseMap<Func, FuncLocalBaseMap>,
    escaped: EscapedBaseMap,
}

// FIXME(eddyb) in theory, something like union-find (or a much more complicated
// directed graph of "set inclusion" style relations) could be used to track
// disjoint "groups" of escaped bases, based on where they were written etc.
#[derive(Default)]
struct EscapedBaseMap {
    /// All bases of pointers that were ever written to memory, with their
    /// corresponding "offset shape" (`Offset<()>`) *of the written pointers*.
    ///
    /// The reason only writes are relevant is that encoding escaped pointers
    /// (into integers) only requires knowing the values that will be encoded,
    /// *not* how they might be used *after* being decoded into the per-function
    /// set of bases and appropriate common-strided offset - which may require
    /// multiplication (i.e. complicating decoding to allow better encodings).
    //
    // FIXME(eddyb) optimize this definition, taking into account disjoint
    // uses of memory, pointers passed between functions, etc.
    bases: FxIndexMap<GlobalBase, Offset<()>>,
}

#[derive(Default)]
struct FuncLocalBaseMap {
    /// All possible bases that `BaseChoice::Dyn` can refer to.
    dyn_choice_bases: FxIndexSet<Base>,

    /// Loop body inputs (of pointer type) are the only case where we can't rely
    /// on having legalized a definition before all of its uses, as one of the
    /// two sources for body inputs is the previous iteration's *body outputs*,
    /// which themselves can depend on previous body inputs, and this relationship
    /// forms a fixpoint-style construct that can only be handled separately.
    //
    // FIXME(eddyb) surely `(BaseChoice<_>, Offset<_>)` should be its own type?
    loop_body_input_base_choices_and_offset_shape:
        FxIndexMap<(Region, u32), (BaseChoice<()>, Offset<()>)>,
}

/// Non-empty set of `Base`s, carrying all dynamic information necessary to pick
/// one specific base when multiple are possible (i.e. `BaseChoice::Dyn`), and
/// also generic over the value type `V`, so that e.g. `BaseChoice<()>` can be
/// used to e.g. abstractly track which bases a pointer *may* be using.
//
// FIXME(eddyb) support `HandleArrayIndex` as well, somehow.
#[derive(Clone)]
enum BaseChoice<V> {
    Fixed(Base),
    Dyn {
        /// Ad-hoc bitset, with each bit index corresponding to the `Base` at
        /// the same index in `FuncLocalBaseMap`'s `dyn_choice_bases` set.
        //
        // FIXME(eddyb) this may be a performance hazard above 64 bases.
        // FIXME(eddyb) consider `Rc` for the non-small case.
        base_index_bitset: SmallVec<[u64; 1]>,

        /// Dynamic integer value (of type `u32`) choosing a base index, from
        /// any in `base_index_bitset` (other values cause undefined behavior).
        base_index_selector: V,
    },
}

// FIXME(eddyb) generalize this.
impl BaseChoice<()> {
    fn insert(&mut self, base: Base, mut base_to_base_idx: impl FnMut(Base) -> usize) {
        let prev_base = match self {
            &mut BaseChoice::Fixed(prev_base) => {
                if prev_base == base {
                    return;
                }
                *self =
                    BaseChoice::Dyn { base_index_bitset: SmallVec::new(), base_index_selector: () };
                Some(prev_base)
            }
            BaseChoice::Dyn { .. } => None,
        };
        let BaseChoice::Dyn { base_index_bitset: chunks, base_index_selector: () } = self else {
            unreachable!();
        };
        for base in prev_base.into_iter().chain([base]) {
            let i = base_to_base_idx(base);
            let (chunk_idx, bit_mask) = (i / 64, 1 << (i % 64));
            if chunk_idx >= chunks.len() {
                chunks.resize(chunk_idx + 1, 0);
            }
            chunks[chunk_idx] |= bit_mask;
        }
    }

    fn merge(self, other: Self, base_to_base_idx: impl FnMut(Base) -> usize) -> Self {
        match (self, other) {
            (BaseChoice::Fixed(base), set) | (set, BaseChoice::Fixed(base)) => {
                let mut merged = set;
                merged.insert(base, base_to_base_idx);
                merged
            }

            (
                BaseChoice::Dyn { base_index_bitset: a, base_index_selector: () },
                BaseChoice::Dyn { base_index_bitset: b, base_index_selector: () },
            ) => {
                let (mut dst, src) = if a.len() > b.len() { (a, b) } else { (b, a) };
                for (dst, src) in dst.iter_mut().zip(src) {
                    *dst |= src;
                }
                BaseChoice::Dyn { base_index_bitset: dst, base_index_selector: () }
            }
        }
    }
}

/// A pointer offset (relative to some `Base`), generic over the value type `V`,
/// so that e.g. `Offset<()>` can be used as an "offset shape".
//
// FIXME(eddyb) does this need a better name?
// FIXME(eddyb) support constant offsets and/or track offset ranges as well.
// FIXME(eddyb) should this be moved further up? (feels too chaotic)
#[derive(Copy, Clone, Default, PartialEq, Eq)]
enum Offset<V> {
    #[default]
    Zero,

    /// When applied to some pointer `ptr`, equivalent to
    /// `QPtrOp::DynOffset { stride }` with `[ptr, index]` as inputs.
    //
    // FIXME(eddyb) track index bounds and/or a fixed offset, as well?
    // FIXME(eddyb) does this need a better name? (`Dyn` is kind of overused).
    Dyn { stride: NonZeroU32, index: V },
}

// HACK(eddyb) helper for `Offset::merge`.
#[derive(Copy, Clone)]
struct Scaled<V> {
    value: V,
    multiplier: NonZeroU32,
}

impl<V> Scaled<V> {
    fn map_value<V2>(self, f: impl FnOnce(V) -> V2) -> Scaled<V2> {
        let Scaled { value, multiplier } = self;
        Scaled { value: f(value), multiplier }
    }
}

impl<V> Offset<V> {
    fn map_value<V2>(self, f: impl FnOnce(V) -> V2) -> Offset<V2> {
        match self {
            Offset::Zero => Offset::Zero,
            Offset::Dyn { stride, index } => Offset::Dyn { stride, index: f(index) },
        }
    }

    /// Merge `Offset`s, resolving stride conflicts between two `Offset::Dyn`s
    /// (with strides `a` vs `b`) by computing a "common stride" `c` such that
    /// `a` and `b` are multiples of `c`, which could be satisfied by the GCD
    /// (greatest common divisor) of `a` and `b`, but the approach taken here
    /// is to use the greatest common *power of two* divisor instead, which is
    /// both cheaper to compute, and more likely to end up being (related to)
    /// the smallest unit of access *anyway*.
    fn merge<V2, V3>(
        self,
        other: Offset<V2>,
        map_self: impl FnOnce(V) -> V3,
        map_other: impl FnOnce(V2) -> V3,
        merge_both: impl FnOnce(Scaled<V>, Scaled<V2>) -> V3,
    ) -> Offset<V3> {
        match (self, other) {
            (Offset::Zero, Offset::Zero) => Offset::Zero,
            (Offset::Dyn { stride, index }, Offset::Zero) => {
                Offset::Dyn { stride, index: map_self(index) }
            }
            (Offset::Zero, Offset::Dyn { stride, index }) => {
                Offset::Dyn { stride, index: map_other(index) }
            }
            (
                Offset::Dyn { stride: a_stride, index: a_index },
                Offset::Dyn { stride: b_stride, index: b_index },
            ) => {
                let stride = if a_stride == b_stride {
                    a_stride
                } else {
                    NonZeroU32::new(1 << a_stride.trailing_zeros().min(b_stride.trailing_zeros()))
                        .unwrap()
                };
                let multiplier_from = |orig_stride: NonZeroU32| {
                    assert_eq!(orig_stride.get() % stride.get(), 0);
                    NonZeroU32::new(orig_stride.get() / stride.get()).unwrap()
                };
                Offset::Dyn {
                    stride,
                    index: merge_both(
                        Scaled { value: a_index, multiplier: multiplier_from(a_stride) },
                        Scaled { value: b_index, multiplier: multiplier_from(b_stride) },
                    ),
                }
            }
        }
    }
}

impl Offset<()> {
    fn merge_in_place(&mut self, other: Self) {
        *self = self.merge(other, |()| (), |()| (), |_, _| ());
    }
}

impl<V> Offset<Option<V>> {
    fn transpose_value(self) -> Option<Offset<V>> {
        match self {
            Offset::Zero => Some(Offset::Zero),
            Offset::Dyn { stride, index } => Some(Offset::Dyn { stride, index: index? }),
        }
    }
}

impl<V, E> Offset<Result<V, E>> {
    fn transpose_value(self) -> Result<Offset<V>, E> {
        match self {
            Offset::Zero => Ok(Offset::Zero),
            Offset::Dyn { stride, index } => Ok(Offset::Dyn { stride, index: index? }),
        }
    }
}

struct ScanBasesInFunc<'a> {
    legalizer: &'a LegalizePtrs,
    parent_map: &'a ParentMap,

    escaped_base_map: &'a mut EscapedBaseMap,
    func_base_map: FuncLocalBaseMap,

    /// If any escaping pointer writes (e.g. `qptr.store` with pointer-typed input)
    /// write pointers that are themselves based on previously escaped pointers
    /// read from memory (i.e. `AnyEscapedBase`), their combined effect is
    /// tracked here.
    write_back_escaped_ptr_offset_shape: Option<(AnyEscapedBase, Offset<()>)>,

    // FIXME(eddyb) how expensive is this cache? (esp. wrt `BaseChoice`)
    // HACK(eddyb) this technically replicates part of `cyclotron::bruteforce`,
    // but without the fixpoint saturation (going for a more any-way traversal).
    // FIXME(eddyb) it's a shame this information gets thrown out and recomputed
    // later, but aspects like `AnyEscapedBase` would require a lot of patching.
    // HACK(eddyb) this is iterable because it's also used to generate all
    // of `loop_body_input_base_choices_and_offset_shape` after the fact, since
    // the case of *only* having `AnyEscapedBase` is not directly representable,
    // and more generally this allows detecting when this function is using any
    // escaped pointers at all (which requires including all escaped `bases`).
    summarized_ptrs: FxIndexMap<Value, PtrSummary>,
}

// HACK(eddyb) marker type used to indicate that a pointer's base could be *any*
// of `EscapedBaseMap`'s *final* `bases` (after all functions have been scanned).
// FIXME(eddyb) in theory, something like union-find (or a much more complicated
// directed graph of "set inclusion" style relations) could be used to track
// disjoint "groups" of escaped bases, based on where they were written etc.
#[derive(Copy, Clone)]
struct AnyEscapedBase;

#[derive(Clone)]
struct PtrSummary {
    bases: EitherOrBoth<BaseChoice<()>, UnknownBases>,
    offset_shape: Offset<()>,
}

/// Summary of pointer bases that couldn't be resolved to a `Base` right away.
//
// FIXME(eddyb) this has the kind of problem `EitherOrBoth` solves, except for
// a set of 3 fields, which shouldn't be able to be `(None, [], None)`.
#[derive(Clone)]
struct UnknownBases {
    any_escaped: Option<AnyEscapedBase>,

    /// Only used while summarizing a loop body input, to detect cyclic recursion,
    /// and to aid in tracking all the potential pointers flowing into the loop
    /// (i.e. this needs to be accurate, not just signaling an error).
    ///
    /// To prevent excessive cloning, `PtrSummarys` are only cached when
    /// `loop_body_input_deps` contains at most a single `(region, input_idx)`
    /// entry, which `summarize_ptr` can use to determine a retry is necessary,
    /// based on `summarized_ptrs[&Value::RegionInput { region, input_idx }]`
    /// losing its own `loop_body_input_deps` (indicating it's completed).
    //
    // FIXME(eddyb) this really wants to be `FxIndexSet` beyond a certain size.
    loop_body_input_deps: SmallVec<[(Region, u32); 1]>,

    /// Unknown/unsupported pointer values encountered, for which errors
    /// need to be later applied (as summarizing cannot mutate the function).
    unsupported: Option<UnsupportedBases>,
}

// HACK(eddyb) only separate for the field/documentation.
#[derive(Clone)]
struct UnsupportedBases {
    /// To avoid excessive errors, `directly_used` values are preserved only for
    /// the erroring pointer (i.e. `summarize_ptr(ptr)` with `directly_used == [ptr]`),
    /// and its direct users, but the latter only keep `directly_used` intact in
    /// `summarized_ptrs`, `summarize_ptr` clearing it before returning to further users.
    directly_used: SmallVec<[Value; 1]>,
}

impl ScanBasesInFunc<'_> {
    // HACK(eddyb) mutable access to the function is only for attaching `Diag`s
    // and the (artificial) `scan_ptr` vs `summarize_ptr` distinction.
    fn scan_func(&mut self, func_def_body: &mut FuncDefBody) {
        let cx = &self.legalizer.cx;

        {
            // FIXME(eddyb) adopt this style of queue-based visiting in more places.
            let mut queue = VecDeque::new();
            queue.push_back(func_def_body.body);
            while let Some(region) = queue.pop_front() {
                let mut func_at_children = func_def_body.at_mut(region).at_children().into_iter();
                while let Some(mut func_at_node) = func_at_children.next() {
                    queue.extend(&func_at_node.reborrow().def().child_regions);
                    self.scan_node(func_at_node);
                }
            }
        }

        // HACK(eddyb) some `summarized_ptrs` entries may only correspond to
        // caching of e.g. `QPtrOp` outputs during loop body input summarization,
        // on which `scan_node` never attempts to call `scan_ptrs` on, so they
        // get left in their `loop_body_input_deps`-carrying transient state,
        // which can just be dropped from the cache.
        self.summarized_ptrs.retain(|&ptr, summary| {
            // HACK(eddyb) opportunistically generate diagnostics here as well.
            if let Some(unsupported) =
                (summary.bases.as_ref().right().and_then(|u| u.unsupported.as_ref()))
                    .filter(|u| !u.directly_used.is_empty())
            {
                let mut generic_diag_added = false;
                for &used_ptr in &unsupported.directly_used {
                    let is_def = used_ptr == ptr;
                    let attach_on_def =
                        matches!(used_ptr, Value::NodeOutput { node: _, output_idx: 0 });
                    if is_def != attach_on_def {
                        continue;
                    }

                    let attrs = match ptr {
                        // FIXME(eddyb) this may become possible with constants
                        // that transform other constants (e.g. constexpr GEP).
                        Value::Const(_) => unreachable!(),

                        Value::RegionInput { region, input_idx } => {
                            &mut func_def_body.regions[region].inputs[input_idx as usize].attrs
                        }
                        Value::NodeOutput { node, output_idx } => {
                            // FIXME(eddyb) consider attaching diagnostics to
                            // the entire `node`, in some cases.
                            &mut func_def_body.nodes[node].outputs[output_idx as usize].attrs
                        }
                    };
                    // FIXME(eddyb) provide more information.
                    if let Value::Const(ct) = used_ptr {
                        attrs.push_diag(
                            cx,
                            Diag::bug(["unsupported pointer `".into(), ct.into(), "`".into()]),
                        );
                    } else if !generic_diag_added {
                        attrs.push_diag(cx, Diag::bug(["unsupported pointer".into()]));
                        generic_diag_added = true;
                    }
                }
            }

            let incomplete =
                summary.bases.as_ref().right().is_some_and(|u| !u.loop_body_input_deps.is_empty());
            if incomplete {
                let is_qptr_offset_op = match ptr {
                    Value::NodeOutput { node, output_idx: 0 } => matches!(
                        func_def_body.at(node).def().kind,
                        NodeKind::QPtr(QPtrOp::Offset(_) | QPtrOp::DynOffset { .. })
                    ),
                    _ => false,
                };
                assert!(is_qptr_offset_op);
            }
            !incomplete
        });
    }

    fn scan_node(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        let cx = &self.legalizer.cx;

        let is_qptr = |ty: Type| matches!(cx[ty].kind, TypeKind::QPtr);

        let node = func_at_node.position;
        let node_def = func_at_node.reborrow().def();
        match &node_def.kind {
            // FIXME(eddyb) consider attaching diagnostics here (via `FuncAtMut`)?
            NodeKind::Select(_) => {
                let num_outputs = node_def.outputs.len();

                let mut func = func_at_node.at(());
                for output_idx in 0..num_outputs {
                    if is_qptr(func.reborrow().at(node).def().outputs[output_idx].ty) {
                        self.scan_ptr(func.reborrow().at(Value::NodeOutput {
                            node,
                            output_idx: output_idx.try_into().unwrap(),
                        }));
                    }
                }
            }

            NodeKind::Loop { .. } => {
                let body = node_def.child_regions[0];

                let mut func = func_at_node.at(());
                let num_body_inputs = func.reborrow().at(body).def().inputs.len();
                for body_input_idx in 0..num_body_inputs {
                    if is_qptr(func.reborrow().at(body).def().inputs[body_input_idx].ty) {
                        self.scan_ptr(func.reborrow().at(Value::RegionInput {
                            region: body,
                            input_idx: body_input_idx.try_into().unwrap(),
                        }));
                    }
                }
            }

            NodeKind::QPtr(QPtrOp::Load { .. }) => {
                if is_qptr(node_def.outputs[0].ty) {
                    self.scan_ptr(func_at_node.at(Value::NodeOutput { node, output_idx: 0 }));
                }
            }
            NodeKind::QPtr(QPtrOp::Store { .. }) => {
                let stored_value = node_def.inputs[0];

                let mut func = func_at_node.at(());
                if is_qptr(func.reborrow().freeze().at(stored_value).type_of(cx)) {
                    let PtrSummary { bases, offset_shape } =
                        self.summarize_ptr(func.reborrow().freeze().at(stored_value));
                    if let Some(unknowns) = bases.as_ref().right() {
                        let UnknownBases { any_escaped, loop_body_input_deps, unsupported: _ } =
                            unknowns;
                        if let Some(AnyEscapedBase) = any_escaped {
                            self.write_back_escaped_ptr_offset_shape
                                .get_or_insert((AnyEscapedBase, Offset::Zero))
                                .1
                                .merge_in_place(offset_shape);
                        }
                        assert!(loop_body_input_deps.is_empty());
                    }
                    let bases = bases.left().into_iter().flat_map(|bases| match bases {
                        BaseChoice::Fixed(bases) => Either::Left([bases].into_iter()),
                        BaseChoice::Dyn { base_index_bitset, .. } => Either::Right(
                            // FIXME(eddyb) move this into a `BaseChoice` method
                            // (or really it should be using a `SmallBitSet`).
                            base_index_bitset
                                .into_iter()
                                .enumerate()
                                .flat_map(|(chunk_idx, mut chunk)| {
                                    let mut i = chunk_idx * 64;
                                    iter::from_fn(move || {
                                        let skip =
                                            NonZeroU64::new(chunk)?.trailing_zeros() as usize;
                                        i += skip;
                                        chunk >>= skip;
                                        let r = Some(i);

                                        // HACK(eddyb) advancing past the `1` bit is
                                        // done separately to avoid `chunk >>= 64`.
                                        i += 1;
                                        chunk >>= 1;

                                        r
                                    })
                                })
                                .map(|base_idx| self.func_base_map.dyn_choice_bases[base_idx]),
                        ),
                    });
                    for base in bases {
                        match base {
                            Base::Global(global_base) => {
                                self.escaped_base_map
                                    .bases
                                    .entry(global_base)
                                    .or_insert(Offset::Zero)
                                    .merge_in_place(offset_shape);
                            }
                            // FIXME(eddyb) support by either lifting the local
                            // variable to a new global (`Private`) one, and/or
                            // relying on the recursion emulator to put it on its
                            // stack (which itself would be a global base).
                            Base::FuncLocalVar(_) => {
                                func.reborrow().at(node).def().attrs.push_diag(
                                    cx,
                                    Diag::bug(["local variable written to memory".into()]),
                                );
                            }
                        }
                    }
                }
            }

            NodeKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(_))
            | NodeKind::Scalar(_)
            | NodeKind::Vector(_)
            | NodeKind::FuncCall { .. }
            | NodeKind::QPtr(
                QPtrOp::FuncLocalVar(_)
                | QPtrOp::HandleArrayIndex
                | QPtrOp::BufferData
                | QPtrOp::BufferDynLen { .. }
                | QPtrOp::Offset(_)
                | QPtrOp::DynOffset { .. }
                | QPtrOp::Copy { .. },
            )
            | NodeKind::SpvInst(..)
            | NodeKind::SpvExtInst { .. } => {}
        }
    }

    // HACK(eddyb) in order to guarantee `summarize_ptr` can't call this,
    // it requires `FuncAtMut` instead of just `FuncAt`.
    fn scan_ptr(&mut self, func_at_ptr: FuncAtMut<'_, Value>) {
        // NOTE(eddyb) a somewhat unusual invariant is that top-level calls to
        // `summarize_ptr` cannot result in any `loop_body_input_deps`, because
        // any loop body inputs encountered will either be already summarized,
        // or never encountered before (and fully summarize themselves), with
        // `loop_body_input_deps` only showing up in the results of calls to
        // `summarize_ptr` *from* the loop body input summarization loop.
        let summary = self.summarize_ptr(func_at_ptr.freeze());
        let incomplete =
            summary.bases.as_ref().right().is_some_and(|u| !u.loop_body_input_deps.is_empty());
        assert!(!incomplete);

        // FIXME(eddyb) consider attaching diagnostics here (via `FuncAtMut`)?
    }

    #[must_use]
    fn summarize_ptr(&mut self, func_at_ptr: FuncAt<'_, Value>) -> PtrSummary {
        let ptr = func_at_ptr.position;

        let propagate = |mut s: PtrSummary| {
            if let Some(unknowns) = s.bases.as_mut().right() {
                if let Some(unsupported) = &mut unknowns.unsupported {
                    // Keep the value for direct users of `ptr`, but clear it for
                    // users of direct users of `ptr` (while keeping it in the cache).
                    if unsupported.directly_used[..] != [ptr] {
                        unsupported.directly_used = [].into_iter().collect();
                    }
                }
            }
            s
        };

        let cached = self.summarized_ptrs.get(&ptr).filter(|cached| {
            let retry = cached.bases.as_ref().right().is_some_and(|unknowns| {
                // Retry if the relevant loop body input has since been processed.
                match unknowns.loop_body_input_deps[..] {
                    [] => false,
                    [(region, input_idx)] => {
                        let input = Value::RegionInput { region, input_idx };
                        input != ptr
                            && self.summarized_ptrs.get(&input).is_some_and(|input_summary| {
                                let incomplete = (input_summary.bases.as_ref().right())
                                    .is_some_and(|u| !u.loop_body_input_deps.is_empty());
                                !incomplete
                            })
                    }
                    // HACK(eddyb) more than one `loop_body_input_deps` entry
                    // is never cached, to make the above check simpler.
                    _ => unreachable!(),
                }
            });
            !retry
        });
        if let Some(cached) = cached {
            return propagate(cached.clone());
        }

        let s = self.summarize_ptr_uncached(func_at_ptr);
        // HACK(eddyb) make it easier to check for retries.
        let avoid_caching =
            (s.bases.as_ref().right()).is_some_and(|u| u.loop_body_input_deps.len() > 1);
        if !avoid_caching {
            self.summarized_ptrs.insert(ptr, s.clone());
        }
        propagate(s)
    }

    fn summarize_ptr_uncached(&mut self, func_at_ptr: FuncAt<'_, Value>) -> PtrSummary {
        if let Some(base) = self.legalizer.ptr_as_base(func_at_ptr) {
            return PtrSummary {
                bases: EitherOrBoth::Left(BaseChoice::Fixed(base)),
                offset_shape: Offset::Zero,
            };
        }

        let ptr = func_at_ptr.position;
        let func = func_at_ptr.at(());

        // FIXME(eddyb) should these be methods on `PtrSummary`?
        let unsupported = || PtrSummary {
            bases: EitherOrBoth::Right(UnknownBases {
                any_escaped: None,
                loop_body_input_deps: [].into_iter().collect(),
                unsupported: Some(UnsupportedBases { directly_used: [ptr].into_iter().collect() }),
            }),
            offset_shape: Offset::Zero,
        };
        let apply_offset =
            |PtrSummary { bases, offset_shape }, new_offset: Offset<()>| PtrSummary {
                bases,
                offset_shape: offset_shape.merge(new_offset, |()| (), |()| (), |_, _| ()),
            };

        let (node, output_idx) = match ptr {
            // FIXME(eddyb) implement more constant address pointers, either
            // here or in `ptr_as_base`.
            Value::Const(_) => return unsupported(),

            Value::NodeOutput { node, output_idx } => (node, output_idx),

            // Loop body inputs are the only values with inherently cyclic sources
            // (i.e. they can, and often do, depend on the previous loop iteration),
            // and as such they require a more free-form "saturating" algorithm.
            Value::RegionInput { region, input_idx } => {
                let Some(&node) = self.parent_map.region_parent.get(region) else {
                    // Not a loop body input, so most likely a function parameter.
                    // FIXME(eddyb) support passing pointers through function calls.
                    return unsupported();
                };
                let node_def = func.at(node).def();
                // FIXME(eddyb) should this generate `Diag::bug` instead?
                assert!(matches!(node_def.kind, NodeKind::Loop { .. }));

                let mut s = PtrSummary {
                    bases: EitherOrBoth::Right(UnknownBases {
                        any_escaped: None,
                        loop_body_input_deps: [(region, input_idx)].into_iter().collect(),
                        unsupported: None,
                    }),
                    offset_shape: Offset::Zero,
                };

                // Avoid cyclic recursion by pre-filling the cache entry.
                self.summarized_ptrs.insert(ptr, s.clone());

                // HACK(eddyb) using `loop_body_input_deps` as a weird queue.
                let mut seen = FxHashSet::default();
                while let Some(candidate) =
                    s.bases.as_mut().right().and_then(|u| u.loop_body_input_deps.pop())
                {
                    if !seen.insert(candidate) {
                        continue;
                    }
                    // NOTE(eddyb) less care needed here because we ensure all
                    // entries in `loop_body_input_deps` come from loops.
                    let candidate_source_ptrs = {
                        let (region, input_idx) = candidate;
                        [
                            &func.at(self.parent_map.region_parent[region]).def().inputs,
                            &func.at(region).def().outputs,
                        ]
                        .map(|inputs_or_outputs| inputs_or_outputs[input_idx as usize])
                    };
                    for candidate_source_ptr in candidate_source_ptrs {
                        let candidate_summary = self.summarize_ptr(func.at(candidate_source_ptr));
                        s = self.merge_summaries(s, candidate_summary);
                    }
                }

                // HACK(eddyb) this should be the only place where `UnknownBases`
                // can end up with `(None, [], None)` fields.
                let any_unknowns = s.bases.as_ref().right().is_some_and(
                    |UnknownBases { any_escaped, loop_body_input_deps, unsupported }| {
                        assert!(loop_body_input_deps.is_empty());
                        any_escaped.is_some() || unsupported.is_some()
                    },
                );
                if !any_unknowns && s.bases.has_left() {
                    s.bases = EitherOrBoth::Left(s.bases.left().unwrap());
                }

                return s;
            }
        };

        let node_def = func.at(node).def();
        match &node_def.kind {
            NodeKind::Select(_) => {
                let mut per_case_output = node_def
                    .child_regions
                    .iter()
                    .map(|&case| func.at(case).def().outputs[output_idx as usize]);

                // HACK(eddyb) can't use `map` + `reduce` due to `self` borrowing.
                let mut s = {
                    let Some(first) = per_case_output.next() else {
                        return unsupported();
                    };
                    self.summarize_ptr(func.at(first))
                };
                for output in per_case_output {
                    let output_summary = self.summarize_ptr(func.at(output));
                    s = self.merge_summaries(s, output_summary);
                }
                s
            }

            // FIXME(eddyb) should these generate `Diag::bug` instead?
            NodeKind::Loop { .. } | NodeKind::Scalar(_) | NodeKind::Vector(_) => unreachable!(),

            NodeKind::FuncCall { .. } => {
                // FIXME(eddyb) support passing pointers through function calls.
                unsupported()
            }
            NodeKind::QPtr(op) => {
                assert_eq!(output_idx, 0);
                match *op {
                    QPtrOp::FuncLocalVar(_) | QPtrOp::BufferData => {
                        // Supported cases handled by `ptr_as_base`.
                        unsupported()
                    }
                    // FIXME(eddyb) implement, either here or in `ptr_as_base`.
                    QPtrOp::HandleArrayIndex => unsupported(),
                    // FIXME(eddyb) should the (Dyn)Offset handling be centralized?
                    QPtrOp::Offset(offset) => apply_offset(
                        self.summarize_ptr(func.at(node_def.inputs[0])),
                        NonZeroU32::new(offset.unsigned_abs())
                            .map_or(Offset::Zero, |stride| Offset::Dyn { stride, index: () }),
                    ),
                    // FIXME(eddyb) ignoring `index_bounds` (and later setting it to
                    // `None`) is lossy and can frustrate e.g. type recovery.
                    QPtrOp::DynOffset { stride, index_bounds: _ } => {
                        apply_offset(self.summarize_ptr(func.at(node_def.inputs[0])), Offset::Dyn {
                            stride,
                            index: (),
                        })
                    }
                    QPtrOp::Load { .. } => PtrSummary {
                        bases: EitherOrBoth::Right(UnknownBases {
                            any_escaped: Some(AnyEscapedBase),
                            loop_body_input_deps: [].into_iter().collect(),
                            unsupported: None,
                        }),
                        offset_shape: Offset::Zero,
                    },

                    // FIXME(eddyb) should these generate `Diag::bug` instead?
                    QPtrOp::BufferDynLen { .. } | QPtrOp::Store { .. } | QPtrOp::Copy { .. } => {
                        unreachable!()
                    }
                }
            }

            NodeKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(_))
            | NodeKind::SpvInst(..)
            | NodeKind::SpvExtInst { .. } => unsupported(),
        }
    }

    fn merge_summaries(&mut self, a: PtrSummary, b: PtrSummary) -> PtrSummary {
        let (a_bases, a_unknowns) = a.bases.map_any(Some, Some).or_default();
        let (b_bases, b_unknowns) = b.bases.map_any(Some, Some).or_default();

        // FIXME(eddyb) this should really be part of `EitherOrBoth`.
        fn maybe_either_or_both<L, R>(
            left: Option<L>,
            right: Option<R>,
        ) -> Option<EitherOrBoth<L, R>> {
            match (left, right) {
                (Some(l), None) => Some(EitherOrBoth::Left(l)),
                (None, Some(r)) => Some(EitherOrBoth::Right(r)),
                (Some(l), Some(r)) => Some(EitherOrBoth::Both(l, r)),
                (None, None) => None,
            }
        }

        let bases = maybe_either_or_both(a_bases, b_bases).map(|ab| {
            ab.reduce(|a, b| {
                a.merge(b, |base| self.func_base_map.dyn_choice_bases.insert_full(base).0)
            })
        });

        let unknowns = maybe_either_or_both(a_unknowns, b_unknowns).map(|ab| {
            // HACK(eddyb) this may be useful elsewhere too?
            fn merge_smallvecs<A: smallvec::Array>(
                mut a: SmallVec<A>,
                mut b: SmallVec<A>,
            ) -> SmallVec<A> {
                if a.len() >= b.len() {
                    a.append(&mut b);
                    a
                } else {
                    b.insert_many(0, a);
                    b
                }
            }
            fn dedup_smallvec<T: Copy + Eq + Hash, const N: usize>(
                mut xs: SmallVec<[T; N]>,
            ) -> SmallVec<[T; N]>
            where
                [T; N]: smallvec::Array<Item = T>,
            {
                // HACK(eddyb) `dedup` can help but only for adjacent elements.
                xs.dedup();
                if xs.len() > 2 {
                    let mut seen = FxHashSet::default();
                    xs.retain(|x| seen.insert(*x));
                }
                if xs.len() <= N {
                    xs.shrink_to_fit();
                }
                xs
            }

            ab.reduce(|a, b| UnknownBases {
                any_escaped: a.any_escaped.or(b.any_escaped),
                loop_body_input_deps: dedup_smallvec(merge_smallvecs(
                    a.loop_body_input_deps,
                    b.loop_body_input_deps,
                )),
                unsupported: maybe_either_or_both(a.unsupported, b.unsupported).map(|ab| {
                    ab.reduce(|a, b| UnsupportedBases {
                        directly_used: merge_smallvecs(a.directly_used, b.directly_used),
                    })
                }),
            })
        });

        PtrSummary {
            bases: maybe_either_or_both(bases, unknowns).unwrap(),
            // FIXME(eddyb) make this less noisy.
            offset_shape: a.offset_shape.merge(b.offset_shape, |()| (), |()| (), |_, _| ()),
        }
    }
}

struct CanonicalizePtrsInFunc<'a> {
    legalizer: &'a LegalizePtrs,

    parent_map: &'a mut ParentMap,

    canonicalized_ptrs: FxHashMap<Value, Result<CanonPtr, CanonError>>,

    // FIXME(eddyb) make this more specific to what it's used for?
    // (see also comment in `transform_value_use`)
    value_replacements: FxHashMap<Value, Value>,
}

/// Canonicalized pointer value, combining any number of offsetting steps, on
/// top of a `base` that (when `BaseChoice::Dyn`) respects `FuncLocalBaseMap`.
///
/// While eagerly combining offsets may be suboptimal for highly local dataflow
/// (e.g. `p2 = if c { p.add(1) } else { p };` can be `p2 = p.add(c as usize);`,
/// but `CanonPtr` will use `p2 = p_base.add(p_idx + (c as usize))` instead),
/// it has the benefit of making merging two `CanonPtr`s into an O(1) operation,
/// without having to go back and dig out some kind of "closest common base".
//
// FIXME(eddyb) reduce duplication between `PtrSummary` and this.
// FIXME(eddyb) canonicalize `HandleArrayIndex`.
#[derive(Clone)]
struct CanonPtr {
    base: BaseChoice<Value>,

    offset: Offset<Value>,

    // HACK(eddyb) if `base` is `BaseChoice::Fixed(GlobalBase::BufferData)`,
    // this is an existing `QPtrOp::BufferData` node for that buffer, that can
    // be used directly.
    reuse_buffer_data: Option<Node>,

    // HACK(eddyb) if `base` is `BaseChoice::Fixed(_)`, this is an existing
    // `QPtrOp::DynIndex` node for its final value, that can be used directly.
    reuse_final_ptr: Option<Node>,
}

/// Error marker type during canonicalization, indicating a diagnostic has been
/// attached already (either earlier, i.e. `UnsupportedBases` during scanning,
/// or `IndexError`s during canonicalization)
///
/// Failure *must* be eagerly propagated (i.e. no merging successes and errors
/// into `CanonPtr` `bases` and/or `offset` should happen), to avoid accidentally
/// discarding the attached diagnostics. (which risks replacing errors with UB).
// HACK(eddyb) the name doesn't contain `Error` for the sake of brevity.
#[derive(Copy, Clone)]
struct CanonError;

/// Errors specific to the types/consts/computation of `Offset::Dyn` indices,
/// that had been ignored during scanning, only to be hit by canonicalization.
#[derive(Clone)]
struct IndexError(Diag);

impl CanonicalizePtrsInFunc<'_> {
    /// Compute the `CanonPtr` representation of a pointer, reusing as much as
    /// possible (e.g. a `QPtrOp::DynIndex` node that still produces the same
    /// pointer value, at least as long as the `base` is `BaseChoice::Fixed`).
    ///
    /// **Note**: this on-demand approach (instead of only local transformations
    /// during traversal) is required due to uses of already-legal pointers
    /// (which would otherwise be left untouched), from e.g. pointer-typed
    /// `Select` outputs that themelves require legalization.
    //
    // FIXME(eddyb) reduce the mountains of duplication... the only part
    // that can't theoretically be done during the scanning step is taking
    // into account the complete set of escaped bases (hence a split design).
    fn canonicalize_ptr(
        &mut self,
        func_at_ptr: FuncAtMut<'_, Value>,
    ) -> Result<CanonPtr, CanonError> {
        let ptr = func_at_ptr.position;
        if let Some(cached) = self.canonicalized_ptrs.get(&ptr) {
            return cached.clone();
        }

        let canon = self.canonicalize_ptr_uncached(func_at_ptr);
        self.canonicalized_ptrs.insert(ptr, canon.clone());
        canon
    }

    fn canonicalize_ptr_uncached(
        &mut self,
        func_at_ptr: FuncAtMut<'_, Value>,
    ) -> Result<CanonPtr, CanonError> {
        let cx = &self.legalizer.cx;
        let wk = self.legalizer.wk;

        let ptr = func_at_ptr.position;
        let mut func = func_at_ptr.at(());

        if let Some(base) = self.legalizer.ptr_as_base(func.reborrow().freeze().at(ptr)) {
            let reuse_final_ptr = match ptr {
                // FIXME(eddyb) should `reuse_final_ptr` include constants, too?
                Value::Const(_) => None,
                Value::NodeOutput { node, output_idx: 0 } => Some(node),
                _ => unreachable!(),
            };
            return Ok(CanonPtr {
                base: BaseChoice::Fixed(base),
                offset: Offset::Zero,
                reuse_buffer_data: reuse_final_ptr
                    .filter(|_| matches!(base, Base::Global(GlobalBase::BufferData { .. }))),
                reuse_final_ptr,
            });
        }

        let (node, output_idx) = match ptr {
            // FIXME(eddyb) implement more constant address pointers, either
            // here or in `ptr_as_base`.
            Value::Const(_) => return unsupported(),

            Value::NodeOutput { node, output_idx } => (node, output_idx),

            // Loop body inputs are the only values with inherently cyclic sources
            // (i.e. they can, and often do, depend on the previous loop iteration),
            // and as such they require a more free-form "saturating" algorithm.
            Value::RegionInput { region, input_idx } => {
                unreachable!("TODO visit order error (check for qptr type as well?)")
            }
        };

        let Value::NodeOutput { node, output_idx: 0 } = ptr else {
            return Err(CanonError);
        };

        // HACK(eddyb) this defers interning a constant too early.
        #[derive(Copy, Clone)]
        enum Index {
            Dyn(Value),
            One,
            MinusOne,
        }

        let node_def = func.reborrow().freeze().at(node).def();
        match &node_def.kind {
            NodeKind::Select(_) => {
                unreachable!("TODO visit order error (check for qptr type as well?)")
            }

            // FIXME(eddyb) should these generate `Diag::bug` instead?
            NodeKind::Loop { .. } | NodeKind::Scalar(_) | NodeKind::Vector(_) => unreachable!(),

            NodeKind::FuncCall { .. } => {
                // FIXME(eddyb) support passing pointers through function calls.
                unsupported()
            }
            NodeKind::QPtr(op) => {
                assert_eq!(output_idx, 0);
                match *op {
                    QPtrOp::FuncLocalVar(_) | QPtrOp::BufferData => {
                        // Supported cases handled by `ptr_as_base`.
                        unsupported()
                    }
                    // FIXME(eddyb) implement, either here or in `ptr_as_base`.
                    QPtrOp::HandleArrayIndex => unsupported(),
                    // FIXME(eddyb) should the (Dyn)Offset handling be centralized?
                    QPtrOp::Offset(offset) => apply_offset(
                        self.summarize_ptr(func.at(node_def.inputs[0])),
                        NonZeroU32::new(offset.unsigned_abs())
                            .map_or(Offset::Zero, |stride| Offset::Dyn { stride, index: () }),
                    ),
                    // FIXME(eddyb) ignoring `index_bounds` (and later setting it to
                    // `None`) is lossy and can frustrate e.g. type recovery.
                    QPtrOp::DynOffset { stride, index_bounds: _ } => {
                        apply_offset(self.summarize_ptr(func.at(node_def.inputs[0])), Offset::Dyn {
                            stride,
                            index: (),
                        })
                    }
                    QPtrOp::Load { .. } => PtrSummary {
                        bases: EitherOrBoth::Right(UnknownBases {
                            any_escaped: Some(AnyEscapedBase),
                            loop_body_input_deps: [].into_iter().collect(),
                            unsupported: None,
                        }),
                        offset_shape: Offset::Zero,
                    },

                    // FIXME(eddyb) should these generate `Diag::bug` instead?
                    QPtrOp::BufferDynLen { .. } | QPtrOp::Store { .. } | QPtrOp::Copy { .. } => {
                        unreachable!()
                    }
                }
            }

            NodeKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(_))
            | NodeKind::SpvInst(..)
            | NodeKind::SpvExtInst { .. } => unsupported(),
        }

        let offset = match node_def.kind {
            NodeKind::QPtr(op) => match op {
                QPtrOp::FuncLocalVar(_) | QPtrOp::BufferData => {
                    // Supported cases handled by `ptr_as_base`.
                    return Err(CanonError);
                }
                // FIXME(eddyb) implement, either here or in `ptr_as_base`.
                QPtrOp::HandleArrayIndex => return Err(CanonError),

                QPtrOp::Offset(offset) => {
                    NonZeroU32::new(offset.unsigned_abs()).map_or(Offset::Zero, |stride| {
                        Offset::Dyn {
                            stride,
                            index: if offset < 0 { Index::MinusOne } else { Index::One },
                        }
                    })
                }
                QPtrOp::DynOffset { stride, .. } => {
                    Offset::Dyn { stride, index: Index::Dyn(node_def.inputs[1]) }
                }
            },
            _ => return Err(CanonError),
        };

        // HACK(eddyb) to avoid terminology issues (since "base" is very taken),
        // `obi_` (short for "offset base input") is used here to refer to the
        // (pointer) input of the offsetting operation and any aspects of it.
        let original_obi_ptr = node_def.inputs[0];

        let CanonPtr {
            base,
            offset: obi_offset,
            mut reuse_buffer_data,
            reuse_final_ptr: reuse_obi_ptr,
        } = self.canonicalize_ptr(func.reborrow().at(original_obi_ptr))?;

        let offset_or_index_error = obi_offset.merge(
            offset,
            Ok,
            |index| {
                match index {
                    Index::Dyn(index) => Ok(index),

                    // FIXME(eddyb) this can lead to index conflicts down the line.
                    Index::One => Ok(Value::Const(cx.intern(scalar::Const::from_u32(0)))),

                    // Directly applying a negative offset onto a base is
                    // *never* legal, as it would *always* leave the bounds.
                    Index::MinusOne => {
                        return Err(IndexError(Diag::bug(["illegal negative offset".into()])));
                    }
                }
            },
            |obi_index, index| {
                let index_ty = {
                    let func = func.reborrow().freeze();
                    let type_of = |v| func.at(v).type_of(cx);
                    let obi_index_ty = type_of(obi_index.value);
                    match index.value {
                        Index::Dyn(index) => {
                            let index_ty = type_of(index);
                            if obi_index_ty != index_ty {
                                // FIXME(eddyb) pick the larger type and then
                                // zero-extend the smaller one's value.
                                return Err(IndexError(Diag::bug([
                                    "index type mismatch: `".into(),
                                    obi_index_ty.into(),
                                    "` vs `".into(),
                                    index_ty.into(),
                                    "`".into(),
                                ])));
                            }
                        }
                        Index::One | Index::MinusOne => {}
                    }
                    obi_index_ty
                };
                let const_index_one = index_ty
                    .as_scalar(cx)
                    .and_then(|ty| scalar::Const::int_try_from_i128(ty, 1))
                    .ok_or_else(|| {
                        IndexError(Diag::bug([
                            "index type `".into(),
                            index_ty.into(),
                            "` not an integer".into(),
                        ]))
                    })?;

                let add_or_sub = match index.value {
                    Index::Dyn(_) | Index::One => scalar::IntBinOp::Add,
                    Index::MinusOne => scalar::IntBinOp::Sub,
                };
                let index = index.map_value(|index| match index {
                    Index::Dyn(index) => index,
                    Index::One | Index::MinusOne => Value::Const(cx.intern(const_index_one)),
                });

                let obi_index = self.materialize_scaled_index(func.reborrow().at(obi_index))?;
                let index = self.materialize_scaled_index(func.reborrow().at(index))?;

                // FIXME(eddyb) constant folding could be helpful here.
                let new_node = func.nodes.define(
                    cx,
                    NodeDef {
                        // FIXME(eddyb) copy at least debuginfo attrs from `node`.
                        attrs: Default::default(),
                        kind: scalar::Op::IntBinary(add_or_sub).into(),
                        inputs: [obi_index, index].into_iter().collect(),
                        child_regions: [].into_iter().collect(),
                        outputs: [NodeOutputDecl { attrs: Default::default(), ty: index_ty }]
                            .into_iter()
                            .collect(),
                    }
                    .into(),
                );
                // FIXME(eddyb) automate all of this with some kind of "insertion cursor".
                let parent_region = self.parent_map.node_parent[node];
                func.regions[parent_region].children.insert_before(new_node, node, func.nodes);
                self.parent_map.node_parent.insert(new_node, parent_region);
                Ok(Value::NodeOutput { node: new_node, output_idx: 0 })
            },
        );
        let offset = offset_or_index_error.transpose_value().map_err(|IndexError(err)| {
            func.reborrow().at(node).def().attrs.push_diag(cx, err);
            CanonError
        })?;

        let reuse_final_ptr = match (&base, offset) {
            (_, Offset::Zero) => reuse_obi_ptr.or(Some(node)),
            (&BaseChoice::Fixed(base), Offset::Dyn { stride, index }) => {
                let base_ptr = match base {
                    Base::FuncLocalVar(node) => Value::NodeOutput { node, output_idx: 0 },
                    Base::Global(global_base) => match global_base {
                        GlobalBase::Null => Value::Const(self.legalizer.null_qptr_const()),
                        GlobalBase::GlobalVar { ptr_to_var } => Value::Const(ptr_to_var),
                        GlobalBase::BufferData { ptr_to_buffer } => {
                            let buffer_data_node = *reuse_buffer_data.get_or_insert_with(|| {
                                let new_node = func.nodes.define(
                                    cx,
                                    NodeDef {
                                        // FIXME(eddyb) copy at least debuginfo attrs from `node`.
                                        attrs: Default::default(),
                                        kind: QPtrOp::BufferData.into(),
                                        inputs: [Value::Const(ptr_to_buffer)].into_iter().collect(),
                                        child_regions: [].into_iter().collect(),
                                        outputs: [NodeOutputDecl {
                                            attrs: Default::default(),
                                            ty: self.legalizer.qptr_type(),
                                        }]
                                        .into_iter()
                                        .collect(),
                                    }
                                    .into(),
                                );
                                // FIXME(eddyb) automate all of this with some kind of "insertion cursor".
                                let parent_region = self.parent_map.node_parent[node];
                                func.regions[parent_region]
                                    .children
                                    .insert_before(new_node, node, func.nodes);
                                self.parent_map.node_parent.insert(new_node, parent_region);
                                new_node
                            });
                            Value::NodeOutput { node: buffer_data_node, output_idx: 0 }
                        }
                    },
                };

                let node_def = func.at(node).def();
                (node_def.kind, node_def.inputs) = (
                    QPtrOp::DynOffset {
                        stride,
                        // FIXME(eddyb) this is lossy and can frustrate type recovery.
                        index_bounds: None,
                    }
                    .into(),
                    [base_ptr, index].into_iter().collect(),
                );

                Some(node)
            }
            (BaseChoice::Dyn { .. }, _) => None,
        };

        Ok(CanonPtr { base, offset, reuse_buffer_data, reuse_final_ptr })
    }

    // FIXME(eddyb) consider memoizing this (esp. for reusing int mul nodes).
    fn materialize_scaled_index(
        &mut self,
        func_at_scaled_index: FuncAtMut<'_, Scaled<Value>>,
    ) -> Result<Value, IndexError> {
        let cx = &self.legalizer.cx;

        let Scaled { value: index, multiplier } = func_at_scaled_index.position;
        let mut func = func_at_scaled_index.at(());

        let (parent_region, insert_after) = match index {
            _ if multiplier.get() == 1 => return Ok(index),

            // FIXME(eddyb) recategorize some `Diag`s as user errors, not just bugs.
            Value::Const(ct) => {
                let (ty, index) = ct
                    .as_scalar(cx)
                    .and_then(|ct| Some((ct.ty(), ct.int_as_u128()?)))
                    .ok_or_else(|| {
                        IndexError(Diag::bug([
                            "index `".into(),
                            ct.into(),
                            "` not a positive integer".into(),
                        ]))
                    })?;

                let r = index
                    .checked_mul(multiplier.get().into())
                    .and_then(|r| scalar::Const::int_try_from_i128(ty, r.try_into().ok()?))
                    .ok_or_else(|| {
                        IndexError(Diag::bug([
                            "index multiplication `".into(),
                            ct.into(),
                            format!(" * {multiplier}` overflowed").into(),
                        ]))
                    })?;
                return Ok(Value::Const(cx.intern(r)));
            }

            Value::RegionInput { region, input_idx: _ } => (region, None),
            Value::NodeOutput { node, output_idx: _ } => {
                (self.parent_map.node_parent[node], Some(node))
            }
        };

        let index_ty = func.reborrow().freeze().at(index).type_of(cx);

        let multiplier = Value::Const(
            cx.intern(
                index_ty
                    .as_scalar(cx)
                    .and_then(|ty| scalar::Const::int_try_from_i128(ty, multiplier.get().into()))
                    .ok_or_else(|| {
                        IndexError(Diag::bug([
                            format!("index type `").into(),
                            index_ty.into(),
                            format!("` can't represent `{multiplier}`").into(),
                        ]))
                    })?,
            ),
        );

        let new_node = func.nodes.define(
            cx,
            NodeDef {
                // FIXME(eddyb) copy at least debuginfo attrs.
                attrs: Default::default(),
                kind: scalar::Op::IntBinary(scalar::IntBinOp::Mul).into(),
                inputs: [index, multiplier].into_iter().collect(),
                child_regions: [].into_iter().collect(),
                outputs: [NodeOutputDecl { attrs: Default::default(), ty: index_ty }]
                    .into_iter()
                    .collect(),
            }
            .into(),
        );
        // FIXME(eddyb) automate all of this with some kind of "insertion cursor".
        let parent_region_children = &mut func.regions[parent_region].children;
        if let Some(node) = insert_after {
            parent_region_children.insert_after(new_node, node, func.nodes);
        } else {
            parent_region_children.insert_first(new_node, func.nodes);
        }
        self.parent_map.node_parent.insert(new_node, parent_region);
        Ok(Value::NodeOutput { node: new_node, output_idx: 0 })
    }
}

// TODO(eddyb) canonicalize `Offset`/`DynOffset` instructions that chain off of
// eachother in ways that couldn't possibly be legal (may require range info?),
// and maybe use it in recursion emulation? (but recursion emulation also wants
// legalized pointers... should recursion re-run legalization *afterwards*?)
impl Transformer for CanonicalizePtrsInFunc<'_> {
    fn transform_value_use(&mut self, v: &Value) -> Transformed<Value> {
        if let Value::Const(_) = v {
            return Transformed::Unchanged;
        }

        // HACK(eddyb) doesn't need transitive replacement because it's always
        // `Select` output/loop body input -> `QPtrOp::DynOffset`
        self.value_replacements.get(v).copied().map_or(Transformed::Unchanged, Transformed::Changed)
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        let cx = &self.legalizer.cx;
        let wk = self.legalizer.wk;

        func_at_node.reborrow().inner_in_place_transform_with(self);

        let node = func_at_node.position;
        let node_def = func_at_node.reborrow().def();
        match &node_def.kind {
            NodeKind::Select(_) => {
                let num_cases = node_def.child_regions.len();
                let num_outputs = node_def.outputs.len();

                // FIXME(eddyb) this should be impossible nowadays.
                if num_cases < 2 {
                    return;
                }

                let mut insert_after_node = node;

                let mut func = func_at_node.at(());
                for output_idx in 0..num_outputs {
                    let output_ty = func.reborrow().at(node).def().outputs[output_idx].ty;
                    if !matches!(cx[output_ty].kind, TypeKind::QPtr) {
                        continue;
                    }

                    let output =
                        Value::NodeOutput { node, output_idx: output_idx.try_into().unwrap() };

                    // HACK(eddyb) can only happen if an outer `Loop` re-visits
                    // its body (to replace some of its body's inputs).
                    if self.value_replacements.contains_key(&output) {
                        continue;
                    }

                    let Ok((
                        common_base,
                        Offset::Dyn { stride: merged_stride, index: common_index_ty },
                    )) = (0..num_cases)
                        .map(|case_idx| {
                            let case = func.reborrow().at(node).def().child_regions[case_idx];
                            let per_case_output =
                                func.reborrow().at(case).def().outputs[output_idx];
                            let CanonPtr { base, offset } = CanonPtr::canonicalize(
                                cx,
                                self.parent_map,
                                func.reborrow().at(per_case_output),
                                Default::default(),
                            );
                            (
                                base,
                                offset.map_value(|index| {
                                    // FIXME(eddyb) should `CanonPtr` cache `type_of(index)`?
                                    func.reborrow().freeze().at(index).type_of(cx)
                                }),
                            )
                        })
                        .coalesce(|a @ (a_base, a_offset), b @ (b_base, b_offset)| {
                            Ok((
                                Some(a_base).filter(|&a| a == b_base).ok_or((a, b))?,
                                a_offset
                                    .merge(b_offset, Some, Some, |a, b| {
                                        Some(a.value).filter(|&a| a == b.value)
                                    })
                                    .transpose_value()
                                    .ok_or((a, b))?,
                            ))
                        })
                        .exactly_one()
                    else {
                        // FIXME(eddyb) record this output for later legalization.
                        continue;
                    };

                    // FIXME(eddyb) final sanity check (necessary for `0` indices).
                    let Some(common_index_scalar_ty) = common_index_ty.as_scalar(cx) else {
                        continue;
                    };

                    // HACK(eddyb) because there are 2+ cases sharing the same
                    // base, we can assume it's defined outside of any one case.

                    // FIXME(eddyb) this name is really bad.
                    let merged_index_output_idx = {
                        // FIXME(eddyb) in theory we could reuse the `qptr` output
                        // for the index, but that can complicate value replacement.
                        let select_outputs = &mut func.reborrow().at(node).def().outputs;
                        let new_output_idx = select_outputs.len();
                        select_outputs.push(NodeOutputDecl {
                            attrs: Default::default(),
                            ty: common_index_ty,
                        });
                        new_output_idx
                    };
                    for case_idx in 0..num_cases {
                        let case = func.reborrow().at(node).def().child_regions[case_idx];
                        let per_case_output = func.reborrow().at(case).def().outputs[output_idx];

                        let CanonPtr { base, offset } = CanonPtr::canonicalize(
                            cx,
                            self.parent_map,
                            func.reborrow().at(per_case_output),
                            Offset::Dyn { stride: merged_stride, index: () },
                        );
                        assert!(base == common_base);

                        let case_outputs = &mut func.reborrow().at(case).def().outputs;
                        assert_eq!(case_outputs.len(), merged_index_output_idx);
                        case_outputs.push(match offset {
                            Offset::Dyn { stride, index } => {
                                assert_eq!(stride, merged_stride);
                                index
                            }
                            Offset::Zero => Value::Const(
                                cx.intern(scalar::Const::from_bits(common_index_scalar_ty, 0)),
                            ),
                        });
                    }

                    let merged_canon_ptr = CanonPtr {
                        base: common_base,
                        offset: Offset::Dyn {
                            stride: merged_stride,
                            index: Value::NodeOutput {
                                node,
                                output_idx: merged_index_output_idx.try_into().unwrap(),
                            },
                        },
                    };
                    let merged_ptr = {
                        let (kind, inputs) = merged_canon_ptr.as_node_kind_and_inputs();
                        let new_node = func.nodes.define(
                            cx,
                            NodeDef {
                                attrs: Default::default(),
                                kind,
                                inputs,
                                child_regions: [].into_iter().collect(),
                                outputs: [NodeOutputDecl {
                                    attrs: Default::default(),
                                    ty: output_ty,
                                }]
                                .into_iter()
                                .collect(),
                            }
                            .into(),
                        );
                        let parent_region = self.parent_map.node_parent[node];
                        self.parent_map.node_parent.insert(new_node, parent_region);
                        func.regions[parent_region].children.insert_after(
                            new_node,
                            insert_after_node,
                            func.nodes,
                        );
                        insert_after_node = new_node;
                        Value::NodeOutput { node: new_node, output_idx: 0 }
                    };

                    self.value_replacements.insert(output, merged_ptr);
                }
            }

            NodeKind::Loop { .. } => {
                let body = node_def.child_regions[0];

                let mut func = func_at_node.at(());

                let body_def = func.reborrow().at(body).def();
                let num_body_inputs_outputs = body_def.inputs.len();
                assert_eq!(num_body_inputs_outputs, body_def.outputs.len());

                // HACK(eddyb) fix-point loop (of a sort), even if monotonic,
                // there's still a risk of cost amplification (w/ nested loops).
                loop {
                    let mut any_changes = false;

                    // FIXME(eddyb) come up with better terminology for this
                    // (but not "loop state" as that clashes with the S in VSDG).
                    for body_in_out_idx in 0..num_body_inputs_outputs {
                        let body_def = func.reborrow().at(body).def();
                        let body_input_ty = body_def.inputs[body_in_out_idx].ty;
                        if !matches!(cx[body_input_ty].kind, TypeKind::QPtr) {
                            continue;
                        }

                        let body_input = Value::RegionInput {
                            region: body,
                            input_idx: body_in_out_idx.try_into().unwrap(),
                        };

                        // HACK(eddyb) can only happen if an outer `Loop` re-visits
                        // its body (to replace some of its body's inputs).
                        if self.value_replacements.contains_key(&body_input) {
                            continue;
                        }

                        // Simple induction detection: `output = input + stride * index`.
                        let body_output = body_def.outputs[body_in_out_idx];

                        let (intra_iteration_stride, index_ty) = match CanonPtr::canonicalize(
                            cx,
                            self.parent_map,
                            func.reborrow().at(body_output),
                            Default::default(),
                        ) {
                            CanonPtr { base, offset: Offset::Dyn { stride, index } }
                                if base == body_input =>
                            {
                                // FIXME(eddyb) should `CanonPtr` cache `type_of(index)`?
                                (stride, func.reborrow().freeze().at(index).type_of(cx))
                            }
                            _ => continue,
                        };

                        let initial_input = func.reborrow().at(node).def().inputs[body_in_out_idx];
                        let mut canon_initial_input = CanonPtr::canonicalize(
                            cx,
                            self.parent_map,
                            func.reborrow().at(initial_input),
                            Offset::Dyn { stride: intra_iteration_stride, index: () },
                        );

                        if let Offset::Dyn { index: initial_index, .. } = canon_initial_input.offset
                        {
                            // FIXME(eddyb) should `CanonPtr` cache `type_of(index)`?
                            if func.reborrow().freeze().at(initial_index).type_of(cx) != index_ty {
                                continue;
                            }
                        } else {
                            // FIXME(eddyb) final sanity check (necessary for `0` indices).
                            let Some(index_scalar_ty) = index_ty.as_scalar(cx) else {
                                continue;
                            };

                            canon_initial_input.offset = Offset::Dyn {
                                stride: intra_iteration_stride,
                                index: Value::Const(
                                    cx.intern(scalar::Const::from_bits(index_scalar_ty, 0)),
                                ),
                            };
                        }

                        // FIXME(eddyb) this name is really bad.
                        let induction_index_in_out_idx = {
                            let body_inputs = &mut func.reborrow().at(body).def().inputs;
                            let new_in_out_idx = body_inputs.len();
                            body_inputs
                                .push(RegionInputDecl { attrs: Default::default(), ty: index_ty });
                            new_in_out_idx
                        };

                        let mut canon_inductive_closed_form = canon_initial_input;
                        let initial_induction_index = match &mut canon_inductive_closed_form.offset
                        {
                            Offset::Dyn { stride: _, index: inductive_closed_form_index } => {
                                mem::replace(inductive_closed_form_index, Value::RegionInput {
                                    region: body,
                                    input_idx: induction_index_in_out_idx.try_into().unwrap(),
                                })
                            }
                            _ => unreachable!(),
                        };

                        {
                            let initial_inputs = &mut func.reborrow().at(node).def().inputs;
                            assert_eq!(initial_inputs.len(), induction_index_in_out_idx);
                            initial_inputs.push(initial_induction_index);
                        }

                        let inductive_closed_form_ptr = {
                            let (kind, inputs) =
                                canon_inductive_closed_form.as_node_kind_and_inputs();
                            let new_node = func.nodes.define(
                                cx,
                                NodeDef {
                                    attrs: Default::default(),
                                    kind,
                                    inputs,
                                    child_regions: [].into_iter().collect(),
                                    outputs: [NodeOutputDecl {
                                        attrs: Default::default(),
                                        ty: body_input_ty,
                                    }]
                                    .into_iter()
                                    .collect(),
                                }
                                .into(),
                            );
                            self.parent_map.node_parent.insert(new_node, body);
                            // FIXME(eddyb) don't reverse the order like this.
                            func.regions[body].children.insert_first(new_node, func.nodes);
                            Value::NodeOutput { node: new_node, output_idx: 0 }
                        };

                        // HACK(eddyb) the only thing missing at this point is the
                        // body output for the induction index, which is coaxed out
                        // of `CanonPtr` by swapping out `body_output`'s input
                        // *in-place* (it must be a `QPtr::DynOffset` after all),
                        // then hoping applying `CanonPtr` again will result in the
                        // same base and stride wrt `canon_inductive_closed_form`,
                        // meaning the index is exactly the desired value.
                        // FIXME(eddyb) that's really "strided pointer subtraction",
                        // and could be more explicitly so, if that made sense.
                        // FIXME(eddyb) due do the fragility of the checks below,
                        // it's possible they could be replaced with a failure state
                        // where the additions above are kept as dead code instead.
                        let body_output_base_slot = match body_output {
                            Value::NodeOutput { node, output_idx: 0 } => {
                                match &mut *func.nodes[node] {
                                    NodeDef {
                                        kind: NodeKind::QPtr(QPtrOp::DynOffset { .. }),
                                        inputs,
                                        ..
                                    } => Some(&mut inputs[0]),
                                    _ => None,
                                }
                            }
                            _ => None,
                        }
                        .filter(|slot| **slot == body_input)
                        .unwrap();
                        *body_output_base_slot = inductive_closed_form_ptr;

                        let induction_index_output = match CanonPtr::canonicalize(
                            cx,
                            self.parent_map,
                            func.reborrow().at(body_output),
                            Default::default(),
                        ) {
                            CanonPtr { base, offset: Offset::Dyn { stride, index } } => {
                                assert!(base == canon_inductive_closed_form.base);
                                match canon_inductive_closed_form.offset {
                                    Offset::Dyn {
                                        stride: inductive_closed_form_stride, ..
                                    } => assert_eq!(stride, inductive_closed_form_stride),

                                    _ => unreachable!(),
                                }
                                index
                            }
                            _ => unreachable!(),
                        };

                        {
                            let body_outputs = &mut func.reborrow().at(body).def().outputs;
                            assert_eq!(body_outputs.len(), induction_index_in_out_idx);
                            body_outputs.push(induction_index_output);
                        }

                        self.value_replacements.insert(body_input, inductive_closed_form_ptr);
                        any_changes = true;
                    }

                    if !any_changes {
                        break;
                    }

                    // HACK(eddyb) this might get pretty expensive but should suffice
                    // to apply all the body region input value replacements etc.
                    func.reborrow().at(body).inner_in_place_transform_with(self);
                }
            }

            NodeKind::SpvInst(spv_inst, lowering)
                if [wk.OpPtrEqual, wk.OpPtrNotEqual].contains(&spv_inst.opcode)
                    && lowering.disaggregated_output.is_none()
                    && lowering.disaggregated_inputs.is_empty()
                    && node_def.inputs.len() == 2 =>
            {
                let cmp_op = match spv_inst.opcode {
                    o if o == wk.OpPtrEqual => scalar::IntBinOp::Eq,
                    o if o == wk.OpPtrNotEqual => scalar::IntBinOp::Ne,
                    _ => unreachable!(),
                };
                let inputs = [node_def.inputs[0], node_def.inputs[1]];

                let [mut a, mut b] = inputs.map(|ptr| {
                    CanonPtr::canonicalize(
                        cx,
                        self.parent_map,
                        func_at_node.reborrow().at(ptr),
                        Default::default(),
                    )
                });

                if a.base != b.base {
                    return;
                }

                // HACK(eddyb) re-canonicalize if necessary to get the same stride.
                let offset_shape_mismatch = a
                    .offset
                    .merge(
                        b.offset,
                        |_| None,
                        |_| None,
                        |a, b| (a.multiplier.get() > 1 || b.multiplier.get() > 1).then_some(()),
                    )
                    .transpose_value();
                if let Some(offset_shape) = offset_shape_mismatch {
                    for (ptr, canon) in inputs.into_iter().zip([&mut a, &mut b]) {
                        let new_canon = CanonPtr::canonicalize(
                            cx,
                            self.parent_map,
                            func_at_node.reborrow().at(ptr),
                            offset_shape,
                        );
                        if new_canon.base != canon.base {
                            return;
                        }
                        *canon = new_canon;
                    }
                }

                let offsets =
                    a.offset.merge(b.offset, EitherOrBoth::Left, EitherOrBoth::Right, |a, b| {
                        assert_eq!((a.multiplier.get(), b.multiplier.get()), (1, 1));
                        EitherOrBoth::Both(a.value, b.value)
                    });

                let mut mark_as_always_equal = || {
                    self.value_replacements.insert(
                        Value::NodeOutput { node, output_idx: 0 },
                        Value::Const(
                            cx.intern(scalar::Const::from_bool(cmp_op == scalar::IntBinOp::Eq)),
                        ),
                    );
                };

                match offsets {
                    Offset::Zero => mark_as_always_equal(),
                    Offset::Dyn { stride: _, index: EitherOrBoth::Both(a_index, b_index) } => {
                        let node_def = func_at_node.def();
                        node_def.kind = scalar::Op::IntBinary(cmp_op).into();
                        node_def.inputs = [a_index, b_index].into_iter().collect();

                        // HACK(eddyb) unlikely but might as well.
                        if a_index == b_index {
                            mark_as_always_equal();
                        }
                    }
                    // FIXME(eddyb) implement by generating a `0` constant.
                    Offset::Dyn {
                        stride: _,
                        index: EitherOrBoth::Left(_) | EitherOrBoth::Right(_),
                    } => {}
                }
            }

            _ => {}
        }
    }
}
