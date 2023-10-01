//! SPIR-T to SPIR-V lifting.

use crate::func_at::FuncAt;
use crate::spv::{self, spec};
use crate::visit::{InnerVisit, Visitor};
use crate::{
    AddrSpace, Attr, AttrSet, Const, ConstDef, ConstKind, Context, DataInst, DataInstDef,
    DataInstKind, DbgSrcLoc, DeclDef, EntityOrientedDenseMap, ExportKey, Exportee, Func, FuncDecl,
    FuncDefBody, FuncParam, FxIndexMap, FxIndexSet, GlobalVar, GlobalVarDefBody, Import, Module,
    ModuleDebugInfo, ModuleDialect, Node, NodeKind, NodeOutputDecl, OrdAssertEq, Region,
    RegionInputDecl, SelectionKind, Type, TypeDef, TypeKind, TypeOrConst, Value, cfg, scalar,
};
use itertools::Itertools as _;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::Path;
use std::{io, iter, mem};

// HACK(eddyb) getting around the lack of a `Step` impl on `spv::Id` (`NonZeroU32`).
trait IdRangeExt {
    fn iter(&self) -> iter::Map<Range<u32>, fn(u32) -> spv::Id>;
}
impl IdRangeExt for Range<spv::Id> {
    fn iter(&self) -> iter::Map<Range<u32>, fn(u32) -> spv::Id> {
        (self.start.get()..self.end.get()).map(|i| spv::Id::new(i).unwrap())
    }
}

impl spv::Dialect {
    fn capability_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.capabilities.iter().map(move |&cap| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpCapability,
                imms: iter::once(spv::Imm::Short(wk.Capability, cap)).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }

    pub fn extension_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.extensions.iter().map(move |ext| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpExtension,
                imms: spv::encode_literal_string(ext).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }
}

impl spv::ModuleDebugInfo {
    fn source_extension_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.source_extensions.iter().map(move |ext| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpSourceExtension,
                imms: spv::encode_literal_string(ext).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }

    fn module_processed_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.module_processes.iter().map(move |proc| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpModuleProcessed,
                imms: spv::encode_literal_string(proc).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }
}

/// ID allocation callback, kept as a closure (instead of having its state
/// be part of `Lifter`) to avoid misuse.
trait AllocIds: FnMut(usize) -> Range<spv::Id> {
    fn one(&mut self) -> spv::Id {
        self(1).start
    }
}

impl<F: FnMut(usize) -> Range<spv::Id>> AllocIds for F {}

struct Lifter<'a, AI: AllocIds> {
    cx: &'a Context,
    module: &'a Module,

    alloc_ids: AI,

    ids: ModuleIds<'a>,

    global_vars_seen: FxIndexSet<GlobalVar>,
}

#[derive(Default)]
struct ModuleIds<'a> {
    ext_inst_imports: BTreeMap<&'a str, spv::Id>,
    debug_strings: BTreeMap<&'a str, spv::Id>,

    // FIXME(eddyb) use `EntityOrientedDenseMap` here.
    globals: FxIndexMap<Global, spv::Id>,
    // FIXME(eddyb) use `EntityOrientedDenseMap` here.
    funcs: FxIndexMap<Func, FuncIds<'a>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Global {
    Type(Type),
    Const(Const),
}

// FIXME(eddyb) this is inconsistently named with `FuncBodyLifting`.
struct FuncIds<'a> {
    spv_func_ret_type: Type,
    // FIXME(eddyb) should we even be interning an `OpTypeFunction` in `Context`?
    // (it's easier this way, but it could also be tracked in `ModuleIds`)
    spv_func_type: Type,

    func_id: spv::Id,
    param_ids: Range<spv::Id>,

    body: Option<FuncBodyLifting<'a>>,
}

impl<AI: AllocIds> Visitor<'_> for Lifter<'_, AI> {
    fn visit_attr_set_use(&mut self, attrs: AttrSet) {
        self.visit_attr_set_def(&self.cx[attrs]);
    }
    fn visit_type_use(&mut self, ty: Type) {
        let global = Global::Type(ty);
        if self.ids.globals.contains_key(&global) {
            return;
        }
        let ty_def = &self.cx[ty];

        // HACK(eddyb) there isn't a great way to handle canonical types, but
        // perhaps this result should be recorded in `self.globals`?
        if let Some((_spv_inst, type_and_const_inputs)) =
            spv::Inst::from_canonical_type(self.cx, &ty_def.kind)
        {
            for ty_or_ct in type_and_const_inputs {
                match ty_or_ct {
                    TypeOrConst::Type(ty) => self.visit_type_use(ty),
                    TypeOrConst::Const(ct) => self.visit_const_use(ct),
                }
            }
        }

        match ty_def.kind {
            TypeKind::Scalar(_) | TypeKind::Vector(_) | TypeKind::SpvInst { .. } => {}

            // FIXME(eddyb) this should be a proper `Result`-based error instead,
            // and/or `spv::lift` should mutate the module for legalization.
            TypeKind::QPtr => {
                unreachable!("`TypeKind::QPtr` should be legalized away before lifting");
            }

            TypeKind::SpvStringLiteralForExtInst => {
                unreachable!(
                    "`TypeKind::SpvStringLiteralForExtInst` should not be used \
                     as a type outside of `ConstKind::SpvStringLiteralForExtInst`"
                );
            }
        }

        self.visit_type_def(ty_def);
        self.ids.globals.insert(global, self.alloc_ids.one());
    }
    fn visit_const_use(&mut self, ct: Const) {
        let global = Global::Const(ct);
        if self.ids.globals.contains_key(&global) {
            return;
        }
        let ct_def = &self.cx[ct];

        // HACK(eddyb) there isn't a great way to handle canonical consts, but
        // perhaps this result should be recorded in `self.globals`?
        if let Some((_spv_inst, const_inputs)) =
            spv::Inst::from_canonical_const(self.cx, &ct_def.kind)
        {
            for ct in const_inputs {
                self.visit_const_use(ct);
            }
        }

        match ct_def.kind {
            ConstKind::Undef
            | ConstKind::Scalar(_)
            | ConstKind::Vector(_)
            | ConstKind::PtrToGlobalVar(_)
            | ConstKind::SpvInst { .. } => {
                self.visit_const_def(ct_def);
                self.ids.globals.insert(global, self.alloc_ids.one());
            }

            // HACK(eddyb) because this is an `OpString` and needs to go earlier
            // in the module than any `OpConstant*`, it needs to be special-cased,
            // without visiting its type, or an entry in `self.globals`.
            ConstKind::SpvStringLiteralForExtInst(s) => {
                let ConstDef { attrs, ty, kind: _ } = ct_def;

                assert!(*attrs == AttrSet::default());
                assert!(
                    self.cx[*ty]
                        == TypeDef {
                            attrs: AttrSet::default(),
                            kind: TypeKind::SpvStringLiteralForExtInst,
                        }
                );

                self.ids.debug_strings.entry(&self.cx[s]).or_insert_with(|| self.alloc_ids.one());
            }
        }
    }

    fn visit_global_var_use(&mut self, gv: GlobalVar) {
        if self.global_vars_seen.insert(gv) {
            self.visit_global_var_decl(&self.module.global_vars[gv]);
        }
    }
    fn visit_func_use(&mut self, func: Func) {
        if self.ids.funcs.contains_key(&func) {
            return;
        }
        let func_decl = &self.module.funcs[func];

        // Synthesize an `OpTypeFunction` type (that SPIR-T itself doesn't carry).
        let wk = &spec::Spec::get().well_known;
        let spv_func_ret_type = match &func_decl.ret_types[..] {
            &[ty] => ty,
            // Reaggregate multiple return types into an `OpTypeStruct`.
            ret_types => {
                let opcode = if ret_types.is_empty() { wk.OpTypeVoid } else { wk.OpTypeStruct };
                self.cx.intern(spv::Inst::from(opcode).into_canonical_type_with(
                    self.cx,
                    ret_types.iter().copied().map(TypeOrConst::Type).collect(),
                ))
            }
        };
        let spv_func_type = self.cx.intern(
            spv::Inst::from(wk.OpTypeFunction).into_canonical_type_with(
                self.cx,
                iter::once(spv_func_ret_type)
                    .chain(func_decl.params.iter().map(|param| param.ty))
                    .map(TypeOrConst::Type)
                    .collect(),
            ),
        );
        self.visit_type_use(spv_func_type);

        // NOTE(eddyb) inserting first produces a different function ordering
        // overall in the final module, but the order doesn't matter, and we
        // need to avoid infinite recursion for recursive functions.
        self.ids.funcs.insert(func, FuncIds {
            spv_func_ret_type,
            spv_func_type,
            func_id: self.alloc_ids.one(),
            param_ids: (self.alloc_ids)(func_decl.params.len()),
            body: None,
        });

        self.visit_func_decl(func_decl);

        // Handle the body last, to minimize recursion hazards (see comment above),
        // and to allow `FuncBodyLifting` to look up its dependencies in `self.ids`.
        match &func_decl.def {
            DeclDef::Imported(_) => {}
            DeclDef::Present(func_def_body) => {
                let func_body_lifting = FuncBodyLifting::from_func_def_body(self, func_def_body);
                self.ids.funcs.get_mut(&func).unwrap().body = Some(func_body_lifting);
            }
        }
    }

    fn visit_spv_module_debug_info(&mut self, debug_info: &spv::ModuleDebugInfo) {
        for sources in debug_info.source_languages.values() {
            // The file operand of `OpSource` has to point to an `OpString`.
            for &s in sources.file_contents.keys() {
                self.ids.debug_strings.entry(&self.cx[s]).or_insert_with(|| self.alloc_ids.one());
            }
        }
    }
    fn visit_attr(&mut self, attr: &Attr) {
        match *attr {
            Attr::Diagnostics(_)
            | Attr::QPtr(_)
            | Attr::SpvAnnotation { .. }
            | Attr::SpvBitflagsOperand(_) => {}
            Attr::DbgSrcLoc(OrdAssertEq(DbgSrcLoc { file_path, .. })) => {
                self.ids
                    .debug_strings
                    .entry(&self.cx[file_path])
                    .or_insert_with(|| self.alloc_ids.one());
            }
        }
        attr.inner_visit_with(self);
    }

    fn visit_node_def(&mut self, func_at_node: FuncAt<'_, Node>) {
        match func_at_node.def().kind {
            NodeKind::FuncCall { .. }
            | NodeKind::Select(_)
            | NodeKind::Loop { .. }
            | NodeKind::ExitInvocation(_)
            | DataInstKind::Scalar(_)
            | DataInstKind::Vector(_)
            | DataInstKind::SpvInst(..) => {}

            // FIXME(eddyb) this should be a proper `Result`-based error instead,
            // and/or `spv::lift` should mutate the module for legalization.
            DataInstKind::QPtr(_) => {
                unreachable!("`DataInstKind::QPtr` should be legalized away before lifting");
            }

            DataInstKind::SpvExtInst { ext_set, .. } => {
                self.ids
                    .ext_inst_imports
                    .entry(&self.cx[ext_set])
                    .or_insert_with(|| self.alloc_ids.one());
            }
        }
        func_at_node.inner_visit_with(self);
    }
}

// FIXME(eddyb) this is inconsistently named with `FuncIds`.
struct FuncBodyLifting<'a> {
    region_inputs_source: EntityOrientedDenseMap<Region, RegionInputsSource>,
    data_insts: EntityOrientedDenseMap<DataInst, DataInstLifting>,

    label_ids: FxHashMap<CfgPoint, spv::Id>,
    blocks: FxIndexMap<CfgPoint, BlockLifting<'a>>,
}

/// What determines the values for [`Value::RegionInput`]s, for a specific
/// region (effectively the subset of "region parents" that support inputs).
///
/// Note that this is not used when a [`cfg::ControlInst`] has `target_inputs`,
/// and the target [`Region`] itself has phis for its `inputs`.
enum RegionInputsSource {
    FuncParams,
    LoopHeaderPhis(Node),
}

struct DataInstLifting {
    result_id: Option<spv::Id>,

    /// If the SPIR-V result type is "aggregate" (`OpTypeStruct`/`OpTypeArray`),
    /// this describes how to extract its leaves, which is necessary as on the
    /// SPIR-T side, [`Value`]s can only refer to individual leaves.
    disaggregate_result: Option<DisaggregateToLeaves>,

    /// `reaggregate_inputs[i]` describes how to recreate the "aggregate" value
    /// demanded by [`spv::InstLowering`]'s `disaggregated_inputs[i]`.
    reaggregate_inputs: SmallVec<[ReaggregateFromLeaves; 1]>,
}

/// All the information necessary to decompose a SPIR-V "aggregate" value into
/// its leaves, with one `OpCompositeExtract` per leaf.
//
// FIXME(eddyb) it might be more efficient to only extract actually used leaves,
// or chain partial extracts following nesting structure - but this is simpler.
struct DisaggregateToLeaves {
    op_composite_extract_result_ids: Range<spv::Id>,
}

/// All the information necessary to recreate a SPIR-V "aggregate" value, with
/// one `OpCompositeInsert` per leaf (starting with an `OpUndef` of that type).
//
// FIXME(eddyb) it might be more efficient to use other strategies, such as
// `OpCompositeConstruct`, special-casing constants, reusing whole results
// of other `DataInstDef`s with an aggregate result, etc. - but this is simpler
// for now, and it reuses the "one instruction per leaf" used for extractions.
struct ReaggregateFromLeaves {
    op_undef: Const,
    op_composite_insert_result_ids: Range<spv::Id>,
}

/// Any of the possible points in structured or unstructured SPIR-T control-flow,
/// that may require a separate SPIR-V basic block.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum CfgPoint {
    RegionEntry(Region),
    RegionExit(Region),

    NodeEntry(Node),
    NodeExit(Node),
}

struct BlockLifting<'a> {
    phis: SmallVec<[Phi; 2]>,
    insts: SmallVec<[DataInst; 4]>,
    terminator: Terminator<'a>,
}

struct Phi {
    attrs: AttrSet,
    ty: Type,

    result_id: spv::Id,
    cases: FxIndexMap<CfgPoint, Value>,

    // HACK(eddyb) used for `Loop` `initial_inputs`, to indicate that any edge
    // to the `Loop` (other than the backedge, which is already in `cases`)
    // should automatically get an entry into `cases`, with this value.
    default_value: Option<Value>,
}

/// Similar to [`cfg::ControlInst`], except:
/// * `targets` use [`CfgPoint`]s instead of [`Region`]s, to be able to
///   reach any of the SPIR-V blocks being created during lifting
/// * φ ("phi") values can be provided for targets regardless of "which side" of
///   the structured control-flow they are for ("region input" vs "node output")
/// * optional `merge` (for `OpSelectionMerge`/`OpLoopMerge`)
/// * existing data is borrowed (from the [`FuncDefBody`](crate::FuncDefBody)),
///   wherever possible
struct Terminator<'a> {
    attrs: AttrSet,

    kind: Cow<'a, cfg::ControlInstKind>,

    /// If this is a [`cfg::ControlInstKind::Return`] with `inputs.len() > 1`,
    /// this ID is for the `OpCompositeConstruct` needed to produce the single
    /// `OpTypeStruct` (`spv_func_ret_type`) value required by `OpReturnValue`.
    reaggregated_return_value_id: Option<spv::Id>,

    // FIXME(eddyb) use `Cow` or something, but ideally the "owned" case always
    // has at most one input, so allocating a whole `Vec` for that seems unwise.
    inputs: SmallVec<[Value; 2]>,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    targets: SmallVec<[CfgPoint; 4]>,

    target_phi_values: FxIndexMap<CfgPoint, &'a [Value]>,

    merge: Option<Merge<CfgPoint>>,
}

#[derive(Copy, Clone)]
enum Merge<L> {
    Selection(L),

    Loop {
        /// The label just after the whole loop, i.e. the `break` target.
        loop_merge: L,

        /// A label that the back-edge block post-dominates, i.e. some point in
        /// the loop body where looping around is inevitable (modulo `break`ing
        /// out of the loop through a `do`-`while`-style conditional back-edge).
        ///
        /// SPIR-V calls this "the `continue` target", but unlike other aspects
        /// of SPIR-V "structured control-flow", there can be multiple valid
        /// choices (any that fit the post-dominator/"inevitability" definition).
        //
        // FIXME(eddyb) https://github.com/EmbarkStudios/spirt/pull/10 tried to
        // set this to the loop body entry, but that may not be valid if the loop
        // body actually diverges, because then the loop body exit will still be
        // post-dominating the back-edge *but* the loop body itself wouldn't have
        // any relationship between its entry and its *unreachable* exit.
        loop_continue: L,
    },
}

/// Helper type for deep traversal of the CFG (as a graph of [`CfgPoint`]s), which
/// tracks the necessary context for navigating a [`Region`]/[`Node`].
#[derive(Copy, Clone)]
struct CfgCursor<'p, P = CfgPoint> {
    point: P,
    parent: Option<&'p CfgCursor<'p, ControlParent>>,
}

enum ControlParent {
    Region(Region),
    Node(Node),
}

impl<'p> FuncAt<'_, CfgCursor<'p>> {
    /// Return the next [`CfgPoint`] (wrapped in [`CfgCursor`]) in a linear
    /// chain within structured control-flow (i.e. no branching to child regions).
    fn unique_successor(self) -> Option<CfgCursor<'p>> {
        let cursor = self.position;
        match cursor.point {
            // Entering a `Region` enters its first `Node` child,
            // or exits the region right away (if it has no children).
            CfgPoint::RegionEntry(region) => Some(CfgCursor {
                point: match self.at(region).def().children.iter().first {
                    Some(first_child) => CfgPoint::NodeEntry(first_child),
                    None => CfgPoint::RegionExit(region),
                },
                parent: cursor.parent,
            }),

            // Exiting a `Region` exits its parent `Node`.
            CfgPoint::RegionExit(_) => cursor.parent.map(|parent| match parent.point {
                ControlParent::Region(_) => unreachable!(),
                ControlParent::Node(parent_node) => {
                    CfgCursor { point: CfgPoint::NodeExit(parent_node), parent: parent.parent }
                }
            }),

            // Entering a `Node` depends entirely on the `NodeKind`.
            CfgPoint::NodeEntry(node) => match self.at(node).def().kind {
                NodeKind::Select { .. }
                | NodeKind::Loop { .. }
                | NodeKind::ExitInvocation { .. } => None,

                NodeKind::FuncCall { .. }
                | DataInstKind::Scalar(_)
                | DataInstKind::Vector(_)
                | DataInstKind::QPtr(_)
                | DataInstKind::SpvInst(..)
                | DataInstKind::SpvExtInst { .. } => {
                    Some(CfgCursor { point: CfgPoint::NodeExit(node), parent: cursor.parent })
                }
            },

            // Exiting a `Node` chains to a sibling/parent.
            CfgPoint::NodeExit(node) => {
                Some(match self.nodes[node].next_in_list() {
                    // Enter the next sibling in the `Region`, if one exists.
                    Some(next_node) => {
                        CfgCursor { point: CfgPoint::NodeEntry(next_node), parent: cursor.parent }
                    }

                    // Exit the parent `Region`.
                    None => {
                        let parent = cursor.parent.unwrap();
                        match cursor.parent.unwrap().point {
                            ControlParent::Region(parent_region) => CfgCursor {
                                point: CfgPoint::RegionExit(parent_region),
                                parent: parent.parent,
                            },
                            ControlParent::Node(_) => unreachable!(),
                        }
                    }
                })
            }
        }
    }
}

impl FuncAt<'_, Region> {
    /// Traverse every [`CfgPoint`] (deeply) contained in this [`Region`],
    /// in reverse post-order (RPO), with `f` receiving each [`CfgPoint`]
    /// in turn (wrapped in [`CfgCursor`], for further traversal flexibility).
    ///
    /// RPO iteration over a CFG provides certain guarantees, most importantly
    /// that dominators are visited before the entire subgraph they dominate.
    fn rev_post_order_for_each(self, mut f: impl FnMut(CfgCursor<'_>)) {
        self.rev_post_order_for_each_inner(&mut f, None);
    }

    fn rev_post_order_for_each_inner(
        self,
        f: &mut impl FnMut(CfgCursor<'_>),
        parent: Option<&CfgCursor<'_, ControlParent>>,
    ) {
        let region = self.position;
        f(CfgCursor { point: CfgPoint::RegionEntry(region), parent });
        for func_at_node in self.at_children() {
            func_at_node.rev_post_order_for_each_inner(f, &CfgCursor {
                point: ControlParent::Region(region),
                parent,
            });
        }
        f(CfgCursor { point: CfgPoint::RegionExit(region), parent });
    }
}

impl FuncAt<'_, Node> {
    fn rev_post_order_for_each_inner(
        self,
        f: &mut impl FnMut(CfgCursor<'_>),
        parent: &CfgCursor<'_, ControlParent>,
    ) {
        let node = self.position;
        let parent = Some(parent);
        f(CfgCursor { point: CfgPoint::NodeEntry(node), parent });
        for &region in &self.def().child_regions {
            self.at(region).rev_post_order_for_each_inner(
                f,
                Some(&CfgCursor { point: ControlParent::Node(node), parent }),
            );
        }
        f(CfgCursor { point: CfgPoint::NodeExit(node), parent });
    }
}

impl<'a> FuncBodyLifting<'a> {
    fn from_func_def_body(
        lifter: &mut Lifter<'_, impl AllocIds>,
        func_def_body: &'a FuncDefBody,
    ) -> Self {
        let cx = lifter.cx;

        let mut region_inputs_source = EntityOrientedDenseMap::new();
        region_inputs_source.insert(func_def_body.body, RegionInputsSource::FuncParams);
        let mut data_insts = EntityOrientedDenseMap::new();

        // Create a SPIR-V block for every CFG point needing one.
        let mut blocks = FxIndexMap::default();
        let mut visit_cfg_point = |point_cursor: CfgCursor<'_>| {
            let point = point_cursor.point;

            let phis = match point {
                CfgPoint::RegionEntry(region) => {
                    if region_inputs_source.get(region).is_some() {
                        // Region inputs handled by the parent of the region.
                        SmallVec::new()
                    } else {
                        func_def_body
                            .at(region)
                            .def()
                            .inputs
                            .iter()
                            .map(|&RegionInputDecl { attrs, ty }| Phi {
                                attrs,
                                ty,

                                result_id: lifter.alloc_ids.one(),
                                cases: FxIndexMap::default(),
                                default_value: None,
                            })
                            .collect()
                    }
                }
                CfgPoint::RegionExit(_) => SmallVec::new(),

                CfgPoint::NodeEntry(node) => {
                    let node_def = func_def_body.at(node).def();
                    match &node_def.kind {
                        // The backedge of a SPIR-V structured loop points to
                        // the "loop header", i.e. the `Entry` of the `Loop`,
                        // so that's where `body` `inputs` phis have to go.
                        NodeKind::Loop { .. } => {
                            let body = node_def.child_regions[0];
                            let loop_body_def = func_def_body.at(body).def();
                            let loop_body_inputs = &loop_body_def.inputs;

                            if !loop_body_inputs.is_empty() {
                                region_inputs_source
                                    .insert(body, RegionInputsSource::LoopHeaderPhis(node));
                            }

                            loop_body_inputs
                                .iter()
                                .enumerate()
                                .map(|(i, &RegionInputDecl { attrs, ty })| Phi {
                                    attrs,
                                    ty,

                                    result_id: lifter.alloc_ids.one(),
                                    cases: FxIndexMap::default(),
                                    default_value: Some(node_def.inputs[i]),
                                })
                                .collect()
                        }
                        _ => SmallVec::new(),
                    }
                }
                CfgPoint::NodeExit(node) => {
                    let node_def = func_def_body.at(node).def();
                    match &node_def.kind {
                        NodeKind::Select(_) => node_def
                            .outputs
                            .iter()
                            .map(|&NodeOutputDecl { attrs, ty }| Phi {
                                attrs,
                                ty,

                                result_id: lifter.alloc_ids.one(),
                                cases: FxIndexMap::default(),
                                default_value: None,
                            })
                            .collect(),
                        _ => SmallVec::new(),
                    }
                }
            };

            let insts = match point {
                CfgPoint::NodeEntry(node) => {
                    let func_at_node = func_def_body.at(node);
                    match func_at_node.def().kind {
                        NodeKind::Select(_)
                        | NodeKind::Loop { .. }
                        | NodeKind::ExitInvocation(_) => SmallVec::new(),

                        NodeKind::FuncCall { .. }
                        | DataInstKind::Scalar(_)
                        | DataInstKind::Vector(_)
                        | DataInstKind::QPtr(_)
                        | DataInstKind::SpvInst(..)
                        | DataInstKind::SpvExtInst { .. } => {
                            data_insts
                                .insert(node, DataInstLifting::from_inst(lifter, func_at_node));

                            [node].into_iter().collect()
                        }
                    }
                }
                _ => SmallVec::new(),
            };

            // Get the terminator, or reconstruct it from structured control-flow.
            let terminator = match (point, func_def_body.at(point_cursor).unique_successor()) {
                // Exiting a `Region` w/o a structured parent.
                (CfgPoint::RegionExit(region), None) => {
                    let unstructured_terminator = func_def_body
                        .unstructured_cfg
                        .as_ref()
                        .and_then(|cfg| cfg.control_inst_on_exit_from.get(region));
                    if let Some(terminator) = unstructured_terminator {
                        let cfg::ControlInst { attrs, kind, inputs, targets, target_inputs } =
                            terminator;
                        Terminator {
                            attrs: *attrs,
                            kind: Cow::Borrowed(kind),
                            reaggregated_return_value_id: match kind {
                                cfg::ControlInstKind::Return if inputs.len() > 1 => {
                                    Some(lifter.alloc_ids.one())
                                }
                                _ => None,
                            },
                            // FIXME(eddyb) borrow these whenever possible.
                            inputs: inputs.clone(),
                            targets: targets
                                .iter()
                                .map(|&target| CfgPoint::RegionEntry(target))
                                .collect(),
                            target_phi_values: target_inputs
                                .iter()
                                .map(|(&target, target_inputs)| {
                                    (CfgPoint::RegionEntry(target), &target_inputs[..])
                                })
                                .collect(),
                            merge: None,
                        }
                    } else {
                        // Structured return out of the function body.
                        assert!(region == func_def_body.body);
                        let inputs = func_def_body.at_body().def().outputs.clone();
                        Terminator {
                            attrs: AttrSet::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::Return),
                            reaggregated_return_value_id: if inputs.len() > 1 {
                                Some(lifter.alloc_ids.one())
                            } else {
                                None
                            },
                            inputs,
                            targets: [].into_iter().collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: None,
                        }
                    }
                }

                // Entering a `Node` with child `Region`s (or diverging).
                (CfgPoint::NodeEntry(node), None) => {
                    let node_def = func_def_body.at(node).def();
                    match &node_def.kind {
                        NodeKind::Select(kind) => Terminator {
                            attrs: AttrSet::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::SelectBranch(kind.clone())),
                            reaggregated_return_value_id: None,
                            inputs: [node_def.inputs[0]].into_iter().collect(),
                            targets: node_def
                                .child_regions
                                .iter()
                                .map(|&case| CfgPoint::RegionEntry(case))
                                .collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: Some(Merge::Selection(CfgPoint::NodeExit(node))),
                        },

                        NodeKind::Loop { repeat_condition: _ } => {
                            let body = node_def.child_regions[0];
                            Terminator {
                                attrs: AttrSet::default(),
                                kind: Cow::Owned(cfg::ControlInstKind::Branch),
                                reaggregated_return_value_id: None,
                                inputs: [].into_iter().collect(),
                                targets: [CfgPoint::RegionEntry(body)].into_iter().collect(),
                                target_phi_values: FxIndexMap::default(),
                                merge: Some(Merge::Loop {
                                    loop_merge: CfgPoint::NodeExit(node),
                                    // NOTE(eddyb) see the note on `Merge::Loop`'s
                                    // `loop_continue` field - in particular, for
                                    // SPIR-T loops, we *could* pick any point
                                    // before/after/between `body`'s `children`
                                    // and it should be valid *but* that had to be
                                    // reverted because it's only true in the absence
                                    // of divergence within the loop body itself!
                                    loop_continue: CfgPoint::RegionExit(body),
                                }),
                            }
                        }

                        NodeKind::ExitInvocation(kind) => Terminator {
                            attrs: AttrSet::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::ExitInvocation(kind.clone())),
                            reaggregated_return_value_id: None,
                            inputs: node_def.inputs.clone(),
                            targets: [].into_iter().collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: None,
                        },

                        NodeKind::FuncCall { .. }
                        | DataInstKind::Scalar(_)
                        | DataInstKind::Vector(_)
                        | DataInstKind::QPtr(_)
                        | DataInstKind::SpvInst(..)
                        | DataInstKind::SpvExtInst { .. } => unreachable!(),
                    }
                }

                // Exiting a `Region` to the parent `Node`.
                (CfgPoint::RegionExit(region), Some(parent_exit_cursor)) => {
                    let region_outputs = Some(&func_def_body.at(region).def().outputs[..])
                        .filter(|outputs| !outputs.is_empty());

                    let parent_exit = parent_exit_cursor.point;
                    let parent_node = match parent_exit {
                        CfgPoint::NodeExit(parent_node) => parent_node,
                        _ => unreachable!(),
                    };

                    match func_def_body.at(parent_node).def().kind {
                        NodeKind::Select { .. } => Terminator {
                            attrs: AttrSet::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::Branch),
                            reaggregated_return_value_id: None,
                            inputs: [].into_iter().collect(),
                            targets: [parent_exit].into_iter().collect(),
                            target_phi_values: region_outputs
                                .map(|outputs| (parent_exit, outputs))
                                .into_iter()
                                .collect(),
                            merge: None,
                        },

                        NodeKind::Loop { repeat_condition } => {
                            let backedge = CfgPoint::NodeEntry(parent_node);
                            let target_phi_values = region_outputs
                                .map(|outputs| (backedge, outputs))
                                .into_iter()
                                .collect();

                            let is_infinite_loop = match repeat_condition {
                                Value::Const(cond) => {
                                    matches!(cx[cond].kind, ConstKind::Scalar(scalar::Const::TRUE))
                                }
                                _ => false,
                            };
                            if is_infinite_loop {
                                Terminator {
                                    attrs: AttrSet::default(),
                                    kind: Cow::Owned(cfg::ControlInstKind::Branch),
                                    reaggregated_return_value_id: None,
                                    inputs: [].into_iter().collect(),
                                    targets: [backedge].into_iter().collect(),
                                    target_phi_values,
                                    merge: None,
                                }
                            } else {
                                Terminator {
                                    attrs: AttrSet::default(),
                                    kind: Cow::Owned(cfg::ControlInstKind::SelectBranch(
                                        SelectionKind::BoolCond,
                                    )),
                                    reaggregated_return_value_id: None,
                                    inputs: [repeat_condition].into_iter().collect(),
                                    targets: [backedge, parent_exit].into_iter().collect(),
                                    target_phi_values,
                                    merge: None,
                                }
                            }
                        }

                        NodeKind::FuncCall { .. }
                        | NodeKind::ExitInvocation { .. }
                        | DataInstKind::Scalar(_)
                        | DataInstKind::Vector(_)
                        | DataInstKind::QPtr(_)
                        | DataInstKind::SpvInst(..)
                        | DataInstKind::SpvExtInst { .. } => unreachable!(),
                    }
                }

                // Siblings in the same `Region` (including the
                // implied edge from a `DataInst`'s `Entry` to its `Exit`).
                //
                // FIXME(eddyb) reduce the cost of generating then removing most
                // "basic blocks" (as each former-`DataInst` gets *two*!),
                // which should be pretty doable in the common case of getting
                // `NodeEntry(a), NodeExit(a), NodeEntry(b), NodeExit(b), ...`
                // from `rev_post_order_try_for_each` and/or introducing an
                // `unique_predecessor` helper (just like `unique_successor`).
                (_, Some(succ_cursor)) => Terminator {
                    attrs: AttrSet::default(),
                    kind: Cow::Owned(cfg::ControlInstKind::Branch),
                    reaggregated_return_value_id: None,
                    inputs: [].into_iter().collect(),
                    targets: [succ_cursor.point].into_iter().collect(),
                    target_phi_values: FxIndexMap::default(),
                    merge: None,
                },

                // Impossible cases, they always return `(_, Some(_))`.
                (CfgPoint::RegionEntry(_) | CfgPoint::NodeExit(_), None) => {
                    unreachable!()
                }
            };

            blocks.insert(point, BlockLifting { phis, insts, terminator });
        };
        match &func_def_body.unstructured_cfg {
            None => {
                func_def_body.at_body().rev_post_order_for_each(visit_cfg_point);
            }
            Some(cfg) => {
                for region in cfg.rev_post_order(func_def_body) {
                    func_def_body.at(region).rev_post_order_for_each(&mut visit_cfg_point);
                }
            }
        }

        // Count the number of "uses" of each block (each incoming edge, plus
        // `1` for the entry block), to help determine which blocks are part
        // of a linear branch chain (and potentially fusable), later on.
        //
        // FIXME(eddyb) use `EntityOrientedDenseMap` here.
        let mut use_counts = FxHashMap::<CfgPoint, usize>::default();
        use_counts.reserve(blocks.len());
        let all_edges = blocks.first().map(|(&entry_point, _)| entry_point).into_iter().chain(
            blocks.values().flat_map(|block| {
                block
                    .terminator
                    .merge
                    .iter()
                    .flat_map(|merge| {
                        let (a, b) = match merge {
                            Merge::Selection(a) => (a, None),
                            Merge::Loop { loop_merge: a, loop_continue: b } => (a, Some(b)),
                        };
                        [a].into_iter().chain(b)
                    })
                    .chain(&block.terminator.targets)
                    .copied()
            }),
        );
        for target in all_edges {
            *use_counts.entry(target).or_default() += 1;
        }

        // Fuse chains of linear branches, when there is no information being
        // lost by the fusion. This is done in reverse order, so that in e.g.
        // `a -> b -> c`, `b -> c` is fused first, then when the iteration
        // reaches `a`, it sees `a -> bc` and can further fuse that into one
        // `abc` block, without knowing about `b` and `c` themselves
        // (this is possible because RPO will always output `[a, b, c]`, when
        // `b` and `c` only have one predecessor each).
        //
        // FIXME(eddyb) while this could theoretically fuse certain kinds of
        // merge blocks (mostly loop bodies) into their unique precedessor, that
        // would require adjusting the `Merge` that points to them.
        //
        // HACK(eddyb) this takes advantage of `blocks` being an `IndexMap`,
        // to iterate at the same time as mutating other entries.
        for block_idx in (0..blocks.len()).rev() {
            let BlockLifting { terminator: original_terminator, .. } = &blocks[block_idx];

            let is_trivial_branch = {
                let Terminator {
                    attrs,
                    kind,
                    reaggregated_return_value_id,
                    inputs,
                    targets,
                    target_phi_values,
                    merge,
                } = original_terminator;

                *attrs == AttrSet::default()
                    && matches!(**kind, cfg::ControlInstKind::Branch)
                    && reaggregated_return_value_id.is_none()
                    && inputs.is_empty()
                    && targets.len() == 1
                    && target_phi_values.is_empty()
                    && merge.is_none()
            };

            if is_trivial_branch {
                let target = original_terminator.targets[0];
                let target_use_count = use_counts.get_mut(&target).unwrap();

                if *target_use_count == 1 {
                    let BlockLifting {
                        phis: ref target_phis,
                        insts: ref mut extra_insts,
                        terminator: ref mut new_terminator,
                    } = blocks[&target];

                    // FIXME(eddyb) check for block-level attributes, once/if
                    // they start being tracked.
                    if target_phis.is_empty() {
                        let extra_insts = mem::take(extra_insts);
                        let new_terminator = mem::replace(new_terminator, Terminator {
                            attrs: Default::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::Unreachable),
                            reaggregated_return_value_id: None,
                            inputs: Default::default(),
                            targets: Default::default(),
                            target_phi_values: Default::default(),
                            merge: None,
                        });
                        *target_use_count = 0;

                        let combined_block = &mut blocks[block_idx];
                        combined_block.insts.extend(extra_insts);
                        combined_block.terminator = new_terminator;
                    }
                }
            }
        }

        // Remove now-unused blocks.
        blocks.retain(|point, _| use_counts.get(point).is_some_and(|&count| count > 0));

        // Collect `OpPhi`s from other blocks' edges into each block.
        //
        // HACK(eddyb) this takes advantage of `blocks` being an `IndexMap`,
        // to iterate at the same time as mutating other entries.
        for source_block_idx in 0..blocks.len() {
            let (&source_point, source_block) = blocks.get_index(source_block_idx).unwrap();
            let targets = source_block.terminator.targets.clone();

            for target in targets {
                let source_values = {
                    let (_, source_block) = blocks.get_index(source_block_idx).unwrap();
                    source_block.terminator.target_phi_values.get(&target).copied()
                };
                let target_block = blocks.get_mut(&target).unwrap();
                for (i, target_phi) in target_block.phis.iter_mut().enumerate() {
                    use indexmap::map::Entry;

                    let source_value =
                        source_values.map(|values| values[i]).or(target_phi.default_value).unwrap();
                    match target_phi.cases.entry(source_point) {
                        Entry::Vacant(entry) => {
                            entry.insert(source_value);
                        }

                        // NOTE(eddyb) the only reason duplicates are allowed,
                        // is that `targets` may itself contain the same target
                        // multiple times (which would result in the same value).
                        Entry::Occupied(entry) => {
                            assert!(*entry.get() == source_value);
                        }
                    }
                }
            }
        }

        Self {
            region_inputs_source,
            data_insts,

            label_ids: blocks.keys().map(|&point| (point, lifter.alloc_ids.one())).collect(),
            blocks,
        }
    }
}

impl DataInstLifting {
    fn from_inst(
        lifter: &mut Lifter<'_, impl AllocIds>,
        func_at_inst: FuncAt<'_, DataInst>,
    ) -> Self {
        let wk = &spec::Spec::get().well_known;
        let cx = lifter.cx;

        let inst_def = func_at_inst.def();
        let output_types = inst_def.outputs.iter().map(|o| o.ty);

        let mut new_spv_inst_lowering = spv::InstLowering::default();
        let spv_inst_lowering = match &inst_def.kind {
            NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::ExitInvocation(_) => {
                unreachable!()
            }

            // Disallowed while visiting.
            DataInstKind::QPtr(_) => unreachable!(),

            DataInstKind::Scalar(_) | DataInstKind::Vector(_) => {
                // FIXME(eddyb) deduplicate creating this `OpTypeStruct`.
                if output_types.len() > 1 {
                    let tuple_ty =
                        cx.intern(spv::Inst::from(wk.OpTypeStruct).into_canonical_type_with(
                            cx,
                            output_types.clone().map(TypeOrConst::Type).collect(),
                        ));
                    lifter.visit_type_use(tuple_ty);
                    new_spv_inst_lowering.disaggregated_output = Some(tuple_ty);
                }
                &new_spv_inst_lowering
            }
            NodeKind::FuncCall { callee } => {
                if output_types.len() > 1 {
                    new_spv_inst_lowering.disaggregated_output =
                        Some(lifter.ids.funcs[callee].spv_func_ret_type);
                }
                &new_spv_inst_lowering
            }
            DataInstKind::SpvInst(_, lowering) | DataInstKind::SpvExtInst { lowering, .. } => {
                lowering
            }
        };

        let reaggregate_inputs = spv_inst_lowering
            .disaggregated_inputs
            .iter()
            .map(|&(_, ty)| {
                let op_undef =
                    cx.intern(ConstDef { attrs: AttrSet::default(), ty, kind: ConstKind::Undef });
                lifter.visit_const_use(op_undef);
                let op_composite_insert_result_ids =
                    (lifter.alloc_ids)(cx[ty].disaggregated_leaf_count());
                ReaggregateFromLeaves { op_undef, op_composite_insert_result_ids }
            })
            .collect();

        // `OpFunctionCall always has a result (but may be `OpTypeVoid`-typed).
        let has_result = matches!(inst_def.kind, NodeKind::FuncCall { .. })
            || spv_inst_lowering.disaggregated_output.is_some()
            || output_types.len() > 0;
        let result_id = if has_result { Some(lifter.alloc_ids.one()) } else { None };

        let disaggregate_result =
            spv_inst_lowering.disaggregated_output.map(|ty| DisaggregateToLeaves {
                op_composite_extract_result_ids: (lifter.alloc_ids)(
                    cx[ty].disaggregated_leaf_count(),
                ),
            });

        DataInstLifting { result_id, disaggregate_result, reaggregate_inputs }
    }

    fn id_for_output(&self, output_idx: u32) -> spv::Id {
        let output_idx = usize::try_from(output_idx).unwrap();
        if let Some(disaggregate_result) = &self.disaggregate_result {
            let result_id = disaggregate_result
                .op_composite_extract_result_ids
                .start
                .checked_add(output_idx.try_into().unwrap())
                .unwrap();
            assert!(disaggregate_result.op_composite_extract_result_ids.contains(&result_id));
            result_id
        } else {
            assert_eq!(output_idx, 0);
            self.result_id.unwrap()
        }
    }
}

/// Maybe-decorated "lazy" SPIR-V instruction, allowing separately emitting
/// *both* decorations (from certain [`Attr`]s), *and* the instruction itself,
/// without eagerly allocating all the instructions.
///
/// Note that SPIR-T disaggregating SPIR-V `OpTypeStruct`/`OpTypeArray`s values
/// may require additional [`spv::Inst`]s for each `LazyInst`, either for
/// reaggregating inputs, or taking apart aggregate outputs.
#[derive(Copy, Clone)]
enum LazyInst<'a, 'b> {
    Global(Global),
    OpFunction {
        func_decl: &'a FuncDecl,
        func_ids: &'b FuncIds<'a>,
    },
    OpFunctionParameter {
        param_id: spv::Id,
        param: &'a FuncParam,
    },
    OpLabel {
        label_id: spv::Id,
    },
    OpPhi {
        parent_func_ids: &'b FuncIds<'a>,
        phi: &'b Phi,
    },
    DataInst {
        parent_func_ids: &'b FuncIds<'a>,
        data_inst_def: &'a DataInstDef,
        lifting: &'b DataInstLifting,
    },
    // FIXME(eddyb) should merge instructions be generated by `Terminator`?
    Merge(Merge<spv::Id>),
    Terminator {
        parent_func_ids: &'b FuncIds<'a>,
        terminator: &'b Terminator<'a>,
    },
    OpFunctionEnd,
}

/// [`Attr::DbgSrcLoc`], extracted from [`AttrSet`], and used for emitting
/// `OpLine`/`OpNoLine` SPIR-V instructions.
#[derive(Copy, Clone, PartialEq, Eq)]
struct SpvDebugLine {
    file_path_id: spv::Id,
    line: u32,
    col: u32,
}

impl LazyInst<'_, '_> {
    fn result_id_attrs_and_import(
        self,
        module: &Module,
        ids: &ModuleIds<'_>,
    ) -> (Option<spv::Id>, AttrSet, Option<Import>) {
        let cx = module.cx_ref();

        #[allow(clippy::match_same_arms)]
        match self {
            Self::Global(global) => {
                let (attrs, import) = match global {
                    Global::Type(ty) => (cx[ty].attrs, None),
                    Global::Const(ct) => {
                        let ct_def = &cx[ct];
                        match ct_def.kind {
                            ConstKind::PtrToGlobalVar(gv) => {
                                let gv_decl = &module.global_vars[gv];
                                let import = match gv_decl.def {
                                    DeclDef::Imported(import) => Some(import),
                                    DeclDef::Present(_) => None,
                                };
                                (gv_decl.attrs, import)
                            }

                            ConstKind::Undef
                            | ConstKind::Scalar(_)
                            | ConstKind::Vector(_)
                            | ConstKind::SpvInst { .. } => (ct_def.attrs, None),

                            // Not inserted into `globals` while visiting.
                            ConstKind::SpvStringLiteralForExtInst(_) => unreachable!(),
                        }
                    }
                };
                (Some(ids.globals[&global]), attrs, import)
            }
            Self::OpFunction { func_decl, func_ids } => {
                let import = match func_decl.def {
                    DeclDef::Imported(import) => Some(import),
                    DeclDef::Present(_) => None,
                };
                (Some(func_ids.func_id), func_decl.attrs, import)
            }
            Self::OpFunctionParameter { param_id, param } => (Some(param_id), param.attrs, None),
            Self::OpLabel { label_id } => (Some(label_id), AttrSet::default(), None),
            Self::OpPhi { parent_func_ids: _, phi } => (Some(phi.result_id), phi.attrs, None),
            Self::DataInst { parent_func_ids: _, data_inst_def, lifting } => {
                (lifting.result_id, data_inst_def.attrs, None)
            }
            Self::Merge(_) => (None, AttrSet::default(), None),
            Self::Terminator { parent_func_ids: _, terminator } => (None, terminator.attrs, None),
            Self::OpFunctionEnd => (None, AttrSet::default(), None),
        }
    }

    /// Expand this `LazyInst` to one or more (see disaggregation/reaggregation
    /// note in [`LazyInst`]'s doc comment for when it can be more than one)
    /// [`spv::Inst`]s (with their respective [`SpvDebugLine`]s, if applicable),
    /// with `each_spv_inst_with_debug_line` being called for each one.
    fn for_each_spv_inst_with_debug_line(
        self,
        module: &Module,
        ids: &ModuleIds<'_>,
        mut each_spv_inst_with_debug_line: impl FnMut(spv::InstWithIds, Option<SpvDebugLine>),
    ) {
        let wk = &spec::Spec::get().well_known;
        let cx = module.cx_ref();

        let value_to_id = |parent_func_ids: &FuncIds<'_>, v| match v {
            Value::Const(ct) => match cx[ct].kind {
                ConstKind::SpvStringLiteralForExtInst(s) => ids.debug_strings[&cx[s]],

                _ => ids.globals[&Global::Const(ct)],
            },
            Value::RegionInput { region, input_idx } => {
                let input_idx = usize::try_from(input_idx).unwrap();
                let parent_func_body_lifting = parent_func_ids.body.as_ref().unwrap();
                match parent_func_body_lifting.region_inputs_source.get(region) {
                    Some(RegionInputsSource::FuncParams) => {
                        let param_id = parent_func_ids
                            .param_ids
                            .start
                            .checked_add(input_idx.try_into().unwrap())
                            .unwrap();
                        assert!(parent_func_ids.param_ids.contains(&param_id));
                        param_id
                    }
                    Some(&RegionInputsSource::LoopHeaderPhis(loop_node)) => {
                        parent_func_body_lifting.blocks[&CfgPoint::NodeEntry(loop_node)].phis
                            [input_idx]
                            .result_id
                    }
                    None => {
                        parent_func_body_lifting.blocks[&CfgPoint::RegionEntry(region)].phis
                            [input_idx]
                            .result_id
                    }
                }
            }
            Value::NodeOutput { node, output_idx } => {
                let parent_func_body_lifting = parent_func_ids.body.as_ref().unwrap();
                if let Some(inst_lifting) = parent_func_body_lifting.data_insts.get(node) {
                    inst_lifting.id_for_output(output_idx)
                } else {
                    parent_func_body_lifting.blocks[&CfgPoint::NodeExit(node)].phis
                        [usize::try_from(output_idx).unwrap()]
                    .result_id
                }
            }
        };

        let (result_id, attrs, _) = self.result_id_attrs_and_import(module, ids);

        let spv_debug_line = attrs.dbg_src_loc(cx).map(|dbg_src_loc| SpvDebugLine {
            file_path_id: ids.debug_strings[&cx[dbg_src_loc.file_path]],
            line: dbg_src_loc.start_line_col.0,
            col: dbg_src_loc.start_line_col.1,
        });

        // HACK(eddyb) there is no need to allow `spv_debug_line` to vary per-inst.
        let mut each_inst = |inst| each_spv_inst_with_debug_line(inst, spv_debug_line);

        match self {
            Self::Global(global) => each_inst(match global {
                Global::Type(ty) => {
                    let ty_def = &cx[ty];
                    match spv::Inst::from_canonical_type(cx, &ty_def.kind)
                        .as_ref()
                        .ok_or(&ty_def.kind)
                    {
                        Err(TypeKind::Scalar(_) | TypeKind::Vector(_)) => {
                            unreachable!("should've been handled as canonical")
                        }

                        Ok((spv_inst, type_and_const_inputs))
                        | Err(TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }) => {
                            spv::InstWithIds {
                                without_ids: spv_inst.clone(),
                                result_type_id: None,
                                result_id,
                                ids: type_and_const_inputs
                                    .iter()
                                    .map(|&ty_or_ct| {
                                        ids.globals[&match ty_or_ct {
                                            TypeOrConst::Type(ty) => Global::Type(ty),
                                            TypeOrConst::Const(ct) => Global::Const(ct),
                                        }]
                                    })
                                    .collect(),
                            }
                        }

                        // Not inserted into `globals` while visiting.
                        Err(TypeKind::QPtr | TypeKind::SpvStringLiteralForExtInst) => {
                            unreachable!()
                        }
                    }
                }
                Global::Const(ct) => {
                    let ct_def = &cx[ct];
                    match spv::Inst::from_canonical_const(cx, &ct_def.kind).ok_or(&ct_def.kind) {
                        // FIXME(eddyb) this duplicates the `ConstKind::SpvInst`
                        // case, only due to an inability to pattern-match `Rc`.
                        Ok((spv_inst, const_inputs)) => spv::InstWithIds {
                            without_ids: spv_inst,
                            result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                            result_id,
                            ids: const_inputs
                                .iter()
                                .map(|&ct| ids.globals[&Global::Const(ct)])
                                .collect(),
                        },
                        Err(ConstKind::SpvInst { spv_inst_and_const_inputs }) => {
                            let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                            spv::InstWithIds {
                                without_ids: spv_inst.clone(),
                                result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                                result_id,
                                ids: const_inputs
                                    .iter()
                                    .map(|&ct| ids.globals[&Global::Const(ct)])
                                    .collect(),
                            }
                        }

                        Err(ConstKind::Undef | ConstKind::Scalar(_) | ConstKind::Vector(_)) => {
                            unreachable!("should've been handled as canonical")
                        }

                        Err(&ConstKind::PtrToGlobalVar(gv)) => {
                            assert!(ct_def.attrs == AttrSet::default());

                            let gv_decl = &module.global_vars[gv];

                            assert!(ct_def.ty == gv_decl.type_of_ptr_to);

                            let storage_class = match gv_decl.addr_space {
                                AddrSpace::Handles => {
                                    unreachable!(
                                        "`AddrSpace::Handles` should be legalized away before lifting"
                                    );
                                }
                                AddrSpace::SpvStorageClass(sc) => {
                                    spv::Imm::Short(wk.StorageClass, sc)
                                }
                            };
                            let initializer = match gv_decl.def {
                                DeclDef::Imported(_) => None,
                                DeclDef::Present(GlobalVarDefBody { initializer }) => initializer
                                    .map(|initializer| ids.globals[&Global::Const(initializer)]),
                            };
                            spv::InstWithIds {
                                without_ids: spv::Inst {
                                    opcode: wk.OpVariable,
                                    imms: iter::once(storage_class).collect(),
                                },
                                result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                                result_id,
                                ids: initializer.into_iter().collect(),
                            }
                        }

                        // Not inserted into `globals` while visiting.
                        Err(ConstKind::SpvStringLiteralForExtInst(_)) => unreachable!(),
                    }
                }
            }),
            Self::OpFunction { func_decl: _, func_ids } => {
                // FIXME(eddyb) make this less of a search and more of a
                // lookup by splitting attrs into key and value parts.
                let func_ctrl = cx[attrs]
                    .attrs
                    .iter()
                    .find_map(|attr| match *attr {
                        Attr::SpvBitflagsOperand(spv::Imm::Short(kind, word))
                            if kind == wk.FunctionControl =>
                        {
                            Some(word)
                        }
                        _ => None,
                    })
                    .unwrap_or(0);

                each_inst(spv::InstWithIds {
                    without_ids: spv::Inst {
                        opcode: wk.OpFunction,
                        imms: iter::once(spv::Imm::Short(wk.FunctionControl, func_ctrl)).collect(),
                    },
                    result_type_id: Some(ids.globals[&Global::Type(func_ids.spv_func_ret_type)]),
                    result_id,
                    ids: iter::once(ids.globals[&Global::Type(func_ids.spv_func_type)]).collect(),
                });
            }
            Self::OpFunctionParameter { param_id: _, param } => each_inst(spv::InstWithIds {
                without_ids: wk.OpFunctionParameter.into(),
                result_type_id: Some(ids.globals[&Global::Type(param.ty)]),
                result_id,
                ids: [].into_iter().collect(),
            }),
            Self::OpLabel { label_id: _ } => each_inst(spv::InstWithIds {
                without_ids: wk.OpLabel.into(),
                result_type_id: None,
                result_id,
                ids: [].into_iter().collect(),
            }),
            Self::OpPhi { parent_func_ids, phi } => each_inst(spv::InstWithIds {
                without_ids: wk.OpPhi.into(),
                result_type_id: Some(ids.globals[&Global::Type(phi.ty)]),
                result_id: Some(phi.result_id),
                ids: phi
                    .cases
                    .iter()
                    .flat_map(|(&source_point, &v)| {
                        [
                            value_to_id(parent_func_ids, v),
                            parent_func_ids.body.as_ref().unwrap().label_ids[&source_point],
                        ]
                    })
                    .collect(),
            }),
            Self::DataInst { parent_func_ids, data_inst_def, lifting } => {
                let kind = &data_inst_def.kind;
                let output_types = data_inst_def.outputs.iter().map(|o| o.ty);

                let mut id_operands = SmallVec::new();

                let mut new_spv_inst_lowering = spv::InstLowering::default();
                let mut override_result_type = None;
                let (inst, spv_inst_lowering) = match spv::Inst::from_canonical_data_inst_kind(kind)
                    .ok_or(kind)
                {
                    Ok(spv_inst) => {
                        // FIXME(eddyb) deduplicate creating this `OpTypeStruct`.
                        if output_types.len() > 1 {
                            new_spv_inst_lowering.disaggregated_output = Some(cx.intern(
                                spv::Inst::from(wk.OpTypeStruct).into_canonical_type_with(
                                    cx,
                                    output_types.clone().map(TypeOrConst::Type).collect(),
                                ),
                            ));
                        }
                        (spv_inst, &new_spv_inst_lowering)
                    }

                    Err(
                        NodeKind::Select(_) | NodeKind::Loop { .. } | NodeKind::ExitInvocation(_),
                    ) => unreachable!(),

                    Err(DataInstKind::Scalar(_) | DataInstKind::Vector(_)) => {
                        unreachable!("should've been handled as canonical")
                    }

                    Err(DataInstKind::QPtr(_)) => {
                        // Disallowed while visiting.
                        unreachable!()
                    }

                    // `OpFunctionCall` always has a result (but may be `OpTypeVoid`-typed).
                    Err(NodeKind::FuncCall { callee }) => {
                        let callee_ids = &ids.funcs[callee];
                        override_result_type = Some(callee_ids.spv_func_ret_type);
                        if output_types.len() > 1 {
                            new_spv_inst_lowering.disaggregated_output = override_result_type;
                        }
                        id_operands.push(callee_ids.func_id);
                        (wk.OpFunctionCall.into(), &new_spv_inst_lowering)
                    }
                    Err(DataInstKind::SpvInst(inst, lowering)) => (inst.clone(), lowering),
                    Err(DataInstKind::SpvExtInst { ext_set, inst, lowering }) => {
                        id_operands.push(ids.ext_inst_imports[&cx[*ext_set]]);
                        (
                            spv::Inst {
                                opcode: wk.OpExtInst,
                                imms: [spv::Imm::Short(wk.LiteralExtInstInteger, *inst)]
                                    .into_iter()
                                    .collect(),
                            },
                            lowering,
                        )
                    }
                };

                let int_imm = |i| spv::Imm::Short(wk.LiteralInteger, i);

                // Emit any `OpCompositeInsert`s needed by the inputs, first,
                // while gathering the `id_operands` for the instruction itself.
                let mut reaggregate_inputs = lifting.reaggregate_inputs.iter();
                for id_operand in spv_inst_lowering.reaggreate_inputs(&data_inst_def.inputs) {
                    let value_to_id = |v| value_to_id(parent_func_ids, v);
                    let id_operand = match id_operand {
                        spv::ReaggregatedIdOperand::Direct(v) => value_to_id(v),
                        spv::ReaggregatedIdOperand::Aggregate { ty, leaves } => {
                            let result_type_id = Some(ids.globals[&Global::Type(ty)]);

                            let ReaggregateFromLeaves { op_undef, op_composite_insert_result_ids } =
                                reaggregate_inputs.next().unwrap();
                            let mut aggregate_id = ids.globals[&Global::Const(*op_undef)];
                            let leaf_paths = ty
                                .disaggregated_leaf_types(cx)
                                .map_with_parent_component_path(|_, leaf_path| {
                                    leaf_path.iter().map(|&(_, i)| i).map(int_imm).collect()
                                });
                            for ((leaf_path_imms, op_composite_insert_result_id), &leaf_value) in
                                leaf_paths
                                    .zip_eq(op_composite_insert_result_ids.iter())
                                    .zip_eq(leaves)
                            {
                                each_inst(spv::InstWithIds {
                                    without_ids: spv::Inst {
                                        opcode: wk.OpCompositeInsert,
                                        imms: leaf_path_imms,
                                    },
                                    result_type_id,
                                    result_id: Some(op_composite_insert_result_id),
                                    ids: [value_to_id(leaf_value), aggregate_id]
                                        .into_iter()
                                        .collect(),
                                });
                                aggregate_id = op_composite_insert_result_id;
                            }
                            aggregate_id
                        }
                    };
                    id_operands.push(id_operand);
                }
                assert!(reaggregate_inputs.next().is_none());

                let result_type = override_result_type
                    .or(spv_inst_lowering.disaggregated_output)
                    .or_else(|| output_types.at_most_one().ok().unwrap());
                each_inst(spv::InstWithIds {
                    without_ids: inst,
                    result_type_id: result_type.map(|ty| ids.globals[&Global::Type(ty)]),
                    result_id,
                    ids: id_operands,
                });

                // Emit any `OpCompositeExtract`s needed for the result, last.
                if let Some(DisaggregateToLeaves { op_composite_extract_result_ids }) =
                    &lifting.disaggregate_result
                {
                    let aggregate_id = result_id.unwrap();
                    let leaf_types_and_paths = spv_inst_lowering
                        .disaggregated_output
                        .unwrap()
                        .disaggregated_leaf_types(cx)
                        .map_with_parent_component_path(|leaf_type, leaf_path| {
                            (leaf_type, leaf_path.iter().map(|&(_, i)| i).map(int_imm).collect())
                        });
                    for ((leaf_type, leaf_path_imms), op_composite_extract_result_id) in
                        leaf_types_and_paths.zip_eq(op_composite_extract_result_ids.iter())
                    {
                        each_inst(spv::InstWithIds {
                            without_ids: spv::Inst {
                                opcode: wk.OpCompositeExtract,
                                imms: leaf_path_imms,
                            },
                            result_type_id: Some(ids.globals[&Global::Type(leaf_type)]),
                            result_id: Some(op_composite_extract_result_id),
                            ids: [aggregate_id].into_iter().collect(),
                        });
                    }
                }
            }
            // FIXME(eddyb) should merge instructions be generated by `Terminator`?
            Self::Merge(Merge::Selection(merge_label_id)) => each_inst(spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpSelectionMerge,
                    imms: [spv::Imm::Short(wk.SelectionControl, 0)].into_iter().collect(),
                },
                result_type_id: None,
                result_id: None,
                ids: [merge_label_id].into_iter().collect(),
            }),
            Self::Merge(Merge::Loop {
                loop_merge: merge_label_id,
                loop_continue: continue_label_id,
            }) => each_inst(spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpLoopMerge,
                    imms: [spv::Imm::Short(wk.LoopControl, 0)].into_iter().collect(),
                },
                result_type_id: None,
                result_id: None,
                ids: [merge_label_id, continue_label_id].into_iter().collect(),
            }),
            Self::Terminator { parent_func_ids, terminator } => {
                let parent_func_body_lifting = parent_func_ids.body.as_ref().unwrap();
                let mut id_operands = terminator
                    .inputs
                    .iter()
                    .map(|&v| value_to_id(parent_func_ids, v))
                    .chain(
                        terminator
                            .targets
                            .iter()
                            .map(|&target| parent_func_body_lifting.label_ids[&target]),
                    )
                    .collect();

                if let Some(reaggregated_value_id) = terminator.reaggregated_return_value_id {
                    assert!(
                        matches!(*terminator.kind, cfg::ControlInstKind::Return)
                            && terminator.inputs.len() > 1
                    );

                    each_inst(spv::InstWithIds {
                        without_ids: wk.OpCompositeConstruct.into(),
                        result_type_id: Some(
                            ids.globals[&Global::Type(parent_func_ids.spv_func_ret_type)],
                        ),
                        result_id: Some(reaggregated_value_id),
                        ids: id_operands,
                    });
                    id_operands = [reaggregated_value_id].into_iter().collect();
                }

                // FIXME(eddyb) move some of this to `spv::canonical`.
                let inst = match &*terminator.kind {
                    cfg::ControlInstKind::Unreachable => wk.OpUnreachable.into(),
                    cfg::ControlInstKind::Return => {
                        if terminator.inputs.is_empty() {
                            wk.OpReturn.into()
                        } else {
                            // Multiple return values get reaggregated above.
                            assert_eq!(id_operands.len(), 1);
                            wk.OpReturnValue.into()
                        }
                    }
                    cfg::ControlInstKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(
                        inst,
                    )) => inst.clone(),

                    cfg::ControlInstKind::Branch => wk.OpBranch.into(),

                    cfg::ControlInstKind::SelectBranch(SelectionKind::BoolCond) => {
                        wk.OpBranchConditional.into()
                    }
                    cfg::ControlInstKind::SelectBranch(SelectionKind::Switch { case_consts }) => {
                        // HACK(eddyb) move the default case from last back to first.
                        let default_target = id_operands.pop().unwrap();
                        id_operands.insert(1, default_target);

                        spv::Inst {
                            opcode: wk.OpSwitch,
                            imms: case_consts
                                .iter()
                                .flat_map(|ct| ct.encode_as_spv_imms())
                                .collect(),
                        }
                    }
                };
                each_inst(spv::InstWithIds {
                    without_ids: inst,
                    result_type_id: None,
                    result_id: None,
                    ids: id_operands,
                });
            }
            Self::OpFunctionEnd => each_inst(spv::InstWithIds {
                without_ids: wk.OpFunctionEnd.into(),
                result_type_id: None,
                result_id: None,
                ids: [].into_iter().collect(),
            }),
        }
    }
}

impl Module {
    pub fn lift_to_spv_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.lift_to_spv_module_emitter()?.write_to_spv_file(path)
    }

    pub fn lift_to_spv_module_emitter(&self) -> io::Result<spv::write::ModuleEmitter> {
        let spv_spec = spec::Spec::get();
        let wk = &spv_spec.well_known;

        let cx = self.cx();
        let (dialect, debug_info) = match (&self.dialect, &self.debug_info) {
            (ModuleDialect::Spv(dialect), ModuleDebugInfo::Spv(debug_info)) => {
                (dialect, debug_info)
            }

            // FIXME(eddyb) support by computing some valid "minimum viable"
            // `spv::Dialect`, or by taking it as additional input.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "not a SPIR-V module"));
            }
        };

        // Because `GlobalVar`s are given IDs by the `Const`s that point to them
        // (i.e. `ConstKind::PtrToGlobalVar`), any `GlobalVar`s in other positions
        // require extra care to ensure the ID-giving `Const` is visited.
        let global_var_to_id_giving_global = |gv| {
            let type_of_ptr_to_global_var = self.global_vars[gv].type_of_ptr_to;
            let ptr_to_global_var = cx.intern(ConstDef {
                attrs: AttrSet::default(),
                ty: type_of_ptr_to_global_var,
                kind: ConstKind::PtrToGlobalVar(gv),
            });
            Global::Const(ptr_to_global_var)
        };

        // Collect uses scattered throughout the module, allocating IDs for them.
        let (ids, id_bound) = {
            let mut id_bound = NonZeroUsize::MIN;
            let mut lifter = Lifter {
                cx: &cx,
                module: self,
                alloc_ids: |count| {
                    let start = id_bound;
                    let end =
                        start.checked_add(count).expect("overflowing `usize` should be impossible");
                    id_bound = end;

                    // NOTE(eddyb) `MAX` is just a placeholder - the check for overflows
                    // is done below, after all IDs that may be allocated, have been
                    // (this is in order to not need this closure to return a `Result`).
                    let from_usize =
                        |id| spv::Id::try_from(id).unwrap_or(spv::Id::new(u32::MAX).unwrap());
                    from_usize(start)..from_usize(end)
                },
                ids: ModuleIds::default(),
                global_vars_seen: FxIndexSet::default(),
            };
            lifter.visit_module(self);

            // See comment on `global_var_to_id_giving_global` for why this is here.
            for &gv in &lifter.global_vars_seen {
                lifter
                    .ids
                    .globals
                    .entry(global_var_to_id_giving_global(gv))
                    .or_insert_with(|| lifter.alloc_ids.one());
            }

            let ids = lifter.ids;

            let id_bound = spv::Id::try_from(id_bound).ok().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "ID bound of SPIR-V module doesn't fit in 32 bits",
                )
            })?;

            (ids, id_bound)
        };

        // HACK(eddyb) allow `move` closures below to reference `cx` or `ids`
        // without causing unwanted moves out of them.
        let (cx, ids) = (&*cx, &ids);

        let global_and_func_insts = ids.globals.keys().copied().map(LazyInst::Global).chain(
            ids.funcs.iter().flat_map(|(&func, func_ids)| {
                let func_decl = &self.funcs[func];
                let body_with_lifting = match (&func_decl.def, &func_ids.body) {
                    (DeclDef::Imported(_), None) => None,
                    (DeclDef::Present(def), Some(func_body_lifting)) => {
                        Some((def, func_body_lifting))
                    }
                    _ => unreachable!(),
                };

                let param_insts = func_ids
                    .param_ids
                    .iter()
                    .zip_eq(&func_decl.params)
                    .map(|(param_id, param)| LazyInst::OpFunctionParameter { param_id, param });
                let body_insts = body_with_lifting.map(|(func_def_body, func_body_lifting)| {
                    func_body_lifting.blocks.iter().flat_map(move |(point, block)| {
                        let BlockLifting { phis, insts, terminator } = block;

                        iter::once(LazyInst::OpLabel {
                            label_id: func_body_lifting.label_ids[point],
                        })
                        .chain(
                            phis.iter()
                                .map(|phi| LazyInst::OpPhi { parent_func_ids: func_ids, phi }),
                        )
                        .chain(insts.iter().copied().map(move |inst| LazyInst::DataInst {
                            parent_func_ids: func_ids,
                            data_inst_def: func_def_body.at(inst).def(),
                            lifting: &func_body_lifting.data_insts[inst],
                        }))
                        .chain(terminator.merge.map(|merge| {
                            LazyInst::Merge(match merge {
                                Merge::Selection(merge) => {
                                    Merge::Selection(func_body_lifting.label_ids[&merge])
                                }
                                Merge::Loop { loop_merge, loop_continue } => Merge::Loop {
                                    loop_merge: func_body_lifting.label_ids[&loop_merge],
                                    loop_continue: func_body_lifting.label_ids[&loop_continue],
                                },
                            })
                        }))
                        .chain([LazyInst::Terminator { parent_func_ids: func_ids, terminator }])
                    })
                });

                iter::once(LazyInst::OpFunction { func_decl, func_ids })
                    .chain(param_insts)
                    .chain(body_insts.into_iter().flatten())
                    .chain([LazyInst::OpFunctionEnd])
            }),
        );

        let reserved_inst_schema = 0;
        let header = [
            spv_spec.magic,
            (u32::from(dialect.version_major) << 16) | (u32::from(dialect.version_minor) << 8),
            debug_info.original_generator_magic.map_or(0, |x| x.get()),
            id_bound.get(),
            reserved_inst_schema,
        ];

        let mut emitter = spv::write::ModuleEmitter::with_header(header);

        for cap_inst in dialect.capability_insts() {
            emitter.push_inst(&cap_inst)?;
        }
        for ext_inst in dialect.extension_insts() {
            emitter.push_inst(&ext_inst)?;
        }
        for (&name, &id) in &ids.ext_inst_imports {
            emitter.push_inst(&spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpExtInstImport,
                    imms: spv::encode_literal_string(name).collect(),
                },
                result_type_id: None,
                result_id: Some(id),
                ids: [].into_iter().collect(),
            })?;
        }
        emitter.push_inst(&spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpMemoryModel,
                imms: [
                    spv::Imm::Short(wk.AddressingModel, dialect.addressing_model),
                    spv::Imm::Short(wk.MemoryModel, dialect.memory_model),
                ]
                .into_iter()
                .collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })?;

        // Collect the various sources of attributes.
        let mut entry_point_insts = vec![];
        let mut execution_mode_insts = vec![];
        let mut debug_name_insts = vec![];
        let mut decoration_insts = vec![];

        for lazy_inst in global_and_func_insts.clone() {
            let (result_id, attrs, import) = lazy_inst.result_id_attrs_and_import(self, ids);

            for attr in cx[attrs].attrs.iter() {
                match attr {
                    Attr::DbgSrcLoc(_)
                    | Attr::Diagnostics(_)
                    | Attr::QPtr(_)
                    | Attr::SpvBitflagsOperand(_) => {}
                    Attr::SpvAnnotation(inst @ spv::Inst { opcode, .. }) => {
                        let target_id = result_id.expect(
                            "FIXME: it shouldn't be possible to attach \
                                 attributes to instructions without an output",
                        );

                        let inst = spv::InstWithIds {
                            without_ids: inst.clone(),
                            result_type_id: None,
                            result_id: None,
                            ids: iter::once(target_id).collect(),
                        };

                        if [wk.OpExecutionMode, wk.OpExecutionModeId].contains(opcode) {
                            execution_mode_insts.push(inst);
                        } else if [wk.OpName, wk.OpMemberName].contains(opcode) {
                            debug_name_insts.push(inst);
                        } else {
                            decoration_insts.push(inst);
                        }
                    }
                }

                if let Some(import) = import {
                    let target_id = result_id.unwrap();
                    match import {
                        Import::LinkName(name) => {
                            decoration_insts.push(spv::InstWithIds {
                                without_ids: spv::Inst {
                                    opcode: wk.OpDecorate,
                                    imms: iter::once(spv::Imm::Short(
                                        wk.Decoration,
                                        wk.LinkageAttributes,
                                    ))
                                    .chain(spv::encode_literal_string(&cx[name]))
                                    .chain([spv::Imm::Short(wk.LinkageType, wk.Import)])
                                    .collect(),
                                },
                                result_type_id: None,
                                result_id: None,
                                ids: iter::once(target_id).collect(),
                            });
                        }
                    }
                }
            }
        }

        for (export_key, &exportee) in &self.exports {
            let target_id = match exportee {
                Exportee::GlobalVar(gv) => ids.globals[&global_var_to_id_giving_global(gv)],
                Exportee::Func(func) => ids.funcs[&func].func_id,
            };
            match export_key {
                &ExportKey::LinkName(name) => {
                    decoration_insts.push(spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpDecorate,
                            imms: iter::once(spv::Imm::Short(wk.Decoration, wk.LinkageAttributes))
                                .chain(spv::encode_literal_string(&cx[name]))
                                .chain([spv::Imm::Short(wk.LinkageType, wk.Export)])
                                .collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(target_id).collect(),
                    });
                }
                ExportKey::SpvEntryPoint { imms, interface_global_vars } => {
                    entry_point_insts.push(spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpEntryPoint,
                            imms: imms.iter().copied().collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(target_id)
                            .chain(
                                interface_global_vars
                                    .iter()
                                    .map(|&gv| ids.globals[&global_var_to_id_giving_global(gv)]),
                            )
                            .collect(),
                    });
                }
            }
        }

        // FIXME(eddyb) maybe make a helper for `push_inst` with an iterator?
        for entry_point_inst in entry_point_insts {
            emitter.push_inst(&entry_point_inst)?;
        }
        for execution_mode_inst in execution_mode_insts {
            emitter.push_inst(&execution_mode_inst)?;
        }

        for (&s, &id) in &ids.debug_strings {
            emitter.push_inst(&spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpString,
                    imms: spv::encode_literal_string(s).collect(),
                },
                result_type_id: None,
                result_id: Some(id),
                ids: [].into_iter().collect(),
            })?;
        }
        for (lang, sources) in &debug_info.source_languages {
            let lang_imms = || {
                [
                    spv::Imm::Short(wk.SourceLanguage, lang.lang),
                    spv::Imm::Short(wk.LiteralInteger, lang.version),
                ]
                .into_iter()
            };
            if sources.file_contents.is_empty() {
                emitter.push_inst(&spv::InstWithIds {
                    without_ids: spv::Inst { opcode: wk.OpSource, imms: lang_imms().collect() },
                    result_type_id: None,
                    result_id: None,
                    ids: [].into_iter().collect(),
                })?;
            } else {
                for (&file, contents) in &sources.file_contents {
                    // The maximum word count is `2**16 - 1`, the first word is
                    // taken up by the opcode & word count, and one extra byte is
                    // taken up by the nil byte at the end of the LiteralString.
                    const MAX_OP_SOURCE_CONT_CONTENTS_LEN: usize = (0xffff - 1) * 4 - 1;

                    // `OpSource` has 3 more operands than `OpSourceContinued`,
                    // and each of them take up exactly one word.
                    const MAX_OP_SOURCE_CONTENTS_LEN: usize =
                        MAX_OP_SOURCE_CONT_CONTENTS_LEN - 3 * 4;

                    let (contents_initial, mut contents_rest) =
                        contents.split_at(contents.len().min(MAX_OP_SOURCE_CONTENTS_LEN));

                    emitter.push_inst(&spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpSource,
                            imms: lang_imms()
                                .chain(spv::encode_literal_string(contents_initial))
                                .collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(ids.debug_strings[&cx[file]]).collect(),
                    })?;

                    while !contents_rest.is_empty() {
                        // FIXME(eddyb) test with UTF-8! this `split_at` should
                        // actually take *less* than the full possible size, to
                        // avoid cutting a UTF-8 sequence.
                        let (cont_chunk, rest) = contents_rest
                            .split_at(contents_rest.len().min(MAX_OP_SOURCE_CONT_CONTENTS_LEN));
                        contents_rest = rest;

                        emitter.push_inst(&spv::InstWithIds {
                            without_ids: spv::Inst {
                                opcode: wk.OpSourceContinued,
                                imms: spv::encode_literal_string(cont_chunk).collect(),
                            },
                            result_type_id: None,
                            result_id: None,
                            ids: [].into_iter().collect(),
                        })?;
                    }
                }
            }
        }
        for ext_inst in debug_info.source_extension_insts() {
            emitter.push_inst(&ext_inst)?;
        }
        for debug_name_inst in debug_name_insts {
            emitter.push_inst(&debug_name_inst)?;
        }
        for mod_proc_inst in debug_info.module_processed_insts() {
            emitter.push_inst(&mod_proc_inst)?;
        }

        for decoration_inst in decoration_insts {
            emitter.push_inst(&decoration_inst)?;
        }

        let mut current_debug_line = None;
        let mut current_block_id = None; // HACK(eddyb) for `current_debug_line` resets.
        for lazy_inst in global_and_func_insts {
            let mut result: Result<(), _> = Ok(());
            lazy_inst.for_each_spv_inst_with_debug_line(self, ids, |inst, new_debug_line| {
                if result.is_err() {
                    return;
                }

                // Reset line debuginfo when crossing/leaving blocks.
                let new_block_id = if inst.opcode == wk.OpLabel {
                    Some(inst.result_id.unwrap())
                } else if inst.opcode == wk.OpFunctionEnd {
                    None
                } else {
                    current_block_id
                };
                if current_block_id != new_block_id {
                    current_debug_line = None;
                }
                current_block_id = new_block_id;

                // Determine whether to emit `OpLine`/`OpNoLine` before `inst`,
                // in order to end up with the expected line debuginfo.
                if current_debug_line != new_debug_line {
                    let (opcode, imms, ids) = match new_debug_line {
                        Some(SpvDebugLine { file_path_id, line, col }) => (
                            wk.OpLine,
                            [
                                spv::Imm::Short(wk.LiteralInteger, line),
                                spv::Imm::Short(wk.LiteralInteger, col),
                            ]
                            .into_iter()
                            .collect(),
                            iter::once(file_path_id).collect(),
                        ),
                        None => (wk.OpNoLine, [].into_iter().collect(), [].into_iter().collect()),
                    };
                    result = emitter.push_inst(&spv::InstWithIds {
                        without_ids: spv::Inst { opcode, imms },
                        result_type_id: None,
                        result_id: None,
                        ids,
                    });
                    if result.is_err() {
                        return;
                    }
                }
                current_debug_line = new_debug_line;

                result = emitter.push_inst(&inst);
            });
            result?;
        }

        Ok(emitter)
    }
}
