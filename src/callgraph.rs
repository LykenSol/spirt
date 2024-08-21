//! Function call graph abstractions and utilities.

use crate::func_at::{FuncAt, FuncAtMut};
use crate::qptr::{self, QPtrOp};
use crate::transform::{InnerInPlaceTransform, Transformed, Transformer};
use crate::visit::{InnerVisit as _, Visitor};
use crate::{
    AddrSpace, AttrSet, Const, ConstDef, ConstKind, Context, ControlNode, ControlNodeDef,
    ControlNodeKind, ControlNodeOutputDecl, ControlRegion, ControlRegionDef,
    ControlRegionInputDecl, DataInst, DataInstDef, DataInstForm, DataInstFormDef, DataInstKind,
    DeclDef, Diag, EntityList, EntityOrientedDenseMap, Exportee, Func, FuncDefBody, FuncParam,
    FxIndexMap, FxIndexSet, GlobalVar, GlobalVarDecl, GlobalVarDefBody, GlobalVarInit, Module,
    SelectionKind, Type, TypeKind, Value, scalar,
};
use itertools::Either;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::hash::Hash;
use std::mem;
use std::num::{NonZeroI32, NonZeroU32};
use std::ops::Range;
use std::rc::Rc;

pub struct CallGraph {
    pub caller_to_callees: FxIndexMap<Func, Callees>,

    // FIXME(eddyb) currently not supported (but entirely doable).
    pub indirect_callees: FxIndexSet<Func>,
}

#[derive(Default)]
pub struct Callees {
    // FIXME(eddyb) this wants to be a multimap, realistically.
    pub direct: FxIndexMap<Func, SmallVec<[CallSite; 2]>>,

    // FIXME(eddyb) currently not supported (but entirely doable).
    pub indirect: SmallVec<[CallSite; 2]>,
}

// HACK(eddyb) this only exists to allow some transforms to be in-place.
#[derive(Copy, Clone)]
pub struct CallSite {
    pub func_call_node: ControlNode,
    pub parent_region: ControlRegion,
}

impl CallGraph {
    pub fn compute(module: &Module) -> Self {
        let mut collector = CallGraphCollector {
            cx: module.cx_ref(),
            module,

            call_graph: Self {
                caller_to_callees: FxIndexMap::default(),
                indirect_callees: FxIndexSet::default(),
            },
            caller: Err("Module"),

            seen_attrs: FxHashSet::default(),
            seen_types: FxHashSet::default(),
            seen_consts: FxHashSet::default(),
            seen_data_inst_forms: FxHashSet::default(),
            seen_global_vars: FxHashSet::default(),
        };

        // FIXME(eddyb) use a queue here to avoid actual recursive visiting.
        // HACK(eddyb) inlined (and customized) `module.inner_visit_with(...)`
        // due to a lack of an overridable `visit_exportee`.
        collector.visit_module_dialect(&module.dialect);
        collector.visit_module_debug_info(&module.debug_info);
        for (export_key, exportee) in &module.exports {
            export_key.inner_visit_with(&mut collector);
            match *exportee {
                Exportee::GlobalVar(gv) => collector.visit_global_var_use(gv),
                Exportee::Func(func) => collector.visit_func_used_by_export_or_callee(func),
            }
        }

        collector.call_graph
    }

    pub fn direct_and_indirect_callees_of(&self, caller: Func) -> impl Iterator<Item = Func> + '_ {
        self.caller_to_callees
            .get(&caller)
            .map(|callees| {
                callees.direct.keys().chain(
                    (!callees.indirect.is_empty())
                        .then_some(&self.indirect_callees)
                        .into_iter()
                        .flatten(),
                )
            })
            .into_iter()
            .flatten()
            .copied()
    }
}

struct CallGraphCollector<'a> {
    cx: &'a Context,
    module: &'a Module,

    call_graph: CallGraph,
    caller: Result<Func, &'static str>,

    // FIXME(eddyb) build some automation to avoid ever repeating these.
    seen_attrs: FxHashSet<AttrSet>,
    seen_types: FxHashSet<Type>,
    seen_consts: FxHashSet<Const>,
    seen_data_inst_forms: FxHashSet<DataInstForm>,
    seen_global_vars: FxHashSet<GlobalVar>,
}

impl CallGraphCollector<'_> {
    fn with_caller<R>(
        &mut self,
        inner_caller: Result<Func, &'static str>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let outer_caller = mem::replace(&mut self.caller, inner_caller);
        let r = f(self);
        self.caller = outer_caller;
        r
    }

    // HACK(eddyb) separate from `visit_func_use` because it's not easy to make
    // `visit_func_use` perfectly know the source of the user.
    fn visit_func_used_by_export_or_callee(&mut self, func: Func) {
        use indexmap::map::Entry;

        if let Entry::Vacant(entry) = self.call_graph.caller_to_callees.entry(func) {
            entry.insert(Default::default());
            self.with_caller(Ok(func), |this| {
                this.visit_func_decl(&this.module.funcs[func]);
            });
        }
    }
}

impl Visitor<'_> for CallGraphCollector<'_> {
    // FIXME(eddyb) build some automation to avoid ever repeating these.
    fn visit_attr_set_use(&mut self, attrs: AttrSet) {
        if self.seen_attrs.insert(attrs) {
            self.with_caller(Err("AttrSet"), |this| {
                this.visit_attr_set_def(&self.cx[attrs]);
            });
        }
    }
    fn visit_type_use(&mut self, ty: Type) {
        if self.seen_types.insert(ty) {
            self.with_caller(Err("Type"), |this| {
                this.visit_type_def(&self.cx[ty]);
            });
        }
    }
    fn visit_const_use(&mut self, ct: Const) {
        if self.seen_consts.insert(ct) {
            let ct_def = &self.cx[ct];
            match ct_def.kind {
                // FIXME(eddyb) special-case function pointer constants once supported.
                ConstKind::Undef
                | ConstKind::Scalar(_)
                | ConstKind::Vector(_)
                | ConstKind::PtrToGlobalVar(_)
                | ConstKind::SpvInst { .. }
                | ConstKind::SpvStringLiteralForExtInst(_) => {
                    self.with_caller(Err("Const"), |this| {
                        this.visit_const_def(ct_def);
                    });
                }
            }
        }
    }
    fn visit_data_inst_form_use(&mut self, data_inst_form: DataInstForm) {
        if self.seen_data_inst_forms.insert(data_inst_form) {
            self.with_caller(Err("DataInstForm"), |this| {
                this.visit_data_inst_form_def(&self.cx[data_inst_form]);
            });
        }
    }

    fn visit_global_var_use(&mut self, gv: GlobalVar) {
        if self.seen_global_vars.insert(gv) {
            self.with_caller(Err("GlobalVar"), |this| {
                this.visit_global_var_decl(&this.module.global_vars[gv]);
            });
        }
    }
    fn visit_func_use(&mut self, _func: Func) {
        unreachable!(
            "Func used directly in {} definition",
            self.caller.map_or_else(|k| k, |_| "Func")
        );
    }

    // HACK(eddyb) needing `parent_region` is the only reason this visitor
    // handles whole `ControlRegion`s instead of just their `ControlNode`s.
    fn visit_control_region_def(&mut self, func_at_control_region: FuncAt<'_, ControlRegion>) {
        let ControlRegionDef { inputs, children, outputs } = func_at_control_region.def();

        for input in inputs {
            input.inner_visit_with(self);
        }
        for func_at_control_node in func_at_control_region.at(*children) {
            let ControlNodeDef { attrs, kind, outputs } = func_at_control_node.def();
            if let (Ok(caller), ControlNodeKind::FuncCall { callee, inputs }) = (self.caller, kind)
            {
                self.visit_attr_set_use(*attrs);

                // HACK(eddyb) bypass `visit_func_use` entirely for static calls.
                let callees = self.call_graph.caller_to_callees.entry(caller).or_default();
                callees.direct.entry(*callee).or_default().push(CallSite {
                    func_call_node: func_at_control_node.position,
                    parent_region: func_at_control_region.position,
                });
                self.visit_func_used_by_export_or_callee(*callee);

                for v in inputs {
                    self.visit_value_use(v);
                }
                for output in outputs {
                    output.inner_visit_with(self);
                }
            } else {
                func_at_control_node.inner_visit_with(self);
            }
        }
        for v in outputs {
            self.visit_value_use(v);
        }
    }
    fn visit_control_node_def(&mut self, _: FuncAt<'_, ControlNode>) {
        unreachable!()
    }
}

// FIXME(eddyb) deduplicate with `cfg::LoopFinder` (almost identical Tarjan SCC
// algorithm, except for `cfg::LoopFinder` having a few loop-specific quirks).
struct CycleFinder<G, N> {
    graph: G,

    /// SCC accumulation stack, where graph nodes collect during the depth-first
    /// traversal, and are only popped when their "SCC root" (cycle entry) is
    /// (note that multiple SCCs on the stack does *not* indicate SCC nesting,
    /// but rather a path between two SCCs, i.e. a cycle *following* another).
    scc_stack: Vec<N>,
    /// Per-graph-node traversal state (often just pointing to a `scc_stack` slot).
    //
    // HACK(eddyb) this also holds (in `SccState::Complete`) the result itself.
    scc_state: FxIndexMap<N, SccState<N>>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SccStackIdx(u32);

#[derive(PartialEq, Eq)]
enum SccState<N> {
    /// Graph node has been reached and ended up somewhere on the `scc_stack`,
    /// where it will remain until the SCC it's part of will be completed.
    Pending(SccStackIdx),

    /// Graph node had been reached once, but is no longer on the `scc_stack`, its
    /// parent SCC having been completed (or it wasn't in an SCC to begin with).
    Complete { parent_scc_root: Option<N> },
}

impl<G, N> CycleFinder<G, N> {
    fn new(graph: G) -> Self {
        Self { graph, scc_stack: vec![], scc_state: FxIndexMap::default() }
    }
}

impl<N: Copy + Eq + Hash, G: Fn(N) -> E, E: Iterator<Item = N>> CycleFinder<G, N> {
    /// Tarjan's SCC algorithm works by computing the "earliest" reachable node,
    /// from every node (often using the name `lowlink`), which will be equal
    /// to the origin node itself iff that node is an "SCC root" (cycle entry),
    /// and always point to an "earlier" node if a cycle is being reached from
    /// somewhere else in the SCC.
    ///
    /// Here we track stack indices (as the stack order is the traversal order),
    /// and distinguish the acyclic case to avoid treating most nodes as self-cycles.
    fn find_earliest_scc_root_of(&mut self, node: N) -> Option<SccStackIdx> {
        use indexmap::map::Entry;

        let state_entry = match self.scc_state.entry(node) {
            Entry::Vacant(entry) => entry,
            Entry::Occupied(entry) => {
                return match *entry.get() {
                    SccState::Pending(scc_stack_idx) => Some(scc_stack_idx),
                    SccState::Complete { .. } => None,
                };
            }
        };
        let scc_stack_idx = SccStackIdx(self.scc_stack.len().try_into().unwrap());
        self.scc_stack.push(node);
        state_entry.insert(SccState::Pending(scc_stack_idx));

        let earliest_scc_root =
            (self.graph)(node).filter_map(|target| self.find_earliest_scc_root_of(target)).min();

        // If this node has been chosen as the root of an SCC, complete that SCC.
        if earliest_scc_root == Some(scc_stack_idx) {
            let scc_start = scc_stack_idx.0 as usize;

            // NOTE(eddyb) this is much simpler than `cfg::LoopFinder`, because
            // there's no need for nested cycles, nor an "exit edge" concept.
            for scc_node in self.scc_stack.drain(scc_start..) {
                *self.scc_state.get_mut(&scc_node).unwrap() =
                    SccState::Complete { parent_scc_root: Some(node) };
            }

            return None;
        }

        // Not actually in an SCC at all, just some node outside any graph cycles.
        if earliest_scc_root.is_none() {
            assert!(self.scc_stack.pop() == Some(node));
            *self.scc_state.get_mut(&node).unwrap() = SccState::Complete { parent_scc_root: None };
        }

        earliest_scc_root
    }
}

// FIXME(eddyb) use proper newtypes for byte amounts.
pub struct CallStackEmuConfig {
    pub layout_config: qptr::LayoutConfig,

    /// The size (in bytes) of call stack elements (that pushes and pops are
    /// rounded up to a multiple of), *and* the largest supported alignment.
    //
    // FIXME(eddyb) remove the need for this by efficiently supporting byte alignment.
    pub stack_unit_bytes: NonZeroU32,

    /// The fixed size (in bytes) of the per-invocation emulated call stack
    /// (must be a multiple of `stack_unit_bytes`).
    ///
    /// Running out of this stack space during recursion is treated as a safe
    /// and deterministic "stack overflow" fatal error (see `build_fatal_error`).
    //
    // FIXME(eddyb) replace this with segmented stacks (see `stacker` crate),
    // replacing "stack overflow" with a global allocator's "out of memory",
    // but ideally at far higher recursive depths (and scaling with "heap" size).
    pub stack_size_bytes: u32,

    /// `CallStackEmulator` will call `build_fatal_error(msg, cx, func_at_region)`
    /// to append a fatal error with some message `msg` (e.g. "stack overflow")
    /// at the end of a `ControlRegion`, expecting that:
    /// - an error *may* be reported (e.g. passing `msg` to some "debug printf")
    /// - control-flow *must* diverge (in `ControlNode`s added to the region),
    ///   and never exit the region normally (into the surrounding function)
    ///   - failure to respect this *will not* directly cause UB, but rather
    ///     infinite looping, which may be treated as UB downstream of SPIR-T,
    ///     but even non-UB infinite looping causing GPU timeouts should be
    ///     avoided, as not all user configurations (OS/drivers/hardware/etc.)
    ///     are robust (enough) wrt hangs and may degrade the rest of the system
    //
    // FIXME(eddyb) consider using an `enum` for the messages?
    pub build_fatal_error: Box<dyn Fn(&str, &Context, FuncAtMut<'_, ControlRegion>)>,
}

/// Potentially-recursive calls require a call stack (as they are equivalent to
/// pushing a "return continuation" onto such a stack, i.e. a code pointer that
/// can then be tail-called to return from the callee).
//
// FIXME(eddyb) consider moving everything below elsewhere (e.g. `emu::callstack`).
pub struct CallStackEmulator<'a> {
    // FIXME(eddyb) does this name make sense? should these two structs be merged?
    global_stack: EmuGlobalStack<'a>,

    call_graph: CallGraph,

    // FIXME(eddyb) is this as necessary given `call_emu_groups`?
    func_emu_summary: EntityOrientedDenseMap<Func, FuncEmuSummary>,
    call_emu_groups: FxIndexMap<CallEmuGroup, FxIndexSet<Func>>,

    next_state_idx: Cell<EmuStateIdx>,
}

/// Potentially-recursive calls (including indirect calls that static analysis
/// cannot prove don't dynamically result in recursion) require emulating a
/// call stack (see also `CallStackEmulator`), and to that end, their callee
/// `Func`s are grouped into independent mutually-recursive cycles (SCCs),
/// each identified by their "SCC root" (i.e. the first `Func` the Tarjan SCC
/// algorithm saw in each group), which has no semantic significance, and only
/// helps in distinguishing *between* such groups.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct CallEmuGroup {
    scc_root: Func,
}

#[derive(Default)]
struct FuncEmuSummary {
    emu_group: Option<CallEmuGroup>,
    called_from_outside_emu_group: bool,

    /// Contains calls to functions belonging to a different `emu_group`
    /// (which may be `None`, i.e. an outermost caller into emulated callees).
    calls_outside_emu_group: bool,
}

impl<'a> CallStackEmulator<'a> {
    pub fn new(module: &mut Module, config: &'a CallStackEmuConfig) -> Self {
        let global_stack = EmuGlobalStack::new(module, config);

        let call_graph = CallGraph::compute(module);

        let mut cycle_finder =
            CycleFinder::new(|func| call_graph.direct_and_indirect_callees_of(func));
        for exportee in module.exports.values() {
            if let &Exportee::Func(func) = exportee {
                cycle_finder.find_earliest_scc_root_of(func);
            }
        }

        let mut func_emu_summary = EntityOrientedDenseMap::new();
        let mut call_emu_groups = FxIndexMap::<_, FxIndexSet<_>>::default();

        // HACK(eddyb) make it simpler to look up any known `Func`.
        for &func in call_graph.caller_to_callees.keys() {
            func_emu_summary.insert(func, FuncEmuSummary::default());
        }

        for (func, state) in cycle_finder.scc_state {
            if let SccState::Complete { parent_scc_root: Some(scc_root) } = state {
                let emu_group = CallEmuGroup { scc_root };
                func_emu_summary[func].emu_group = Some(emu_group);
                call_emu_groups.entry(emu_group).or_default().insert(func);
            }
        }

        // FIXME(eddyb) this is probably less efficient than it could be.
        for &caller in call_graph.caller_to_callees.keys() {
            let caller_emu_group = func_emu_summary[caller].emu_group;
            let mut any_callees_outside_emu_group = false;
            for callee in call_graph.direct_and_indirect_callees_of(caller) {
                let callee_emu_summary = &mut func_emu_summary[callee];
                if callee_emu_summary.emu_group != caller_emu_group {
                    callee_emu_summary.called_from_outside_emu_group = true;
                    any_callees_outside_emu_group = true;
                }
            }
            if any_callees_outside_emu_group {
                func_emu_summary[caller].calls_outside_emu_group = true;
            }
        }

        Self {
            global_stack,

            call_graph,
            func_emu_summary,
            call_emu_groups,

            next_state_idx: Cell::new(EmuStateIdx(NonZeroI32::new(1).unwrap())),
        }
    }

    // NOTE(eddyb) `module` passed outside `self` to avoid borrow issues.
    pub fn transform_module(self, module: &mut Module) {
        for (&emu_group, funcs) in &self.call_emu_groups {
            self.transform_all_funcs_in_emu_group(module, emu_group, funcs.iter().copied());
        }
    }

    fn transform_all_funcs_in_emu_group(
        &self,
        module: &mut Module,
        emu_group: CallEmuGroup,
        funcs_in_group: impl Iterator<Item = Func>,
    ) {
        // FIXME(eddyb) support indirect calls in this as well!
        assert!(self.call_graph.indirect_callees.is_empty());

        // HACK(eddyb) this shouldn't matter, except there's assumptions made
        // elsewhere that `emu_group.scc_root` is the first function in a chain
        // of "next state" handlers (which includes all functions in the group).
        let funcs_in_group = [emu_group.scc_root]
            .into_iter()
            .chain(funcs_in_group.filter(|&func| func != emu_group.scc_root));

        let cx = &self.global_stack.cx;
        let per_func_states: FxIndexMap<_, _> = funcs_in_group
            .map(|func| {
                let func_def_body = match &module.funcs[func].def {
                    DeclDef::Present(func_def_body) => func_def_body,
                    DeclDef::Imported(_) => unreachable!(),
                };

                // FIXME(eddyb) move this check elsewhere.
                assert!(func_def_body.unstructured_cfg.is_none());

                let mut state_reserver = EmuStateReserver {
                    func_emu_summary: &self.func_emu_summary,
                    emu_group,

                    next_state_idx: self.next_state_idx.get(),

                    states: Default::default(),
                    body_stack: [func_def_body.body].into_iter().collect(),
                };
                func_def_body.inner_visit_with(&mut state_reserver);

                self.next_state_idx.set(state_reserver.next_state_idx);

                (func, state_reserver.states)
            })
            .collect();

        // FIXME(eddyb) does this need to be a separate map?
        let func_call_emu_cont: FxIndexMap<_, _> = per_func_states
            .iter()
            .map(|(&func, func_states)| {
                let func_decl = &module.funcs[func];
                let body_region = match &func_decl.def {
                    DeclDef::Present(func_def_body) => func_def_body.body,
                    DeclDef::Imported(_) => unreachable!(),
                };
                (func, EmuContClosure {
                    origin: Ok(Either::Left((body_region, func_states.for_region[body_region]))),
                    input_count: func_decl.params.len(),
                    captures: FxIndexSet::default(),
                })
            })
            .collect();

        // After entry states for all functions in the `emu_group` are reserved,
        // all calls from anywhere outside `emu_group` can be replaced with the
        // state machine loop (which advances states until the final return).
        //
        // FIXME(eddyb) this is probably an inefficient scan even with the
        // pre-collection of `CallSite`s.
        for (&caller, callees) in &self.call_graph.caller_to_callees {
            if self.func_emu_summary[caller].emu_group == Some(emu_group) {
                continue;
            }

            for (&callee, call_sites) in &callees.direct {
                if let Some(call_emu_cont) = func_call_emu_cont.get(&callee) {
                    let caller_func_def_body = match &mut module.funcs[caller].def {
                        DeclDef::Present(func_def_body) => func_def_body,
                        DeclDef::Imported(_) => unreachable!(),
                    };
                    for &call_site in call_sites {
                        self.transform_inter_emu_group_func_call(
                            caller_func_def_body,
                            call_site,
                            call_emu_cont,
                            emu_group,
                        );
                    }
                }
            }
        }

        // Turn each function in the group into a "next state" handler,
        // i.e. an `EmuStateIdx -> EmuStateIdx` (concretely, `s32 -> s32`)
        // single step `switch`, with all inter-state dataflow going solely
        // through the global stack.
        let state_ty = cx.intern(EmuStateIdx::TYPE);
        for (func_idx, (&func, func_states)) in per_func_states.iter().enumerate() {
            let func_decl = &mut module.funcs[func];
            let func_def_body = match &mut func_decl.def {
                DeclDef::Present(func_def_body) => func_def_body,
                DeclDef::Imported(_) => unreachable!(),
            };

            let orig_params = mem::replace(
                &mut func_decl.params,
                [FuncParam { attrs: AttrSet::default(), ty: state_ty }].into_iter().collect(),
            );
            let orig_ret_types =
                mem::replace(&mut func_decl.ret_types, [state_ty].into_iter().collect());

            let new_body = func_def_body.control_regions.define(cx, ControlRegionDef::default());
            let orig_body = mem::replace(&mut func_def_body.body, new_body);

            let mut func = func_def_body.at_mut(());
            let ret_cont = {
                let mut popper = self.global_stack.popper(func.reborrow());
                let popped_state = popper.pop(func.reborrow(), state_ty);
                let pops_block = popper.finish(func.reborrow());
                func.control_regions[orig_body]
                    .children
                    .insert_last(pops_block, func.control_nodes);
                EmuContClosure {
                    origin: Err(popped_state),
                    input_count: orig_ret_types.len(),
                    captures: FxIndexSet::default(),
                }
            };

            let state_switch_cases = {
                let mut state_splitter = EmuStateSplitter {
                    global_stack: &self.global_stack,

                    states: func_states,
                    func_call_emu_cont: &func_call_emu_cont,

                    state_switch_cases: BTreeMap::new(),
                };
                state_splitter
                    .maybe_split_states_in_region(func.reborrow().at(orig_body), &ret_cont);
                state_splitter.state_switch_cases
            };

            let current_state = {
                func.reborrow()
                    .at(new_body)
                    .def()
                    .inputs
                    .push(ControlRegionInputDecl { attrs: AttrSet::default(), ty: state_ty });
                Value::ControlRegionInput { region: new_body, input_idx: 0 }
            };

            // HACK(eddyb) this is how the whole group gets chained together.
            let default_case = {
                let mut children = EntityList::empty();
                let next_state_after =
                    match per_func_states.get_index(func_idx + 1).map(|(&f, _)| f) {
                        Some(next_func_in_group) => {
                            // FIXME(eddyb) DRY vs `transform_inter_emu_group_func_call`?
                            let call_node = func.control_nodes.define(
                                cx,
                                ControlNodeDef {
                                    attrs: AttrSet::default(),
                                    kind: ControlNodeKind::FuncCall {
                                        callee: next_func_in_group,
                                        inputs: [current_state].into_iter().collect(),
                                    },
                                    outputs: [ControlNodeOutputDecl {
                                        attrs: AttrSet::default(),
                                        ty: state_ty,
                                    }]
                                    .into_iter()
                                    .collect(),
                                }
                                .into(),
                            );
                            children.insert_last(call_node, func.control_nodes);
                            Value::ControlNodeOutput { control_node: call_node, output_idx: 0 }
                        }
                        None => EmuStateIdx::UNKNOWN_STATE.to_value(cx),
                    };
                func.control_regions
                    .define(cx, EmuContBody { children, next_state_after }.into_region_def())
            };

            let state_switch_node = {
                let (case_consts, mut cases): (_, SmallVec<_>) = state_switch_cases
                    .into_iter()
                    .map(|(case_const, case)| (case_const.as_scalar(), case))
                    .unzip();
                cases.push(default_case);
                func.control_nodes.define(
                    cx,
                    ControlNodeDef {
                        attrs: AttrSet::default(),
                        kind: ControlNodeKind::Select {
                            kind: SelectionKind::Switch { case_consts },
                            scrutinee: current_state,
                            cases,
                        },

                        outputs: [ControlNodeOutputDecl {
                            attrs: AttrSet::default(),
                            ty: state_ty,
                        }]
                        .into_iter()
                        .collect(),
                    }
                    .into(),
                )
            };
            let new_body_def = &mut func.control_regions[new_body];
            new_body_def.children.insert_last(state_switch_node, func.control_nodes);
            new_body_def
                .outputs
                .push(Value::ControlNodeOutput { control_node: state_switch_node, output_idx: 0 });
        }
    }

    fn transform_inter_emu_group_func_call(
        &self,
        func_def_body: &mut FuncDefBody,
        call_site: CallSite,
        call_emu_cont: &EmuContClosure,
        callee_emu_group: CallEmuGroup,
    ) {
        let cx = &self.global_stack.cx;
        let mut func = func_def_body.at_mut(());

        // HACK(eddyb) allocate an unique `EmuStateIdx` for each callsite.
        let call_site_ret_cont_state = {
            let state = self.next_state_idx.get();
            self.next_state_idx.set(state.checked_next().unwrap());
            state
        };

        let call_args = match &mut func.reborrow().at(call_site.func_call_node).def().kind {
            ControlNodeKind::FuncCall { callee: _, inputs } => mem::take(inputs),
            _ => unreachable!(),
        };

        // FIXME(eddyb) is `call_emu_cont` unnecessary?
        assert!(call_emu_cont.captures.is_empty());

        let (args_pushes_node, args_pushes_node2, initial_state) = {
            let mut pusher = self.global_stack.pusher(func.reborrow());
            pusher.push(func.reborrow(), call_site_ret_cont_state.to_value(cx));
            for v in call_args.into_iter().rev() {
                pusher.push(func.reborrow(), v);
            }
            pusher.finish_for_state(func.reborrow(), call_emu_cont.entry_state_value(cx))
        };
        func.control_regions[call_site.parent_region].children.insert_before(
            args_pushes_node,
            call_site.func_call_node,
            func.control_nodes,
        );
        if let Some(args_pushes_node2) = args_pushes_node2 {
            func.control_regions[call_site.parent_region].children.insert_before(
                args_pushes_node2,
                call_site.func_call_node,
                func.control_nodes,
            );
        }

        let call_node_outputs = {
            let count = func.reborrow().at(call_site.func_call_node).def().outputs.len();
            (0..u32::try_from(count).unwrap()).map(|output_idx| Value::ControlNodeOutput {
                control_node: call_site.func_call_node,
                output_idx,
            })
        };
        let (ret_vals_pops_node, ret_vals) = {
            let mut popper = self.global_stack.popper(func.reborrow());
            let ret_vals: SmallVec<_> = call_node_outputs
                .clone()
                .map(|call_output| {
                    let ty = func.reborrow().freeze().at(call_output).type_of(cx);
                    popper.pop(func.reborrow(), ty)
                })
                .collect();
            (popper.finish(func.reborrow()), ret_vals)
        };

        let state_ty = cx.intern(EmuStateIdx::TYPE);
        let state_machine_loop_body = func.control_regions.define(cx, ControlRegionDef::default());
        let current_state = {
            func.reborrow()
                .at(state_machine_loop_body)
                .def()
                .inputs
                .push(ControlRegionInputDecl { attrs: AttrSet::default(), ty: state_ty });
            Value::ControlRegionInput { region: state_machine_loop_body, input_idx: 0 }
        };
        let next_state = {
            let call_node = func.control_nodes.define(
                cx,
                ControlNodeDef {
                    attrs: AttrSet::default(),
                    kind: ControlNodeKind::FuncCall {
                        callee: callee_emu_group.scc_root,
                        inputs: [current_state].into_iter().collect(),
                    },
                    outputs: [ControlNodeOutputDecl { attrs: AttrSet::default(), ty: state_ty }]
                        .into_iter()
                        .collect(),
                }
                .into(),
            );
            func.control_regions[state_machine_loop_body]
                .children
                .insert_last(call_node, func.control_nodes);
            Value::ControlNodeOutput { control_node: call_node, output_idx: 0 }
        };
        func.control_regions[state_machine_loop_body].outputs.push(next_state);

        // NOTE(eddyb) the `switch` default case is where all intermediary states
        // will end up (i.e. the state machine loop keeps going), with only the
        // successful return, and error states, being matched for explicitly.
        let non_default_next_state_switch_cases = [
            (call_site_ret_cont_state, Ok((ret_vals_pops_node, ret_vals))),
            (EmuStateIdx::STACK_OVERFLOW, Err("stack overflow")),
            (EmuStateIdx::UNKNOWN_STATE, Err("unknown state")),
        ];
        let next_state_switch_case_consts =
            non_default_next_state_switch_cases.iter().map(|(s, _)| s.as_scalar()).collect();
        let next_state_switch_cases = non_default_next_state_switch_cases
            .into_iter()
            .map(Some)
            .chain([None])
            .map(|maybe_non_default_case| {
                let case_region = func.control_regions.define(cx, ControlRegionDef::default());
                let break_with_outputs = match maybe_non_default_case {
                    Some((_, Ok((ret_vals_pops_node, ret_vals)))) => {
                        func.control_regions[case_region]
                            .children
                            .insert_last(ret_vals_pops_node, func.control_nodes);
                        Some(ret_vals)
                    }
                    Some((_, Err(msg))) => {
                        (self.global_stack.config.build_fatal_error)(
                            msg,
                            cx,
                            func.reborrow().at(case_region),
                        );
                        None
                    }
                    None => None,
                };
                let loop_repeat_cond =
                    Value::Const(cx.intern(scalar::Const::from_bool(break_with_outputs.is_none())));

                let mut outputs = break_with_outputs.unwrap_or_else(|| {
                    // FIXME(eddyb) collect these undefs only once.
                    call_node_outputs
                        .clone()
                        .map(|call_output| {
                            let ty = func.reborrow().freeze().at(call_output).type_of(cx);
                            Value::Const(cx.intern(ConstDef {
                                attrs: AttrSet::default(),
                                ty,
                                kind: ConstKind::Undef,
                            }))
                        })
                        .collect()
                });
                outputs.push(loop_repeat_cond);
                func.reborrow().at(case_region).def().outputs = outputs;

                case_region
            })
            .collect();

        // HACK(eddyb) because loop nodes don't output values yet, and to avoid
        // having to replace uses of the original call node in the caller, the
        // original call node becomes the `switch` over `next_state`, and outputs
        // both call return values, *and* the loop exit decision - additionally,
        // this requires some region membership juggling of the `switch` & `loop`.
        let next_state_switch = call_site.func_call_node;
        let state_machine_loop_repeat_cond = {
            let next_state_switch_def = func.reborrow().at(next_state_switch).def();
            next_state_switch_def.kind = ControlNodeKind::Select {
                kind: SelectionKind::Switch { case_consts: next_state_switch_case_consts },
                scrutinee: next_state,
                cases: next_state_switch_cases,
            };

            let output_idx = u32::try_from(next_state_switch_def.outputs.len()).unwrap();
            next_state_switch_def.outputs.push(ControlNodeOutputDecl {
                attrs: AttrSet::default(),
                ty: cx.intern(scalar::Type::Bool),
            });
            Value::ControlNodeOutput { control_node: next_state_switch, output_idx }
        };

        let state_machine_loop_node = func.control_nodes.define(
            cx,
            ControlNodeDef {
                attrs: AttrSet::default(),
                kind: ControlNodeKind::Loop {
                    initial_inputs: [initial_state].into_iter().collect(),
                    body: state_machine_loop_body,
                    repeat_condition: state_machine_loop_repeat_cond,
                },
                outputs: [].into_iter().collect(),
            }
            .into(),
        );

        // HACK(eddyb) this is the juggling mentioned in the above comment, to
        // get the `loop` where the call node (now a `switch`) used to be, and
        // add `switch` node to the body of the `loop`.
        {
            let parent_region_children =
                &mut func.control_regions[call_site.parent_region].children;
            parent_region_children.insert_before(
                state_machine_loop_node,
                call_site.func_call_node,
                func.control_nodes,
            );
            parent_region_children.remove(call_site.func_call_node, func.control_nodes);

            assert!(call_site.func_call_node == next_state_switch);
            func.control_regions[state_machine_loop_body]
                .children
                .insert_last(next_state_switch, func.control_nodes);
        }
    }
}

struct EmuStateReserver<'a> {
    func_emu_summary: &'a EntityOrientedDenseMap<Func, FuncEmuSummary>,

    next_state_idx: EmuStateIdx,

    emu_group: CallEmuGroup,
    states: FuncEmuStates,

    // HACK(eddyb) to keep state reservation in traversal order, function and
    // loop bodies must have their `RegionEmuStates`' `entry_state` reserved
    // before anything nested in them.
    body_stack: SmallVec<[ControlRegion; 4]>,
}

// HACK(eddyb) state indices are signed, so that negative states can be used to
// encode failure modes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct EmuStateIdx(NonZeroI32);

impl EmuStateIdx {
    // HACK(eddyb) changed from `S32` for testing.
    const TYPE: scalar::Type = scalar::Type::U32;

    const STACK_OVERFLOW: Self = Self(match NonZeroI32::new(-1) {
        Some(x) => x,
        None => unreachable!(),
    });
    const UNKNOWN_STATE: Self = Self(match NonZeroI32::new(-2) {
        Some(x) => x,
        None => unreachable!(),
    });

    fn checked_next(self) -> Option<Self> {
        Some(Self(NonZeroI32::new(self.0.get().checked_add(1)?)?))
    }

    fn as_scalar(self) -> scalar::Const {
        scalar::Const::from_bits(Self::TYPE, self.0.get() as u32 as u128)
        // scalar::Const::int_try_from_i128(Self::TYPE, self.0.get().into()).unwrap()
    }

    fn to_value(self, cx: &Context) -> Value {
        Value::Const(cx.intern(self.as_scalar()))
    }
}

#[derive(Default)]
struct FuncEmuStates {
    for_region: EntityOrientedDenseMap<ControlRegion, RegionEmuStates>,
    for_control_node: EntityOrientedDenseMap<ControlNode, ControlNodeEmuStates>,
}

/// Indicates a `ControlRegion` requiring state-splitting (due to its children),
/// and may include additional helper state(s) where necessary.
//
// FIXME(eddyb) better names/organization?
#[derive(Copy, Clone)]
struct RegionEmuStates {
    /// Only used for function bodies (i.e. as the target of a call) and
    /// loop bodies (i.e. as the target of a backedge).
    entry_state: Option<EmuStateIdx>,
}

/// Indicates a `ControlNode` requiring state-splitting (an emulated `FuncCall`,
/// or due to its children), and includes additional helper state(s).
//
// FIXME(eddyb) better names/organization?
#[derive(Copy, Clone)]
struct ControlNodeEmuStates {
    /// The "continuation" (or "exit"), into the parent `ControlRegion`, of this
    /// node, receiving the outputs of this node (e.g. `FuncCall` return values,
    /// or `Select` outputs, etc.).
    merge: EmuStateIdx,
}

impl EmuStateReserver<'_> {
    fn reserve_state(&mut self) -> EmuStateIdx {
        let mut next_state = || {
            let state = self.next_state_idx;
            self.next_state_idx = state.checked_next().unwrap();
            state
        };

        // HACK(eddyb) ensure all surrounding loops have their body entry state
        // reserved first (see comment on `body_stack`).
        for body in self.body_stack.drain(..) {
            assert!(
                self.states
                    .for_region
                    .entry(body)
                    .get_or_insert_with(|| RegionEmuStates { entry_state: Some(next_state()) })
                    .entry_state
                    .is_some()
            );
        }

        next_state()
    }

    fn any_states_reserved_during(&mut self, f: impl FnOnce(&mut Self)) -> bool {
        let orig_next_state_idx = self.next_state_idx;
        f(self);
        orig_next_state_idx != self.next_state_idx
    }
}

impl Visitor<'_> for EmuStateReserver<'_> {
    fn visit_attr_set_use(&mut self, _: AttrSet) {}
    fn visit_type_use(&mut self, _: Type) {}
    fn visit_const_use(&mut self, _: Const) {}
    fn visit_data_inst_form_use(&mut self, _: DataInstForm) {}
    fn visit_global_var_use(&mut self, _: GlobalVar) {}
    fn visit_func_use(&mut self, _: Func) {}

    fn visit_control_region_def(&mut self, func_at_region: FuncAt<'_, ControlRegion>) {
        let region = func_at_region.position;
        let is_body = self.body_stack.last() == Some(&region);
        if self.any_states_reserved_during(|this| func_at_region.inner_visit_with(this)) {
            assert!(self.body_stack.is_empty());
            if is_body {
                assert!(self.states.for_region[region].entry_state.is_some());
            } else {
                assert!(
                    self.states
                        .for_region
                        .insert(region, RegionEmuStates { entry_state: None })
                        .is_none()
                );
            }
        } else {
            if is_body {
                assert!(self.body_stack.pop().unwrap() == region);
            }
            assert!(self.states.for_region.get(region).is_none());
        }
    }
    fn visit_control_node_def(&mut self, func_at_node: FuncAt<'_, ControlNode>) {
        let (loop_body, needs_merge_state) = match func_at_node.def().kind {
            // FIXME(eddyb) include indirect calls in this as well!
            ControlNodeKind::FuncCall { callee, .. } => {
                (None, self.func_emu_summary[callee].emu_group == Some(self.emu_group))
            }
            ControlNodeKind::Loop { body, .. } => (Some(body), false),
            _ => (None, false),
        };

        self.body_stack.extend(loop_body);
        if self.any_states_reserved_during(|this| func_at_node.inner_visit_with(this))
            | needs_merge_state
        {
            let states = ControlNodeEmuStates { merge: self.reserve_state() };
            assert!(self.states.for_control_node.insert(func_at_node.position, states).is_none());
        }
    }
}

struct EmuGlobalStack<'a> {
    cx: Rc<Context>,

    config: &'a CallStackEmuConfig,

    layout_cache: qptr::layout::LayoutCache<'a>,

    // HACK(eddyb) currently always a cached `qptr` type.
    type_of_stack_ptr: Type,

    // HACK(eddyb) currently always a cached `u32` type.
    type_of_stack_top: Type,

    /// Global variable that directly contains the per-invocation emulated stack
    /// contents (likely lifted to an array of `stack_unit_bytes`-sized elements).
    stack_array_global: GlobalVar,
    ptr_to_stack_array_global: Const,

    /// Global variable (of integer type `type_of_stack_top`) holding the offset
    /// (divided by `stack_unit_bytes`) of the most recent push (that is still
    /// "active", i.e. it hasn't been popped already).
    ///
    /// Initialized with `stack_array_global`'s size (divided by `stack_unit_bytes`),
    /// decreasing with pushes, and increasing with pops (back to that maximum
    /// initial value only when the stack is fully empty again).
    stack_top_global: GlobalVar,
    ptr_to_stack_top_global: Const,
}

impl<'a> EmuGlobalStack<'a> {
    fn new(module: &mut Module, config: &'a CallStackEmuConfig) -> Self {
        let cx = module.cx();

        let qptr_ty = cx.intern(TypeKind::QPtr);

        // FIXME(eddyb) make such addrspaces first-class in SPIR-T.
        let invocation_local_addr_space =
            AddrSpace::SpvStorageClass(crate::spv::spec::Spec::get().well_known.Private);
        let mut invocation_local_global_var_and_ptr_to = |size, initializer| {
            let align = config.stack_unit_bytes.get();
            assert_eq!(size % align, 0);
            let global_var =
                module.global_vars.define(&cx, GlobalVarDecl {
                    attrs: AttrSet::default(),
                    type_of_ptr_to: qptr_ty,
                    shape: Some(qptr::shapes::GlobalVarShape::UntypedData(
                        qptr::shapes::MemLayout { align, legacy_align: align, size },
                    )),
                    addr_space: invocation_local_addr_space,
                    def: DeclDef::Present(GlobalVarDefBody { initializer }),
                });
            (
                global_var,
                cx.intern(ConstDef {
                    attrs: AttrSet::default(),
                    ty: qptr_ty,
                    kind: ConstKind::PtrToGlobalVar(global_var),
                }),
            )
        };

        let (stack_array_global, ptr_to_stack_array_global) =
            invocation_local_global_var_and_ptr_to(config.stack_size_bytes, None);

        let scalar_type_of_stack_top = scalar::Type::U32;
        let (stack_top_global, ptr_to_stack_top_global) = invocation_local_global_var_and_ptr_to(
            scalar_type_of_stack_top.bit_width() / 8,
            Some(GlobalVarInit::Direct(cx.intern(scalar::Const::from_bits(
                scalar_type_of_stack_top,
                (config.stack_size_bytes / config.stack_unit_bytes).into(),
            )))),
        );

        Self {
            cx: cx.clone(),

            config,

            layout_cache: qptr::layout::LayoutCache::new(cx.clone(), &config.layout_config),

            type_of_stack_ptr: qptr_ty,
            type_of_stack_top: cx.intern(scalar_type_of_stack_top),

            stack_array_global,
            ptr_to_stack_array_global,

            stack_top_global,
            ptr_to_stack_top_global,
        }
    }

    fn size_of_for_stack_in_stack_units(&self, ty: Type, reason: &str) -> Result<NonZeroU32, Diag> {
        let mem_layout = self.layout_cache.fixed_mem_layout_of(ty, reason)?;

        if mem_layout.align > self.config.stack_unit_bytes.get() {
            return Err(Diag::bug([
                format!("alignment {} of type `", mem_layout.align).into(),
                ty.into(),
                format!("` not supported for {reason}").into(),
            ]));
        }

        // FIXME(eddyb) should be guaranteed by `Value` never being an aggregate.
        let size_in_bytes = NonZeroU32::new(mem_layout.size).unwrap();
        Ok(NonZeroU32::new(size_in_bytes.get().div_ceil(self.config.stack_unit_bytes.get()))
            .unwrap())
    }

    fn pusher(&self, func: FuncAtMut<'_, ()>) -> EmuStackPusherPopper<'a, '_, true> {
        self.pusher_popper(func)
    }
    fn popper(&self, func: FuncAtMut<'_, ()>) -> EmuStackPusherPopper<'a, '_, false> {
        self.pusher_popper(func)
    }

    // FIXME(eddyb) find a better name for this abstraction.
    fn pusher_popper<const CAN_PUSH: bool>(
        &self,
        func: FuncAtMut<'_, ()>,
    ) -> EmuStackPusherPopper<'a, '_, CAN_PUSH> {
        let stack_top_initial = func.data_insts.define(
            &self.cx,
            DataInstDef {
                attrs: AttrSet::default(),
                form: self.cx.intern(DataInstFormDef {
                    kind: DataInstKind::QPtr(QPtrOp::Load { offset: 0 }),
                    output_types: [self.type_of_stack_top].into_iter().collect(),
                }),
                inputs: [Value::Const(self.ptr_to_stack_top_global)].into_iter().collect(),
            }
            .into(),
        );

        let stack_ptr = func.data_insts.define(
            &self.cx,
            DataInstDef {
                attrs: AttrSet::default(),
                form: self.cx.intern(DataInstFormDef {
                    kind: DataInstKind::QPtr(QPtrOp::DynOffset {
                        stride: self.config.stack_unit_bytes,
                        index_bounds: Some(
                            0..self
                                .config
                                .stack_size_bytes
                                .checked_div(self.config.stack_unit_bytes.get())
                                .unwrap()
                                .try_into()
                                .unwrap(),
                        ),
                    }),
                    output_types: [self.type_of_stack_ptr].into_iter().collect(),
                }),
                inputs: [Value::Const(self.ptr_to_stack_array_global), Value::DataInstOutput {
                    inst: stack_top_initial,
                    output_idx: 0,
                }]
                .into_iter()
                .collect(),
            }
            .into(),
        );

        EmuStackPusherPopper {
            global_stack: self,
            stack_top_initial,
            stack_ptr,
            push_pop_insts: EntityList::empty(),
            offset_in_stack_units: 0,
            accessed_stack_unit_offsets: 0..0,
        }
    }

    // FIXME(eddyb) is this the right API?
    fn collect_cont_closure(
        &self,
        mut func: FuncAtMut<'_, ()>,
        origin: Either<(ControlRegion, RegionEmuStates), (ControlNode, ControlNodeEmuStates)>,
        cont_body: EmuContBody,
    ) -> (EmuContClosure, ControlRegionDef) {
        // FIXME(eddyb) implement closing over local variables, ideally handled
        // as a "stack frame" (just above continuation captures, and only popped
        // by the whole function returning, not any sooner - that means it has
        // to be handled at the whole-function level, where returns are handled).
        let first_func_local_var = func
            .reborrow()
            .freeze()
            .at(cont_body.children)
            .into_iter()
            .map_while(|func_at_child_node| match func_at_child_node.def().kind {
                ControlNodeKind::Block { insts } => Some(func_at_child_node.at(insts)),
                _ => None,
            })
            .flatten()
            .next()
            .filter(|func_at_inst| {
                matches!(
                    self.cx[func_at_inst.def().form].kind,
                    DataInstKind::QPtr(qptr::QPtrOp::FuncLocalVar(_))
                )
            })
            .map(|func_at_inst| func_at_inst.position);
        if let Some(first_func_local_var) = first_func_local_var {
            func.reborrow().at(first_func_local_var).def().attrs.push_diag(
                &self.cx,
                Diag::bug(["local variables NYI in emulated recursive call stack".into()]),
            );
        }

        let mut collector = EmuContClosureCollector {
            global_stack: self,
            popper: None,

            closure: EmuContClosure {
                origin: Ok(origin),
                input_count: origin.either_with(
                    func.reborrow().freeze(),
                    |func, (region, _)| func.at(region).def().inputs.len(),
                    |func, (node, _)| func.at(node).def().outputs.len(),
                ),
                captures: FxIndexSet::default(),
            },

            pops_of_inputs_and_captures: vec![],
            defined_region_inputs: EntityOrientedDenseMap::new(),
            defined_control_node_outputs: EntityOrientedDenseMap::new(),
            defined_data_inst_outputs: EntityOrientedDenseMap::new(),
        };

        // HACK(eddyb) guarantee the first `cont.input_count` pops.
        for i in 0..collector.closure.input_count {
            assert_eq!(collector.pops_of_inputs_and_captures.len(), i);
            let v = origin.either(
                |(region, _)| Value::ControlRegionInput {
                    region,
                    input_idx: i.try_into().unwrap(),
                },
                |(control_node, _)| Value::ControlNodeOutput {
                    control_node,
                    output_idx: i.try_into().unwrap(),
                },
            );
            match collector.transform_value_use_in_func(func.reborrow().at(v)) {
                Transformed::Unchanged => unreachable!(),
                Transformed::Changed(new) => {
                    assert!(
                        new == Value::DataInstOutput {
                            inst: collector.pops_of_inputs_and_captures[i],
                            output_idx: 0
                        }
                    );
                }
            }
        }
        assert_eq!(collector.pops_of_inputs_and_captures.len(), collector.closure.input_count);
        assert!(collector.closure.captures.is_empty());

        func.reborrow()
            .at(cont_body.children)
            .into_iter()
            .inner_in_place_transform_with(&mut collector);

        let mut cont_body_def = cont_body.into_region_def();

        if let Some(popper) = collector.popper.take() {
            let pops_block = popper.finish(func.reborrow());
            cont_body_def.children.insert_first(pops_block, func.control_nodes);
        }

        (collector.closure, cont_body_def)
    }
}

// FIXME(eddyb) find a better name for this abstraction.
struct EmuStackPusherPopper<
    'a,
    'b,
    // HACK(eddyb) this only exists to avoid having both stack overflow checks
    // (i.e. from pushes), and value definitions (i.e. pops) at the same time,
    // as it would need nesting an arbitrary user `ControlRegion` and plumbing
    // its outputs (so that it can access the popped values in the first place).
    const CAN_PUSH: bool,
> {
    global_stack: &'b EmuGlobalStack<'a>,

    stack_top_initial: DataInst,
    stack_ptr: DataInst,

    /// `DataInst`s for pushes (i.e. `qptr.store`) and pops (i.e `qptr.load`),
    /// without any e.g. global variable manipulation helper instructions.
    push_pop_insts: EntityList<DataInst>,

    offset_in_stack_units: i32,

    /// Stack unit offset range including all written/read bytes by pushes/pops,
    /// relative to `stack_ptr`.
    accessed_stack_unit_offsets: Range<i32>,
}

impl EmuStackPusherPopper<'_, '_, /*CAN_PUSH=*/ true> {
    fn push(&mut self, mut func: FuncAtMut<'_, ()>, v: Value) {
        let cx = &self.global_stack.cx;

        let mut attrs = AttrSet::default();
        let ty = func.reborrow().freeze().at(v).type_of(cx);
        let size_in_stack_units = self
            .global_stack
            .size_of_for_stack_in_stack_units(ty, "pushing state to emulated stack")
            .map_err(|err| {
                attrs.push_diag(cx, err);
            });

        self.accessed_stack_unit_offsets.end =
            self.accessed_stack_unit_offsets.end.max(self.offset_in_stack_units);
        self.offset_in_stack_units = self
            .offset_in_stack_units
            .checked_sub(size_in_stack_units.map_or(0, |size| size.get()).try_into().unwrap())
            .unwrap();
        self.accessed_stack_unit_offsets.start =
            self.accessed_stack_unit_offsets.start.min(self.offset_in_stack_units);

        let inst = func.data_insts.define(
            cx,
            DataInstDef {
                attrs,
                form: cx.intern(DataInstFormDef {
                    kind: DataInstKind::QPtr(QPtrOp::Store {
                        offset: self
                            .offset_in_stack_units
                            .checked_mul(
                                self.global_stack.config.stack_unit_bytes.get().try_into().unwrap(),
                            )
                            .unwrap(),
                    }),
                    output_types: [].into_iter().collect(),
                }),
                inputs: [Value::DataInstOutput { inst: self.stack_ptr, output_idx: 0 }, v]
                    .into_iter()
                    .collect(),
            }
            .into(),
        );
        self.push_pop_insts.insert_last(inst, func.data_insts);
    }
}

impl EmuStackPusherPopper<'_, '_, /*CAN_PUSH=*/ false> {
    fn pop(&mut self, func: FuncAtMut<'_, ()>, ty: Type) -> Value {
        let cx = &self.global_stack.cx;

        let mut attrs = AttrSet::default();
        let size_in_stack_units = self
            .global_stack
            .size_of_for_stack_in_stack_units(ty, "popping state from emulated stack")
            .map_err(|err| {
                attrs.push_diag(cx, err);
            });
        let inst = func.data_insts.define(
            cx,
            DataInstDef {
                attrs,
                form: cx.intern(DataInstFormDef {
                    kind: DataInstKind::QPtr(QPtrOp::Load {
                        offset: self
                            .offset_in_stack_units
                            .checked_mul(
                                self.global_stack.config.stack_unit_bytes.get().try_into().unwrap(),
                            )
                            .unwrap(),
                    }),
                    output_types: [ty].into_iter().collect(),
                }),
                inputs: [Value::DataInstOutput { inst: self.stack_ptr, output_idx: 0 }]
                    .into_iter()
                    .collect(),
            }
            .into(),
        );
        self.push_pop_insts.insert_last(inst, func.data_insts);

        self.accessed_stack_unit_offsets.start =
            self.accessed_stack_unit_offsets.start.min(self.offset_in_stack_units);
        self.offset_in_stack_units = self
            .offset_in_stack_units
            .checked_add(size_in_stack_units.map_or(0, |size| size.get()).try_into().unwrap())
            .unwrap();
        self.accessed_stack_unit_offsets.end =
            self.accessed_stack_unit_offsets.end.max(self.offset_in_stack_units);

        Value::DataInstOutput { inst, output_idx: 0 }
    }

    // HACK(eddyb) popping doesn't need to worry about stack overflows.
    fn finish(self, func: FuncAtMut<'_, ()>) -> ControlNode {
        let dummy_in = EmuStateIdx::UNKNOWN_STATE.to_value(&self.global_stack.cx);
        let (node, node2, dummy_out) = self.finish_for_state(func, dummy_in);
        assert!(node2.is_none() && dummy_out == dummy_in);
        node
    }
}

impl<const CAN_PUSH: bool> EmuStackPusherPopper<'_, '_, CAN_PUSH> {
    // HACK(eddyb) if stack overflows are possible (i.e. through pushes), the
    // `Value` returned will (dynamically) be a choice between `next_state` or
    // `EmuStateIdx::STACK_OVERFLOW`, and the caller needs to rely on it.
    fn finish_for_state(
        self,
        mut func: FuncAtMut<'_, ()>,
        next_state: Value,
    ) -> (ControlNode, Option<ControlNode>, Value) {
        let cx = &self.global_stack.cx;

        let min_neg_offset = Some(self.accessed_stack_unit_offsets.start).filter(|&x| x < 0);
        let final_offset = self.offset_in_stack_units;

        let (mut pre_check_insts, mut stack_overflow_checked_insts) = if min_neg_offset.is_some() {
            // FIXME(eddyb) support mixing pushes and pops w/ overflow checks
            // (right now they're mutually exclusive, so `self.insts` is stores-only).
            assert!(CAN_PUSH);

            (EntityList::empty(), Some(self.push_pop_insts))
        } else {
            (self.push_pop_insts, None)
        };

        // HACK(eddyb) `stack_top_initial` will be the first `pre_check_insts`
        // instruction, so this is a weird workaround for lacking `insert_after`.
        stack_overflow_checked_insts
            .as_mut()
            .unwrap_or(&mut pre_check_insts)
            .insert_first(self.stack_ptr, func.data_insts);
        pre_check_insts.insert_first(self.stack_top_initial, func.data_insts);

        let mut mk_stack_top_plus = |offset_in_stack_units: i32| {
            let inst = func.data_insts.define(
                cx,
                DataInstDef {
                    attrs: AttrSet::default(),
                    form: cx.intern(DataInstFormDef {
                        kind: DataInstKind::Scalar(scalar::Op::IntBinary(
                            if offset_in_stack_units < 0 {
                                scalar::IntBinOp::Sub
                            } else {
                                scalar::IntBinOp::Add
                            },
                        )),
                        output_types: [self.global_stack.type_of_stack_top].into_iter().collect(),
                    }),
                    inputs: [
                        Value::DataInstOutput { inst: self.stack_top_initial, output_idx: 0 },
                        Value::Const(
                            cx.intern(scalar::Const::from_u32(
                                offset_in_stack_units.unsigned_abs(),
                            )),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                }
                .into(),
            );

            if stack_overflow_checked_insts.is_some() {
                pre_check_insts.insert_last(inst, func.data_insts);
            } else {
                pre_check_insts.insert_before(inst, self.stack_ptr, func.data_insts);
            }

            inst
        };

        let stack_top_plus_min_neg_offset = min_neg_offset.map(&mut mk_stack_top_plus);
        let stack_top_plus_final_offset = if Some(final_offset) == min_neg_offset {
            // HACK(eddyb) reuse identical instruction (common case for push-only).
            stack_top_plus_min_neg_offset.unwrap()
        } else {
            mk_stack_top_plus(final_offset)
        };

        // HACK(eddyb) adjust `self.stack_ptr` so that all offsets of pushes'
        // `qptr.store`s are positive (before negative offsets are handled).
        if let Some(stack_top_plus_min_neg_offset) = stack_top_plus_min_neg_offset {
            let stack_ptr_idx_input = &mut func.reborrow().at(self.stack_ptr).def().inputs[1];
            assert!(
                *stack_ptr_idx_input
                    == Value::DataInstOutput { inst: self.stack_top_initial, output_idx: 0 }
            );
            *stack_ptr_idx_input =
                Value::DataInstOutput { inst: stack_top_plus_min_neg_offset, output_idx: 0 };

            let offset_delta = (-min_neg_offset.unwrap())
                .checked_mul(self.global_stack.config.stack_unit_bytes.get().try_into().unwrap())
                .unwrap();

            let mut func_at_insts =
                func.reborrow().at(stack_overflow_checked_insts.unwrap()).into_iter();
            assert!(func_at_insts.next().unwrap().position == self.stack_ptr);
            while let Some(func_at_inst) = func_at_insts.next() {
                let inst_def = func_at_inst.def();
                let mut inst_form_def = cx[inst_def.form].clone();
                match &mut inst_form_def.kind {
                    DataInstKind::QPtr(
                        qptr::QPtrOp::Load { offset } | qptr::QPtrOp::Store { offset },
                    ) => {
                        *offset = offset.checked_add(offset_delta).unwrap();
                    }
                    _ => unreachable!(),
                }
                inst_def.form = cx.intern(inst_form_def);
                assert!(
                    inst_def.inputs[0]
                        == Value::DataInstOutput { inst: self.stack_ptr, output_idx: 0 }
                );
            }
        }

        let stack_top_store = func.data_insts.define(
            cx,
            DataInstDef {
                attrs: AttrSet::default(),
                form: cx.intern(DataInstFormDef {
                    kind: DataInstKind::QPtr(QPtrOp::Store { offset: 0 }),
                    output_types: [].into_iter().collect(),
                }),
                inputs: [
                    Value::Const(self.global_stack.ptr_to_stack_top_global),
                    Value::DataInstOutput { inst: stack_top_plus_final_offset, output_idx: 0 },
                ]
                .into_iter()
                .collect(),
            }
            .into(),
        );
        stack_overflow_checked_insts
            .as_mut()
            .unwrap_or(&mut pre_check_insts)
            .insert_last(stack_top_store, func.data_insts);

        let def_block = |func: FuncAtMut<'_, ()>, insts| {
            func.control_nodes.define(
                cx,
                ControlNodeDef {
                    attrs: AttrSet::default(),
                    kind: ControlNodeKind::Block { insts },
                    outputs: [].into_iter().collect(),
                }
                .into(),
            )
        };

        let Some(checked_insts) = stack_overflow_checked_insts else {
            return (def_block(func, pre_check_insts), None, next_state);
        };

        let would_overflow_stack = func.data_insts.define(
            cx,
            DataInstDef {
                attrs: AttrSet::default(),
                form: cx.intern(DataInstFormDef {
                    kind: DataInstKind::Scalar(scalar::Op::IntBinary(scalar::IntBinOp::GtU)),
                    output_types: [cx.intern(scalar::Type::Bool)].into_iter().collect(),
                }),
                inputs: [
                    Value::DataInstOutput {
                        inst: stack_top_plus_min_neg_offset.unwrap(),
                        output_idx: 0,
                    },
                    Value::DataInstOutput { inst: self.stack_top_initial, output_idx: 0 },
                ]
                .into_iter()
                .collect(),
            }
            .into(),
        );
        pre_check_insts.insert_last(would_overflow_stack, func.data_insts);

        let pre_check_block = def_block(func.reborrow(), pre_check_insts);

        let cases =
            [(None, EmuStateIdx::STACK_OVERFLOW.to_value(cx)), (Some(checked_insts), next_state)]
                .into_iter()
                .map(|(insts, output)| {
                    let mut children = EntityList::empty();
                    if let Some(insts) = insts {
                        children.insert_last(def_block(func.reborrow(), insts), func.control_nodes);
                    }
                    func.control_regions.define(cx, ControlRegionDef {
                        inputs: [].into_iter().collect(),
                        children,
                        outputs: [output].into_iter().collect(),
                    })
                })
                .collect();

        let check_node = func.control_nodes.define(
            cx,
            ControlNodeDef {
                attrs: AttrSet::default(),
                kind: ControlNodeKind::Select {
                    kind: SelectionKind::BoolCond,
                    scrutinee: Value::DataInstOutput { inst: would_overflow_stack, output_idx: 0 },
                    cases,
                },
                outputs: [ControlNodeOutputDecl {
                    attrs: AttrSet::default(),
                    ty: cx.intern(EmuStateIdx::TYPE),
                }]
                .into_iter()
                .collect(),
            }
            .into(),
        );
        (pre_check_block, Some(check_node), Value::ControlNodeOutput {
            control_node: check_node,
            output_idx: 0,
        })
    }
}

struct EmuContClosureCollector<'a, 'b> {
    global_stack: &'b EmuGlobalStack<'a>,
    popper: Option<EmuStackPusherPopper<'a, 'b, false>>,

    /// The continuation whose `captures` are being collected.
    closure: EmuContClosure,

    /// `qptr.load` instructions (each added to `pops_block`) popping all inputs
    /// (`cont.input_count`) and all captures (keyed/ordered by `cont.captures`,
    /// after the inputs) from the stack, ordered by increasing stack offset
    /// (from the stack top upwards, i.e. towards earlier pushed values).
    ///
    /// Note that pushing has to be done in reverse (moving the stack top downwards),
    /// though this is mainly relevant when inputs are pushed separately, which
    /// is why inputs are popped first (`0..cont.input_count`), and pushed last.
    pops_of_inputs_and_captures: Vec<DataInst>,

    // HACK(eddyb) efficient tracking to allow determining if a `Value` is part
    // of the continuation itself, or a capture (see `popped_values`).
    // FIXME(eddyb) there should be (sparse) bitset versions of these.
    defined_region_inputs: EntityOrientedDenseMap<ControlRegion, ()>,
    defined_control_node_outputs: EntityOrientedDenseMap<ControlNode, ()>,
    defined_data_inst_outputs: EntityOrientedDenseMap<DataInst, ()>,
}

/// The "closure" of an emulated continuation, which includes everything needed
/// to invoke it (i.e. pushing `captures` and inputs to the stack).
struct EmuContClosure {
    /// The reason for this continuation to exist, including the `EmuStateIdx`
    /// associated with it (for `RegionEmuStates`, `entry_state` must be `Some`).
    //
    // HACK(eddyb) the `Result` around the whole thing allows "dynamic" states,
    // but the whole abstraction should be refactored.
    origin: Result<
        Either<(ControlRegion, RegionEmuStates), (ControlNode, ControlNodeEmuStates)>,
        Value,
    >,
    /// The count of, depending on `origin`:
    /// - when `(ControlRegion, _)`: all region (i.e. func/loop body) inputs
    /// - when `(ControlNode, _)`: all node (i.e. being merged) outputs
    input_count: usize,
    /// Set of `Value`s defined outside of the continuation, requiring each a
    /// stack pop in the continuation, and a matching push on the other side.
    captures: FxIndexSet<Value>,
}

impl EmuContClosure {
    fn entry_state_idx(&self) -> Result<EmuStateIdx, Value> {
        self.origin.map(|o| {
            o.either(|(_, states)| states.entry_state.unwrap(), |(_, states)| states.merge)
        })
    }
    fn entry_state_value(&self, cx: &Context) -> Value {
        match self.entry_state_idx() {
            Ok(s) => s.to_value(cx),
            Err(v) => v,
        }
    }
}

/// The "body" of an emulated continuation, which will be executed when invoking
/// the respective [`EmuContClosure`], i.e. a `ControlRegion` with:
/// - no inputs
/// - child `ControlNode`s (including all necessary stack manipulation)
/// - one output: `next_state_after` (see also its documentation)
struct EmuContBody {
    children: EntityList<ControlNode>,

    /// The potentially-dynamic `EmuStateIdx` to switch to after `children`, e.g.:
    /// - constant `EmuStateIdx` (chain into another continuation)
    /// - dynamic `EmuStateIdx` popped off the stack (return from an emulated call)
    /// - output of a tail `Select` choosing between N instances of the above
    ///   (`Select` cases acting like `EmuContBody`s for "immediately invoked"
    ///   continuations that weren't wastefully each given distinct states)
    ///
    /// Regardless of which state is being switched to, `children` must end in
    /// the appropriate pushes (e.g. as described by the `EmuContClosure` of a
    /// destination continuation).
    next_state_after: Value,
}

impl EmuContBody {
    fn into_region_def(self) -> ControlRegionDef {
        let EmuContBody { children, next_state_after } = self;
        ControlRegionDef {
            inputs: [].into_iter().collect(),
            children,
            outputs: [next_state_after].into_iter().collect(),
        }
    }
}

impl Transformer for EmuContClosureCollector<'_, '_> {
    fn transform_value_use_in_func(
        &mut self,
        func_at_val: FuncAtMut<'_, Value>,
    ) -> Transformed<Value> {
        let v = func_at_val.position;
        let mut func = func_at_val.at(());

        let already_valid = match v {
            Value::Const(_) => true,
            Value::ControlRegionInput { region, .. } => {
                self.defined_region_inputs.get(region).is_some()
            }
            Value::ControlNodeOutput { control_node, .. } => {
                self.defined_control_node_outputs.get(control_node).is_some()
            }
            Value::DataInstOutput { inst, .. } => {
                self.defined_data_inst_outputs.get(inst).is_some()
            }
        };
        if already_valid {
            return Transformed::Unchanged;
        }

        let origin_region_or_node =
            self.closure.origin.ok().unwrap().map_either(|(region, _)| region, |(node, _)| node);
        let cont_input_idx = match (origin_region_or_node, v) {
            (Either::Left(r1), Value::ControlRegionInput { region: r2, input_idx }) if r1 == r2 => {
                Some(input_idx)
            }
            (Either::Right(n1), Value::ControlNodeOutput { control_node: n2, output_idx })
                if n1 == n2 =>
            {
                Some(output_idx)
            }
            _ => None,
        };

        let pop_idx = match cont_input_idx {
            Some(input_idx) => input_idx.try_into().unwrap(),
            None => self.closure.input_count + self.closure.captures.insert_full(v).0,
        };

        if let Some(&inst) = self.pops_of_inputs_and_captures.get(pop_idx) {
            // Already seen (i.e. effectively cached).
            return Transformed::Changed(Value::DataInstOutput { inst, output_idx: 0 });
        }

        // Generate the pop (and scaffolding, if not yet present).
        assert_eq!(pop_idx, self.pops_of_inputs_and_captures.len());

        let ty = func.reborrow().freeze().at(v).type_of(&self.global_stack.cx);
        let popped_value = self
            .popper
            .get_or_insert_with(|| self.global_stack.popper(func.reborrow()))
            .pop(func.reborrow(), ty);
        match popped_value {
            Value::DataInstOutput { inst, output_idx: 0 } => {
                self.pops_of_inputs_and_captures.push(inst);
            }
            _ => unreachable!(),
        }
        Transformed::Changed(popped_value)
    }

    fn in_place_transform_control_region_def(
        &mut self,
        mut func_at_control_region: FuncAtMut<'_, ControlRegion>,
    ) {
        self.defined_region_inputs.insert(func_at_control_region.position, ());
        func_at_control_region.inner_in_place_transform_with(self);
    }

    fn in_place_transform_control_node_def(
        &mut self,
        mut func_at_control_node: FuncAtMut<'_, ControlNode>,
    ) {
        func_at_control_node.inner_in_place_transform_with(self);
        self.defined_control_node_outputs.insert(func_at_control_node.position, ());
    }

    fn in_place_transform_data_inst_def(&mut self, mut func_at_data_inst: FuncAtMut<'_, DataInst>) {
        func_at_data_inst.inner_in_place_transform_with(self);
        self.defined_data_inst_outputs.insert(func_at_data_inst.position, ());
    }
}

struct EmuStateSplitter<'a> {
    global_stack: &'a EmuGlobalStack<'a>,

    states: &'a FuncEmuStates,
    func_call_emu_cont: &'a FxIndexMap<Func, EmuContClosure>,

    state_switch_cases: BTreeMap<EmuStateIdx, ControlRegion>,
}

impl EmuStateSplitter<'_> {
    fn maybe_split_states_in_region(
        &mut self,
        func_at_region: FuncAtMut<'_, ControlRegion>,
        next_cont_after_region: &EmuContClosure,
    ) {
        let cx = &self.global_stack.cx;

        let region = func_at_region.position;
        let mut func = func_at_region.at(());

        let maybe_region_states = self.states.for_region.get(region).copied();

        let mut cont_body = {
            let outputs = mem::take(&mut func.reborrow().at(region).def().outputs);
            assert_eq!(outputs.len(), next_cont_after_region.input_count);

            let values_in_pop_order =
                outputs.into_iter().chain(next_cont_after_region.captures.iter().copied());
            let values_in_push_order = values_in_pop_order.rev();

            let mut pusher = self.global_stack.pusher(func.reborrow());
            for v in values_in_push_order {
                pusher.push(func.reborrow(), v);
            }
            let (pushes_node, pushes_node2, next_state_after) = pusher
                .finish_for_state(func.reborrow(), next_cont_after_region.entry_state_value(cx));

            let mut children = EntityList::empty();
            children.insert_last(pushes_node, func.control_nodes);
            if let Some(pushes_node2) = pushes_node2 {
                children.insert_last(pushes_node2, func.control_nodes);
            }

            EmuContBody { children, next_state_after }
        };

        let Some(region_states) = maybe_region_states else {
            let region_def = func.at(region).def();
            assert_eq!(region_def.inputs.len(), 0);
            *region_def = cont_body.into_region_def();
            return;
        };

        while let Some(node) = func.control_regions[region].children.remove_last(func.control_nodes)
        {
            if let Some(&node_states) = self.states.for_control_node.get(node) {
                let (merge_cont, merge_cont_body_def) = self.global_stack.collect_cont_closure(
                    func.reborrow(),
                    Either::Right((node, node_states)),
                    cont_body,
                );
                assert_eq!(merge_cont.entry_state_idx().ok().unwrap(), node_states.merge);

                self.state_switch_cases.insert(
                    node_states.merge,
                    func.control_regions.define(cx, merge_cont_body_def),
                );

                cont_body = self.split_states_in_control_node(func.reborrow().at(node), merge_cont);
            } else {
                cont_body.children.insert_first(node, func.control_nodes);
            }
        }

        // FIXME(eddyb) maybe `split_states_in_region`'s return type should
        // indicate whether `region` is still usable in its original parent,
        // or if it has been added to the state `switch`.
        let new_region_def = if let Some(entry_state) = region_states.entry_state {
            let (entry_cont, entry_cont_body_def) = self.global_stack.collect_cont_closure(
                func.reborrow(),
                Either::Left((region, region_states)),
                cont_body,
            );
            assert_eq!(entry_cont.entry_state_idx().ok().unwrap(), entry_state);

            self.state_switch_cases.insert(entry_state, region);

            entry_cont_body_def
        } else {
            cont_body.into_region_def()
        };
        *func.at(region).def() = new_region_def;
    }

    fn split_states_in_control_node(
        &mut self,
        func_at_node: FuncAtMut<'_, ControlNode>,
        next_cont_after_node: EmuContClosure,
    ) -> EmuContBody {
        let cx = &self.global_stack.cx;

        let node = func_at_node.position;
        let mut func = func_at_node.at(());

        // FIXME(eddyb) cache this somewhere.
        let state_ty = cx.intern(EmuStateIdx::TYPE);

        let mut children = EntityList::empty();
        let next_state_after = match &mut func.control_nodes[node].kind {
            ControlNodeKind::Block { .. } | ControlNodeKind::ExitInvocation { .. } => {
                unreachable!()
            }

            // FIXME(eddyb) DRY this vs `maybe_split_states_in_region`.
            ControlNodeKind::FuncCall { callee, inputs } => {
                let callee_entry_cont = &self.func_call_emu_cont[&*callee];

                let inputs = mem::take(inputs);
                assert_eq!(inputs.len(), callee_entry_cont.input_count);
                assert!(callee_entry_cont.captures.is_empty());

                // FIXME(eddyb) the `.rev()` usage here is not self-explanatory.
                let values_in_push_order = (next_cont_after_node.captures.iter().copied().rev())
                    .chain([next_cont_after_node.entry_state_value(cx)])
                    .chain(inputs.iter().copied().rev());

                let mut pusher = self.global_stack.pusher(func.reborrow());
                for v in values_in_push_order {
                    pusher.push(func.reborrow(), v);
                }
                let (pushes_node, pushes_node2, next_state_after) = pusher
                    .finish_for_state(func.reborrow(), callee_entry_cont.entry_state_value(cx));

                children.insert_last(pushes_node, func.control_nodes);
                if let Some(pushes_node2) = pushes_node2 {
                    children.insert_last(pushes_node2, func.control_nodes);
                }

                next_state_after
            }
            ControlNodeKind::Select { cases, .. } => {
                for case_idx in 0..cases.len() {
                    let case = match &func.control_nodes[node].kind {
                        ControlNodeKind::Select { cases, .. } => cases[case_idx],
                        _ => unreachable!(),
                    };

                    self.maybe_split_states_in_region(
                        func.reborrow().at(case),
                        &next_cont_after_node,
                    );
                }

                children.insert_last(node, func.control_nodes);

                func.reborrow().at(node).def().outputs =
                    [ControlNodeOutputDecl { attrs: AttrSet::default(), ty: state_ty }]
                        .into_iter()
                        .collect();
                Value::ControlNodeOutput { control_node: node, output_idx: 0 }
            }
            ControlNodeKind::Loop { .. } => todo!(),
        };

        EmuContBody { children, next_state_after }
    }
}
