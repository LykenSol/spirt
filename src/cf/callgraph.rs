//! Function call graph abstractions and utilities.
//
// TODO(eddyb) split this into at least `cf::callgraph` and `cf::reentrant`/`cf::stackful`,
// maybe something like `cf::cps` for the state machine stuff.

use crate::cf::SelectionKind;
use crate::func_at::{FuncAt, FuncAtMut};
use crate::mem::MemOp;
use crate::qptr::QPtrOp;
use crate::transform::{InnerInPlaceTransform, Transformed, Transformer};
use crate::visit::{self, InnerVisit as _, Visitor};
use crate::{
    AddrSpace, AttrSet, Const, ConstDef, ConstKind, Context, DataInst, DataInstDef, DataInstKind,
    DeclDef, Diag, EntityList, EntityOrientedDenseMap, Exportee, Func, FuncDefBody, FuncParam,
    FxIndexMap, FxIndexSet, GlobalVar, GlobalVarDecl, GlobalVarDefBody, GlobalVarInit, Module,
    Node, NodeDef, NodeKind, Region, RegionDef, Type, TypeKind, Value, Var, VarDecl, VarKind,
    scalar, spv,
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
    pub func_call_node: Node,
    pub parent_region: Region,
}

impl CallGraph {
    pub fn compute(module: &Module) -> Self {
        let mut collector = CallGraphCollector {
            cx: module.cx_ref(),
            wk: &spv::spec::Spec::get().well_known,
            module,

            call_graph: Self {
                caller_to_callees: FxIndexMap::default(),
                indirect_callees: FxIndexSet::default(),
            },
            caller: Err("Module"),

            parent_region: None,

            seen_attrs: FxHashSet::default(),
            seen_types: FxHashSet::default(),
            seen_consts: FxHashSet::default(),
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
    wk: &'static spv::spec::WellKnown,
    module: &'a Module,

    call_graph: CallGraph,
    caller: Result<Func, &'static str>,

    parent_region: Option<Region>,

    // FIXME(eddyb) build some automation to avoid ever repeating these.
    seen_attrs: FxHashSet<AttrSet>,
    seen_types: FxHashSet<Type>,
    seen_consts: FxHashSet<Const>,
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
            if let ConstKind::PtrToFunc(func) = ct_def.kind {
                let ConstDef { attrs, ty, kind: _ } = *ct_def;

                self.visit_attr_set_use(attrs);
                self.visit_type_use(ty);

                // HACK(eddyb) bypass `visit_func_use` entirely for fn pointers.
                if self.call_graph.indirect_callees.insert(func) {
                    self.visit_func_used_by_export_or_callee(func);
                }
            } else {
                self.with_caller(Err("Const"), |this| {
                    this.visit_const_def(ct_def);
                });
            }
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
    fn visit_region_def(&mut self, func_at_region: FuncAt<'_, Region>) {
        let outer_region = self.parent_region.replace(func_at_region.position);
        func_at_region.inner_visit_with(self);
        self.parent_region = outer_region;
    }
    fn visit_node_def(&mut self, func_at_node: FuncAt<'_, Node>) {
        let NodeDef { attrs, kind, inputs, child_regions: _, outputs } = func_at_node.def();
        if let (Ok(caller), DataInstKind::FuncCall(callee)) = (self.caller, kind) {
            self.visit_attr_set_use(*attrs);

            // HACK(eddyb) bypass `visit_func_use` entirely for static calls.
            let callees = self.call_graph.caller_to_callees.entry(caller).or_default();
            callees.direct.entry(*callee).or_default().push(CallSite {
                func_call_node: func_at_node.position,
                parent_region: self.parent_region.unwrap(),
            });
            self.visit_func_used_by_export_or_callee(*callee);

            for v in inputs {
                self.visit_value_use(v);
            }
            for &output in outputs {
                self.visit_var_decl(func_at_node.at(output));
            }
            return;
        }

        if let (Ok(caller), DataInstKind::SpvInst(spv_inst, _)) = (self.caller, kind)
            && spv_inst.opcode == self.wk.OpFunctionPointerCallINTEL
        {
            let callees = self.call_graph.caller_to_callees.entry(caller).or_default();
            callees.indirect.push(CallSite {
                func_call_node: func_at_node.position,
                parent_region: self.parent_region.unwrap(),
            });
        }

        func_at_node.inner_visit_with(self);
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
    pub layout_config: crate::mem::LayoutConfig,

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
    /// at the end of a `Region`, expecting that:
    /// - an error *may* be reported (e.g. passing `msg` to some "debug printf")
    /// - control-flow *must* diverge (in `Node`s added to the region),
    ///   and never exit the region normally (into the surrounding function)
    ///   - failure to respect this *will not* directly cause UB, but rather
    ///     infinite looping, which may be treated as UB downstream of SPIR-T,
    ///     but even non-UB infinite looping causing GPU timeouts should be
    ///     avoided, as not all user configurations (OS/drivers/hardware/etc.)
    ///     are robust (enough) wrt hangs and may degrade the rest of the system
    //
    // FIXME(eddyb) consider using an `enum` for the messages?
    pub build_fatal_error: Box<dyn Fn(&str, &Context, FuncAtMut<'_, Region>)>,
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

    // HACK(eddyb) all indirectly-callable functions must share a group, for
    // indirect calls to become calls in that group, with dynamic entry state.
    // FIXME(eddyb) address some of the inefficiencies, by e.g. clustering
    // indirect calls using (a simplified form of) their signature, and/or
    // avoiding the need for non-recursive indirect calls to share a group etc.
    indirect_callee_emu_group: Option<CallEmuGroup>,

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

        // HACK(eddyb) all indirectly-callable functions must share a group, for
        // indirect calls to become calls in that group, with dynamic entry state.
        let indirect_callee_emu_group = (!call_graph.indirect_callees.is_empty()).then(|| {
            call_graph
                .indirect_callees
                .iter()
                .find_map(|&func| func_emu_summary[func].emu_group)
                .unwrap_or_else(|| CallEmuGroup {
                    scc_root: *call_graph.indirect_callees.first().unwrap(),
                })
        });

        // HACK(eddyb) as indirectly called functions are accurately processed
        // by `CycleFinder`, they can easily end up in separate `CallEmuGroup`s
        // (or in none at all, if they contain no recursive/indirect calls),
        // so now all of their `CallEmuGroup`s have to be artificially unified.
        if let Some(indirect_callee_emu_group) = indirect_callee_emu_group {
            for &func in &call_graph.indirect_callees {
                let emu_group = func_emu_summary[func].emu_group;
                if emu_group == Some(indirect_callee_emu_group) {
                    continue;
                }

                if let Some(emu_group) = emu_group {
                    let emu_group_funcs = mem::take(&mut call_emu_groups[&emu_group]);
                    for &emu_group_func in &emu_group_funcs {
                        func_emu_summary[emu_group_func].emu_group =
                            Some(indirect_callee_emu_group);
                    }
                    call_emu_groups
                        .entry(indirect_callee_emu_group)
                        .or_default()
                        .extend(emu_group_funcs);
                } else {
                    func_emu_summary[func].emu_group = Some(indirect_callee_emu_group);
                    call_emu_groups.entry(indirect_callee_emu_group).or_default().insert(func);
                }
            }
            call_emu_groups.retain(|_, funcs| !funcs.is_empty());
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

            indirect_callee_emu_group,

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
                let reserved_body_states =
                    state_reserver.any_states_reserved_during(|state_reserver| {
                        func_def_body.inner_visit_with(state_reserver);
                    });

                // HACK(eddyb) leaf indirect callees won't reserve any states
                // on their own, even for their whole function body.
                if !reserved_body_states {
                    let body_states =
                        RegionEmuStates { entry_state: Some(state_reserver.reserve_state()) };
                    state_reserver.states.for_region.insert(func_def_body.body, body_states);
                }

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
                (
                    func,
                    EmuContClosure {
                        origin: Ok(Either::Left((
                            body_region,
                            func_states.for_region[body_region],
                        ))),
                        input_count: func_decl.params.len(),
                        captures: FxIndexSet::default(),
                    },
                )
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

            if self.indirect_callee_emu_group == Some(emu_group) {
                let caller_func_def_body = match &mut module.funcs[caller].def {
                    DeclDef::Present(func_def_body) => func_def_body,
                    DeclDef::Imported(_) => unreachable!(),
                };
                for &call_site in &callees.indirect {
                    let call_node_def = &mut caller_func_def_body.nodes[call_site.func_call_node];
                    // HACK(eddyb) temporarily let the call become malformed.
                    // FIXME(eddyb) pass the call args explicitly into
                    // `transform_inter_emu_group_func_call` to avoid this.
                    let indirect_callee = call_node_def.inputs.remove(0);
                    let indirect_call_emu_cont = EmuContClosure {
                        origin: Err(indirect_callee),
                        input_count: call_node_def.inputs.len(),
                        captures: FxIndexSet::default(),
                    };
                    self.transform_inter_emu_group_func_call(
                        caller_func_def_body,
                        call_site,
                        &indirect_call_emu_cont,
                        emu_group,
                    );
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

            let new_body = func_def_body.regions.define(cx, RegionDef::default());
            let orig_body = mem::replace(&mut func_def_body.body, new_body);

            let mut func = func_def_body.at_mut(());
            let ret_cont = {
                let mut popper = self.global_stack.popper(func.reborrow());
                let popped_state = Value::Var(popper.pop(func.reborrow(), state_ty));
                let pops_nodes = popper.finish(func.reborrow());
                func.regions[orig_body].children.append(pops_nodes, func.nodes);
                EmuContClosure {
                    origin: Err(popped_state),
                    input_count: orig_ret_types.len(),
                    captures: FxIndexSet::default(),
                }
            };

            let state_switch_cases = {
                let mut fracker = EmuFuncFracker {
                    global_stack: &self.global_stack,

                    states: func_states,
                    func_call_emu_cont: &func_call_emu_cont,

                    state_switch_cases: BTreeMap::new(),
                };
                let entry_cont = fracker
                    .frack_region_as_needed(
                        func.reborrow().at(orig_body),
                        |global_stack, mut func, children, outputs| {
                            let mut cont_body = global_stack.invoke_cont_closure(
                                func.reborrow(),
                                &ret_cont,
                                outputs,
                            );
                            cont_body.children.prepend(children, func.nodes);
                            cont_body
                        },
                    )
                    .unwrap();
                assert!(entry_cont.captures.is_empty());
                fracker.state_switch_cases
            };

            let current_state = {
                let input_var = func.vars.define(
                    cx,
                    VarDecl {
                        attrs: AttrSet::default(),
                        ty: state_ty,

                        def_parent: Either::Left(new_body),
                        def_idx: 0,
                    },
                );
                func.regions[new_body].inputs.push(input_var);
                Value::Var(input_var)
            };

            // HACK(eddyb) this is how the whole group gets chained together.
            let default_case = {
                let mut children = EntityList::empty();
                let next_state_after =
                    match per_func_states.get_index(func_idx + 1).map(|(&f, _)| f) {
                        Some(next_func_in_group) => {
                            // FIXME(eddyb) DRY vs `transform_inter_emu_group_func_call`?
                            let call_inst = func.nodes.define(
                                cx,
                                DataInstDef {
                                    attrs: AttrSet::default(),
                                    kind: DataInstKind::FuncCall(next_func_in_group),
                                    inputs: [current_state].into_iter().collect(),
                                    child_regions: [].into_iter().collect(),
                                    outputs: [].into_iter().collect(),
                                }
                                .into(),
                            );

                            // FIXME(eddyb) automate this (insertion cursor?).
                            let call_output_var = func.vars.define(
                                cx,
                                VarDecl {
                                    attrs: Default::default(),
                                    ty: state_ty,
                                    def_parent: Either::Right(call_inst),
                                    def_idx: 0,
                                },
                            );
                            func.nodes[call_inst].outputs.push(call_output_var);

                            children.insert_last(call_inst, func.nodes);
                            Value::Var(call_output_var)
                        }
                        None => EmuStateIdx::UNKNOWN_STATE.to_value(cx),
                    };
                func.regions
                    .define(cx, EmuContBody { children, next_state_after }.into_region_def())
            };

            let state_switch_node = {
                let (case_consts, mut cases): (_, SmallVec<_>) = state_switch_cases
                    .into_iter()
                    .map(|(case_const, case)| (case_const.as_scalar(), case))
                    .unzip();
                cases.push(default_case);
                func.nodes.define(
                    cx,
                    NodeDef {
                        attrs: AttrSet::default(),
                        kind: NodeKind::Select(SelectionKind::Switch { case_consts }),
                        inputs: [current_state].into_iter().collect(),
                        child_regions: cases,
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                )
            };

            // FIXME(eddyb) automate this (insertion cursor?).
            let state_switch_output_var = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: state_ty,
                    def_parent: Either::Right(state_switch_node),
                    def_idx: 0,
                },
            );
            func.nodes[state_switch_node].outputs.push(state_switch_output_var);

            let new_body_def = &mut func.regions[new_body];
            new_body_def.children.insert_last(state_switch_node, func.nodes);
            new_body_def.outputs.push(Value::Var(state_switch_output_var));
        }

        // HACK(eddyb) function pointers can be replaced by entry states indices,
        // but this only handles consts in constant data (i.e. vtables).
        if Some(emu_group) == self.indirect_callee_emu_group {
            // FIXME(eddyb) automate this and/or make it a `Module` wrapper.
            let all_uses_from_module = visit::AllUses::from_module(module);

            // HACK(eddyb) this is copied from `qptr::legalize`, DRY it!
            for &gv in &all_uses_from_module.global_vars {
                let gv_decl = &mut module.global_vars[gv];
                let transform_const_use = |ct: Const, untyped_size: u32| -> Transformed<Const> {
                    let func_ptr_size =
                        self.global_stack.config.layout_config.logical_ptr_size_align.0;
                    // FIXME(eddyb) support other fn ptr sizes.
                    assert_eq!(func_ptr_size * 8, EmuStateIdx::TYPE.bit_width());

                    let maybe_func_ptr_entry_state = if untyped_size == func_ptr_size
                        && let ConstKind::PtrToFunc(func) = cx[ct].kind
                    {
                        func_call_emu_cont
                            .get(&func)
                            .and_then(|call_emu_cont| call_emu_cont.entry_state_idx().ok())
                    } else {
                        None
                    };
                    match maybe_func_ptr_entry_state {
                        Some(state) => Transformed::Changed(cx.intern(state.as_scalar())),
                        None => Transformed::Unchanged,
                    }
                };
                if let DeclDef::Present(gv_def) = &mut gv_decl.def
                    && let Some(init) = &mut gv_def.initializer
                {
                    match init {
                        GlobalVarInit::Data(const_data) => {
                            use crate::mem::const_data::Part;

                            let mut next_offset = 0;
                            while next_offset < const_data.size() {
                                let part =
                                    const_data.read(next_offset..const_data.size()).next().unwrap();

                                let offset = next_offset;
                                next_offset += part.size().get();

                                let encoded_part = match part {
                                    Part::Uninit { .. } | Part::Bytes(_) => Transformed::Unchanged,
                                    Part::Symbolic { size, maybe_partial_slice: _, value } => {
                                        transform_const_use(value, size.get())
                                    }
                                };

                                // FIXME(eddyb) deduplicate this with `qptr::lower`.
                                let mut write_scalar = |scalar: scalar::Const| {
                                    let byte_len = match scalar.ty() {
                                        scalar::Type::Bool => {
                                            self.global_stack
                                                .config
                                                .layout_config
                                                .abstract_bool_size_align
                                                .0
                                        }
                                        scalar::Type::SInt(_)
                                        | scalar::Type::UInt(_)
                                        | scalar::Type::Float(_) => {
                                            let bit_width = scalar.ty().bit_width();
                                            assert_eq!(bit_width % 8, 0);
                                            bit_width / 8
                                        }
                                    };

                                    // HACK(eddyb) only perfectly overwriting is allowed.
                                    assert_eq!(byte_len, next_offset - offset);

                                    let mut bytes = scalar.bits().to_le_bytes();
                                    let bytes = &mut bytes[..byte_len as usize];
                                    if self.global_stack.config.layout_config.is_big_endian {
                                        bytes.reverse();
                                    }
                                    const_data.write_bytes(offset, bytes).unwrap();
                                };

                                if let Transformed::Changed(encoded) = encoded_part {
                                    write_scalar(*encoded.as_scalar(cx).unwrap());
                                }
                            }
                        }
                        GlobalVarInit::Direct(ct) => {
                            let untyped_size = match gv_decl.shape {
                                Some(crate::mem::shapes::GlobalVarShape::UntypedData(
                                    mem_layout,
                                )) => Some(mem_layout.size),
                                Some(
                                    crate::mem::shapes::GlobalVarShape::Handles { .. }
                                    | crate::mem::shapes::GlobalVarShape::TypedInterface(_),
                                )
                                | None => None,
                            };
                            if let Some(size) = untyped_size {
                                transform_const_use(*ct, size).apply_to(ct);
                            }
                        }
                        // HACK(eddyb) all errors will have been reported earlier.
                        GlobalVarInit::SpvAggregate { .. } => {}
                    }
                }
            }
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

        let call_args = mem::take(&mut func.reborrow().at(call_site.func_call_node).def().inputs);

        // FIXME(eddyb) is `call_emu_cont` unnecessary?
        assert!(call_emu_cont.captures.is_empty());

        // TODO(eddyb) consider using `invoke_cont_closure` here.
        let (args_pushes_nodes, initial_state) = {
            let mut pusher = self.global_stack.pusher(func.reborrow());
            pusher.push(func.reborrow(), call_site_ret_cont_state.to_value(cx));
            for v in call_args.into_iter().rev() {
                pusher.push(func.reborrow(), v);
            }
            pusher.finish_for_state(func.reborrow(), call_emu_cont.entry_state_value(cx))
        };
        {
            // HACK(eddyb) support splicing lists to make this O(1).
            let mut args_pushes_nodes = args_pushes_nodes;
            while let Some(node) = args_pushes_nodes.remove_first(func.nodes) {
                func.regions[call_site.parent_region].children.insert_before(
                    node,
                    call_site.func_call_node,
                    func.nodes,
                );
            }
        }

        let call_node_output_indices = 0..func.nodes[call_site.func_call_node].outputs.len();
        let (ret_vals_pops_nodes, ret_vals) = {
            let mut popper = self.global_stack.popper(func.reborrow());
            let ret_vals: SmallVec<_> = call_node_output_indices
                .clone()
                .map(|call_output_idx| {
                    let call_output = func.nodes[call_site.func_call_node].outputs[call_output_idx];
                    let ty = func.vars[call_output].ty;
                    Value::Var(popper.pop(func.reborrow(), ty))
                })
                .collect();
            (popper.finish(func.reborrow()), ret_vals)
        };

        let state_ty = cx.intern(EmuStateIdx::TYPE);
        let state_machine_loop_body = func.regions.define(cx, RegionDef::default());
        let current_state = {
            let input_var = func.vars.define(
                cx,
                VarDecl {
                    attrs: AttrSet::default(),
                    ty: state_ty,

                    def_parent: Either::Left(state_machine_loop_body),
                    def_idx: 0,
                },
            );
            func.regions[state_machine_loop_body].inputs.push(input_var);
            Value::Var(input_var)
        };
        let next_state = {
            let call_inst = func.nodes.define(
                cx,
                DataInstDef {
                    attrs: AttrSet::default(),
                    kind: DataInstKind::FuncCall(callee_emu_group.scc_root),
                    inputs: [current_state].into_iter().collect(),
                    child_regions: [].into_iter().collect(),
                    outputs: [].into_iter().collect(),
                }
                .into(),
            );

            // FIXME(eddyb) automate this (insertion cursor?).
            let call_output_var = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: state_ty,
                    def_parent: Either::Right(call_inst),
                    def_idx: 0,
                },
            );
            func.nodes[call_inst].outputs.push(call_output_var);

            func.regions[state_machine_loop_body].children.insert_last(call_inst, func.nodes);
            Value::Var(call_output_var)
        };
        func.regions[state_machine_loop_body].outputs.push(next_state);

        // NOTE(eddyb) the `switch` default case is where all intermediary states
        // will end up (i.e. the state machine loop keeps going), with only the
        // successful return, and error states, being matched for explicitly.
        let non_default_next_state_switch_cases = [
            (call_site_ret_cont_state, Ok((ret_vals_pops_nodes, ret_vals))),
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
                let case_region = func.regions.define(cx, RegionDef::default());
                let break_with_outputs = match maybe_non_default_case {
                    Some((_, Ok((ret_vals_pops_nodes, ret_vals)))) => {
                        func.regions[case_region].children.append(ret_vals_pops_nodes, func.nodes);
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
                    call_node_output_indices
                        .clone()
                        .map(|call_output_idx| {
                            let call_output =
                                func.nodes[call_site.func_call_node].outputs[call_output_idx];
                            let ty = func.vars[call_output].ty;
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
            let next_state_switch_def = &mut func.nodes[next_state_switch];
            next_state_switch_def.kind = NodeKind::Select(SelectionKind::Switch {
                case_consts: next_state_switch_case_consts,
            });
            next_state_switch_def.inputs = [next_state].into_iter().collect();
            next_state_switch_def.child_regions = next_state_switch_cases;

            // FIXME(eddyb) automate this (insertion cursor?).
            let state_machine_loop_repeat_cond = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: cx.intern(scalar::Type::Bool),
                    def_parent: Either::Right(next_state_switch),
                    def_idx: next_state_switch_def.outputs.len().try_into().unwrap(),
                },
            );
            next_state_switch_def.outputs.push(state_machine_loop_repeat_cond);

            Value::Var(state_machine_loop_repeat_cond)
        };

        let state_machine_loop_node = func.nodes.define(
            cx,
            NodeDef {
                attrs: AttrSet::default(),
                kind: NodeKind::Loop { repeat_condition: state_machine_loop_repeat_cond },
                inputs: [initial_state].into_iter().collect(),
                child_regions: [state_machine_loop_body].into_iter().collect(),
                outputs: [].into_iter().collect(),
            }
            .into(),
        );

        // HACK(eddyb) this is the juggling mentioned in the above comment, to
        // get the `loop` where the call node (now a `switch`) used to be, and
        // add `switch` node to the body of the `loop`.
        {
            let parent_region_children = &mut func.regions[call_site.parent_region].children;
            parent_region_children.insert_before(
                state_machine_loop_node,
                call_site.func_call_node,
                func.nodes,
            );
            parent_region_children.remove(call_site.func_call_node, func.nodes);

            assert!(call_site.func_call_node == next_state_switch);
            func.regions[state_machine_loop_body]
                .children
                .insert_last(next_state_switch, func.nodes);
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
    body_stack: SmallVec<[Region; 4]>,
}

// HACK(eddyb) state indices are signed, so that negative states can be used to
// encode failure modes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct EmuStateIdx(NonZeroI32);

impl EmuStateIdx {
    const TYPE: scalar::Type = scalar::Type::S32;

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
        scalar::Const::int_try_from_i128(Self::TYPE, self.0.get().into()).unwrap()
    }

    fn to_value(self, cx: &Context) -> Value {
        Value::Const(cx.intern(self.as_scalar()))
    }
}

#[derive(Default)]
struct FuncEmuStates {
    for_region: EntityOrientedDenseMap<Region, RegionEmuStates>,
    for_node: EntityOrientedDenseMap<Node, NodeEmuStates>,
}

/// Indicates a `Region` requiring state-splitting (due to its children),
/// and may include additional helper state(s) where necessary.
//
// FIXME(eddyb) better names/organization?
#[derive(Copy, Clone)]
struct RegionEmuStates {
    /// Only used for function bodies (i.e. as the target of a call) and
    /// loop bodies (i.e. as the target of a backedge).
    entry_state: Option<EmuStateIdx>,
}

/// Indicates a `Node` requiring state-splitting (an emulated `FuncCall`,
/// or due to its children), and includes additional helper state(s).
//
// FIXME(eddyb) better names/organization?
#[derive(Copy, Clone)]
struct NodeEmuStates {
    /// The "continuation" (or "exit"), into the parent `Region`, of this
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
    fn visit_global_var_use(&mut self, _: GlobalVar) {}
    fn visit_func_use(&mut self, _: Func) {}

    fn visit_region_def(&mut self, func_at_region: FuncAt<'_, Region>) {
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
    fn visit_node_def(&mut self, func_at_node: FuncAt<'_, Node>) {
        let node_def = func_at_node.def();
        let needs_merge_state = match node_def.kind {
            NodeKind::Loop { .. } => {
                self.body_stack.push(node_def.child_regions[0]);
                false
            }
            // FIXME(eddyb) include indirect calls in this as well!
            DataInstKind::FuncCall(callee) => {
                self.func_emu_summary[callee].emu_group == Some(self.emu_group)
            }
            _ => false,
        };

        if self.any_states_reserved_during(|this| func_at_node.inner_visit_with(this))
            | needs_merge_state
        {
            let states = NodeEmuStates { merge: self.reserve_state() };
            assert!(self.states.for_node.insert(func_at_node.position, states).is_none());
        }
    }
}

struct EmuGlobalStack<'a> {
    cx: Rc<Context>,

    config: &'a CallStackEmuConfig,

    layout_cache: crate::mem::layout::LayoutCache<'a>,

    // HACK(eddyb) currently always a cached `qptr` type.
    type_of_stack_ptr: Type,

    // HACK(eddyb) currently always a cached `u32` type.
    type_of_stack_top: Type,

    /// Global that directly contains the per-invocation emulated stack contents
    /// (likely lifted to an array of `stack_unit_bytes`-sized elements).
    stack_array_global: GlobalVar,
    ptr_to_stack_array_global: Const,

    /// Global (of integer type `type_of_stack_top`) holding the offset
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
        let mut invocation_local_global_and_ptr_to = |size, initializer| {
            let align = config.stack_unit_bytes.get();
            assert_eq!(size % align, 0);
            let global = module.global_vars.define(
                &cx,
                GlobalVarDecl {
                    attrs: AttrSet::default(),
                    type_of_ptr_to: qptr_ty,
                    shape: Some(crate::mem::shapes::GlobalVarShape::UntypedData(
                        crate::mem::shapes::MemLayout { align, legacy_align: align, size },
                    )),
                    addr_space: invocation_local_addr_space,
                    def: DeclDef::Present(GlobalVarDefBody { initializer }),
                },
            );
            (
                global,
                cx.intern(ConstDef {
                    attrs: AttrSet::default(),
                    ty: qptr_ty,
                    kind: ConstKind::PtrToGlobalVar(global),
                }),
            )
        };

        let (stack_array_global, ptr_to_stack_array_global) =
            invocation_local_global_and_ptr_to(config.stack_size_bytes, None);

        let scalar_type_of_stack_top = scalar::Type::U32;
        let (stack_top_global, ptr_to_stack_top_global) = invocation_local_global_and_ptr_to(
            scalar_type_of_stack_top.bit_width() / 8,
            Some(GlobalVarInit::Direct(cx.intern(scalar::Const::from_bits(
                scalar_type_of_stack_top,
                (config.stack_size_bytes / config.stack_unit_bytes).into(),
            )))),
        );

        Self {
            cx: cx.clone(),

            config,

            layout_cache: crate::mem::layout::LayoutCache::new(cx.clone(), &config.layout_config),

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
        let stack_top_initial_inst = func.nodes.define(
            &self.cx,
            DataInstDef {
                attrs: AttrSet::default(),
                kind: DataInstKind::Mem(MemOp::Load { offset: None }),
                inputs: [Value::Const(self.ptr_to_stack_top_global)].into_iter().collect(),
                child_regions: [].into_iter().collect(),
                outputs: [].into_iter().collect(),
            }
            .into(),
        );

        // FIXME(eddyb) automate this (insertion cursor?).
        let stack_top_initial = func.vars.define(
            &self.cx,
            VarDecl {
                attrs: Default::default(),
                ty: self.type_of_stack_top,
                def_parent: Either::Right(stack_top_initial_inst),
                def_idx: 0,
            },
        );
        func.nodes[stack_top_initial_inst].outputs.push(stack_top_initial);

        let stack_ptr_inst = func.nodes.define(
            &self.cx,
            DataInstDef {
                attrs: AttrSet::default(),
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
                inputs: [
                    Value::Const(self.ptr_to_stack_array_global),
                    Value::Var(stack_top_initial),
                ]
                .into_iter()
                .collect(),
                child_regions: [].into_iter().collect(),
                outputs: [].into_iter().collect(),
            }
            .into(),
        );

        // FIXME(eddyb) automate this (insertion cursor?).
        let stack_ptr = func.vars.define(
            &self.cx,
            VarDecl {
                attrs: Default::default(),
                ty: self.type_of_stack_ptr,
                def_parent: Either::Right(stack_ptr_inst),
                def_idx: 0,
            },
        );
        func.nodes[stack_ptr_inst].outputs.push(stack_ptr);

        EmuStackPusherPopper {
            global_stack: self,
            stack_top_initial_inst,
            stack_top_initial,
            stack_ptr_inst,
            stack_ptr,
            push_pop_insts: EntityList::empty(),
            offset_in_stack_units: 0,
            accessed_stack_unit_offsets: 0..0,
        }
    }

    // FIXME(eddyb) is this the right API?
    // TODO(eddyb) document as building `λ(...inputs). cont_body`.
    fn collect_cont_closure(
        &self,
        func: FuncAtMut<'_, ()>,
        origin: Either<(Region, RegionEmuStates), (Node, NodeEmuStates)>,
        cont_body: EmuContBody,
    ) -> (EmuContClosure, EmuContBody) {
        self.collect_cont_closure_with_collector_access(func, origin, cont_body, |_, _| {})
    }

    // TODO(eddyb) document like `collect_cont_closure` while allowing self-invocation.
    fn collect_cont_closure_with_collector_access(
        &self,
        mut func: FuncAtMut<'_, ()>,
        origin: Either<(Region, RegionEmuStates), (Node, NodeEmuStates)>,
        mut cont_body: EmuContBody,
        // FIXME(eddyb) this is only used by loops for self-invocation, find some
        // better way (builder pattern?) to access this API.
        access_collector: impl FnOnce(FuncAtMut<'_, ()>, &mut EmuContClosureCollector<'_, '_>),
    ) -> (EmuContClosure, EmuContBody) {
        // FIXME(eddyb) implement closing over locals, ideally handled as a
        // "stack frame" (just above continuation captures, and only popped
        // by the whole function returning, not any sooner - that means it has
        // to be handled at the whole-function level, where returns are handled).
        let first_func_local = func
            .reborrow()
            .freeze()
            .at(cont_body.children)
            .into_iter()
            .next()
            .filter(|func_at_inst| {
                matches!(func_at_inst.def().kind, DataInstKind::Mem(MemOp::FuncLocalVar(_)))
            })
            .map(|func_at_inst| func_at_inst.position);
        if let Some(first_func_local) = first_func_local {
            func.reborrow().at(first_func_local).def().attrs.push_diag(
                &self.cx,
                Diag::bug(["locals NYI in emulated recursive call stack".into()]),
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
            defined_vars: EntityOrientedDenseMap::new(),
        };

        // HACK(eddyb) guarantee the first `cont.input_count` pops.
        for i in 0..collector.closure.input_count {
            assert_eq!(collector.pops_of_inputs_and_captures.len(), i);
            let v = Value::Var(origin.either(
                |(region, _)| func.regions[region].inputs[i],
                |(node, _)| func.nodes[node].outputs[i],
            ));
            match collector.transform_value_use_in_func(func.reborrow().at(v)) {
                Transformed::Unchanged => unreachable!(),
                Transformed::Changed(new) => {
                    assert!(new == Value::Var(collector.pops_of_inputs_and_captures[i]));
                }
            }
        }
        assert_eq!(collector.pops_of_inputs_and_captures.len(), collector.closure.input_count);
        assert!(collector.closure.captures.is_empty());

        func.reborrow()
            .at(cont_body.children)
            .into_iter()
            .inner_in_place_transform_with(&mut collector);

        access_collector(func.reborrow(), &mut collector);

        if let Some(popper) = collector.popper.take() {
            let pops_nodes = popper.finish(func.reborrow());
            cont_body.children.prepend(pops_nodes, func.nodes);
        }

        (collector.closure, cont_body)
    }

    // FIXME(eddyb) is this the right API?
    // TODO(eddyb) document as building `λ(). cont(...inputs)`.
    fn invoke_cont_closure(
        &self,
        mut func: FuncAtMut<'_, ()>,
        cont: &EmuContClosure,
        inputs: &[Value],
    ) -> EmuContBody {
        assert_eq!(inputs.len(), cont.input_count);

        let values_in_pop_order =
            inputs.iter().copied().chain(cont.captures.iter().map(|&v| Value::Var(v)));
        let values_in_push_order = values_in_pop_order.rev();

        let mut pusher = self.pusher(func.reborrow());
        for v in values_in_push_order {
            pusher.push(func.reborrow(), v);
        }
        let (children, next_state_after) =
            pusher.finish_for_state(func, cont.entry_state_value(&self.cx));

        EmuContBody { children, next_state_after }
    }
}

// FIXME(eddyb) find a better name for this abstraction.
struct EmuStackPusherPopper<
    'a,
    'b,
    // HACK(eddyb) this only exists to avoid having both stack overflow checks
    // (i.e. from pushes), and value definitions (i.e. pops) at the same time,
    // as it would need nesting an arbitrary user `Region` and plumbing
    // its outputs (so that it can access the popped values in the first place).
    const CAN_PUSH: bool,
> {
    global_stack: &'b EmuGlobalStack<'a>,

    // FIXME(eddyb) remove the non-`Var` fields.
    stack_top_initial_inst: DataInst,
    stack_top_initial: Var,
    stack_ptr_inst: DataInst,
    stack_ptr: Var,

    /// `DataInst`s for pushes (i.e. `mem.store`) and pops (i.e `mem.load`),
    /// without any e.g. global manipulation helper instructions.
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

        let inst = func.nodes.define(
            cx,
            DataInstDef {
                attrs,
                kind: DataInstKind::Mem(MemOp::Store {
                    offset: NonZeroI32::new(
                        self.offset_in_stack_units
                            .checked_mul(
                                self.global_stack.config.stack_unit_bytes.get().try_into().unwrap(),
                            )
                            .unwrap(),
                    ),
                }),
                inputs: [Value::Var(self.stack_ptr), v].into_iter().collect(),
                child_regions: [].into_iter().collect(),
                outputs: [].into_iter().collect(),
            }
            .into(),
        );
        self.push_pop_insts.insert_last(inst, func.nodes);
    }
}

impl EmuStackPusherPopper<'_, '_, /*CAN_PUSH=*/ false> {
    fn pop(&mut self, func: FuncAtMut<'_, ()>, ty: Type) -> Var {
        let cx = &self.global_stack.cx;

        let mut attrs = AttrSet::default();
        let size_in_stack_units = self
            .global_stack
            .size_of_for_stack_in_stack_units(ty, "popping state from emulated stack")
            .map_err(|err| {
                attrs.push_diag(cx, err);
            });
        let inst = func.nodes.define(
            cx,
            DataInstDef {
                attrs,
                kind: DataInstKind::Mem(MemOp::Load {
                    offset: NonZeroI32::new(
                        self.offset_in_stack_units
                            .checked_mul(
                                self.global_stack.config.stack_unit_bytes.get().try_into().unwrap(),
                            )
                            .unwrap(),
                    ),
                }),
                inputs: [Value::Var(self.stack_ptr)].into_iter().collect(),
                child_regions: [].into_iter().collect(),
                outputs: [].into_iter().collect(),
            }
            .into(),
        );

        // FIXME(eddyb) automate this (insertion cursor?).
        let output_var = func.vars.define(
            cx,
            VarDecl { attrs: Default::default(), ty, def_parent: Either::Right(inst), def_idx: 0 },
        );
        func.nodes[inst].outputs.push(output_var);

        self.push_pop_insts.insert_last(inst, func.nodes);

        self.accessed_stack_unit_offsets.start =
            self.accessed_stack_unit_offsets.start.min(self.offset_in_stack_units);
        self.offset_in_stack_units = self
            .offset_in_stack_units
            .checked_add(size_in_stack_units.map_or(0, |size| size.get()).try_into().unwrap())
            .unwrap();
        self.accessed_stack_unit_offsets.end =
            self.accessed_stack_unit_offsets.end.max(self.offset_in_stack_units);

        output_var
    }

    // HACK(eddyb) popping doesn't need to worry about stack overflows.
    fn finish(self, func: FuncAtMut<'_, ()>) -> EntityList<Node> {
        let dummy_in = EmuStateIdx::UNKNOWN_STATE.to_value(&self.global_stack.cx);
        let (nodes, dummy_out) = self.finish_for_state(func, dummy_in);
        assert!(dummy_out == dummy_in);
        nodes
    }
}

impl<const CAN_PUSH: bool> EmuStackPusherPopper<'_, '_, CAN_PUSH> {
    // HACK(eddyb) if stack overflows are possible (i.e. through pushes), the
    // `Value` returned will (dynamically) be a choice between `next_state` or
    // `EmuStateIdx::STACK_OVERFLOW`, and the caller needs to rely on it.
    // FIXME(eddyb) should this return `EmuContBody`?
    fn finish_for_state(
        self,
        mut func: FuncAtMut<'_, ()>,
        next_state: Value,
    ) -> (EntityList<Node>, Value) {
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
            .insert_first(self.stack_ptr_inst, func.nodes);
        pre_check_insts.insert_first(self.stack_top_initial_inst, func.nodes);

        let mut mk_stack_top_plus = |offset_in_stack_units: i32| {
            let inst =
                func.nodes.define(
                    cx,
                    DataInstDef {
                        attrs: AttrSet::default(),
                        kind: DataInstKind::Scalar(scalar::Op::IntBinary(
                            if offset_in_stack_units < 0 {
                                scalar::IntBinOp::Sub
                            } else {
                                scalar::IntBinOp::Add
                            },
                        )),
                        inputs: [
                            Value::Var(self.stack_top_initial),
                            Value::Const(cx.intern(scalar::Const::from_u32(
                                offset_in_stack_units.unsigned_abs(),
                            ))),
                        ]
                        .into_iter()
                        .collect(),
                        child_regions: [].into_iter().collect(),
                        outputs: [].into_iter().collect(),
                    }
                    .into(),
                );

            // FIXME(eddyb) automate this (insertion cursor?).
            let output_var = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: self.global_stack.type_of_stack_top,
                    def_parent: Either::Right(inst),
                    def_idx: 0,
                },
            );
            func.nodes[inst].outputs.push(output_var);

            if stack_overflow_checked_insts.is_some() {
                pre_check_insts.insert_last(inst, func.nodes);
            } else {
                pre_check_insts.insert_before(inst, self.stack_ptr_inst, func.nodes);
            }

            Value::Var(output_var)
        };

        let stack_top_plus_min_neg_offset = min_neg_offset.map(&mut mk_stack_top_plus);
        let stack_top_plus_final_offset = if Some(final_offset) == min_neg_offset {
            // HACK(eddyb) reuse identical instruction (common case for push-only).
            stack_top_plus_min_neg_offset.unwrap()
        } else {
            mk_stack_top_plus(final_offset)
        };

        // HACK(eddyb) adjust `self.stack_ptr` so that all offsets of pushes'
        // `mem.store`s are positive (before negative offsets are handled).
        if let Some(stack_top_plus_min_neg_offset) = stack_top_plus_min_neg_offset {
            let stack_ptr_idx_input = &mut func.reborrow().at(self.stack_ptr_inst).def().inputs[1];
            assert!(*stack_ptr_idx_input == Value::Var(self.stack_top_initial));
            *stack_ptr_idx_input = stack_top_plus_min_neg_offset;

            let offset_delta = (-min_neg_offset.unwrap())
                .checked_mul(self.global_stack.config.stack_unit_bytes.get().try_into().unwrap())
                .unwrap();

            let mut func_at_insts =
                func.reborrow().at(stack_overflow_checked_insts.unwrap()).into_iter();
            assert!(func_at_insts.next().unwrap().position == self.stack_ptr_inst);
            while let Some(func_at_inst) = func_at_insts.next() {
                let inst_def = func_at_inst.def();
                match &mut inst_def.kind {
                    DataInstKind::Mem(MemOp::Load { offset } | MemOp::Store { offset }) => {
                        *offset = NonZeroI32::new(
                            offset.map_or(0, |o| o.get()).checked_add(offset_delta).unwrap(),
                        );
                    }
                    _ => unreachable!(),
                }
                assert!(inst_def.inputs[0] == Value::Var(self.stack_ptr));
            }
        }

        let stack_top_store = func.nodes.define(
            cx,
            DataInstDef {
                attrs: AttrSet::default(),
                kind: DataInstKind::Mem(MemOp::Store { offset: None }),
                inputs: [
                    Value::Const(self.global_stack.ptr_to_stack_top_global),
                    stack_top_plus_final_offset,
                ]
                .into_iter()
                .collect(),
                child_regions: [].into_iter().collect(),
                outputs: [].into_iter().collect(),
            }
            .into(),
        );
        stack_overflow_checked_insts
            .as_mut()
            .unwrap_or(&mut pre_check_insts)
            .insert_last(stack_top_store, func.nodes);

        let mut all_nodes = pre_check_insts;

        let Some(checked_insts) = stack_overflow_checked_insts else {
            return (all_nodes, next_state);
        };

        let would_overflow_stack_node = func.nodes.define(
            cx,
            DataInstDef {
                attrs: AttrSet::default(),
                kind: DataInstKind::Scalar(scalar::Op::IntBinary(scalar::IntBinOp::GtU)),
                inputs: [
                    stack_top_plus_min_neg_offset.unwrap(),
                    Value::Var(self.stack_top_initial),
                ]
                .into_iter()
                .collect(),
                child_regions: [].into_iter().collect(),
                outputs: [].into_iter().collect(),
            }
            .into(),
        );

        // FIXME(eddyb) automate this (insertion cursor?).
        let would_overflow_stack = func.vars.define(
            cx,
            VarDecl {
                attrs: Default::default(),
                ty: cx.intern(scalar::Type::Bool),
                def_parent: Either::Right(would_overflow_stack_node),
                def_idx: 0,
            },
        );
        func.nodes[would_overflow_stack_node].outputs.push(would_overflow_stack);

        all_nodes.insert_last(would_overflow_stack_node, func.nodes);

        let cases =
            [(None, EmuStateIdx::STACK_OVERFLOW.to_value(cx)), (Some(checked_insts), next_state)]
                .into_iter()
                .map(|(insts, output)| {
                    func.regions.define(
                        cx,
                        RegionDef {
                            inputs: [].into_iter().collect(),
                            children: insts.unwrap_or_default(),
                            outputs: [output].into_iter().collect(),
                        },
                    )
                })
                .collect();

        let check_node = func.nodes.define(
            cx,
            NodeDef {
                attrs: AttrSet::default(),
                kind: NodeKind::Select(SelectionKind::BoolCond),
                inputs: [Value::Var(would_overflow_stack)].into_iter().collect(),
                child_regions: cases,
                outputs: [].into_iter().collect(),
            }
            .into(),
        );

        // FIXME(eddyb) automate this (insertion cursor?).
        let check_output_var = func.vars.define(
            cx,
            VarDecl {
                attrs: Default::default(),
                ty: cx.intern(EmuStateIdx::TYPE),
                def_parent: Either::Right(check_node),
                def_idx: 0,
            },
        );
        func.nodes[check_node].outputs.push(check_output_var);

        all_nodes.insert_last(check_node, func.nodes);

        (all_nodes, Value::Var(check_output_var))
    }
}

struct EmuContClosureCollector<'a, 'b> {
    global_stack: &'b EmuGlobalStack<'a>,
    popper: Option<EmuStackPusherPopper<'a, 'b, false>>,

    /// The continuation whose `captures` are being collected.
    closure: EmuContClosure,

    /// Outputs of `mem.load`s (each added to `pops_block`) popping all inputs
    /// (`cont.input_count`) and all captures (keyed/ordered by `cont.captures`,
    /// after the inputs) from the stack, ordered by increasing stack offset
    /// (from the stack top upwards, i.e. towards earlier pushed values).
    ///
    /// Note that pushing has to be done in reverse (moving the stack top downwards),
    /// though this is mainly relevant when inputs are pushed separately, which
    /// is why inputs are popped first (`0..cont.input_count`), and pushed last.
    pops_of_inputs_and_captures: Vec<Var>,

    // HACK(eddyb) efficient tracking to allow determining if a `Value` is part
    // of the continuation itself, or a capture (see `popped_values`).
    // FIXME(eddyb) there should be a (sparse) bitset version of this.
    defined_vars: EntityOrientedDenseMap<Var, ()>,
}

/// The "closure" of an emulated continuation, which includes everything needed
/// to invoke it (i.e. pushing `captures` and inputs to the stack).
struct EmuContClosure {
    /// The reason for this continuation to exist, including the `EmuStateIdx`
    /// associated with it (for `RegionEmuStates`, `entry_state` must be `Some`).
    //
    // HACK(eddyb) the `Result` around the whole thing allows "dynamic" states,
    // but the whole abstraction should be refactored.
    origin: Result<Either<(Region, RegionEmuStates), (Node, NodeEmuStates)>, Value>,
    /// The count of, depending on `origin`:
    /// - when `(Region, _)`: all region (i.e. func/loop body) inputs
    /// - when `(Node, _)`: all node (i.e. being merged) outputs
    input_count: usize,
    /// Set of `Value`s defined outside of the continuation, requiring each a
    /// stack pop in the continuation, and a matching push on the other side.
    //
    // FIXME(eddyb) there should be a (sparse) bitset version of this.
    captures: FxIndexSet<Var>,
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
/// the respective [`EmuContClosure`], i.e. a `Region` with:
/// - no inputs
/// - child `Node`s (including all necessary stack manipulation)
/// - one output: `next_state_after` (see also its documentation)
struct EmuContBody {
    children: EntityList<Node>,

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
    fn into_region_def(self) -> RegionDef {
        let EmuContBody { children, next_state_after } = self;
        RegionDef {
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

        let Value::Var(v) = v else {
            return Transformed::Unchanged;
        };

        let already_valid = self.defined_vars.get(v).is_some();
        if already_valid {
            return Transformed::Unchanged;
        }

        let origin_region_or_node =
            self.closure.origin.ok().unwrap().map_either(|(region, _)| region, |(node, _)| node);
        let cont_input_idx = match (origin_region_or_node, func.vars[v].kind()) {
            (Either::Left(r1), VarKind::RegionInput { region: r2, input_idx }) if r1 == r2 => {
                Some(input_idx)
            }
            (Either::Right(n1), VarKind::NodeOutput { node: n2, output_idx }) if n1 == n2 => {
                Some(output_idx)
            }
            _ => None,
        };

        let pop_idx = match cont_input_idx {
            Some(input_idx) => input_idx.try_into().unwrap(),
            None => self.closure.input_count + self.closure.captures.insert_full(v).0,
        };

        if let Some(&v) = self.pops_of_inputs_and_captures.get(pop_idx) {
            // Already seen (i.e. effectively cached).
            return Transformed::Changed(Value::Var(v));
        }

        // Generate the pop (and scaffolding, if not yet present).
        assert_eq!(pop_idx, self.pops_of_inputs_and_captures.len());

        let ty = func.vars[v].ty;
        let popped_value = self
            .popper
            .get_or_insert_with(|| self.global_stack.popper(func.reborrow()))
            .pop(func.reborrow(), ty);
        self.pops_of_inputs_and_captures.push(popped_value);
        Transformed::Changed(Value::Var(popped_value))
    }

    fn in_place_transform_region_def(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        for &input_var in &func_at_region.reborrow().def().inputs {
            self.defined_vars.insert(input_var, ());
        }
        func_at_region.inner_in_place_transform_with(self);
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        func_at_node.inner_in_place_transform_with(self);
        for &output_var in &func_at_node.def().outputs {
            self.defined_vars.insert(output_var, ());
        }
    }
}

struct EmuFuncFracker<'a> {
    global_stack: &'a EmuGlobalStack<'a>,

    states: &'a FuncEmuStates,
    func_call_emu_cont: &'a FxIndexMap<Func, EmuContClosure>,

    state_switch_cases: BTreeMap<EmuStateIdx, Region>,
}

// FIXME(eddyb) reconsider name or at least document?
impl EmuFuncFracker<'_> {
    fn frack_cont_body_nodes_as_needed(
        &mut self,
        mut func: FuncAtMut<'_, ()>,
        mut cont_body: EmuContBody,
    ) -> EmuContBody {
        let cx = &self.global_stack.cx;

        let mut children = mem::take(&mut cont_body.children);
        while let Some(node) = children.remove_last(func.nodes) {
            if let Some(&node_states) = self.states.for_node.get(node) {
                let (merge_cont, merge_cont_body) = self.global_stack.collect_cont_closure(
                    func.reborrow(),
                    Either::Right((node, node_states)),
                    cont_body,
                );
                assert_eq!(merge_cont.entry_state_idx().ok().unwrap(), node_states.merge);

                self.state_switch_cases.insert(
                    node_states.merge,
                    func.regions.define(cx, merge_cont_body.into_region_def()),
                );

                cont_body = self.frack_node(func.reborrow().at(node), merge_cont);
            } else {
                cont_body.children.insert_first(node, func.nodes);
            }
        }
        cont_body
    }

    // FIXME(eddyb) turn the below comment into proper docs:
    // HACK(eddyb) returns one of:
    // - `None`: `region` is a `Select` case, remains in-place, and outputs state
    // - `Some(entry_cont)`: `states.for_region[region].entry_state` is `Some`,
    //   i.e. `region` is a func/loop body, and was moved to the state `switch`,
    //   so it can only be reached by invoking `entry_cont`
    // TODO(eddyb) deal with the fact that this has only two callers, and only
    // one of them can even have `states.for_region[region]` be `None`.
    fn frack_region_as_needed(
        &mut self,
        func_at_region: FuncAtMut<'_, Region>,
        invoke_merge: impl FnOnce(
            &EmuGlobalStack<'_>,
            FuncAtMut<'_, ()>,
            EntityList<Node>,
            &[Value],
        ) -> EmuContBody,
    ) -> Option<EmuContClosure> {
        let region = func_at_region.position;
        let mut func = func_at_region.at(());

        let cont_body = {
            let region_def = &mut func.regions[region];
            let children = mem::take(&mut region_def.children);
            let outputs = mem::take(&mut region_def.outputs);
            invoke_merge(self.global_stack, func.reborrow(), children, &outputs)
        };

        let Some(&region_states) = self.states.for_region.get(region) else {
            let region_def = &mut func.regions[region];
            assert_eq!(region_def.inputs.len(), 0);
            *region_def = cont_body.into_region_def();
            return None;
        };

        let cont_body = self.frack_cont_body_nodes_as_needed(func.reborrow(), cont_body);

        let (maybe_entry_cont, cont_body) = if let Some(entry_state) = region_states.entry_state {
            let (entry_cont, entry_cont_body) = self.global_stack.collect_cont_closure(
                func.reborrow(),
                Either::Left((region, region_states)),
                cont_body,
            );
            assert_eq!(entry_cont.entry_state_idx().ok().unwrap(), entry_state);

            self.state_switch_cases.insert(entry_state, region);

            (Some(entry_cont), entry_cont_body)
        } else {
            assert_eq!(func.regions[region].inputs.len(), 0);
            (None, cont_body)
        };
        func.regions[region] = cont_body.into_region_def();

        maybe_entry_cont
    }

    fn frack_node(
        &mut self,
        func_at_node: FuncAtMut<'_, Node>,
        merge: EmuContClosure,
    ) -> EmuContBody {
        let cx = &self.global_stack.cx;

        let node = func_at_node.position;
        let mut func = func_at_node.at(());

        // FIXME(eddyb) cache this somewhere.
        let state_ty = cx.intern(EmuStateIdx::TYPE);

        let node_def = &mut func.nodes[node];
        match &node_def.kind {
            NodeKind::ExitInvocation { .. } => {
                unreachable!()
            }
            NodeKind::Select(_) => {
                for case_idx in 0..node_def.child_regions.len() {
                    let case = func.nodes[node].child_regions[case_idx];

                    let case_entry_cont = self.frack_region_as_needed(
                        func.reborrow().at(case),
                        |global_stack, mut func, children, case_outputs| {
                            let mut cont_body = global_stack.invoke_cont_closure(
                                func.reborrow(),
                                &merge,
                                case_outputs,
                            );
                            cont_body.children.prepend(children, func.nodes);
                            cont_body
                        },
                    );
                    assert!(case_entry_cont.is_none());
                }

                let mut children = EntityList::empty();
                children.insert_last(node, func.nodes);

                // FIXME(eddyb) automate this (insertion cursor?).
                let output_var = func.vars.define(
                    cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: state_ty,
                        def_parent: Either::Right(node),
                        def_idx: 0,
                    },
                );
                func.nodes[node].outputs = [output_var].into_iter().collect();

                EmuContBody { children, next_state_after: Value::Var(output_var) }
            }
            &NodeKind::Loop { repeat_condition } => {
                let body = node_def.child_regions[0];

                let body_states = self.states.for_region[body];
                let body_entry_state = body_states.entry_state.unwrap();

                // Structured loops are equivalent to their `body` being trailed
                // by an implied conditional backedge, akin to an unstructured
                // `if repeat { goto body(...body.outputs) } else { goto merge }`,
                // so *correctly* closure-converting `body` needs it to already
                // contain both the "backedge" (`body`) and "break" (`merge`)
                // targets (or rather, `body.outputs` and `merge.captures` must
                // *both* be present within `body`, to be accurately tracked).
                //
                // HACK(eddyb) the above problem cannot be solved by "simply"
                // invoking `body_entry_cont` from `body`, as that would
                // would make closure-converting `body` cyclically depend on its
                // own result (yet another way in which loops act like "fixpoints"),
                // so this first* injects (into `body`) *invalid* SPIR-T like
                // `if repeat {} else { invoke merge }` (i.e. a `Select` with
                // mismatched output arity, but accounting for `merge.captures`),
                // then uses `collect_cont_closure_with_collector_access` to be
                // able to invoke `body_entry_cont` *before* it's completed,
                // and while being able to access the capture collector, so that
                // the correct value replacements are performed just before that
                // self-invocation is injected into `backedge_region`, and all
                // that is over before ever fracking any of `body`'s nodes.
                let [backedge_region, merge_region] = [
                    RegionDef::default(),
                    self.global_stack
                        .invoke_cont_closure(func.reborrow(), &merge, &[])
                        .into_region_def(),
                ]
                .map(|def| func.regions.define(cx, def));
                let whole_body_cont_body = {
                    let cb_node = func.nodes.define(
                        cx,
                        NodeDef {
                            attrs: AttrSet::default(),
                            kind: NodeKind::Select(SelectionKind::BoolCond),
                            inputs: [repeat_condition].into_iter().collect(),
                            child_regions: [backedge_region, merge_region].into_iter().collect(),
                            outputs: [].into_iter().collect(),
                        }
                        .into(),
                    );

                    // FIXME(eddyb) automate this (insertion cursor?).
                    let cb_output_var = func.vars.define(
                        cx,
                        VarDecl {
                            attrs: Default::default(),
                            ty: state_ty,
                            def_parent: Either::Right(cb_node),
                            def_idx: 0,
                        },
                    );
                    func.nodes[cb_node].outputs = [cb_output_var].into_iter().collect();

                    let mut children = mem::take(&mut func.regions[body].children);
                    children.insert_last(cb_node, func.nodes);

                    EmuContBody { children, next_state_after: Value::Var(cb_output_var) }
                };
                let (body_entry_cont, whole_body_cont_body) =
                    self.global_stack.collect_cont_closure_with_collector_access(
                        func.reborrow(),
                        Either::Left((body, body_states)),
                        whole_body_cont_body,
                        |mut func, collector| {
                            let body_outputs = mem::take(&mut func.regions[body].outputs);

                            // FIXME(eddyb) move all this into a method on `collector`.
                            for &v in &body_outputs {
                                // HACK(eddyb) ensure captures consider outputs,
                                // but discard the transformation result, as all
                                // of these values will be seen as part of the
                                // self-invocation, later transformed below.
                                let _ =
                                    collector.transform_value_use_in_func(func.reborrow().at(v));
                            }
                            let frozen_capture_count = collector.closure.captures.len();
                            func.regions[backedge_region] = self
                                .global_stack
                                .invoke_cont_closure(
                                    func.reborrow(),
                                    &collector.closure,
                                    &body_outputs,
                                )
                                .into_region_def();
                            collector.in_place_transform_region_def(func.at(backedge_region));
                            // FIXME(eddyb) there should be a field on `collector`
                            // that contains an `Option<usize>` and performs this
                            // check every time it rewrites any `Value`s.
                            assert_eq!(collector.closure.captures.len(), frozen_capture_count);
                        },
                    );

                // HACK(eddyb) this is done *after* closure-converting `body`
                // (via `collect_cont_closure_with_collector_access`), so that
                // the self-invocation performed above is *already* aware of
                // any captures needed *anywhere* inside the `body`, before
                // fracking its child nodes may split it into separate regions.
                func.regions[body] = self
                    .frack_cont_body_nodes_as_needed(func.reborrow(), whole_body_cont_body)
                    .into_region_def();

                // HACK(eddyb) this used to be part of `frack_region_as_needed`.
                {
                    assert_eq!(body_entry_cont.entry_state_idx().ok().unwrap(), body_entry_state);
                    self.state_switch_cases.insert(body_entry_state, body);
                }
                let loop_initial_inputs = mem::take(&mut func.nodes[node].inputs);
                self.global_stack.invoke_cont_closure(func, &body_entry_cont, &loop_initial_inputs)
            }

            // TODO(eddyb) use `invoke_cont_closure` here.
            &DataInstKind::FuncCall(callee) => {
                let callee_entry_cont = &self.func_call_emu_cont[&callee];

                let inputs = mem::take(&mut node_def.inputs);
                assert_eq!(inputs.len(), callee_entry_cont.input_count);
                assert!(callee_entry_cont.captures.is_empty());

                // FIXME(eddyb) the `.rev()` usage here is not self-explanatory.
                let values_in_push_order = (merge.captures.iter().map(|&v| Value::Var(v)).rev())
                    .chain([merge.entry_state_value(cx)])
                    .chain(inputs.iter().copied().rev());

                let mut pusher = self.global_stack.pusher(func.reborrow());
                for v in values_in_push_order {
                    pusher.push(func.reborrow(), v);
                }
                let (pushes_nodes, next_state_after) = pusher
                    .finish_for_state(func.reborrow(), callee_entry_cont.entry_state_value(cx));

                let mut children = EntityList::empty();
                children.append(pushes_nodes, func.nodes);

                EmuContBody { children, next_state_after }
            }

            DataInstKind::Scalar(_)
            | DataInstKind::Vector(_)
            | DataInstKind::Mem(_)
            | DataInstKind::QPtr(_)
            | DataInstKind::SpvInst(..)
            | DataInstKind::SpvExtInst { .. } => unreachable!(),
        }
    }
}
