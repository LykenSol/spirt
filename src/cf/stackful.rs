//! Stackful function recursion/pointers emulation.
//
// TODO(eddyb) choose between `cf::stackful` and `cf::reentrant`,
// maybe move some of the state machine "continuation" stuff to `cf::cps`.

use crate::cf::SelectionKind;
use crate::cf::callgraph::{CallGraph, CallSite};
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
use smallvec::SmallVec;
use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::hash::Hash;
use std::mem;
use std::num::{NonZeroI32, NonZeroU32};
use std::ops::Range;
use std::rc::Rc;

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

// TODO(eddyb) rename this to `DispatchGroup`!
//
/// Potentially-recursive calls (including indirect calls that static analysis
/// cannot prove don't dynamically result in recursion) require emulating a
/// call stack (see also `CallStackEmulator`), and to that end, their callee
/// `Func`s are grouped into independent mutually-recursive cycles (SCCs),
/// each identified by their "SCC root" (i.e. the first `Func` the Tarjan SCC
/// algorithm saw in each group), which has no semantic significance, and only
/// helps in distinguishing *between* such groups.
//
// TODO(eddyb) rename this to `DispatchGroup`!
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

        let mut cycle_finder = CycleFinder::new(|func| {
            // HACK(eddyb) this is like `direct_and_indirect_calless_of`,
            // but takes into account the need to group all indirectly
            // callable functions into a single "dispatch group".

            // call_graph.direct_and_indirect_callees_of(func)
            call_graph
                .caller_to_callees
                .get(&func)
                .map(|callees| {
                    callees.direct.keys().chain(
                        (!callees.indirect.is_empty()
                            || call_graph.indirect_callees.contains(&func))
                        .then_some(
                            call_graph.indirect_callees.iter().chain(
                                // HACK(eddyb) also include indirect *callers*, hoping to reduce
                                // the impact of quasi-exponential inlining amplification.
                                // TODO(eddyb) remove or make configurable.
                                call_graph
                                    .caller_to_callees
                                    .iter()
                                    .filter(|&(&caller, callees)| {
                                        false
                                            && !callees.indirect.is_empty()
                                            && !call_graph.spv_entry_points.contains(&caller)
                                    })
                                    .map(|(caller, _)| caller),
                            ),
                        )
                        .into_iter()
                        .flatten(),
                    )
                })
                .into_iter()
                .flatten()
                .copied()
        });
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
            for func in call_graph.indirect_callees.iter().copied().chain(
                // HACK(eddyb) also include indirect *callers*, hoping to reduce
                // the impact of quasi-exponential inlining amplification.
                // TODO(eddyb) remove or make configurable.
                call_graph
                    .caller_to_callees
                    .iter()
                    .filter(|(_, callees)| !callees.indirect.is_empty() && false)
                    .map(|(&caller, _)| caller),
            ) {
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
            eprintln!("call emu groups left: {}", call_emu_groups.len());
            // FIXME(eddyb) use the cycle finder to add functions in between!!!
            // FIXME(eddyb) does that even make sense? maybe the issue is that
            // the cycle finder just doesn't take into account this whole idea
            // that there is a "dispatch" to indirect callees... hmmmm
            // maybe it's fixable with just treating any indirect callee function
            // as being able to call any other indirect callee function?
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

        // FIXME(eddyb) maybe track whether function pointer types are present,
        // to be able to skip this step?
        self.retype_func_ptrs_in_module(module);
    }

    // HACK(eddyb) replace the function pointer type with the state index type,
    // across all functions in the module, to complete function pointer emulation.
    // FIXME(eddyb) is this missing anything outside functions?
    fn retype_func_ptrs_in_module(&self, module: &mut Module) {
        let cx = &self.global_stack.cx;
        let wk = &spv::spec::Spec::get().well_known;

        let func_ptr_size = self.global_stack.config.layout_config.logical_ptr_size_align.0;
        // FIXME(eddyb) support other fn ptr sizes.
        assert_eq!(func_ptr_size * 8, EmuStateIdx::TYPE.bit_width());

        // FIXME(eddyb) cache this.
        let func_ptr_type = cx.intern(
            spv::Inst {
                opcode: wk.OpTypeUntypedPointerKHR,
                imms: [spv::Imm::Short(wk.StorageClass, wk.CodeSectionINTEL)].into_iter().collect(),
            }
            .into_canonical_type_with(cx, [].into_iter().collect()),
        );
        let state_ty: Type = cx.intern(EmuStateIdx::TYPE);

        let mut reusable_queue = VecDeque::new();
        for &func in self.call_graph.caller_to_callees.keys() {
            let func_decl = &mut module.funcs[func];
            let DeclDef::Present(func_def_body) = &mut func_decl.def else {
                // FIXME(eddyb) do imports even make sense here? still, even so,
                // the signature should probably be updated, or at least checked?
                continue;
            };

            let mut func_ptr_vars = vec![];

            // FIXME(eddyb) adopt this style of queue-based visiting in more places.
            let queue = &mut reusable_queue;
            queue.clear();
            queue.push_back(func_def_body.body);
            while let Some(region) = queue.pop_front() {
                for &input_var in &func_def_body.at(region).def().inputs {
                    if func_def_body.at(input_var).decl().ty == func_ptr_type {
                        func_ptr_vars.push(input_var);
                    }
                }
                let mut func_at_children = func_def_body.at_mut(region).at_children().into_iter();
                while let Some(mut func_at_node) = func_at_children.next() {
                    let node = func_at_node.position;
                    let func = func_at_node.reborrow().freeze();
                    let node_def = func.at(node).def();

                    let mut new_attrs = node_def.attrs;
                    let mut new_node_kind = None;

                    for (i, &input) in node_def.inputs.iter().enumerate() {
                        if func.at(input).type_of(cx) == func_ptr_type {
                            // HACK(eddyb) as long as the replacement type has
                            // the same properties (see `func_ptr_size` assert),
                            // it should be sound to change a memory access type
                            // (and direct dataflow even more so).
                            match &node_def.kind {
                                NodeKind::Mem(MemOp::Store { offset: _ }) if i == 1 => {}
                                NodeKind::FuncCall(_) => {}

                                NodeKind::SpvInst(spv_inst, _)
                                    if spv_inst.opcode == wk.OpBitcast => {}

                                // FIXME(eddyb) `qptr::legalize` handles such
                                // instructions for the "data pointer" case.
                                NodeKind::SpvInst(spv_inst, _)
                                    if spv_inst.opcode == wk.OpPtrEqual =>
                                {
                                    new_node_kind =
                                        Some(scalar::Op::IntBinary(scalar::IntBinOp::Eq).into());
                                }
                                NodeKind::SpvInst(spv_inst, _)
                                    if spv_inst.opcode == wk.OpPtrNotEqual =>
                                {
                                    new_node_kind =
                                        Some(scalar::Op::IntBinary(scalar::IntBinOp::Ne).into());
                                }

                                _ => {
                                    new_attrs.push_diag(
                                        cx,
                                        Diag::bug([
                                            "unsupported `".into(),
                                            func_ptr_type.into(),
                                            "` input (will not retype to `".into(),
                                            state_ty.into(),
                                            "`)".into(),
                                        ]),
                                    );
                                }
                            }
                        }
                    }
                    for &output_var in &node_def.outputs {
                        if func.at(output_var).decl().ty == func_ptr_type {
                            // HACK(eddyb) as long as the replacement type has
                            // the same properties (see `func_ptr_size` assert),
                            // it should be sound to change a memory access type
                            // (and direct dataflow even more so).
                            match &node_def.kind {
                                NodeKind::Mem(MemOp::Load { offset: _ })
                                | NodeKind::FuncCall(_)
                                | NodeKind::Select(_) => {}

                                NodeKind::SpvInst(spv_inst, _)
                                    if spv_inst.opcode == wk.OpBitcast => {}

                                _ => {
                                    new_attrs.push_diag(
                                        cx,
                                        Diag::bug([
                                            "unsupported `".into(),
                                            func_ptr_type.into(),
                                            "` output (will not retype to `".into(),
                                            state_ty.into(),
                                            "`)".into(),
                                        ]),
                                    );
                                    continue;
                                }
                            }

                            func_ptr_vars.push(output_var);
                        }
                    }

                    let node_def = func_at_node.def();
                    node_def.attrs = new_attrs;
                    if let Some(new_node_kind) = new_node_kind {
                        node_def.kind = new_node_kind;
                    }

                    queue.extend(node_def.child_regions.iter().copied());
                }
            }

            // FIXME(eddyb) consider gating this on there being no errors above
            // (except it's near-impossible to avoid type mismatches from errors).
            for var in func_ptr_vars {
                let var_decl = &mut func_def_body.vars[var];
                assert!(var_decl.ty == func_ptr_type);
                var_decl.ty = state_ty;
            }
            for ty in func_decl.params.iter_mut().map(|p| &mut p.ty).chain(&mut func_decl.ret_types)
            {
                if *ty == func_ptr_type {
                    *ty = state_ty;
                }
            }
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
                    indirect_callee_emu_group: self.indirect_callee_emu_group,
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

            {
                // FIXME(eddyb) reuse more `EmuStackPusherPopper` machinery here.
                let mut stack_manip = None;
                let mut last_func_local_var = None;
                let mut func_at_orig_body_children =
                    func_def_body.at_mut(orig_body).at_children().into_iter();
                while let Some(func_at_node) = func_at_orig_body_children.next() {
                    let node = func_at_node.position;
                    let mut func = func_at_node.at(());
                    let NodeKind::Mem(MemOp::FuncLocalVar(layout)) = func.nodes[node].kind else {
                        break;
                    };
                    last_func_local_var = Some(node);

                    let stack_manip = stack_manip.get_or_insert_with(|| {
                        let stack_manip = self.global_stack.pusher(func.reborrow());
                        let orig_body_children = &mut func.regions[orig_body].children;
                        for new_node in
                            [stack_manip.stack_top_initial_inst, stack_manip.stack_ptr_inst]
                        {
                            orig_body_children.insert_before(new_node, node, func.nodes);
                        }
                        stack_manip
                    });

                    let node_def = func.at(node).def();

                    // TODO(eddyb) support (by injecting a `mem.store`
                    // just after the original declaration position).
                    assert_eq!(node_def.inputs.len(), 0);

                    let offset = if layout.size == 0 {
                        0
                    } else {
                        stack_manip
                            .mem_offset_for_push(
                                layout,
                                "allocating function-local variable on emulated stack",
                            )
                            .map_err(|diag| node_def.attrs.push_diag(cx, diag))
                            .unwrap_or(0)
                    };
                    node_def.kind = QPtrOp::Offset(offset).into();
                    node_def.inputs = [Value::Var(stack_manip.stack_ptr)].into_iter().collect();
                }

                // TODO(eddyb) also check for stack overflows!
                // (might be worth using `build_fatal_error` directly, instead of
                // returning the `EmuStateIdx::STACK_OVERFLOW` state, just to
                // keep things simple?)
                if let Some(stack_manip) = stack_manip {
                    let last_func_local_var = last_func_local_var.unwrap();
                    assert!(stack_manip.push_pop_insts.is_empty());

                    let mut func = func_def_body.at_mut(());

                    // FIXME(eddyb) dedup with `EmuStackPusherPopper::finish_for_state`.
                    let final_offset = stack_manip.offset_in_stack_units;
                    let (stack_top_plus_final_offset_inst, stack_top_plus_final_offset) =
                        stack_manip
                            .stack_top_plus_offset_in_stack_units(func.reborrow(), final_offset);

                    let mut mk_stack_top_store = |offset_in_stack_units| {
                        func.nodes.define(
                            cx,
                            DataInstDef {
                                attrs: AttrSet::default(),
                                kind: DataInstKind::Mem(MemOp::Store { offset: None }),
                                inputs: [
                                    Value::Const(self.global_stack.ptr_to_stack_top_global),
                                    offset_in_stack_units,
                                ]
                                .into_iter()
                                .collect(),
                                child_regions: [].into_iter().collect(),
                                outputs: [].into_iter().collect(),
                            }
                            .into(),
                        )
                    };

                    let store_final_offset_stack_top =
                        mk_stack_top_store(stack_top_plus_final_offset);
                    let restore_original_stack_top =
                        mk_stack_top_store(Value::Var(stack_manip.stack_top_initial));

                    let orig_body_children = &mut func.regions[orig_body].children;
                    let mut prev_node = last_func_local_var;
                    for new_node in [stack_top_plus_final_offset_inst, store_final_offset_stack_top]
                    {
                        orig_body_children.insert_after(new_node, prev_node, func.nodes);
                        prev_node = new_node;
                    }
                    orig_body_children.insert_last(restore_original_stack_top, func.nodes);
                }
            }

            // TODO(eddyb) instead of popping the return state when returning,
            // consider instead adding it as an extra argument ahead of time
            // (sadly that doesn't account for captures of the ret cont, when
            // it's intra-emu-group).
            let mut func = func_def_body.at_mut(());
            let ret_cont = {
                let popped_state_var = func.vars.define(
                    &self.global_stack.cx,
                    VarDecl {
                        attrs: Default::default(),
                        ty: state_ty,
                        // HACK(eddyb) using an existing region to declare an "orphan" `Var`.
                        def_parent: Either::Left(new_body),
                        def_idx: !0,
                    },
                );

                let mut popper = self.global_stack.popper(func.reborrow());
                popper.pop_into(func.reborrow(), popped_state_var);
                let pops_nodes = popper.finish(func.reborrow());
                func.regions[orig_body].children.append(pops_nodes, func.nodes);
                EmuContClosure {
                    origin: Err(Value::Var(popped_state_var)),
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
                                None,
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
                func.regions.define(
                    cx,
                    RegionDef {
                        inputs: [].into_iter().collect(),
                        children,
                        outputs: [next_state_after].into_iter().collect(),
                    },
                )
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

        // Function pointers can now be replaced by entry states indices.
        if Some(emu_group) == self.indirect_callee_emu_group {
            // FIXME(eddyb) automate this and/or make it a `Module` wrapper.
            let all_uses_from_module = visit::AllUses::from_module(module);

            let wk = &spv::spec::Spec::get().well_known;

            // FIXME(eddyb) cache this.
            let func_ptr_type = cx.intern(
                spv::Inst {
                    opcode: wk.OpTypeUntypedPointerKHR,
                    imms: [spv::Imm::Short(wk.StorageClass, wk.CodeSectionINTEL)]
                        .into_iter()
                        .collect(),
                }
                .into_canonical_type_with(cx, [].into_iter().collect()),
            );

            // HACK(eddyb) this is copied from `qptr::legalize`, DRY it!
            let transform_const_use = |ct: Const, untyped_size: u32| -> Transformed<Const> {
                let func_ptr_size = self.global_stack.config.layout_config.logical_ptr_size_align.0;
                // FIXME(eddyb) support other fn ptr sizes.
                assert_eq!(func_ptr_size * 8, EmuStateIdx::TYPE.bit_width());

                let ct_def = &cx[ct];
                if untyped_size != func_ptr_size || ct_def.ty != func_ptr_type {
                    return Transformed::Unchanged;
                }

                let maybe_state_idx_const = match &ct_def.kind {
                    ConstKind::Undef => Some(cx.intern(ConstDef {
                        attrs: AttrSet::default(),
                        ty: cx.intern(EmuStateIdx::TYPE),
                        kind: ConstKind::Undef,
                    })),

                    ConstKind::SpvInst { spv_inst_and_const_inputs }
                        if {
                            // FIXME(eddyb) maybe `qptr` should have its own null constant?
                            let (spv_inst, _) = &**spv_inst_and_const_inputs;
                            spv_inst.opcode == wk.OpConstantNull
                        } =>
                    {
                        Some(cx.intern(scalar::Const::from_bits(EmuStateIdx::TYPE, 0)))
                    }

                    ConstKind::PtrToFunc(func) => {
                        func_call_emu_cont.get(func).and_then(|call_emu_cont| {
                            Some(cx.intern(call_emu_cont.entry_state_idx().ok()?.as_scalar()))
                        })
                    }
                    _ => None,
                };
                maybe_state_idx_const.map_or(Transformed::Unchanged, Transformed::Changed)
            };
            for &gv in &all_uses_from_module.global_vars {
                let gv_decl = &mut module.global_vars[gv];
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

                                if let Transformed::Changed(encoded) = encoded_part {
                                    let written_range = const_data
                                        .write_scalar(
                                            offset,
                                            *encoded.as_scalar(cx).unwrap(),
                                            &self.global_stack.config.layout_config,
                                        )
                                        .unwrap();

                                    // HACK(eddyb) only perfectly overwriting is allowed.
                                    assert_eq!(written_range, offset..next_offset);
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

            let in_place_transform_values = |values: &mut [Value]| {
                for v in values {
                    if let Value::Const(ct) = v {
                        let func_ptr_size =
                            self.global_stack.config.layout_config.logical_ptr_size_align.0;
                        transform_const_use(*ct, func_ptr_size).apply_to(ct);
                    }
                }
            };

            // FIXME(eddyb) dedup this with `retype_func_ptrs_in_module`.
            let mut reusable_queue = VecDeque::new();
            for &func in self.call_graph.caller_to_callees.keys() {
                let func_decl = &mut module.funcs[func];
                let DeclDef::Present(func_def_body) = &mut func_decl.def else {
                    // FIXME(eddyb) do imports even make sense here? still, even so,
                    // the signature should probably be updated, or at least checked?
                    continue;
                };

                // FIXME(eddyb) adopt this style of queue-based visiting in more places.
                let queue = &mut reusable_queue;
                queue.clear();
                queue.push_back(func_def_body.body);
                while let Some(region) = queue.pop_front() {
                    let mut func_at_region = func_def_body.at_mut(region);

                    let mut func_at_region_children =
                        func_at_region.reborrow().at_children().into_iter();
                    while let Some(func_at_node) = func_at_region_children.next() {
                        let node_def = func_at_node.def();
                        in_place_transform_values(&mut node_def.inputs);
                        queue.extend(node_def.child_regions.iter().copied());
                    }

                    in_place_transform_values(&mut func_at_region.def().outputs);
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

        let mut call_start_cont_body = self.global_stack.invoke_cont_closure(
            func.reborrow(),
            call_emu_cont,
            &call_args,
            Some(&EmuContClosure {
                origin: Err(call_site_ret_cont_state.to_value(cx)),
                input_count: 0,
                captures: FxIndexSet::default(),
            }),
        );

        // HACK(eddyb) support splicing lists to make this O(1).
        while let Some(node) = call_start_cont_body.children.remove_first(func.nodes) {
            func.regions[call_site.parent_region].children.insert_before(
                node,
                call_site.func_call_node,
                func.nodes,
            );
        }

        let call_node_output_indices = 0..func.nodes[call_site.func_call_node].outputs.len();
        let ret_cont_def = {
            let dummy_state_in = EmuStateIdx::UNKNOWN_STATE.to_value(cx);

            let ret_cont_origin = Either::Right((
                call_site.func_call_node,
                NodeEmuStates { merge: call_site_ret_cont_state },
            ));
            let (ret_cont_closure, ret_cont_def) = self.global_stack.collect_cont_closure(
                func.reborrow(),
                ret_cont_origin,
                EmuContBody { children: EntityList::empty(), next_state_after: dummy_state_in },
            );

            {
                let EmuContClosure { origin, input_count, captures } = ret_cont_closure;
                assert!(origin == Ok(ret_cont_origin));
                assert_eq!(input_count, call_node_output_indices.len());
                assert!(captures.is_empty());
            }

            {
                let EmuContDef {
                    inputs_and_captures,
                    body: EmuContBody { children: _, next_state_after: dummy_state_out },
                } = &ret_cont_def;

                assert_eq!(inputs_and_captures.len(), call_node_output_indices.len());
                assert!(*dummy_state_out == dummy_state_in);
            }

            ret_cont_def
        };

        let state_machine_loop_body = func.regions.define(cx, RegionDef::default());

        // HACK(eddyb) these variables are never used across loop iterations,
        // but they must still exist to be able to pass callee returned values
        // out of the loop (now that loops are hermetic on the output side).
        for call_output_idx in call_node_output_indices.clone() {
            let call_output = func.nodes[call_site.func_call_node].outputs[call_output_idx];
            let ty = func.vars[call_output].ty;

            let state_machine_loop_body_def = &mut func.regions[state_machine_loop_body];
            let var = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty,
                    def_parent: Either::Left(state_machine_loop_body),
                    def_idx: state_machine_loop_body_def.inputs.len().try_into().unwrap(),
                },
            );
            state_machine_loop_body_def.inputs.push(var);
        }

        let state_ty = cx.intern(EmuStateIdx::TYPE);
        let current_state = {
            let state_machine_loop_body_def = &mut func.regions[state_machine_loop_body];
            let input_var = func.vars.define(
                cx,
                VarDecl {
                    attrs: AttrSet::default(),
                    ty: state_ty,

                    def_parent: Either::Left(state_machine_loop_body),
                    def_idx: state_machine_loop_body_def.inputs.len().try_into().unwrap(),
                },
            );
            state_machine_loop_body_def.inputs.push(input_var);
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

        // FIXME(eddyb) come up with a common naming scheme.
        let call_outputs_typed_undefs: SmallVec<[Value; 2]> = call_node_output_indices
            .clone()
            .map(|call_output_idx| {
                let call_output = func.nodes[call_site.func_call_node].outputs[call_output_idx];
                let ty = func.vars[call_output].ty;
                Value::Const(cx.intern(ConstDef {
                    attrs: AttrSet::default(),
                    ty,
                    kind: ConstKind::Undef,
                }))
            })
            .collect();

        // NOTE(eddyb) the `switch` default case is where all intermediary states
        // will end up (i.e. the state machine loop keeps going), with only the
        // successful return, and error states, being matched for explicitly.
        let non_default_next_state_switch_cases = [
            (call_site_ret_cont_state, Ok(ret_cont_def)),
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
                    Some((_, Ok(ret_cont_def))) => {
                        let ret_vals = ret_cont_def
                            .inputs_and_captures
                            .iter()
                            .map(|&v| Value::Var(v))
                            .collect();
                        ret_cont_def
                            .define_into(func.reborrow().at(case_region), &self.global_stack);
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

                let mut outputs =
                    break_with_outputs.unwrap_or_else(|| call_outputs_typed_undefs.clone());
                outputs.push(loop_repeat_cond);
                func.reborrow().at(case_region).def().outputs = outputs;

                case_region
            })
            .collect();
        let next_state_switch = func.nodes.define(
            cx,
            NodeDef {
                attrs: AttrSet::default(),
                kind: NodeKind::Select(SelectionKind::Switch {
                    case_consts: next_state_switch_case_consts,
                }),
                inputs: [next_state].into_iter().collect(),
                child_regions: next_state_switch_cases,
                outputs: [].into_iter().collect(),
            }
            .into(),
        );
        func.regions[state_machine_loop_body].children.insert_last(next_state_switch, func.nodes);

        for call_output_idx in call_node_output_indices.clone() {
            let call_output = func.nodes[call_site.func_call_node].outputs[call_output_idx];
            let ty = func.vars[call_output].ty;

            let next_state_switch_def = &mut func.nodes[next_state_switch];
            let call_output_var_on_ret = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty,
                    def_parent: Either::Right(next_state_switch),
                    def_idx: next_state_switch_def.outputs.len().try_into().unwrap(),
                },
            );
            next_state_switch_def.outputs.push(call_output_var_on_ret);
            func.regions[state_machine_loop_body].outputs.push(Value::Var(call_output_var_on_ret));
        }
        func.regions[state_machine_loop_body].outputs.push(next_state);

        let state_machine_loop_repeat_cond = {
            // FIXME(eddyb) automate this (insertion cursor?).
            let next_state_switch_def = &mut func.nodes[next_state_switch];
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

        let state_machine_loop_node = call_site.func_call_node;
        {
            let state_machine_loop_node_def = &mut func.nodes[state_machine_loop_node];
            state_machine_loop_node_def.kind =
                NodeKind::Loop { repeat_condition: state_machine_loop_repeat_cond };
            state_machine_loop_node_def.inputs = call_outputs_typed_undefs;
            state_machine_loop_node_def.child_regions =
                [state_machine_loop_body].into_iter().collect();

            state_machine_loop_node_def.inputs.push(call_start_cont_body.next_state_after);

            // HACK(eddyb) `.outputs` starts out as all of the original call's
            // output `Var`s, which are already in the right place to take the
            // right values on the last loop iteration (when the callee returns),
            // so the only addition needed here is an (effectively unused) `Var`
            // matching the "next state" that each loop iteration updates.
            let next_state_output_var = func.vars.define(
                cx,
                VarDecl {
                    attrs: Default::default(),
                    ty: state_ty,
                    def_parent: Either::Right(state_machine_loop_node),
                    def_idx: state_machine_loop_node_def.outputs.len().try_into().unwrap(),
                },
            );
            state_machine_loop_node_def.outputs.push(next_state_output_var);
        }
    }
}

struct EmuStateReserver<'a> {
    // FIXME(eddyb) put both of these fields behind a single reference?
    indirect_callee_emu_group: Option<CallEmuGroup>,
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

    // TODO(eddyb) consider in-place stack overflow aborts, than at call sites.
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
#[derive(Copy, Clone, PartialEq, Eq)]
struct RegionEmuStates {
    /// Only used for function bodies (i.e. as the target of a call) and
    /// loop bodies (i.e. as the target of a backedge).
    entry_state: Option<EmuStateIdx>,
}

/// Indicates a `Node` requiring state-splitting (an emulated `FuncCall`,
/// or due to its children), and includes additional helper state(s).
//
// FIXME(eddyb) better names/organization?
#[derive(Copy, Clone, PartialEq, Eq)]
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
        let wk = &spv::spec::Spec::get().well_known;

        let node_def = func_at_node.def();
        let needs_merge_state = match &node_def.kind {
            NodeKind::Loop { .. } => {
                self.body_stack.push(node_def.child_regions[0]);
                false
            }
            &DataInstKind::FuncCall(callee) => {
                self.func_emu_summary[callee].emu_group == Some(self.emu_group)
            }
            DataInstKind::SpvInst(spv_inst, _)
                if spv_inst.opcode == wk.OpFunctionPointerCallINTEL =>
            {
                self.indirect_callee_emu_group == Some(self.emu_group)
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
                    kind: ConstKind::PtrToGlobalVar { global_var: global, offset: None },
                }),
            )
        };

        let (stack_array_global, ptr_to_stack_array_global) = invocation_local_global_and_ptr_to(
            config.stack_size_bytes,
            // HACK(eddyb) this is equivalent to `None`, but forces `qptr::legalize`
            // to fuse it with the `Private` globals that also have initializers.
            Some(GlobalVarInit::Data(crate::mem::const_data::ConstData::new(
                config.stack_size_bytes,
            )))
            .filter(|_| false),
        );

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

    fn size_of_type_for_stack_in_stack_units(
        &self,
        ty: Type,
        reason: &str,
    ) -> Result<NonZeroU32, Diag> {
        let mem_layout = self.layout_cache.fixed_mem_layout_of(ty, reason)?;
        self.size_for_stack_in_stack_units(mem_layout, reason).map_err(|mut diag| {
            diag.message.extend([", due to type `".into(), ty.into(), "`".into()]);
            diag
        })
    }

    fn size_for_stack_in_stack_units(
        &self,
        mem_layout: crate::mem::shapes::MemLayout,
        reason: &str,
    ) -> Result<NonZeroU32, Diag> {
        // HACK(eddyb) the condition used to assume "stack unit" alignment
        // (i.e. `mem_layout.align > self.config.stack_unit_bytes`), but
        // as long as the total stack size is compatible, it should be fine.
        if !self.config.stack_size_bytes.is_multiple_of(mem_layout.align) {
            return Err(Diag::bug([format!(
                "alignment {} not supported for {reason}",
                mem_layout.align
            )
            .into()]));
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
    #[must_use]
    fn collect_cont_closure(
        &self,
        func: FuncAtMut<'_, ()>,
        origin: Either<(Region, RegionEmuStates), (Node, NodeEmuStates)>,
        cont_body: EmuContBody,
    ) -> (EmuContClosure, EmuContDef) {
        self.collect_cont_closure_with_collector_access(func, origin, cont_body, |_, _| {})
    }

    // TODO(eddyb) document like `collect_cont_closure` while allowing self-invocation.
    #[must_use]
    fn collect_cont_closure_with_collector_access(
        &self,
        mut func: FuncAtMut<'_, ()>,
        origin: Either<(Region, RegionEmuStates), (Node, NodeEmuStates)>,
        mut cont_body: EmuContBody,
        // FIXME(eddyb) this is only used by loops for self-invocation, find some
        // better way (builder pattern?) to access this API.
        access_collector: impl FnOnce(FuncAtMut<'_, ()>, &mut EmuContClosureCollector<'_>),
    ) -> (EmuContClosure, EmuContDef) {
        let mut collector = EmuContClosureCollector {
            cx: &self.cx,

            closure: EmuContClosure {
                origin: Ok(origin),
                input_count: origin.either_with(
                    func.reborrow().freeze(),
                    |func, (region, _)| func.at(region).def().inputs.len(),
                    |func, (node, _)| func.at(node).def().outputs.len(),
                ),
                captures: FxIndexSet::default(),
            },

            inputs_and_captures: vec![],
            defined_vars: EntityOrientedDenseMap::new(),
        };

        // HACK(eddyb) guarantee the first `cont.input_count` pops.
        for i in 0..collector.closure.input_count {
            assert_eq!(collector.inputs_and_captures.len(), i);
            let v = Value::Var(origin.either(
                |(region, _)| func.regions[region].inputs[i],
                |(node, _)| func.nodes[node].outputs[i],
            ));
            match collector.transform_value_use_in_func(func.reborrow().at(v)) {
                Transformed::Unchanged => unreachable!(),
                Transformed::Changed(new) => {
                    assert!(new == Value::Var(collector.inputs_and_captures[i]));
                }
            }
        }
        assert_eq!(collector.inputs_and_captures.len(), collector.closure.input_count);
        assert!(collector.closure.captures.is_empty());

        func.reborrow()
            .at(cont_body.children)
            .into_iter()
            .inner_in_place_transform_with(&mut collector);

        // HACK(eddyb) this allows access to `collector.closure`, and also the
        // ability to apply existing remappings via `EmuContClosureCollector`,
        // without `access_collector` being able to introduce *extra* captures.
        let frozen_capture_count = collector.closure.captures.len();
        access_collector(func.reborrow(), &mut collector);
        assert_eq!(collector.closure.captures.len(), frozen_capture_count);

        (
            collector.closure,
            EmuContDef { inputs_and_captures: collector.inputs_and_captures, body: cont_body },
        )
    }

    // FIXME(eddyb) is this the right API?
    // TODO(eddyb) document as building `λ(). cont(...inputs)`.
    fn invoke_cont_closure(
        &self,
        mut func: FuncAtMut<'_, ()>,
        cont: &EmuContClosure,
        inputs: &[Value],
        // FIXME(eddyb) get rid of this by encoding everything with e.g. thunks.
        ret_cont_input: Option<&EmuContClosure>,
    ) -> EmuContBody {
        assert_eq!(inputs.len(), cont.input_count);

        let ret_cont_state = ret_cont_input.map(|ret_cont| ret_cont.entry_state_value(&self.cx));
        let captures = match ret_cont_input {
            Some(ret_cont) => {
                assert!(cont.captures.is_empty());
                &ret_cont.captures
            }
            None => &cont.captures,
        };

        let values_in_pop_order = inputs
            .iter()
            .copied()
            .chain(ret_cont_state)
            .chain(captures.iter().map(|&v| Value::Var(v)));
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
    // HACK(eddyb) helper shared by `mem.func_local_var` and `push` below.
    fn mem_offset_for_push(
        &mut self,
        mem_layout: crate::mem::shapes::MemLayout,
        reason: &str,
    ) -> Result<i32, Diag> {
        let size_in_stack_units =
            self.global_stack.size_for_stack_in_stack_units(mem_layout, reason)?;

        self.accessed_stack_unit_offsets.end =
            self.accessed_stack_unit_offsets.end.max(self.offset_in_stack_units);
        self.offset_in_stack_units = self
            .offset_in_stack_units
            .checked_sub(size_in_stack_units.get().try_into().unwrap())
            .unwrap();
        self.accessed_stack_unit_offsets.start =
            self.accessed_stack_unit_offsets.start.min(self.offset_in_stack_units);

        Ok(self
            .offset_in_stack_units
            .checked_mul(self.global_stack.config.stack_unit_bytes.get().try_into().unwrap())
            .unwrap())
    }

    fn push(&mut self, mut func: FuncAtMut<'_, ()>, v: Value) {
        let cx = &self.global_stack.cx;

        let mut attrs = AttrSet::default();
        let ty = func.reborrow().freeze().at(v).type_of(cx);
        let reason = "pushing state to emulated stack";
        let offset = self
            .global_stack
            .layout_cache
            .fixed_mem_layout_of(ty, reason)
            .and_then(|mem_layout| self.mem_offset_for_push(mem_layout, reason))
            .map_err(|diag| {
                attrs.push_diag(cx, diag);
            });

        let inst = func.nodes.define(
            cx,
            DataInstDef {
                attrs,
                kind: DataInstKind::Mem(MemOp::Store {
                    offset: NonZeroI32::new(offset.unwrap_or(0)),
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
    fn pop_into(&mut self, func: FuncAtMut<'_, ()>, output_var: Var) {
        let cx = &self.global_stack.cx;

        let ty = func.vars[output_var].ty;

        let mut attrs = AttrSet::default();
        let size_in_stack_units = self
            .global_stack
            .size_of_type_for_stack_in_stack_units(ty, "popping state from emulated stack")
            .map_err(|diag| {
                attrs.push_diag(cx, diag);
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
                outputs: [output_var].into_iter().collect(),
            }
            .into(),
        );

        {
            let output_var_decl = &mut func.vars[output_var];
            output_var_decl.def_parent = Either::Right(inst);
            output_var_decl.def_idx = 0;
        }

        // FIXME(eddyb) automate this (insertion cursor?).
        self.push_pop_insts.insert_last(inst, func.nodes);

        self.accessed_stack_unit_offsets.start =
            self.accessed_stack_unit_offsets.start.min(self.offset_in_stack_units);
        self.offset_in_stack_units = self
            .offset_in_stack_units
            .checked_add(size_in_stack_units.map_or(0, |size| size.get()).try_into().unwrap())
            .unwrap();
        self.accessed_stack_unit_offsets.end =
            self.accessed_stack_unit_offsets.end.max(self.offset_in_stack_units);
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
    // HACK(eddyb) helper shared by `mem.func_local_var` and `finish_for_state` below.
    // FIXME(eddyb) consider reusing the `qptr::{legalize,lift}` "insertion cursor".
    fn stack_top_plus_offset_in_stack_units(
        &self,
        func: FuncAtMut<'_, ()>,
        offset_in_stack_units: i32,
    ) -> (Node, Value) {
        let cx = &self.global_stack.cx;

        let inst = func.nodes.define(
            cx,
            DataInstDef {
                attrs: AttrSet::default(),
                kind: DataInstKind::Scalar(scalar::Op::IntBinary(if offset_in_stack_units < 0 {
                    scalar::IntBinOp::Sub
                } else {
                    scalar::IntBinOp::Add
                })),
                inputs: [
                    Value::Var(self.stack_top_initial),
                    Value::Const(
                        cx.intern(scalar::Const::from_u32(offset_in_stack_units.unsigned_abs())),
                    ),
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

        (inst, Value::Var(output_var))
    }

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

        let mut mk_stack_top_plus = |offset_in_stack_units| {
            let (inst, output) =
                self.stack_top_plus_offset_in_stack_units(func.reborrow(), offset_in_stack_units);

            if stack_overflow_checked_insts.is_some() {
                pre_check_insts.insert_last(inst, func.nodes);
            } else {
                pre_check_insts.insert_before(inst, self.stack_ptr_inst, func.nodes);
            }

            output
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
        // TODO(eddyb) remove this, feels unnecessary.
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

struct EmuContClosureCollector<'a> {
    cx: &'a Context,

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
    //
    // TODO(eddyb) update the docs, now that the pops are not done on the fly.
    inputs_and_captures: Vec<Var>,

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
//
// TODO(eddyb) update docs (after `EmuContDef` addition)
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

// HACK(eddyb) this is the "def-side" of an emulated continuation, which can be
// invoked via `EmuContClosure` (making that the "use-side").
// TODO(eddyb) document (might be gone after thunkification?)
struct EmuContDef {
    inputs_and_captures: Vec<Var>,
    body: EmuContBody,
}

impl EmuContDef {
    fn define_into(
        self,
        mut func_at_region: FuncAtMut<'_, Region>,
        global_stack: &EmuGlobalStack<'_>,
    ) {
        let EmuContDef { inputs_and_captures, mut body } = self;

        if !inputs_and_captures.is_empty() {
            let mut func = func_at_region.reborrow().at(());

            let mut popper = global_stack.popper(func.reborrow());
            for v in inputs_and_captures {
                popper.pop_into(func.reborrow(), v);
            }

            let pops_nodes = popper.finish(func.reborrow());
            body.children.prepend(pops_nodes, func.nodes);
        }

        *func_at_region.def() = body.into_region_def();
    }
}

impl Transformer for EmuContClosureCollector<'_> {
    fn transform_value_use_in_func(
        &mut self,
        func_at_val: FuncAtMut<'_, Value>,
    ) -> Transformed<Value> {
        let v = func_at_val.position;
        let func = func_at_val.at(());

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

        let input_or_capture_idx = match cont_input_idx {
            Some(input_idx) => input_idx.try_into().unwrap(),
            None => self.closure.input_count + self.closure.captures.insert_full(v).0,
        };

        if let Some(&v) = self.inputs_and_captures.get(input_or_capture_idx) {
            // Already seen (i.e. effectively cached).
            return Transformed::Changed(Value::Var(v));
        }

        // Reserve a new `Var` that can eventually be used as a pop destination.
        assert_eq!(input_or_capture_idx, self.inputs_and_captures.len());

        let ty = func.vars[v].ty;

        let new_var = func.vars.define(
            self.cx,
            VarDecl {
                attrs: Default::default(),
                ty,
                // HACK(eddyb) using an existing parent to declare an "orphan" `Var`.
                def_parent: func.vars[v].def_parent,
                def_idx: !0,
            },
        );
        self.inputs_and_captures.push(new_var);

        Transformed::Changed(Value::Var(new_var))
    }

    fn in_place_transform_region_def(&mut self, mut func_at_region: FuncAtMut<'_, Region>) {
        for &input_var in &func_at_region.reborrow().def().inputs {
            self.defined_vars.insert(input_var, ());
        }
        func_at_region.inner_in_place_transform_with(self);
    }

    fn in_place_transform_node_def(&mut self, mut func_at_node: FuncAtMut<'_, Node>) {
        func_at_node.inner_in_place_transform_with(self);
        let node_def = func_at_node.def();
        if let NodeKind::Mem(MemOp::FuncLocalVar(_)) = node_def.kind {
            node_def.attrs.push_diag(
                self.cx,
                Diag::bug(["unexpected local not at the start of the function".into()]),
            );
        }
        for &output_var in &node_def.outputs {
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
                let (merge_cont, merge_cont_def) = self.global_stack.collect_cont_closure(
                    func.reborrow(),
                    Either::Right((node, node_states)),
                    cont_body,
                );
                assert_eq!(merge_cont.entry_state_idx().ok().unwrap(), node_states.merge);

                let merge_cont_region = func.regions.define(cx, RegionDef::default());
                merge_cont_def
                    .define_into(func.reborrow().at(merge_cont_region), self.global_stack);
                self.state_switch_cases.insert(node_states.merge, merge_cont_region);

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
        let cx = &self.global_stack.cx;

        let region = func_at_region.position;
        let mut func = func_at_region.at(());

        let cont_body = {
            let region_def = &mut func.regions[region];
            let children = mem::take(&mut region_def.children);
            let outputs = mem::take(&mut region_def.outputs);

            // HACK(eddyb) avoid merging an always-aborting branch.
            if let Some(last) = children.iter().last
                && let NodeKind::ExitInvocation(_) = func.nodes[last].kind
            {
                EmuContBody {
                    children,
                    next_state_after: Value::Const(cx.intern(ConstDef {
                        attrs: AttrSet::default(),
                        ty: cx.intern(EmuStateIdx::TYPE),
                        kind: ConstKind::Undef,
                    })),
                }
            } else {
                invoke_merge(self.global_stack, func.reborrow(), children, &outputs)
            }
        };

        let Some(&region_states) = self.states.for_region.get(region) else {
            let region_def = &mut func.regions[region];
            assert_eq!(region_def.inputs.len(), 0);
            *region_def = cont_body.into_region_def();
            return None;
        };

        let cont_body = self.frack_cont_body_nodes_as_needed(func.reborrow(), cont_body);

        if let Some(entry_state) = region_states.entry_state {
            let (entry_cont, entry_cont_def) = self.global_stack.collect_cont_closure(
                func.reborrow(),
                Either::Left((region, region_states)),
                cont_body,
            );
            assert_eq!(entry_cont.entry_state_idx().ok().unwrap(), entry_state);

            entry_cont_def.define_into(func.at(region), self.global_stack);
            self.state_switch_cases.insert(entry_state, region);

            Some(entry_cont)
        } else {
            assert_eq!(func.regions[region].inputs.len(), 0);

            func.regions[region] = cont_body.into_region_def();

            None
        }
    }

    fn frack_node(
        &mut self,
        func_at_node: FuncAtMut<'_, Node>,
        merge: EmuContClosure,
    ) -> EmuContBody {
        let cx = &self.global_stack.cx;
        let wk = &spv::spec::Spec::get().well_known;

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
                                None,
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
                let body_outputs = mem::take(&mut func.regions[body].outputs);

                let body_states = self.states.for_region[body];
                let body_entry_state = body_states.entry_state.unwrap();

                // Structured loops are equivalent to their `body` being trailed
                // by an implied conditional backedge, akin to an unstructured
                // `if repeat { goto body(...body.outputs) } else { goto merge(...body.outputs) }`,
                // so *correctly* closure-converting `body` needs it to already
                // contain both the "backedge" (`body`) and "break" (`merge`)
                // targets (or rather, `body.outputs` and `merge.captures` must
                // *both* be present within `body`, to be accurately tracked)
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
                        .invoke_cont_closure(func.reborrow(), &merge, &body_outputs, None)
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
                let (body_entry_cont, mut whole_body_cont_def) =
                    self.global_stack.collect_cont_closure_with_collector_access(
                        func.reborrow(),
                        Either::Left((body, body_states)),
                        whole_body_cont_body,
                        |mut func, collector| {
                            func.regions[backedge_region] = self
                                .global_stack
                                .invoke_cont_closure(
                                    func.reborrow(),
                                    &collector.closure,
                                    &body_outputs,
                                    None,
                                )
                                .into_region_def();
                            collector.in_place_transform_region_def(func.at(backedge_region));
                        },
                    );

                // HACK(eddyb) this is done *after* closure-converting `body`
                // (via `collect_cont_closure_with_collector_access`), so that
                // the self-invocation performed above is *already* aware of
                // any captures needed *anywhere* inside the `body`, before
                // fracking its child nodes may split it into separate regions.
                whole_body_cont_def.body =
                    self.frack_cont_body_nodes_as_needed(func.reborrow(), whole_body_cont_def.body);
                whole_body_cont_def.define_into(func.reborrow().at(body), self.global_stack);

                // HACK(eddyb) this used to be part of `frack_region_as_needed`.
                {
                    assert_eq!(body_entry_cont.entry_state_idx().ok().unwrap(), body_entry_state);
                    self.state_switch_cases.insert(body_entry_state, body);
                }
                let loop_initial_inputs = mem::take(&mut func.nodes[node].inputs);
                self.global_stack.invoke_cont_closure(
                    func,
                    &body_entry_cont,
                    &loop_initial_inputs,
                    None,
                )
            }

            &DataInstKind::FuncCall(callee) => {
                // TODO(eddyb) for this to be compatible with "frame (base) pointers",
                // the `merge` (return) continuation should be pushed separately
                // from the actual function inputs, and/or maybe there should
                // be that and `merge.entry_state_value(cx)` in actual inputs
                // or something idk... OH YES. GOT IT!!!
                //
                // !!! instead of assuming the return continuation is on top of
                // the stack once the function args are popped off, it should
                // be passed in as a `(code_ptr, data_ptr)` pair, that then
                // `qptr::legalize` has to handle (the data one, that is), and
                // the stack being restored is done by that return continuation
                // happening to capture the next up frame pointer or w/e etc.
                let inputs = mem::take(&mut node_def.inputs);
                self.global_stack.invoke_cont_closure(
                    func,
                    &self.func_call_emu_cont[&callee],
                    &inputs,
                    Some(&merge),
                )
            }

            // FIXME(eddyb) deduplicate with `FuncCall` above.
            DataInstKind::SpvInst(spv_inst, _)
                if spv_inst.opcode == wk.OpFunctionPointerCallINTEL =>
            {
                let inputs = mem::take(&mut node_def.inputs);
                let (&callee, inputs) = inputs.split_first().unwrap();

                self.global_stack.invoke_cont_closure(
                    func,
                    &EmuContClosure {
                        origin: Err(callee),
                        input_count: inputs.len(),
                        captures: FxIndexSet::default(),
                    },
                    inputs,
                    Some(&merge),
                )
            }

            DataInstKind::Scalar(_)
            | DataInstKind::Vector(_)
            | DataInstKind::Mem(_)
            | DataInstKind::QPtr(_)
            | DataInstKind::ThunkBind(_)
            | DataInstKind::SpvInst(..)
            | DataInstKind::SpvExtInst { .. } => unreachable!(),
        }
    }
}
