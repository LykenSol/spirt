//! Function call graph abstraction.

use crate::func_at::{FuncAt, FuncAtMut};
use crate::mem::MemOp;
use crate::transform::{Transformed, Transformer};
use crate::visit::{InnerVisit as _, Visitor};
use crate::{
    AttrSet, Const, ConstDef, ConstKind, Context, DataInstKind, DeclDef, Diag,
    EntityOrientedDenseMap, ExportKey, Exportee, Func, FuncDefBody, FxIndexMap, FxIndexSet,
    GlobalVar, Import, Module, Node, NodeDef, NodeKind, Region, RegionDef, Type, Value, Var, cf,
    spv,
};
use itertools::Either;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::mem;

pub struct CallGraph {
    // FIXME(eddyb) integrate these a bit better, as "call graph roots" maybe?
    pub spv_entry_points: FxIndexSet<Func>,

    pub caller_to_callees: FxIndexMap<Func, Callees>,

    pub indirect_callees: FxIndexSet<Func>,
}

#[derive(Default)]
pub struct Callees {
    // FIXME(eddyb) this wants to be a multimap, realistically.
    pub direct: FxIndexMap<Func, SmallVec<[CallSite; 2]>>,

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
                spv_entry_points: FxIndexSet::default(),
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
                Exportee::Func(func) => {
                    match export_key {
                        ExportKey::LinkName(_) => {}
                        ExportKey::SpvEntryPoint { .. } => {
                            collector.call_graph.spv_entry_points.insert(func);
                        }
                    }
                    collector.visit_func_used_by_export_or_callee(func);
                }
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

    /// Order functions using a post-order traversal, i.e. callees before callers.
    pub fn post_order(&self) -> Vec<Func> {
        // FIXME(eddyb) use a bitset for this.
        let mut visited = FxHashSet::default();
        let mut post_order = Vec::with_capacity(self.caller_to_callees.len());

        // Visit the call graph with entry points as roots.
        for &entry in &self.spv_entry_points {
            self.post_order_step(entry, &mut visited, &mut post_order);
        }

        // Also visit any functions that were not reached from entry points
        // (they might be dead but they should be processed nonetheless).
        for &func in self.caller_to_callees.keys() {
            self.post_order_step(func, &mut visited, &mut post_order);
        }

        post_order
    }

    fn post_order_step(
        &self,
        caller: Func,
        visited: &mut FxHashSet<Func>,
        post_order: &mut Vec<Func>,
    ) {
        if !visited.insert(caller) {
            return;
        }

        for callee in self.direct_and_indirect_callees_of(caller) {
            self.post_order_step(callee, visited, post_order);
        }

        post_order.push(caller);
    }

    // FIXME(eddyb) expose a more fine-grained inliner interface.
    pub fn exhaustively_inline_calls_in_module(&mut self, module: &mut Module) {
        self.inline_calls_in_module_with_filter(module, |_, _| true);
    }

    fn inline_calls_in_module_with_filter(
        &mut self,
        module: &mut Module,
        should_inline: impl Fn(FuncAt<'_, CallSite>, &FuncDefBody) -> bool,
    ) {
        let cx = module.cx();

        // TODO(eddyb) replace this with injecting diagnostics, and on call sites,
        // not on the callees (which can just be because of function pointers).
        assert!(self.indirect_callees.is_empty(), "inlining does not support indirect calls");

        // FIXME(eddyb) implement or use diagnostics instead of assert?
        for &func in self.caller_to_callees.keys() {
            let DeclDef::Present(func_def_body) = &module.funcs[func].def else {
                unreachable!();
            };
            assert!(
                func_def_body.unstructured_cfg.is_none(),
                "inlining does not support unstructured control-flow"
            );
        }

        // FIXME(eddyb) should `CallGraph` itself contain this reverse mapping?
        let mut callee_to_callers = {
            let mut callee_to_callers = FxIndexMap::<_, SmallVec<[_; 8]>>::default();
            for (&caller, callees) in &self.caller_to_callees {
                for &callee in callees.direct.keys() {
                    callee_to_callers.entry(callee).or_default().push(caller);
                }
            }
            callee_to_callers
        };

        for callee in self.post_order() {
            let Some(callers) = callee_to_callers.get_mut(&callee) else {
                continue;
            };

            // HACK(eddyb) in order to avoid borrow conflicts, the callee body
            // is stolen by replacing it with an import, while a diagnostic is
            // added just in case the function is observed in this state.
            let callee_decl = &mut module.funcs[callee];
            let callee_attrs = callee_decl.attrs;
            callee_decl
                .attrs
                .push_diag(&cx, Diag::bug(["function removed during inlining".into()]));
            let DeclDef::Present(callee_def) = mem::replace(
                &mut callee_decl.def,
                DeclDef::Imported(Import::LinkName(cx.intern(""))),
            ) else {
                unreachable!()
            };

            callers.retain(|&mut caller| {
                // FIXME(eddyb) enforce that no recursion exists to begin with,
                // but the bottom-up (i.e. postorder) approach should heavily
                // limit any damage leftover recursion may cause.
                if caller == callee {
                    return true;
                }

                let all_direct_callees =
                    &mut self.caller_to_callees.get_mut(&caller).unwrap().direct;
                let call_sites = all_direct_callees.get_mut(&callee).unwrap();

                let DeclDef::Present(caller_def) = &mut module.funcs[caller].def else {
                    unreachable!()
                };
                call_sites.retain(|&mut call_site| {
                    let inline = should_inline(caller_def.at(call_site), &callee_def);
                    if inline {
                        inline_call(&cx, caller_def, call_site, &callee_def);
                    }
                    !inline
                });

                !call_sites.is_empty()
            });

            let callee_still_used = !callers.is_empty()
                || self.spv_entry_points.contains(&callee)
                || self.indirect_callees.contains(&callee);
            if callee_still_used {
                let callee_decl = &mut module.funcs[callee];
                callee_decl.attrs = callee_attrs;
                callee_decl.def = DeclDef::Present(callee_def);
            } else {
                // HACK(eddyb) by not putting back `callee_def` (like above),
                // callees which no longer have any calls to them get "removed",
                // avoiding inefficient (quadratic) memory usage.
            }
        }

        // Prune removed calls/callees form the callgraph, to keep it usable.
        self.caller_to_callees.retain(|&func, callees| {
            let func_still_used =
                callee_to_callers.get(&func).is_some_and(|callers| !callers.is_empty())
                    || self.spv_entry_points.contains(&func)
                    || self.indirect_callees.contains(&func);
            if !func_still_used {
                return false;
            }

            callees.direct.retain(|_, call_sites| !call_sites.is_empty());

            true
        });
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

fn inline_call(cx: &Context, caller: &mut FuncDefBody, call_site: CallSite, callee: &FuncDefBody) {
    assert!(callee.unstructured_cfg.is_none());

    let inlined_callee = clone_region(cx, caller.at_mut(()), callee.at_body());

    // FIXME(eddyb) this would likely have to become more complicated if/when
    // `clone_region` stops wastefully allocating a new outermost `Region`.
    let call_args = mem::take(&mut caller.nodes[call_site.func_call_node].inputs);
    ReplaceVarUsesWith(|func_at_var| {
        let var_decl = func_at_var.decl();
        if var_decl.def_parent.left() == Some(inlined_callee) {
            Some(call_args[var_decl.def_idx as usize])
        } else {
            None
        }
    })
    .in_place_transform_region_def(caller.at_mut(inlined_callee));

    // HACK(eddyb) `spv::lift` misbehaves when `ExitInvocation` is found anywhere
    // except the last node in a region, so this is the best place to remove all
    // following nodes, if the callee ended in `ExitInvocation`.
    let callee_diverges = caller.regions[inlined_callee]
        .children
        .iter()
        .last
        .is_some_and(|node| matches!(caller.nodes[node].kind, NodeKind::ExitInvocation(_)));
    if callee_diverges {
        let parent_region_def = &mut caller.regions[call_site.parent_region];
        while let Some(node) = caller.nodes[call_site.func_call_node].next_in_list() {
            parent_region_def.children.remove(node, &mut caller.nodes);
        }
        for output in &mut parent_region_def.outputs {
            let ty = match *output {
                Value::Const(ct) => cx[ct].ty,
                Value::Var(var) => caller.vars[var].ty,
            };
            *output = Value::Const(cx.intern(ConstDef {
                attrs: AttrSet::default(),
                ty,
                kind: ConstKind::Undef,
            }));
        }
    }

    // NOTE(eddyb) with SPIR-T being "outward"/"output-side" hermetic, there
    // can only be uses of the call node output `Var`s, in the parent region.
    // FIXME(eddyb) consider even ignoring the nodes before the call node.
    let ret_vals = mem::take(&mut caller.regions[inlined_callee].outputs);
    ReplaceVarUsesWith(|func_at_var| {
        let var_decl = func_at_var.decl();
        if var_decl.def_parent.right() == Some(call_site.func_call_node) {
            Some(ret_vals[var_decl.def_idx as usize])
        } else {
            None
        }
    })
    .in_place_transform_region_def(caller.at_mut(call_site.parent_region));

    {
        let wk = &spv::spec::Spec::get().well_known;
        let is_func_local_var = |node_def: &NodeDef| match &node_def.kind {
            NodeKind::Mem(MemOp::FuncLocalVar(_)) => true,
            // HACK(eddyb) only needed because this can run after lifting,
            // and `mem.func_local_var` isn't yet used instead of `OpVariable`.
            NodeKind::SpvInst(spv_inst, _) => spv_inst.opcode == wk.OpVariable,
            _ => false,
        };
        let mut last_func_local_var = caller
            .at_body()
            .at_children()
            .into_iter()
            .take_while(|func_at_node| is_func_local_var(func_at_node.def()))
            .map(|func_at_node| func_at_node.position)
            .last();

        // HACK(eddyb) support splicing lists to make this O(1).
        let mut inlined_nodes = mem::take(&mut caller.regions[inlined_callee].children);
        while let Some(node) = inlined_nodes.remove_first(&mut caller.nodes) {
            // HACK(eddyb) inject locals at the start of the caller's body.
            if is_func_local_var(&caller.nodes[node]) {
                let caller_body_children = &mut caller.regions[caller.body].children;
                if let Some(last_func_local_var) = last_func_local_var {
                    caller_body_children.insert_after(node, last_func_local_var, &mut caller.nodes);
                } else {
                    caller_body_children.insert_first(node, &mut caller.nodes);
                }
                last_func_local_var = Some(node);
                continue;
            }

            caller.regions[call_site.parent_region].children.insert_before(
                node,
                call_site.func_call_node,
                &mut caller.nodes,
            );
        }
    }

    caller.regions[call_site.parent_region]
        .children
        .remove(call_site.func_call_node, &mut caller.nodes);
}

// FIXME(eddyb) this shouldn't allocate a new `Region` in the parent, for the
// outermost source region, but this is the easiest way to keep it all sound.
fn clone_region(cx: &Context, dst: FuncAtMut<'_, ()>, src: FuncAt<'_, Region>) -> Region {
    let dst_region = dst.regions.define(cx, RegionDef::default());
    let src_region = src.position;
    let src = src.at(());

    // HACK(eddyb) `region_mapping` exists only for `thunk.bind`'s sake.
    let mut region_mapping = EntityOrientedDenseMap::new();
    let mut var_mapping = EntityOrientedDenseMap::new();

    // FIXME(eddyb) adopt this style of queue-based visiting in more places.
    let mut queue = VecDeque::new();
    region_mapping.insert(src_region, dst_region);
    queue.push_back(src_region);
    while let Some(src_region) = queue.pop_front() {
        let dst_region = region_mapping[src_region];
        let src_region_def = src.at(src_region).def();

        let mut clone_vars = |vars: &[Var], dst_def_parent, src_def_parent| {
            vars.iter()
                .enumerate()
                .map(|(var_idx, &src_var)| {
                    let mut var_decl = src.at(src_var).decl().clone();
                    assert!(var_decl.def_parent == src_def_parent);
                    assert_eq!(var_decl.def_idx as usize, var_idx);
                    var_decl.def_parent = dst_def_parent;
                    let dst_var = dst.vars.define(cx, var_decl);
                    var_mapping.insert(src_var, dst_var);
                    dst_var
                })
                .collect()
        };

        dst.regions[dst_region].inputs =
            clone_vars(&src_region_def.inputs, Either::Left(dst_region), Either::Left(src_region));

        for src_at_node in src.at(src_region_def.children) {
            let dst_node = dst.nodes.define(cx, src_at_node.def().clone().into());
            let dst_node_def = &mut dst.nodes[dst_node];

            let mut clone_other_region = |src_other_region: Region| {
                *region_mapping.entry(src_other_region).get_or_insert_with(|| {
                    let dst_other_region = dst.regions.define(cx, RegionDef::default());
                    queue.push_back(src_other_region);
                    dst_other_region
                })
            };

            // FIXME(eddyb) find a better way to replace `Region` references.
            match &mut dst_node_def.kind {
                NodeKind::Select(_)
                | NodeKind::Loop { .. }
                | NodeKind::ExitInvocation(_)
                | NodeKind::Scalar(_)
                | NodeKind::Vector(_)
                | NodeKind::FuncCall(_)
                | NodeKind::Mem(_)
                | NodeKind::QPtr(_)
                | NodeKind::SpvInst(..)
                | NodeKind::SpvExtInst { .. } => {}

                NodeKind::ThunkBind(target) => match target {
                    cf::unstructured::ControlTarget::Region(target) => {
                        *target = clone_other_region(*target);
                    }
                    cf::unstructured::ControlTarget::Return => {}
                },
            }

            for child_region in &mut dst_node_def.child_regions {
                *child_region = clone_other_region(*child_region);
            }
            dst_node_def.outputs = clone_vars(
                &dst_node_def.outputs,
                Either::Right(dst_node),
                Either::Right(src_at_node.position),
            );

            dst.regions[dst_region].children.insert_last(dst_node, dst.nodes);
        }

        dst.regions[dst_region].outputs = src_region_def.outputs.clone();
    }

    ReplaceVarUsesWith(|dst_at_src_var| {
        let src_var = dst_at_src_var.position;
        Some(Value::Var(var_mapping[src_var]))
    })
    .in_place_transform_region_def(dst.at(dst_region));

    dst_region
}

// FIXME(eddyb) maybe this should be provided by `spirt::transform`.
struct ReplaceVarUsesWith<F: FnMut(FuncAt<'_, Var>) -> Option<Value>>(F);
impl<F: FnMut(FuncAt<'_, Var>) -> Option<Value>> Transformer for ReplaceVarUsesWith<F> {
    fn transform_value_use_in_func(
        &mut self,
        func_at_val: FuncAtMut<'_, Value>,
    ) -> Transformed<Value> {
        match func_at_val.position {
            Value::Const(_) => None,
            Value::Var(v) => self.0(func_at_val.freeze().at(v)),
        }
        .map_or(Transformed::Unchanged, Transformed::Changed)
    }
}
