//! Function call graph abstraction.

use crate::func_at::FuncAt;
use crate::visit::{InnerVisit as _, Visitor};
use crate::{
    AttrSet, Const, ConstDef, ConstKind, Context, DataInstKind, Exportee, Func, FxIndexMap,
    FxIndexSet, GlobalVar, Module, Node, NodeDef, Region, Type, spv,
};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::mem;

pub struct CallGraph {
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
