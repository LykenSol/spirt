//! [`QPtr`](crate::TypeKind::QPtr) transforms.

use crate::visit::{InnerVisit, Visitor};
use crate::{AttrSet, Const, Context, DeclDef, Func, FxIndexSet, GlobalVar, Module, Type, qptr};

pub fn lower_from_spv_ptrs(module: &mut Module, layout_config: &qptr::LayoutConfig) {
    let cx = &module.cx();

    let (seen_global_vars, seen_funcs) = {
        // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
        let mut collector = ReachableUseCollector {
            cx,
            module,

            seen_types: FxIndexSet::default(),
            seen_consts: FxIndexSet::default(),
            seen_global_vars: FxIndexSet::default(),
            seen_funcs: FxIndexSet::default(),
        };
        for (export_key, &exportee) in &module.exports {
            export_key.inner_visit_with(&mut collector);
            exportee.inner_visit_with(&mut collector);
        }
        (collector.seen_global_vars, collector.seen_funcs)
    };

    let lowerer = qptr::lower::LowerFromSpvPtrs::new(cx.clone(), layout_config);
    for &global_var in &seen_global_vars {
        lowerer.lower_global_var(&mut module.global_vars[global_var]);
    }
    for &func in &seen_funcs {
        lowerer.lower_func(&mut module.funcs[func]);
    }
}

// FIXME(eddyb) split this into separate passes, but the looping complicates things.
pub fn partition_and_propagate(module: &mut Module, layout_config: &qptr::LayoutConfig) {
    let cx = &module.cx();

    let (_seen_global_vars, seen_funcs) = {
        // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
        let mut collector = ReachableUseCollector {
            cx,
            module,

            seen_types: FxIndexSet::default(),
            seen_consts: FxIndexSet::default(),
            seen_global_vars: FxIndexSet::default(),
            seen_funcs: FxIndexSet::default(),
        };
        for (export_key, &exportee) in &module.exports {
            export_key.inner_visit_with(&mut collector);
            exportee.inner_visit_with(&mut collector);
        }
        (collector.seen_global_vars, collector.seen_funcs)
    };

    for func in seen_funcs {
        if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
            // FIXME(eddyb) reuse `LayoutCache` and whatnot, between functions,
            // or at least iterations of this loop.
            loop {
                qptr::simplify::partition_local_vars_in_func(
                    cx.clone(),
                    layout_config,
                    func_def_body,
                );

                let report = qptr::simplify::propagate_contents_of_local_vars_in_func(
                    cx.clone(),
                    layout_config,
                    func_def_body,
                );
                if !report.any_qptrs_propagated {
                    break;
                }
            }
        }
    }
}

pub fn analyze_uses(module: &mut Module, layout_config: &qptr::LayoutConfig) {
    qptr::analyze::InferUsage::new(module.cx(), layout_config).infer_usage_in_module(module);
}

pub fn lift_to_spv_ptrs(module: &mut Module, layout_config: &qptr::LayoutConfig) {
    let cx = &module.cx();

    let (seen_global_vars, seen_funcs) = {
        // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
        let mut collector = ReachableUseCollector {
            cx,
            module,

            seen_types: FxIndexSet::default(),
            seen_consts: FxIndexSet::default(),
            seen_global_vars: FxIndexSet::default(),
            seen_funcs: FxIndexSet::default(),
        };
        for (export_key, &exportee) in &module.exports {
            export_key.inner_visit_with(&mut collector);
            exportee.inner_visit_with(&mut collector);
        }
        (collector.seen_global_vars, collector.seen_funcs)
    };

    let lifter = qptr::lift::LiftToSpvPtrs::new(cx.clone(), layout_config);
    for &global_var in &seen_global_vars {
        lifter.lift_global_var(&mut module.global_vars[global_var]);
    }
    lifter.lift_all_funcs(module, seen_funcs);
}

struct ReachableUseCollector<'a> {
    cx: &'a Context,
    module: &'a Module,

    // FIXME(eddyb) build some automation to avoid ever repeating these.
    seen_types: FxIndexSet<Type>,
    seen_consts: FxIndexSet<Const>,
    seen_global_vars: FxIndexSet<GlobalVar>,
    seen_funcs: FxIndexSet<Func>,
}

impl Visitor<'_> for ReachableUseCollector<'_> {
    // FIXME(eddyb) build some automation to avoid ever repeating these.
    fn visit_attr_set_use(&mut self, _attrs: AttrSet) {
        // FIXME(eddyb) if `AttrSet`s are ignored, why not `Type`s too?
    }
    fn visit_type_use(&mut self, ty: Type) {
        if self.seen_types.insert(ty) {
            self.visit_type_def(&self.cx[ty]);
        }
    }
    fn visit_const_use(&mut self, ct: Const) {
        if self.seen_consts.insert(ct) {
            self.visit_const_def(&self.cx[ct]);
        }
    }

    fn visit_global_var_use(&mut self, gv: GlobalVar) {
        if self.seen_global_vars.insert(gv) {
            self.visit_global_var_decl(&self.module.global_vars[gv]);
        }
    }
    fn visit_func_use(&mut self, func: Func) {
        if self.seen_funcs.insert(func) {
            self.visit_func_decl(&self.module.funcs[func]);
        }
    }
}
