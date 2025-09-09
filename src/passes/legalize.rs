use crate::visit;
use crate::{DeclDef, Module, cf};

/// Apply the [`cf::unstructured::Structurizer`] algorithm to all function definitions in `module`.
pub fn structurize_func_cfgs(module: &mut Module) {
    let cx = &module.cx();

    // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
    let visit::AllUses { funcs, .. } = visit::AllUses::from_module(module);

    for &func in &funcs {
        let func_decl = &mut module.funcs[func];
        if let DeclDef::Present(_) = func_decl.def {
            cf::structurize::Structurizer::new(cx, func_decl).structurize_func();
        }
    }
}

// FIXME(eddyb) properly make this configurable.
pub fn emulate_call_stack(module: &mut Module, config: &cf::stackful::CallStackEmuConfig) {
    cf::stackful::CallStackEmulator::new(module, config).transform_module(module);
}
