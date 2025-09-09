use crate::visit;
use crate::{DeclDef, Module, cf};

/// Apply the [`cf::unstructured::Structurizer`] algorithm to all function definitions in `module`.
pub fn structurize_func_cfgs(module: &mut Module) {
    let cx = &module.cx();

    // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
    let visit::AllUses { funcs, .. } = visit::AllUses::from_module(module);

    for &func in &funcs {
        if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
            cf::structurize::Structurizer::new(cx, func_def_body).structurize_func();
        }
    }
}
