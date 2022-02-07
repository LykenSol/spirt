use crate::{
    cfg, spv, AddrSpace, Attr, AttrSet, AttrSetDef, Const, ConstCtor, ConstDef, ControlNode,
    ControlNodeDef, ControlNodeKind, ControlNodeOutputDecl, DataInstDef, DataInstKind, DeclDef,
    ExportKey, Exportee, Func, FuncAt, FuncDecl, FuncDefBody, FuncParam, GlobalVar, GlobalVarDecl,
    GlobalVarDefBody, Import, Module, ModuleDebugInfo, ModuleDialect, Type, TypeCtor, TypeCtorArg,
    TypeDef, Value,
};

// FIXME(eddyb) `Sized` bound shouldn't be needed but removing it requires
// writing `impl Visitor<'a> + ?Sized` in `fn inner_visit_with` signatures.
pub trait Visitor<'a>: Sized {
    // Context-interned leaves (no default provided).
    // FIXME(eddyb) treat these separately somehow and allow e.g. automatic deep
    // visiting (with a set to avoid repeat visits) if a `Rc<Context>` is provided.
    fn visit_attr_set_use(&mut self, attrs: AttrSet);
    fn visit_type_use(&mut self, ty: Type);
    fn visit_const_use(&mut self, ct: Const);

    // Module-stored entity leaves (no default provided).
    fn visit_global_var_use(&mut self, gv: GlobalVar);
    fn visit_func_use(&mut self, func: Func);

    // Leaves (noop default behavior).
    fn visit_spv_dialect(&mut self, _dialect: &spv::Dialect) {}
    fn visit_spv_module_debug_info(&mut self, _debug_info: &spv::ModuleDebugInfo) {}
    fn visit_attr(&mut self, _attr: &Attr) {}
    fn visit_import(&mut self, _import: &Import) {}

    // Non-leaves (defaulting to calling `.inner_visit_with(self)`).
    fn visit_module(&mut self, module: &'a Module) {
        module.inner_visit_with(self);
    }
    fn visit_module_dialect(&mut self, dialect: &'a ModuleDialect) {
        dialect.inner_visit_with(self);
    }
    fn visit_module_debug_info(&mut self, debug_info: &'a ModuleDebugInfo) {
        debug_info.inner_visit_with(self);
    }
    fn visit_attr_set_def(&mut self, attrs_def: &'a AttrSetDef) {
        attrs_def.inner_visit_with(self);
    }
    fn visit_type_def(&mut self, ty_def: &'a TypeDef) {
        ty_def.inner_visit_with(self);
    }
    fn visit_const_def(&mut self, ct_def: &'a ConstDef) {
        ct_def.inner_visit_with(self);
    }
    fn visit_global_var_decl(&mut self, gv_decl: &'a GlobalVarDecl) {
        gv_decl.inner_visit_with(self);
    }
    fn visit_func_decl(&mut self, func_decl: &'a FuncDecl) {
        func_decl.inner_visit_with(self);
    }
    fn visit_control_node_def(&mut self, func_at_control_node: FuncAt<'a, ControlNode>) {
        func_at_control_node.inner_visit_with(self);
    }
    fn visit_data_inst_def(&mut self, data_inst_def: &'a DataInstDef) {
        data_inst_def.inner_visit_with(self);
    }
    fn visit_value_use(&mut self, v: &'a Value) {
        v.inner_visit_with(self);
    }
}

/// Trait implemented on "visitable" types, to further "explore" a type by
/// visiting its "interior" (i.e. variants and/or fields).
///
/// That is, an `impl InnerVisit for X` will call the relevant `Visitor` method
/// for each `X` field, effectively performing a single level of a deep visit.
/// Also, if `Visitor::visit_X` exists for a given `X`, its default should be to
/// call `X::inner_visit_with` (i.e. so that visiting is mostly-deep by default).
pub trait InnerVisit {
    // FIXME(eddyb) the naming here isn't great, can it be improved?
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>);
}

/// Dynamic dispatch version of `InnerVisit`.
///
/// `dyn DynInnerVisit<'a, V>` is possible, unlike `dyn InnerVisit`, because of
/// the `trait`-level type parameter `V`, which replaces the method parameter.
pub trait DynInnerVisit<'a, V> {
    fn dyn_inner_visit_with(&'a self, visitor: &mut V);
}

impl<'a, T: InnerVisit, V: Visitor<'a>> DynInnerVisit<'a, V> for T {
    fn dyn_inner_visit_with(&'a self, visitor: &mut V) {
        self.inner_visit_with(visitor);
    }
}

// FIXME(eddyb) should the impls be here, or next to definitions? (maybe derived?)
impl InnerVisit for Module {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        // FIXME(eddyb) this can't be exhaustive because of the private `cx` field.
        let Self {
            dialect,
            debug_info,
            global_vars: _,
            funcs: _,
            exports,
            ..
        } = self;

        visitor.visit_module_dialect(dialect);
        visitor.visit_module_debug_info(debug_info);
        for (export_key, exportee) in exports {
            export_key.inner_visit_with(visitor);
            exportee.inner_visit_with(visitor);
        }
    }
}

impl InnerVisit for ModuleDialect {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::Spv(dialect) => visitor.visit_spv_dialect(dialect),
        }
    }
}

impl InnerVisit for ModuleDebugInfo {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::Spv(debug_info) => {
                visitor.visit_spv_module_debug_info(debug_info);
            }
        }
    }
}

impl InnerVisit for ExportKey {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::LinkName(_) => {}

            Self::SpvEntryPoint {
                imms: _,
                interface_global_vars,
            } => {
                for &gv in interface_global_vars {
                    visitor.visit_global_var_use(gv);
                }
            }
        }
    }
}

impl InnerVisit for Exportee {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match *self {
            Self::GlobalVar(gv) => visitor.visit_global_var_use(gv),
            Self::Func(func) => visitor.visit_func_use(func),
        }
    }
}

impl InnerVisit for AttrSetDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs } = self;

        for attr in attrs {
            visitor.visit_attr(attr);
        }
    }
}

impl InnerVisit for TypeDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            ctor,
            ctor_args,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        match ctor {
            TypeCtor::SpvInst(_) => {}
        }
        for &arg in ctor_args {
            match arg {
                TypeCtorArg::Type(ty) => visitor.visit_type_use(ty),
                TypeCtorArg::Const(ct) => visitor.visit_const_use(ct),
            }
        }
    }
}

impl InnerVisit for ConstDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            ty,
            ctor,
            ctor_args,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        visitor.visit_type_use(*ty);
        match *ctor {
            ConstCtor::PtrToGlobalVar(gv) => visitor.visit_global_var_use(gv),
            ConstCtor::SpvInst(_) => {}
        }
        for &ct in ctor_args {
            visitor.visit_const_use(ct);
        }
    }
}

impl<D: InnerVisit> InnerVisit for DeclDef<D> {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match self {
            Self::Imported(import) => visitor.visit_import(import),
            Self::Present(def) => def.inner_visit_with(visitor),
        }
    }
}

impl InnerVisit for GlobalVarDecl {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            type_of_ptr_to,
            addr_space,
            def,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        visitor.visit_type_use(*type_of_ptr_to);
        match addr_space {
            AddrSpace::SpvStorageClass(_) => {}
        }
        def.inner_visit_with(visitor);
    }
}

impl InnerVisit for GlobalVarDefBody {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { initializer } = self;

        if let Some(initializer) = *initializer {
            visitor.visit_const_use(initializer);
        }
    }
}

impl InnerVisit for FuncDecl {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            ret_type,
            params,
            def,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        visitor.visit_type_use(*ret_type);
        for param in params {
            param.inner_visit_with(visitor);
        }
        def.inner_visit_with(visitor);
    }
}

impl InnerVisit for FuncParam {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, ty } = *self;

        visitor.visit_attr_set_use(attrs);
        visitor.visit_type_use(ty);
    }
}

impl InnerVisit for FuncDefBody {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            data_insts: _,
            control_nodes,
            body: _,
            cfg,
        } = self;

        for point in cfg.rev_post_order(self) {
            // HACK(eddyb) this needs to visit `UnstructuredMerge`s on `Exit`
            // instead of `Entry`, because they don't have have `Entry`s.
            let can_uniquely_visit = match control_nodes[point.control_node()].kind {
                ControlNodeKind::UnstructuredMerge => {
                    assert!(matches!(point, cfg::ControlPoint::Exit(_)));
                    true
                }
                _ => matches!(point, cfg::ControlPoint::Entry(_)),
            };

            if can_uniquely_visit {
                visitor.visit_control_node_def(self.at(point.control_node()));
            }

            if let Some(control_inst) = cfg.control_insts.get(point) {
                control_inst.inner_visit_with(visitor);
            }
        }
    }
}

// FIXME(eddyb) this can't implement `InnerVisit` because of the `&'a self`
// requirement, whereas this has `'a` in `self: FuncAt<'a, ControlNode>`.
impl<'a> FuncAt<'a, ControlNode> {
    fn inner_visit_with(self, visitor: &mut impl Visitor<'a>) {
        let ControlNodeDef { kind, outputs } = &*self.control_nodes[self.position];

        match kind {
            ControlNodeKind::UnstructuredMerge => {}
            ControlNodeKind::Block { insts } => {
                for &inst in insts {
                    visitor.visit_data_inst_def(&self.data_insts[inst]);
                }
            }
        }
        for output in outputs {
            output.inner_visit_with(visitor);
        }
    }
}

impl InnerVisit for ControlNodeOutputDecl {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self { attrs, ty } = *self;

        visitor.visit_attr_set_use(attrs);
        visitor.visit_type_use(ty);
    }
}

impl InnerVisit for DataInstDef {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            kind,
            output_type,
            inputs,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        match *kind {
            DataInstKind::FuncCall(func) => visitor.visit_func_use(func),
            DataInstKind::SpvInst(_) | DataInstKind::SpvExtInst { .. } => {}
        }
        if let Some(ty) = *output_type {
            visitor.visit_type_use(ty);
        }
        for v in inputs {
            visitor.visit_value_use(v);
        }
    }
}

impl InnerVisit for cfg::ControlInst {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        let Self {
            attrs,
            kind,
            inputs,
            targets: _,
            target_merge_outputs,
        } = self;

        visitor.visit_attr_set_use(*attrs);
        match kind {
            cfg::ControlInstKind::Unreachable
            | cfg::ControlInstKind::Return
            | cfg::ControlInstKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(_))
            | cfg::ControlInstKind::Branch
            | cfg::ControlInstKind::SelectBranch(
                cfg::SelectionKind::BoolCond | cfg::SelectionKind::SpvInst(_),
            ) => {}
        }
        for v in inputs {
            visitor.visit_value_use(v);
        }
        for outputs in target_merge_outputs.values() {
            for v in outputs {
                visitor.visit_value_use(v);
            }
        }
    }
}

impl InnerVisit for Value {
    fn inner_visit_with<'a>(&'a self, visitor: &mut impl Visitor<'a>) {
        match *self {
            Self::Const(ct) => visitor.visit_const_use(ct),
            Self::FuncParam { idx: _ }
            | Self::ControlNodeOutput {
                control_node: _,
                output_idx: _,
            }
            | Self::DataInstOutput(_) => {}
        }
    }
}
