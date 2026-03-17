use arrayvec::ArrayVec;
use itertools::Itertools as _;
use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use spirt::cf::{self, SelectionKind};
use spirt::func_at::FuncAt;
use spirt::mem::MemOp;
use spirt::qptr::QPtrOp;
use spirt::{
    AddrSpace, Attr, Const, ConstKind, Context, DeclDef, EntityOrientedDenseMap, FuncDecl,
    GlobalVar, GlobalVarDefBody, GlobalVarInit, InternedStr, Module, Node, NodeDef, NodeKind,
    Region, RegionDef, TypeKind, Value, Var, scalar, spv, vector, visit,
};
use std::fmt;
use std::fmt::Write as _;
use std::num::{NonZeroI32, NonZeroU32};
use std::path::Path;
use std::rc::Rc;

// HACK(eddyb) `spv::spec::Spec` with extra `WellKnown`s.
macro_rules! def_spv_spec_with_extra_well_known {
    ($($group:ident: $ty:ty = [$($entry:ident),+ $(,)?]),+ $(,)?) => {
        struct SpvSpecWithExtras {
            __base_spec: &'static spv::spec::Spec,

            well_known: SpvWellKnownWithExtras,
        }

        #[allow(non_snake_case)]
        pub struct SpvWellKnownWithExtras {
            __base_well_known: &'static spv::spec::WellKnown,

            $($(pub $entry: $ty,)+)+
        }

        impl std::ops::Deref for SpvSpecWithExtras {
            type Target = spv::spec::Spec;
            fn deref(&self) -> &Self::Target {
                self.__base_spec
            }
        }

        impl std::ops::Deref for SpvWellKnownWithExtras {
            type Target = spv::spec::WellKnown;
            fn deref(&self) -> &Self::Target {
                self.__base_well_known
            }
        }

        impl SpvSpecWithExtras {
            #[inline(always)]
            #[must_use]
            pub fn get() -> &'static SpvSpecWithExtras {
                lazy_static! {
                    static ref SPEC: SpvSpecWithExtras = {
                        #[allow(non_camel_case_types)]
                        struct PerWellKnownGroup<$($group),+> {
                            $($group: $group),+
                        }

                        let spv_spec = spv::spec::Spec::get();
                        let wk = &spv_spec.well_known;

                        let storage_classes = match &spv_spec.operand_kinds[wk.StorageClass] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };
                        let decorations = match &spv_spec.operand_kinds[wk.Decoration] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };

                        let execution_models = match &spv_spec.operand_kinds[spv_spec.operand_kinds.lookup("ExecutionModel").unwrap()] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };
                        let builtins = match &spv_spec.operand_kinds[spv_spec.operand_kinds.lookup("BuiltIn").unwrap()] {
                            spv::spec::OperandKindDef::ValueEnum { variants } => variants,
                            _ => unreachable!(),
                        };

                        let glsl_std_450_ops = spv_spec
                            .get_ext_inst_set_by_lowercase_name("glsl.std.450")
                            .unwrap()
                            .instructions
                            .iter()
                            .map(|(&op, inst_desc)| (&inst_desc.name[..], op))
                            .collect::<FxHashMap<_, _>>();

                        let lookup_fns = PerWellKnownGroup {
                            opcode: |name| spv_spec.instructions.lookup(name).unwrap(),
                            operand_kind: |name| spv_spec.operand_kinds.lookup(name).unwrap(),
                            storage_class: |name| storage_classes.lookup(name).unwrap().into(),
                            decoration: |name| decorations.lookup(name).unwrap().into(),
                            execution_model: |name| execution_models.lookup(name).unwrap().into(),
                            builtin: |name| builtins.lookup(name).unwrap().into(),
                            glsl_std_450_op: |name| glsl_std_450_ops.get(name).copied().unwrap(),
                        };

                        SpvSpecWithExtras {
                            __base_spec: spv_spec,

                            well_known: SpvWellKnownWithExtras {
                                __base_well_known: &spv_spec.well_known,

                                $($($entry: (lookup_fns.$group)(stringify!($entry)),)+)+
                            },
                        }
                    };
                }
                &SPEC
            }
        }
    };
}
def_spv_spec_with_extra_well_known! {
    opcode: spv::spec::Opcode = [
        OpSpecConstant,

        OpSelect,

        OpAtomicLoad,
        OpAtomicCompareExchange,
        OpAtomicIAdd,
    ],
    operand_kind: spv::spec::OperandKind = [
        ExecutionModel,
    ],
    storage_class: u32 = [
        PushConstant,
    ],
    decoration: u32 = [
        BuiltIn,
    ],
    execution_model: u32 = [
        Vertex,
        Fragment,
        GLCompute,
    ],
    builtin: u32 = [
        FragCoord,
        GlobalInvocationId,
    ],
    glsl_std_450_op: u32 = [
        FindILsb,
        FindSMsb,
        FindUMsb,

        FAbs,
        Floor,
        Ceil,
        Round,
        Trunc,

        Exp,
        Sqrt,
        Sin,
        Cos,

        FMin,
        FMax,
        Pow,

        Fma,
    ],
}

pub enum PtxEntryParamSource {
    // FIXME(eddyb) extract `(DescriptorSet, Binding)` and/or fully emulate
    // descriptor set buffers, descriptor heaps, etc.
    // FIXME(eddyb) be more precise with address spaces.
    BufferDataPtrGeneric64 { buffer: GlobalVar },
    BufferLen { buffer: GlobalVar, fixed_base_size: u32, dyn_unit_stride: NonZeroU32 },
}

pub fn spv_file_to_ptx(in_file_path: &Path) -> (String, Vec<PtxEntryParamSource>) {
    fn eprint_duration<R>(f: impl FnOnce() -> R) -> R {
        let start = std::time::Instant::now();
        let r = f();
        eprint!("[{:8.3}ms] ", start.elapsed().as_secs_f64() * 1000.0);
        r
    }

    let mut module = eprint_duration(|| {
        Module::lower_from_spv_file(Rc::new(Context::new()), in_file_path).unwrap()
    });
    eprintln!("Module::lower_from_spv_file({})", in_file_path.display());

    let cx = module.cx();

    eprint_duration(|| spirt::passes::legalize::structurize_func_cfgs(&mut module));
    eprintln!("legalize::structurize_func_cfgs");

    let layout_config = &spirt::mem::LayoutConfig::VULKAN_SCALAR_LAYOUT_LE;

    eprint_duration(|| spirt::passes::qptr::lower_from_spv_ptrs(&mut module, layout_config));
    eprintln!("qptr::lower_from_spv_ptrs");

    if false {
        // HACK(eddyb) this shouldn't be necessary, but if `spirv-opt` was used,
        // it may have left behind spurious function-local variables.
        eprint_duration(|| {
            spirt::passes::qptr::partition_and_propagate(&mut module, layout_config)
        });
        eprintln!("qptr::partition_and_propagate");
    }

    {
        use spirt::visit;

        // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
        let visit::AllUses { funcs, .. } = visit::AllUses::from_module(&module);

        eprint_duration(|| {
            for &func in &funcs {
                if let DeclDef::Present(func_def_body) = &mut module.funcs[func].def {
                    spirt::cf::hermetic::seal(&cx, func_def_body.at_mut_body());
                }
            }
        });
        eprintln!("cf::hermetic::seal");
    }

    let ptx = eprint_duration(|| module_to_ptx(&module));
    eprintln!("spirt_ptx::module_to_ptx");
    ptx
}

fn module_to_ptx(module: &Module) -> (String, Vec<PtxEntryParamSource>) {
    let wk = &SpvSpecWithExtras::get().well_known;

    let cx = module.cx_ref();

    let visit::AllUses { global_vars, funcs, .. } = visit::AllUses::from_module(module);

    // FIXME(eddyb) should there be a `PtxModule` type?
    let mut ptx = "\
.version 9.0
.target sm_80
.address_size 64

"
    .to_string();

    // FIXME(eddyb) this is confusing wrt "global" as a PTX address space.
    let mut num_globals: u32 = 0;

    let mut ptx_module_interface = PtxModuleInterface::default();
    for gv in global_vars {
        let gv_decl = &module.global_vars[gv];

        let builtin = if gv_decl.addr_space == AddrSpace::SpvStorageClass(wk.Input) {
            cx[gv_decl.attrs].attrs.iter().find_map(|attr| match attr {
                Attr::SpvAnnotation(spv_inst) if spv_inst.opcode == wk.OpDecorate => match spv_inst
                    .imms
                    .strip_prefix(&[spv::Imm::Short(wk.Decoration, wk.BuiltIn)])?
                {
                    &[spv::Imm::Short(builtin_kind, builtin)] => Some((builtin_kind, builtin)),
                    _ => None,
                },
                _ => None,
            })
        } else {
            None
        };

        let init = match &gv_decl.def {
            DeclDef::Imported(_) => None,
            DeclDef::Present(GlobalVarDefBody { initializer }) => initializer.as_ref(),
        };

        let encoding = if let Some((builtin_kind, builtin)) = builtin {
            // FIXME(eddyb) use imports for such declarations?
            assert!(init.is_none());

            let print_builtin = || {
                spv::print::operand_from_imms([spv::Imm::Short(builtin_kind, builtin)])
                    .concat_to_plain_text()
            };
            if builtin == wk.GlobalInvocationId {
                PtxGlobalVarEncoding::GlobalInvocationId
            } else {
                todo!("unknown {}", print_builtin());
            }
        } else {
            use spirt::mem::shapes::{GlobalVarShape, Handle};

            match gv_decl.shape.unwrap() {
                GlobalVarShape::Handles {
                    handle: Handle::Buffer(_, layout),
                    fixed_count: Some(count),
                } if count.get() == 1 => {
                    // FIXME(eddyb) use imports for such declarations?
                    assert!(init.is_none());

                    let data_ptr_generic64_entry_param_idx =
                        ptx_module_interface.entry_param_sources.len().try_into().unwrap();
                    ptx_module_interface
                        .entry_param_sources
                        .push(PtxEntryParamSource::BufferDataPtrGeneric64 { buffer: gv });

                    let dyn_len_entry_param_idx = layout.dyn_unit_stride.map(|stride| {
                        let dyn_len_entry_param_idx =
                            ptx_module_interface.entry_param_sources.len().try_into().unwrap();
                        ptx_module_interface.entry_param_sources.push(
                            PtxEntryParamSource::BufferLen {
                                buffer: gv,
                                fixed_base_size: layout.fixed_base.size,
                                dyn_unit_stride: stride,
                            },
                        );
                        dyn_len_entry_param_idx
                    });

                    PtxGlobalVarEncoding::Buffer {
                        layout,
                        data_ptr_generic64_entry_param_idx,
                        dyn_len_entry_param_idx,
                    }
                }
                GlobalVarShape::Handles { handle: _, fixed_count: _ } => todo!(),
                GlobalVarShape::UntypedData(layout) => {
                    // FIXME(eddyb) should these just be fused together?
                    // FIXME(eddyb) use `.global`/`.const` instead of `.local`,
                    // for read-only globals (`qptr::legalize` could avoid fusing
                    // those with actually mutable globals - but it's a tradeoff).
                    let idx = num_globals;
                    num_globals = idx.checked_add(1).unwrap();

                    let addr_space = if gv_decl.addr_space == AddrSpace::SpvStorageClass(wk.Private)
                    {
                        "local"
                    } else if gv_decl.addr_space == AddrSpace::SpvStorageClass(wk.Workgroup) {
                        // HACK(eddyb) "overshare" when necessary.
                        if init.is_some() { "global" } else { "shared" }
                    } else {
                        todo!()
                    };

                    // HACK(eddyb) approximating a `try {...}` block.
                    (|| {
                        write!(
                            ptx,
                            ".{addr_space} .align {} .b8 %G{idx}[{}]",
                            layout.align, layout.size
                        )?;

                        if let Some(init) = init {
                            // FIXME(eddyb) can this do better than byte array?
                            match init {
                                // HACK(eddyb) avoid unnecessary zero-init.
                                &GlobalVarInit::Direct(ct)
                                    if ct.as_scalar(cx).is_some_and(|ct| ct.bits() == 0)
                                        && addr_space == "global" => {}

                                GlobalVarInit::Direct(_) => todo!(),
                                GlobalVarInit::SpvAggregate { .. } => unreachable!(),
                                GlobalVarInit::Data(const_data) => {
                                    write!(ptx, " = {{")?;
                                    let mut offset = 0;
                                    let mut init_size = 0;
                                    for part in const_data.read(0..const_data.size()) {
                                        match part {
                                            spirt::mem::const_data::Part::Uninit { .. } => {}
                                            spirt::mem::const_data::Part::Bytes(bytes) => {
                                                // HACK(eddyb) this chooses `0`
                                                // as the `undef` encoding, and
                                                // only avoids emitting the last
                                                // `undef` range as `0` bytes.
                                                let uninit_filler = (init_size..offset).map(|_| 0);

                                                for byte in
                                                    uninit_filler.chain(bytes.iter().copied())
                                                {
                                                    if init_size > 1 {
                                                        write!(ptx, ", ")?;
                                                    }
                                                    write!(ptx, "{byte}")?;
                                                    init_size += 1;
                                                }
                                            }
                                            // FIXME(eddyb) use the "mask operators"
                                            // to encode the individual bytes.
                                            spirt::mem::const_data::Part::Symbolic { .. } => {
                                                todo!()
                                            }
                                        }
                                        offset += part.size().get();
                                    }
                                    write!(ptx, "}}")?;
                                }
                            }
                        }

                        writeln!(ptx, ";")
                    })()
                    .unwrap();

                    PtxGlobalVarEncoding::ModuleScoped { prefix: "%G", idx }
                }
                GlobalVarShape::TypedInterface(_) => todo!(),
            }
        };

        // FIXME(eddyb) update uses of `insert` (e.g. in `spirti`) to `assert!`
        // uniqueness like this (maybe using a new method that always does it?).
        assert!(ptx_module_interface.global_var_encodings.insert(gv, encoding).is_none());
    }

    let mut ptx_funcs = vec![];
    for func in funcs {
        let func_decl = &module.funcs[func];

        let sched = match &func_decl.def {
            DeclDef::Imported(_) => spirt::sched::Schedule::default(),
            DeclDef::Present(func_def_body) => {
                spirt::sched::Schedule::compute(func_def_body.at_body())
            }
        };

        ptx_funcs.push(FuncToPtx::new(cx, &ptx_module_interface, &sched).func_to_ptx(func_decl));
    }

    // FIXME(eddyb) use `module.exports` to label everything appropriately.

    for ptx_func in ptx_funcs {
        writeln!(ptx, "{ptx_func}").unwrap();
    }

    (ptx, ptx_module_interface.entry_param_sources)
}

// FIXME(eddyb) is this a good representation? (consider e.g. lg2 bitwidth)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PtxRegBank {
    Pred,
    B8,
    B16,
    B32,
    B64,
    V4B32,
}

impl spirt::sched::RegBank for PtxRegBank {
    const ALL: &'static [Self] = Self::ALL;

    fn index(self) -> usize {
        self as usize
    }
}

impl PtxRegBank {
    const ALL: &'static [Self] =
        &[Self::Pred, Self::B8, Self::B16, Self::B32, Self::B64, Self::V4B32];

    fn prefix(self) -> &'static str {
        (const { &["rp", "rb", "rs", "r", "rd", "rv"] })[self as usize]
    }

    fn ty(self) -> &'static str {
        (const { &["pred", "b8", "b16", "b32", "b64", "v4.b32"] })[self as usize]
    }
}

// FIXME(eddyb) should PTX insts be buffered like this, or streamed?
#[derive(Default)]
struct PtxFunc {
    param_types: Vec<&'static str>,

    locals: Vec<spirt::mem::shapes::MemLayout>,
    used_banks_with_reg_counts: Vec<(PtxRegBank, NonZeroU32)>,

    insts: Vec<PtxInst>,
}

impl fmt::Display for PtxFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { param_types, locals, used_banks_with_reg_counts, insts } = self;

        writeln!(f, ".visible .entry kernel(")?;
        for (i, param_type) in param_types.iter().enumerate() {
            writeln!(
                f,
                "  .param .{param_type} %p{i}{}",
                if i < param_types.len() - 1 { "," } else { "" }
            )?;
        }
        writeln!(f, ")")?;
        writeln!(f, "{{")?;
        writeln!(f)?;
        // FIXME(eddyb) should these just be fused together?
        for (i, local) in locals.iter().enumerate() {
            writeln!(f, "  .local .align {} .b8 %l{i}[{}];", local.align, local.size)?;
        }
        writeln!(f)?;
        for &(bank, count) in used_banks_with_reg_counts {
            writeln!(f, "  .reg .{} %{}<{count}>;", bank.ty(), bank.prefix())?;
        }
        writeln!(f)?;
        for inst in insts {
            writeln!(f, "  {}", inst)?;
        }
        writeln!(f, "}}")
    }
}

struct PtxInst {
    // FIXME(eddyb) generalize the idea of a small string + `u32` counter.
    prefix: Option<(&'static str, u32)>,

    op: &'static str,
    // FIXME(eddyb) optimize some of these for "one [a-z] letter + log2 bitwidth".
    suffix: Option<&'static str>,
    // FIXME(eddyb) split out the destination operand, if it exists?
    operands: SmallVec<[PtxOperand; 4]>,
}

impl fmt::Display for PtxInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { prefix, op, suffix, operands } = self;

        if let Some((prefix, prefix_idx)) = prefix {
            write!(f, "{prefix}{prefix_idx} ")?;
        }

        write!(f, "{op}")?;

        // HACK(eddyb) avoid emitting an invalid `:;` sequence.
        // FIXME(eddyb) consider making `PtxInst` an `enum` instead?
        // TODO(eddyb) is this even needed? is `:;` *actually* invalid?
        if *op == ":" && suffix.is_none() && operands.is_empty() {
            return Ok(());
        }

        if let Some(suffix) = suffix {
            write!(f, "{suffix}")?;
        }

        for (i, operand) in operands.iter().enumerate() {
            write!(f, "{} {operand}", if i > 0 { "," } else { "" })?;
        }

        write!(f, ";")
    }
}

// FIXME(eddyb) these variants are almost orthogonal, except for the fact that
// PTX lacks `reg+imm`, but has `[reg+imm]` and `X+imm` (for non-register `X`).
#[derive(Clone)]
enum PtxOperand {
    // FIXME(eddyb) factor out into a `struct`/`enum` `PtxConst`.
    Const { ty: scalar::Type, bits: u64 },
    Reg(PtxReg),
    Special(&'static str),
    MemRef { base: PtxMemBase, offset: Option<NonZeroI32> },
    // TODO(eddyb) make this wrap a struct that's more like an "anon name".
    PtrToNamedVar { base: (&'static str, u32), offset: Option<NonZeroI32> },

    // TODO(eddyb) make this wrap a struct that's more like an "anon name".
    Label { idx: u32 },
}

impl fmt::Display for PtxOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            &PtxOperand::Const { ty, bits } => match ty {
                scalar::Type::Bool | scalar::Type::UInt(_) => write!(f, "{bits}U"),
                scalar::Type::SInt(_) => write!(f, "{}", bits as i64),
                scalar::Type::F32 => write!(f, "0f{:08x}", bits as u32),
                scalar::Type::F64 => write!(f, "0d{:016x}", bits),
                scalar::Type::Float(_) => todo!(),
            },
            PtxOperand::Reg(reg) => write!(f, "{reg}"),
            PtxOperand::Special(sreg) => write!(f, "{sreg}"),
            PtxOperand::MemRef { base, offset } => {
                write!(f, "[{base}")?;
                if let Some(offset) = offset {
                    write!(
                        f,
                        "{}{}",
                        if offset.get() < 0 { "-" } else { "+" },
                        offset.unsigned_abs()
                    )?;
                }
                write!(f, "]")
            }
            PtxOperand::PtrToNamedVar { base: (prefix, idx), offset } => {
                write!(f, "{prefix}{idx}")?;
                if let Some(offset) = offset {
                    write!(
                        f,
                        "{}{}",
                        if offset.get() < 0 { "-" } else { "+" },
                        offset.unsigned_abs()
                    )?;
                }
                Ok(())
            }
            PtxOperand::Label { idx } => write!(f, "%L{idx}"),
        }
    }
}

#[derive(Copy, Clone)]
struct PtxReg {
    bank: PtxRegBank,
    idx: u32,

    // FIXME(eddyb) remove this, maybe sneak it in somewhere else?
    suffix: Option<&'static str>,
}

impl fmt::Display for PtxReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { bank, idx, suffix } = self;

        write!(f, "%{}{idx}", bank.prefix())?;
        if let Some(suffix) = suffix {
            write!(f, "{suffix}")?;
        }

        Ok(())
    }
}

// FIXME(eddyb) should `RegName` and `PtxReg` be different types?
impl From<spirt::sched::RegName<PtxRegBank>> for PtxReg {
    fn from(spirt::sched::RegName { bank, idx }: spirt::sched::RegName<PtxRegBank>) -> Self {
        PtxReg { bank, idx, suffix: None }
    }
}

// FIXME(eddyb) should `RegName` and `PtxReg` be different types?
impl Into<spirt::sched::RegName<PtxRegBank>> for PtxReg {
    fn into(self) -> spirt::sched::RegName<PtxRegBank> {
        let PtxReg { bank, idx, suffix } = self;
        assert_eq!(suffix, None);
        spirt::sched::RegName { bank, idx }
    }
}

#[derive(Clone)]
enum PtxMemBase {
    // TODO(eddyb) make this wrap a struct that's more like an "anon name".
    NamedVar { prefix: &'static str, idx: u32 },
    Ptr(PtxReg),
}

impl fmt::Display for PtxMemBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PtxMemBase::NamedVar { prefix, idx } => write!(f, "{prefix}{idx}"),
            PtxMemBase::Ptr(reg) => write!(f, "{reg}"),
        }
    }
}

#[derive(Default)]
struct PtxModuleInterface {
    global_var_encodings: EntityOrientedDenseMap<GlobalVar, PtxGlobalVarEncoding>,
    entry_param_sources: Vec<PtxEntryParamSource>,
}

#[derive(Copy, Clone)]
enum PtxGlobalVarEncoding {
    ModuleScoped {
        prefix: &'static str,
        idx: u32,
    },

    // FIXME(eddyb) be more precise with address spaces.
    Buffer {
        layout: spirt::mem::shapes::MaybeDynMemLayout,
        data_ptr_generic64_entry_param_idx: u32,
        dyn_len_entry_param_idx: Option<u32>,
    },

    // FIXME(eddyb) generalize this to any built-in?
    GlobalInvocationId,
}

// FIXME(eddyb) consider mapping `Var`s to this, to avoid extra registers.
enum PtxValueEncoding<'a> {
    // FIXME(eddyb) factor out into a `struct`/`enum` `PtxConst`.
    Const { ty: scalar::Type, bits: u64 },
    Reg(PtxReg),
    PtrToGlobalVar { global: &'a PtxGlobalVarEncoding, offset: Option<NonZeroI32> },

    Undef,
    SpvStringLiteralForExtInst(InternedStr),
}

impl PtxValueEncoding<'_> {
    #[track_caller]
    fn as_src_operand(&self) -> PtxOperand {
        match *self {
            PtxValueEncoding::Const { ty, bits } => PtxOperand::Const { ty, bits },
            PtxValueEncoding::Reg(reg) => PtxOperand::Reg(reg),
            PtxValueEncoding::PtrToGlobalVar { global, offset } => match *global {
                PtxGlobalVarEncoding::ModuleScoped { prefix, idx } => {
                    PtxOperand::PtrToNamedVar { base: (prefix, idx), offset }
                }
                PtxGlobalVarEncoding::Buffer { .. } => {
                    unreachable!("buffer handle used outside of `qptr.buffer_data`");
                }
                PtxGlobalVarEncoding::GlobalInvocationId => {
                    unreachable!("input builtin used outside of `mem.load`");
                }
            },
            PtxValueEncoding::Undef => {
                todo!("`undef` failed to get special-cased");
            }
            PtxValueEncoding::SpvStringLiteralForExtInst(_) => {
                unreachable!("SPIR-V `OpString` used outside of `OpExtInst`");
            }
        }
    }

    fn as_mem_operand(&self, offset: i32) -> PtxOperand {
        let (base, base_offset) = match self.as_src_operand() {
            PtxOperand::Const { .. } => todo!(),
            PtxOperand::Reg(reg) => (PtxMemBase::Ptr(reg), 0),
            PtxOperand::Special(_) | PtxOperand::MemRef { .. } | PtxOperand::Label { .. } => {
                unreachable!()
            }
            PtxOperand::PtrToNamedVar { base: (prefix, idx), offset } => {
                (PtxMemBase::NamedVar { prefix, idx }, offset.map_or(0, |o| o.get()))
            }
        };
        PtxOperand::MemRef {
            base,
            offset: NonZeroI32::new(base_offset.checked_add(offset).unwrap()),
        }
    }
}

#[derive(Copy, Clone)]
enum PtxCond {
    If(PtxReg),
    IfNot(PtxReg),
}

impl PtxCond {
    fn inst_prefix(&self) -> (&'static str, u32) {
        let (pred, flip) = match *self {
            PtxCond::If(pred) => (pred, false),
            PtxCond::IfNot(pred) => (pred, true),
        };

        let PtxReg { bank, idx, suffix } = pred;
        assert_eq!((bank, suffix), (PtxRegBank::Pred, None));

        (if flip { "@!%rp" } else { "@%rp" }, idx)
    }
}

struct FuncToPtx<'a> {
    wk: &'static SpvWellKnownWithExtras,

    // FIXME(eddyb) consider an `interned` field to hold this.
    glsl_std_450: InternedStr,

    cx: &'a Context,
    ptx_module_interface: &'a PtxModuleInterface,

    sched: &'a spirt::sched::Schedule,

    func: PtxFunc,

    num_labels: u32,
    loaded_params: Vec<PtxReg>,
    regs: spirt::sched::OnlineRegAlloc<'a, PtxRegBank>,
}

impl<'a> FuncToPtx<'a> {
    // FIXME(eddyb) consider merging `new` and `func_to_ptx`?
    fn new(
        cx: &'a Context,
        ptx_module_interface: &'a PtxModuleInterface,
        sched: &'a spirt::sched::Schedule,
    ) -> Self {
        Self {
            wk: &SpvSpecWithExtras::get().well_known,

            glsl_std_450: cx.intern("GLSL.std.450"),

            cx,
            ptx_module_interface,

            sched,

            func: Default::default(),

            num_labels: Default::default(),
            loaded_params: Default::default(),
            regs: spirt::sched::OnlineRegAlloc::new(sched),
        }
    }

    fn func_to_ptx(mut self, func_decl: &FuncDecl) -> PtxFunc {
        match &func_decl.def {
            DeclDef::Imported(_) => todo!(),
            DeclDef::Present(func_def_body) => {
                for (i, param) in self.ptx_module_interface.entry_param_sources.iter().enumerate() {
                    let ty_kind = match param {
                        PtxEntryParamSource::BufferDataPtrGeneric64 { .. } => &TypeKind::QPtr,
                        PtxEntryParamSource::BufferLen { .. } => {
                            &TypeKind::Scalar(scalar::Type::U32)
                        }
                    };
                    let reg = self.alloc_reg_with_type_kind(ty_kind);
                    self.regs.pin_temp(reg.into());

                    assert_eq!(self.func.param_types.len(), i);
                    self.func.param_types.push(reg.bank.ty());

                    assert_eq!(self.loaded_params.len(), i);
                    self.loaded_params.push(reg);

                    self.func.insts.push(PtxInst {
                        prefix: None,
                        op: "ld.param.",
                        suffix: Some(reg.bank.ty()),
                        operands: [
                            PtxOperand::Reg(reg),
                            PtxOperand::MemRef {
                                base: PtxMemBase::NamedVar {
                                    prefix: "%p",
                                    idx: i.try_into().unwrap(),
                                },
                                offset: None,
                            },
                        ]
                        .into_iter()
                        .collect(),
                    });
                }

                self.region_to_ptx_with(func_def_body.at_body(), [].into_iter(), |this| {
                    this.func.insts.push(PtxInst {
                        prefix: None,
                        op: "ret",
                        suffix: None,
                        operands: [].into_iter().collect(),
                    });
                });

                assert!(self.func.used_banks_with_reg_counts.is_empty());
                self.func.used_banks_with_reg_counts =
                    self.regs.used_banks_with_reg_counts().collect();
            }
        }
        self.func
    }

    fn region_to_ptx_with<R>(
        &mut self,
        func_at_region: FuncAt<'_, Region>,
        // FIXME(eddyb) try sketching a more flexible "destination passing" system.
        output_regs: impl ExactSizeIterator<Item = Option<PtxReg>>,

        // HACK(eddyb) needed for handling branches/returns and some dataflow.
        before_exit: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let RegionDef { inputs: input_vars, children: _, outputs } = func_at_region.def();

        // FIXME(eddyb) use `self.sched` for the actual order!
        for func_at_node in func_at_region.at_children() {
            self.node_to_ptx(func_at_node);

            // TODO(eddyb) also release registers used for variables.
            self.regs.release_temps();
        }

        assert_eq!(outputs.len(), output_regs.len());
        for (maybe_output_reg, &output) in output_regs.zip_eq(outputs) {
            if let Some(output_reg) = maybe_output_reg {
                // FIXME(eddyb) abstract this into a `value_to_ptx` method?
                let output_encoding = match output {
                    Value::Const(ct) => self.const_to_ptx(ct),
                    Value::Var(var) => PtxValueEncoding::Reg(self.regs.use_var(var).into()),
                };
                self.mov(None, output_reg, &output_encoding);
            }
        }

        let extra = before_exit(self);

        // TODO(eddyb) use "last use position" information.
        if true {
            let vars_defined_in_region = input_vars
                .iter()
                .chain(
                    func_at_region
                        .at_children()
                        .into_iter()
                        .flat_map(|fan| fan.def().outputs.iter()),
                )
                .copied();

            // TODO(eddyb) this doesn't actually work, because the aliases
            // would have to be released in the exact order they're created...
            // WAIT, no, that's wrong... aliases should only be used for things
            // like capture-shaped `Var`s...., right?
            for var in vars_defined_in_region.rev() {
                // HACK(eddyb) this allows `Var`s to not be defined, is that fine?
                self.regs.release_var_if_defined(var);
            }
        }

        extra
    }

    fn node_to_ptx(&mut self, func_at_node: FuncAt<'_, Node>) {
        let cx = self.cx;

        let func = func_at_node.at(());

        let NodeDef { attrs, kind, inputs, child_regions, outputs: output_vars } =
            func_at_node.def();

        // FIXME(eddyb) make this more useful/share it with the rest of the codebase.
        let describe_spv_inst = || match kind {
            NodeKind::SpvInst(spv_inst, _) => {
                format!(
                    "SPIR-V instruction {}({})",
                    spv_inst.opcode.name(),
                    spv::print::inst_operands(
                        spv_inst.opcode,
                        spv_inst.imms.iter().copied(),
                        inputs.iter().enumerate().map(|(i, _)| format!("<input #{i}>"))
                    )
                    .map(|operand| operand.concat_to_plain_text())
                    .collect::<Vec<_>>()
                    .join(", ")
                )
            }
            &NodeKind::SpvExtInst { ext_set, inst, .. } => {
                let ext_set = &cx[ext_set];
                format!(
                    "SPIR-V extended ({ext_set}) instruction #{inst} ({:?})",
                    spv::spec::Spec::get()
                        .get_ext_inst_set_by_lowercase_name(&ext_set.to_ascii_lowercase())
                        .and_then(|ext_inst_set_desc| Some(
                            &ext_inst_set_desc.instructions.get(&inst)?.name
                        )),
                )
            }
            _ => unreachable!(),
        };

        let input_encodings: SmallVec<[_; 4]> = inputs
            .iter()
            .map(|&v| match v {
                Value::Const(ct) => self.const_to_ptx(ct),
                Value::Var(var) => PtxValueEncoding::Reg(self.regs.use_var(var).into()),
            })
            .collect();

        let outputs: SmallVec<[_; 4]> = match kind {
            NodeKind::Select(kind) => {
                let selector = input_encodings[0].as_src_operand();

                let output_regs: SmallVec<[_; 4]> =
                    output_vars.iter().map(|&var| self.alloc_reg_for_var(func.at(var))).collect();

                let case_consts = match kind {
                    SelectionKind::BoolCond => &[scalar::Const::TRUE][..],
                    SelectionKind::Switch { case_consts } => case_consts,
                };

                // FIXME(eddyb) consider some of these improvements:
                // - ranges for labels
                // - not making the default case first
                // - using `bra.idx` (for sequential `case_consts` values)
                // - encoding the lack of a default case in SPIR-T
                // - deduplicate code w/ `spv::lift`'s CFG regeneration
                // - heuristic for predicating instructions inside each case
                //   (instead of using branches at all)
                let (&default_case, non_default_cases) = child_regions.split_last().unwrap();
                assert_eq!(non_default_cases.len(), case_consts.len());
                let non_default_case_labels: SmallVec<[_; 4]> =
                    non_default_cases.iter().map(|_| self.alloc_label()).collect();

                for (&case_const, case_label) in case_consts.iter().zip_eq(&non_default_case_labels)
                {
                    let cond = match selector {
                        PtxOperand::Const { ty, bits } => {
                            assert!(ty == case_const.ty());
                            if case_const.bits() == bits.into() {
                                None
                            } else {
                                continue;
                            }
                        }
                        PtxOperand::Reg(selector) => Some(match case_const {
                            scalar::Const::TRUE => PtxCond::If(selector),
                            scalar::Const::FALSE => PtxCond::IfNot(selector),
                            _ => {
                                let (selector_eq_const,) = self
                                    .scalar_op_to_ptx(
                                        scalar::IntBinOp::Eq.into(),
                                        [
                                            PtxOperand::Reg(selector),
                                            PtxOperand::Const {
                                                ty: scalar::Type::UInt(
                                                    scalar::IntWidth::try_from_bits(
                                                        case_const.ty().bit_width(),
                                                    )
                                                    .unwrap(),
                                                ),
                                                bits: case_const.bits().try_into().unwrap(),
                                            },
                                        ]
                                        .into_iter()
                                        .collect(),
                                        [scalar::Type::Bool].into_iter().collect(),
                                    )
                                    .into_iter()
                                    .collect_tuple()
                                    .unwrap();
                                PtxCond::If(selector_eq_const)
                            }
                        }),
                        _ => unreachable!(),
                    };
                    self.branch(cond, case_label.clone());
                    if cond.is_none() {
                        break;
                    }
                }

                let merge_label = self.alloc_label();

                for (maybe_case_label, &case) in ([(None, &default_case)].into_iter())
                    .chain(non_default_case_labels.into_iter().map(Some).zip_eq(non_default_cases))
                {
                    if let Some(case_label) = maybe_case_label {
                        self.start_label(case_label);
                    }

                    let func_at_case = func.at(case);
                    let case_def = func_at_case.def();

                    if case_def.inputs.is_empty() {
                        assert_eq!(input_encodings.len(), 1);
                    } else {
                        for (&case_input, select_input) in
                            case_def.inputs.iter().zip_eq(&input_encodings)
                        {
                            if let &PtxValueEncoding::Reg(input_reg) = select_input {
                                self.regs.def_var_aliasing(case_input, input_reg.into());
                            } else {
                                // HACK(eddyb) avoid allocating new registers
                                // for `Var`s that are never used anyway.
                                // FIXME(eddyb) also consider allowing vars
                                // to map to constants, not just registers.
                                if self.sched.vars[case_input].last_use_pos.is_none() {
                                    continue;
                                }

                                let case_input_var = self.alloc_reg_for_var(func.at(case_input));
                                self.mov(None, case_input_var, select_input);
                            };
                        }
                    }

                    self.region_to_ptx_with(
                        func_at_case,
                        output_regs.iter().map(|&r| Some(r)),
                        |this| {
                            // FIXME(eddyb) the very last one of these is unnecessary,
                            // as execution would just fall through anyway.
                            this.branch(None, merge_label.clone());
                        },
                    );
                }

                self.start_label(merge_label);

                // HACK(eddyb) `output_vars` register assignments already handled above.
                return;
            }
            NodeKind::Loop { repeat_condition } => {
                let body = child_regions[0];
                let body_def = func.at(body).def();

                // HACK(eddyb) this has `None` for every "loop state" that is
                // actually unmodified by the body (`body_output == body_input`),
                // and `Some((input_reg, output_reg))` otherwise, where:
                // - `input_reg` is the "current state", used as the body input
                // - `output_reg` is the "next state", taking in the body output
                //   and making it available to both the loop output and further
                //   iterations (via a copy into `input_reg`, before the backedge)
                // FIXME(eddyb) separate "next state" registers could be avoided,
                // in theory, but then the exact ordering of body outputs matter,
                // as they need to be copied into the "current state" registers
                // all at once ("atomically"), with "current state" registers
                // likely being present in the sources of those copies.
                let stateful_input_and_output_regs: SmallVec<[_; 4]> = (input_encodings
                    .iter()
                    .zip_eq(&body_def.inputs)
                    .zip_eq(&body_def.outputs)
                    .zip_eq(output_vars))
                .map(|(((initial_input, &body_input), &body_output), &loop_output)| {
                    match initial_input {
                        // TODO(eddyb) implement some kind of "alias locking"
                        // mechanism, which would block writes to registers
                        // that are being reused in this way.
                        &PtxValueEncoding::Reg(initial_input_reg)
                            if body_output == Value::Var(body_input) =>
                        {
                            // HACK(eddyb) this ordering is needed to be able to
                            // release `body_input` sooner than `loop_output`.
                            // FIXME(eddyb) it might be better to never alias
                            // a loop input to a loop output!
                            if self.sched.vars[loop_output].last_use_pos.is_some() {
                                self.regs.def_var_aliasing(loop_output, initial_input_reg.into());
                            }
                            self.regs.def_var_aliasing(body_input, initial_input_reg.into());

                            None
                        }
                        _ => {
                            let body_input_reg = self.alloc_reg_for_var(func.at(body_input));
                            self.mov(None, body_input_reg, initial_input);

                            Some((body_input_reg, self.alloc_reg_for_var(func.at(loop_output))))
                        }
                    }
                })
                .collect();

                let body_start_label = self.alloc_label();
                self.start_label(body_start_label.clone());

                self.region_to_ptx_with(
                    func.at(body),
                    stateful_input_and_output_regs.iter().map(|&input_and_output| {
                        let (_input_reg, output_reg) = input_and_output?;
                        Some(output_reg)
                    }),
                    |this| {
                        // FIXME(eddyb) as only the output (or "next state") registers
                        // are ever read outside the loop, this could be moved around
                        // into some kind of "unconditionally reaching the backedge"
                        // section (which would be bypassed when loop is being exited),
                        // but it's easier to have a single branch, for the backedge,
                        // and always exit the loop as a "fallthrough", after it.
                        for (input_reg, output_reg) in
                            stateful_input_and_output_regs.iter().copied().flatten()
                        {
                            this.mov(None, input_reg, &PtxValueEncoding::Reg(output_reg));
                        }

                        match *repeat_condition {
                            Value::Const(ct) => match *ct.as_scalar(cx).unwrap() {
                                scalar::Const::TRUE => this.branch(None, body_start_label),
                                scalar::Const::FALSE => {}
                                _ => unreachable!(),
                            },
                            Value::Var(repeat_condition) => {
                                this.branch(
                                    Some(PtxCond::If(this.regs.use_var(repeat_condition).into())),
                                    body_start_label,
                                );
                            }
                        }
                    },
                );

                // HACK(eddyb) `output_vars` register assignments already handled above.
                return;
            }
            NodeKind::ExitInvocation(kind) => match kind {
                cf::ExitInvocationKind::SpvInst(inst) if inst.opcode == self.wk.OpReturn => {
                    // FIXME(eddyb) assert this is an entry and use `exit`?
                    self.func.insts.push(PtxInst {
                        prefix: None,
                        op: "ret",
                        suffix: None,
                        operands: [].into_iter().collect(),
                    });
                    [].into_iter().collect()
                }
                cf::ExitInvocationKind::SpvInst(_) => todo!(),
                cf::ExitInvocationKind::Abort => {
                    self.func.insts.push(PtxInst {
                        prefix: None,
                        op: "exit",
                        suffix: None,
                        operands: [].into_iter().collect(),
                    });
                    [].into_iter().collect()
                }
            },
            &NodeKind::Scalar(op) => self
                .scalar_op_to_ptx(
                    op,
                    input_encodings.iter().map(|input| input.as_src_operand()).collect(),
                    output_vars.iter().map(|&v| func.vars[v].ty.as_scalar(cx).unwrap()).collect(),
                )
                .into_iter()
                .collect(),
            NodeKind::Vector(op) => match op {
                vector::Op::Distribute(op) => todo!(),
                vector::Op::Reduce(reduce_op) => todo!(),
                vector::Op::Whole(op) => match op {
                    vector::WholeOp::New => todo!(),
                    &vector::WholeOp::Extract { elem_idx } => {
                        let ty = func.vars[output_vars[0]].ty;
                        let output_reg = self.alloc_reg_with_type_kind(&cx[ty].kind);

                        self.mov(
                            None,
                            output_reg,
                            &match input_encodings[0].as_src_operand() {
                                PtxOperand::Reg(PtxReg { bank, idx, suffix: None }) => {
                                    assert_eq!(bank, PtxRegBank::V4B32);
                                    PtxValueEncoding::Reg(PtxReg {
                                        bank,
                                        idx,
                                        suffix: Some([".x", ".y", ".z", ".w"][elem_idx as usize]),
                                    })
                                }
                                _ => unreachable!(),
                            },
                        );

                        [output_reg].into_iter().collect()
                    }
                    vector::WholeOp::Insert { elem_idx } => todo!(),
                    vector::WholeOp::DynExtract => todo!(),
                    vector::WholeOp::DynInsert => todo!(),
                    vector::WholeOp::Mul => todo!(),
                },
            },
            NodeKind::FuncCall(func) => todo!(),
            NodeKind::Mem(op) => match *op {
                MemOp::FuncLocalVar(layout) => {
                    let ty = func.vars[output_vars[0]].ty;
                    let output_reg = self.alloc_reg_with_type_kind(&cx[ty].kind);

                    let local_idx = self.func.locals.len();
                    self.func.locals.push(layout);

                    self.func.insts.push(PtxInst {
                        prefix: None,
                        op: "cvta.local.",
                        suffix: Some("u64"),
                        operands: [
                            PtxOperand::Reg(output_reg),
                            PtxOperand::PtrToNamedVar {
                                base: ("%l", local_idx.try_into().unwrap()),
                                offset: None,
                            },
                        ]
                        .into_iter()
                        .collect(),
                    });

                    // FIXME(eddyb) emit stores for initializers.
                    assert!(input_encodings.is_empty());

                    [output_reg].into_iter().collect()
                }
                MemOp::Load { offset } => {
                    let ty = func.vars[output_vars[0]].ty;

                    let mut convert_to_bool = false;
                    let loaded_value_reg = self.alloc_reg_with_type_kind(match &cx[ty].kind {
                        TypeKind::Scalar(scalar::Type::Bool) => {
                            convert_to_bool = true;
                            &TypeKind::Scalar(scalar::Type::UInt(scalar::IntWidth::I8))
                        }
                        k => k,
                    });

                    if let [
                        PtxValueEncoding::PtrToGlobalVar {
                            global: PtxGlobalVarEncoding::GlobalInvocationId,
                            offset: base_offset,
                        },
                    ] = input_encodings[..]
                    {
                        assert_eq!((base_offset, offset), (None, None));
                        assert_eq!(loaded_value_reg.bank, PtxRegBank::V4B32);

                        // FIXME(eddyb) `%tid` isn't exactly the complete ID,
                        // it should also be combined with `%ctaid`.
                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "mov.",
                            suffix: Some(loaded_value_reg.bank.ty()),
                            operands: [
                                PtxOperand::Reg(loaded_value_reg),
                                PtxOperand::Special("%tid"),
                            ]
                            .into_iter()
                            .collect(),
                        });
                    } else {
                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "ld.",
                            suffix: Some(loaded_value_reg.bank.ty()),
                            operands: [
                                PtxOperand::Reg(loaded_value_reg),
                                input_encodings[0].as_mem_operand(offset.map_or(0, |o| o.get())),
                            ]
                            .into_iter()
                            .collect(),
                        });
                    }

                    if convert_to_bool {
                        self.scalar_op_to_ptx(
                            scalar::IntBinOp::Ne.into(),
                            [
                                PtxOperand::Reg(loaded_value_reg),
                                PtxOperand::Const {
                                    ty: scalar::Type::UInt(scalar::IntWidth::I8),
                                    bits: 0,
                                },
                            ]
                            .into_iter()
                            .collect(),
                            [scalar::Type::Bool].into_iter().collect(),
                        )
                        .into_iter()
                        .collect()
                    } else {
                        [loaded_value_reg].into_iter().collect()
                    }
                }
                MemOp::Store { offset } => {
                    let tmp;
                    let mut stored_value = &input_encodings[1];

                    let store_type = match stored_value {
                        // HACK(eddyb) skipping `mem.store(p, undef)` entirely.
                        PtxValueEncoding::Undef => return,

                        PtxValueEncoding::Const { ty, .. } => match ty.bit_width() {
                            1 | 8 => "b8",
                            16 => "b16",
                            32 => "b32",
                            64 => "b64",
                            w => unreachable!("unsupported width {w}"),
                        },

                        PtxValueEncoding::Reg(reg) => match reg.bank {
                            PtxRegBank::Pred => {
                                let u8_type = scalar::Type::UInt(scalar::IntWidth::I8);
                                let byte_reg =
                                    self.alloc_reg_with_type_kind(&TypeKind::Scalar(u8_type));
                                self.select(
                                    byte_reg,
                                    stored_value,
                                    &PtxValueEncoding::Const { ty: u8_type, bits: 1 },
                                    &PtxValueEncoding::Const { ty: u8_type, bits: 0 },
                                );
                                tmp = PtxValueEncoding::Reg(byte_reg);
                                stored_value = &tmp;

                                byte_reg.bank.ty()
                            }
                            _ => reg.bank.ty(),
                        },
                        _ => unreachable!(),
                    };

                    self.func.insts.push(PtxInst {
                        prefix: None,
                        op: "st.",
                        suffix: Some(store_type),
                        operands: [
                            input_encodings[0].as_mem_operand(offset.map_or(0, |o| o.get())),
                            stored_value.as_src_operand(),
                        ]
                        .into_iter()
                        .collect(),
                    });
                    [].into_iter().collect()
                }
                MemOp::Copy { size } => {
                    // HACK(eddyb) in the absence of even a basic (greedy) regalloc,
                    // this avoids using more than 3 registers total.
                    let mut last_used_reg = None::<(PtxReg, scalar::Type)>;

                    // HACK(eddyb) this is mostly taken from `qptr::lift`.
                    let mut offset = 0;
                    let size = i32::try_from(size.get()).unwrap();
                    while let Some(remaining_bytes @ 1..) = size.checked_sub(offset) {
                        // HACK(eddyb) prefer `u32` over `u16` over `u8`.
                        // FIXME(eddyb) consider using even vector load/store pairs.
                        let copy_unit_size = 1 << remaining_bytes.trailing_zeros().clamp(0, 2);
                        let copy_unit = scalar::Type::UInt(
                            scalar::IntWidth::try_from_bits(copy_unit_size * 8).unwrap(),
                        );

                        let copy_reg =
                            last_used_reg.filter(|&(_, last_ty)| last_ty == copy_unit).map_or_else(
                                || self.alloc_reg_with_type_kind(&TypeKind::Scalar(copy_unit)),
                                |(last_reg, _)| last_reg,
                            );
                        last_used_reg = Some((copy_reg, copy_unit));

                        let copy_ptx_type = match copy_unit.bit_width() {
                            8 => "b8",
                            16 => "b16",
                            32 => "b32",
                            _ => unreachable!(),
                        };

                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "ld.",
                            suffix: Some(copy_ptx_type),
                            operands: [
                                PtxOperand::Reg(copy_reg),
                                input_encodings[1].as_mem_operand(offset),
                            ]
                            .into_iter()
                            .collect(),
                        });
                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "st.",
                            suffix: Some(copy_ptx_type),
                            operands: [
                                input_encodings[0].as_mem_operand(offset),
                                PtxOperand::Reg(copy_reg),
                            ]
                            .into_iter()
                            .collect(),
                        });

                        offset += copy_unit_size as i32;
                    }

                    [].into_iter().collect()
                }
            },
            NodeKind::QPtr(op) => match op {
                QPtrOp::HandleArrayIndex => todo!(),
                QPtrOp::BufferData => match input_encodings[..] {
                    [
                        PtxValueEncoding::PtrToGlobalVar {
                            global:
                                &PtxGlobalVarEncoding::Buffer {
                                    data_ptr_generic64_entry_param_idx, ..
                                },
                            offset,
                        },
                    ] => {
                        assert_eq!(offset, None);

                        // HACK(eddyb) avoid attempting to take ownership.
                        self.regs.def_var_aliasing(
                            output_vars[0],
                            self.loaded_params[data_ptr_generic64_entry_param_idx as usize].into(),
                        );
                        return;
                    }
                    _ => unreachable!(),
                },
                &QPtrOp::BufferDynLen { fixed_base_size, dyn_unit_stride } => match input_encodings
                    [..]
                {
                    [
                        PtxValueEncoding::PtrToGlobalVar {
                            global:
                                &PtxGlobalVarEncoding::Buffer {
                                    layout, dyn_len_entry_param_idx, ..
                                },
                            offset,
                        },
                    ] => {
                        assert_eq!(offset, None);
                        assert_eq!(layout.fixed_base.size, fixed_base_size);
                        assert_eq!(layout.dyn_unit_stride, Some(dyn_unit_stride));

                        // HACK(eddyb) avoid attempting to take ownership.
                        self.regs.def_var_aliasing(
                            output_vars[0],
                            self.loaded_params[dyn_len_entry_param_idx.unwrap() as usize].into(),
                        );
                        return;
                    }
                    _ => unreachable!(),
                },
                &QPtrOp::Offset(offset) => self
                    .scalar_op_to_ptx(
                        scalar::IntBinOp::Add.into(),
                        [
                            input_encodings[0].as_src_operand(),
                            PtxOperand::Const { ty: scalar::Type::S32, bits: offset as i64 as u64 },
                        ]
                        .into_iter()
                        .collect(),
                        [scalar::Type::UInt(scalar::IntWidth::I64)].into_iter().collect(),
                    )
                    .into_iter()
                    .collect(),
                QPtrOp::DynOffset { stride, index_bounds: _ } => {
                    let (index,) = self
                        .scalar_op_to_ptx(
                            scalar::IntUnOp::TruncOrSignExtend.into(),
                            [input_encodings[1].as_src_operand()].into_iter().collect(),
                            [scalar::Type::SInt(scalar::IntWidth::I64)].into_iter().collect(),
                        )
                        .into_iter()
                        .collect_tuple()
                        .unwrap();

                    let (offset,) = self
                        .scalar_op_to_ptx(
                            scalar::IntBinOp::Mul.into(),
                            [
                                PtxOperand::Reg(index),
                                PtxOperand::Const {
                                    ty: scalar::Type::SInt(scalar::IntWidth::I64),
                                    bits: stride.get().into(),
                                },
                            ]
                            .into_iter()
                            .collect(),
                            [scalar::Type::SInt(scalar::IntWidth::I64)].into_iter().collect(),
                        )
                        .into_iter()
                        .collect_tuple()
                        .unwrap();

                    self.scalar_op_to_ptx(
                        scalar::IntBinOp::Add.into(),
                        [input_encodings[0].as_src_operand(), PtxOperand::Reg(offset)]
                            .into_iter()
                            .collect(),
                        [scalar::Type::UInt(scalar::IntWidth::I64)].into_iter().collect(),
                    )
                    .into_iter()
                    .collect()
                }
            },
            NodeKind::ThunkBind(control_target) => todo!(),
            NodeKind::SpvInst(spv_inst, inst_lowering) => {
                // FIXME(eddyb) support these usecases.
                assert!(inst_lowering.disaggregated_inputs.is_empty());
                assert!(inst_lowering.disaggregated_output.is_none());

                // TODO(eddyb) look into e.g. `fence.sc` is needed for atomics.
                if spv_inst.opcode == self.wk.OpAtomicLoad {
                    let ty = func.vars[output_vars[0]].ty;
                    let output_reg = self.alloc_reg_with_type_kind(&cx[ty].kind);

                    self.func.insts.push(PtxInst {
                        prefix: None,
                        // TODO(eddyb) don't ignore the scope/consistency operands.
                        op: "ld.relaxed.sys.",
                        suffix: Some(output_reg.bank.ty()),
                        operands: [
                            PtxOperand::Reg(output_reg),
                            input_encodings[0].as_mem_operand(0),
                        ]
                        .into_iter()
                        .collect(),
                    });

                    [output_reg].into_iter().collect()
                } else if spv_inst.opcode == self.wk.OpAtomicCompareExchange {
                    let ty = func.vars[output_vars[0]].ty;
                    let output_reg = self.alloc_reg_with_type_kind(&cx[ty].kind);

                    self.func.insts.push(PtxInst {
                        prefix: None,
                        // TODO(eddyb) don't ignore the scope/consistency operands.
                        op: "atom.relaxed.sys.cas.",
                        suffix: Some(output_reg.bank.ty()),
                        operands: [
                            PtxOperand::Reg(output_reg),
                            input_encodings[0].as_mem_operand(0),
                            input_encodings[5].as_src_operand(),
                            input_encodings[4].as_src_operand(),
                        ]
                        .into_iter()
                        .collect(),
                    });

                    [output_reg].into_iter().collect()
                } else if spv_inst.opcode == self.wk.OpAtomicIAdd {
                    let ty = func.vars[output_vars[0]].ty;
                    let output_reg = self.alloc_reg_with_type_kind(&cx[ty].kind);

                    let output_reg_type = match output_reg.bank {
                        PtxRegBank::B8 => "u8",
                        PtxRegBank::B16 => "u16",
                        PtxRegBank::B32 => "u32",
                        PtxRegBank::B64 => "u64",
                        _ => unreachable!("unsupported reg bank {:?}", output_reg.bank),
                    };

                    self.func.insts.push(PtxInst {
                        prefix: None,
                        // TODO(eddyb) don't ignore the scope/consistency operands.
                        op: "atom.relaxed.sys.add.",
                        suffix: Some(output_reg_type),
                        operands: [
                            PtxOperand::Reg(output_reg),
                            input_encodings[0].as_mem_operand(0),
                            input_encodings[3].as_src_operand(),
                        ]
                        .into_iter()
                        .collect(),
                    });

                    [output_reg].into_iter().collect()
                } else if spv_inst.opcode == self.wk.OpSelect {
                    let ty = func.vars[output_vars[0]].ty;
                    let output_reg = self.alloc_reg_with_type_kind(&cx[ty].kind);

                    // FIXME(eddyb) avoid indexing, and use e.g. `collect_tuple` instead.
                    self.select(
                        output_reg,
                        &input_encodings[0],
                        &input_encodings[1],
                        &input_encodings[2],
                    );

                    [output_reg].into_iter().collect()
                } else if spv_inst.opcode == self.wk.OpBitcast {
                    let ty = func.vars[output_vars[0]].ty;
                    let output_reg = self.alloc_reg_with_type_kind(&cx[ty].kind);

                    // FIXME(eddyb) `mov` can only be avoided via regalloc
                    // (registers can't be reused, as they could be overriden
                    // at a later point - in theory, that should only ever
                    // happen strictly past the end of the scope of this `Var`,
                    // but better to be safe than sorry).
                    self.mov(None, output_reg, &input_encodings[0]);

                    [output_reg].into_iter().collect()
                } else {
                    todo!("unsupported {}", describe_spv_inst())
                }
            }
            &NodeKind::SpvExtInst { ext_set, inst, .. } if ext_set == self.glsl_std_450 => {
                let inputs: ArrayVec<_, 3> =
                    input_encodings.iter().map(|input| input.as_src_operand()).collect();

                // FIXME(eddyb) properly encode type shape in `PtxOperand`
                // (or some intermediary type that then gets turned into it).
                let input0_reg_bank = match inputs[0] {
                    PtxOperand::Const { ty, .. } => match ty.bit_width() {
                        1 => PtxRegBank::Pred,
                        8 => PtxRegBank::B8,
                        16 => PtxRegBank::B16,
                        32 => PtxRegBank::B32,
                        64 => PtxRegBank::B64,
                        w => unreachable!("unsupported width {w}"),
                    },
                    PtxOperand::Reg(reg) => reg.bank,
                    _ => unreachable!(),
                };
                let int_or_float_suffix_from_input0 = match input0_reg_bank {
                    PtxRegBank::Pred => Err("`inputs[0]` is a `bool`"),
                    PtxRegBank::B8 => Ok("8"),
                    PtxRegBank::B16 => Ok("16"),
                    PtxRegBank::B32 => Ok("32"),
                    PtxRegBank::B64 => Ok("64"),
                    PtxRegBank::V4B32 => Err("`inputs[0]` is a vector"),
                };

                let mut suffix_override = None;
                let op_name = if inst == self.wk.FindILsb {
                    // FIXME(eddyb) this should be "count trailing zeros", and
                    // PTX has a "count leading zeros", plus "bit reverse",
                    // i.e. `brev` + `clz` may be used to implement this.
                    "trap; // TODO_ctz.b"
                } else if inst == self.wk.FindSMsb {
                    // FIXME(eddyb) check the semantics (SPIR-V vs PTX).
                    "bfind.s"
                } else if inst == self.wk.FindUMsb {
                    // FIXME(eddyb) check the semantics (SPIR-V vs PTX).
                    "bfind.u"
                } else if inst == self.wk.FAbs {
                    "abs.f"
                } else if inst == self.wk.Floor {
                    match input0_reg_bank {
                        PtxRegBank::B32 => "cvt.rmi.f32.f",
                        PtxRegBank::B64 => "cvt.rmi.f64.f",
                        _ => unreachable!(),
                    }
                } else if inst == self.wk.Ceil {
                    match input0_reg_bank {
                        PtxRegBank::B32 => "cvt.rpi.f32.f",
                        PtxRegBank::B64 => "cvt.rpi.f64.f",
                        _ => unreachable!(),
                    }
                } else if inst == self.wk.Round {
                    // TODO(eddyb) this can be handled relatively easily, based
                    // on libdevice - i.e. `cvt.rzi(add.rz(x, copysign(x, 0.5)))`
                    "trap; // TODO_round.f"
                } else if inst == self.wk.Trunc {
                    match input0_reg_bank {
                        PtxRegBank::B32 => "cvt.rzi.f32.f",
                        PtxRegBank::B64 => "cvt.rzi.f64.f",
                        _ => unreachable!(),
                    }
                } else if inst == self.wk.Exp {
                    "TODO_exp.f"
                } else if inst == self.wk.Sqrt {
                    "sqrt.rn.f"
                } else if inst == self.wk.Sin {
                    "TODO_sin.f"
                } else if inst == self.wk.Cos {
                    "TODO_cos.f"
                } else if inst == self.wk.FMin {
                    "min.f"
                } else if inst == self.wk.FMax {
                    "max.f"
                } else if inst == self.wk.Pow {
                    "TODO_pow.f"
                } else if inst == self.wk.Fma {
                    "fma.rn.f"
                } else {
                    if true {
                        let inst = spv::spec::Spec::get()
                            .get_ext_inst_set_by_lowercase_name(&cx[ext_set].to_ascii_lowercase())
                            .and_then(|ext_inst_set_desc| {
                                Some(&ext_inst_set_desc.instructions.get(&inst)?.name)
                            })
                            .unwrap();
                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "TODO_glsl_std_450_",
                            suffix: Some(String::leak(inst.to_string())),
                            operands: [].into_iter().collect(),
                        });
                        for &v in output_vars {
                            self.alloc_reg_for_var(func.at(v));
                        }
                        return;
                    }

                    todo!("unsupported {}", describe_spv_inst())
                };

                let suffix =
                    suffix_override.unwrap_or_else(|| int_or_float_suffix_from_input0.unwrap());

                let output_regs: SmallVec<[_; 4]> = output_vars
                    .iter()
                    .map(|&var| self.alloc_reg_with_type_kind(&cx[func.vars[var].ty].kind))
                    .collect();

                assert_eq!(output_regs.len(), 1);

                self.func.insts.push(PtxInst {
                    prefix: None,
                    op: op_name,
                    suffix: Some(suffix),
                    operands: [PtxOperand::Reg(output_regs[0])].into_iter().chain(inputs).collect(),
                });

                output_regs
            }
            NodeKind::SpvExtInst { ext_set, inst, lowering } => {
                // FIXME(eddyb) support these usecases.
                assert!(lowering.disaggregated_inputs.is_empty());
                assert!(lowering.disaggregated_output.is_none());

                // HACK(eddyb) ignoring `OpExtInst "NonSemantic.*"` and hoping
                // that nothing accesses the `OpTypeVoid`-typed output `Var`.
                // FIXME(eddyb) implement at least `DebugPrintf`.
                if cx[*ext_set].starts_with("NonSemantic.") {
                    return;
                }

                if true {
                    let ext_set = &cx[*ext_set];
                    let inst = *inst;
                    let inst = spv::spec::Spec::get()
                        .get_ext_inst_set_by_lowercase_name(&ext_set.to_ascii_lowercase())
                        .and_then(|ext_inst_set_desc| {
                            Some(&ext_inst_set_desc.instructions.get(&inst)?.name)
                        })
                        .unwrap();
                    self.func.insts.push(PtxInst {
                        prefix: None,
                        op: "TODO_",
                        suffix: Some(String::leak(format!("{ext_set:?}_{inst}"))),
                        operands: [].into_iter().collect(),
                    });
                    for &v in output_vars {
                        self.alloc_reg_for_var(func.at(v));
                    }
                    return;
                }

                todo!("unsupported {}", describe_spv_inst());
            }
        };

        assert_eq!(outputs.len(), output_vars.len());
        for (&output_var, output) in output_vars.iter().zip_eq(outputs) {
            self.regs.def_var_taking_ownership_of_temp(output_var, output.into());
        }
    }

    fn scalar_op_to_ptx(
        &mut self,
        op: scalar::Op,
        inputs: ArrayVec<PtxOperand, 2>,
        output_types: ArrayVec<scalar::Type, 2>,
    ) -> ArrayVec<PtxReg, 2> {
        // HACK(eddyb) strength-reduce a few easy cases.
        match (op, &inputs[..]) {
            // FIXME(eddyb) handle signed division too (may round differently?).
            (
                scalar::Op::IntBinary(op @ (scalar::IntBinOp::Mul | scalar::IntBinOp::DivU)),
                [a, PtxOperand::Const { ty: scalar::Type::UInt(_), bits: b }],
            ) if b.is_power_of_two() => {
                return self.scalar_op_to_ptx(
                    match op {
                        scalar::IntBinOp::Mul => scalar::IntBinOp::Shl,
                        scalar::IntBinOp::DivU => scalar::IntBinOp::ShrU,
                        _ => unreachable!(),
                    }
                    .into(),
                    [
                        a.clone(),
                        PtxOperand::Const {
                            ty: scalar::Type::U32,
                            bits: b.trailing_zeros().into(),
                        },
                    ]
                    .into_iter()
                    .collect(),
                    output_types,
                );
            }

            // HACK(eddyb) map 8-bit operations to e.g. 16-bit ones.
            // FIXME(eddyb) also do the same for unsupported 16-bit operations!
            (scalar::Op::IntUnary(_) | scalar::Op::IntBinary(_), _)
                if inputs.iter().any(|input| {
                    matches!(input, PtxOperand::Reg(PtxReg { bank: PtxRegBank::B8, .. }))
                }) =>
            {
                let wider_op_type = match op {
                    scalar::Op::IntUnary(op) => match op {
                        scalar::IntUnOp::TruncOrZeroExtend | scalar::IntUnOp::TruncOrSignExtend => {
                            None
                        }

                        scalar::IntUnOp::Neg => Some(scalar::Type::SInt(scalar::IntWidth::I16)),
                        scalar::IntUnOp::Not => Some(scalar::Type::UInt(scalar::IntWidth::I16)),
                        scalar::IntUnOp::CountOnes => Some(scalar::Type::U32),
                    },
                    scalar::Op::IntBinary(op) => match op {
                        // FIXME(eddyb) implement these using plain add/sub/mul
                        // on the wider type, then splitting that into halves.
                        scalar::IntBinOp::CarryingAdd => todo!(),
                        scalar::IntBinOp::BorrowingSub => todo!(),
                        scalar::IntBinOp::WideningMulU => todo!(),
                        scalar::IntBinOp::WideningMulS => todo!(),

                        scalar::IntBinOp::Add
                        | scalar::IntBinOp::Sub
                        | scalar::IntBinOp::Mul
                        | scalar::IntBinOp::DivU
                        | scalar::IntBinOp::ModU
                        | scalar::IntBinOp::ShrU
                        | scalar::IntBinOp::Shl
                        | scalar::IntBinOp::Or
                        | scalar::IntBinOp::Xor
                        | scalar::IntBinOp::And
                        | scalar::IntBinOp::Eq
                        | scalar::IntBinOp::Ne
                        | scalar::IntBinOp::GtU
                        | scalar::IntBinOp::GeU
                        | scalar::IntBinOp::LtU
                        | scalar::IntBinOp::LeU => Some(scalar::Type::UInt(scalar::IntWidth::I16)),

                        scalar::IntBinOp::DivS
                        | scalar::IntBinOp::RemS
                        | scalar::IntBinOp::ModS
                        | scalar::IntBinOp::ShrS
                        | scalar::IntBinOp::GtS
                        | scalar::IntBinOp::GeS
                        | scalar::IntBinOp::LtS
                        | scalar::IntBinOp::LeS => Some(scalar::Type::SInt(scalar::IntWidth::I16)),
                    },
                    _ => unreachable!(),
                };

                if let Some(wider_op_type) = wider_op_type {
                    let cvt_op_to_or_from_wider_op_type = match wider_op_type {
                        scalar::Type::UInt(_) => scalar::IntUnOp::TruncOrZeroExtend,
                        scalar::Type::SInt(_) => scalar::IntUnOp::TruncOrSignExtend,
                        _ => unreachable!(),
                    };
                    let wider_inputs = inputs
                        .into_iter()
                        .map(|input| {
                            if let PtxOperand::Const { ty, bits } = input
                                && let Ok(wider_input) = cvt_op_to_or_from_wider_op_type.try_eval(
                                    scalar::Const::from_bits(ty, bits.into()),
                                    wider_op_type,
                                )
                            {
                                return PtxOperand::Const {
                                    ty: wider_input.ty(),
                                    bits: wider_input.bits().try_into().unwrap(),
                                };
                            }

                            let (wider_input,) = self
                                .scalar_op_to_ptx(
                                    cvt_op_to_or_from_wider_op_type.into(),
                                    [input].into_iter().collect(),
                                    [wider_op_type].into_iter().collect(),
                                )
                                .into_iter()
                                .collect_tuple()
                                .unwrap();
                            PtxOperand::Reg(wider_input)
                        })
                        .collect();

                    if let [scalar::Type::Bool] = output_types[..] {
                        return self.scalar_op_to_ptx(op, wider_inputs, output_types);
                    }

                    let (wider_output,) = self
                        .scalar_op_to_ptx(op, wider_inputs, [wider_op_type].into_iter().collect())
                        .into_iter()
                        .collect_tuple()
                        .unwrap();

                    return self.scalar_op_to_ptx(
                        cvt_op_to_or_from_wider_op_type.into(),
                        [PtxOperand::Reg(wider_output)].into_iter().collect(),
                        output_types,
                    );
                }
            }

            _ => {}
        }

        // FIXME(eddyb) make it possible to feed `output_regs` from the caller.
        let output_regs: ArrayVec<_, 2> = output_types
            .iter()
            .map(|&ty| self.alloc_reg_with_type_kind(&TypeKind::Scalar(ty)))
            .collect();

        // HACK(eddyb) older `spirv-opt` can generate fully-const-foldable ops.
        let const_inputs: Option<ArrayVec<_, 2>> = inputs
            .iter()
            .map(|input| match input {
                &PtxOperand::Const { ty, bits } => Some(scalar::Const::from_bits(ty, bits.into())),
                _ => None,
            })
            .collect();
        if let Some(const_inputs) = const_inputs
            && let Ok(const_outputs) = op.try_eval(&const_inputs, &output_types)
        {
            for (&output_reg, const_output) in output_regs.iter().zip_eq(const_outputs) {
                self.mov(
                    None,
                    output_reg,
                    &PtxValueEncoding::Const {
                        ty: const_output.ty(),
                        bits: const_output.bits().try_into().unwrap(),
                    },
                )
            }
            return output_regs;
        }

        // FIXME(eddyb) properly encode type shape in `PtxOperand`
        // (or some intermediary type that then gets turned into it).
        let int_or_float_suffix_from_input0 = match inputs[0] {
            PtxOperand::Const { ty, .. } => match ty.bit_width() {
                1 => Err("`inputs[0]` is a `bool`"),
                8 => Ok("8"),
                16 => Ok("16"),
                32 => Ok("32"),
                64 => Ok("64"),
                w => unreachable!("unsupported width {w}"),
            },
            PtxOperand::Reg(reg) => match reg.bank {
                PtxRegBank::Pred => Err("`inputs[0]` is a `bool`"),
                PtxRegBank::B8 => Ok("8"),
                PtxRegBank::B16 => Ok("16"),
                PtxRegBank::B32 => Ok("32"),
                PtxRegBank::B64 => Ok("64"),
                PtxRegBank::V4B32 => Err("`inputs[0]` is a vector"),
            },
            PtxOperand::PtrToNamedVar { .. } => Ok("64"),
            _ => unreachable!(),
        };

        // HACK(eddyb) only used to implement `bool.eq` using `xor.pred`.
        let mut bool_not_output = false;

        let (op_name, suffix) = match op {
            scalar::Op::BoolUnary(op) => (
                match op {
                    scalar::BoolUnOp::Not => "not.",
                },
                "pred",
            ),
            scalar::Op::BoolBinary(op) => (
                match op {
                    scalar::BoolBinOp::Eq => {
                        bool_not_output = true;
                        "xor."
                    }
                    scalar::BoolBinOp::Ne => "xor.",
                    scalar::BoolBinOp::Or => "or.",
                    scalar::BoolBinOp::And => "and.",
                },
                "pred",
            ),
            scalar::Op::IntUnary(op) => {
                let op_name = match op {
                    scalar::IntUnOp::Neg => "neg.s",
                    scalar::IntUnOp::Not => "not.b",
                    scalar::IntUnOp::CountOnes => "popc.b",
                    scalar::IntUnOp::TruncOrZeroExtend => match output_regs[0].bank {
                        PtxRegBank::B8 => "cvt.u8.u",
                        PtxRegBank::B16 => "cvt.u16.u",
                        PtxRegBank::B32 => "cvt.u32.u",
                        PtxRegBank::B64 => "cvt.u64.u",
                        _ => unreachable!("unsupported reg bank {:?}", output_regs[0].bank),
                    },
                    scalar::IntUnOp::TruncOrSignExtend => match output_regs[0].bank {
                        PtxRegBank::B8 => "cvt.s8.s",
                        PtxRegBank::B16 => "cvt.s16.s",
                        PtxRegBank::B32 => "cvt.s32.s",
                        PtxRegBank::B64 => "cvt.s64.s",
                        _ => unreachable!("unsupported reg bank {:?}", output_regs[0].bank),
                    },
                };

                (op_name, int_or_float_suffix_from_input0.unwrap())
            }
            scalar::Op::IntBinary(op) => {
                let suffix = int_or_float_suffix_from_input0.unwrap();

                let op_name = match op {
                    scalar::IntBinOp::Add => "add.u",
                    scalar::IntBinOp::Sub => "sub.u",
                    scalar::IntBinOp::Mul => "mul.lo.u",
                    scalar::IntBinOp::DivU => "div.u",
                    scalar::IntBinOp::DivS => "div.s",
                    scalar::IntBinOp::ModU => "rem.u",
                    // FIXME(eddyb) the PTX docs for `rem.s` state:
                    // > The behavior for negative numbers is machine-dependent
                    // > and depends on whether divide rounds
                    // > towards zero or negative infinity.
                    scalar::IntBinOp::RemS => "rem.s",
                    scalar::IntBinOp::ModS => todo!(),
                    scalar::IntBinOp::ShrU => "shr.u",
                    scalar::IntBinOp::ShrS => "shr.s",
                    scalar::IntBinOp::Shl => "shl.b",
                    scalar::IntBinOp::Or => "or.b",
                    scalar::IntBinOp::Xor => "xor.b",
                    scalar::IntBinOp::And => "and.b",
                    scalar::IntBinOp::CarryingAdd => {
                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "add.cc.u",
                            suffix: Some(suffix),
                            operands: [PtxOperand::Reg(output_regs[0])]
                                .into_iter()
                                .chain(inputs)
                                .collect(),
                        });
                        // HACK(eddyb) read out the `CC.CF` register.
                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "addc.u",
                            suffix: Some(suffix),
                            operands: [
                                PtxOperand::Reg(output_regs[1]),
                                PtxOperand::Const { ty: output_types[1], bits: 0 },
                                PtxOperand::Const { ty: output_types[1], bits: 0 },
                            ]
                            .into_iter()
                            .collect(),
                        });
                        return output_regs;
                    }
                    scalar::IntBinOp::BorrowingSub => {
                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "sub.cc.u",
                            suffix: Some(suffix),
                            operands: [PtxOperand::Reg(output_regs[0])]
                                .into_iter()
                                .chain(inputs)
                                .collect(),
                        });
                        // HACK(eddyb) read out the `CC.CF` register.
                        self.func.insts.push(PtxInst {
                            prefix: None,
                            op: "addc.u",
                            suffix: Some(suffix),
                            operands: [
                                PtxOperand::Reg(output_regs[1]),
                                PtxOperand::Const { ty: output_types[1], bits: 0 },
                                PtxOperand::Const { ty: output_types[1], bits: 0 },
                            ]
                            .into_iter()
                            .collect(),
                        });
                        return output_regs;
                    }
                    scalar::IntBinOp::WideningMulU => {
                        for (i, &output_reg) in output_regs.iter().enumerate() {
                            self.func.insts.push(PtxInst {
                                prefix: None,
                                op: ["mul.lo.u", "mul.hi.u"][i],
                                suffix: Some(suffix),
                                operands: [PtxOperand::Reg(output_reg)]
                                    .into_iter()
                                    .chain(inputs.clone())
                                    .collect(),
                            });
                        }
                        return output_regs;
                    }
                    scalar::IntBinOp::WideningMulS => todo!(),
                    scalar::IntBinOp::Eq => "setp.eq.b",
                    scalar::IntBinOp::Ne => "setp.ne.b",
                    scalar::IntBinOp::GtU => "setp.gt.u",
                    scalar::IntBinOp::GtS => "setp.gt.s",
                    scalar::IntBinOp::GeU => "setp.ge.u",
                    scalar::IntBinOp::GeS => "setp.ge.s",
                    scalar::IntBinOp::LtU => "setp.lt.u",
                    scalar::IntBinOp::LtS => "setp.lt.s",
                    scalar::IntBinOp::LeU => "setp.le.u",
                    scalar::IntBinOp::LeS => "setp.le.s",
                };

                (op_name, suffix)
            }
            scalar::Op::FloatUnary(op) => {
                let op_name = match op {
                    scalar::FloatUnOp::Neg => "neg.f",
                    scalar::FloatUnOp::IsNan => "testp.notanumber.f",
                    scalar::FloatUnOp::IsInf => "testp.infinite.f",
                    scalar::FloatUnOp::FromUInt => match output_regs[0].bank {
                        PtxRegBank::B32 => "cvt.rn.f32.u",
                        PtxRegBank::B64 => "cvt.rn.f64.u",
                        _ => unreachable!("unsupported reg bank {:?}", output_regs[0].bank),
                    },
                    scalar::FloatUnOp::FromSInt => match output_regs[0].bank {
                        PtxRegBank::B32 => "cvt.rn.f32.s",
                        PtxRegBank::B64 => "cvt.rn.f64.s",
                        _ => unreachable!("unsupported reg bank {:?}", output_regs[0].bank),
                    },
                    scalar::FloatUnOp::ToUInt => match output_regs[0].bank {
                        PtxRegBank::B8 => "cvt.rzi.u8.f",
                        PtxRegBank::B16 => "cvt.rzi.u16.f",
                        PtxRegBank::B32 => "cvt.rzi.u32.f",
                        PtxRegBank::B64 => "cvt.rzi.u64.f",
                        _ => unreachable!("unsupported reg bank {:?}", output_regs[0].bank),
                    },
                    scalar::FloatUnOp::ToSInt => match output_regs[0].bank {
                        PtxRegBank::B8 => "cvt.rzi.s8.f",
                        PtxRegBank::B16 => "cvt.rzi.s16.f",
                        PtxRegBank::B32 => "cvt.rzi.s32.f",
                        PtxRegBank::B64 => "cvt.rzi.s64.f",
                        _ => unreachable!("unsupported reg bank {:?}", output_regs[0].bank),
                    },
                    scalar::FloatUnOp::Convert => match output_regs[0].bank {
                        PtxRegBank::B32 => "cvt.rn.f32.u",
                        PtxRegBank::B64 => "cvt.rn.f64.u",
                        _ => unreachable!("unsupported reg bank {:?}", output_regs[0].bank),
                    },
                    // FIXME(eddyb) should work as `T -> f16 -> T` conversions.
                    scalar::FloatUnOp::QuantizeAsF16 => todo!(),
                };

                (op_name, int_or_float_suffix_from_input0.unwrap())
            }
            scalar::Op::FloatBinary(op) => {
                // NOTE(eddyb) `.rn` is optional in some cases, but is included
                // here for IEEE-754 compliance (and to disable FMA folding).
                // FIXME(eddyb) take into account SPIR-V float semantics flags.
                let op_name = match op {
                    scalar::FloatBinOp::Add => "add.rn.f",
                    scalar::FloatBinOp::Sub => "sub.rn.f",
                    scalar::FloatBinOp::Mul => "mul.rn.f",
                    scalar::FloatBinOp::Div => "div.rn.f",
                    scalar::FloatBinOp::Rem => "trap; // TODO_rem.f",
                    scalar::FloatBinOp::Mod => "TODO_mod.f",
                    scalar::FloatBinOp::Cmp(cmp) => match cmp {
                        scalar::FloatCmp::Eq => "setp.eq.f",
                        scalar::FloatCmp::Ne => "setp.ne.f",
                        scalar::FloatCmp::Lt => "setp.lt.f",
                        scalar::FloatCmp::Gt => "setp.gt.f",
                        scalar::FloatCmp::Le => "setp.le.f",
                        scalar::FloatCmp::Ge => "setp.ge.f",
                    },
                    scalar::FloatBinOp::CmpOrUnord(cmp) => match cmp {
                        scalar::FloatCmp::Eq => "setp.equ.f",
                        scalar::FloatCmp::Ne => "setp.neu.f",
                        scalar::FloatCmp::Lt => "setp.ltu.f",
                        scalar::FloatCmp::Gt => "setp.gtu.f",
                        scalar::FloatCmp::Le => "setp.leu.f",
                        scalar::FloatCmp::Ge => "setp.geu.f",
                    },
                };

                (op_name, int_or_float_suffix_from_input0.unwrap())
            }
        };

        // HACK(eddyb) PTX shifts require a 32-bit shift amount.
        let inputs = {
            let mut inputs = inputs;
            if let scalar::Op::IntBinary(
                scalar::IntBinOp::Shl | scalar::IntBinOp::ShrU | scalar::IntBinOp::ShrS,
            ) = op
                && let PtxOperand::Reg(rhs) = &mut inputs[1]
                && rhs.bank != PtxRegBank::B32
            {
                (*rhs,) = self
                    .scalar_op_to_ptx(
                        scalar::IntUnOp::TruncOrZeroExtend.into(),
                        [PtxOperand::Reg(*rhs)].into_iter().collect(),
                        [scalar::Type::U32].into_iter().collect(),
                    )
                    .into_iter()
                    .collect_tuple()
                    .unwrap();
            }
            inputs
        };

        assert_eq!(output_regs.len(), 1);

        self.func.insts.push(PtxInst {
            prefix: None,
            op: op_name,
            suffix: Some(suffix),
            operands: [PtxOperand::Reg(output_regs[0])].into_iter().chain(inputs).collect(),
        });

        // HACK(eddyb) only used to implement `bool.eq` using `xor.pred`.
        if bool_not_output {
            self.func.insts.push(PtxInst {
                prefix: None,
                op: "not.",
                suffix: Some("pred"),
                operands: [PtxOperand::Reg(output_regs[0]), PtxOperand::Reg(output_regs[0])]
                    .into_iter()
                    .collect(),
            });
        }

        output_regs
    }

    fn const_to_ptx(&self, ct: Const) -> PtxValueEncoding<'a> {
        match &self.cx[ct].kind {
            ConstKind::Undef => PtxValueEncoding::Undef,
            ConstKind::Scalar(ct) => {
                PtxValueEncoding::Const { ty: ct.ty(), bits: ct.bits().try_into().unwrap() }
            }
            ConstKind::Vector(_) => todo!(),
            &ConstKind::PtrToGlobalVar { global_var, offset } => PtxValueEncoding::PtrToGlobalVar {
                global: &self.ptx_module_interface.global_var_encodings[global_var],
                offset: offset.map(|o| o.try_into().unwrap()),
            },
            ConstKind::PtrToFunc(_) => todo!(),
            ConstKind::SpvInst { .. } => todo!(),
            &ConstKind::SpvStringLiteralForExtInst(s) => {
                PtxValueEncoding::SpvStringLiteralForExtInst(s)
            }
        }
    }

    fn alloc_reg_for_var(&mut self, func_at_var: FuncAt<'_, Var>) -> PtxReg {
        let var = func_at_var.position;
        let ty = func_at_var.decl().ty;
        let reg = self.alloc_reg_with_type_kind(&self.cx[ty].kind);

        self.regs.def_var_taking_ownership_of_temp(var, reg.into());

        reg
    }

    fn alloc_reg_with_type_kind(&mut self, ty_kind: &TypeKind) -> PtxReg {
        let bank = match ty_kind {
            TypeKind::Scalar(ty) => match ty.bit_width() {
                1 => PtxRegBank::Pred,
                8 => PtxRegBank::B8,
                16 => PtxRegBank::B16,
                32 => PtxRegBank::B32,
                64 => PtxRegBank::B64,
                w => todo!(".reg .b{w}"),
            },
            TypeKind::Vector(ty) => match (ty.elem_count.get(), ty.elem.bit_width()) {
                (3..=4, 32) => PtxRegBank::V4B32,
                _ => todo!(),
            },
            TypeKind::QPtr => PtxRegBank::B64,
            _ => todo!(),
        };

        self.regs.alloc_temp(bank).into()
    }

    fn select(
        &mut self,
        dst: PtxReg,
        cond: &PtxValueEncoding,
        t: &PtxValueEncoding,
        e: &PtxValueEncoding,
    ) {
        match (cond, t, e) {
            (PtxValueEncoding::Undef, _, _) => {
                // HACK(eddyb) denoting immediate UB using `trap`.
                // FIXME(eddyb) does `OpSelect` even have immediate UB?
                self.func.insts.push(PtxInst {
                    prefix: None,
                    op: "trap",
                    suffix: None,
                    operands: [].into_iter().collect(),
                });
            }

            (PtxValueEncoding::Const { bits: 1, .. }, x, _)
            | (PtxValueEncoding::Const { bits: 0, .. }, _, x)
            | (_, PtxValueEncoding::Undef, x)
            | (_, x, PtxValueEncoding::Undef) => self.mov(None, dst, x),

            _ => {
                if let PtxRegBank::B16 | PtxRegBank::B32 | PtxRegBank::B64 = dst.bank {
                    self.func.insts.push(PtxInst {
                        prefix: None,
                        op: "selp.",
                        suffix: Some(dst.bank.ty()),
                        operands: [
                            PtxOperand::Reg(dst),
                            t.as_src_operand(),
                            e.as_src_operand(),
                            cond.as_src_operand(),
                        ]
                        .into_iter()
                        .collect(),
                    });
                } else {
                    let cond = match cond.as_src_operand() {
                        PtxOperand::Reg(pred) => pred,
                        _ => unreachable!(),
                    };
                    self.mov(Some(PtxCond::If(cond)), dst, t);
                    self.mov(Some(PtxCond::IfNot(cond)), dst, e);
                }
            }
        }
    }

    // FIXME(eddyb) with destination-passing, this could be an "SSA copy",
    // if an "allocate the new destination yourself" was passed in.
    fn mov(&mut self, cond: Option<PtxCond>, dst: PtxReg, src: &PtxValueEncoding) {
        if let PtxValueEncoding::Undef = src {
            return;
        }

        // FIXME(eddyb) redundant `mov` can probably be avoided w/ regalloc?

        // HACK(eddyb) work around PTX 8-bit register limitations.
        // FIXME(eddyb) expand 8-bit operations to 16-bit or 32-bit ones.
        let (op, suffix) =
            if dst.bank == PtxRegBank::B8 { ("cvt.u8.", "u8") } else { ("mov.", dst.bank.ty()) };

        self.func.insts.push(PtxInst {
            prefix: cond.map(|cond| cond.inst_prefix()),
            op,
            suffix: Some(suffix),
            operands: [PtxOperand::Reg(dst), src.as_src_operand()].into_iter().collect(),
        });
    }

    // FIXME(eddyb) consider returning a range of labels?
    fn alloc_label(&mut self) -> PtxOperand {
        let idx = self.num_labels;
        self.num_labels = idx.checked_add(1).unwrap();
        PtxOperand::Label { idx }
    }

    // FIXME(eddyb) consider newtyping the index inside `PtxOperand::Label`.
    fn start_label(&mut self, label: PtxOperand) {
        let PtxOperand::Label { idx } = label else {
            unreachable!();
        };

        self.func.insts.push(PtxInst {
            prefix: Some(("%L", idx)),
            op: ":",
            suffix: None,
            operands: [].into_iter().collect(),
        });
    }

    fn branch(&mut self, cond: Option<PtxCond>, label: PtxOperand) {
        // FIXME(eddyb) use `bra.uni` when possible.
        self.func.insts.push(PtxInst {
            prefix: cond.map(|cond| cond.inst_prefix()),
            op: "bra",
            suffix: None,
            operands: [label].into_iter().collect(),
        });
    }
}
