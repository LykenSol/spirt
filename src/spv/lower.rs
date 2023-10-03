//! SPIR-V to SPIR-T lowering.

use crate::spv::{self, spec};
// FIXME(eddyb) import more to avoid `crate::` everywhere.
use crate::{
    AddrSpace, Attr, AttrSet, Const, ConstDef, ConstKind, Context, DataInstDef, DataInstKind,
    DbgSrcLoc, DeclDef, Diag, EntityDefs, ExportKey, Exportee, Func, FuncDecl, FuncDefBody,
    FuncParam, FxIndexMap, GlobalVarDecl, GlobalVarDefBody, GlobalVarInit, Import, InternedStr,
    Module, NodeDef, NodeKind, NodeOutputDecl, Region, RegionDef, RegionInputDecl, SelectionKind,
    Type, TypeDef, TypeKind, TypeOrConst, Value, cfg, print, scalar,
};
use itertools::Either;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
use std::{io, mem};

/// SPIR-T definition of a SPIR-V ID.
enum IdDef {
    Type(Type),
    Const(Const),

    /// Like `Const`, but for SPIR-V "aggregate" (`OpTypeStruct`/`OpTypeArray`)
    /// constants (e.g. `OpConstantComposite`s of those types, but also more
    /// general constants like `OpUndef`/`OpConstantNull` etc.).
    AggregateConst {
        whole_type: Type,

        leaves: SmallVec<[Const; 4]>,
    },

    Func(Func),

    SpvExtInstImport(InternedStr),
    SpvDebugString(InternedStr),
}

impl IdDef {
    fn descr(&self, cx: &Context) -> String {
        match *self {
            // FIXME(eddyb) print these with some kind of "maximum depth",
            // instead of just describing the kind of definition.
            // FIXME(eddyb) replace these with the `Diag` embedding system.
            IdDef::Type(_) => "a type".into(),
            IdDef::Const(_) => "a constant".into(),
            IdDef::AggregateConst { .. } => "an aggregate constant".into(),

            IdDef::Func(_) => "a function".into(),

            IdDef::SpvExtInstImport(name) => {
                format!("`OpExtInstImport {:?}`", &cx[name])
            }
            IdDef::SpvDebugString(s) => format!("`OpString {:?}`", &cx[s]),
        }
    }
}

impl Type {
    // HACK(eddyb) `indices` is a `&mut` because it specifically only consumes
    // the indices it needs, so when this function returns `Some`, all remaining
    // indices will be left over for the caller to process itself.
    fn aggregate_component_path_type_and_leaf_range(
        self,
        cx: &Context,
        indices: &mut impl Iterator<Item = u32>,
    ) -> Option<(Type, Range<usize>)> {
        let (mut leaf_type, mut leaf_range) =
            self.aggregate_component_type_and_leaf_range(cx, indices.next()?)?;

        while let spv::ValueLowering::Disaggregate(_) = cx[leaf_type].spv_value_lowering() {
            let (sub_leaf_type, sub_leaf_range) = match indices.next() {
                Some(i) => leaf_type.aggregate_component_type_and_leaf_range(cx, i)?,
                None => break,
            };

            assert!(sub_leaf_range.end <= leaf_range.len());
            leaf_range.end = leaf_range.start + sub_leaf_range.end;
            leaf_range.start += sub_leaf_range.start;
            leaf_type = sub_leaf_type;
        }

        Some((leaf_type, leaf_range))
    }
}

/// Deferred export, needed because the IDs are initially forward refs.
enum Export {
    Linkage {
        name: InternedStr,
        target_id: spv::Id,
    },
    EntryPoint {
        func_id: spv::Id,
        imms: SmallVec<[spv::Imm; 2]>,
        interface_ids: SmallVec<[spv::Id; 4]>,
    },
}

/// Deferred [`FuncDefBody`], needed because some IDs are initially forward refs.
struct FuncBody {
    func_id: spv::Id,
    func: Func,
    insts: Vec<IntraFuncInst>,
}

struct IntraFuncInst {
    // Instruction aspects that can be pre-lowered:
    attrs: AttrSet,
    result_type: Option<Type>,

    without_ids: spv::Inst,

    // Instruction aspects that cannot be lowered initially (due to forward refs):
    result_id: Option<spv::Id>,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    ids: SmallVec<[spv::Id; 4]>,
}

// FIXME(eddyb) stop abusing `io::Error` for error reporting and switch to `Diag`.
fn invalid(reason: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("malformed SPIR-V ({reason})"))
}

fn invalid_factory_for_spv_inst(
    inst: &spv::Inst,
    result_id: Option<spv::Id>,
    ids: &[spv::Id],
) -> impl Fn(&str) -> io::Error {
    let opcode = inst.opcode;
    let first_id_operand = ids.first().copied();
    move |msg: &str| {
        let result_prefix = result_id.map(|id| format!("%{id} = ")).unwrap_or_default();
        let operand_suffix = first_id_operand.map(|id| format!(" %{id} ...")).unwrap_or_default();
        invalid(&format!("in {result_prefix}{}{operand_suffix}: {msg}", opcode.name()))
    }
}

// FIXME(eddyb) provide more information about any normalization that happened:
// * stats about deduplication that occured through interning
// * sets of unused global vars and functions (and types+consts only they use)
// FIXME(eddyb) use `Diag` instead of `io::Error`, maybe with a return type like
// `Result<Module, IncompletelyLoweredModule>` where `IncompletelyLoweredModule`
// contains a `Module`, maps of all the SPIR-V IDs (to the SPIR-T definitions),
// global `Diag`s (where they can't be attached to specific `AttrSet`s), etc.
impl Module {
    pub fn lower_from_spv_file(cx: Rc<Context>, path: impl AsRef<Path>) -> io::Result<Self> {
        Self::lower_from_spv_module_parser(cx, spv::read::ModuleParser::read_from_spv_file(path)?)
    }

    pub fn lower_from_spv_bytes(cx: Rc<Context>, spv_bytes: Vec<u8>) -> io::Result<Self> {
        Self::lower_from_spv_module_parser(
            cx,
            spv::read::ModuleParser::read_from_spv_bytes(spv_bytes)?,
        )
    }

    pub fn lower_from_spv_module_parser(
        cx: Rc<Context>,
        parser: spv::read::ModuleParser,
    ) -> io::Result<Self> {
        let spv_spec = spec::Spec::get();
        let wk = &spv_spec.well_known;

        // HACK(eddyb) used to quickly check whether an `OpVariable` is global.
        let storage_class_function_imm = spv::Imm::Short(wk.StorageClass, wk.Function);

        let mut module = {
            let [magic, version, generator_magic, id_bound, reserved_inst_schema] = parser.header;

            // Ensured above (this is the value after any endianness swapping).
            assert_eq!(magic, spv_spec.magic);

            let [version_reserved_hi, version_major, version_minor, version_reserved_lo] =
                version.to_be_bytes();

            if (version_reserved_lo, version_reserved_hi) != (0, 0) {
                return Err(invalid(&format!(
                    "version 0x{version:08x} is not in expected (0.major.minor.0) form"
                )));
            }

            // FIXME(eddyb) maybe use this somehow? (e.g. check IDs against it)
            let _ = id_bound;

            if reserved_inst_schema != 0 {
                return Err(invalid(&format!(
                    "unknown instruction schema {reserved_inst_schema} - only 0 is supported"
                )));
            }

            Self::new(
                cx.clone(),
                crate::ModuleDialect::Spv(spv::Dialect {
                    version_major,
                    version_minor,

                    capabilities: BTreeSet::new(),
                    extensions: BTreeSet::new(),

                    addressing_model: 0,
                    memory_model: 0,
                }),
                crate::ModuleDebugInfo::Spv(spv::ModuleDebugInfo {
                    original_generator_magic: NonZeroU32::new(generator_magic),

                    source_languages: BTreeMap::new(),
                    source_extensions: vec![],
                    module_processes: vec![],
                }),
            )
        };

        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        enum Seq {
            Capability,
            Extension,
            ExtInstImport,
            MemoryModel,
            EntryPoint,
            ExecutionMode,
            DebugStringAndSource,
            DebugName,
            DebugModuleProcessed,
            Decoration,

            // NOTE(eddyb) not its own section, but only a "checkpoint", forcing
            // instructions following `OpLine`/`OpNoLine` into later sections.
            DebugLine,

            TypeConstOrGlobalVar,
            Function,
        }
        let mut seq = None;

        let mut has_memory_model = false;
        let mut pending_attrs = FxHashMap::<spv::Id, crate::AttrSetDef>::default();
        let mut pending_imports = FxHashMap::<spv::Id, Import>::default();
        let mut pending_exports = vec![];
        let mut current_dbg_src_loc = None;
        let mut current_block_id = None; // HACK(eddyb) for `current_dbg_src_loc` resets.
        let mut id_defs = FxHashMap::default();
        let mut pending_func_bodies = vec![];
        let mut current_func_body = None;

        let mut spv_insts = parser.peekable();
        while let Some(mut inst) = spv_insts.next().transpose()? {
            let opcode = inst.opcode;

            let invalid = invalid_factory_for_spv_inst(&inst, inst.result_id, &inst.ids);

            // Handle line debuginfo early, as it doesn't have its own section,
            // but rather can go almost anywhere among globals and functions.
            //
            // FIXME(eddyb) also support debuginfo "extended instruction sets"
            // (e.g. `OpenCL.DebugInfo.100`, `NonSemantic.Shader.DebugInfo.100`)
            if [wk.OpLine, wk.OpNoLine].contains(&opcode) {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                current_dbg_src_loc = if opcode == wk.OpLine {
                    match (&inst.imms[..], &inst.ids[..]) {
                        (
                            &[spv::Imm::Short(l_kind, line), spv::Imm::Short(c_kind, col)],
                            &[file_path_id],
                        ) => {
                            assert_eq!([l_kind, c_kind], [wk.LiteralInteger; 2]);
                            let file_path = match id_defs.get(&file_path_id) {
                                Some(&IdDef::SpvDebugString(s)) => s,
                                _ => {
                                    return Err(invalid(&format!(
                                        "%{file_path_id} is not an OpString"
                                    )));
                                }
                            };
                            Some(DbgSrcLoc {
                                file_path,
                                start_line_col: (line, col),
                                end_line_col: (line, col),
                                inlined_callee_name_and_call_site: None,
                            })
                        }
                        _ => unreachable!(),
                    }
                } else {
                    assert!(inst.imms.is_empty() && inst.ids.is_empty());
                    None
                };

                // Advance to `Seq::DebugLine` if we're not there yet, forcing
                // any following instructions to not be in earlier sections.
                seq = seq.max(Some(Seq::DebugLine));
                continue;
            }

            // Reset line debuginfo when crossing/leaving blocks.
            let new_block_id = if opcode == wk.OpLabel {
                Some(inst.result_id.unwrap())
            } else if opcode == wk.OpFunctionEnd {
                None
            } else {
                current_block_id
            };
            if current_block_id != new_block_id {
                current_dbg_src_loc = None;
            }
            current_block_id = new_block_id;

            let mut attrs =
                inst.result_id.and_then(|id| pending_attrs.remove(&id)).unwrap_or_default();

            if let Some(dbg_src_loc) = current_dbg_src_loc {
                attrs.set_dbg_src_loc(dbg_src_loc);
            }

            // Take certain bitflags operands out of the instruction and rewrite
            // them into attributes instead.
            inst.imms.retain(|imm| match *imm {
                spv::Imm::Short(kind, word) if kind == wk.FunctionControl => {
                    if word != 0 {
                        attrs.attrs.insert(Attr::SpvBitflagsOperand(*imm));
                    }
                    false
                }
                _ => true,
            });

            let mut attrs = cx.intern(attrs);

            // FIXME(eddyb) move this kind of lookup into methods on some sort
            // of "lowering context" type.
            let result_type = inst
                .result_type_id
                .map(|type_id| match id_defs.get(&type_id) {
                    Some(&IdDef::Type(ty)) => Ok(ty),
                    Some(id_def) => Err(invalid(&format!(
                        "result type %{} should be a type, not a {}",
                        type_id,
                        id_def.descr(&cx)
                    ))),
                    None => Err(invalid(&format!("result type %{type_id} not defined"))),
                })
                .transpose()?;

            let inst_category = spv_spec.instructions[opcode].category;

            let next_seq = if opcode == wk.OpCapability {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let cap = match (&inst.imms[..], &inst.ids[..]) {
                    (&[spv::Imm::Short(kind, cap)], &[]) => {
                        assert_eq!(kind, wk.Capability);
                        cap
                    }
                    _ => unreachable!(),
                };

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.capabilities.insert(cap);
                    }
                }

                Seq::Capability
            } else if opcode == wk.OpExtension {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let ext = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.extensions.insert(ext);
                    }
                }

                Seq::Extension
            } else if opcode == wk.OpExtInstImport {
                assert!(inst.result_type_id.is_none() && inst.ids.is_empty());
                let id = inst.result_id.unwrap();
                let name = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                id_defs.insert(id, IdDef::SpvExtInstImport(cx.intern(name)));

                Seq::ExtInstImport
            } else if opcode == wk.OpMemoryModel {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (addressing_model, memory_model) = match (&inst.imms[..], &inst.ids[..]) {
                    (&[spv::Imm::Short(am_kind, am), spv::Imm::Short(mm_kind, mm)], &[]) => {
                        assert_eq!([am_kind, mm_kind], [wk.AddressingModel, wk.MemoryModel]);
                        (am, mm)
                    }
                    _ => unreachable!(),
                };

                if has_memory_model {
                    return Err(invalid("duplicate OpMemoryModel"));
                }
                has_memory_model = true;

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.addressing_model = addressing_model;
                        dialect.memory_model = memory_model;
                    }
                }

                Seq::MemoryModel
            } else if opcode == wk.OpString {
                assert!(inst.result_type_id.is_none() && inst.ids.is_empty());
                let id = inst.result_id.unwrap();
                let s = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                id_defs.insert(id, IdDef::SpvDebugString(cx.intern(s)));

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpSource {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (lang, version) = match inst.imms[..] {
                    [spv::Imm::Short(l_kind, lang), spv::Imm::Short(v_kind, version), ..] => {
                        assert_eq!([l_kind, v_kind], [wk.SourceLanguage, wk.LiteralInteger]);
                        (lang, version)
                    }
                    _ => unreachable!(),
                };

                let debug_sources = match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => debug_info
                        .source_languages
                        .entry(spv::DebugSourceLang { lang, version })
                        .or_default(),
                };

                match (&inst.imms[2..], &inst.ids[..]) {
                    (contents, &[file_path_id]) => {
                        let file_path = match id_defs.get(&file_path_id) {
                            Some(&IdDef::SpvDebugString(s)) => s,
                            _ => {
                                return Err(invalid(&format!(
                                    "%{file_path_id} is not an OpString"
                                )));
                            }
                        };
                        let mut contents = if contents.is_empty() {
                            String::new()
                        } else {
                            spv::extract_literal_string(contents)
                                .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?
                        };

                        // Absorb all following `OpSourceContinued` into `contents`.
                        while let Some(Ok(cont_inst)) = spv_insts.peek() {
                            if cont_inst.opcode != wk.OpSourceContinued {
                                break;
                            }
                            let cont_inst = spv_insts.next().unwrap().unwrap();

                            assert!(
                                cont_inst.result_type_id.is_none()
                                    && cont_inst.result_id.is_none()
                                    && cont_inst.ids.is_empty()
                            );
                            let cont_contents = spv::extract_literal_string(&cont_inst.imms)
                                .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;
                            contents += &cont_contents;
                        }

                        debug_sources.file_contents.insert(file_path, contents);
                    }
                    (&[], &[]) => {}
                    _ => unreachable!(),
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpSourceContinued {
                return Err(invalid("must follow OpSource"));
            } else if opcode == wk.OpSourceExtension {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let ext = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => {
                        debug_info.source_extensions.push(ext);
                    }
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpModuleProcessed {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let proc = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => {
                        debug_info.module_processes.push(proc);
                    }
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugModuleProcessed
            } else if opcode == wk.OpEntryPoint {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                pending_exports.push(Export::EntryPoint {
                    func_id: inst.ids[0],
                    imms: inst.without_ids.imms,
                    interface_ids: inst.ids[1..].iter().copied().collect(),
                });

                Seq::EntryPoint
            } else if [
                wk.OpExecutionMode,
                wk.OpExecutionModeId, // FIXME(eddyb) not actually supported
                wk.OpName,
                wk.OpMemberName,
                wk.OpDecorate,
                wk.OpMemberDecorate,
                wk.OpDecorateId, // FIXME(eddyb) not actually supported
                wk.OpDecorateString,
                wk.OpMemberDecorateString,
            ]
            .contains(&opcode)
            {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                let target_id = inst.ids[0];
                if inst.ids.len() > 1 {
                    return Err(invalid("unsupported decoration with ID"));
                }

                match inst.imms[..] {
                    // Special-case `OpDecorate LinkageAttributes ... Import|Export`.
                    [
                        decoration @ spv::Imm::Short(..),
                        ref name @ ..,
                        spv::Imm::Short(lt_kind, linkage_type),
                    ] if opcode == wk.OpDecorate
                        && decoration == spv::Imm::Short(wk.Decoration, wk.LinkageAttributes)
                        && lt_kind == wk.LinkageType
                        && [wk.Import, wk.Export].contains(&linkage_type) =>
                    {
                        let name = spv::extract_literal_string(name)
                            .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;
                        let name = cx.intern(name);

                        if linkage_type == wk.Import {
                            pending_imports.insert(target_id, Import::LinkName(name));
                        } else {
                            pending_exports.push(Export::Linkage { name, target_id });
                        }
                    }

                    _ => {
                        pending_attrs
                            .entry(target_id)
                            .or_default()
                            .attrs
                            .insert(Attr::SpvAnnotation(inst.without_ids));
                    }
                };

                if [wk.OpExecutionMode, wk.OpExecutionModeId].contains(&opcode) {
                    Seq::ExecutionMode
                } else if [wk.OpName, wk.OpMemberName].contains(&opcode) {
                    Seq::DebugName
                } else {
                    Seq::Decoration
                }
            } else if [wk.OpDecorationGroup, wk.OpGroupDecorate, wk.OpGroupMemberDecorate]
                .contains(&opcode)
            {
                return Err(invalid("unsupported decoration groups (officially deprecated)"));
            } else if opcode == wk.OpTypeForwardPointer {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (id, sc) = match (&inst.imms[..], &inst.ids[..]) {
                    (&[sc], &[id]) => (id, sc),
                    _ => unreachable!(),
                };

                // HACK(eddyb) this is not a proper implementation - one would
                // require fixpoint (aka "μ" aka "mu") types - but for now this
                // serves as a first approximation for a "deferred error".
                let ty = cx.intern(TypeDef {
                    attrs: mem::take(&mut attrs),
                    kind: TypeKind::SpvInst {
                        spv_inst: spv::Inst { opcode, imms: [sc].into_iter().collect() },
                        type_and_const_inputs: [].into_iter().collect(),
                        value_lowering: Default::default(),
                    },
                });
                id_defs.insert(id, IdDef::Type(ty));

                Seq::TypeConstOrGlobalVar
            } else if inst_category == spec::InstructionCategory::Type {
                assert!(inst.result_type_id.is_none());
                let id = inst.result_id.unwrap();
                let type_and_const_inputs: SmallVec<_> = inst
                    .ids
                    .iter()
                    .map(|&id| match id_defs.get(&id) {
                        Some(&IdDef::Type(ty)) => Ok(TypeOrConst::Type(ty)),
                        Some(&IdDef::Const(ct)) => Ok(TypeOrConst::Const(ct)),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("a forward reference to %{id}")),
                    })
                    .map(|result| {
                        result.map_err(|descr| {
                            invalid(&format!("unsupported use of {descr} in a type"))
                        })
                    })
                    .collect::<Result<_, _>>()?;

                let ty = cx.intern(TypeDef {
                    attrs: mem::take(&mut attrs),
                    kind: inst.without_ids.into_canonical_type_with(&cx, type_and_const_inputs),
                });
                id_defs.insert(id, IdDef::Type(ty));

                Seq::TypeConstOrGlobalVar
            } else if inst_category == spec::InstructionCategory::Const
                || inst.always_lower_as_const()
            {
                let id = inst.result_id.unwrap();

                let ty = result_type.unwrap();

                // HACK(eddyb) while creating constants of unsized array types
                // is *technically* illegal in SPIR-V, array semantics always
                // are length-independent, so we can pretend this is an array
                // of the right length (as long as we track the error on it).
                let maybe_fixup_unsized_array_type = |ty: Type| {
                    if ![wk.OpConstantComposite, wk.OpSpecConstantComposite].contains(&opcode) {
                        return None;
                    };
                    let actual_component_count = u32::try_from(inst.ids.len()).ok()?;

                    let ty_def = &cx[ty];
                    let elem_type_of_unsized_array = match &ty_def.kind {
                        TypeKind::SpvInst { spv_inst: ty_inst, type_and_const_inputs, .. } => {
                            match type_and_const_inputs[..] {
                                [TypeOrConst::Type(elem_type), TypeOrConst::Const(len)]
                                    if ty_inst.opcode == wk.OpTypeArray
                                        && len.as_scalar(&cx).is_none() =>
                                {
                                    elem_type
                                }
                                [TypeOrConst::Type(elem_type)]
                                    if ty_inst.opcode == wk.OpTypeRuntimeArray =>
                                {
                                    elem_type
                                }
                                _ => return None,
                            }
                        }
                        _ => return None,
                    };
                    let mut attrs = ty_def.attrs;
                    attrs.push_diag(
                        &cx,
                        Diag::err([
                            "illegal constant: values of type `".into(),
                            ty.into(),
                            "` should only be accessed through pointers".into(),
                        ]),
                    );
                    Some(
                        cx.intern(TypeDef {
                            attrs,
                            kind: spv::Inst::from(wk.OpTypeArray).into_canonical_type_with(
                                &cx,
                                [
                                    TypeOrConst::Type(elem_type_of_unsized_array),
                                    TypeOrConst::Const(
                                        cx.intern(scalar::Const::from_u32(actual_component_count)),
                                    ),
                                ]
                                .into_iter()
                                .collect(),
                            ),
                        }),
                    )
                };
                let ty = maybe_fixup_unsized_array_type(ty).unwrap_or(ty);

                let mut all_leaves = SmallVec::new();
                match cx[ty].spv_value_lowering() {
                    spv::ValueLowering::Direct => {
                        all_leaves.reserve(inst.ids.len());
                    }
                    spv::ValueLowering::Disaggregate(_) => {
                        // HACK(eddyb) this expands `OpUndef`/`OpConstantNull`.
                        // FIXME(eddyb) this could potentially create a very
                        // inefficient large array, even when the intent can
                        // be expressed much more compactly in theory.
                        if inst.lower_const_by_distributing_to_aggregate_leaves() {
                            assert_eq!(inst.ids.len(), 0);
                            all_leaves.extend(ty.disaggregated_leaf_types(&cx).map(|leaf_type| {
                                cx.intern(ConstDef {
                                    attrs: Default::default(),
                                    ty: leaf_type,
                                    kind: inst
                                        .as_canonical_const(&cx, leaf_type, &[])
                                        .unwrap_or_else(|| ConstKind::SpvInst {
                                            spv_inst_and_const_inputs: Rc::new((
                                                inst.without_ids.clone(),
                                                [].into_iter().collect(),
                                            )),
                                        }),
                                })
                            }));
                        } else if [wk.OpConstantComposite, wk.OpSpecConstantComposite]
                            .contains(&opcode)
                        {
                            all_leaves.reserve(cx[ty].disaggregated_leaf_count());
                        } else {
                            attrs.push_diag(
                                &cx,
                                Diag::bug(["unsupported aggregate-producing constant".into()]),
                            );
                        }
                    }
                }

                let invalid = |descr| invalid(&format!("unsupported use of {descr} in a constant"));
                for &id in &inst.ids {
                    match id_defs.get(&id) {
                        Some(&IdDef::Const(ct)) => {
                            all_leaves.push(ct);
                        }
                        Some(IdDef::AggregateConst { whole_type, leaves }) => {
                            all_leaves.extend(leaves.iter().copied());

                            match cx[ty].spv_value_lowering() {
                                // FIXME(eddyb) this also covers invalid consts
                                // of e.g. unsized aggregate types, as well.
                                spv::ValueLowering::Direct => {
                                    attrs.push_diag(
                                        &cx,
                                        Diag::err([
                                            "unexpected aggregate constant of type `".into(),
                                            (*whole_type).into(),
                                            "`".into(),
                                        ]),
                                    );
                                }
                                spv::ValueLowering::Disaggregate(_) => {}
                            }
                        }
                        Some(id_def) => return Err(invalid(&id_def.descr(&cx))),
                        None => return Err(invalid(&format!("a forward reference to %{id}"))),
                    }
                }

                let lowering = &cx[ty].spv_value_lowering();
                let lowering = match lowering {
                    spv::ValueLowering::Disaggregate(_)
                        if cx[ty].disaggregated_leaf_count() != all_leaves.len() =>
                    {
                        attrs.push_diag(
                            &cx,
                            Diag::err([format!(
                                "aggregate leaf count mismatch (expected {}, found {})",
                                cx[ty].disaggregated_leaf_count(),
                                all_leaves.len()
                            )
                            .into()]),
                        );
                        // HACK(eddyb) pretend the type isn't an aggregate, so
                        // that it doesn't end up using `IdDef::AggregateConst`,
                        // which requires having the exact number of leaves.
                        &spv::ValueLowering::Direct
                    }
                    _ => lowering,
                };

                let attrs = mem::take(&mut attrs);
                id_defs.insert(id, match lowering {
                    spv::ValueLowering::Direct => IdDef::Const(cx.intern(ConstDef {
                        attrs,
                        ty,
                        kind: inst.as_canonical_const(&cx, ty, &all_leaves).unwrap_or_else(|| {
                            ConstKind::SpvInst {
                                spv_inst_and_const_inputs: Rc::new((inst.without_ids, all_leaves)),
                            }
                        }),
                    })),
                    spv::ValueLowering::Disaggregate(_) => {
                        // FIXME(eddyb) this may lose semantic `attrs`.
                        IdDef::AggregateConst { whole_type: ty, leaves: all_leaves }
                    }
                });

                if inst_category != spec::InstructionCategory::Const {
                    // `OpUndef` can appear either among constants, or in a
                    // function, so at most advance `seq` to globals.
                    seq.max(Some(Seq::TypeConstOrGlobalVar)).unwrap()
                } else {
                    Seq::TypeConstOrGlobalVar
                }
            } else if opcode == wk.OpVariable && current_func_body.is_none() {
                let global_var_id = inst.result_id.unwrap();
                let type_of_ptr_to_global_var = result_type.unwrap();

                if inst.imms[0] == storage_class_function_imm {
                    return Err(invalid("`Function` storage class outside function"));
                }

                let storage_class = match inst.imms[..] {
                    [spv::Imm::Short(kind, storage_class)] => {
                        assert_eq!(kind, wk.StorageClass);
                        storage_class
                    }
                    _ => unreachable!(),
                };
                let initializer = match inst.ids[..] {
                    [initializer] => Some(initializer),
                    [] => None,
                    _ => unreachable!(),
                };

                let initializer = initializer
                    .map(|id| match id_defs.get(&id) {
                        Some(&IdDef::Const(ct)) => Ok(GlobalVarInit::Direct(ct)),
                        Some(IdDef::AggregateConst { whole_type, leaves }) => {
                            Ok(GlobalVarInit::SpvAggregate {
                                ty: *whole_type,
                                leaves: leaves.clone(),
                            })
                        }
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("a forward reference to %{id}")),
                    })
                    .transpose()
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the initializer of a global variable"
                        ))
                    })?;

                let def = match pending_imports.remove(&global_var_id) {
                    Some(import @ Import::LinkName(name)) => {
                        if initializer.is_some() {
                            return Err(invalid(&format!(
                                "global variable with initializer decorated as `Import` of {:?}",
                                &cx[name]
                            )));
                        }
                        DeclDef::Imported(import)
                    }
                    None => DeclDef::Present(GlobalVarDefBody { initializer }),
                };

                let global_var = module.global_vars.define(&cx, GlobalVarDecl {
                    attrs: mem::take(&mut attrs),
                    type_of_ptr_to: type_of_ptr_to_global_var,
                    shape: None,
                    addr_space: AddrSpace::SpvStorageClass(storage_class),
                    def,
                });
                let ptr_to_global_var = cx.intern(ConstDef {
                    attrs: AttrSet::default(),
                    ty: type_of_ptr_to_global_var,
                    kind: ConstKind::PtrToGlobalVar(global_var),
                });
                id_defs.insert(global_var_id, IdDef::Const(ptr_to_global_var));

                Seq::TypeConstOrGlobalVar
            } else if opcode == wk.OpFunction {
                if current_func_body.is_some() {
                    return Err(invalid("nested OpFunction while still in a function"));
                }

                let func_id = inst.result_id.unwrap();
                let func_ret_type = result_type.unwrap();

                let func_type_id = match (&inst.imms[..], &inst.ids[..]) {
                    // NOTE(eddyb) the `FunctionControl` operand is already gone,
                    // having been converted into an attribute above.
                    (&[], &[func_type_id]) => func_type_id,
                    _ => unreachable!(),
                };

                let (func_type_ret_type, func_type_param_types) =
                    match id_defs.get(&func_type_id) {
                        Some(&IdDef::Type(ty)) => match &cx[ty].kind {
                            TypeKind::SpvInst { spv_inst, type_and_const_inputs, .. }
                                if spv_inst.opcode == wk.OpTypeFunction =>
                            {
                                let mut types =
                                    type_and_const_inputs.iter().map(|&ty_or_ct| match ty_or_ct {
                                        TypeOrConst::Type(ty) => ty,
                                        TypeOrConst::Const(_) => unreachable!(),
                                    });
                                Some((types.next().unwrap(), types))
                            }
                            _ => None,
                        },
                        _ => None,
                    }
                    .ok_or_else(|| {
                        invalid(&format!("function type %{func_type_id} not an `OpTypeFunction`"))
                    })?;

                if func_ret_type != func_type_ret_type {
                    // FIXME(remove) embed IDs in errors by moving them to the
                    // `let invalid = |...| ...;` closure that wraps insts.
                    return Err(invalid(
                        &print::Plan::for_root(
                            &cx,
                            &Diag::err([
                                format!("in %{}, ", func_id).into(),
                                "return type differs between `OpFunction` (".into(),
                                func_ret_type.into(),
                                ") and `OpTypeFunction` (".into(),
                                func_type_ret_type.into(),
                                ")".into(),
                            ])
                            .message,
                        )
                        .pretty_print()
                        .to_string(),
                    ));
                }

                let def = match pending_imports.remove(&func_id) {
                    Some(import) => DeclDef::Imported(import),
                    None => {
                        let mut regions = EntityDefs::default();
                        let body = regions.define(&cx, RegionDef::default());
                        DeclDef::Present(FuncDefBody {
                            regions,
                            nodes: Default::default(),
                            body,
                            unstructured_cfg: Some(cfg::ControlFlowGraph::default()),
                        })
                    }
                };

                // Always flatten aggregates in param and return types.
                let ret_types = match &cx[func_ret_type].kind {
                    // HACK(eddyb) `OpTypeVoid` special-cased here as if it were
                    // an aggregate with `0` leaves.
                    TypeKind::SpvInst { spv_inst: func_ret_type_spv_inst, .. }
                        if func_ret_type_spv_inst.opcode == wk.OpTypeVoid =>
                    {
                        [].into_iter().collect()
                    }

                    _ => func_ret_type.disaggregated_leaf_types(&cx).collect(),
                };
                let mut params = SmallVec::with_capacity(func_type_param_types.len());
                for param_type in func_type_param_types {
                    params.extend(
                        param_type
                            .disaggregated_leaf_types(&cx)
                            .map(|ty| FuncParam { attrs: AttrSet::default(), ty }),
                    );
                }

                let func = module.funcs.define(&cx, FuncDecl {
                    attrs: mem::take(&mut attrs),
                    ret_types,
                    params,
                    def,
                });
                id_defs.insert(func_id, IdDef::Func(func));

                current_func_body = Some(FuncBody { func_id, func, insts: vec![] });

                Seq::Function
            } else if opcode == wk.OpFunctionEnd {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                assert!(inst.imms.is_empty() && inst.ids.is_empty());

                let func_body = current_func_body
                    .take()
                    .ok_or_else(|| invalid("nested OpFunction while still in a function"))?;

                pending_func_bodies.push(func_body);

                Seq::Function
            } else {
                let func_body = current_func_body
                    .as_mut()
                    .ok_or_else(|| invalid("expected only inside a function"))?;
                assert_eq!(seq, Some(Seq::Function));

                func_body.insts.push(IntraFuncInst {
                    attrs: mem::take(&mut attrs),
                    result_type,

                    without_ids: spv::Inst { opcode, imms: inst.without_ids.imms },
                    result_id: inst.result_id,
                    ids: inst.ids,
                });

                Seq::Function
            };
            if let Some(prev_seq) = seq {
                if prev_seq > next_seq {
                    return Err(invalid(&format!(
                        "out of order: {next_seq:?} instructions must precede {prev_seq:?} instructions"
                    )));
                }
            }
            seq = Some(next_seq);

            if attrs != Default::default() {
                return Err(invalid("unused decorations / line debuginfo"));
            }
        }

        if !has_memory_model {
            return Err(invalid("missing OpMemoryModel"));
        }

        if !pending_attrs.is_empty() {
            let ids = pending_attrs.keys().collect::<BTreeSet<_>>();
            return Err(invalid(&format!("decorated IDs never defined: {ids:?}")));
        }

        if current_func_body.is_some() {
            return Err(invalid("OpFunction without matching OpFunctionEnd"));
        }

        // Process function bodies, having seen the whole module.
        for func_body in pending_func_bodies {
            let FuncBody { func_id, func, insts: raw_insts } = func_body;

            let func_decl = &mut module.funcs[func];

            #[derive(PartialEq, Eq, Hash)]
            struct PhiKey {
                source_block_id: spv::Id,
                target_block_id: spv::Id,
                // FIXME(eddyb) remove this, key phis only by the edge, and keep
                // a per-edge list of phi input `spv::Id`s (with validation for
                // missing entries/duplicates).
                target_phi_idx: u32,
            }

            struct BlockDetails {
                label_id: spv::Id,
                phi_count: usize,

                // FIXME(eddyb) how inefficient is `FxIndexMap<spv::Id, Type>`?
                // (vs e.g. a bitset combined with not duplicating `Type`s per-block?)
                cfgssa_inter_block_uses: FxIndexMap<spv::Id, Type>,
            }

            // Gather `OpLabel`s and `OpPhi`s early (so they can be random-accessed).
            let mut phi_to_values = FxIndexMap::<PhiKey, SmallVec<[spv::Id; 1]>>::default();
            // FIXME(eddyb) wouldn't `EntityOrientedDenseMap` make more sense?
            let mut block_details = FxIndexMap::<Region, BlockDetails>::default();
            let mut has_blocks = false;
            let mut cfgssa_def_map = {
                // FIXME(eddyb) in theory, this could be a toggle, but there is
                // very little value in allowing dominance-based SSA use rules.
                const SPIRT_CFGSSA_UNDOMINATE: bool = true;

                SPIRT_CFGSSA_UNDOMINATE.then(|| {
                    let mut def_map = crate::cfgssa::DefMap::new();

                    // HACK(eddyb) allow e.g. `OpFunctionParameter` to
                    // be treated like `OpPhi`s of the entry block.
                    if let DeclDef::Present(func_def_body) = &func_decl.def {
                        def_map.add_block(func_def_body.body);
                    }

                    def_map
                })
            };
            {
                for raw_inst in &raw_insts {
                    let IntraFuncInst {
                        without_ids: spv::Inst { opcode, ref imms },
                        result_type,
                        result_id,
                        ..
                    } = *raw_inst;

                    if opcode == wk.OpFunctionParameter {
                        continue;
                    }

                    let is_entry_block = !has_blocks;
                    has_blocks = true;

                    let func_def_body = match &mut func_decl.def {
                        // Error will be emitted later, below.
                        DeclDef::Imported(_) => continue,
                        DeclDef::Present(def) => def,
                    };

                    if opcode == wk.OpLabel {
                        let block = if is_entry_block {
                            // A `Region` was defined earlier,
                            // to be able to create the `FuncDefBody`.
                            func_def_body.body
                        } else {
                            func_def_body.regions.define(&cx, RegionDef::default())
                        };
                        block_details.insert(block, BlockDetails {
                            label_id: result_id.unwrap(),
                            phi_count: 0,
                            cfgssa_inter_block_uses: Default::default(),
                        });
                    } else if opcode == wk.OpPhi {
                        let (_, block_details) = match block_details.last_mut() {
                            Some(entry) => entry,
                            // Error will be emitted later, below.
                            None => continue,
                        };

                        let phi_idx = block_details.phi_count;
                        block_details.phi_count = phi_idx.checked_add(1).unwrap();
                        let phi_idx = u32::try_from(phi_idx).unwrap();

                        assert!(imms.is_empty());
                        // FIXME(eddyb) use `array_chunks` when that's stable.
                        for value_and_source_block_id in raw_inst.ids.chunks(2) {
                            let &[value_id, source_block_id]: &[_; 2] =
                                value_and_source_block_id.try_into().unwrap();

                            phi_to_values
                                .entry(PhiKey {
                                    source_block_id,
                                    target_block_id: block_details.label_id,
                                    target_phi_idx: phi_idx,
                                })
                                .or_default()
                                .push(value_id);
                        }
                    }

                    if let Some(def_map) = &mut cfgssa_def_map {
                        if let DeclDef::Present(func_def_body) = &func_decl.def {
                            let current_block = match block_details.last() {
                                Some((&current_block, _)) => current_block,
                                // HACK(eddyb) ensure e.g. `OpFunctionParameter`
                                // are treated like `OpPhi`s of the entry block.
                                None => func_def_body.body,
                            };

                            if opcode == wk.OpLabel {
                                // HACK(eddyb) the entry block was already added.
                                if current_block != func_def_body.body {
                                    def_map.add_block(current_block);
                                }
                                continue;
                            }

                            if let Some(id) = result_id {
                                def_map.add_def(current_block, id, result_type.unwrap());
                            }
                        }
                    }
                }
            }

            let mut params = SmallVec::<[_; 8]>::new();

            let mut func_def_body = if has_blocks {
                match &mut func_decl.def {
                    DeclDef::Imported(Import::LinkName(name)) => {
                        return Err(invalid(&format!(
                            "non-empty function %{} decorated as `Import` of {:?}",
                            func_id, &cx[*name]
                        )));
                    }
                    DeclDef::Present(def) => Some(def),
                }
            } else {
                match func_decl.def {
                    DeclDef::Imported(Import::LinkName(_)) => {}
                    DeclDef::Present(_) => {
                        // FIXME(remove) embed IDs in errors by moving them to the
                        // `let invalid = |...| ...;` closure that wraps insts.
                        return Err(invalid(&format!(
                            "function %{func_id} lacks any blocks, \
                             but isn't an import either"
                        )));
                    }
                }

                None
            };

            // HACK(eddyb) this is generic to allow `IdDef::AggregateConst`s
            // to be converted to `LocalIdDef::Value`s, inside `lookup_id`.
            enum LocalIdDef<VL = Either<ValueRange, SmallVec<[Value; 4]>>> {
                Value { whole_type: Type, leaves: VL },
                BlockLabel(Region),
            }

            struct ValueRange {
                start: Value,
                count: u32,
            }

            impl ValueRange {
                fn iter(&self) -> impl ExactSizeIterator<Item = Value> + Clone {
                    let start = self.start;
                    (0..self.count).map(move |i| match start {
                        Value::Const(_) => unreachable!(),
                        Value::RegionInput { region, input_idx } => Value::RegionInput {
                            region,
                            input_idx: input_idx.checked_add(i).unwrap(),
                        },
                        Value::NodeOutput { node, output_idx } => Value::NodeOutput {
                            node,
                            output_idx: output_idx.checked_add(i).unwrap(),
                        },
                    })
                }
            }

            let mut local_id_defs = FxIndexMap::<spv::Id, LocalIdDef>::default();

            // Labels can be forward-referenced, so always have them present.
            local_id_defs.extend(
                block_details
                    .iter()
                    .map(|(&region, details)| (details.label_id, LocalIdDef::BlockLabel(region))),
            );

            // HACK(eddyb) an entire separate traversal is required to find
            // all inter-block uses, before any blocks get lowered to SPIR-T.
            let mut cfgssa_use_accumulator = cfgssa_def_map
                .as_ref()
                .filter(|_| func_def_body.is_some())
                .map(crate::cfgssa::UseAccumulator::new);
            if let Some(use_acc) = &mut cfgssa_use_accumulator {
                // HACK(eddyb) ensure e.g. `OpFunctionParameter`
                // are treated like `OpPhi`s of the entry block.
                let mut current_block = func_def_body.as_ref().unwrap().body;
                for raw_inst in &raw_insts {
                    let IntraFuncInst {
                        without_ids: spv::Inst { opcode, ref imms },
                        result_id,
                        ..
                    } = *raw_inst;

                    if opcode == wk.OpLabel {
                        current_block = match local_id_defs[&result_id.unwrap()] {
                            LocalIdDef::BlockLabel(region) => region,
                            LocalIdDef::Value { .. } => unreachable!(),
                        };
                        continue;
                    }

                    if opcode == wk.OpPhi {
                        assert!(imms.is_empty());
                        // FIXME(eddyb) use `array_chunks` when that's stable.
                        for value_and_source_block_id in raw_inst.ids.chunks(2) {
                            let &[value_id, source_block_id]: &[_; 2] =
                                value_and_source_block_id.try_into().unwrap();

                            if let Some(&LocalIdDef::BlockLabel(source_block)) =
                                local_id_defs.get(&source_block_id)
                            {
                                // HACK(eddyb) `value_id` would be explicitly used
                                // in `source_block`, in a "BB args" representation,
                                // but phis move the use to the edge's target.
                                use_acc.add_use(source_block, value_id);
                            }
                        }
                        continue;
                    }

                    // HACK(eddyb) while including merges as edges may seem useful,
                    // they don't participate in dominance (and thus SSA validity),
                    // and if there's any chance `current_block` is *not* the
                    // closest dominator of a merge, that merge could contain
                    // uses that don't belong/are illegal in `current_block`.
                    if [wk.OpSelectionMerge, wk.OpLoopMerge].contains(&opcode) {
                        continue;
                    }

                    for &id in &raw_inst.ids {
                        // HACK(eddyb) treat all mentions of `OpLabel` IDs as
                        // CFG edge targets, which turns out to be accurate,
                        // except for `OpPhi`/`OpSelectionMerge`/`OpLoopMerge`
                        // (which are already special-cased above).
                        if let Some(&LocalIdDef::BlockLabel(target_block)) = local_id_defs.get(&id)
                        {
                            use_acc.add_edge(current_block, target_block);
                        } else {
                            // HACK(eddyb) this heavily relies on `add_use(_, id)`
                            // ignoring `id`s which aren't recognized by `def_map`.
                            use_acc.add_use(current_block, id);
                        }
                    }
                }
            }
            if let Some(use_acc) = cfgssa_use_accumulator {
                for (block, inter_block_uses) in use_acc.into_inter_block_uses() {
                    block_details[&block].cfgssa_inter_block_uses = inter_block_uses;
                }
            }

            struct CurrentBlock<'a> {
                region: Region,

                // FIXME(eddyb) figure out a better name and/or organization for this.
                details: &'a BlockDetails,

                // HACK(eddyb) this is probably very inefficient but allows easy
                // access to inter-block-used IDs, in a form directly usable in
                // the current block (i.e. `Region` inputs).
                shadowed_local_id_defs: FxIndexMap<spv::Id, LocalIdDef>,
            }

            let mut current_block = None;
            for (raw_inst_idx, raw_inst) in raw_insts.iter().enumerate() {
                let lookahead_raw_inst =
                    |dist| raw_inst_idx.checked_add(dist).and_then(|i| raw_insts.get(i));

                let IntraFuncInst {
                    attrs,
                    result_type,
                    without_ids: spv::Inst { opcode, ref imms },
                    result_id,
                    ref ids,
                } = *raw_inst;

                let invalid = invalid_factory_for_spv_inst(&raw_inst.without_ids, result_id, ids);

                let is_last_in_block = lookahead_raw_inst(1)
                    .map_or(true, |next_raw_inst| next_raw_inst.without_ids.opcode == wk.OpLabel);

                // HACK(eddyb) this is handled early because it's the only case
                // where a `result_id` isn't a value, and `OpFunctionParameter`
                // wants to be able to use common value result helpers.
                if opcode == wk.OpLabel {
                    if is_last_in_block {
                        return Err(invalid("block lacks terminator instruction"));
                    }

                    // An empty `Region` was defined earlier,
                    // to be able to have an entry in `local_id_defs`.
                    let region = match local_id_defs[&result_id.unwrap()] {
                        LocalIdDef::BlockLabel(region) => region,
                        LocalIdDef::Value { .. } => unreachable!(),
                    };
                    let details = &block_details[&region];
                    assert_eq!(details.label_id, result_id.unwrap());
                    current_block = Some(CurrentBlock {
                        region,
                        details,

                        // HACK(eddyb) reuse `shadowed_local_id_defs` storage.
                        shadowed_local_id_defs: current_block
                            .take()
                            .map(|CurrentBlock { mut shadowed_local_id_defs, .. }| {
                                shadowed_local_id_defs.clear();
                                shadowed_local_id_defs
                            })
                            .unwrap_or_default(),
                    });
                    continue;
                }

                // Helper shared by `OpFunctionParameter` and `OpPhi`.
                let attrs_for_result_leaf = |leaf_type: Type| {
                    if result_type == Some(leaf_type) {
                        attrs
                    } else {
                        // FIXME(eddyb) this may lose semantic `attrs`.
                        AttrSet::default()
                    }
                };

                if opcode == wk.OpFunctionParameter {
                    let result_type = result_type.unwrap();

                    if current_block.is_some() {
                        return Err(invalid(
                            "out of order: `OpFunctionParameter`s should come \
                             before the function's blocks",
                        ));
                    }

                    assert!(imms.is_empty() && ids.is_empty());

                    let param_start = params.len();
                    params.extend(
                        result_type
                            .disaggregated_leaf_types(&cx)
                            .map(|ty| FuncParam { attrs: attrs_for_result_leaf(ty), ty }),
                    );
                    let param_end = params.len();

                    if let Some(func_def_body) = &mut func_def_body {
                        let body_inputs = &mut func_def_body.at_mut_body().def().inputs;
                        let start = u32::try_from(body_inputs.len()).unwrap();
                        body_inputs.extend(
                            params[param_start..param_end]
                                .iter()
                                .map(|&FuncParam { attrs, ty }| RegionInputDecl { attrs, ty }),
                        );
                        let end = u32::try_from(body_inputs.len()).unwrap();

                        local_id_defs.insert(result_id.unwrap(), LocalIdDef::Value {
                            whole_type: result_type,
                            leaves: Either::Left(ValueRange {
                                start: Value::RegionInput {
                                    region: func_def_body.body,
                                    input_idx: start,
                                },
                                count: end - start,
                            }),
                        });
                    }
                    continue;
                }
                let func_def_body = func_def_body.as_deref_mut().unwrap();

                let current_block = current_block.as_mut().ok_or_else(|| {
                    invalid("out of order: not expected before the function's blocks")
                })?;
                let current_block_region_def = &mut func_def_body.regions[current_block.region];

                // HACK(eddyb) the `Region` inputs for inter-block uses
                // have to be inserted just after all the `OpPhi`s' region inputs,
                // or right away (e.g. on `OpLabel`) when there are no `OpPhi`s,
                // so the easiest place to insert them is before handling the
                // first instruction in the block that's not `OpLabel`/`OpPhi`.
                if opcode != wk.OpPhi
                    && current_block.shadowed_local_id_defs.is_empty()
                    && !current_block.details.cfgssa_inter_block_uses.is_empty()
                {
                    assert!(current_block_region_def.children.is_empty());

                    current_block.shadowed_local_id_defs.extend(
                        current_block.details.cfgssa_inter_block_uses.iter().map(
                            |(&used_id, &ty)| {
                                let inputs = &mut current_block_region_def.inputs;
                                let start = u32::try_from(inputs.len()).unwrap();
                                inputs.extend(
                                    ty.disaggregated_leaf_types(&cx).map(|ty| RegionInputDecl {
                                        attrs: AttrSet::default(),
                                        ty,
                                    }),
                                );
                                let end = u32::try_from(inputs.len()).unwrap();

                                (used_id, LocalIdDef::Value {
                                    whole_type: ty,
                                    leaves: Either::Left(ValueRange {
                                        start: Value::RegionInput {
                                            region: current_block.region,
                                            input_idx: start,
                                        },
                                        count: end - start,
                                    }),
                                })
                            },
                        ),
                    );
                }

                // FIXME(eddyb) this returns `LocalIdDef` even for global values.
                let lookup_id = |id| match id_defs.get(&id) {
                    None => {
                        let local_id_def = (current_block.shadowed_local_id_defs.get(&id))
                            .or_else(|| local_id_defs.get(&id))
                            .ok_or_else(|| {
                                // FIXME(eddyb) scan the rest of the function for any
                                // instructions returning this ID, to report an invalid
                                // forward reference (use before def).
                                invalid(&format!("undefined ID %{id}"))
                            })?;
                        // HACK(eddyb) change the type of `leaves` within
                        // `LocalIdDef::Value` to support consts
                        // (see `IdDef::AggregateConst` case just below).
                        Ok(match local_id_def {
                            LocalIdDef::Value { whole_type, leaves } => LocalIdDef::Value {
                                whole_type: *whole_type,
                                leaves: Either::Left(
                                    leaves
                                        .as_ref()
                                        .map_left(|leaves| leaves.iter())
                                        .map_right(|leaves| leaves.iter().copied()),
                                ),
                            },
                            &LocalIdDef::BlockLabel(label) => LocalIdDef::BlockLabel(label),
                        })
                    }
                    Some(&IdDef::Const(ct)) => Ok(LocalIdDef::Value {
                        whole_type: cx[ct].ty,
                        leaves: Either::Right(Either::Left([Value::Const(ct)].into_iter())),
                    }),
                    Some(IdDef::AggregateConst { whole_type, leaves }) => Ok(LocalIdDef::Value {
                        whole_type: *whole_type,
                        leaves: Either::Right(Either::Right(
                            leaves.iter().copied().map(Value::Const),
                        )),
                    }),
                    Some(id_def @ IdDef::Type(_)) => Err(invalid(&format!(
                        "unsupported use of {} as an operand for \
                         an instruction in a function",
                        id_def.descr(&cx),
                    ))),
                    Some(id_def @ IdDef::Func(_)) => Err(invalid(&format!(
                        "unsupported use of {} outside `OpFunctionCall`",
                        id_def.descr(&cx),
                    ))),
                    Some(id_def @ IdDef::SpvDebugString(s)) => {
                        if opcode == wk.OpExtInst {
                            // HACK(eddyb) intern `OpString`s as `Const`s on
                            // the fly, as it's a less likely usage than the
                            // `OpLine` one.
                            let ty = cx.intern(TypeKind::SpvStringLiteralForExtInst);
                            let ct = cx.intern(ConstDef {
                                attrs: AttrSet::default(),
                                ty,
                                kind: ConstKind::SpvStringLiteralForExtInst(*s),
                            });
                            Ok(LocalIdDef::Value {
                                whole_type: ty,
                                leaves: Either::Right(Either::Left([Value::Const(ct)].into_iter())),
                            })
                        } else {
                            Err(invalid(&format!(
                                "unsupported use of {} outside `OpSource`, \
                                 `OpLine`, or `OpExtInst`",
                                id_def.descr(&cx),
                            )))
                        }
                    }
                    Some(id_def @ IdDef::SpvExtInstImport(_)) => Err(invalid(&format!(
                        "unsupported use of {} outside `OpExtInst`",
                        id_def.descr(&cx),
                    ))),
                };

                if is_last_in_block {
                    if opcode.def().category != spec::InstructionCategory::ControlFlow
                        || [wk.OpPhi, wk.OpSelectionMerge, wk.OpLoopMerge].contains(&opcode)
                    {
                        return Err(invalid(
                            "non-control-flow instruction cannot be used \
                             as the terminator instruction of a block",
                        ));
                    }

                    let mut target_inputs = FxIndexMap::default();
                    let mut record_cfg_edge = |target_block| -> io::Result<()> {
                        use indexmap::map::Entry;

                        let target_block_details = &block_details[&target_block];

                        if target_block_details.phi_count == 0
                            && target_block_details.cfgssa_inter_block_uses.is_empty()
                        {
                            return Ok(());
                        }

                        // Only resolve `OpPhi`s exactly once (per target).
                        let target_inputs_entry = match target_inputs.entry(target_block) {
                            Entry::Occupied(_) => return Ok(()),
                            Entry::Vacant(entry) => entry,
                        };

                        let mut target_inputs = SmallVec::new();
                        for target_phi_idx in 0..target_block_details.phi_count {
                            let phi_key = PhiKey {
                                source_block_id: current_block.details.label_id,
                                target_block_id: target_block_details.label_id,
                                target_phi_idx: target_phi_idx.try_into().unwrap(),
                            };
                            let descr_phi_case = || {
                                format!(
                                    "`OpPhi` (#{} in %{})'s case for source block %{}",
                                    phi_key.target_phi_idx,
                                    phi_key.target_block_id,
                                    phi_key.source_block_id,
                                )
                            };

                            let phi_value_ids =
                                phi_to_values.swap_remove(&phi_key).unwrap_or_default();

                            let phi_value_id = match phi_value_ids[..] {
                                [] => {
                                    return Err(invalid(&format!(
                                        "{} is missing",
                                        descr_phi_case()
                                    )));
                                }
                                [id] => id,
                                [..] => {
                                    return Err(invalid(&format!(
                                        "{} is duplicated",
                                        descr_phi_case()
                                    )));
                                }
                            };

                            match lookup_id(phi_value_id)? {
                                LocalIdDef::Value { leaves, .. } => {
                                    target_inputs.extend(leaves);
                                }
                                LocalIdDef::BlockLabel(_) => {
                                    return Err(invalid(&format!(
                                        "unsupported use of block label as the value for {}",
                                        descr_phi_case()
                                    )));
                                }
                            }
                        }

                        for &used_id in target_block_details.cfgssa_inter_block_uses.keys() {
                            match lookup_id(used_id)? {
                                LocalIdDef::Value { leaves, .. } => {
                                    target_inputs.extend(leaves);
                                }
                                LocalIdDef::BlockLabel(_) => unreachable!(),
                            }
                        }

                        target_inputs_entry.insert(target_inputs);

                        Ok(())
                    };

                    // Split the operands into value inputs (e.g. a branch's
                    // condition or an `OpSwitch`'s selector) and target blocks.
                    let mut inputs = SmallVec::new();
                    let mut input_types = SmallVec::<[_; 2]>::new();
                    let mut targets = SmallVec::new();
                    for &id in ids {
                        match lookup_id(id)? {
                            LocalIdDef::Value { whole_type, leaves, .. } => {
                                if !targets.is_empty() {
                                    return Err(invalid(
                                        "out of order: value operand \
                                         after target label ID",
                                    ));
                                }

                                match cx[whole_type].spv_value_lowering() {
                                    spv::ValueLowering::Direct => {}

                                    // Returns are "lossily" disaggregated, just like
                                    // function's signatures and calls to them.
                                    spv::ValueLowering::Disaggregate(_)
                                        if opcode == wk.OpReturnValue => {}

                                    spv::ValueLowering::Disaggregate(_) => {
                                        return Err(invalid(
                                            "unsupported aggregate value operand, \
                                             in non-return terminator instruction",
                                        ));
                                    }
                                }

                                inputs.extend(leaves);
                                input_types.push(whole_type);
                            }
                            LocalIdDef::BlockLabel(target) => {
                                record_cfg_edge(target)?;
                                targets.push(target);
                            }
                        }
                    }

                    // FIXME(eddyb) move some of this to `spv::canonical`.
                    let kind = if opcode == wk.OpUnreachable {
                        assert!(targets.is_empty() && inputs.is_empty());
                        cfg::ControlInstKind::Unreachable
                    } else if [wk.OpReturn, wk.OpReturnValue].contains(&opcode) {
                        assert!(targets.is_empty());
                        cfg::ControlInstKind::Return
                    } else if targets.is_empty() {
                        cfg::ControlInstKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(
                            raw_inst.without_ids.clone(),
                        ))
                    } else if opcode == wk.OpBranch {
                        assert_eq!((targets.len(), inputs.len()), (1, 0));
                        cfg::ControlInstKind::Branch
                    } else if opcode == wk.OpBranchConditional {
                        assert_eq!((targets.len(), inputs.len()), (2, 1));
                        cfg::ControlInstKind::SelectBranch(SelectionKind::BoolCond)
                    } else if opcode == wk.OpSwitch {
                        assert_eq!(inputs.len(), 1);

                        // HACK(eddyb) `spv::read` has to "redundantly" validate
                        // that such a type is `OpTypeInt`/`OpTypeFloat`, but
                        // there is still a limitation when it comes to `scalar::Const`.
                        // FIXME(eddyb) don't hardcode the 128-bit limitation,
                        // but query `scalar::Const` somehow instead.
                        let scrutinee_type = input_types[0];
                        let scrutinee_type = scrutinee_type
                            .as_scalar(&cx)
                            .filter(|ty| {
                                matches!(ty, scalar::Type::UInt(_) | scalar::Type::SInt(_))
                                    && ty.bit_width() <= 128
                            })
                            .ok_or_else(|| {
                                invalid(
                                    &print::Plan::for_root(
                                        &cx,
                                        &Diag::err([
                                            "unsupported `OpSwitch` scrutinee type `".into(),
                                            scrutinee_type.into(),
                                            "`".into(),
                                        ])
                                        .message,
                                    )
                                    .pretty_print()
                                    .to_string(),
                                )
                            })?;

                        // FIXME(eddyb) move some of this to `spv::canonical`.
                        let imm_words_per_case =
                            usize::try_from(scrutinee_type.bit_width().div_ceil(32)).unwrap();

                        // NOTE(eddyb) these sanity-checks are redundant with `spv::read`.
                        assert_eq!(imms.len() % imm_words_per_case, 0);
                        assert_eq!(targets.len(), 1 + imms.len() / imm_words_per_case);

                        let case_consts = imms
                            .chunks(imm_words_per_case)
                            .map(|case_imms| {
                                scalar::Const::try_decode_from_spv_imms(scrutinee_type, case_imms)
                                    .ok_or_else(|| {
                                        invalid(&format!(
                                            "invalid {}-bit `OpSwitch` case constant",
                                            scrutinee_type.bit_width()
                                        ))
                                    })
                            })
                            .collect::<Result<_, _>>()?;

                        // HACK(eddyb) move the default case from first to last.
                        let default_target = targets.remove(0);
                        targets.push(default_target);

                        cfg::ControlInstKind::SelectBranch(SelectionKind::Switch { case_consts })
                    } else {
                        return Err(invalid("unsupported control-flow instruction"));
                    };

                    func_def_body
                        .unstructured_cfg
                        .as_mut()
                        .unwrap()
                        .control_inst_on_exit_from
                        .insert(current_block.region, cfg::ControlInst {
                            attrs,
                            kind,
                            inputs,
                            targets,
                            target_inputs,
                        });
                    continue;
                }

                if opcode == wk.OpPhi {
                    let result_type = result_type.unwrap();

                    if !current_block_region_def.children.is_empty() {
                        return Err(invalid(
                            "out of order: `OpPhi`s should come before \
                             the rest of the block's instructions",
                        ));
                    }

                    let inputs = &mut current_block_region_def.inputs;
                    let start = u32::try_from(inputs.len()).unwrap();
                    inputs.extend(
                        result_type
                            .disaggregated_leaf_types(&cx)
                            .map(|ty| RegionInputDecl { attrs: attrs_for_result_leaf(ty), ty }),
                    );
                    let end = u32::try_from(inputs.len()).unwrap();

                    local_id_defs.insert(result_id.unwrap(), LocalIdDef::Value {
                        whole_type: result_type,
                        leaves: Either::Left(ValueRange {
                            start: Value::RegionInput {
                                region: current_block.region,
                                input_idx: start,
                            },
                            count: end - start,
                        }),
                    });
                    continue;
                }

                if [wk.OpSelectionMerge, wk.OpLoopMerge].contains(&opcode) {
                    let is_second_to_last_in_block = lookahead_raw_inst(2)
                        .map_or(true, |next_raw_inst| {
                            next_raw_inst.without_ids.opcode == wk.OpLabel
                        });

                    if !is_second_to_last_in_block {
                        return Err(invalid(
                            "out of order: a merge instruction should be the last \
                             instruction before the block's terminator",
                        ));
                    }

                    // HACK(eddyb) we want to at least record `OpLoopMerge`s'
                    // impact on the shape of a loop, for restructurization.
                    if opcode == wk.OpLoopMerge {
                        assert_eq!(ids.len(), 2);
                        let loop_merge_target = match lookup_id(ids[0])? {
                            LocalIdDef::Value { .. } => return Err(invalid("expected label ID")),
                            LocalIdDef::BlockLabel(target) => target,
                        };

                        func_def_body
                            .unstructured_cfg
                            .as_mut()
                            .unwrap()
                            .loop_merge_to_loop_header
                            .insert(loop_merge_target, current_block.region);
                    }

                    // HACK(eddyb) merges are mostly ignored - this may be lossy,
                    // especially wrt the `SelectionControl` and `LoopControl`
                    // operands, but it's not obvious how they should map to
                    // some "structured regions" replacement for the CFG.
                    continue;
                }

                // All control-flow instructions have been handled above.
                // Only `DataInst`s get generated below here.

                let mut append_node = |node_def: NodeDef| {
                    let node = func_def_body.nodes.define(&cx, node_def.into());
                    current_block_region_def.children.insert_last(node, &mut func_def_body.nodes);
                    node
                };

                let lookup_value_id = |id| match lookup_id(id)? {
                    LocalIdDef::Value { whole_type, leaves } => Ok((whole_type, leaves)),
                    LocalIdDef::BlockLabel(_) => Err(invalid(
                        "unsupported use of block label as a value, \
                         in non-terminator instruction",
                    )),
                };

                // Special-case instructions which deal with aggregates as
                // "containers" for their leaves, and so have an effect which
                // can be interpreted eagerly on the disaggregated form.
                // FIXME(eddyb) this may lose semantic `attrs`
                let eagerly_lowered_result = if opcode == wk.OpCompositeConstruct {
                    let result_type = result_type.unwrap();

                    match cx[result_type].spv_value_lowering() {
                        spv::ValueLowering::Direct => None,
                        spv::ValueLowering::Disaggregate(_) => {
                            let mut all_leaves =
                                SmallVec::with_capacity(cx[result_type].disaggregated_leaf_count());
                            for &id in ids {
                                let (_, leaves) = lookup_value_id(id)?;
                                all_leaves.extend(leaves);
                            }
                            if all_leaves.len() == cx[result_type].disaggregated_leaf_count() {
                                Some(LocalIdDef::Value {
                                    whole_type: result_type,
                                    leaves: Either::Right(all_leaves),
                                })
                            } else {
                                None
                            }
                        }
                    }
                } else if [wk.OpCompositeExtract, wk.OpCompositeInsert].contains(&opcode) {
                    let result_type = result_type.unwrap();

                    let (&composite_id, ids_without_last) = ids.split_last().unwrap();
                    let (composite_type, leaves) = lookup_value_id(composite_id)?;

                    // HACK(eddyb) `replace_component` and `rebuild_composite`
                    // are always both `None` or both `Some`, but splitting the
                    // two aspects of `OpCompositeInsert` makes it easier later.
                    let (component_type, replace_component, rebuild_composite);
                    match ids_without_last[..] {
                        [] => {
                            component_type = result_type;
                            replace_component = None;
                            rebuild_composite = None;
                        }
                        [replacement_component_id] => {
                            let (replacement_component_type, replacement_component_leaves) =
                                lookup_value_id(replacement_component_id)?;

                            component_type = replacement_component_type;
                            replace_component = Some(replacement_component_leaves);
                            rebuild_composite = Some(result_type);
                        }
                        _ => unreachable!(),
                    }

                    // HACK(eddyb) this is a `try {...}`-like use of a closure.
                    (|| {
                        if let Some(expected_type) = rebuild_composite {
                            if composite_type != expected_type {
                                return None;
                            }
                        }

                        let mut imms = imms.iter();
                        let (leaf_type, leaf_range) = match cx[composite_type].spv_value_lowering()
                        {
                            spv::ValueLowering::Direct => return None,
                            spv::ValueLowering::Disaggregate(_) => composite_type
                                .aggregate_component_path_type_and_leaf_range(
                                    &cx,
                                    &mut imms.by_ref().map(|&imm| match imm {
                                        spv::Imm::Short(_, i) => i,
                                        _ => unreachable!(),
                                    }),
                                )?,
                        };
                        let non_aggregate_indexing_imms = imms.as_slice();

                        if non_aggregate_indexing_imms.is_empty() && leaf_type != component_type {
                            return None;
                        }

                        let mut component_leaves =
                            leaves.clone().skip(leaf_range.start).take(leaf_range.len());

                        // If there's any leftover indices they must be indexing
                        // into a vector/matrix, which requires separate handling.
                        let component_leaves = if !non_aggregate_indexing_imms.is_empty() {
                            assert_eq!(component_leaves.len(), 1);
                            let non_aggregate_composite = component_leaves.next().unwrap();

                            let leaf_spv_inst = spv::Inst {
                                opcode,
                                imms: non_aggregate_indexing_imms.iter().copied().collect(),
                            };
                            let leaf_output_types = [match rebuild_composite {
                                Some(_) => leaf_type,
                                None => component_type,
                            }];
                            let leaf_inst = append_node(DataInstDef {
                                attrs,
                                kind: leaf_spv_inst
                                    .as_canonical_data_inst_kind(&cx, leaf_output_types.into_iter())
                                    .unwrap_or(DataInstKind::SpvInst(
                                        leaf_spv_inst,
                                        spv::InstLowering::default(),
                                    )),
                                inputs: replace_component
                                    .map(|mut replacement_leaves| {
                                        assert_eq!(replacement_leaves.len(), 1);
                                        replacement_leaves.next().unwrap()
                                    })
                                    .into_iter()
                                    .chain([non_aggregate_composite])
                                    .collect(),
                                child_regions: [].into_iter().collect(),
                                outputs: leaf_output_types
                                    .into_iter()
                                    .map(|ty| {
                                        // FIXME(eddyb) this may lose semantic `attrs`.
                                        NodeOutputDecl { attrs: AttrSet::default(), ty }
                                    })
                                    .collect(),
                            });
                            Either::Left(
                                [Value::NodeOutput { node: leaf_inst, output_idx: 0 }].into_iter(),
                            )
                        } else {
                            Either::Right(
                                replace_component
                                    .map_or(Either::Left(component_leaves), Either::Right),
                            )
                        };

                        assert_eq!(
                            component_leaves.len(),
                            cx[component_type].disaggregated_leaf_count()
                        );

                        let leaves = match rebuild_composite {
                            Some(_) => leaves
                                .clone()
                                .take(leaf_range.start)
                                .chain(component_leaves)
                                .chain(leaves.skip(leaf_range.end))
                                .collect(),
                            None => component_leaves.collect(),
                        };

                        Some(LocalIdDef::Value {
                            whole_type: result_type,
                            // FIXME(eddyb) avoid allocating somehow, like
                            // try "recompressing" into a `ValueRange`, or
                            // preserving that form throughout above?
                            leaves: Either::Right(leaves),
                        })
                    })()
                } else {
                    None
                };
                if let Some(def) = eagerly_lowered_result {
                    local_id_defs.insert(result_id.unwrap(), def);
                    continue;
                }

                let mut ids = &ids[..];
                let mut kind = if opcode == wk.OpFunctionCall {
                    assert!(imms.is_empty());
                    let callee_id = ids[0];
                    let maybe_callee = id_defs
                        .get(&callee_id)
                        .map(|id_def| match *id_def {
                            IdDef::Func(func) => Ok(func),
                            _ => Err(id_def.descr(&cx)),
                        })
                        .transpose()
                        .map_err(|descr| {
                            invalid(&format!(
                                "unsupported use of {descr} as the `OpFunctionCall` callee"
                            ))
                        })?;

                    match maybe_callee {
                        Some(callee) => {
                            ids = &ids[1..];
                            NodeKind::FuncCall { callee }
                        }

                        // HACK(eddyb) this should be an error, but it shows
                        // up in Rust-GPU output (likely a zombie?).
                        None => DataInstKind::SpvInst(
                            raw_inst.without_ids.clone(),
                            spv::InstLowering::default(),
                        ),
                    }
                } else if opcode == wk.OpExtInst {
                    let ext_set_id = ids[0];
                    ids = &ids[1..];

                    let inst = match imms[..] {
                        [spv::Imm::Short(kind, inst)] => {
                            assert_eq!(kind, wk.LiteralExtInstInteger);
                            inst
                        }
                        _ => unreachable!(),
                    };

                    let ext_set = match id_defs.get(&ext_set_id) {
                        Some(&IdDef::SpvExtInstImport(name)) => Ok(name),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("unknown ID %{ext_set_id}")),
                    }
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the `OpExtInst` \
                             extended instruction set ID"
                        ))
                    })?;

                    DataInstKind::SpvExtInst {
                        ext_set,
                        inst,
                        lowering: spv::InstLowering::default(),
                    }
                } else {
                    DataInstKind::SpvInst(
                        raw_inst.without_ids.clone(),
                        spv::InstLowering::default(),
                    )
                };

                // HACK(eddyb) only factored out due to `kind`'s mutable borrow.
                let call_ret_type = match kind {
                    NodeKind::FuncCall { .. } => Some(result_type.unwrap()),
                    _ => None,
                };

                let mut spv_inst_lowering = match &mut kind {
                    DataInstKind::SpvInst(_, lowering)
                    | DataInstKind::SpvExtInst { lowering, .. } => Some(lowering),

                    // NOTE(eddyb) function signatures and calls keep their
                    // disaggregation even when lifting back to SPIR-V, so
                    // no `spv::InstLowering` is tracked for them.
                    NodeKind::FuncCall { .. } => None,

                    NodeKind::Select(_)
                    | NodeKind::Loop { .. }
                    | NodeKind::ExitInvocation(_)
                    | DataInstKind::Scalar(_)
                    | DataInstKind::Vector(_)
                    | DataInstKind::QPtr(_) => {
                        unreachable!()
                    }
                };

                let outputs: SmallVec<[_; 2]> = result_id
                    .and_then(|_| {
                        let result_type = result_type.unwrap();
                        if let Some(spv_inst_lowering) = &mut spv_inst_lowering {
                            spv_inst_lowering.disaggregated_output =
                                match cx[result_type].spv_value_lowering() {
                                    spv::ValueLowering::Direct => None,
                                    spv::ValueLowering::Disaggregate(_) => Some(result_type),
                                };
                        }

                        // HACK(eddyb) `OpTypeVoid` special-cased for calls
                        // as if it were an aggregate with `0` leaves.
                        let ret_void = call_ret_type.is_some_and(|ty| match &cx[ty].kind {
                            TypeKind::SpvInst { spv_inst: ret_type_spv_inst, .. } => {
                                ret_type_spv_inst.opcode == wk.OpTypeVoid
                            }
                            _ => false,
                        });
                        (!ret_void).then_some(result_type)
                    })
                    .into_iter()
                    .flat_map(|result_type| result_type.disaggregated_leaf_types(&cx))
                    .map(|ty| {
                        // FIXME(eddyb) this may lose semantic `attrs`.
                        NodeOutputDecl { attrs: AttrSet::default(), ty }
                    })
                    .collect();
                let output_count = outputs.len();

                let mut inputs = SmallVec::with_capacity(ids.len());
                for &id in ids {
                    let (whole_input_type, leaves) = lookup_value_id(id)?;

                    let start = u32::try_from(inputs.len()).unwrap();
                    inputs.extend(leaves);
                    let end = u32::try_from(inputs.len()).unwrap();

                    if let spv::ValueLowering::Disaggregate(_) =
                        cx[whole_input_type].spv_value_lowering()
                    {
                        if let Some(lowering) = &mut spv_inst_lowering {
                            lowering.disaggregated_inputs.push((start..end, whole_input_type));
                        }
                    }
                }

                if let DataInstKind::SpvInst(spv_inst, lowering) = &kind {
                    if lowering.disaggregated_inputs.is_empty() {
                        if let Some(canonical_kind) =
                            spv_inst.as_canonical_data_inst_kind(&cx, outputs.iter().map(|o| o.ty))
                        {
                            // FIXME(eddyb) sanity-check the number/types of inputs.
                            kind = canonical_kind;
                        }
                    }
                }

                let node = append_node(NodeDef {
                    attrs,
                    kind,
                    inputs,
                    child_regions: [].into_iter().collect(),
                    outputs,
                });

                if let Some(result_id) = result_id {
                    local_id_defs.insert(result_id, LocalIdDef::Value {
                        whole_type: result_type.unwrap(),
                        leaves: Either::Left(ValueRange {
                            start: Value::NodeOutput { node, output_idx: 0 },
                            count: u32::try_from(output_count).unwrap(),
                        }),
                    });
                }
            }

            // FIXME(eddyb) all functions should have the appropriate number of
            // `OpFunctionParameter`, even imports.
            if !params.is_empty() {
                if func_decl.params.len() != params.len() {
                    // FIXME(remove) embed IDs in errors by moving them to the
                    // `let invalid = |...| ...;` closure that wraps insts.
                    return Err(invalid(&format!(
                        "in %{}, param count differs between `OpTypeFunction` ({}) \
                         and `OpFunctionParameter`s ({})",
                        func_id,
                        func_decl.params.len(),
                        params.len(),
                    )));
                }

                for (i, (func_decl_param, param)) in
                    func_decl.params.iter_mut().zip(params).enumerate()
                {
                    func_decl_param.attrs = param.attrs;
                    if func_decl_param.ty != param.ty {
                        // FIXME(remove) embed IDs in errors by moving them to the
                        // `let invalid = |...| ...;` closure that wraps insts.
                        return Err(invalid(
                            &print::Plan::for_root(
                                &cx,
                                &Diag::err([
                                    format!("in %{}, ", func_id).into(),
                                    format!("param #{i}'s type differs between `OpTypeFunction` (")
                                        .into(),
                                    func_decl_param.ty.into(),
                                    ") and `OpFunctionParameter` (".into(),
                                    param.ty.into(),
                                    ")".into(),
                                ])
                                .message,
                            )
                            .pretty_print()
                            .to_string(),
                        ));
                    }
                }
            }

            if !phi_to_values.is_empty() {
                let mut edges = phi_to_values
                    .keys()
                    .map(|key| format!("%{} -> %{}", key.source_block_id, key.target_block_id))
                    .collect::<Vec<_>>();
                edges.dedup();
                // FIXME(remove) embed IDs in errors by moving them to the
                // `let invalid = |...| ...;` closure that wraps insts.
                return Err(invalid(&format!(
                    "in %{}, `OpPhi`s refer to non-existent edges: {}",
                    func_id,
                    edges.join(", ")
                )));
            }

            // Sanity-check the entry block.
            if let Some(func_def_body) = func_def_body {
                if block_details[&func_def_body.body].phi_count > 0 {
                    // FIXME(remove) embed IDs in errors by moving them to the
                    // `let invalid = |...| ...;` closure that wraps insts.
                    return Err(invalid(&format!(
                        "in %{func_id}, the entry block contains `OpPhi`s"
                    )));
                }
            }
        }

        assert!(module.exports.is_empty());
        module.exports = pending_exports
            .into_iter()
            .map(|export| match export {
                Export::Linkage { name, target_id } => {
                    let exportee = match id_defs.get(&target_id) {
                        Some(id_def @ &IdDef::Const(ct)) => match cx[ct].kind {
                            ConstKind::PtrToGlobalVar(gv) => Ok(Exportee::GlobalVar(gv)),
                            _ => Err(id_def.descr(&cx)),
                        },
                        Some(&IdDef::Func(func)) => Ok(Exportee::Func(func)),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("unknown ID %{target_id}")),
                    }
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the `LinkageAttributes` target"
                        ))
                    })?;

                    Ok((ExportKey::LinkName(name), exportee))
                }

                Export::EntryPoint {
                    func_id,
                    imms,
                    interface_ids,
                } => {
                    let func = match id_defs.get(&func_id) {
                        Some(&IdDef::Func(func)) => Ok(func),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("unknown ID %{func_id}")),
                    }
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the `OpEntryPoint` target"
                        ))
                    })?;
                    let interface_global_vars = interface_ids
                        .into_iter()
                        .map(|id| match id_defs.get(&id) {
                            Some(id_def @ &IdDef::Const(ct)) => match cx[ct].kind {
                                ConstKind::PtrToGlobalVar(gv) => Ok(gv),
                                _ => Err(id_def.descr(&cx)),
                            },
                            Some(id_def) => Err(id_def.descr(&cx)),
                            None => Err(format!("unknown ID %{id}")),
                        })
                        .map(|result| {
                            result.map_err(|descr| {
                                invalid(&format!(
                                    "unsupported use of {descr} as an `OpEntryPoint` interface variable"
                                ))
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    Ok((
                        ExportKey::SpvEntryPoint {
                            imms,
                            interface_global_vars,
                        },
                        Exportee::Func(func),
                    ))
                }
            })
            .collect::<io::Result<_>>()?;

        Ok(module)
    }
}
