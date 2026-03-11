//! Host-side snapshot optimization for Python (shared-modules) components.
//!
//! # Overview
//!
//! Python components built with shared-everything dynamic linking pay a
//! significant initialization cost: each time the component is instantiated the
//! CPython interpreter must start up, import the standard library, and execute
//! top-level module code. This module eliminates that repeated cost by taking a
//! *snapshot* of the Wasm linear memory and mutable globals immediately after
//! initialization, then baking those values directly into the component binary.
//! Subsequent instantiations start from the post-init state with no interpreter
//! startup overhead.
//!
//! # Snapshot Pipeline
//!
//! The end-to-end process performed by [`snapshot_component()`] is:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                          Snapshot Pipeline                               │
//! └──────────────────────────────────────────────────────────────────────────┘
//!
//!   Original Component
//!          │
//!          ▼
//!   1. instrument()
//!      Rewrite the component to expose memory contents and mutable globals
//!      through synthesized getter functions (e.g., `snapshot-memory-N`,
//!      `snapshot-global-N`). A `prepare-snapshot` export triggers the
//!      original initialization code and then makes the state readable.
//!          │
//!          ▼
//!   2. Instantiate + call prepare-snapshot
//!      Compile and instantiate the instrumented component with a full
//!      linker (including shared modules). Calling `prepare-snapshot`
//!      runs the real initialization (CPython startup, user code, etc.).
//!          │
//!          ▼
//!   3. Read back snapshot data
//!      Invoke the synthesized getter functions to capture every page of
//!      linear memory and the value of every mutable global.
//!          │
//!          ▼
//!   4. apply()
//!      Produce a new component binary that:
//!        - Replaces data segments with the captured memory snapshot
//!        - Replaces global initializers with the captured values
//!        - Strips the original start/init functions so the heavy
//!          initialization is never executed again
//!
//!   Result: a component that instantiates into the post-init state directly.
//! ```
//!
//! # Stripped Shared Modules
//!
//! A snapshotted component already contains the fully initialized memory image,
//! including the portions that would normally be populated by the data segments
//! and start functions of shared modules (e.g., libc, CPython). If those shared
//! modules were instantiated normally, their data segments and start sections
//! would overwrite the snapshot. [`strip_module_data()`] produces *stripped*
//! variants of shared modules — identical except that data segments, the data
//! count section, and start sections are removed — so that instantiation of a
//! snapshotted component preserves the captured state.
//!
//! # Public API
//!
//! This module exposes two `pub(super)` entry points consumed by the parent
//! runtime module:
//!
//! - [`snapshot_component()`] — end-to-end orchestrator: instruments, runs,
//!   captures, and applies the snapshot, returning the new component bytes.
//! - [`strip_module_data()`] — strips data/start sections from a shared module
//!   so it can be used when instantiating snapshotted components.
//!
//! # Source Attribution
//!
//! The `instrument()` and `apply()` functions are adapted from the
//! `component_init_transform` crate:
//!   <https://github.com/dicej/component-init>
//!   rev 1de5906ca8c5f7093eaa9f6565f1dde5fc9608d3
//!   file: transform/src/lib.rs
//!
//! Key adaptation: both functions count `ComponentTypeRef::Module` imports in
//! `module_count`, correctly handling shared-everything dynamically linked
//! components that import core modules (e.g., libc, CPython).

use {
    anyhow::{Context, Result, anyhow, bail},
    async_trait::async_trait,
    std::{
        collections::{HashMap, hash_map::Entry},
        iter,
        ops::Range,
    },
    wasm_encoder::{
        Alias, CanonicalFunctionSection, CanonicalOption, CodeSection,
        Component as EncoderComponent, ComponentAliasSection, ComponentExportKind,
        ComponentExportSection, ComponentTypeSection, ComponentValType, ConstExpr,
        DataCountSection, DataSection, ExportKind, ExportSection, Function, FunctionSection,
        GlobalType, ImportSection, InstanceSection, Instruction as Ins, MemArg, MemorySection,
        Module as EncoderModule, ModuleArg, ModuleSection, NestedComponentSection,
        PrimitiveValType, RawSection, TypeSection, ValType,
        reencode::{Reencode, RoundtripReencoder as Encode},
    },
    wasmparser::{
        CanonicalFunction, ComponentAlias, ComponentExternalKind, ComponentTypeRef, ExternalKind,
        Imports, Instance, Operator, Parser, Payload, TypeRef, Validator, WasmFeatures,
    },
    wasmtime::{
        Engine, Store,
        component::{Component, ComponentNamedList, Instance as WasmtimeInstance, Lift, Linker},
    },
};

const PAGE_SIZE_BYTES: i32 = 64 * 1024;
const MAX_CONSECUTIVE_ZEROS: usize = 64;

// ---------------------------------------------------------------------------
// Invoker trait
// ---------------------------------------------------------------------------

#[async_trait]
trait Invoker: Send {
    async fn call_s32(&mut self, function: &str) -> Result<i32>;
    async fn call_s64(&mut self, function: &str) -> Result<i64>;
    async fn call_f32(&mut self, function: &str) -> Result<f32>;
    async fn call_f64(&mut self, function: &str) -> Result<f64>;
    async fn call_list_u8(&mut self, function: &str) -> Result<Vec<u8>>;
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct MemoryInfo {
    module_index: u32,
    export_name: String,
    ty: wasmparser::MemoryType,
}

type GlobalMap<T> = HashMap<u32, HashMap<u32, T>>;

#[derive(Debug)]
enum GlobalExport {
    Existing {
        module_index: u32,
        global_index: u32,
        export_name: String,
    },
    Synthesized {
        module_index: u32,
        global_index: u32,
    },
}

impl GlobalExport {
    fn module_export(&self) -> String {
        match self {
            Self::Existing { export_name, .. } => export_name.clone(),
            Self::Synthesized { global_index, .. } => format!("component-init:{global_index}"),
        }
    }
    fn component_export(&self) -> String {
        match self {
            Self::Existing {
                module_index,
                global_index,
                ..
            }
            | Self::Synthesized {
                module_index,
                global_index,
            } => format!("component-init-get-module{module_index}-global{global_index}"),
        }
    }
}

#[derive(Default)]
struct Instrumentation {
    memory: Option<MemoryInfo>,
    globals: GlobalMap<(GlobalExport, wasmparser::ValType)>,
}

impl Instrumentation {
    fn register_memory(
        &mut self,
        module_index: u32,
        name: impl AsRef<str>,
        ty: wasmparser::MemoryType,
    ) -> Result<()> {
        if self.memory.is_some() {
            bail!("only one memory allowed per component");
        }
        self.memory = Some(MemoryInfo {
            module_index,
            export_name: name.as_ref().to_string(),
            ty,
        });
        Ok(())
    }

    fn register_global(&mut self, module_index: u32, global_index: u32, ty: wasmparser::ValType) {
        self.globals.entry(module_index).or_default().insert(
            global_index,
            (
                GlobalExport::Synthesized {
                    module_index,
                    global_index,
                },
                ty,
            ),
        );
    }

    fn register_global_export(
        &mut self,
        module_index: u32,
        global_index: u32,
        export_name: impl AsRef<str>,
    ) {
        if let Some((name, _)) = self
            .globals
            .get_mut(&module_index)
            .and_then(|map| map.get_mut(&global_index))
        {
            let export_name = export_name.as_ref().to_string();
            *name = GlobalExport::Existing {
                module_index,
                global_index,
                export_name,
            };
        }
    }

    fn amend_module_exports(&self, module_index: u32, exports: &mut ExportSection) {
        if let Some(g_map) = self.globals.get(&module_index) {
            for (export, _ty) in g_map.values() {
                if let GlobalExport::Synthesized { global_index, .. } = export {
                    exports.export(&export.module_export(), ExportKind::Global, *global_index);
                }
            }
        }
    }

    async fn measure(&self, invoker: &mut Box<dyn Invoker>) -> Result<Measurement> {
        let mut globals = HashMap::new();
        for (module_index, globals_to_export) in &self.globals {
            let mut my_global_values = HashMap::new();
            for (global_index, (global_export, ty)) in globals_to_export {
                my_global_values.insert(
                    *global_index,
                    match ty {
                        wasmparser::ValType::I32 => ConstExpr::i32_const(
                            invoker
                                .call_s32(&global_export.component_export())
                                .await
                                .with_context(|| {
                                    format!("retrieving global value {global_export:?}")
                                })?,
                        ),
                        wasmparser::ValType::I64 => ConstExpr::i64_const(
                            invoker
                                .call_s64(&global_export.component_export())
                                .await
                                .with_context(|| {
                                    format!("retrieving global value {global_export:?}")
                                })?,
                        ),
                        wasmparser::ValType::F32 => ConstExpr::f32_const(
                            invoker
                                .call_f32(&global_export.component_export())
                                .await
                                .with_context(|| {
                                    format!("retrieving global value {global_export:?}")
                                })?
                                .into(),
                        ),
                        wasmparser::ValType::F64 => ConstExpr::f64_const(
                            invoker
                                .call_f64(&global_export.component_export())
                                .await
                                .with_context(|| {
                                    format!("retrieving global value {global_export:?}")
                                })?
                                .into(),
                        ),
                        wasmparser::ValType::V128 => bail!("V128 not yet supported"),
                        wasmparser::ValType::Ref(_) => bail!("reference types not supported"),
                    },
                );
            }
            globals.insert(*module_index, my_global_values);
        }

        let memory = if let Some(info) = &self.memory {
            let name = "component-init-get-memory";
            Some((
                info.module_index,
                invoker
                    .call_list_u8(name)
                    .await
                    .with_context(|| format!("retrieving memory with {name}"))?,
            ))
        } else {
            None
        };
        Ok(Measurement { memory, globals })
    }
}

#[allow(dead_code)]
struct Measurement {
    memory: Option<(u32, Vec<u8>)>,
    globals: GlobalMap<wasm_encoder::ConstExpr>,
}

impl Measurement {
    fn data_section(&self, module_index: u32) -> (Option<DataSection>, u32) {
        if let Some((m_ix, value)) = &self.memory
            && *m_ix == module_index
        {
            let mut data = DataSection::new();
            let mut data_segment_count = 0;
            for (start, len) in Segments::new(value) {
                data_segment_count += 1;
                data.active(
                    0,
                    &ConstExpr::i32_const(start.try_into().unwrap()),
                    value[start..][..len].iter().copied(),
                );
            }
            (Some(data), data_segment_count)
        } else {
            (None, 0)
        }
    }

    fn memory_initial(&self, module_index: u32) -> Option<u64> {
        if let Some((m_ix, value)) = &self.memory
            && *m_ix == module_index
        {
            Some(
                u64::try_from((value.len() / usize::try_from(PAGE_SIZE_BYTES).unwrap()) + 1)
                    .unwrap(),
            )
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn global_init(&self, module_index: u32, global_index: u32) -> Option<wasm_encoder::ConstExpr> {
        self.globals
            .get(&module_index)
            .and_then(|m| m.get(&global_index).cloned())
    }
}

// ---------------------------------------------------------------------------
// Segments iterator
// ---------------------------------------------------------------------------

struct Segments<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Segments<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
}

impl<'a> Iterator for Segments<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let mut zero_count = 0;
        let mut start = 0;
        let mut length = 0;
        for (index, value) in self.bytes[self.offset..].iter().enumerate() {
            if *value == 0 {
                zero_count += 1;
            } else {
                if zero_count > MAX_CONSECUTIVE_ZEROS {
                    if length > 0 {
                        start += self.offset;
                        self.offset += index;
                        return Some((start, length));
                    } else {
                        start = index;
                        length = 1;
                    }
                } else {
                    length += zero_count + 1;
                }
                zero_count = 0;
            }
        }
        if length > 0 {
            start += self.offset;
            self.offset = self.bytes.len();
            Some((start, length))
        } else {
            self.offset = self.bytes.len();
            None
        }
    }
}

// ---------------------------------------------------------------------------
// instrument()
// ---------------------------------------------------------------------------

fn instrument(component_bytes: &[u8]) -> Result<(Vec<u8>, Instrumentation)> {
    let mut module_count = 0;
    let mut instance_count = 0;
    let mut core_function_count = 0;
    let mut function_count = 0;
    let mut type_count = 0;
    let mut instrumentation = Instrumentation::default();
    let mut instantiations = HashMap::new();
    let mut instrumented_component = EncoderComponent::new();
    let mut parser = Parser::new(0).parse_all(component_bytes);
    #[allow(clippy::while_let_on_iterator)]
    while let Some(payload) = parser.next() {
        let payload = payload?;
        let section = payload.as_section();
        match payload {
            Payload::ComponentSection {
                unchecked_range, ..
            } => {
                let mut subcomponent = EncoderComponent::new();
                while let Some(payload) = parser.next() {
                    let payload = payload?;
                    let section = payload.as_section();
                    let my_range = section.as_ref().map(|(_, range)| range.clone());
                    copy_component_section(section, component_bytes, &mut subcomponent);
                    if let Some(my_range) = my_range
                        && my_range.end >= unchecked_range.end
                    {
                        break;
                    }
                }
                instrumented_component.section(&NestedComponentSection(&subcomponent));
            }

            Payload::ModuleSection {
                unchecked_range, ..
            } => {
                let module_index = get_and_increment(&mut module_count);
                let mut global_types = Vec::new();
                let mut instrumented_module = EncoderModule::new();
                let mut global_count = 0;
                while let Some(payload) = parser.next() {
                    let payload = payload?;
                    let section = payload.as_section();
                    let my_range = section.as_ref().map(|(_, range)| range.clone());
                    match payload {
                        Payload::ImportSection(reader) => {
                            for import in reader {
                                match import? {
                                    Imports::Single(_, import) => {
                                        if let TypeRef::Global(_) = import.ty {
                                            global_count += 1;
                                        }
                                    }
                                    Imports::Compact1 { .. } | Imports::Compact2 { .. } => todo!(),
                                }
                            }
                            copy_module_section(section, component_bytes, &mut instrumented_module);
                        }

                        Payload::MemorySection(reader) => {
                            for memory in reader {
                                instrumentation.register_memory(module_index, "memory", memory?)?;
                            }
                            copy_module_section(section, component_bytes, &mut instrumented_module);
                        }

                        Payload::GlobalSection(reader) => {
                            for global in reader {
                                let global = global?;
                                let ty = global.ty;
                                global_types.push(ty);
                                let global_index = get_and_increment(&mut global_count);
                                if global.ty.mutable {
                                    instrumentation.register_global(
                                        module_index,
                                        global_index,
                                        ty.content_type,
                                    )
                                }
                            }
                            copy_module_section(section, component_bytes, &mut instrumented_module);
                        }

                        Payload::ExportSection(reader) => {
                            let mut exports = ExportSection::new();
                            for export in reader {
                                let export = export?;
                                if let ExternalKind::Global = export.kind {
                                    instrumentation.register_global_export(
                                        module_index,
                                        export.index,
                                        export.name,
                                    )
                                }
                                exports.export(
                                    export.name,
                                    Encode.export_kind(export.kind)?,
                                    export.index,
                                );
                            }
                            instrumentation.amend_module_exports(module_index, &mut exports);
                            instrumented_module.section(&exports);
                        }

                        Payload::CodeSectionEntry(body) => {
                            for operator in body.get_operators_reader()? {
                                match operator? {
                                    Operator::TableCopy { .. }
                                    | Operator::TableFill { .. }
                                    | Operator::TableGrow { .. }
                                    | Operator::TableInit { .. }
                                    | Operator::TableSet { .. } => {
                                        bail!("table operations not allowed");
                                    }
                                    _ => (),
                                }
                            }
                            copy_module_section(section, component_bytes, &mut instrumented_module);
                        }

                        _ => {
                            copy_module_section(section, component_bytes, &mut instrumented_module)
                        }
                    }

                    if let Some(my_range) = my_range
                        && my_range.end >= unchecked_range.end
                    {
                        break;
                    }
                }
                instrumented_component.section(&ModuleSection(&instrumented_module));
            }

            Payload::InstanceSection(reader) => {
                for instance in reader {
                    let instance_index = get_and_increment(&mut instance_count);
                    if let Instance::Instantiate { module_index, .. } = instance? {
                        match instantiations.entry(module_index) {
                            Entry::Vacant(entry) => {
                                entry.insert(instance_index);
                            }
                            Entry::Occupied(_) => bail!("modules may be instantiated at most once"),
                        }
                    }
                }
                copy_component_section(section, component_bytes, &mut instrumented_component);
            }

            Payload::ComponentAliasSection(reader) => {
                for alias in reader {
                    match alias? {
                        ComponentAlias::CoreInstanceExport {
                            kind: ExternalKind::Func,
                            ..
                        } => {
                            core_function_count += 1;
                        }
                        ComponentAlias::InstanceExport {
                            kind: ComponentExternalKind::Type,
                            ..
                        } => {
                            type_count += 1;
                        }
                        ComponentAlias::InstanceExport {
                            kind: ComponentExternalKind::Func,
                            ..
                        } => {
                            function_count += 1;
                        }
                        _ => (),
                    }
                }
                copy_component_section(section, component_bytes, &mut instrumented_component);
            }

            Payload::ComponentCanonicalSection(reader) => {
                for function in reader {
                    match function? {
                        CanonicalFunction::Lower { .. }
                        | CanonicalFunction::ResourceNew { .. }
                        | CanonicalFunction::ResourceDrop { .. }
                        | CanonicalFunction::ResourceRep { .. }
                        | CanonicalFunction::BackpressureInc
                        | CanonicalFunction::BackpressureDec
                        | CanonicalFunction::TaskCancel
                        | CanonicalFunction::TaskReturn { .. }
                        | CanonicalFunction::ContextGet(_)
                        | CanonicalFunction::ContextSet(_)
                        | CanonicalFunction::ThreadYield { .. }
                        | CanonicalFunction::SubtaskDrop
                        | CanonicalFunction::WaitableSetNew
                        | CanonicalFunction::WaitableSetWait { .. }
                        | CanonicalFunction::WaitableSetPoll { .. }
                        | CanonicalFunction::WaitableSetDrop
                        | CanonicalFunction::WaitableJoin
                        | CanonicalFunction::StreamNew { .. }
                        | CanonicalFunction::StreamRead { .. }
                        | CanonicalFunction::StreamWrite { .. }
                        | CanonicalFunction::StreamCancelRead { .. }
                        | CanonicalFunction::StreamCancelWrite { .. }
                        | CanonicalFunction::StreamDropReadable { .. }
                        | CanonicalFunction::StreamDropWritable { .. }
                        | CanonicalFunction::FutureNew { .. }
                        | CanonicalFunction::FutureRead { .. }
                        | CanonicalFunction::FutureWrite { .. }
                        | CanonicalFunction::FutureCancelRead { .. }
                        | CanonicalFunction::FutureCancelWrite { .. }
                        | CanonicalFunction::FutureDropReadable { .. }
                        | CanonicalFunction::FutureDropWritable { .. }
                        | CanonicalFunction::ErrorContextNew { .. }
                        | CanonicalFunction::ErrorContextDebugMessage { .. }
                        | CanonicalFunction::ErrorContextDrop => {
                            core_function_count += 1;
                        }
                        CanonicalFunction::Lift { .. } => {
                            function_count += 1;
                        }
                        _ => {}
                    }
                }
                copy_component_section(section, component_bytes, &mut instrumented_component);
            }

            Payload::ComponentImportSection(reader) => {
                for import in reader {
                    match import?.ty {
                        ComponentTypeRef::Func(_) => {
                            function_count += 1;
                        }
                        ComponentTypeRef::Type(_) => {
                            type_count += 1;
                        }
                        ComponentTypeRef::Module(_) => {
                            module_count += 1;
                        }
                        _ => (),
                    }
                }
                copy_component_section(section, component_bytes, &mut instrumented_component);
            }

            Payload::ComponentExportSection(reader) => {
                for export in reader {
                    match export?.kind {
                        ComponentExternalKind::Func => {
                            function_count += 1;
                        }
                        ComponentExternalKind::Type => {
                            type_count += 1;
                        }
                        _ => (),
                    }
                }
                copy_component_section(section, component_bytes, &mut instrumented_component);
            }

            Payload::ComponentTypeSection(reader) => {
                for _ in reader {
                    type_count += 1;
                }
                copy_component_section(section, component_bytes, &mut instrumented_component);
            }

            _ => copy_component_section(section, component_bytes, &mut instrumented_component),
        }
    }

    let mut types = TypeSection::new();
    let mut imports = ImportSection::new();
    let mut functions = FunctionSection::new();
    let mut exports = ExportSection::new();
    let mut code = CodeSection::new();
    let mut aliases = ComponentAliasSection::new();
    let mut lifts = CanonicalFunctionSection::new();
    let mut component_types = ComponentTypeSection::new();
    let mut component_exports = ComponentExportSection::new();

    for (module_index, module_globals) in &instrumentation.globals {
        for (global_export, ty) in module_globals.values() {
            let ty = Encode.val_type(*ty)?;
            let offset = types.len();
            types.ty().function([], [ty]);
            imports.import(
                &module_index.to_string(),
                &global_export.module_export(),
                GlobalType {
                    val_type: ty,
                    mutable: true,
                    shared: false,
                },
            );
            functions.function(offset);
            let mut function = Function::new([]);
            function.instruction(&Ins::GlobalGet(offset));
            function.instruction(&Ins::End);
            code.function(&function);
            let export_name = global_export.component_export();
            exports.export(&export_name, ExportKind::Func, offset);
            aliases.alias(Alias::CoreInstanceExport {
                instance: instance_count,
                kind: ExportKind::Func,
                name: &export_name,
            });
            component_types
                .function()
                .params(iter::empty::<(_, ComponentValType)>())
                .result(Some(ComponentValType::Primitive(match ty {
                    ValType::I32 => PrimitiveValType::S32,
                    ValType::I64 => PrimitiveValType::S64,
                    ValType::F32 => PrimitiveValType::F32,
                    ValType::F64 => PrimitiveValType::F64,
                    ValType::V128 => bail!("V128 not yet supported"),
                    ValType::Ref(_) => bail!("reference types not supported"),
                })));
            lifts.lift(
                core_function_count + offset,
                type_count + component_types.len() - 1,
                [CanonicalOption::UTF8],
            );
            component_exports.export(
                &export_name,
                ComponentExportKind::Func,
                function_count + offset,
                None,
            );
        }
    }

    if let Some(memory_info) = &instrumentation.memory {
        let offset = types.len();
        types.ty().function([], [ValType::I32]);
        imports.import(
            &memory_info.module_index.to_string(),
            &memory_info.export_name,
            Encode.entity_type(TypeRef::Memory(memory_info.ty))?,
        );
        functions.function(offset);

        let mut function = Function::new([(1, ValType::I32)]);
        function.instruction(&Ins::MemorySize(0));
        function.instruction(&Ins::I32Const(PAGE_SIZE_BYTES));
        function.instruction(&Ins::I32Mul);
        function.instruction(&Ins::LocalTee(0));
        function.instruction(&Ins::I32Const(1));
        function.instruction(&Ins::MemoryGrow(0));
        function.instruction(&Ins::I32Const(0));
        function.instruction(&Ins::I32LtS);
        function.instruction(&Ins::If(wasm_encoder::BlockType::Empty));
        function.instruction(&Ins::Unreachable);
        function.instruction(&Ins::Else);
        function.instruction(&Ins::End);
        function.instruction(&Ins::I32Const(0));
        function.instruction(&Ins::I32Store(mem_arg(0, 1)));
        function.instruction(&Ins::LocalGet(0));
        function.instruction(&Ins::LocalGet(0));
        function.instruction(&Ins::I32Store(mem_arg(4, 1)));
        function.instruction(&Ins::LocalGet(0));
        function.instruction(&Ins::End);
        code.function(&function);

        let export_name = "component-init-get-memory".to_owned();
        exports.export(&export_name, ExportKind::Func, offset);
        aliases.alias(Alias::CoreInstanceExport {
            instance: instance_count,
            kind: ExportKind::Func,
            name: &export_name,
        });
        let list_type = type_count + component_types.len();
        component_types.defined_type().list(PrimitiveValType::U8);
        component_types
            .function()
            .params(iter::empty::<(_, ComponentValType)>())
            .result(Some(ComponentValType::Type(list_type)));
        lifts.lift(
            core_function_count + offset,
            type_count + component_types.len() - 1,
            [CanonicalOption::UTF8, CanonicalOption::Memory(0)],
        );
        component_exports.export(
            &export_name,
            ComponentExportKind::Func,
            function_count + offset,
            None,
        );
    }

    let mut instances = InstanceSection::new();
    instances.instantiate(
        module_count,
        instantiations
            .into_iter()
            .map(|(module_index, instance_index)| {
                (
                    module_index.to_string(),
                    ModuleArg::Instance(instance_index),
                )
            }),
    );

    let mut module = EncoderModule::new();
    module.section(&types);
    module.section(&imports);
    module.section(&functions);
    module.section(&exports);
    module.section(&code);

    instrumented_component.section(&ModuleSection(&module));
    instrumented_component.section(&instances);
    instrumented_component.section(&component_types);
    instrumented_component.section(&aliases);
    instrumented_component.section(&lifts);
    instrumented_component.section(&component_exports);

    let instrumented_component = instrumented_component.finish();
    Ok((instrumented_component, instrumentation))
}

// ---------------------------------------------------------------------------
// strip_module_data
// ---------------------------------------------------------------------------

pub(super) fn strip_module_data(module_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut out = EncoderModule::new();
    for payload in Parser::new(0).parse_all(module_bytes) {
        let payload = payload?;
        let section = payload.as_section();
        match payload {
            Payload::DataSection(_)
            | Payload::DataCountSection { .. }
            | Payload::StartSection { .. } => (),

            _ => {
                copy_module_section(section, module_bytes, &mut out);
            }
        }
    }
    Ok(out.finish())
}

// ---------------------------------------------------------------------------
// rewrite_init_module_for_snapshot
// ---------------------------------------------------------------------------

fn rewrite_init_module_for_snapshot<'a>(
    module_bytes: &[u8],
    outer_parser: &mut impl Iterator<Item = Result<Payload<'a>, wasmparser::BinaryReaderError>>,
    component_bytes: &[u8],
    unchecked_range: Range<usize>,
) -> Result<EncoderModule> {
    let mut type_params: HashMap<u32, usize> = HashMap::new();
    struct FuncInfo {
        fn_name: String,
        param_count: usize,
    }
    let mut func_info: HashMap<u32, FuncInfo> = HashMap::new();
    {
        let mut func_idx = 0u32;
        let sub = Parser::new(0).parse_all(module_bytes);
        for payload in sub {
            let payload = payload?;
            match payload {
                Payload::TypeSection(reader) => {
                    for (idx, rec_group) in reader.into_iter_err_on_gc_types().enumerate() {
                        let ft = rec_group?;
                        type_params.insert(idx as u32, ft.params().len());
                    }
                }
                Payload::ImportSection(reader) => {
                    for import in reader {
                        let import = import?;
                        match import {
                            Imports::Single(_, imp) => {
                                if let TypeRef::Func(type_idx) = imp.ty {
                                    let param_count =
                                        type_params.get(&type_idx).copied().unwrap_or(0);
                                    func_info.insert(
                                        func_idx,
                                        FuncInfo {
                                            fn_name: imp.name.to_owned(),
                                            param_count,
                                        },
                                    );
                                    func_idx += 1;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let should_skip = |func_idx: u32| -> Option<usize> {
        if let Some(info) = func_info.get(&func_idx) {
            match info.fn_name.as_str() {
                "_initialize"
                | "__wasm_call_ctors"
                | "__wasm_apply_data_relocs"
                | "__set_app_data"
                | "__wasm_set_libraries" => return Some(info.param_count),
                _ => {}
            }
        }
        None
    };

    let mut module_buf = EncoderModule::new();
    while let Some(payload) = outer_parser.next() {
        let payload = payload?;
        let section = payload.as_section();
        let my_range = section.as_ref().map(|(_, range)| range.clone());

        match payload {
            Payload::DataSection(_) | Payload::DataCountSection { .. } => (),

            Payload::CodeSectionStart { count, .. } => {
                let mut code = CodeSection::new();
                for _ in 0..count {
                    let entry = loop {
                        let p = outer_parser
                            .next()
                            .ok_or_else(|| anyhow!("unexpected end of __init module"))??;
                        if let Payload::CodeSectionEntry(body) = p {
                            break body;
                        }
                    };
                    let mut new_func = Function::new([]);
                    let ops = entry.get_operators_reader()?;
                    for op in ops {
                        let op = op?;
                        match op {
                            Operator::Call { function_index } => {
                                if let Some(param_count) = should_skip(function_index) {
                                    for _ in 0..param_count {
                                        new_func.instruction(&Ins::Drop);
                                    }
                                } else {
                                    new_func.instruction(
                                        &Encode
                                            .instruction(op)
                                            .context("re-encoding __init instruction")?,
                                    );
                                }
                            }
                            _ => {
                                new_func.instruction(
                                    &Encode
                                        .instruction(op)
                                        .context("re-encoding __init instruction")?,
                                );
                            }
                        }
                    }
                    code.function(&new_func);
                }
                module_buf.section(&code);
            }

            _ => {
                copy_module_section(section, component_bytes, &mut module_buf);
            }
        }

        if let Some(my_range) = my_range
            && my_range.end >= unchecked_range.end
        {
            break;
        }
    }

    Ok(module_buf)
}

// ---------------------------------------------------------------------------
// strip_module_section
// ---------------------------------------------------------------------------

fn strip_module_section<'a>(
    outer_parser: &mut impl Iterator<Item = Result<Payload<'a>, wasmparser::BinaryReaderError>>,
    component_bytes: &[u8],
    unchecked_range: Range<usize>,
) -> Result<EncoderModule> {
    let mut module_buf = EncoderModule::new();
    while let Some(payload) = outer_parser.next() {
        let payload = payload?;
        let section = payload.as_section();
        let my_range = section.as_ref().map(|(_, range)| range.clone());

        match payload {
            Payload::DataSection(_)
            | Payload::DataCountSection { .. }
            | Payload::StartSection { .. } => (),

            _ => {
                copy_module_section(section, component_bytes, &mut module_buf);
            }
        }

        if let Some(my_range) = my_range
            && my_range.end >= unchecked_range.end
        {
            break;
        }
    }

    Ok(module_buf)
}

// ---------------------------------------------------------------------------
// apply()
// ---------------------------------------------------------------------------

fn apply(measurement: Measurement, component_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut initialized_component = EncoderComponent::new();
    let mut parser = Parser::new(0).parse_all(component_bytes);
    let mut module_count: u32 = 0;
    #[allow(clippy::while_let_on_iterator)]
    while let Some(payload) = parser.next() {
        let payload = payload?;
        let section = payload.as_section();
        match payload {
            Payload::ComponentSection {
                unchecked_range, ..
            } => {
                let mut subcomponent = EncoderComponent::new();
                while let Some(payload) = parser.next() {
                    let payload = payload?;
                    let section = payload.as_section();
                    let my_range = section.as_ref().map(|(_, range)| range.clone());
                    copy_component_section(section, component_bytes, &mut subcomponent);
                    if let Some(my_range) = my_range
                        && my_range.end >= unchecked_range.end
                    {
                        break;
                    }
                }
                initialized_component.section(&NestedComponentSection(&subcomponent));
            }

            Payload::ModuleSection {
                unchecked_range, ..
            } => {
                let module_index = get_and_increment(&mut module_count);
                let is_target = measurement
                    .memory
                    .as_ref()
                    .map(|(idx, _)| *idx == module_index)
                    .unwrap_or(false);

                if is_target {
                    let mut initialized_module = EncoderModule::new();
                    let (data_section, data_segment_count) = measurement.data_section(module_index);
                    while let Some(payload) = parser.next() {
                        let payload = payload?;
                        let section = payload.as_section();
                        let my_range = section.as_ref().map(|(_, range)| range.clone());
                        match payload {
                            Payload::MemorySection(reader) => {
                                let mut memories = MemorySection::new();
                                for memory in reader {
                                    let mut memory = memory?;
                                    memory.initial = measurement
                                        .memory_initial(module_index)
                                        .expect("measurement for module's memory");
                                    memories.memory(Encode.memory_type(memory)?);
                                }
                                initialized_module.section(&memories);
                            }

                            Payload::DataSection(_) | Payload::StartSection { .. } => (),

                            Payload::DataCountSection { .. } => {
                                initialized_module.section(&DataCountSection {
                                    count: data_segment_count,
                                });
                            }

                            _ => copy_module_section(
                                section,
                                component_bytes,
                                &mut initialized_module,
                            ),
                        }

                        if let Some(my_range) = my_range
                            && my_range.end >= unchecked_range.end
                        {
                            break;
                        }
                    }
                    if let Some(data_section) = data_section {
                        initialized_module.section(&data_section);
                    }
                    initialized_component.section(&ModuleSection(&initialized_module));
                } else {
                    let saved_bytes = &component_bytes[unchecked_range.start..unchecked_range.end];

                    let mut has_prepare_snapshot = false;
                    {
                        let sub = Parser::new(0).parse_all(saved_bytes);
                        for payload in sub {
                            let payload = payload?;
                            if let Payload::ExportSection(reader) = payload {
                                for export in reader {
                                    let export = export?;
                                    if export.name == "__prepare_snapshot" {
                                        has_prepare_snapshot = true;
                                    }
                                }
                            }
                        }
                    }

                    if has_prepare_snapshot {
                        initialized_component.section(&ModuleSection(
                            &rewrite_init_module_for_snapshot(
                                saved_bytes,
                                &mut parser,
                                component_bytes,
                                unchecked_range,
                            )?,
                        ));
                    } else {
                        let stripped =
                            strip_module_section(&mut parser, component_bytes, unchecked_range)?;
                        initialized_component.section(&ModuleSection(&stripped));
                    }
                }
            }

            Payload::ComponentImportSection(reader) => {
                for import in reader {
                    match import?.ty {
                        ComponentTypeRef::Module(_) => {
                            module_count += 1;
                        }
                        _ => (),
                    }
                }
                copy_component_section(section, component_bytes, &mut initialized_component);
            }

            _ => copy_component_section(section, component_bytes, &mut initialized_component),
        }
    }

    let initialized_component = initialized_component.finish();

    let mut add = wasm_metadata::AddMetadata::default();
    add.processed_by = vec![("host-snapshot".to_owned(), "0.1.0".to_owned())];
    let initialized_component = add.to_wasm(&initialized_component)?;

    Validator::new_with_features(WasmFeatures::all()).validate_all(&initialized_component)?;

    Ok(initialized_component)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_and_increment(n: &mut u32) -> u32 {
    let v = *n;
    *n += 1;
    v
}

fn mem_arg(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

fn copy_component_section(
    section: Option<(u8, Range<usize>)>,
    component: &[u8],
    result: &mut EncoderComponent,
) {
    if let Some((id, range)) = section {
        result.section(&RawSection {
            id,
            data: &component[range],
        });
    }
}

fn copy_module_section(
    section: Option<(u8, Range<usize>)>,
    module: &[u8],
    result: &mut EncoderModule,
) {
    if let Some((id, range)) = section {
        result.section(&RawSection {
            id,
            data: &module[range],
        });
    }
}

// ===========================================================================
// Host-side snapshot integration
// ===========================================================================

struct HostInvoker<T: 'static> {
    instance: WasmtimeInstance,
    store: Store<T>,
}

impl<T: Send + 'static> HostInvoker<T> {
    async fn call<R: ComponentNamedList + Lift + Send + Sync + 'static>(
        &mut self,
        name: &str,
    ) -> Result<R> {
        let export = self
            .instance
            .get_export_index(&mut self.store, None, name)
            .ok_or_else(|| anyhow!("{name} is not exported"))?;
        let func = self
            .instance
            .get_func(&mut self.store, export)
            .ok_or_else(|| anyhow!("{name} export is not a func"))?;
        let func = func
            .typed::<(), R>(&mut self.store)
            .with_context(|| format!("type of {name} func"))?;
        let r = func
            .call_async(&mut self.store, ())
            .await
            .with_context(|| format!("executing {name}"))?;
        func.post_return_async(&mut self.store)
            .await
            .with_context(|| format!("post-return {name}"))?;
        Ok(r)
    }
}

#[async_trait]
impl<T: Send + 'static> Invoker for HostInvoker<T> {
    async fn call_s32(&mut self, name: &str) -> Result<i32> {
        Ok(self.call::<(i32,)>(name).await?.0)
    }
    async fn call_s64(&mut self, name: &str) -> Result<i64> {
        Ok(self.call::<(i64,)>(name).await?.0)
    }
    async fn call_f32(&mut self, name: &str) -> Result<f32> {
        Ok(self.call::<(f32,)>(name).await?.0)
    }
    async fn call_f64(&mut self, name: &str) -> Result<f64> {
        Ok(self.call::<(f64,)>(name).await?.0)
    }
    async fn call_list_u8(&mut self, name: &str) -> Result<Vec<u8>> {
        Ok(self.call::<(Vec<u8>,)>(name).await?.0)
    }
}

// ---------------------------------------------------------------------------
// snapshot_component -- instruments, initializes, measures, and applies.
// ---------------------------------------------------------------------------

pub(super) async fn snapshot_component<T: Send + 'static>(
    engine: &Engine,
    original_bytes: &[u8],
    linker: &Linker<T>,
    make_store: &(dyn Fn(&Engine) -> Result<Store<T>> + Send + Sync),
) -> Result<Vec<u8>> {
    let (instrumented_bytes, instrumentation) = instrument(original_bytes)?;

    Validator::new_with_features(WasmFeatures::all())
        .validate_all(&instrumented_bytes)
        .context("validating instrumented component")?;

    let instrumented_component =
        Component::new(engine, &instrumented_bytes).context("compiling instrumented component")?;

    let mut store = make_store(engine)?;

    let instance = linker
        .instantiate_async(&mut store, &instrumented_component)
        .await
        .context("instantiating instrumented component")?;

    let export_idx = instance
        .get_export_index(&mut store, None, "prepare-snapshot")
        .ok_or_else(|| anyhow!("missing prepare-snapshot export"))?;
    let func = instance
        .get_func(&mut store, export_idx)
        .ok_or_else(|| anyhow!("prepare-snapshot export is not a func"))?;
    let typed = func.typed::<(), ()>(&mut store)?;
    typed.call_async(&mut store, ()).await?;
    typed.post_return_async(&mut store).await?;

    let mut invoker: Box<dyn Invoker> = Box::new(HostInvoker { instance, store });
    let measurement = instrumentation.measure(&mut invoker).await?;

    apply(measurement, original_bytes)
}
