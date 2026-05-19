# Rust Migration for the Pie Layout Loader

This document describes the north-star Rust migration for Pie's weight-loading
compiler.

The split is deliberate:

```
C++ metadata/config/ABI data
    -> Rust loader compiler
    -> flat StorageProgram
    -> C++ executor (CUDA native or portable ggml)
    -> WeightStore
```

Rust owns model/schema adaptation, the layout algebra, type inference,
optimization, storage planning, plan diagnostics, and CPU reference tests. C++
owns file IO, mmap/GDS/pread, CUDA allocation, ggml allocation, streams,
kernels, FlashInfer / CUTLASS bindings, the CUDA `WeightStore`, portable ggml
backend tensors, and model forward/bind code.

The boundary between them is plain data. No C++ object implements a Rust trait.
No Rust code owns CUDA resources.

## 1. Goals

1. **Pure compiler.** Rust has no file descriptors, CUDA handles, streams, or
   kernel launches.
2. **Data ABI, not callback ABI.** C++ passes checkpoint metadata, model config,
   runtime ABI contracts, and backend policy as flat C views.
3. **Rich internal IR.** Rust uses normal enums, vectors, strings, and errors
   internally; only the FFI layer is flattened.
4. **Explicit dependencies.** Quant side tensors such as scales, zero-points,
   `g_idx`, MXFP4 block scales, and output quant scales are explicit IR inputs
   or outputs.
5. **Runtime layout ownership.** `RuntimeAbiView` declares the final tensor
   contracts: names, dtype, encoding, layout, alignment, sharding, metadata, and
   policy.
6. **Executor-native output.** `StorageProgram` contains concrete allocations,
   source byte ranges, tiled transforms, views, metadata attachments, releases,
   and finalization. It contains no QKV, MoE, GPT-OSS, Gemma, or Qwen logic.
7. **Deterministic artifact.** The same metadata/config/ABI/target input
   produces stable tensor IDs, buffer IDs, instruction IDs, plan dumps, and cache
   keys.
8. **CPU reference correctness.** Most algebra and optimizer correctness is
   tested without a GPU by executing layout expressions against small CPU
   tensors and comparing final values/metadata.

## 2. Non-Goals

- Not a runtime, allocator, kernel library, or CUDA wrapper.
- Not a full C++ rewrite. The CUDA executor, byte sources, kernels, binders, and
  `WeightStore` stay in C++.
- Not a Python API in v0.
- Not a Rust-owned ggml runtime. The portable driver may consume the same
  Rust-compiled `StorageProgram`, but ggml backend tensor ownership and graph
  execution stay in C++.
- Not a serialization requirement for execution. Serialization is for plan
  caching and diagnostics; FFI execution uses flat views.

## 3. Crate Layout

```
driver/weight_loader/
├─ Cargo.toml
├─ cbindgen.toml
├─ build.rs
├─ include/
│  └─ weight_loader.h        # generated and CI-verified
├─ spec/
│  ├─ semantic_groups.md     # built-in group/role vocabulary
│  └─ storage_program.md     # versioned program format
├─ src/
│  ├─ lib.rs                 # internal Rust API + compile_from_views()
│  ├─ types.rs               # DType, Shape, Layout, Encoding, TensorDecl
│  ├─ source.rs              # internal CheckpointMetadata + conversion from FFI
│  ├─ config.rs              # ModelConfig, schema-facing normalized config
│  ├─ semantic.rs            # SemanticGraph, SemanticRole, SemanticGroupKind
│  ├─ schema.rs              # ModelSchema trait + registry
│  ├─ schemas/
│  │  ├─ llama.rs
│  │  ├─ qwen.rs
│  │  ├─ gemma.rs
│  │  ├─ mixtral.rs
│  │  ├─ ministral.rs
│  │  ├─ phi3.rs
│  │  ├─ olmo.rs
│  │  └─ gpt_oss.rs
│  ├─ abi.rs                 # RuntimeAbi data model, not trait boundary
│  ├─ ir.rs                  # LayoutExpr, LayoutPlan
│  ├─ typecheck.rs           # TensorDecl propagation + side conditions
│  ├─ frontend.rs            # SemanticGraph + RuntimeAbi -> LayoutPlan
│  ├─ optimizer.rs           # fixed-point algebra rewrites
│  ├─ storage.rs             # StorageProgram internal model
│  ├─ storage_compiler.rs    # algebra -> StorageProgram
│  ├─ reference.rs           # CPU evaluator for algebra/storage tests
│  ├─ ffi.rs                 # extern "C" entry points
│  ├─ ffi_types.rs           # #[repr(C)] tags/views
│  ├─ ffi_arena.rs           # stable flat buffers owned by ProgramHandle
│  ├─ dump.rs                # JSON/YAML plan dumps
│  └─ error.rs
├─ tests/
│  ├─ typecheck.rs
│  ├─ optimizer.rs
│  ├─ reference_executor.rs
│  ├─ plan_snapshots.rs
│  ├─ ffi_layout.rs
│  └─ cxx_compat.rs
└─ examples/
   └─ compile_dump.rs
```

Target size for the first complete migration is roughly 3k-4k LOC of Rust plus
one small schema file per family. The split above is a working boundary, not a
license to add framework machinery.

## 4. Internal Types

Internal Rust types should be pleasant Rust. They do not need to be C ABI
friendly.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExprId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Axis(pub u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I32,
    I16,
    I8,
    U32,
    U16,
    U8,
    Bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Encoding {
    Raw(DType),
    Quant(QuantSpec),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantSpec {
    pub scheme: QuantScheme,
    pub logical_dtype: DType,
    pub bits_per_element: u8,
    pub group_size: u32,
    pub channel_axis: Option<Axis>,
    pub scale_dtype: Option<DType>,
    pub zero_point_dtype: Option<DType>,
    pub block_shape: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantScheme {
    Fp8E4M3,
    Fp8E5M2,
    Int8Symmetric,
    Int8Asymmetric,
    AwqInt4,
    GptqInt4,
    Mxfp4E2M1E8M0,
    GgufQ4_0,
    GgufQ4_K,
    GgufQ5_0,
    GgufQ5_K,
    GgufQ8_0,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDecl {
    pub name: String,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub layout: Layout,
    pub sharding: Sharding,
    pub alignment: u32,
}
```

`TensorDecl` is the type carried by every algebra expression. Type inference
must be able to recompute each declared output and turn silent layout bugs into
compile-time errors.

Idiomatic Rust guardrails:

- use newtype IDs (`TensorId`, `ExprId`, `BufferId`, `FileId`) internally rather
  than `type Foo = u32`; flatten to raw `u32` only at FFI
- use `Axis` internally and validate FFI `i32` axes at conversion time
- keep rich Rust enums inside the compiler; convert to C tag/payload structs at
  the boundary
- return typed `thiserror` errors internally; convert to structured FFI errors
  once
- avoid self-referential structs; FFI views borrow an owned arena
- avoid trait-object plumbing across the C boundary; runtime ABI is data
- prefer explicit `Result<T, E>` side-condition failures over panics
- keep deterministic iteration by sorting checkpoint tensors and runtime
  contracts before assigning IDs

## 5. Checkpoint Metadata

Rust receives metadata as data. v0 does not require Rust to own filesystem IO.

```rust
pub struct CheckpointMetadata {
    pub files: Vec<CheckpointFile>,
    pub tensors: Vec<RawTensor>,
}

pub struct CheckpointFile {
    pub id: FileId,
    pub path: String,
    pub size_bytes: u64,
    pub format: CheckpointFormat,
}

pub struct RawTensor {
    pub id: TensorId,
    pub name: String,
    pub file_id: FileId,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub layout: Layout,
}
```

Important rule: **source identity is never implicit**. Every lowered
`ExtentWrite` must include `file_id`, `tensor_id`, source offset, source span,
and striding. The C++ executor should never need to reverse-map a tensor name
back to a file.

Rust may later add pure safetensors/GGUF header parsers for tooling and tests,
but the execution ABI starts from explicit metadata supplied by C++.

## 6. Model Schemas and Semantic Graph

Model schemas convert raw checkpoint naming into model-family semantics. The
name intentionally avoids "adapter", which is overloaded with LoRA/adapters in
model-serving code.

```rust
pub trait ModelSchema: Send + Sync {
    fn matches(&self, model_type: &str) -> bool;
    fn build(&self, metadata: &CheckpointMetadata, cfg: &ModelConfig)
        -> Result<SemanticGraph, SchemaError>;
}

pub fn find_schema(model_type: &str) -> Option<&'static dyn ModelSchema> {
    BUILTIN_SCHEMAS.iter().find(|s| s.matches(model_type)).copied()
}
```

```rust
pub struct SemanticGraph {
    pub tensors: Vec<SemanticTensor>,
    pub groups: Vec<SemanticGroup>,
}

pub struct SemanticTensor {
    pub id: SemanticId,
    pub role: SemanticRole,
    pub raw: TensorId,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticRole {
    TokenEmbedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    MlpGate,
    MlpUp,
    MlpDown,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    ExpertBias,
    Norm,
    QuantScale,
    QuantZeroPoint,
    QuantGroupIndex,
    Extension(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticGroupKind {
    AttentionQkv,
    MlpGateUp,
    ExpertBank,
    ExpertGateUpInterleaved,
    GptOssMxfp4,
    QuantizedTensor,
    Extension(String),
}

pub struct SemanticGroup {
    pub kind: SemanticGroupKind,
    pub members: Vec<SemanticId>,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
    pub metadata: GroupMetadata,
}
```

Built-in roles/groups are closed enums so typos become compile errors. Extension
strings remain available for experiments, but the paper claim should rely on
the built-in vocabulary.

## 7. Runtime ABI as Data

The runtime ABI is a declarative final-layout contract. It is not a Rust trait
implemented by C++.

```rust
pub struct RuntimeAbi {
    pub name: String,
    pub version: u32,
    pub tensors: Vec<RuntimeTensorContract>,
    pub groups: Vec<RuntimeGroupContract>,
    pub aliases: Vec<RuntimeAliasContract>,
    pub quant_policies: Vec<QuantPolicy>,
}

pub struct RuntimeTensorContract {
    pub match_role: SemanticRole,
    pub layer: LayerSelector,
    pub output: RuntimeNamePattern,
    pub decl: RuntimeDeclTemplate,
    pub recipe: Vec<RecipeOp>,
}

pub struct RuntimeGroupContract {
    pub match_group: SemanticGroupKind,
    pub layer: LayerSelector,
    pub output: Vec<RuntimeNamePattern>,
    pub decl: Vec<RuntimeDeclTemplate>,
    pub recipe: Vec<RecipeOp>,
}

pub enum RecipeOp {
    Select { axis: Axis, start: DimExpr, length: DimExpr },
    Partition { axis: Axis, parts: u32, index: RankExpr },
    Join { axis: Axis },
    Stack { axis: Axis },
    Unzip { axis: Axis },
    Reorder { perm: Vec<u8> },
    View { layout: LayoutTemplate },
    Cast { dtype: DType },
    Decode { scheme: QuantScheme },
    Encode { scheme: QuantScheme },
    Transcode { from: QuantScheme, to: QuantScheme },
    AttachQuantMetadata,
}
```

C++ declares what its runtime wants. Rust decides how to construct it from the
checkpoint metadata and semantic graph.

This is the key simplification: **new runtime = new ABI data; new model/format =
new schema/parser data.** The compiler does not grow `if runtime == cuda &&
model == gpt_oss` branches.

## 8. Backend Target and Policy

Backend policy is explicit input to planning.

```rust
pub struct BackendTarget {
    pub backend: BackendKind,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub max_tile_bytes: u64,
    pub preferred_alignment: u32,
    pub quant: QuantTargetPolicy,
    pub capabilities: BackendCapabilities,
}

pub struct BackendCapabilities {
    pub native_mxfp4_moe: bool,
    pub runtime_int8: bool,
    pub runtime_fp8: bool,
    pub gds: bool,
    pub async_copy: bool,
}

pub enum Mxfp4MoePolicy {
    NativeGemm,
    RoutedDecode,
    EagerBf16,
}
```

The planner may produce different algebra for different targets. That is a
feature, not a fallback hidden inside the executor.

## 9. Layout Algebra

Quant metadata is explicit. `Unzip` stays in the IR because gate/up
interleaving and deinterleaving are real layout transformations that the
optimizer and storage compiler should see.

```rust
pub enum LayoutExpr {
    Source {
        tensor: TensorId,
        decl: TensorDecl,
    },
    Select {
        input: ExprId,
        axis: Axis,
        start: i64,
        length: i64,
        decl: TensorDecl,
    },
    Partition {
        input: ExprId,
        axis: Axis,
        parts: u32,
        index: u32,
        decl: TensorDecl,
    },
    Join {
        inputs: Vec<ExprId>,
        axis: Axis,
        decl: TensorDecl,
    },
    Stack {
        inputs: Vec<ExprId>,
        axis: Axis,
        decl: TensorDecl,
    },
    Unzip {
        input: ExprId,
        axis: Axis,
        outputs: Vec<TensorDecl>,
    },
    Reorder {
        input: ExprId,
        perm: Vec<u8>,
        decl: TensorDecl,
    },
    View {
        input: ExprId,
        layout: Layout,
        decl: TensorDecl,
    },
    Cast {
        input: ExprId,
        dtype: DType,
        decl: TensorDecl,
    },
    Decode {
        scheme: QuantScheme,
        data: ExprId,
        metadata: Vec<ExprId>,
        decl: TensorDecl,
    },
    Encode {
        scheme: QuantScheme,
        input: ExprId,
        metadata_outputs: Vec<TensorDecl>,
        decl: TensorDecl,
    },
    Transcode {
        from: QuantScheme,
        to: QuantScheme,
        data: ExprId,
        metadata: Vec<ExprId>,
        metadata_outputs: Vec<TensorDecl>,
        decl: TensorDecl,
    },
    Attach {
        data: ExprId,
        metadata: Vec<ExprId>,
        decl: TensorDecl,
    },
    Realize {
        input: ExprId,
        runtime_name: String,
        decl: TensorDecl,
    },
}
```

`Release` is not authored in the algebra. Lifetimes are computed by the storage
compiler after lowering.

## 10. Optimizer

The optimizer is a fixed-point rewrite loop with mandatory side conditions.

Required rewrites:

- selection and partition pushdown through `Join` and `Stack`
- partition-join cancellation for resharding
- encode hoisting when packing alignment allows it
- decode/select fusion when metadata side tensors can be sliced consistently
- cast sinking for widening casts
- transcode fusion through a format-pair registry
- reorder coalescing
- identity view elision
- dead expression elimination

Each rewrite has two responsibilities:

1. Preserve inferred `TensorDecl`.
2. Preserve reference-evaluator semantics.

The test oracle is not byte-for-byte equality of optimized and unoptimized
`StorageProgram`s. Optimized programs should differ. The oracle is equality of
final tensors and attached metadata under a CPU reference executor.

## 11. Storage Program

`StorageProgram` is the executable loading plan. It contains no model-family
vocabulary.

```rust
pub struct StorageProgram {
    pub version: u32,
    pub target: StorageTarget,
    pub tensors: Vec<TensorDecl>,
    pub buffers: Vec<BufferDecl>,
    pub instrs: Vec<StorageInstr>,
    pub schedule: Vec<InstrId>,
    pub memory: MemoryPlan,
    pub diagnostics: PlanDiagnostics,
}

pub struct BufferDecl {
    pub id: BufferId,
    pub tensor: Option<TensorId>,
    pub bytes: u64,
    pub alignment: u32,
    pub lifetime: Lifetime,
}

pub enum StorageInstr {
    Allocate {
        id: InstrId,
        buffer: BufferId,
    },
    ExtentWrite {
        id: InstrId,
        source: SourceExtent,
        dest: DestExtent,
    },
    TileMap {
        id: InstrId,
        kind: TileMapKind,
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        tile: TileSpec,
        transform: TransformSpec,
    },
    CreateView {
        id: InstrId,
        input: BufferId,
        output: BufferId,
        layout: Layout,
    },
    Attach {
        id: InstrId,
        tensor: BufferId,
        metadata: Vec<BufferId>,
        spec: MetadataSpec,
    },
    Release {
        id: InstrId,
        buffer: BufferId,
    },
    Finalize {
        id: InstrId,
        tensor: BufferId,
        name: String,
    },
}

pub struct SourceExtent {
    pub file_id: FileId,
    pub tensor_id: TensorId,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub stride: StridedExtent,
}

pub struct DestExtent {
    pub buffer: BufferId,
    pub offset: u64,
    pub stride: StridedExtent,
}
```

The executor should be able to interpret this program with only:

- the program view
- byte sources keyed by `file_id`
- CUDA allocation/copy/transform services
- `WeightStoreBuilder`

No checkpoint naming, semantic roles, model config, or runtime ABI lookup should
be needed during execution.

## 12. FFI Boundary

The C ABI has flat input views and an opaque output handle.

```c
int pie_loader_compile(
    const PieLoaderCompileInput* input,
    PieLoaderProgramHandle** out_program,
    PieLoaderError* out_error);

PieLoaderStorageProgramView pie_loader_program_view(
    const PieLoaderProgramHandle* program);

void pie_loader_program_free(PieLoaderProgramHandle* program);

void pie_loader_error_free(PieLoaderError* error);
```

`PieLoaderCompileInput` contains:

- `PieCheckpointFileView[]`
- `PieCheckpointTensorView[]`
- `PieModelConfigView`
- `PieRuntimeAbiView`
- `PieBackendTargetView`

The output handle owns a flattened FFI arena:

```rust
pub struct ProgramHandle {
    program: StorageProgram,
    arena: FfiArena,
}
```

`FfiArena` contains owned C-layout arrays, string bytes, and payload arrays.
`StorageProgramView` borrows from the arena, not from self-referential pointers
into `program`. This avoids self-referential Rust structs and makes C++ RAII
straightforward.

FFI rules:

- internal Rust enums may be rich data-carrying enums
- public C enums use `#[repr(C)]` integer tags
- public payloads use tag + struct payload, not Rust enum layout
- public strings are `(ptr, len)` UTF-8 byte spans
- public slices are `(ptr, len)`
- C++ never mutates memory owned by the handle
- CI runs `cbindgen --verify` plus size/offset tests

## 13. Plan Dumps and Caching

There are two artifacts:

1. **FFI view:** executable in-process view consumed by C++.
2. **Serialized dump/cache:** versioned serde artifact for debugging, snapshot
   tests, and optional plan caching.

They must not be conflated. The cache format can use `serde`/`bincode` or
`rkyv`; the execution ABI remains flat C data.

Every dump includes:

- compiler version
- ABI name/version
- target policy/capabilities
- source file/tensor table hash
- model config hash
- stable tensor/buffer/instruction IDs
- memory plan
- unsupported-feature diagnostics when compilation fails

## 14. C++ Side

C++ keeps the physical executor:

```cpp
class RustStorageProgram {
public:
    explicit RustStorageProgram(PieLoaderProgramHandle* handle);
    ~RustStorageProgram();
    PieLoaderStorageProgramView view() const noexcept;

private:
    PieLoaderProgramHandle* handle_ = nullptr;
};

class StorageProgramExecutor {
public:
    LoadExecutionStats execute(
        const PieLoaderStorageProgramView& program,
        ByteSourceRegistry& sources,
        WeightStoreBuilder& weights);
};
```

The remaining C++ loader surface should shrink toward:

- `rust_storage_program.{hpp,cpp}`: handle wrapper and view translation
- `rust_storage_executor.hpp`: interpreter over flat program views
- `pie_cuda_abi.{hpp,cpp}`: declarative ABI construction
- `transforms.{hpp,cpp}`: CUDA/FlashInfer/CUTLASS transform dispatch

Model binders consume a finalized `WeightStore`; they do not see planner
selection or compatibility paths.

## 15. Testing Strategy

Rust tests:

- type inference failures for shape/dtype/layout/encoding mismatches
- property tests for rewrite soundness against CPU reference tensors
- optimizer-on/off final tensor equality, not program equality
- snapshot tests for `(family, quantization, tp)` plan dumps
- storage compiler validation for coverage, dependency order, lifetimes, and
  memory high-water
- FFI layout and cbindgen header stability

C++ tests:

- compile Rust program and execute through the CUDA loader golden tests
- compare final tensors and quant metadata for representative transforms
- run plan parity against the current C++ planner while dual-planning is enabled
- e2e load smoke for claimed model families

Evidence tests:

- planned vs actual load memory high-water
- startup/load time comparison with C++ planner
- later, vLLM/SGLang comparison for the paper artifact

## 16. Migration Phases

### Phase 0: Commit the boundary

- Add the Rust crate skeleton.
- Define FFI input/output structs.
- Generate and verify the cbindgen header.
- Add C++ RAII wrapper.
- `pie_loader_compile` returns an empty valid program.

Acceptance: C++ can call Rust, get a handle, view it, and free it under ASAN.

### Phase 1: Metadata/config/ABI data path

- C++ builds `PieLoaderCompileInput` from existing metadata and config.
- C++ builds `PieRuntimeAbiView` from the current Pie CUDA ABI.
- Rust validates input and emits diagnostic dumps.

Acceptance: no planning yet, but all current model snapshots can be described as
Rust input data.

### Phase 2: Dense BF16 side-by-side compiler

- Port `Source`, `Select`, `Partition`, `Join`, `View`, `Cast`, `Realize`.
- Add Llama/Qwen dense schema coverage.
- Lower direct copies and simple tile casts.
- Run C++ planner and Rust planner side by side.

Acceptance: Rust-produced `StorageProgram` executes dense BF16 loads through the
existing C++ executor.

### Phase 3: Groups and MoE

- Port `Stack`, `Unzip`, grouped views, expert banks, Qwen/Gemma/Mixtral MoE.
- Add direct expert extent writes.

Acceptance: Rust plans cover current BF16 dense/MoE claim set with matching
golden outputs.

### Phase 4: Quantization and Transcode

- Port `Decode`, `Encode`, `Transcode`, `Attach`.
- Add AWQ, GPTQ, compressed-tensors FP8/INT8, GPT-OSS MXFP4, and GGUF metadata
  coverage.
- Add CPU golden fixtures for representative quant blocks.

Acceptance: Rust covers the current C++ quantized loader claim set and adds
golden transcode fixtures for at least FP8->INT8 and one INT4 path.

### Phase 5: Cutover

- Make Rust the only planner/executor path for CUDA and portable loading.
- Remove `cpp`/`dual` planner modes and all production fallback execution.
- Compare deterministic dumps and final tensors in CI.
- Delete C++ planning once the remaining tests are rewritten against Rust
  plan fixtures.

Acceptance: production weight loading always executes a Rust-compiled
`StorageProgram`; unsupported coverage fails at compile time instead of falling
back.

### Phase 6: Paper-Grade Artifact

- Versioned plan cache and deterministic dumps.
- CPU property-test corpus.
- Bug-archeology examples mapped to type errors.
- Memory high-water telemetry.
- Benchmark script regenerating loader evidence.
- Native backend hooks for MXFP4 MoE once a FlashInfer/CUTLASS path is selected.

Acceptance: the artifact supports the paper claim: weight loading is a typed
compiler from checkpoint metadata and runtime ABI to a physical storage program.

## 17. Missing North-Star Requirements

These are easy to forget and should be treated as first-class requirements:

1. **Stable IDs and deterministic ordering.** Plan snapshots and caches are only
   useful if tensor, buffer, and instruction IDs are stable across machines.
2. **Versioned failure diagnostics.** Unsupported quant formats should produce
   structured errors with missing capability/format names, not ad hoc strings.
3. **Runtime ABI validation.** Rust must validate that ABI contracts do not
   request impossible shapes, duplicate final names, dangling aliases, or
   unsupported transform chains.
4. **Explicit memory accounting.** The storage compiler reports persistent
   bytes, temporary peak, tile scratch, checkpoint read bytes, and device write
   bytes. Runtime-only scratch remains a separate executor statistic.
5. **No hidden source lookup.** Every storage read names the file and tensor ID.
6. **CPU reference semantics.** Every algebra op that claims a rewrite law has a
   CPU evaluator implementation.
7. **Plan cache invalidation.** Cache keys include compiler version, ABI version,
   model config hash, checkpoint metadata hash, and backend target policy.
8. **Feature gates for partial migration.** The cutover needs `rust`, `cpp`, and
   `dual` planner modes until parity is proven.
9. **FFI fuzz/ASAN coverage.** Malformed views should fail safely in Rust; valid
   views should survive ASAN/UBSAN on the C++ side.
10. **Backend integration discipline.** FlashInfer/CUTLASS/Marlin integrations are
    executor transform backends, not model-loader branches.

## 18. Summary

The Rust migration should not be a Rust port of `model_schema.cpp`. It should be
a compiler boundary:

```
PieLoaderCompileInput
  { checkpoint metadata, model config, runtime ABI, backend target }
      -> Rust model schemas
      -> typed LayoutAlgebra
      -> optimizer
      -> StorageProgram
      -> flat FFI view
      -> C++ CUDA executor
      -> WeightStore
```

This keeps the architecture small and paper-aligned. Rust gives us the typed
compiler and testable rewrite system; C++ keeps the physical CUDA integration.

## 19. Implementation Tracker

Current Rust implementation status:

- Added `driver/weight_loader` as a workspace crate with generated
  `include/weight_loader.h`.
- Implemented flat FFI inputs for checkpoint files/tensors, model config,
  runtime tensor contracts, backend target policy, and structured errors.
- Implemented schema-driven semantic graph construction with built-in role and
  group enums; runtime contracts can use either direct tensor IDs or semantic
  role/layer/expert matching.
- Implemented typed layout algebra, type inference, CPU reference semantics,
  optimizer rewrites, and algebra-only storage lowering.
- Implemented flat executable `StorageProgram` views with instruction payloads
  for allocate, extent write, tile map, create view, attach, release, and
  finalize.
- Made `TileMap` placement explicit: tiled transforms now carry an optional
  destination extent, so buffer-to-buffer joins/stacks/reblocks have the same
  physical addressability as raw `ExtentWrite`s instead of relying on implicit
  output offsets.
- Added C++ boundary headers for owning Rust program handles and for building
  flat `PieLoaderCompileInput` views from CUDA-side metadata/config data.
- Wired the CUDA loader to compile/dump Rust storage programs as the only
  weight-loading path and to execute dense direct, compact cast, create-view,
  release, and identity reblock Rust-compiled paths through the CUDA
  `WeightStoreBuilder`.
- Wired the portable ggml loader to compile/dump Rust storage programs as the
  only weight-loading path. The old portable C++/dual planner fallback has
  been removed; all ggml backends, including Metal builds, execute the same
  Rust-compiled direct/cast, strided extent-write, create-view, identity
  reblock/reorder, and decode paths into backend tensors.
- Added a generic `ByteSpans` runtime source/IR term for byte-range assembly:
  fused slices, stacked expert writes, source/destination offset writes, and
  GGUF-style byte passthrough now lower to ordinary `ExtentWrite`
  instructions instead of model-family compatibility code.
- Added explicit runtime metadata tensors to the ABI. Quantized source tensors
  now lower as `Decode(data, metadata)` rather than relying on executor-side
  name conventions; portable FP8 checkpoints use this for scalar and per-row
  scale tensors.
- Moved CUDA compatibility-view ranges into `RuntimeABI`/`TensorDecl` as
  explicit `view_axis/start/length` fields. The Rust bridge now validates
  QKV/gate-up view ranges against the ABI-owned contract before emitting
  `Select` contracts.
- Extended the CUDA Rust input surface to match portable: direct contracts can
  carry metadata tensor IDs, and `ByteSpans` contracts are available for
  source/destination offset assembly.
- Lowered CUDA MoE expert-bank `Stack` roots into generic `ByteSpans`
  contracts. Qwen MoE gate/up/down expert tensors now compile to ordinary
  storage-program extent writes instead of requiring a model-specific Rust
  executor path.
- Added generic quant metadata attachment for CUDA Rust execution. The bridge
  carries `TensorDecl.quant` scale contracts into the executor, and the
  executor restores `WeightStore` quant metadata before finalization. This is
  used by GPT-OSS packed MXFP4 expert tensors.
- Added a Rust optimizer rewrite that distributes `Cast` over `Join`/`Stack`,
  allowing the storage compiler to avoid materializing a wide joined temporary
  when inputs can be tiled directly into the final dtype.
- Made Rust the single canonical layout optimizer. The former CUDA C++
  `layout_optimizer` rewrite path has been removed from production and test
  builds, and the CUDA loader no longer exposes `cpp`/`dual` planner modes.
- Ported the remaining CUDA C++ algebra rewrites into Rust: select-through-
  decode with metadata slicing, conservative encode-through-select hoisting,
  encode/decode transcode fusion, decode/cast fusion, dead output binding
  elimination, and structured optimizer pass reporting on `StorageProgram`.
  The optimizer report is also exposed through the flat C ABI so CUDA and
  portable storage-program dumps show the canonical Rust passes that ran.
- Updated the CUDA Rust executor to honor compiled physical file offsets for
  compact `ExtentWrite` instructions via safetensors byte-range copies.
- Extended the portable Rust executor with CPU temporary buffers and executable
  `TileMap::Decode` for FP8_E4M3 -> BF16, so metadata-backed transforms run
  from the compiled storage program.
- Added a storage-compiler regression test proving tiled buffer joins carry
  exact destination offsets.
- Fixed GGUF declaration policy so tensors with `ggml_type_override` keep the
  GGUF-owned runtime type instead of passing through safetensors F32/BF16
  cast heuristics.
- Kept generated portable constants outside checkpoint coverage: synthesized
  tensors are produced by the runtime and copied after Rust materializes all
  checkpoint-backed weights.
- Added Rust tests for FFI safety, type errors, reference semantics, optimizer
  rewrites, semantic-role contracts, generated C++ header stability, and wrapper
  header compilation. Optimizer tests now cover every rewrite that used to
  live in CUDA C++.
- Added `examples/compile_dump.rs` and `bench_loader.sh` for deterministic dump
  and compile-time evidence generation.
- Extended `benches/run_loader_evidence.py` with `--pie-driver` so CUDA and
  portable Rust storage-program dumps can be regenerated by the same evidence
  harness.

Current e2e smoke evidence:

- CUDA native, default Rust planner, `Qwen/Qwen3-0.6B`,
  `cuda:0`: 367/367 contracts, 1045 Rust storage instructions, optimizer
  report present in the plan dump, one-token latency smoke passed.
- CUDA native, default Rust loader, `Qwen/Qwen3-32B`,
  `cuda:0`, small KV pool: 835/835 contracts, 62.5 GiB loaded through Rust
  storage, one-token smoke passed.
- CUDA native, default Rust loader, `Qwen/Qwen3.6-27B`,
  `cuda:0`, small KV pool: 1199/1199 contracts, 52.9 GiB loaded through Rust
  storage, one-token smoke passed.
- CUDA native, default Rust loader, `Qwen/Qwen3-30B-A3B`,
  `cuda:0`, small KV pool: 531/531 contracts, MoE expert bank byte-span
  lowering, one-token smoke passed.
- CUDA native, default Rust loader, `Qwen/Qwen3.6-35B-A3B`,
  `cuda:0`, small KV pool: 1045/1045 contracts, one-token smoke passed.
- CUDA native, default Rust loader, `openai/gpt-oss-20b`,
  `cuda:0`, `mxfp4_moe=packed`, small KV pool: 459/459 contracts, packed
  MXFP4 quant metadata restored, one-token smoke passed.
- CUDA native, default Rust planner, `google/gemma-4-E2B`: 2011/2011
  contracts, 9.5 GiB loaded through Rust storage, one-token smoke passed.
- CUDA native, default Rust planner, `google/gemma-4-E4B`: 2130/2130
  contracts, 14.9 GiB loaded through Rust storage, one-token smoke passed.
- CUDA native, default Rust planner, `google/gemma-4-26B-A4B`: 1013/1013
  contracts, 48.1 GiB loaded through Rust storage, one-token MoE smoke passed.
- Portable ggml, default Rust planner, `Qwen/Qwen3-0.6B`, CPU backend:
  311/311 contracts, 933 Rust storage instructions, optimizer report present
  in the plan dump, one-token latency smoke passed.
- Portable ggml, Rust storage loader,
  `microsoft/Phi-3-mini-4k-instruct`: 291/291 checkpoint-backed contracts,
  fused QKV/gate-up slice smoke passed.
- Portable ggml, Rust storage loader, `google/gemma-4-E2B`:
  505/505 checkpoint-backed contracts plus generated constants, one-token
  smoke passed.
- Portable ggml, Rust storage loader, `google/gemma-4-E4B`:
  623/623 checkpoint-backed contracts plus generated constants, one-token
  smoke passed.
- Portable ggml, Rust storage loader,
  `google/gemma-4-26B-A4B`: 657/657 checkpoint-backed contracts plus generated
  constants, one-token MoE smoke passed.
- Portable ggml, Rust storage loader,
  `allenai/Olmo-3-7B-Instruct`: 355/355 contracts, one-token smoke passed.
- Portable ggml, Rust storage loader,
  `RedHatAI/Qwen3-0.6B-FP8-dynamic`: 311/311 contracts,
  metadata-backed FP8 decode smoke passed after the explicit TileMap
  destination change.
- Portable ggml, Rust storage loader,
  `mistralai/Ministral-3-3B-Instruct-2512`: 236/236 contracts,
  static FP8 `weight_scale_inv` decode smoke passed.
- Portable ggml offline driver runner, Rust storage loader,
  `unsloth/Qwen3-0.6B-GGUF`: Q4_K_M, Q5_K_M, and Q8_0 all loaded as
  310/310-contract Rust storage programs and completed one-token generation;
  Q4_K_M was rechecked after the explicit TileMap destination change.

Current cutover checkpoint:

- CUDA now compiles safetensors metadata directly through the Rust default ABI,
  algebra optimizer, and storage compiler. The former CUDA C++ `LayoutPlan`,
  `model_schema`, `RuntimeABI`, semantic graph, planner, typecheck, and
  `test_layout_plan` build target have been removed.
- The Rust storage compiler preserves source strides for non-leading-axis TP
  slices, and the CUDA executor materializes those strided extents through the
  generic safetensors slice copier.
- CUDA `runtime_quant={fp8,int8}` is back on the Rust path: the default
  `RuntimeABI` emits Quant contracts for dense projection weights, the algebra
  lowers Raw -> Quant as `Encode` plus generated `_scale_inv` metadata outputs,
  and the CUDA executor materializes those outputs through Encode TileMaps.
  The storage FFI now carries the physical tile budget, so compact Encode
  sources execute as bounded row tiles rather than full-source scratch tensors.
- `cargo test --manifest-path driver/weight_loader/Cargo.toml`,
  `cmake --build driver/cuda/build --target pie_driver_cuda_lib -j2`,
  `cmake --build driver/cuda/build --target pie_driver_cuda -j2`,
  `ctest --test-dir driver/cuda/build --output-on-failure`, and
  `cmake --build driver/portable/build --target pie_driver_portable_lib -j2`
  pass on 2026-05-19. `cargo build -p pie-server --release --no-default-features --features driver-cuda`
  also links.

Remaining cutover work is now concentrated on large-model and production
evidence: running the very large MoE families where host/GPU memory permits,
checking CUDA Rust execution on the same coverage matrix, and broadening
quantized executor kernels beyond FP8/MXFP4/AWQ/GPTQ reference paths.
