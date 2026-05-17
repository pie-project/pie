# Compiling Weight Loading: Redesign and Refactor Plan

This document is a standalone redesign proposal for Pie's model loading
system. It is intentionally separate from `LOADER.md`: `LOADER.md` tracks the
current implementation, while this document describes the cleaner architecture
we want the codebase to converge toward.

The goal is to align Pie with the "Compiling Weight Loading" proposal:

```text
Semantic graph -> Layout algebra -> Storage program -> Runtime weight store
```

The key change is conceptual. The loader should not be a sequence of
architecture-specific mutations. It should be a compiler from checkpoint
layouts to runtime layouts, with a typed algebraic IR and a storage program as
the only executable artifact.

## Current Status And Gap

The current CUDA loader already has many of the right components:

- `CheckpointSource`
- `SafetensorsCheckpointSource`
- `LayoutPlan`
- `StorageProgram`
- `LoadExecutor`
- `RuntimeABI`
- `WeightStore`

The implementation has started the migration toward the compiler architecture:

- `LoadExecutor` now requires a `StorageProgram` and walks `StorageInstr`
  records. The old top-level compatibility executor switch has been removed.
- `StorageInstr` carries typed storage payloads, and tiled transforms are
  represented as `TileMap` records.
- `StorageCompiler` lowers algebra directly. `Source`, `Select`, `Partition`,
  `Join`, and direct expert `Stack` lower to byte extents; `Cast`, `Encode`,
  `Decode`, and `Transcode` lower to typed tile maps. Unsupported algebra fails
  compilation explicitly.
- `RuntimeABI` exposes final tensor contracts with dtype, encoding, layout,
  ownership, alignment, sharding, quant metadata, and quant policy.
- `LayoutOptimizer` has concrete normalization rewrites for
  selection/partition pushdown through joins, select/decode movement,
  encode/select movement, cast sinking through joins/stacks,
  partition-join cancellation, decode-cast fusion, and Encode-Decode
  transcode fusion.
- `GgufCheckpointSource` parses dense tensors and initial quant block metadata
  (`gguf.q4_0`), with a decoded Q4_0 golden fixture.
- GPT-OSS MXFP4 planning is algebra-native for both packed routed-dequant and
  eager-BF16 fallback policies; storage compilation schedules the corresponding
  extent writes, decode tile maps, deinterleave/select transforms, attach
  steps, and releases without compatibility metadata.

The remaining gap is narrower but still important:

- Schema adaptation, target policy, runtime ABI decisions, layout planning, and
  lowering are still heavily concentrated in `model_schema.cpp`.
- Type checking now propagates `TensorDecl` through the algebra, but coverage
  should continue expanding for quantization metadata and secondary outputs.
- The remaining cleanup is modularity, not compatibility removal: split
  per-family adapters from planner/lowering modules and keep the algebra IR
  model-neutral.

The refactor target is not to add layers for their own sake. The target is to
make ownership boundaries exact:

```text
Source describes bytes.
Adapter describes meaning.
ABI describes final runtime layout.
Compiler connects them.
Executor only runs the compiled storage program.
```

## North-Star Architecture

### 1. CheckpointSource

`CheckpointSource` describes raw checkpoint storage. It owns file/container
metadata, tensor byte ranges, raw tensor names, raw shapes, raw dtypes, and
checkpoint-side encodings.

It must not know:

- model semantics
- runtime tensor names
- CUDA kernel layout preferences
- tensor-parallel policy
- whether Q/K/V or gate/up should be packed

Required interface:

```cpp
class CheckpointSource {
public:
    virtual bool contains(TensorId raw) const = 0;
    virtual TensorDecl raw_decl(TensorId raw) const = 0;
    virtual StorageExtent storage_extent(TensorId raw) const = 0;
    virtual std::vector<TensorId> tensors() const = 0;
};
```

Format implementations:

- `SafetensorsCheckpointSource`
- `GgufCheckpointSource`
- future: object-store shards, remote byte-range sources, training checkpoints

GGUF support should be implemented here as format metadata and byte extents,
not as a separate semantic loader path.

### 2. ModelAdapter

`ModelAdapter` maps checkpoint tensors to canonical semantic symbols and groups.
It is the only layer that understands model-family naming conventions.

Its output is a `SemanticGraph`:

```cpp
struct SemanticTensor {
    SemanticId id;
    TensorId raw;
    SemanticRole role;
    TensorDecl checkpoint_decl;
};

struct SemanticGroup {
    SemanticGroupKind kind;
    std::vector<SemanticId> members;
    GroupMetadata metadata;
};

struct SemanticGraph {
    std::vector<SemanticTensor> tensors;
    std::vector<SemanticGroup> groups;
};
```

Examples of semantic roles:

- `Embedding`
- `LmHead`
- `Norm`
- `AttentionInputProjection`
- `AttentionOutputProjection`
- `FeedForwardInputProjection`
- `FeedForwardOutputProjection`
- `ExpertInputProjection`
- `ExpertOutputProjection`
- `QuantData`
- `QuantScale`
- `QuantZeroPoint`

The adapter may use suffix matching internally, because checkpoint naming is a
format/model-family concern. But suffix matching must not leak into lowering or
runtime ABI selection.

Target state:

- Qwen dense/MoE adapter emits semantic tensors and groups.
- Gemma dense/MoE adapter emits semantic tensors and groups.
- GPT-OSS adapter emits MXFP4 semantic tensors and metadata groups.
- Mixtral, Ministral, Phi-3, OLMo adapters emit explicit groups.
- GGUF adapter maps GGUF tensor names and quant block metadata into the same
  semantic graph.

### 3. RuntimeABI and BackendTarget

`RuntimeABI` owns final runtime tensor contracts. This is the boundary that
decides final GPU names and layouts.

`BackendTarget` owns device and backend policy:

```cpp
struct BackendTarget {
    DeviceCapability device;
    TensorParallelPolicy tp;
    ExpertParallelPolicy ep;
    QuantPolicy quant;
    StoragePolicy storage;
};
```

`RuntimeABI` answers:

- What final tensors does the runtime require?
- What are their names?
- What dtype, encoding, layout, alignment, and sharding do they require?
- Which tensors are views or aliases?
- Which quantized formats are native on this backend?
- Which formats require tiled decode, routed decode, or eager decode?

This means final GPU layout is explicitly owned by the runtime, not by
checkpoint naming and not by ad-hoc schema lowering.

### 4. Layout Algebra

The central IR is a typed expression DAG. A `LayoutPlan` is a set of output
bindings from runtime tensor IDs to `LayoutExpr` roots.

Core expression grammar:

```text
Expr ::= Source(raw_tensor)
       | Select(axis, start, length, Expr)
       | Partition(axis, parts, index, Expr)
       | Join(axis, Expr...)
       | Stack(axis, Expr...)
       | Reorder(perm, Expr)
       | View(layout, Expr)
       | Cast(dtype, Expr)
       | Encode(encoding, Expr)
       | Decode(encoding, Expr)
       | Attach(metadata, Expr...)
       | Realize(runtime_tensor, Expr)
```

Derived forms such as QKV packing, gate/up packing, row-range split, expert
bank fusion, and tensor-parallel shard are planner conveniences. They must
lower into the core algebra before optimization and storage compilation.

### 5. TensorDecl as the Type System

Every expression has an inferred `TensorDecl`:

```cpp
struct TensorDecl {
    Shape shape;
    DType dtype;
    Encoding encoding;
    Layout layout;
    Sharding sharding;
    MetadataSchema metadata;
};
```

The type checker must validate:

- shape compatibility for `Join` and `Stack`
- range validity for `Select`
- divisibility and boundary alignment for `Partition`
- dtype legality for `Cast`
- encoding legality for `Encode` and `Decode`
- metadata availability for quantized tensors
- layout compatibility for `View`
- runtime ABI compatibility for `Realize`

This turns many silent loader failures into compile-time plan errors.

### 6. Layout Optimizer

The optimizer rewrites layout algebra into a cheaper equivalent form. The first
implementation should be rule-based. A cost model can be added later.

Must-have rewrites:

- selection pushdown through `Join`
- partition pushdown through `Join`
- partition-join cancellation for resharding
- cast sinking toward `Realize`
- decode sinking or fusion into tiled realization
- adjacent select coalescing
- adjacent reorder coalescing
- identity view elimination
- dead expression elimination

The paper should avoid claiming a global unique normal form until the rewrite
system is formally constrained. A safer claim is:

```text
The compiler canonicalizes supported algebra fragments under explicit side
conditions and validates every rewrite by type preservation.
```

### 7. Storage Compiler

The storage compiler lowers optimized layout algebra into `StorageProgram`.

The storage program is the physical execution plan:

```text
Allocate
ExtentWrite
TileMap
CreateView
Attach
Release
Finalize
```

Responsibilities:

- compute exact checkpoint byte extents
- compute exact destination offsets
- schedule writes by file order when useful
- coalesce adjacent source ranges
- emit tiled transforms for cast/decode/encode/reblock
- account for persistent bytes, temporary bytes, and scratch bytes
- express dependencies and lifetimes explicitly
- reject any layout expression that cannot lower physically

The storage compiler is where the system becomes a compiler rather than a
renamed loader.

### 8. LoadExecutor

`LoadExecutor` should execute only `StorageProgram`.

It should not switch on model-family semantics. It should not know model-family
concepts. It should not decide that QKV, MoE, or MXFP4 need special handling.

Allowed responsibilities:

- allocate device tensors
- read from `CheckpointByteSource`
- invoke copy kernels
- invoke tiled transform kernels
- create views and aliases
- attach metadata
- release temporaries
- collect telemetry
- finalize an immutable `WeightStore`

Target invariant:

```text
If a transformation happens during load, it appears as a typed StorageInstr.
```

## Quantization and Encoding

Quantization should be modeled as encoding, not as a collection of unrelated
special cases.

Examples:

- BF16: raw dense encoding
- FP8 E4M3: encoded scalar format
- AWQ/GPTQ: packed int4 plus scale/zero metadata
- MXFP4: packed E2M1 data plus E8M0 scales
- GGUF Q4_K/Q5_K/etc.: block encodings with scheme-specific metadata

Each encoding spec must declare:

- logical element dtype
- stored dtype or byte representation
- block shape
- packing factor
- alignment constraints
- scale and zero-point metadata
- supported decode targets
- supported native runtime targets
- legal selection boundaries

The `Encode`/`Decode` algebra should be precise, but the paper should not claim
all encodings form one universal algebra. Each encoding contributes laws under
explicit side conditions, such as block alignment and rounding policy.

## GGUF Support

GGUF changes the design by adding a different `CheckpointSource` and additional
encoding specs. It should not change the runtime ABI or require separate
runtime-specific conversion paths.

Implementation shape:

```text
GgufCheckpointSource
  -> raw GGUF tensor declarations
  -> GGUF encoding metadata
  -> ModelAdapter maps GGUF names to semantic IDs
  -> RuntimeABI requests final CUDA tensors
  -> LayoutPlanner emits Decode/View/Partition/Realize expressions
  -> StorageCompiler emits ExtentWrite/TileMap instructions
```

The important design point:

```text
Safetensors and GGUF differ at the source and encoding layers.
They should converge before runtime layout planning.
```

## Refactor Plan

### Phase 1: Make the Layer Boundaries Real

Deliverables:

- introduce `SemanticGraph` and `ModelAdapter` interfaces
- move adapter-specific suffix matching out of layout lowering
- make current Qwen/Phi/Gemma/GPT-OSS/Mixtral logic emit semantic groups
- keep existing `LayoutPlan` lowering working through a compatibility builder

Acceptance criteria:

- lowering no longer reparses runtime suffixes to determine tensor roles
- each claimed model family has a semantic graph snapshot test
- runtime tensor names are selected only by `RuntimeABI`

### Phase 2: Replace Linear Layout Ops With Algebraic Expressions

Deliverables:

- introduce algebraic `ExprId` DAG representation
- add expression payloads for `Source`, `Select`, `Partition`, `Join`, `Stack`,
  `Reorder`, `View`, `Cast`, `Encode`, `Decode`, `Attach`, `Realize`
- implement type inference for every expression
- express existing direct-copy and packing paths using the DAG

Acceptance criteria:

- no transformer-specific op is required in the core algebra
- current model plan dumps show algebra roots per runtime tensor
- invalid shape/dtype/layout plans fail before storage lowering

### Phase 3: Add Optimizer Pass Infrastructure

Deliverables:

- pass manager over layout DAG
- before/after plan dumps for every pass
- selection and partition pushdown
- cast sinking
- decode/tile fusion marking
- view/alias canonicalization
- dead expression elimination

Acceptance criteria:

- optimizer can be disabled for debugging
- optimized and unoptimized plans produce identical golden tensors
- plan snapshots prove expected rewrites fire for representative models

### Phase 4: Make StorageProgram the Sole Executable Artifact

Deliverables:

- lower every supported algebra op into `StorageInstr`s
- remove direct layout-op interpretation from `LoadExecutor`
- represent views, metadata, lifetime, and finalization explicitly
- make unsupported algebra fail compilation

Acceptance criteria:

- `LoadExecutor` switches only on `StorageInstrKind`
- no post-load transform path exists outside `StorageProgram`
- storage validation proves full coverage for every `Realize`

Progress:

- Done: the executor's top-level compatibility switch has been removed.
- Done: extent writes and fused raw-read-plus-cast are scheduled storage
  instructions.
- Done: GPT-OSS MXFP4 plans compile through `StorageProgram` without
  compatibility metadata.
- Done: transform payloads are represented as typed storage/algebra payloads.

### Phase 5: Generalize TileMap

Deliverables:

- extend `TileMap` from cast/dequantize to generic tiled transforms
- add transform descriptors for `Decode`, `Encode`, `Transcode`, `Reblock`,
  `Reorder`, and `Cast`
- add backend dispatch table for transform kernels
- keep MXFP4 native handling behind backend capability declarations

Acceptance criteria:

- MXFP4 routed-dequant and eager-BF16 are selected by target policy
- future native MXFP4 is an ABI/backend option, not a model-loader fork
- GGUF block decode can use the same tile-map mechanism

Progress:

- Done: `Encode` and `Transcode` are first-class algebra/storage tile-map
  kinds.
- Done: optimizer tests cover the five proposal rewrites: selection pushdown,
  encode hoisting, decode fusion, cast sinking, partition-join cancellation,
  and transcode fusion.
- Remaining: executor kernels/reference paths for real quantized transcode
  formats.

### Phase 6: Add GGUF as a First-Class Source

Deliverables:

- parse GGUF metadata and tensor directory
- expose GGUF tensors through `CheckpointSource`
- define GGUF encoding specs for initial small formats
- add model adapter mapping for supported GGUF model families
- lower GGUF quant tensors into generic `Decode` expressions

Acceptance criteria:

- no runtime code path checks whether the source was safetensors or GGUF
- GGUF plan dumps contain the same algebraic operators as safetensors plans
- golden tests compare decoded small fixtures against a reference

Progress:

- Done: dense GGUF tensors expose byte extents through `CheckpointSource`.
- Done: `gguf.q4_0` tensors expose logical shape plus physical block metadata.
- Done: a Q4_0 decoded block fixture verifies nibble order and fp16 scale
  interpretation.

### Phase 7: Evidence Package

Deliverables:

- semantic graph snapshot tests for each claimed architecture
- layout algebra golden tests
- optimizer rewrite tests
- storage program coverage tests
- memory high-water telemetry tests
- one-command benchmark script for Pie/vLLM/SGLang comparison

Acceptance criteria:

- every supported transform has a small exact expected-output fixture
- planned peak and measured CUDA high-water are reported together
- benchmarks include startup time, load memory, latency, and throughput

## Codebase Restructuring

Proposed source layout:

```text
driver/cuda/src/loader/
  checkpoint_source.hpp
  safetensors_source.hpp/.cpp
  gguf_source.hpp/.cpp

  semantic_graph.hpp/.cpp
  model_adapter.hpp/.cpp
  adapters/
    qwen_adapter.hpp/.cpp
    gemma_adapter.hpp/.cpp
    gpt_oss_adapter.hpp/.cpp
    mixtral_adapter.hpp/.cpp
    phi3_adapter.hpp/.cpp
    olmo_adapter.hpp/.cpp

  runtime_abi.hpp/.cpp
  backend_target.hpp/.cpp

  layout_expr.hpp/.cpp
  layout_typecheck.hpp/.cpp
  layout_planner.hpp/.cpp
  layout_optimizer.hpp/.cpp
  passes/
    selection_pushdown.hpp/.cpp
    cast_sinking.hpp/.cpp
    partition_join_cancel.hpp/.cpp
    view_canonicalize.hpp/.cpp

  storage_program.hpp/.cpp
  storage_compiler.hpp/.cpp
  storage_executor.hpp/.cpp
  load_executor.hpp/.cpp
```

The old `model_schema.cpp` should shrink until it disappears. Its current
responsibilities should move into adapters, planner, ABI, and compiler code.

## Paper-Scope Claims

The implementation should support these claims confidently:

- Weight loading can be modeled as compilation from source layout to runtime
  layout.
- A typed layout algebra catches shape, dtype, sharding, and encoding errors
  before materialization.
- Source formats and runtime ABIs compose through semantic symbols, reducing
  format-runtime integration work.
- Storage compilation enables direct-to-final writes, bounded tiled transforms,
  and explicit memory accounting.
- Optimizer rewrites have measurable impact on memory or load time for concrete
  model/source/runtime combinations.

Claims to avoid until proven:

- a global unique normal form for the full algebra
- universal Encode/Decode Galois connection across all quantization formats
- arbitrary cross-format transcoding without reference implementations
- O(N + M) as a theorem rather than an empirical extensibility result

## Must-Do Before Calling It Paper-Grade

- explicit model adapters for every claimed architecture
- algebraic expression DAG with type inference
- runtime ABI ownership of final layouts
- storage compiler as the only lowering path to execution
- executor that runs only storage instructions
- golden tests for all algebra ops and transforms
- memory high-water telemetry tied to storage program estimates
- benchmark evidence against vLLM and SGLang
- GGUF support through `CheckpointSource` and encoding specs

## Nice-To-Have

- cost-based optimization
- GDS byte source
- remote object-store byte source
- native MXFP4 expert GEMM backend
- production cross-format quantized transcoding kernels and references
- formal proof of canonicalization for selected algebra fragments

## Design Summary

The clean design is:

```text
CheckpointSource
  gives raw tensors and byte extents.

ModelAdapter
  gives semantic identity.

RuntimeABI + BackendTarget
  gives final GPU tensor contracts.

LayoutPlanner
  builds typed algebraic expressions.

LayoutOptimizer
  rewrites expressions under explicit laws.

StorageCompiler
  lowers expressions to byte writes, tile maps, views, metadata, and lifetimes.

LoadExecutor
  executes the storage program and finalizes WeightStore.
```

The main refactor is replacing "loader ops that know model details" with
"typed algebra that model adapters and runtime ABIs compile through."
