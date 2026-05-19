# Pie Model Loading System

## Abstract

Pie treats model loading as **layout compilation under a memory budget**.
The loader is not merely a checkpoint reader. It compiles checkpoint tensor
layouts into the runtime tensor layout consumed by CUDA kernels.

This document captures the north-star design for Pie's CUDA model loading
system and tracks implementation progress. It is intended to serve two
purposes:

1. Provide a paper-quality system description for future writing.
2. Keep implementation work aligned with a simple, extensible architecture.

The central design principle is:

> Separate semantic identity, runtime representation, and storage
> materialization.

This yields the following simplified north-star architecture:

```text
CheckpointSource
  -> ModelAdapter
  -> RuntimeABI / BackendTarget
  -> LoadCompiler
  -> LayoutPlan
  -> StorageProgram
  -> LoadExecutor
  -> WeightStore / LoadedModel
  -> BoundCudaModel
```

The implementation still uses transitional names such as
`SafetensorsCheckpointSource`, `ModelSchema`, `LayoutPlan`, `StorageProgram`, and
`LoadExecutor`. The refactor target is not to add more layers, but to collapse
the design around three contracts:

```text
Source describes bytes.
Adapter describes meaning.
ABI describes final GPU layout.
```

The compiler connects those contracts in two algebraic forms:

```text
LayoutPlan: tensor layout algebra over semantic tensors.
StorageProgram: executable storage algebra over byte extents and tiles.
```

`StorageProgram` is the only executable artifact.

## Motivation

Large language model checkpoints are not stored in the layout that inference
kernels want to consume. A checkpoint is optimized for portability and training
framework conventions. The runtime wants different properties:

- tensor parallel shards instead of full tensors
- packed QKV and packed MLP gate/up projections
- fused MoE expert tensors
- quantized runtime formats
- architecture-specific compatibility views
- bounded temporary memory during startup

A naive loader materializes checkpoint tensors first and transforms them later.
That creates high peak memory usage. For large models, the transient extra
copies can be the difference between successful startup and OOM. Qwen3-32B was
the motivating example: loading raw projection weights and then fusing them
created avoidable memory pressure. The correct strategy is to plan final
runtime tensors first, then stream checkpoint bytes into final or near-final
storage.

The loader therefore behaves like a compiler:

```text
checkpoint program -> runtime tensor program -> scheduled storage execution
```

## Design Principles

### 1. Checkpoint Storage Is Separate From Meaning

A `CheckpointSource` declares what raw tensors and byte ranges exist. It owns:

- file/container metadata
- raw tensor names
- raw dtype or encoded dtype
- raw shape
- storage offsets and extents
- format-specific encodings such as safetensors, GGUF, or future containers

It does not decide whether Q/K/V should be packed, whether a tensor should be
runtime-quantized, or what CUDA kernels will consume.

### 2. Semantic Identity Is Separate From Storage

A `ModelAdapter` declares what tensors mean:

```text
Attention.Q
Attention.K
Attention.V
Attention.O
MLP.Gate
MLP.Up
MLP.Down
MoE.Expert.Gate
MoE.Expert.Up
MoE.Expert.Down
Embedding
LMHead
Norm
```

The adapter does not perform CUDA allocation and does not copy checkpoint
bytes. It maps raw checkpoint tensors into semantic tensor IDs and semantic
groups, validates architecture invariants, and records checkpoint-side
encodings.

### 3. Runtime Representation Is Owned By RuntimeABI

The final GPU layout is not checkpoint-owned and should not be invented by
schema lowering. It is the backend ABI.

`RuntimeABI` declares:

- final runtime tensor IDs and names
- final dtype, shape, layout, alignment, and view relationships
- backend quantized representations
- legal aliases and compatibility views
- the exact contract consumed by binders and kernels

This boundary matters because checkpoint layouts change slowly, while runtime
layouts evolve with kernels, devices, tensor parallel policies, and quantized
backends.

### 4. Runtime Representation Is Explicit

Runtime tensors are represented by typed specifications:

```text
name
shape
dtype
layout
ownership
parallelism
quantization format
quantization metadata
backing tensor, when this tensor is a view or alias
```

The runtime representation is not implicit in scattered string suffix checks.
It is encoded in the load IR and validated before materialization.

### 5. Storage Materialization Owns Lifetime

The `LoadExecutor` is responsible for:

- allocation
- streaming reads
- direct-to-final copies
- temporary lifetime
- aliases and views
- quant metadata ownership
- peak memory accounting

No later binding stage should need to ask how a tensor was loaded. It should
only consume a validated runtime tensor store.

## System Overview

### CheckpointSource

`CheckpointSource` owns checkpoint metadata and byte movement.

Its intended responsibility is narrow:

- parse container metadata
- expose tensor dtype, shape, file offsets, and shard location
- copy full tensors or slices to device memory
- provide generic read primitives for the executor

It should not know about Qwen, Gemma, MoE, quant policy, packed QKV, or CUDA
kernel binding.

Current implementation:

```text
SafetensorsCheckpointSource        -> initial SafetensorsCheckpointSource
MmapByteSource/GdsSource -> storage byte-source implementations
```

Planned source implementations:

```text
SafetensorsCheckpointSource
GgufCheckpointSource
```

The repository resolver that feeds `SafetensorsCheckpointSource` should also become
artifact-selective. Runtime loading needs `config.json`, tokenizer artifacts,
`model.safetensors`, and/or `model.safetensors.index.json` plus referenced
safetensors shards. It should not fetch alternate checkpoint formats such as
`.pt` or `.bin` files when safetensors are available.

### ModelAdapter

`ModelAdapter` is the architecture and checkpoint-dialect semantic
description. It maps source tensors into architecture-owned semantic IDs.

It declares:

- model family
- logical tensor roles
- checkpoint naming aliases
- checkpoint dialect, such as HF safetensors naming or GGUF naming
- required and optional tensors
- shape constraints
- tensor parallel rules
- legal semantic groups
- quantization eligibility
- tied or aliased tensors
- multimodal prefix strip and skip behavior

Examples:

- dense Llama-like models prefer packed QKV and packed gate/up layouts
- Phi-3 has fused checkpoint tensors that may need logical row-range splits
- Qwen MoE models require fused expert runtime tensors
- compressed-tensors checkpoints require quant metadata normalization
- GPT-OSS MXFP4 checkpoints declare packed E2M1 blocks, E8M0 scales, expert
  bias tensors, and the legal runtime representations for those tensors

The adapter may declare that Q/K/V form a semantic group. It should not decide
that the final CUDA runtime tensor is named `qkv_proj.weight`. That final name
and layout belong to `RuntimeABI`.

### RuntimeABI and BackendTarget

`RuntimeABI` is the source of truth for final GPU tensor layout. It declares
what binders and kernels consume.

Examples:

```text
SeparateQKV
PackedQKVRows
PackedGateUpRows
FusedExpertBank
NativePackedMXFP4Experts
RoutedMXFP4Dequant
MarlinInt4Packed
RuntimeFp8PerChannel
```

`BackendTarget` describes the concrete deployment target:

- device capabilities
- tensor parallel topology
- enabled kernel backends
- runtime quant policy
- scratch and memory budget
- chosen MXFP4, FP8, INT4, and MoE policies

The layout compiler chooses a representation that is legal for both the
semantic model and the runtime ABI.

### Layout Algebra

The layout algebra describes tensor transformations independent of model
family. There are two levels:

- a `LayoutPlan` that records semantic tensor values and selected runtime
  representations
- a `StorageProgram` that realizes that layout as storage instructions

The layout plan is allowed to be target-dependent. For example, GPT-OSS MXFP4
can lower to native packed MXFP4 tensors on a target with a registered native
MoE backend, or to BF16 expert tensors on a target without that backend. The
operation vocabulary remains generic in both cases.

The current implementation has an operational IR:

```text
Read / Copy
Slice
Shard / RowRangeShard
GroupedSliceConcat / GroupedSlice
Cast
Concat / AxisConcat
View
Alias
Drop
QuantizeRuntime
Dequantize
Deinterleave
RepackLayout
AttachMetadata
StackGroups
Materialize
```

The north-star `LayoutExpr` vocabulary should be algebraic but concrete:

```text
Source
Select
Partition
Join(axis)
Stack(axis)
View
Alias
Cast
Encode
Decode
Unzip
Reorder
Attach
Release
Realize
```

Architecture-specific names such as `QKV`, `GateUp`, or `MoE` exist only as
adapter semantics and diagnostics. The layout algebra uses generic terms such
as `Select`, `Partition`, `Join`, `Stack`, `Unzip`, `Reorder`, and `Attach`.

### StorageProgram

`StorageProgram` is the executable storage program. It is the only artifact
the executor should interpret.

The current implementation calls this layer `StorageProgram`. It lowers
copy-expressible logical ops into:

```text
ExtentWrite {
  raw checkpoint tensor
  optional rectangular slices
  destination runtime tensor
  destination byte offset
  compact destination shape
  source path, shard, byte offset, and byte span
  exact byte count
  exact source range count
}
```

The north-star instruction vocabulary is:

```text
Allocate
ExtentWrite
TileMap
CreateView
Attach
Release
Finalize
```

Non-copy operations lower to:

```text
TileMap {
  kind = Cast | Dequantize
  inputs
  output
  tile byte budget
  scratch byte estimate
}
```

The storage compiler also emits an explicit dependency-checked schedule:

```text
StorageInstr {
  kind = ExtentWrite | TileMap | DeviceMap | Attach | View | Release
  layout op index
  extent-write or tile-map references
  inputs
  outputs
  dependencies
}
```

Ready `ExtentWrite` steps are scheduled by checkpoint file order
(`source_path`, `source_shard_id`, `source_offset_bytes`) to reduce avoidable
seek/page-cache churn without changing semantic dependencies. The schedule is
validated statically: every extent write appears exactly once, every tile map
appears exactly once, and every dependency must precede its consumer.

The same storage program can execute with different byte sources:

```text
MmapByteSource: mmap + cudaMemcpy/cudaMemcpy2D into final VRAM
GdsByteSource: cuFileRead into final VRAM for direct contiguous writes
```

Storage writes run through a `StorageWriteExecutor` abstraction. The default
sync executor preserves the byte-source contract; the mmap path can use a
pipelined executor that issues async CUDA copies over a small stream pool for
ready writes. Transform fusion remains represented as `TileMap` rather
than hidden post-load mutation.

The production CUDA loader now enters through the Rust storage program only.
The old C++ byte-source executor is no longer linked into the driver. GDS
support should re-enter as a Rust `ByteSource` backend after the storage-source
interface is stable; it must not be exposed as a hidden fallback path.

### LoadCompiler

The `LoadCompiler` lowers source bytes, adapter semantics, and runtime ABI
requirements into a `LayoutPlan` and executable `StorageProgram`.

Logical passes:

```text
SourceBindingPass
  raw source tensors -> semantic tensors

AdapterValidationPass
  semantic tensors -> validated model graph

RuntimeABIPass
  semantic tensors -> final runtime tensors

TargetSelectionPass
  device, CUDA, kernel registry, memory budget -> target profile

RepresentationSelectionPass
  legal runtime representations -> chosen representation

TensorParallelPass
  insert shard and slice operations

QuantizationPass
  insert runtime quant, offline quant repack, dequant, and metadata ops

DTypeNormalizationPass
  convert remaining fp16/fp32 runtime tensors to bf16

FusionPass
  insert packed QKV, packed gate/up, and fused expert layouts

MemoryPlanningPass
  order operations and annotate temporary lifetimes

StorageCompilationPass
  runtime layout expressions -> Allocate/ExtentWrite/TileMap/View/Release
```

The compiler must estimate:

- persistent bytes
- maximum temporary bytes
- estimated peak bytes
- scratch requirements

### LoadExecutor

The `LoadExecutor` executes `StorageProgram`.

Rules:

- allocate final runtime buffers directly when possible
- stream checkpoint tensors into final storage
- avoid keeping checkpoint-only tensors alive
- expose compatibility names as views instead of copies
- release temporaries at last use
- attach quant metadata only after referenced tensors exist
- validate actual materialization against planned tensor specs

The executor should not know that a tensor is QKV, MoE, Phi-3, GPT-OSS, GGUF,
or safetensors. Those decisions must already be compiled into instructions.

For dense packed QKV, the desired compiled storage program is:

```text
Allocate qkv_final
ExtentWrite q shard -> qkv_final rows
ExtentWrite k shard -> qkv_final rows
ExtentWrite v shard -> qkv_final rows
CreateView q -> qkv_final
CreateView k -> qkv_final
CreateView v -> qkv_final
```

This avoids:

```text
load q persistent
load k persistent
load v persistent
allocate qkv
copy q/k/v into qkv
keep all raw tensors alive until later
```

### WeightStore and BoundCudaModel

`WeightStore` is the validated materialized artifact. Its contract is:

- all final runtime tensors are present
- non-owning views have valid owners
- quantized tensors have complete quant metadata
- tensor parallel shards have checked shapes
- unsupported dtype/layout combinations are rejected before binding

`BoundCudaModel` should consume only the final store. It should not perform
checkpoint interpretation or late layout synthesis.

## Tensor and Quantization Model

### TensorDecl

A runtime tensor spec contains:

```text
name
dtype
shape
layout
ownership
parallelism
quant spec
backing tensor
```

Important layout kinds:

```text
Dense
AxisConcatenated
Grouped
QuantPacked
View
```

Important ownership kinds:

```text
Owned
BorrowedView
Alias
Temporary
```

Important parallel kinds:

```text
Replicated
Column
Row
Expert
Custom
```

### QuantSpec

Quantization is a runtime layout property, not a side effect.

Supported and planned formats:

```text
None
RuntimeFp8E4M3
RuntimeInt8
GptqInt4
AwqInt4
CompressedFp8E4M3
CompressedInt8
Mxfp4E2M1E8M0
```

Supported granularity:

```text
PerTensor
PerChannel
PerGroup
```

Quantized tensors must specify:

- quant format
- scale tensor
- optional zero-point tensor
- group size, when per-group
- channel axis

The loader must guarantee that quant metadata is present before GEMM dispatch
can observe the weight.

## Invariants

The design relies on explicit invariants:

- Every final runtime tensor is produced by the plan.
- Every runtime tensor has a complete `TensorDecl`.
- Every view has a valid backing tensor.
- No view outlives its backing tensor.
- Every quantized tensor has matching quant metadata.
- Tensor parallel shards are shape-checked before allocation.
- Checkpoint-only tensors do not become persistent runtime tensors.
- The load executor does not allocate unplanned persistent tensors.
- `BoundCudaModel` consumes only validated runtime tensors.

Additional ownership invariants:

- `CheckpointSource` owns raw tensor names and byte extents.
- `ModelAdapter` owns semantic tensor IDs and groups.
- `RuntimeABI` owns final runtime tensor IDs, names, layouts, and binder ABI.
- `StorageProgram` is the only executable representation.
- `LoadExecutor` does not infer model semantics or final layout.

These invariants make extension easier: new checkpoint formats add sources,
new model/dialect variants add adapters, and new CUDA layouts add runtime ABI
representations plus compiler lowerings.

## Quantization Strategy

Quantization should be represented in the same IR as packing and sharding.

### Runtime FP8 and INT8

Runtime quantization lowers from BF16 runtime tensors to quantized runtime
tensors:

```text
Read/Materialize BF16 source
compute per-row absmax
all-reduce row-parallel scales when tensor parallelism requires it
cast to FP8 or INT8
attach scale metadata
drop BF16 source
```

The quantized tensor replaces the original runtime tensor name.

### GPTQ and AWQ

Offline INT4 checkpoints should lower to either:

- `RepackLayout` into Marlin-compatible storage, or
- `Dequantize` into BF16 when correctness or kernel support requires eager
  fallback.

The choice must be explicit in the plan.

### Compressed-Tensors FP8 and INT8

Compressed-tensors checkpoints lower by:

- loading quantized weights
- normalizing scale tensor dtype and shape
- registering canonical scale metadata
- attaching quant metadata to the quantized weight
- optionally dequantizing eagerly when native kernels are unavailable

### MXFP4

MXFP4 is represented as a quantized runtime format, not as an instruction to
always dequantize. The model adapter declares the checkpoint facts:

```text
checkpoint_format = MXFP4_E2M1_E8M0
block_size = 32
packed_data = uint8 nibbles
scale_data = uint8 E8M0 block scales
gate_up_layout = interleaved
legal_representations = [BackendNativePackedMoe, RoutedDequant, DenseBf16Fallback]
```

The target profile chooses one executable lowering.

Packed MXFP4 MoE runtime lowering:

```text
Copy packed blocks into final QuantPacked expert tensor
Copy E8M0 block scales into final scale tensor
Shard packed rows or block groups when tensor parallelism requires it
RepackLayout to backend layout when the selected kernel requires a swizzle
AttachMetadata
Expose final packed expert tensors and fused expert biases
```

This lowering keeps MXFP4 as the final runtime representation. Backends are
pluggable capabilities: the default CUDA native backend dequantizes only the
routed experts into bounded runtime scratch before invoking existing BF16 GEMMs,
while a FlashInfer/TRT-LLM, CUTLASS, or Marlin backend can consume the same
packed artifact with true native MXFP4 MoE kernels. Pie should not maintain a
naive in-tree MXFP4 GEMM; native execution is an adapter over a proven backend.

BF16 fallback lowering:

```text
Copy packed blocks
Copy E8M0 block scales
Dequantize MXFP4 blocks to temporary BF16 tensor
Deinterleave gate/up rows into final expert tensors
Slice tensor-parallel bands when needed
Drop packed blocks, scales, and temporary BF16 tensors
```

GPT-OSS uses these lowerings for expert MLP weights and biases. Packed MXFP4
is now the default representation for the CUDA native driver: `bind_gpt_oss`
consumes `QuantPacked` expert tensors and the Mixtral-style MoE backend
dequantizes routed experts into fixed BF16 scratch buffers at dispatch time.
`[model].mxfp4_moe = "bf16"` or `"eager_bf16"` forces the eager load-time BF16
fallback when comparing representation and memory behavior; `"native"` requires
a registered backend such as FlashInfer/TRT-LLM, CUTLASS, or Marlin.

### FP8 Fallback

On devices without native FP8 GEMM support, the plan may replace FP8 weights
with BF16 weights during load. This follows the same target-aware rule as
MXFP4: fallback is selected by lowering, not discovered late by the binder.

## Extensibility

### Adding a Model Family

To add a model family:

1. Add or extend a `ModelAdapter`.
2. Map raw source tensors into semantic tensor IDs and semantic groups.
3. Declare shape constraints, optional tensors, and model invariants.
4. Declare tensor parallel rules as semantic sharding constraints.
5. Add plan tests using synthetic `CheckpointSource` metadata.
6. Bind the resulting runtime ABI tensors in the model-specific binder.

The `LoadExecutor` should not need model-family-specific branches unless the
family requires a genuinely new reusable storage instruction.

### Adding a Checkpoint Format

To add a checkpoint format such as GGUF:

1. Add a `CheckpointSource` implementation that can list raw tensors, expose
   metadata, and read byte extents.
2. Add format-specific raw encodings and quant codecs as checkpoint-side
   properties.
3. Extend the relevant `ModelAdapter` to map the format's naming dialect into
   the same semantic tensor IDs used by safetensors.
4. Reuse the existing `RuntimeABI` and `LoadCompiler` lowerings whenever the
   semantic graph is equivalent.
5. Add source-level metadata tests, layout-plan snapshot tests, and at least
   one startup smoke.

GGUF support should not create a GGUF-specific CUDA binder path. If the same
semantic model is loaded from safetensors or GGUF, both should compile into the
same runtime ABI when the selected backend representation is supported.

### Adding a Quantization Format

To add a quantization format:

1. Add a `QuantFormat` value.
2. Add validation rules for metadata and supported tensor layouts.
3. Add lowering rules from checkpoint representation to runtime
   representation.
4. Add load executor handlers for dequant, repack, or runtime quantization.
5. Add GEMM dispatch support or an eager fallback.
6. Add plan tests and at least one startup smoke.

### Adding a Runtime Layout

To add a runtime layout:

1. Add a `RuntimeABI` representation only if existing generic layout
   descriptors cannot express the layout clearly.
2. Add backend capability and policy selection in `BackendTarget`.
3. Add validation rules for shape, dtype, alignment, sharding, and quant
   metadata.
4. Add lowering from semantic tensors to runtime tensor declarations.
5. Add storage compiler support that emits `StorageProgram` instructions.
6. Add binder support for the new ABI representation.

Prefer generic operations over model-specific op names.

## Simplified North-Star Refactor Plan

The refactor should be staged and behavior-preserving. The target is a simpler
architecture, not a larger framework.

### Algebraic IR Naming

The north-star names should sound like tensor and storage algebra, while
remaining concrete enough for maintainable C++:

```text
Current name             North-star name
------------------------------------------------
LayoutPlan                 LayoutPlan
LayoutExpr                   LayoutExpr
TensorDecl               TensorDecl
StorageProgram         StorageProgram
StorageInstr     StorageInstr
ExtentWrite           ExtentWrite
TileMap           TileMap
LoadExecutor             LoadExecutor
WeightStoreBuilder       MutableWeightStoreBuilder
```

Operation vocabulary:

```text
Read / Copy              Source / Realize
Slice                    Select
Shard                    Partition
RowRangeShard            Select + Partition
Concat / AxisConcat      Join(axis)
StackGroups              Stack(axis)
GroupedSlice             SelectGroup
GroupedSliceConcat       SelectGroup + Join(axis)
QuantizeRuntime          Encode
Dequantize               Decode
Deinterleave             Unzip
RepackLayout             Reorder
AttachMetadata             Attach
Drop                     Release
Materialize              Realize
```

The storage instruction vocabulary is smaller:

```text
Allocate
ExtentWrite
TileMap
CreateView
Attach
Release
Finalize
```

### Phase 1: Name the Boundaries

- Introduce `CheckpointSource`, `ModelAdapter`, `RuntimeABI`, `LayoutPlan`,
  `StorageProgram`, and `LoadExecutor` types as wrappers or aliases around
  the current implementation.
- Keep `SafetensorsCheckpointSource`, `LayoutPlan`, `StorageProgram`, and
  `LoadExecutor` working while the new names are introduced.
- Update dumps and tests to report both layout algebra and executable storage
  program concepts clearly.

### Phase 2: Move Final Layout Ownership Into RuntimeABI

- Move final runtime tensor names, packed layouts, aliases, quantized runtime
  representations, and binder ABI declarations out of schema lowering.
- Make adapters emit semantic IDs and groups only.
- Make the runtime ABI declare final tensor IDs such as packed QKV, packed
  gate/up, fused expert banks, native MXFP4 expert tensors, Marlin INT4
  tensors, and compatibility views.
- Make binders consume runtime ABI declarations instead of relying on
  checkpoint-derived names.

### Phase 3: Split Source Format From Model Semantics

- Rename the safetensors path toward `SafetensorsCheckpointSource`.
- Add a metadata-only `CheckpointSource` test implementation that is not tied
  to safetensors names.
- Add `GgufCheckpointSource` for GGUF metadata and byte extents.
- Add GGUF naming mappings inside the relevant model adapters, not inside the
  executor or CUDA binders.

### Phase 4: Make StorageProgram the Only Executable IR

- Compile all copy-like work into `Allocate`, `ExtentWrite`, `CreateView`, and
  `Release` instructions.
- Compile all casts, dequantization, repacks, and runtime quantization into
  typed `TileMap` instructions with explicit scratch/tile budgets.
- Move remaining layout-op interpretation out of the load executor.
- Make `LoadExecutor` execute only `StorageProgram` instructions.

### Phase 5: Strengthen Static Validation

- Validate that every semantic tensor required by the selected runtime ABI has
  exactly one producer.
- Validate that every executable instruction is covered by source extents,
  transform kernels, metadata binding, or lifetime actions.
- Validate view/backing lifetime before execution.
- Validate planned peak memory against instruction-level allocations,
  transform scratch, and backend runtime scratch.

### Phase 6: Optimize Behind the Same Program Interface

- Keep file-order scheduling and adjacent extent coalescing as compiler passes.
- Add stream assignment and read-copy pipelining behind the executor interface.
- Add source-read plus cast/dequant fusion by producing different
  `TileMap` instructions, not by adding binder or model-specific
  fallbacks.
- Add GDS alignment-aware batching only after the source/executor contract is
  stable.

### Phase 7: Evidence

- Snapshot `ModelAdapter` semantic graphs for every claimed architecture and
  checkpoint dialect.
- Snapshot `RuntimeABI` declarations for each backend target profile.
- Snapshot final `StorageProgram`s and prove no hidden post-load transform path.
- Keep golden transform tests for exact tensors.
- Continue reporting planned versus actual CUDA memory high-water and startup
  load time.

## Current Implementation Progress

Last updated: 2026-05-17.

### Implemented

- `SafetensorsCheckpointSource` parses checkpoint metadata and copies full tensors or
  generic contiguous/strided tensor slices into caller-owned device memory. It
  has no allocation-returning API and no MoE-specific copy entry points; all
  allocation, layout, and lifetime decisions live in the load executor.
- `CheckpointSource`, `LayoutPlan`, `TensorDecl`, `LayoutExpr`,
  `StorageProgram`, `ExtentWrite`, `TileMap`, `StorageInstr`, and
  `LoadExecutor` are present as the north-star boundary names while the older
  implementation names remain source-compatible during the staged refactor.
- `RuntimeABI` now owns the final CUDA-facing names and layout declarations for
  packed projection storage, compatibility views, fused expert banks, and
  canonical quant scale companion tensors. Schema lowering asks the ABI for
  those declarations instead of constructing final runtime names inline.
- `GgufCheckpointSource` parses GGUF v2/v3 metadata, dense tensor dtypes,
  shapes, and absolute byte extents into the same `CheckpointSource` contract as
  safetensors. Unsupported GGUF quant block types fail explicitly until a GGUF
  quant dialect adapter lowers them into the layout algebra.
- `ModelSchema` exists as a first explicit schema resolution layer.
- `SemanticGraph` is a distinct checkpoint-binding stage that records raw
  checkpoint names, runtime names, adapter-declared semantic roles, dtype,
  shape, and tensor-parallel shard axes before layout-plan lowering.
  It also records logical tensor groups such as packed QKV, packed gate/up,
  row-range splits, per-expert MoE, fused MoE, GPT-OSS MXFP4, and FP8+scale
  pairs. Groups carry source and runtime roles, so lowering consumes declared
  group structure instead of reparsing QKV/GateUp/Phi-3/GPT-OSS suffixes.
- `LayoutPlan` contains typed runtime tensor specs.
- `LayoutExpr` uses typed payload families instead of a flat overloaded field bag.
  Constructors build raw-load, row-range, tensor, slice, axis-concat, and
  stack-groups payloads; validation and materialization consume those payloads via
  typed accessors.
- `LayoutPlan` contains explicit executable operations for:
  - GPTQ/AWQ dequantization and Marlin repack
  - compressed-tensors FP8 and INT8 metadata registration
  - runtime FP8/INT8 quantization
  - eager FP8 and MXFP4 BF16 fallbacks
  - packed MXFP4 runtime loading
  - Qwen per-expert MoE fusion
- The core IR uses generic operation names in C++ types, dumps, and
  diagnostics: `AxisConcat`, `GroupedSliceConcat`, `GroupedSlice`,
  `StackGroups`, `Deinterleave`, `RepackLayout`, and `AttachMetadata`.
  Architecture-specific terms such as QKV, gate/up, and MoE remain schema
  semantics, not executable IR terms.
- `LayoutPlan` tracks persistent, temporary, and estimated peak memory.
- Dense Llama-like Qwen/Qwen2/Llama/Mistral/Olmo loading can plan packed QKV
  and packed gate/up tensors.
- Packed QKV and packed gate/up expose compatibility names as views.
- `StorageProgram` lowers copy-expressible algebra into exact `ExtentWrite`
  records with destination tensor, destination offset, compact slice shape,
  byte count, source range count, and concrete checkpoint source path/offset
  metadata.
- Plan dumps now identify the semantic layer as `LayoutPlan`/`LayoutExpr` and
  the executable storage layer as `StorageProgram` with `StorageInstr`,
  `ExtentWrite`, and `TileMap` records.
- `CheckpointByteSource` is the stable storage IO interface. `MmapByteSource`
  owns the mmap + CUDA-copy path; `GdsByteSource` dynamically loads libcufile
  and issues `cuFileRead` directly into planned destination allocations for
  contiguous full-tensor and leading-axis slice writes.
- `StorageProgram` now includes a dependency-checked storage schedule. Ready
  extent writes are ordered by checkpoint file location, and plan dumps expose
  `schedule`, `scheduled_extent_writes`, and source offsets for every write.
- `StorageProgram` records carry native algebra identity (`expr_id`,
  `binding_index`). There is no compatibility coverage table or planner-op
  index in the executable storage plan.
- `LoadExecutor` consumes compiled storage extent writes when a
  `StorageProgram` is available; it no longer re-lowers an unoptimized byte
  mapping for execution.
- `LoadExecutor` no longer depends on compatibility planner op records.
  `StorageInstr` carries typed payloads for slice/view/axis metadata, and the
  executor derives its internal execution op from the compiled storage
  instruction stream.
- `StorageWriteExecutor` is the execution abstraction below the load executor.
  The sync executor works with any byte source; the mmap executor can issue
  async stream-pooled copies through the byte-source interface.
- The storage compiler validates the algebra-to-storage boundary directly.
  Copy-like algebra lowers to `ExtentWrite`; `Cast`, `Encode`, `Decode`, and
  `Transcode` lower to typed `TileMap`; view, lifetime, metadata, and
  device-transform algebra lower to typed `StorageInstr` records. Plan
  construction fails if unsupported algebra reaches storage compilation.
- The storage optimizer currently coalesces adjacent compatible extent writes
  and records optimized/coalesced write counts in plan dumps. This is the
  first optimizing compiler pass; the interface is ready for alignment-aware
  GDS batching and transform fusion passes.
- `Join(axis)` materialization streams sources into final storage rather than
  keeping all packed inputs alive at once.
- `Stack(axis)` can stream per-expert checkpoint rows directly into final
  fused expert tensors instead of staging checkpoint-shaped source tensors.
- Raw `Source`, `Select`, `Partition`, and `Join` expressions lower to generic
  storage extent writes and copy checkpoint bytes directly into final or
  planned temporary destinations.
  BF16 dense Qwen-style packed plans no longer create full checkpoint-shaped
  device buffers for projection tensors before packing.
- `Source -> Cast -> Release` sequences execute through a tiled raw-cast path:
  checkpoint bytes stream through bounded scratch and are cast directly into
  the final runtime tensor, avoiding a full source tensor in the `WeightStore`.
- FP8 and MXFP4 BF16 fallback dequantization execute the output transform in
  bounded tiles. Native MXFP4 remains a backend adapter target; Pie does not
  grow a custom in-tree MXFP4 GEMM.
- `WeightStore` stores `TensorRecord { spec, tensor }`, counts only owned
  resident tensors for memory accounting, and rejects erasing a backing tensor
  while a registered view or alias still depends on it.
- `LoadExecutor` executes layout plans into `WeightStore` without depending on
  `LoadedModel`.
- Materialization validates produced tensor dtype, shape, and view backings
  against the plan.
- The load executor supports generic `Read`, `Copy`, `Shard`, `RowRangeShard`,
  `GroupedSliceConcat`, `GroupedSlice`, `Slice`, `Cast`, `Concat`,
  `AxisConcat`, `View`, `Alias`, `Drop`, `QuantizeRuntime`, `Dequantize`,
  `Deinterleave`, `RepackLayout`, `AttachMetadata`, `StackGroups`, and
  `Materialize` operations.
- FP16/FP32 checkpoint tensors lower into scheduled `Copy -> Cast -> Drop`
  sequences when the runtime representation is BF16. The same producer helper
  covers normal copies, Phi-3 row-range shards, and Qwen MoE direct expert
  shard ops; canonical quant scale tensors that must remain FP32 are left
  untouched.
- `RepackLayout` is a first-class scheduled load executor operation for
  symmetric GPTQ int4 checkpoints that target Marlin W4A16 storage.
- AWQ int4 checkpoints and GPTQ int4 fallback variants, including GPTQ
  act-order, lower into scheduled
  `Copy(qweight/qzeros/scales[/g_idx]) -> Dequantize -> Drop` op sequences.
  The checkpoint-only int4 tensors are temporary `TensorDecl`s tagged with
  `AwqInt4` or `GptqInt4`; final runtime tensors are BF16 dense weights when
  no native runtime int4 representation is selected. Tensor-parallel
  act-order row shards keep full GPTQ scale/zero metadata and slice only
  qweight/g_idx, preserving the checkpoint's global group ids without an
  ad-hoc remap.
- Runtime `fp8` and `int8` quantization for supported dense projection
  weights lowers into explicit `Copy -> QuantizeRuntime -> AttachMetadata
  -> Drop` op sequences. Temporary BF16/FP16/FP32 source tensors are not
  registered as final runtime tensors.
- `QuantizeRuntime` materialization computes per-row absmax scales, performs
  TP all-reduce for row-parallel weights, writes quantized final tensors, and
  attaches scale metadata through the `WeightStore`.
- Compressed-tensors FP8 weights lower into scheduled load ops. Native-FP8
  hardware keeps FP8 weights and attaches canonical FP32 scale metadata; the
  non-native fallback lowers through scheduled `Dequantize`.
- Mistral-style FP8 weights with `weight_scale_inv` side tensors lower through
  scheduled `Copy(weight/scale) -> Dequantize -> Drop` into BF16 runtime
  weights; the binder no longer performs this fallback.
- Symmetric compressed-tensors INT8 weights lower into scheduled
  `Copy(weight) -> Copy/Cast(scale) -> AttachMetadata` plans. The final
  runtime tensor is `QuantPacked` INT8 with canonical scale metadata; variants
  that require zero-points or unsupported layouts remain rejected before
  materialization.
- `Dequantize` materialization supports compressed-tensors FP8 E4M3, AWQ
  int4, GPTQ int4, and GPT-OSS MXFP4 E2M1/E8M0 blockwise dequantization to
  BF16.
- Symmetric GPTQ int4 checkpoints with `desc_act=false` and no zero-points
  lower into scheduled
  `Copy(qweight/scales) -> RepackLayout -> AttachMetadata -> Drop`. The final
  runtime tensor is Marlin-packed `INT4_PACKED` with a BF16 canonical
  `weight_scale_inv` side tensor. Tensor-parallel plans slice local GPTQ
  qweight and scale temporaries before repacking: column-parallel projections
  shard GPTQ `N`, while row-parallel projections shard packed `K/8` and
  scale groups. If the target has no Marlin int4 backend, the same symmetric
  GPTQ checkpoint lowers to scheduled BF16 `Dequantize` when qzeros metadata is
  present, instead of emitting a plan that fails later in materialization.
- Cargo CUDA builds can opt into the same Marlin-backed path with
  `PIE_CUDA_BUILD_MARLIN=1`; the default remains off because the vendored
  template kernels are expensive to compile.
- AWQ int4 checkpoints now share the target-aware int4 architecture. On
  Marlin-capable targets they lower to scheduled
  `Copy(qweight/qzeros/scales) -> RepackLayout -> AttachMetadata -> Drop`
  plans that produce Marlin-packed INT4 runtime tensors with canonical BF16
  scale and INT32 zero-point side tensors. On non-Marlin targets they lower to
  scheduled BF16 `Dequantize`.
- The CUDA layout planner receives a `BackendTarget` profile instead of a raw
  `fp8_native` flag. The target carries FP8 capability, GPTQ Marlin int4
  availability, and the selected MXFP4 MoE representation.
- The common config path accepts flat `rope_parameters` as a RoPE config source
  in addition to `rope_scaling`, and serving maps the full `RopeScaling` enum
  into runtime `RopeKind`. This preserves Original YaRN for OLMo-3, GPT-OSS,
  and Ministral-3 instead of collapsing it to standard RoPE.
- GPT-OSS 20B MXFP4 expert tensors have two explicit lowerings. The default
  CUDA native lowering emits final `QuantPacked` MXFP4 expert tensors with
  E8M0 scale metadata and fused expert biases. The runtime MoE backend
  consumes those packed tensors and dequantizes only routed experts into
  bounded BF16 scratch before existing BF16 GEMMs. The eager load-time BF16
  fallback remains available through `[model].mxfp4_moe = "bf16"` or
  `"eager_bf16"`. True native MXFP4 is represented as a `BackendTarget`
  capability and should bind to FlashInfer/TRT-LLM, CUTLASS, or Marlin rather
  than an architecture-local custom GEMM.
- `PIE_CUDA_RUST_LAYOUT_PLAN_DUMP=/path/to/plan.json` writes a JSON plan artifact with
  algebra expressions, tensor specs, storage extent writes, tile maps, and
  semantic/storage memory estimates.
- GPT-OSS MXFP4 policy is expressed as target policy through
  `[model].mxfp4_moe`, not hidden env-var behavior.
- The planner computes persistent bytes from final owned tensors and computes
  semantic temporary high-water bytes from algebra lifetimes.
- The storage compiler computes resident temporary high-water from the actual
  load executor lifetime protocol, including `Drop` ops, raw-copy-to-cast
  fusion, and bounded transform scratch.
- The load executor samples CUDA memory during load through a device-allocation
  callback, and reports planned storage peak/temp versus actual CUDA
  high-water memory.
- Golden CUDA materialization tests cover exact `AxisConcat`,
  `GroupedSliceConcat`, `GroupedSlice`, `StackGroups`, `Cast`, `Dequantize`,
  and `AttachMetadata` behavior on small real safetensors fixtures.
- Synthetic architecture plan tests cover Qwen dense/MoE, Qwen3.6 dense/MoE,
  Gemma4 dense/MoE, GPT-OSS MXFP4, Mixtral, Ministral, Phi-3, and OLMo-3.
- `benches/run_loader_evidence.py` is the one-command evidence harness for
  loader tests plus Pie/vLLM/SGLang latency, throughput, startup, plan dumps,
  and Pie load-memory telemetry.
- Qwen3-32B starts successfully with the CUDA native driver.
- The BF16 CUDA native e2e smoke matrix covers dense Qwen3, Qwen3 MoE,
  Qwen3.6 dense, Qwen3.6 MoE, Gemma4 dense, and Gemma4 MoE checkpoints.
- `BoundCudaModel` centralizes architecture binding from the validated loaded
  model.
- Model binders consume `const LoadedModel&`; remaining checkpoint-layout
  synthesis has moved into layout-plan lowering/materialization.
- `bind_gpt_oss` consumes loader-materialized `QuantPacked` MXFP4 expert
  tensors when the packed runtime path is selected, and consumes BF16 expert
  tensors only for the explicit eager fallback path.
- HF snapshot resolution is runtime-artifact selective by default for both
  `serve` and `pie model download`: config/tokenizer artifacts plus
  `model*.safetensors` are fetched, while duplicate `.pt`, `.bin`, `.gguf`,
  and non-loader safetensors artifacts are skipped. `pie model download --all`
  remains available when a complete HF snapshot is explicitly needed.
- The Rust loader accepts a metadata-only tensor source and can synthesize the
  CUDA `RuntimeABI` directly from safetensors metadata plus `config.json`.
  CUDA no longer builds or translates the former C++ `LayoutPlan`.
- Rust is the production semantic-to-algebra migration target. The production
  loader now emits algebra directly from Rust schema/default ABI rules; the
  storage compiler lowers those plans without a compatibility op stream.
- Algebra-only packed row-group plans now emit typed `CreateView` schedule
  records for compatibility views. The executor can materialize those views
  from `LayoutExpr::View` metadata without consulting a planner op.
- Dense, quantized, MoE, Phi-3 row-split, GPT-OSS MXFP4, and packed Llama-like
  production plans now have no compatibility step stream.
- Rust unit tests cover algebra rewrites, FFI safety, semantic-role contracts,
  explicit `ByteSpans`, default CUDA ABI synthesis, tensor-parallel row/column
  sharding, packed quant byte alignment, storage lowering, and generated C++
  wrapper/header compatibility.

### Remaining Implementation Work

- Schema adaptation, runtime ABI decisions, and lowering now live in Rust.
  The next cleanup is to split `abi.rs` default-contract synthesis into
  per-family `schema.rs` modules plus reusable ABI builders.
- The IR is typed and broad enough for current layouts, but some less common
  checkpoint quantization sub-modes still need schema coverage before they can
  lower into scheduled algebra.
- Compressed-tensors FP8 and symmetric INT8 lowerings are scheduled for the
  variants Pie can bind today; asymmetric INT8, compressed INT4, and other
  compressed-tensors sub-modes still need schema and backend coverage.
- MoE fused expert loading has scheduled schema support for current Qwen
  per-expert checkpoint layouts and GPT-OSS packed MXFP4 layouts. Future MoE
  variants should reuse the same generic `StackGroups`, quant metadata, and
  backend-selection machinery rather than adding architecture-local transforms.
- Semantic memory planning tracks scheduled temporary tensors. Runtime backend
  scratch, such as GPT-OSS routed-expert MXFP4 dequant scratch, is tracked
  outside layout-plan temporary high-water marks and should be reported
  explicitly as runtime scratch.
- Native transform fusion is still conservative: the storage compiler records
  tiled `Cast`/`Dequantize` actions and the load executor executes bounded
  transform paths, but deeper source-read plus transform fusion is still a
  future optimizer pass.

### Not Yet Implemented

- True hardware-native MXFP4 MoE GEMM kernels. The current packed runtime
  backend keeps MXFP4 as the final loaded representation and dequantizes only
  routed experts into bounded BF16 scratch before GEMM.
- Future architecture adapters for checkpoint formats not yet in the local
  claim set.

## Test and Evaluation Plan

### Unit and Static Plan Tests

Required plan tests:

- Qwen3 dense BF16 emits packed QKV and packed gate/up plans.
- Qwen3-32B metadata does not produce persistent raw Q/K/V/gate/up tensors
  when packed layouts are selected.
- Phi-3 TP emits row-range shard operations for fused checkpoint tensors.
- Qwen MoE TP emits expert-aware sharding.
- Runtime `fp8` and `int8` emit `QuantizeRuntime` op sequences for supported
  dense projection weights.
- Symmetric GPTQ emits scheduled `RepackLayout` on Marlin-capable targets and
  scheduled BF16 `Dequantize` on non-Marlin targets with qzeros metadata. AWQ
  emits scheduled `RepackLayout` on Marlin-capable targets and scheduled BF16
  `Dequantize` on non-Marlin targets. GPTQ act-order TP keeps full scale/zero
  metadata for row-parallel shards so `g_idx` remains valid without rewriting.
- Compressed-tensors FP8 and symmetric INT8 emit scale normalization and quant
  metadata attachment.
- GPT-OSS MXFP4 emits target-dependent plans: packed-runtime targets produce
  `QuantPacked` MXFP4 tensors with attached scales; eager fallback targets
  produce `Dequantize` and `Deinterleave` operations with no persistent raw
  block/scale tensors.
- FP16/FP32 checkpoint tensors emit scheduled `Cast` ops, and quant scale
  tensors that must remain FP32 are not normalized.

### Runtime Smokes

Required runtime smokes:

- Qwen3-0.6B BF16 latency smoke.
- Qwen3-32B BF16 startup and short latency smoke.
- Runtime INT8 startup and short latency smoke.
- Runtime FP8 startup smoke on native FP8 hardware.
- Compressed-tensors FP8 startup smoke when a local checkpoint is available.
- GPT-OSS 20B MXFP4 startup smoke.
- MoE startup smoke for Qwen MoE variants when local checkpoints are available.
- Gemma4 MoE startup smoke when a local checkpoint is available.
- Ministral-3 3B startup smoke.
- Mixtral 8x7B startup smoke with a small KV budget.
- OLMo-3 7B startup smoke.
- Phi-3 mini startup smoke.

### Paper Metrics

Useful metrics for the system paper:

- peak GPU memory during load
- final persistent weight memory
- startup time
- number of materialized tensors
- number of compatibility views
- temporary memory high-water mark
- throughput and latency after loading
- comparison with naive load-then-transform strategy
- comparison with vLLM and SGLang startup memory for the same models

Latest verification:

- Rust-default CUDA ABI cutover on 2026-05-19:
  `cargo test --manifest-path driver/weight_loader/Cargo.toml`,
  `cmake -S driver/cuda -B driver/cuda/build`,
  `cmake --build driver/cuda/build --target pie_driver_cuda_lib -j2`,
  `cmake --build driver/cuda/build --target pie_driver_cuda -j2`,
  `ctest --test-dir driver/cuda/build --output-on-failure`, and
  `cmake --build driver/portable/build --target pie_driver_portable_lib -j2`
  pass. `cargo build -p pie-server --release --no-default-features --features driver-cuda`
  also links. The former CUDA C++ `LayoutPlan`/`model_schema`/`RuntimeABI` sources
  and `test_layout_plan` target have been removed from the build tree; CUDA now
  compiles safetensors metadata directly through the Rust default ABI, algebra,
  optimizer, and storage compiler.

- Storage schedule/executor implementation pass on 2026-05-17:
  `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -S driver/cuda -B /tmp/pie-cuda-loader-build -DCMAKE_BUILD_TYPE=Release`,
  `cmake --build /tmp/pie-cuda-loader-build --target test_layout_plan pie_driver_cuda_lib -j2`,
  and `/tmp/pie-cuda-loader-build/bin/test_layout_plan` pass. After building
  `test_brle` and `test_driver_common`,
  `ctest --test-dir /tmp/pie-cuda-loader-build --output-on-failure` passes
  3/3 tests. `cargo build -p pie-server --release --no-default-features --features driver-cuda`
  also passes. This covers
  source-offset extent writes, file-ordered storage schedules, static schedule
  validation, compiled-extent-write materialization plumbing, and CUDA library
  compilation of the Rust storage-program executor.
- North-star implementation pass on 2026-05-17:
  `cmake --build driver/cuda/build --target pie_driver_cuda test_layout_plan -j 8`,
  `ctest --test-dir driver/cuda/build --output-on-failure`,
  `cargo test -p pie-server cuda -- --nocapture`,
  `cargo build -p pie-server --release --no-default-features --features driver-cuda`,
  and `python3 -m py_compile benches/run_loader_evidence.py benches/pie_bench.py benches/sglang_bench.py benches/vllm_bench.py`
  all pass.
- One-command evidence:
  `benches/run_loader_evidence.py --model Qwen/Qwen3-32B --engines pie,vllm,sglang --modes latency,tput --requests 1 --num-requests 2 --concurrency 2 --max-tokens 8 --warmup 0 --max-model-len 512`.
  Artifact: `.tmp/loader_evidence/qwen32b-final/evidence.json`.
- Qwen3-32B Pie loader telemetry from that artifact:
  Rust storage program had 707 `ExtentWrite` records, 62,488 MiB
  checkpoint/device bytes, planned storage temp <= 0 MiB, planned peak ~=
  62,488 MiB, actual CUDA delta ~= 62,490 MiB, and free-memory high-water
  96,704 -> 34,214 -> 34,214 MiB
  across 1,032 samples.
- Qwen3-32B short benchmark from that artifact:
  Pie latency 493.0 ms p50 / 16.23 output tok/s and throughput 18.69 output
  tok/s; vLLM latency 445.9 ms p50 / 17.94 output tok/s and throughput
  31.09 output tok/s; SGLang latency 884.3 ms p50 / 9.05 output tok/s and
  throughput 16.11 output tok/s with Triton attention, PyTorch sampling, CUDA
  graphs disabled, `FLASHINFER_CUDA_ARCH_LIST=12.0a`, and
  `SGLANG_DISABLE_PDL=1`.
- Current implementation pass:
  `c++ -std=c++20 ... driver/cuda/tests/test_layout_plan.cpp ...` and
  `cargo build -p pie-server --release --no-default-features --features driver-cuda`.
  The synthetic plan tests now include typed `LayoutExpr` payload validation and
  scheduled FP8 `weight_scale_inv` dequant lowering.
- `cargo build -p pie-server --release --no-default-features --features driver-cuda`
- `PIE_CUDA_BUILD_MARLIN=1` path verified through
  `cmake --build .tmp/cuda-marlin-build --target pie_driver_cuda_lib -j2`
  before removing the temporary build tree.
- `cargo test -p pie-server hf:: --no-default-features --features driver-cuda`
- `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -S driver/cuda -B .tmp/cuda-layout-plan-tests -DCMAKE_BUILD_TYPE=Release`
- `cmake --build .tmp/cuda-layout-plan-tests --target test_layout_plan -j2`
- `.tmp/cuda-layout-plan-tests/bin/test_layout_plan`
- Synthetic layout-plan tests include scheduled GPTQ symmetric int4
  `RepackLayout` lowering with canonical BF16 scale metadata, tensor-parallel
  local qweight/scale slicing, and no `OfflineGptqAwq` transform.
- Synthetic layout-plan tests include scheduled AWQ int4, asymmetric GPTQ int4,
  and GPTQ act-order BF16 fallback lowerings, including tensor-parallel local
  qweight, qzeros, scale, and g_idx handling with no `OfflineGptqAwq`
  transform. They also cover target-dependent symmetric GPTQ and AWQ:
  Marlin-capable targets emit `RepackLayout`; non-Marlin targets emit scheduled
  BF16 `Dequantize` when the checkpoint has qzeros metadata.
- Current-turn verification: Qwen3-0.6B CUDA native one-token smoke completed
  1/1 with 98.9 ms mean latency. Dumped plan: 171 `Copy`, 56 `AxisConcat`,
  persistent 1,433 MiB, temp 6 MiB, and zero post-load transforms.
- Qwen3-32B direct CUDA native load after direct-to-final materialization:
  layout compiler emitted algebra-native bindings and 835 runtime tensor specs, persistent
  62,488 MiB, load-time temp <= 0 MiB, peak ~= 62,488 MiB, and 128
  `AxisConcat` groups. The model loaded 835 tensors in about 5 seconds on the
  RTX PRO 6000 Blackwell Server Edition.
- Qwen3-32B storage program dump after the byte-plan compiler:
  707 `ExtentWrite` records, 707 source ranges, 62,488 MiB checkpoint
  bytes read, 62,488 MiB device bytes written, storage temp <= 0 MiB, and
  storage peak ~= 62,488 MiB.
- Synthetic layout-plan tests include Qwen MoE TP expert sharding and GPT-OSS
  MXFP4 target selection: BF16 fallback emits `Decode` /
  `Unzip` / `Select` algebra, while packed-runtime targets emit
  `QuantPacked` MXFP4 tensors with attached scale metadata.
- Fresh-cache selective download smoke:
  `HF_HOME=.tmp/hf-selective-qwen pie model download Qwen/Qwen3-0.6B`
  fetched only config/tokenizer artifacts plus `model.safetensors`.
- Qwen3-0.6B CUDA native one-token smoke from that selective cache.
- Qwen3-0.6B BF16 latency smoke.
- RedHatAI/Qwen3-0.6B-FP8-dynamic compressed-tensors FP8 latency smoke on
  native-FP8 hardware.
- openai/gpt-oss-20b MXFP4 latency smoke.
- Qwen3-32B BF16 short latency smoke.
- Qwen3-0.6B INT8 runtime-quant latency smoke.
- Qwen3-32B INT8 runtime-quant short latency smoke.
- Qwen3-0.6B FP8 runtime-quant latency smoke on native-FP8 hardware.
- Dumped Qwen3-0.6B INT8 plan: 196 `QuantizeRuntime` ops, 196
  `AttachMetadata` ops, 196 `Drop` ops, 196 quantized runtime tensors, and no
  `RuntimeQuantize` transform.
- Dumped RedHatAI/Qwen3-0.6B-FP8-dynamic plan: 196 compressed-FP8 E4M3
  quantized runtime tensors, 196 `AttachMetadata` ops, canonical FP32 scale
  tensors, and no compressed-FP8 post-load transform.
- Dumped openai/gpt-oss-20b packed-runtime plan: 459 `Copy` ops and
  48 `AttachMetadata` ops, zero post-load transforms, persistent weight
  estimate 13,123 MiB, and load-time temporary estimate 0 MiB. Runtime MoE
  dispatch uses bounded BF16 scratch for only the routed MXFP4 experts.
- Small real GPTQ/AWQ smokes:
  `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` completed 1/1 with 98.5 ms mean
  latency, and `Qwen/Qwen2-0.5B-Instruct-AWQ` completed 1/1 with 98.8 ms mean
  latency. Both dumped plans had zero post-load transforms on the default
  non-Marlin release build.
- Additional CUDA native one-token smokes on 2026-05-17:
  - `openai/gpt-oss-20b`: completed 1/1, 1 output token, 332.6 ms mean
    latency. Packed-runtime plan: 459 `Copy`, 48 `AttachMetadata`,
    persistent 13,123 MiB, load-time temp 0 MiB.
  - `microsoft/Phi-3-mini-4k-instruct`: completed 1/1, 1 output token,
    111.7 ms mean latency. Plan: 195 `Copy`; persistent 7,288 MiB.
  - `allenai/Olmo-3-7B-Instruct`: completed 1/1, 1 output token,
    113.2 ms mean latency. Plan: 195 `Copy`, 64 `AxisConcat`; persistent
    13,919 MiB, temp 86 MiB.
  - `mistralai/Ministral-3-3B-Instruct-2512`: completed 1/1, 1 output token,
    110.1 ms mean latency. Plan: 600 `Copy`; persistent 3,654 MiB.
  - `mistralai/Mixtral-8x7B-Instruct-v0.1`: completed 1/1, 1 output token,
    177.2 ms mean latency with `max_batch_tokens=128` and `kv_pages=16`.
    Plan: 995 `Copy`; persistent 89,078 MiB. No new load IR operation was
    needed, but BF16 weights leave limited memory headroom on a 98 GB GPU.
- BF16 CUDA native one-token e2e matrix:
  - `Qwen/Qwen3-0.6B`
  - `Qwen/Qwen3-1.7B`
  - `Qwen/Qwen3-4B`
  - `Qwen/Qwen3-8B`
  - `Qwen/Qwen3-14B`
  - `Qwen/Qwen3-32B`
  - `Qwen/Qwen3-30B-A3B`
  - `Qwen/Qwen3.6-27B`
  - `Qwen/Qwen3.6-35B-A3B`
  - `google/gemma-4-E2B`
  - `google/gemma-4-E4B`
  - `google/gemma-4-26B-A4B`
  - `microsoft/Phi-3-mini-4k-instruct`
  - `allenai/Olmo-3-7B-Instruct`
  - `mistralai/Ministral-3-3B-Instruct-2512`
  - `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Qwen3-32B latency/throughput comparison on the same host, using
  `max_model_len=1024`, `max_tokens=16`, 3 latency requests with 1 warmup, and
  16 throughput requests at concurrency 4:
  - Pie CUDA native: 788.9 ms mean latency, 20.28 output tok/s latency run,
    76.95 output tok/s throughput run.
  - vLLM 0.10.2 with Torch 2.8.0+cu128, Transformers 4.55.4, and
    `VLLM_USE_FLASHINFER_SAMPLER=0` for CUDA 12.8/SM120 sampler
    compatibility: 742.8 ms mean latency, 21.54 output tok/s latency run,
    79.89 output tok/s throughput run.
  - SGLang local checkout with Torch 2.8.0+cu128, local SM120 `sgl-kernel`,
    Triton attention, PyTorch sampling, CUDA graphs disabled, piecewise CUDA
    graphs disabled, FlashInfer arch override `FLASHINFER_CUDA_ARCH_LIST=12.0a`,
    and PDL disabled for CUDA 12.8/SM120 compatibility: 786.1 ms mean latency,
    20.35 output tok/s latency run, 76.95 output tok/s throughput run.
- Current algebra compiler milestone:
  - GPT-OSS MXFP4 is emitted as native algebra for packed routed-dequant and
    eager-BF16 fallback, with no planner ops in either path.
  - Algebra-only storage compilation covers GPT-OSS MXFP4 extent writes,
    decode tile maps, deinterleave/select transforms, attach steps, and
    releases.
  - `LayoutExprKind::Transcode` is now first-class, and optimizer tests cover
    selection pushdown, encode/select movement, select/decode movement, cast
    sinking, partition-join cancellation, and Encode-Decode transcode fusion.
  - Verification: full CUDA CMake build, CTest (`brle`, `driver_common`,
    `layout_plan`), Rust CUDA-feature build, Rust CUDA config tests, evidence
    smoke, and `git diff --check`.

## Paper Framing

The paper contribution can be stated as:

1. A compiler-style formulation of model loading as checkpoint-to-runtime
   layout compilation.
2. A typed tensor layout IR covering packing, sharding, aliasing, quantization,
   and materialization.
3. A memory-aware materialization planner that bounds startup peak memory.
4. A schema mechanism that decouples model-family semantics from CUDA runtime
   binding.
5. An empirical demonstration that large models can load without OOM by
   avoiding unnecessary checkpoint-layout persistence.

The key thesis:

> Model loading should be optimized as a layout compilation problem, not
> patched as a sequence of architecture-specific post-load transformations.
