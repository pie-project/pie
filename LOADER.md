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

> Separate semantic identity, runtime representation, and physical
> materialization.

This yields the following architecture:

```text
SafetensorsLoader
  -> ModelSchema
  -> Logical tensor graph
  -> Load IR
  -> target-aware lowering and optimization passes
  -> memory plan
  -> Materializer
  -> WeightStore / LoadedModel
  -> BoundCudaModel
```

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
checkpoint program -> runtime tensor program -> scheduled physical execution
```

## Design Principles

### 1. Semantic Identity Is Separate From Storage

A model schema declares what tensors mean:

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

The schema does not perform CUDA allocation and does not copy checkpoint bytes.
It describes the logical tensor contract for an architecture.

### 2. Runtime Representation Is Explicit

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

### 3. Physical Materialization Owns Lifetime

The materializer is responsible for:

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

### SafetensorsLoader

The `SafetensorsLoader` owns checkpoint metadata and byte movement.

Its intended responsibility is narrow:

- parse safetensors metadata
- expose tensor dtype, shape, file offsets, and shard location
- copy full tensors or slices to device memory
- provide generic read primitives for the materializer

It should not know about Qwen, Gemma, MoE, quant policy, packed QKV, or CUDA
kernel binding.

The repository resolver that feeds `SafetensorsLoader` should also become
artifact-selective. Runtime loading needs `config.json`, tokenizer artifacts,
`model.safetensors`, and/or `model.safetensors.index.json` plus referenced
safetensors shards. It should not fetch alternate checkpoint formats such as
`.pt` or `.bin` files when safetensors are available.

### ModelSchema

`ModelSchema` is the architecture-owned semantic description.

It declares:

- model family
- logical tensor roles
- checkpoint naming aliases
- required and optional tensors
- shape constraints
- tensor parallel rules
- runtime layout preferences
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

### Load IR

The Load IR describes tensor transformations independent of model family.
There are two levels of IR:

- a portable logical layout IR that records checkpoint semantics and allowed
  runtime representations
- an executable `LoadPlan` selected for a concrete target profile

The executable plan is allowed to be target-dependent. For example, GPT-OSS
MXFP4 can lower to native packed MXFP4 tensors on a target with a registered
native MoE backend, or to BF16 expert tensors on a target without that backend.
The operation vocabulary remains generic in both cases.

The current implementation has an operational IR:

```text
Read
Copy
Slice
Shard
RowRangeShard
MoeGateUpShard
MoeDownShard
Cast
Concat
PackRows
View
Alias
Drop
QuantizeRuntime
Dequantize
SplitInterleaved
RepackQuant
AttachQuantMeta
FuseMoeExperts
Materialize
```

The north-star IR should cover the following reusable operation families:

```text
Read
Slice
Shard
Cast
Concat
PackRows
View
Alias
Drop
QuantizeRuntime
Dequantize
SplitInterleaved
RepackQuant
AttachQuantMeta
FuseMoeExperts
Materialize
```

Architecture-specific names such as `PackQKV`, `PackGateUp`, or
`FuseExperts` may exist as high-level lowering helpers, but their semantics
should reduce to generic layout operations.

### Planner

The planner lowers schema declarations into an executable load plan.

Logical passes:

```text
CheckpointBindingPass
  raw safetensors names -> logical tensors

RuntimeLayoutPass
  logical tensors -> final runtime tensors

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
```

The planner must estimate:

- persistent bytes
- maximum temporary bytes
- estimated peak bytes
- scratch requirements

### Materializer

The materializer executes the plan.

Rules:

- allocate final runtime buffers directly when possible
- stream checkpoint tensors into final storage
- avoid keeping checkpoint-only tensors alive
- expose compatibility names as views instead of copies
- release temporaries at last use
- attach quant metadata only after referenced tensors exist
- validate actual materialization against planned tensor specs

For dense packed QKV, the desired execution is:

```text
allocate qkv_final
read q shard -> copy into qkv_final rows
release q temporary
read k shard -> copy into qkv_final rows
release k temporary
read v shard -> copy into qkv_final rows
release v temporary
register q/k/v compatibility views into qkv_final
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

### TensorSpec

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
PackedQkv
PackedGateUp
FusedMoeExperts
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
- Every runtime tensor has a complete `TensorSpec`.
- Every view has a valid backing tensor.
- No view outlives its backing tensor.
- Every quantized tensor has matching quant metadata.
- Tensor parallel shards are shape-checked before allocation.
- Checkpoint-only tensors do not become persistent runtime tensors.
- The materializer does not allocate unplanned persistent tensors.
- `BoundCudaModel` consumes only validated runtime tensors.

These invariants make extension easier: new architectures add schema rules,
and new layouts add reusable IR operations.

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

- `RepackQuant` into Marlin-compatible storage, or
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
always dequantize. The schema declares the checkpoint facts:

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
RepackQuant to backend layout when the selected kernel requires a swizzle
AttachQuantMeta
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
SplitInterleaved gate/up rows into final expert tensors
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

1. Add a `ModelSchema` adapter.
2. Declare logical tensor roles and checkpoint naming aliases.
3. Declare shape constraints and optional tensors.
4. Declare tensor parallel rules.
5. Declare preferred runtime layouts.
6. Add plan tests using synthetic safetensors metadata.
7. Bind the resulting `WeightStore` in the model-specific binder.

The materializer should not need model-family-specific branches unless the
family requires a genuinely new reusable physical operation.

### Adding a Quantization Format

To add a quantization format:

1. Add a `QuantFormat` value.
2. Add validation rules for metadata and supported tensor layouts.
3. Add lowering rules from checkpoint representation to runtime
   representation.
4. Add materializer handlers for dequant, repack, or runtime quantization.
5. Add GEMM dispatch support or an eager fallback.
6. Add plan tests and at least one startup smoke.

### Adding a Runtime Layout

To add a runtime layout:

1. Add a layout kind only if existing generic layout operations cannot express
   the layout clearly.
2. Add validation rules.
3. Add memory planning behavior.
4. Add materializer support.
5. Add binder support.

Prefer generic operations over model-specific op names.

## Current Implementation Progress

Last updated: 2026-05-17.

### Implemented

- `SafetensorsLoader` parses checkpoint metadata and copies full tensors or
  generic contiguous/strided tensor slices into caller-owned device memory. It
  has no allocation-returning API and no MoE-specific copy entry points; all
  allocation, layout, and lifetime decisions live in the materializer.
- `ModelSchema` exists as a first explicit schema resolution layer.
- `LogicalTensorGraph` is a distinct checkpoint-binding stage that records raw
  checkpoint names, runtime names, inferred semantic roles, dtype, and shape
  before load-plan lowering.
  It also records logical tensor groups such as packed QKV, packed gate/up,
  per-expert MoE, fused MoE, GPT-OSS MXFP4, and FP8+scale pairs so adapters
  can identify semantic groups before lowering selects physical ops.
- `LoadPlan` contains typed runtime tensor specs.
- `LoadOp` uses typed payload families instead of a flat overloaded field bag.
  Constructors build raw-load, row-range, tensor, slice, pack-rows, and
  fuse-MoE payloads; validation and materialization consume those payloads via
  typed accessors.
- `LoadPlan` contains explicit executable operations for:
  - GPTQ/AWQ dequantization and Marlin repack
  - compressed-tensors FP8 and INT8 metadata registration
  - runtime FP8/INT8 quantization
  - eager FP8 and MXFP4 BF16 fallbacks
  - packed MXFP4 runtime loading
  - Qwen per-expert MoE fusion
- `LoadPlan` tracks persistent, temporary, and estimated peak memory.
- Dense Llama-like Qwen/Qwen2/Llama/Mistral/Olmo loading can plan packed QKV
  and packed gate/up tensors.
- Packed QKV and packed gate/up expose compatibility names as views.
- `PackRows` materialization streams sources into final storage rather than
  keeping all packed inputs alive at once.
- `FuseMoeExperts` can stream per-expert checkpoint rows directly into final
  fused expert tensors instead of staging checkpoint-shaped source tensors.
- Raw `Copy`, `Shard`, `Read`, `RowRangeShard`, `MoeGateUpShard`, and
  `MoeDownShard` ops copy checkpoint bytes directly into final or planned
  temporary destinations. BF16 dense Qwen-style packed plans no longer create
  full checkpoint-shaped device buffers for projection tensors before packing.
- `WeightStore` stores `TensorRecord { spec, tensor }`, counts only owned
  resident tensors for memory accounting, and rejects erasing a backing tensor
  while a registered view or alias still depends on it.
- `Materializer` executes load plans into `WeightStore` without depending on
  `LoadedModel`.
- Materialization validates produced tensor dtype, shape, and view backings
  against the plan.
- The materializer supports generic `Read`, `Copy`, `Shard`, `RowRangeShard`,
  `MoeGateUpShard`, `MoeDownShard`, `Slice`, `Cast`, `Concat`, `PackRows`,
  `View`, `Alias`, `Drop`, `QuantizeRuntime`, `Dequantize`,
  `SplitInterleaved`, `RepackQuant`, `AttachQuantMeta`, `FuseMoeExperts`,
  and `Materialize` operations.
- FP16/FP32 checkpoint tensors lower into scheduled `Copy -> Cast -> Drop`
  sequences when the runtime representation is BF16. The same producer helper
  covers normal copies, Phi-3 row-range shards, and Qwen MoE direct expert
  shard ops; canonical quant scale tensors that must remain FP32 are left
  untouched.
- `RepackQuant` is a first-class scheduled materializer operation for
  symmetric GPTQ int4 checkpoints that target Marlin W4A16 storage.
- AWQ int4 checkpoints and GPTQ int4 fallback variants, including GPTQ
  act-order, lower into scheduled
  `Copy(qweight/qzeros/scales[/g_idx]) -> Dequantize -> Drop` op sequences.
  The checkpoint-only int4 tensors are temporary `TensorSpec`s tagged with
  `AwqInt4` or `GptqInt4`; final runtime tensors are BF16 dense weights when
  no native runtime int4 representation is selected. Tensor-parallel
  act-order row shards keep full GPTQ scale/zero metadata and slice only
  qweight/g_idx, preserving the checkpoint's global group ids without an
  ad-hoc remap.
- Runtime `fp8` and `int8` quantization for supported dense projection
  weights lowers into explicit `Copy -> QuantizeRuntime -> AttachQuantMeta
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
  `Copy(weight) -> Copy/Cast(scale) -> AttachQuantMeta` plans. The final
  runtime tensor is `QuantPacked` INT8 with canonical scale metadata; variants
  that require zero-points or unsupported layouts remain rejected before
  materialization.
- `Dequantize` materialization supports compressed-tensors FP8 E4M3, AWQ
  int4, GPTQ int4, and GPT-OSS MXFP4 E2M1/E8M0 blockwise dequantization to
  BF16.
- Symmetric GPTQ int4 checkpoints with `desc_act=false` and no zero-points
  lower into scheduled
  `Copy(qweight/scales) -> RepackQuant -> AttachQuantMeta -> Drop`. The final
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
  `Copy(qweight/qzeros/scales) -> RepackQuant -> AttachQuantMeta -> Drop`
  plans that produce Marlin-packed INT4 runtime tensors with canonical BF16
  scale and INT32 zero-point side tensors. On non-Marlin targets they lower to
  scheduled BF16 `Dequantize`.
- The CUDA load planner receives a `LoadTarget` profile instead of a raw
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
  `"eager_bf16"`. True native MXFP4 is represented as a `LoadTarget`
  capability and should bind to FlashInfer/TRT-LLM, CUTLASS, or Marlin rather
  than an architecture-local custom GEMM.
- `PIE_CUDA_LOAD_PLAN_DUMP=/path/to/plan.json` writes a JSON plan artifact with
  ops, tensor specs, and memory estimates.
- The planner computes persistent bytes from final owned tensors and computes
  temporary high-water bytes with a last-use scan over the executable op stream,
  while preserving conservative scratch estimates for operations that allocate
  unregistered scratch.
- Qwen3-32B starts successfully with the CUDA native driver.
- The BF16 CUDA native e2e smoke matrix covers dense Qwen3, Qwen3 MoE,
  Qwen3.6 dense, Qwen3.6 MoE, Gemma4 dense, and Gemma4 MoE checkpoints.
- `BoundCudaModel` centralizes architecture binding from the validated loaded
  model.
- Model binders consume `const LoadedModel&`; remaining checkpoint-layout
  synthesis has moved into load-plan lowering/materialization.
- `bind_gpt_oss` consumes loader-materialized `QuantPacked` MXFP4 expert
  tensors when the packed runtime path is selected, and consumes BF16 expert
  tensors only for the explicit eager fallback path.
- HF snapshot resolution is runtime-artifact selective by default for both
  `serve` and `pie model download`: config/tokenizer artifacts plus
  `model*.safetensors` are fetched, while duplicate `.pt`, `.bin`, `.gguf`,
  and non-loader safetensors artifacts are skipped. `pie model download --all`
  remains available when a complete HF snapshot is explicitly needed.
- The planner accepts a metadata-only tensor source, so load-plan tests can run
  on synthetic checkpoint metadata without real safetensors files or GPU
  allocation.
- CPU-only load-plan tests cover dense Qwen packed QKV/gate-up lowering,
  scheduled runtime INT8 quantization, scheduled FP16/FP32-to-BF16 casts,
  Phi-3 tensor-parallel row-range sharding, symmetric GPTQ lowering into
  scheduled `RepackQuant`, and GPTQ tensor-parallel local slicing for column-
  and row-parallel projections. They also cover AWQ Marlin repack and BF16
  fallback, asymmetric GPTQ and GPTQ act-order dequant fallbacks,
  compressed-tensors INT8 metadata attachment, scheduled Qwen per-expert MoE
  fusion, Qwen MoE expert tensor-parallel sharding, and GPT-OSS MXFP4
  target-dependent plans for BF16 fallback and packed runtime representations.
  The Phi-3 test caught and fixed a moved-from `LoadOp::output_name` bug in
  row-range spec registration.

### Partially Implemented

- The IR is typed and broad enough for current layouts, but some less common
  checkpoint quantization sub-modes still need schema coverage before they can
  lower into scheduled graph ops.
- Compressed-tensors FP8 and symmetric INT8 lowerings are scheduled for the
  variants Pie can bind today; asymmetric INT8, compressed INT4, and other
  compressed-tensors sub-modes still need schema and backend coverage.
- MoE fused expert loading has scheduled schema support for current Qwen
  per-expert checkpoint layouts and GPT-OSS packed MXFP4 layouts. Future MoE
  variants should reuse the same generic `FuseMoeExperts`, quant metadata, and
  backend-selection machinery rather than adding architecture-local transforms.
- Memory planning now performs last-use analysis for scheduled temporary
  tensors. Runtime backend scratch, such as GPT-OSS routed-expert MXFP4
  dequant scratch, is tracked outside load-plan temporary high-water marks and
  should be reported explicitly as runtime scratch.
- `LogicalTensorGraph` exists, but current lowering still infers many roles
  from suffix conventions. Future schema adapters should declare roles and
  groups directly instead of relying on inference.

### Not Yet Implemented

- True hardware-native MXFP4 MoE GEMM kernels. The current packed runtime
  backend keeps MXFP4 as the final loaded representation and dequantizes only
  routed experts into bounded BF16 scratch before GEMM.
- Complete schema adapters for every architecture in the tree.

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
- Symmetric GPTQ emits scheduled `RepackQuant` on Marlin-capable targets and
  scheduled BF16 `Dequantize` on non-Marlin targets with qzeros metadata. AWQ
  emits scheduled `RepackQuant` on Marlin-capable targets and scheduled BF16
  `Dequantize` on non-Marlin targets. GPTQ act-order TP keeps full scale/zero
  metadata for row-parallel shards so `g_idx` remains valid without rewriting.
- Compressed-tensors FP8 and symmetric INT8 emit scale normalization and quant
  metadata attachment.
- GPT-OSS MXFP4 emits target-dependent plans: packed-runtime targets produce
  `QuantPacked` MXFP4 tensors with attached scales; eager fallback targets
  produce `Dequantize` and `SplitInterleaved` operations with no persistent raw
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

- Current implementation pass:
  `c++ -std=c++20 ... driver/cuda/tests/test_load_plan.cpp ...` and
  `cargo build -p pie-server --release --no-default-features --features driver-cuda`.
  The synthetic plan tests now include typed `LoadOp` payload validation and
  scheduled FP8 `weight_scale_inv` dequant lowering.
- `cargo build -p pie-server --release --no-default-features --features driver-cuda`
- `PIE_CUDA_BUILD_MARLIN=1` path verified through
  `cmake --build .tmp/cuda-marlin-build --target pie_driver_cuda_lib -j2`
  before removing the temporary build tree.
- `cargo test -p pie-server hf:: --no-default-features --features driver-cuda`
- `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -S driver/cuda -B .tmp/cuda-load-plan-tests -DCMAKE_BUILD_TYPE=Release`
- `cmake --build .tmp/cuda-load-plan-tests --target test_load_plan -j2`
- `.tmp/cuda-load-plan-tests/bin/test_load_plan`
- Synthetic load-plan tests include scheduled GPTQ symmetric int4
  `RepackQuant` lowering with canonical BF16 scale metadata, tensor-parallel
  local qweight/scale slicing, and no `OfflineGptqAwq` transform.
- Synthetic load-plan tests include scheduled AWQ int4, asymmetric GPTQ int4,
  and GPTQ act-order BF16 fallback lowerings, including tensor-parallel local
  qweight, qzeros, scale, and g_idx handling with no `OfflineGptqAwq`
  transform. They also cover target-dependent symmetric GPTQ and AWQ:
  Marlin-capable targets emit `RepackQuant`; non-Marlin targets emit scheduled
  BF16 `Dequantize` when the checkpoint has qzeros metadata.
- Current-turn verification: Qwen3-0.6B CUDA native one-token smoke completed
  1/1 with 98.9 ms mean latency. Dumped plan: 171 `Copy`, 56 `PackRows`,
  persistent 1,433 MiB, temp 6 MiB, and zero post-load transforms.
- Qwen3-32B direct CUDA native load after direct-to-final materialization:
  load compiler emitted 515 load ops and 835 runtime tensor specs, persistent
  62,488 MiB, load-time temp <= 0 MiB, peak ~= 62,488 MiB, with 64 packed QKV
  groups and 64 packed gate/up groups. The model loaded 835 tensors in about
  5 seconds on the RTX PRO 6000 Blackwell Server Edition.
- Synthetic load-plan tests include Qwen MoE TP expert sharding and GPT-OSS
  MXFP4 target selection: BF16 fallback emits `Dequantize` /
  `SplitInterleaved` / `Slice` ops, while packed-runtime targets emit
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
  `AttachQuantMeta` ops, 196 `Drop` ops, 196 quantized runtime tensors, and no
  `RuntimeQuantize` transform.
- Dumped RedHatAI/Qwen3-0.6B-FP8-dynamic plan: 196 compressed-FP8 E4M3
  quantized runtime tensors, 196 `AttachQuantMeta` ops, canonical FP32 scale
  tensors, and no compressed-FP8 post-load transform.
- Dumped openai/gpt-oss-20b packed-runtime plan: 459 `Copy` ops and
  48 `AttachQuantMeta` ops, zero post-load transforms, persistent weight
  estimate 13,123 MiB, and load-time temporary estimate 0 MiB. Runtime MoE
  dispatch uses bounded BF16 scratch for only the routed MXFP4 experts.
- Small real GPTQ/AWQ smokes:
  `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` completed 1/1 with 98.5 ms mean
  latency, and `Qwen/Qwen2-0.5B-Instruct-AWQ` completed 1/1 with 98.8 ms mean
  latency. Both dumped plans had zero post-load transforms on the default
  non-Marlin release build.
- Additional CUDA native one-token smokes on 2026-05-17:
  - `openai/gpt-oss-20b`: completed 1/1, 1 output token, 332.6 ms mean
    latency. Packed-runtime plan: 459 `Copy`, 48 `AttachQuantMeta`,
    persistent 13,123 MiB, load-time temp 0 MiB.
  - `microsoft/Phi-3-mini-4k-instruct`: completed 1/1, 1 output token,
    111.7 ms mean latency. Plan: 195 `Copy`; persistent 7,288 MiB.
  - `allenai/Olmo-3-7B-Instruct`: completed 1/1, 1 output token,
    113.2 ms mean latency. Plan: 195 `Copy`, 64 `PackRows`; persistent
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
  - Pie CUDA native: 789.5 ms mean latency, 20.26 output tok/s latency run,
    76.89 output tok/s throughput run.
  - vLLM 0.10.2 with Torch 2.8.0+cu128 and Transformers 4.55.4: 747.6 ms mean
    latency, 21.40 output tok/s latency run, 78.69 output tok/s throughput
    run.
  - SGLang local checkout with Torch 2.8.0+cu128, local SM120 `sgl-kernel`,
    Triton attention, PyTorch sampling, CUDA graphs disabled, piecewise CUDA
    graphs disabled, FlashInfer arch override `FLASHINFER_CUDA_ARCH_LIST=12.0a`,
    and PDL disabled for CUDA 12.8/SM120 compatibility: 785.9 ms mean latency,
    20.36 output tok/s latency run, 73.93 output tok/s throughput run.

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
