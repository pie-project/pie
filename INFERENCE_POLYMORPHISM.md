# Inference Polymorphism

This document captures a design observation from the CUDA optimization work:
the forward pass of an LLM should be treated as one logical computation with
many valid physical implementations. The model architecture should describe the
math. The serving engine should compile that math into hardware-specific
execution plans.

The idea is not specific to Pie. The same abstraction applies to vLLM, SGLang,
TensorRT-LLM, and other inference engines.

## Thesis

Inference polymorphism is the separation between:

```text
Logical forward pass:
  the pure model computation and its cache/state effects

Physical forward plan:
  the kernels, fusions, layouts, graph buckets, memory arenas, and parallel
  strategy used to execute that computation on a particular machine
```

The runtime-visible operation is stable:

```text
forward(tokens, positions, sequence_state) -> logits, updated_sequence_state
```

The implementation is late-bound:

```text
generic paged attention
XQA decode
Hopper/Blackwell prefill kernels
MLA kernels
fused QKV postprocess
fused gate/up MLP
TP sharded matmuls and collectives
different CUDA graph shape buckets
different KV page sizes and arena layouts
```

The north star is:

> Treat each LLM forward pass as a stable logical operator, then specialize it
> into hardware-specific physical plans using model traits, tensor shapes,
> memory availability, topology, and accelerator capabilities.

## Why This Is Needed

Modern LLM inference is no longer one implementation per model. The same model
math can be executed through very different physical paths depending on GPU
generation, workload shape, memory pressure, tensor-parallel topology, and
kernel availability.

If these choices leak into public config or scheduler code, the system becomes
fragile:

- users must tune physical knobs such as block size, page size, max batch
  tokens, graph sizes, and backend names;
- the scheduler starts encoding assumptions about CUDA kernels and memory
  layouts;
- model ports become a pile of device-specific special cases;
- a kernel optimization for one shape can silently regress another shape;
- TP and graph-capture bugs look like model bugs even when the logical forward
  expression is unchanged.

The cleaner design is to make the model architecture declarative and let a
forward compiler choose the physical plan.

## Concrete Examples

### Attention Kernel Selection

The logical operation may be simply:

```text
causal_attention(Q, K_state, V_state, mask, scale) -> A
```

That should not name FlashInfer, XQA, FA3, or any other kernel. Those are
physical implementations.

Different implementations are valid under different support predicates:

- generic paged attention works broadly and is the fallback;
- XQA decode may be best for pure decode when dtype, head dimension, GQA ratio,
  page size, softcap, and architecture match the kernel;
- Hopper/Blackwell prefill kernels can be best for prefill or mixed batches;
- prefill-decode kernels can help once KV histories are deep enough, but can
  hurt shallow decode;
- MLA kernels are valid only when the model architecture and cache state are
  actually MLA-shaped.

The optimization work showed why this cannot be a static model-level choice.
Forcing a prefill-oriented path into pure decode can regress throughput. XQA is
excellent only when the model shape and cache layout match its requirements.
MLA primitives being available in a library is not enough; the logical model
state must expose MLA semantics before the compiler can legally select them.

### Fusion Strategy

The logical model can define independent affine maps:

```text
Q = Affine_Q(X)
K = Affine_K(X)
V = Affine_V(X)
```

The IR should not say `fused_qkv`, `split_qkv`, `reshape_heads`, or
`write_paged_kv`. Those are physical choices.

A compiler may implement the same math as:

- three independent GEMMs;
- one fused QKV GEMM;
- TP column-sharded QKV projection;
- fused Q/K normalization plus RoPE;
- fused QKV postprocess plus KV-state write;
- a graph-captured decode-specific path;
- an eager fallback for rare shapes.

Similarly, the MLP expression:

```text
Down(SiLU(Gate(X)) * Up(X))
```

can lower to separate projections, fused gate/up projection, quantized kernels,
or architecture-specific epilogues. The model spec should not encode which one.

### Tensor Parallelism

Tensor parallelism does not change the logical forward expression. It changes
the physical realization:

- weights may be column-sharded or row-sharded;
- partial activations may require all-reduce or reduce-scatter;
- some fusions can move across collective boundaries;
- topology determines whether a custom collective is worthwhile;
- rank-local memory availability can differ, so capacity is the minimum across
  ranks.

A TP=2 hang where one GPU is busy and the other is idle is not a semantic model
problem. It is a physical-plan or collective-path problem. The forward IR should
remain the same; the TP planner should be responsible for selecting legal
partitions, collective placement, and rank-local memory plans.

### Memory Layout and Capacity

KV page size, number of KV pages, forward arena size, graph reserve, page-ref
capacity, and output scratch rows are not model semantics. They are physical
plan parameters.

The model says that each layer reads and updates sequence state. The physical
planner decides:

- the KV or state-cache layout;
- page size;
- number of resident pages;
- whether a model needs KV cache, MLA state, linear-attention state, or a hybrid;
- how much memory to reserve for activations, attention scratch, outputs, and
  graph capture;
- the reported forward capacity used by the runtime scheduler.

This is why manual config knobs such as `max_batch_tokens`, page size, and KV
page count are the wrong user-facing abstraction. They are derived properties
of the selected physical plan.

### CUDA Graph Shape Lattices

CUDA graph capture is also physical. The logical forward pass says nothing
about graph buckets. The compiler may choose:

- a small upfront lattice of decode graph shapes;
- a larger vLLM-style lattice;
- eager execution for uncommon tail shapes;
- different graph keys for different attention layout classes.

This choice depends on shape distribution, graph memory cost, capture overhead,
and kernel support. It should not appear in the model spec.

## Architecture Stress Test

The local codebase already supports or recognizes a broad set of architecture
families:

- Llama-like dense decoders: Qwen 2/2.5/3, Llama 3/3.1/3.2, Mistral,
  Ministral/Mistral Small, Phi-3/Phi-4, OLMo 2/3.
- Gemma-family dense decoders: Gemma 2, Gemma 3, Gemma 4, Gemma 3n.
- Sparse-MoE decoders: Mixtral, GPT-OSS, Qwen 3 MoE, Qwen 3.5/3.6 MoE,
  Phi-3.5-MoE, Gemma 4 MoE.
- Hybrid attention/state models: Qwen 3.5 and Qwen 3.6, which mix
  gated-delta-rule linear-attention layers with full-attention layers.
- Multimodal wrappers whose inference target here is the text tower:
  Gemma 3/4 wrappers, Mistral 3 wrappers, Qwen 3.5/3.6 wrappers.

This scan changes the IR design in one important way: the IR cannot be a
hard-coded "Transformer layer" template. It must be a small semantic language
for composing model math. A basic dense decoder is one instance of the language,
not the language itself.

### Dense Llama-Like Models

Qwen, Llama, Mistral, Phi, and OLMo mostly fit the simple decoder-layer shape:

```text
norm -> attention -> residual -> norm -> MLP -> residual
```

But even inside this family the IR needs semantic attributes:

- Q/K normalization may be present or absent.
- QKV/O biases may be present or absent.
- norm placement can be pre-norm or post-norm.
- RoPE can be standard, Llama-3-style scaled, or original YaRN.
- attention can be full or sliding-window per layer.
- Phi-style fused checkpoint tensors must not leak into the IR; they are weight
  storage layout, not model math.

IR implication: the dense layer should be expressed through semantic ops such
as `Norm`, `Affine`, `RoPE`, `Attention`, `MLP`, and `Residual`, with attributes
for norm kind, positional encoding, and attention mask. It should not bake in a
single tensor layout or projection packing.

### Gemma 2 / Gemma 3

Gemma adds several semantic differences:

- embedding output is scaled by `sqrt(hidden_size)`;
- RMSNorm may use Gemma-style centered weights;
- Gemma 2 uses GeGLU/tanh-like MLP behavior instead of the usual SwiGLU path;
- Gemma 2 has attention and final-logit soft caps;
- Gemma 3 adds Q/K norm and alternating full/sliding attention with different
  RoPE bases.

IR implication: norm flavor, activation flavor, logit soft cap, attention logit
soft cap, embedding scale, and per-layer attention type must be first-class
semantic attributes.

### Gemma 4

Gemma 4 is a strong test of the abstraction:

- Per-Layer Embeddings (PLE) inject layer-specific token-derived residual input;
- the last layers can share K/V with earlier source layers of the same
  attention type;
- sliding and full-attention layers may use different head dimensions;
- full-attention layers can use fewer global KV heads;
- some variants derive V from K (`k_eq_v`) and then apply V-norm;
- full-attention layers can use partial RoPE;
- layer outputs have a learned scalar;
- Gemma 4 MoE runs a routed expert branch in parallel with the dense MLP branch.

IR implication: the IR needs explicit named state effects. A layer may append
K/V to its own logical state, or read K/V from a different source layer. Head
dimension and KV-head count can be per-layer semantic properties. PLE is not an
implementation detail; it is extra model math that injects an auxiliary
per-layer signal into the residual stream.

The IR should be able to say:

```text
PLE[l, t, p] =
  PerLayerInput(layer = l, token = token[t], hidden = H[t])

State1 =
  AppendKV(layer = source_layer(l), ...)

A =
  Attention(query = Q_l, key_state = State1.K[source_layer(l)], ...)
```

without saying how the per-layer embedding table is packed or how the
per-layer KV cache is laid out.

### Gemma 3n

Gemma 3n goes beyond ordinary decoder structure:

- AltUp maintains multiple residual streams and alternates updates through
  predict/correct steps;
- Laurel adds a low-rank residual branch;
- activation sparsity can hard-sparsify early-layer MLP gates;
- PLE is present;
- intermediate size can be per-layer;
- KV sharing can be present.

IR implication: "hidden state" cannot always be one tensor. The IR needs a
named residual-state object with multiple streams, and layer bodies can read and
write only one active stream while updating the others through semantic
operators:

```text
Streams1 = AltUpPredict(layer = l, streams = Streams0)
Active   = Streams1.active
Active'  = DecoderBody(layer = l, hidden = Active)
Streams2 = AltUpCorrect(layer = l, streams = Streams1, active = Active')
```

A compiler may materialize this as K buffers, fused updates, or specialized
kernels, but the logical effect is multi-stream state transition.

### GPT-OSS

GPT-OSS combines several non-standard pieces:

- sparse MoE in every layer;
- router bias and per-expert biases;
- clipped SwiGLU variant;
- per-head attention sinks that extend the softmax denominator;
- alternating sliding/full attention;
- MXFP4 expert weight storage in some checkpoints.

IR implication: MoE routing and attention sinks are semantic, while MXFP4
packing is physical. The IR needs an `MoE` operator with router logits,
top-k selection, optional renormalization, and expert application. It also
needs attention to optionally include learned sink terms:

```text
A =
  CausalAttention(
    query = Q,
    key_state = K,
    value_state = V,
    sink = AttentionSink[l, qh],
    mask = layer_mask[l]
  )
```

The compiler can implement sinks as a native attention-kernel feature, an LSE
post-rescale, or a fallback path. The IR should only state the denominator
semantics.

### Mixtral / Qwen MoE / Phi MoE

MoE families share the high-level operator but differ in details:

- expert tensor names and packed layouts differ;
- top-k count differs;
- some models renormalize top-k probabilities;
- Qwen 3.5/3.6 MoE has an always-on shared expert in parallel with routed
  experts;
- Gemma 4 MoE has a routed branch in parallel with a dense branch;
- Phi-3.5-MoE adds its own norm and bias conventions.

IR implication: `MoE` should be parameterized by routing semantics, not by
weight layout:

```text
Y =
  MoE(
    input = X,
    router = Router[l],
    experts = Experts[l],
    top_k = K,
    renormalize = true_or_false,
    shared_expert = optional,
    parallel_dense_branch = optional
  )
```

The physical compiler decides whether this becomes grouped GEMM, expert loops,
batched GEMM, fused Blackwell MXFP4 kernels, or a dense fallback.

### Qwen 3.5 / Qwen 3.6

Qwen 3.5 and 3.6 are the clearest proof that a decoder IR cannot assume every
layer is attention over KV cache:

- most layers are gated-delta-rule linear attention with recurrent matrix
  state;
- every Nth layer is standard full attention over KV cache;
- linear layers include depthwise causal conv state;
- full-attention layers use MRoPE/partial RoPE;
- full-attention output can be gated by a second projection output;
- Qwen 3.6 MoE replaces the dense MLP with routed experts plus a shared expert.

IR implication: cache effects must be generalized beyond KV pages. The IR needs
typed state effects:

```text
KVState' =
  AppendKV(layer = l, ...)

LinearState' =
  GatedDeltaUpdate(
    layer = l,
    state = LinearState,
    conv_state = ConvState,
    key = K,
    value = V,
    gate = Z,
    beta = Beta,
    decay = A
  )
```

A single architecture can therefore contain both:

```text
LayerKind.FullAttention
LayerKind.LinearRecurrent
```

and the physical planner must allocate both KV pages and recurrent state-cache
slots. This is exactly why "KV cache" is too narrow as the only state concept in
the IR.

### Phi-3-Small Blocksparse Attention

Phi-3-small introduces per-request blocksparse-clipped attention. That is not a
kernel name; it is a mask semantic.

IR implication: masks should be first-class:

```text
mask = Causal
mask = SlidingWindow(window)
mask = Blocksparse(pattern)
mask = Custom(mask_id)
```

The compiler can lower these masks to FlashInfer plan variants, custom-mask
prefill, dense fallback, or specialized kernels.

### Multimodal Wrappers

Some checkpoints are multimodal containers but the current inference path runs
only the text tower. This is not a forward-IR complication if the IR is scoped
correctly:

```text
Model = Wrapper(
  text_tower = TextDecoder,
  vision_tower = ignored_or_external,
  projector = ignored_or_external
)
```

IR implication: the forward IR should describe the selected subgraph. Weight
prefix stripping and skipped vision tensors are loader concerns, not forward
math.

### Summary Of Required IR Features

The architecture scan suggests this minimal semantic vocabulary:

- named axes: token, sequence, layer, hidden, query head, KV head, feature,
  expert, residual stream;
- affine maps over semantic axes, independent of packed checkpoint layout;
- norm operators with flavor attributes;
- activation operators including SwiGLU, GeGLU/tanh, clipped SwiGLU, and
  activation sparsity;
- positional encoding operators: RoPE, scaled RoPE, YaRN, MRoPE, partial RoPE;
- attention operators parameterized by mask, sink, scale, soft cap, GQA/MHA,
  and per-layer KV source;
- typed state effects: KV append/read, recurrent linear-attention state update,
  conv-state update, multi-stream residual update;
- MoE operators with routing, top-k, renormalization, shared experts, and
  optional parallel dense branches;
- output transforms such as final norm, tied/untied LM head, and logit soft cap.

This vocabulary keeps the IR pure while still covering the supported model
families. Physical implementation choices remain below the compiler boundary:
packed QKV, fused postprocess, XQA, FA3, grouped GEMM, tensor-parallel
partitioning, graph capture, page size, and memory arena layout.

## The Forward IR Should Be Pure

The forward IR should be close to the mathematical definition of the model. It
should use named semantic axes and explicit state effects. It should not expose
physical tensor layout operations.

Important semantic axes include:

```text
t      token position within the current forward batch
s      logical sequence
m      model hidden axis
qh     query-head axis
kvh    key/value-head axis
d      per-head feature axis
f      MLP intermediate axis
l      layer axis
```

These axes are semantic. They do not say how tensors are packed in memory.

For one decoder layer, a pure IR could look like:

```text
X1[t, m] =
  RMSNorm(X[t, m], weight = AttnNorm[l, m])

Q[t, qh, d] =
  Affine_Q[l](X1[t, m])

K[t, kvh, d] =
  Affine_K[l](X1[t, m])

V[t, kvh, d] =
  Affine_V[l](X1[t, m])

Qp[t, qh, d] =
  RoPE(Q[t, qh, d], position[t])

Kp[t, kvh, d] =
  RoPE(K[t, kvh, d], position[t])

State1 =
  AppendKV(
    state = State0,
    layer = l,
    sequence = sequence_id[t],
    position = position[t],
    key = Kp[t, kvh, d],
    value = V[t, kvh, d]
  )

A[t, qh, d] =
  CausalAttention(
    query = Qp[t, qh, d],
    key_state = State1.K[
      layer = l,
      sequence = sequence_id[t],
      kv_head = HeadGroup(qh),
      position <= position[t],
      feature = d
    ],
    value_state = State1.V[
      layer = l,
      sequence = sequence_id[t],
      kv_head = HeadGroup(qh),
      position <= position[t],
      feature = d
    ],
    scale = AttentionScale[l],
    mask = causal
  )

O[t, m] =
  Affine_O[l](A[t, qh, d])

Y[t, m] =
  X[t, m] + O[t, m]

Z[t, m] =
  RMSNorm(Y[t, m], weight = MlpNorm[l, m])

G[t, f] =
  Affine_Gate[l](Z[t, m])

U[t, f] =
  Affine_Up[l](Z[t, m])

M[t, f] =
  SiLU(G[t, f]) * U[t, f]

D[t, m] =
  Affine_Down[l](M[t, f])

Out[t, m] =
  Y[t, m] + D[t, m]
```

At model level:

```text
H0[t, m] =
  TokenEmbedding(input_token[t])

for l in Layers:
  H(l + 1), State(l + 1) =
    DecoderLayer(l, H(l), State(l), position, sequence_id)

HN[t, m] =
  RMSNorm(H_last[t, m], weight = FinalNorm[m])

Logits[t, vocab] =
  LmHead(HN[t, m])

return Logits, StateFinal
```

This IR is deliberately not written as a sequence of reshapes. It says what the
model computes over semantic axes. The compiler is free to choose how those axes
are materialized.

## What Must Not Be In The IR

The pure forward IR should not contain:

- kernel names such as FlashInfer, XQA, FA3, or cuBLAS;
- page sizes or physical cache block sizes;
- rank IDs, TP shard IDs, or NCCL/custom collective names;
- graph bucket sizes;
- `split_head`, `merge_head`, `reshape_heads`, or packed-QKV layout operations;
- scratch-buffer offsets;
- CUDA stream or event details;
- concrete memory strides unless they are part of an external ABI boundary.

Those belong to the physical compiler and runtime backend.

## Physical Forward Compiler

The compiler consumes:

```text
ModelSpec:
  pure forward IR, semantic axes, dtype constraints, cache/state semantics

HardwareSpec:
  GPU architecture, SM count, memory, bandwidth class, kernel availability,
  graph support, TP topology

ShapeSpec:
  forward token count, request count, prompt/decode mix, KV history

PolicySpec:
  latency, throughput, capacity, memory utilization, determinism requirements

BackendCatalog:
  support predicates and cost models for kernels and fused implementations
```

It produces:

```text
PhysicalForwardPlan:
  selected attention backend per phase
  fusion boundaries
  KV/state layout
  page size and resident page count
  forward arena layout
  graph capture lattice
  TP sharding and collective placement
  output-row and sampler scratch capacity
  runtime capability contract
```

The runtime should schedule against the capability contract. It should not
choose kernels or infer physical cache geometry.

## Support Predicates and Cost Models

Each physical implementation should advertise when it is legal:

```text
supports(model, hardware, shape, layout) -> bool
```

and how expensive it is expected to be:

```text
estimate(model, hardware, shape, layout) -> cost
```

For example, an XQA decode backend may require:

```text
dtype = bf16
head_dim = 128
page_size in supported_page_sizes
attention has no unsupported softcap
architecture has the required kernel variant
GQA grouping is supported
shape is pure decode or decode-dominant
```

A Hopper prefill backend may require:

```text
architecture >= sm90
prefill or mixed-prefill shape
workspace fits in the forward arena
mask form is supported
```

The planner should select the best legal implementation, not the most specific
one. A specialized kernel that is legal but slower for a shallow-KV shape should
lose to the generic path.

## Correctness Contract

Two physical plans are equivalent if they implement the same logical forward
operator:

```text
same semantic inputs
same cache/state reads and writes
same output positions
same masking semantics
same sampling-visible logits/probabilities within accepted numerical tolerance
```

The tolerance must account for dtype, accumulation order, fused epilogues, and
collective ordering. But physical plans must not change model semantics. If an
implementation requires a different cache state, such as MLA instead of standard
KV, that requirement must be represented in the model/state spec before the
planner can select it.

## Why The Abstraction Should Be Engine-Agnostic

vLLM, SGLang, and Pie all face the same problem:

- one model family can run on Ampere, Ada, Hopper, Blackwell, and future GPUs;
- one forward pass can be prefill, decode, mixed, speculative, or graph-captured;
- one attention expression can map to many kernels;
- one TP topology can make a fusion good or bad;
- one memory policy can decide whether the workload runs as one resident cohort
  or several waves.

The engines differ in implementation, but the abstraction pressure is the same.
Physical execution choices should be derived from model, shape, and hardware,
not exposed as permanent user knobs.

## Lessons From The Optimization Work

Several concrete findings point toward this abstraction:

- Memory planner tuning alone cannot explain every performance gap. Once KV
  residency is close to vLLM, remaining gaps often come from attention-path or
  model-body implementation choices.
- Bigger physical plans are not always better. Larger forward-token arenas,
  larger graph lattices, or more aggressive prefill-decode paths can reduce KV
  residency or hurt cadence.
- Page size is a physical choice. It affects fragmentation, kernel support,
  metadata size, and KV residency, but it is not part of model semantics.
- Graph capture must be keyed by physical layout class. Capturing an unsafe
  plan can produce coherent-looking execution with corrupted outputs.
- TP correctness and performance depend on collective placement and topology.
  The logical model does not change when the TP plan changes.
- Kernel availability is not enough. The model/state semantics and physical
  layout must satisfy the kernel's assumptions.
- A generic fallback is necessary. Specialized kernels are optimizations, not
  the semantic definition of attention.

## Design Principle

The model should provide a pure, effect-aware forward spec. The backend should
own physical planning.

```text
Pure model architecture:
  "This is the forward function."

Physical planner:
  "This is how to run that function well on this hardware for this shape."

Runtime:
  "This is the capacity contract I can safely schedule against."
```

Inference polymorphism is the discipline of keeping those three roles separate.
