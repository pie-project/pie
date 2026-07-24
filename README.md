# gpu-lr1

GPU feasibility prototype for heterogeneous batched LR(1) and JSON Schema
constrained decoding.

## Result

The central idea is practical in three stages:

1. Precompiled, non-recursive JSON Schemas with canonical JSON serialization
   can be compiled into a tokenizer-aware byte DFA.
2. Arbitrary deterministic LR(1) grammars can be compiled into sparse
   ACTION/GOTO tables and executed on GPU over an already-segmented grammar
   terminal stream.
3. For a configured stack-depth bound, reachable LR stack configurations can be
   expanded into finite token states. A terminal trie then compiles tokenizer
   tokens spanning several grammar terminals into sparse
   `(token, next_configuration)` rows.

For the typically sparse token rows in the DFA backend, the best measured layout
is not a dense `M_next[Q, V]` table but a packed CSR edge table:

```text
row_ptr[state] -> [(allowed_token, next_state), ...]
```

A single Triton program gathers only allowed logits, selects the token, and
writes the next state. Every schema uses one global state namespace, so a batch
contains only global state IDs; there is no schema padding or device pointer
chasing.

On an NVIDIA L40S with a 32,768-token GPT-2 vocabulary:

- 14 heterogeneous schemas / 3,533 states: fused CSR runtime tables use
  **25.2 MiB** and the mixed-batch kernel takes about **27.5-28.2 us** of CUDA
  time.
- 64 heterogeneous schemas / 17,424 states: fused CSR tables use **102.1 MiB**.
- Mixed-schema wall latency is **34.9 us at batch 1** and **45.5 us at batch
  512**.
- Against XGrammar 0.2.3's CPU mask generation, pinned H2D bitmask copy, GPU
  mask application, and argmax, the measured wall-clock speedup grows from
  **2.2x at batch 1** to **17.3x at batch 512**.
- The canonical LR(1) backend packs 17 heterogeneous grammars into 734 global
  states and 20.3 KiB of ACTION/GOTO/production tables. Its fused
  select+reduce+shift kernel takes about **63-65 us** through batch 512 and
  **64.7 us** for a mixed batch of 2,048 sequences on the same L40S.
- With the full Qwen3 151,669-token vocabulary, bounded byte-terminal arithmetic
  and recursive balanced grammars compile into 675 configurations and 4,465
  token edges. The runtime tables use **40.8 KiB**, compile in **0.094 s**, and
  the fused CSR token step takes about **28 us** through batch 2,048.

This is evidence for GPU-native high-batch execution. General lexer integration,
unbounded tokenizer-aware LR(1), and arbitrary JSON Schema semantics are not yet
solved.

## What is implemented

1. A canonical JSON Schema compiler builds an epsilon NFA and determinizes it
   into a byte DFA.
2. The compiler computes the DFA/tokenizer cross-product on GPU, correctly
   handling tokenizer tokens that span multiple JSON terminals.
3. Per-schema states are relocated into flat packed tables with global state
   IDs.
4. CPU references and Triton kernels implement:
   - dense byte-mask lookup and logits masking;
   - packed int32 bitset lookup and masking;
   - dense and bitset two-stage argmax;
   - CSR/ragged allowed-token argmax;
   - fused CSR `(token, next_state)` selection;
   - dense token-transition lookup;
   - compact byte-DFA token traversal;
   - row-major and depth-major divergent stack updates.
5. Reproducible benchmarks compare homogeneous, 4-schema, all-schema, sparse,
   and dense batches.
6. A separate canonical LR(1) backend implements:
   - FIRST sets, LR(1) closure/goto construction, and conflict detection;
   - sparse ACTION/GOTO tables and production metadata;
   - relocation of differently sized grammars into one global state,
     nonterminal, and production namespace;
   - bounded ragged stack segments with different capacities per sequence;
   - a CPU reference parser;
   - fused and split fast/slow Triton shift-reduce kernels.
7. A bounded tokenizer-aware LR(1) compiler implements:
   - explicit token-to-terminal sequences and a real tokenizer-byte adapter;
   - a terminal trie that shares work across token prefixes;
   - exact reachable LR stack configurations up to a configured depth;
   - heterogeneous global packing into sparse token/next-state CSR rows;
   - the existing fused CSR GPU sampler with no runtime parser stack.

## Important finding about the original LR/PDA hypothesis

For a general LR parser, a reduction pops states and the subsequent `goto`
depends on the newly exposed stack state. Therefore, `M_next[q, token]` is not
generally independent of the stack.

The new LR(1) backend executes this stack-dependent reduce/goto closure directly
on GPU. The tables are a disjoint global packing of:

```text
ACTION[state, terminal] -> SHIFT / REDUCE / ACCEPT
GOTO[state, nonterminal] -> state
PRODUCTION[id] -> (lhs, rhs_length)
```

The direct LR kernel consumes one grammar terminal per step. The bounded token
compiler now bridges real tokenizer byte strings when grammar terminals are
bytes, or any explicit token-to-terminal sequence supplied by a lexer adapter.
It is exact for reachable stacks within `max_stack_depth`, but converts those
full stacks into finite configuration IDs. General regex/lexer states and
unbounded recursion still need a compact stack classifier or transition-program
representation.

## Related work

The literature survey in [`related-work/`](related-work/) separates tokenizer
alignment, LR/PDA construction, CPU-versus-GPU grammar execution, serving
integration, and JSON Schema evaluation methodology.

The closest direct prior art is
[Gram2Token](https://icml.cc/virtual/2026/poster/62392), whose ICML 2026 abstract
also describes GPU-resident token transition tables under schema-diverse
continuous batching. [Pre3](https://aclanthology.org/2025.acl-long.551/) and
[PSC](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/81/Efficient-Grammar-Constrained-Decoding-via-Parser-Stack-Classification)
are the closest stack-aware LR/parser predecessors. The defensible gpu-lr1
contribution is the public, measured combination of flat heterogeneous state
packing, direct sparse LR execution, bounded stack-configuration expansion,
CSR/bitset representations, and fused Triton token selection plus state update.

## Benchmark results

Hardware and software:

- NVIDIA L40S 46 GiB, compute capability 8.9
- PyTorch 2.8.0+cu129
- Triton 3.4.0
- XGrammar 0.2.3
- GPT-2 token bytes truncated to a 32,768-token vocabulary
- Qwen/Qwen3-0.6B tokenizer with all 151,669 token IDs

### End-to-end constraint step vs XGrammar

Both paths include token selection. The gpu-lr1 CSR path also writes the next
automaton state. The XGrammar baseline is optimistic: it uses one CPU worker,
pinned memory, precompiled/static matcher states, and excludes
`accept_token`/rollback and model execution.

| Batch | gpu-lr1 CSR p50 | XGrammar p50 | Speedup |
|---:|---:|---:|---:|
| 1 | 34.9 us | 78.3 us | 2.2x |
| 8 | 34.6 us | 86.0 us | 2.5x |
| 32 | 36.8 us | 106.8 us | 2.9x |
| 128 | 37.1 us | 185.5 us | 5.0x |
| 512 | 45.5 us | 789.4 us | 17.3x |

The original blanket "10x faster" claim is not supported at batch 1. It becomes
plausible only at high heterogeneous batch sizes, where CPU mask construction
and host/device staging scale with the number of sequences.

### Kernel comparison, 14-schema mixed batch, batch 512

| Strategy | CUDA time |
|---|---:|
| CSR argmax only | 25.0 us |
| Fused CSR argmax + next state | 28.1 us |
| Dense two-stage argmax | 49.8 us |
| Bitset two-stage argmax | 49.7 us |
| Bitset argmax + dense next table | 62.9 us |
| Bitset argmax + byte-DFA advance | 66.0 us |
| Optimistic CPU-sync baseline | 376.1 us wall |

The CSR kernel is launched with a fixed 32,768-element upper bound, so it does
not need a CPU-side maximum-row-length calculation. Masked lanes do not issue
loads. With rotating logits buffers that exceed L2 cache at high batch, CSR
selection at batch 512 measured about 25 us through 1,024 allowed tokens, 31 us
around 1,025-4,096, 46 us around 4,097-8,192, and 51 us above 8,192. Dense and
bitset selection stayed near 50 us. A production sampler should therefore use
a CSR/bitset hybrid rather than CSR for every state.

All headline runs rotate through fresh logits buffers during timed iterations.
This avoids overstating throughput by repeatedly reading one cache-resident
logits tensor.

### Runtime table memory

| Schemas / states | CSR token + next | Bitset + byte DFA | Bitset + dense next |
|---|---:|---:|---:|
| 14 / 3,533 | 25.2 MiB | 19.4 MiB | 455.4 MiB |
| 64 / 17,424 | 102.1 MiB | 87.2 MiB | 2,246.1 MiB |

The dense `M_next[Q, V]` design is the main memory failure. Sparse
`(token,next_state)` edges or a compact byte-DFA transition after sampling are
the viable representations.

The tokenizer cross-product compilation took 1.4 seconds for 14 schemas and
6.2 seconds for 64 schemas, versus 0.07 seconds for the 14-schema XGrammar
compile in this workload. Production use therefore requires persistent
compiled-table caching.

### Stack-depth divergence

The stack microbenchmark tested row-major and depth-major stacks at depths
distributed identically, narrowly, and uniformly across a maximum depth of
1,024. Up to batch 8,192, updates generally took 22-24 us. The largest observed
layout/divergence penalty was about 21%, not the hypothesized 10x collapse.

This only measures one top lookup and one branchless push/pop update. It does
not cover variable-length LR reduction loops.

### Canonical LR(1) terminal-step backend

The LR benchmark compiles arithmetic, JSON structure, balanced-parenthesis,
wide-choice, long-sequence, and reduction-chain grammars. It packs 17 grammars,
734 states, 526 terminals, and 465 productions into 20,808 bytes of sparse
runtime tables. Logits have 32,768 columns, while sparse ACTION rows contain at
most 256 terminals.

| Mix | Batch | Mean stack depth | Mean reductions | Fused CUDA | Split CUDA |
|---|---:|---:|---:|---:|---:|
| shift | 1 | 1.0 | 0.0 | 66.0 us | 97.0 us |
| shift | 2,048 | 24.5 | 0.0 | 63.1 us | 93.6 us |
| depth-divergent | 2,048 | 43.3 | 1.0 | 63.5 us | 95.7 us |
| reduction-divergent | 512 | 2.0 | 22.3 | 64.5 us | 95.0 us |
| reduction-divergent | 2,048 | 2.0 | 22.3 | 79.3 us | 94.3 us |
| mixed | 2,048 | 21.2 | 5.9 | 64.7 us | 94.4 us |

The fused kernel is consistently faster than launching a separate shift fast
path and reduction slow path. At this grammar scale, the second launch costs
more than it saves. Stack depth from 2 through 129 has little effect; variable
reduction work is the more meaningful divergence axis, and only becomes visible
at the largest reduction-heavy batch.

The CPU number in this benchmark is a Python reference over terminal logits
already resident on the host. It excludes real serving costs such as device
logits, tokenizer alignment, mask transfer, and model sampling, so it is a
correctness and scaling reference rather than a production CPU baseline.

### Bounded tokenizer-aware LR(1) backend

The bounded compiler turns each reachable LR state stack into a finite
configuration and walks a trie of tokenizer-terminal sequences from every
configuration. This makes the online step identical to the fast DFA path:

```text
configuration -> CSR [(token_id, next_configuration), ...]
```

The full stress table uses a 1,024-token BPE-like arithmetic alphabet and four
heterogeneous grammar/depth combinations:

| Metric | Result |
|---|---:|
| Configurations | 554 |
| Token edges | 190,886 |
| Compile time | 3.77 s |
| Runtime token+next tables | 1.46 MiB |
| Direct ACTION/GOTO tables for the same four grammar/depth entries | 9.14 KiB |
| Memory expansion versus direct LR tables | 164x |
| GPU fused step, typical CUDA time | 28-29 us |

The actual Qwen3 full-vocabulary probe is smaller because only
grammar-compatible byte
tokens are representable:

| Grammar | Depth bound | Representable Qwen3 tokens | Configurations | Edges | Compile |
|---|---:|---:|---:|---:|---:|
| Byte arithmetic | 4 | 87 | 81 | 783 | 0.034 s |
| Balanced parentheses | 16 | 20 | 594 | 3,682 | 0.060 s |

Packed together, these tables use 41,819 runtime bytes. The heterogeneous GPU
step remains about 28 us from batch 1 through 2,048.

The failure mode is state explosion. With a synthetic 4,096-token grammar
alphabet, arithmetic depth 4 compiles to 81 configurations and 100,641 edges in
2.10 s, while depth 6 and 8 exceed the explicit 10-second compile limit.
Across the 18 compile-scaling cases, three time out. Balanced parentheses remain
small because their branching structure is much simpler.

The conclusion is a hybrid architecture: bounded configuration expansion is an
excellent GPU-only fast path for shallow or low-branching grammars, but direct
stack execution or Pre3/PSC-style stack classifiers are required when
configuration expansion grows too large.

Full machine-readable results are in [`results/`](results/).

## Supported schema subset

Accepted documents are canonical valid subsets of the input schema:

- objects with sorted, declared properties;
- `required`, `minProperties`, and `maxProperties`;
- bounded arrays and capped unbounded arrays;
- `const`, `enum`, and `anyOf`;
- booleans, null, integers, numbers, and ASCII JSON strings;
- string length constraints;
- bounded integer constraints and positive integer `multipleOf`;
- acyclic local `$ref`.

The compiler rejects, rather than silently approximates:

- recursive `$ref`;
- `oneOf`, `allOf`, conditionals, and `not`;
- regex/pattern and dynamic property constraints;
- `contains`, `uniqueItems`, dependent schemas, and related evaluation
  semantics;
- unconstrained `true` schemas and unconstrained array items.

`const`, `enum`, and `anyOf` with unnormalized sibling assertions are also
rejected; declared `type` is the only assertion intersected directly with
finite `const`/`enum` values.

Production JSON Schema support needs regex DFAs, counters/side state, object-key
tracking, recursion, and precise Draft 2020-12 evaluation semantics. Existing
engines have similar limitations; see
[JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench).

## Reproduce

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[tokenizers,baselines]'

.venv/bin/python -m unittest discover -s tests -v

.venv/bin/gpu-lr1-bench \
  --profile full \
  --schemas 14 \
  --vocab gpt2 \
  --vocab-size 32768 \
  --output results/l40s-gpt2-full.json

.venv/bin/gpu-lr1-xgrammar-bench \
  --schemas 14 \
  --vocab-size 32768 \
  --threads 1 \
  --output results/l40s-xgrammar-full.json

.venv/bin/gpu-lr1-density-bench \
  --output results/l40s-density-crossover.json

.venv/bin/gpu-lr1-stack-bench \
  --output results/l40s-stack-layout.json

.venv/bin/gpu-lr1-lr-bench \
  --profile full \
  --logit-columns 32768 \
  --output results/l40s-lr1-full.json

.venv/bin/gpu-lr1-lr-token-bench \
  --profile full \
  --output results/l40s-lr1-token-full.json
```

The 64-schema scaling run is:

```bash
.venv/bin/gpu-lr1-bench \
  --profile quick \
  --schemas 64 \
  --vocab gpt2 \
  --vocab-size 32768 \
  --batch-sizes 64 512 \
  --mixes mixed_all mixed_sparse mixed_dense \
  --warmup 5 \
  --iterations 20 \
  --output results/l40s-gpt2-64schemas.json
```

## What remains before serving integration

- Add a general regex/lexer frontend; the current tokenizer bridge is exact for
  byte-terminal grammars or explicitly supplied terminal sequences.
- Add an adaptive compiler that chooses bounded configuration expansion,
  direct stack execution, or a Pre3/PSC-style stack classifier from measured
  state-growth estimates.
- Compile recursive JSON Schema structure into the LR(1) backend and add side
  state for schema semantics that are not context-free.
- Implement categorical/temperature/top-k/top-p sampling directly over sparse
  rows; the current backends benchmark greedy argmax.
- Add LR(1) stack rollback/fork for speculative decoding.
- Fuse penalties and sampling semantics used by a real serving engine.
- Benchmark full model TPOT/throughput instead of the isolated constraint step.
- Run JSONSchemaBench coverage and MaskBench-style parser workloads.

Most current production serving systems keep parser/matcher work on CPU and
stage packed masks to GPU; relevant implementations include
[XGrammar](https://github.com/mlc-ai/xgrammar),
[vLLM structured outputs](https://github.com/vllm-project/vllm), and
[SGLang constrained decoding](https://github.com/sgl-project/sglang). Gram2Token
is the closest published GPU-native counterexample at the abstract level, but
its declared implementation was not publicly reachable when the related-work
survey was written.
