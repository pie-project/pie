# gpu-lr1 handoff

Snapshot: 2026-07-24

Repository: `git@github.com:ingim/gpu-lr1.git`

Branch: `main`

Last pushed base commit before this handoff: `3004b40`

## Current objective

Build a practical GPU-native constrained-decoding engine for heterogeneous
batches of LR(1) grammars, with JSON Schema as the main application.

The project now has three related execution paths:

1. **Canonical JSON Schema DFA**
   - Compiles an acyclic canonical JSON Schema subset to a byte DFA.
   - Computes tokenizer/DFA transitions and packs heterogeneous schemas into
     global CSR, bitset, or dense tables.
   - Fused Triton kernels select tokens and write next states.

2. **Direct canonical LR(1)**
   - Compiles arbitrary deterministic LR(1) grammars into sparse ACTION/GOTO
     tables and production metadata.
   - Packs grammar-local states, nonterminals, and productions into global ID
     spaces.
   - Executes terminal selection, reductions, goto transitions, and shifts on
     GPU using bounded ragged stacks.

3. **Bounded tokenizer-aware LR(1)**
   - Maps tokenizer byte strings to sequences of grammar terminals for
     byte-terminal grammars, or accepts explicitly supplied terminal sequences.
   - Uses a terminal trie to share token-prefix work.
   - Enumerates exact reachable LR state stacks up to `max_stack_depth`.
   - Emits finite sparse CSR rows:

     ```text
     configuration -> [(token_id, next_configuration), ...]
     ```

   - Reuses the fused CSR token-selection/next-state Triton kernel, so no parser
     stack is needed at runtime for compiled configurations.

## Newly implemented but not yet pushed

- Prevalidated CSR launch plans that move shape and device checks out of the hot
  path.
- Stable launch-shape autotuning over warp counts and packed-row candidates.
- Multi-row packed CSR and ELLPACK experimental kernels.
- CUDA Graph capture with safe in-place grammar-state updates.
- A row-width/batch sweep over 1-8,192 allowed tokens and batch 1-2,048.
- Exact finite-depth arithmetic and balanced grammars shared with XGrammar.
- Configuration witness tokens and differential mask tests.
- A fair Qwen3 benchmark that verifies every selected XGrammar mask is
  bit-for-bit identical to gpu-lr1 before timing.
- Strong XGrammar baselines:
  - native mask application plus argmax;
  - fused bitset argmax;
  - CUDA Graph fused bitset argmax;
  - captured pinned-H2D mask copy plus argmax;
  - optional matcher `accept_token` and rollback.
- New result artifacts:
  - `results/l40s-csr-optimization.json`
  - `results/l40s-fair-xgrammar-qwen3.json`

## Qwen3 result

Tokenizer: `Qwen/Qwen3-0.6B`

Full tokenizer size: **151,669 token IDs**

The benchmark reads the tokenizer's byte-level BPE tokens, preserves the real
token IDs and EOS ID, and runs logits with the full vocabulary width.

Bounded grammar tables:

| Grammar | Depth bound | Compatible Qwen3 tokens | Configurations | Token edges | Compile |
|---|---:|---:|---:|---:|---:|
| Byte arithmetic | 4 | 87 | 81 | 783 | 0.034 s |
| Balanced parentheses | 16 | 20 | 594 | 3,682 | 0.060 s |

Packed result:

- 675 configurations
- 4,465 sparse token edges
- 41,819 runtime bytes excluding diagnostic reduction counts
- 59,679 bytes including diagnostics
- 0.094 s aggregate compile time

Original validated-wrapper L40S CUDA time:

| Batch | CUDA time | Wall p50 |
|---:|---:|---:|
| 1 | 28.28 us | 36.63 us |
| 8 | 28.38 us | 36.00 us |
| 32 | 27.84 us | 34.81 us |
| 128 | 28.64 us | 35.16 us |
| 512 | 28.00 us | 35.00 us |
| 2,048 | 28.17 us | 35.54 us |

Optimized launch paths:

| Path | Typical CUDA | Typical wall |
|---|---:|---:|
| Validated wrapper | 28-29 us | 37-38 us |
| Prevalidated plan | 25-27 us | 34-36 us |
| Four-buffer CUDA Graph | 4.7-5.2 us | about 11.5 us |

Benchmark methodology:

- Full 151,669-column FP16 logits.
- Four rotating logits buffers to reduce one-buffer cache bias without
  allocating an excessive 2,048 x 151,669 x iteration-count pool.
- Isolated constraint step only; no language model execution.
- The compatible-token count is intentionally small because the test grammars
  use narrow byte alphabets. A full JSON or programming-language lexer will
  expose many more Qwen tokens and larger rows.

Kernel sweep conclusions:

- CUDA Graph replay is the largest optimization.
- Packed rows help a few very narrow, high-batch cases but not the real Qwen3
  mixture consistently.
- ELLPACK uses 105,300 bytes versus 41,819 CSR bytes and is not consistently
  faster.
- Wide 8,192-entry rows reach roughly 72 us at batch 2,048; Qwen3's 19-entry
  rows remain near 5 us of actual device work.

## State-explosion result

Bounded configuration expansion is exact within the configured depth, but it is
not a general compact representation of an unbounded LR stack.

Synthetic arithmetic grammar results:

| Vocabulary | Depth | Configurations | Edges | Compile |
|---:|---:|---:|---:|---:|
| 256 | 4 | 81 | 8,463 | 0.18 s |
| 256 | 6 | 318 | 42,279 | 0.88 s |
| 256 | 8 | 1,042 | 142,803 | 3.00 s |
| 1,024 | 4 | 81 | 31,002 | 0.62 s |
| 1,024 | 6 | 318 | 159,472 | 3.13 s |
| 1,024 | 8 | timeout | timeout | >10 s |
| 4,096 | 4 | 81 | 100,641 | 2.10 s |
| 4,096 | 6 | timeout | timeout | >10 s |
| 4,096 | 8 | timeout | timeout | >10 s |

Three of eighteen compile-scaling cases hit the explicit 10-second timeout.

Conclusion:

- Bounded expansion is a strong GPU-only fast path for shallow or
  low-branching grammars.
- Direct stack execution remains necessary for configurations that do not
  compile compactly.
- The next architecture should choose among bounded expansion, direct stack
  execution, and a Pre3/PSC-like stack classifier based on predicted growth.

## Important correctness boundary

The tokenizer-aware backend currently supports:

- byte-terminal grammars with byte-level tokenizers;
- explicit mappings from each model token to a sequence of grammar terminals;
- exact behavior for reachable LR stacks within a configured maximum depth.

It does not yet implement:

- a general regex lexer with mid-lexeme state;
- unbounded recursion through finite configuration expansion;
- complete JSON Schema semantics;
- speculative rollback/fork;
- top-k, top-p, or temperature sampling;
- full serving-engine integration.

Do not claim general unbounded tokenizer-aware LR(1) constrained decoding yet.

## Existing benchmark highlights

- Canonical JSON Schema DFA vs XGrammar:
  - 2.2x at batch 1
  - 17.3x at batch 512
- Direct terminal-level LR(1):
  - 17 grammars, 734 states
  - roughly 63-65 us fused CUDA time through batch 512
  - 64.7 us mixed batch 2,048
- Bounded Qwen3 tokenizer LR(1):
  - 25-27 us prevalidated launch CUDA time
  - 4.7-5.2 us CUDA Graph time through batch 2,048

## Fair XGrammar result

Benchmark invariants:

- same Qwen3 151,669-token vocabulary;
- same finite-depth grammars and exact prefixes;
- same full-width FP16 logits;
- masks compared bit-for-bit before timing;
- fastest XGrammar thread count selected from 1/2/4/8/auto;
- equivalent CUDA Graph optimization applied to both GPU paths.

Strongest optimistic XGrammar means CPU mask fill + captured pinned H2D copy +
graphed fused bitset argmax, without accepting the selected token.

| Batch | gpu-lr1 graph | XGrammar optimistic | XGrammar stateful | Speedup optimistic | Speedup stateful |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.7 us | 23.9 us | 45.1 us | 2.0x | 3.9x |
| 8 | 11.8 us | 75.5 us | 115.2 us | 6.4x | 9.7x |
| 32 | 11.7 us | 273.1 us | 350.7 us | 23.3x | 29.9x |
| 128 | 11.7 us | 953.9 us | 1,183.0 us | 81.4x | 101.0x |
| 512 | 11.8 us | 2,664.6 us | 3,764.1 us | 225.2x | 318.1x |
| 2,048 | 11.8 us | 10,222.4 us | 17,388.9 us | 863.5x | 1,468.9x |

Counterweights:

- Without graphs, gpu-lr1 is slower at batch 1 and wins from batch 8.
- gpu-lr1 compile: 0.856 s; XGrammar compile: 0.0032 s.
- gpu-lr1 runtime tables: about 292 KiB excluding diagnostics; XGrammar compiler
  cache: about 52 KiB.
- gpu-lr1 plan autotuning and four graph captures add about 1.1 s per batch
  shape in this benchmark and must be cached.
- Model execution is excluded, and XGrammar can overlap CPU matching with it.

## Commands

Install:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[tokenizers,baselines]'
```

Tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Direct LR benchmark:

```bash
.venv/bin/gpu-lr1-lr-bench \
  --profile full \
  --logit-columns 32768 \
  --output results/l40s-lr1-full.json
```

Tokenizer-aware LR benchmark:

```bash
.venv/bin/gpu-lr1-lr-token-bench \
  --profile full \
  --qwen-model Qwen/Qwen3-0.6B \
  --output results/l40s-lr1-token-full.json
```

CSR optimization:

```bash
.venv/bin/gpu-lr1-csr-opt-bench \
  --output results/l40s-csr-optimization.json
```

Fair XGrammar comparison:

```bash
.venv/bin/gpu-lr1-fair-xgrammar-bench \
  --output results/l40s-fair-xgrammar-qwen3.json
```

## Validation at handoff

- Full test suite: **43 tests passed**.
- Qwen3 result artifact:
  - vocabulary size confirmed as 151,669;
  - six batch-size runtime records present;
  - three expected synthetic compile-timeout records present.
- Editable package and `gpu-lr1-lr-token-bench` entrypoint refreshed.

## Recommended next milestone

1. Add a byte/regex lexer state to the LR configuration key.
2. Compile a realistic JSON or tool-call grammar using Qwen3's full tokenizer.
3. Estimate configuration growth before compilation.
4. Route compact states to bounded CSR tables and large states to the direct
   ragged-stack kernel.
5. Integrate the graphed constraint step into a serving engine/model CUDA graph.
6. Add rollback/fork and top-k/top-p/temperature sampling.

## Workflow preference

At every major milestone:

1. run focused tests and the relevant final benchmark;
2. update documentation and results;
3. commit with the Copilot co-author trailer;
4. push to GitHub automatically.

Avoid subagents unless strictly necessary.
