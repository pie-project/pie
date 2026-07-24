# gpu-lr1 handoff

Snapshot: 2026-07-24

Repository: `git@github.com:ingim/gpu-lr1.git`

Branch: `main`

Last pushed base commit before this handoff: `a5a1508`

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

- `src/gpu_lr1/lr1_tokens.py`
  - `LR1TokenVocabulary`
  - Hugging Face byte-token and explicit terminal-sequence adapters
  - terminal trie
  - bounded LR stack-configuration compiler
  - compile timeout and configuration-count limits
  - heterogeneous global configuration packing
  - CPU reference and Triton runtime wrapper

- `src/gpu_lr1/lr1_token_benchmark.py`
  - synthetic BPE-like vocabulary/depth compile-scaling matrix
  - Qwen3 full-vocabulary probe
  - heterogeneous GPU runtime benchmark

- `src/gpu_lr1/vocab.py`
  - Hugging Face byte-level tokenizer extraction
  - currently requires a byte-level decoder

- `src/gpu_lr1/lr1_workloads.py`
  - byte-terminal arithmetic grammar

- `tests/test_lr1_tokens.py`
  - multi-terminal tokenizer tokens
  - byte-token conversion
  - depth-bound behavior
  - explicit timeout/configuration-limit failures
  - heterogeneous packing
  - CPU/GPU CSR equivalence

- `results/l40s-lr1-token-full.json`
  - final L40S benchmark including Qwen3.

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

L40S fused CSR CUDA time:

| Batch | CUDA time | Wall p50 |
|---:|---:|---:|
| 1 | 28.28 us | 36.63 us |
| 8 | 28.38 us | 36.00 us |
| 32 | 27.84 us | 34.81 us |
| 128 | 28.64 us | 35.16 us |
| 512 | 28.00 us | 35.00 us |
| 2,048 | 28.17 us | 35.54 us |

Benchmark methodology:

- Full 151,669-column FP16 logits.
- Four rotating logits buffers to reduce one-buffer cache bias without
  allocating an excessive 2,048 x 151,669 x iteration-count pool.
- Isolated constraint step only; no language model execution.
- The compatible-token count is intentionally small because the test grammars
  use narrow byte alphabets. A full JSON or programming-language lexer will
  expose many more Qwen tokens and larger rows.

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
  - roughly 28 us fused CUDA time through batch 2,048

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

## Validation at handoff

- Full test suite: **39 tests passed**
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
5. Add rollback/fork and full serving-engine sampling semantics.

## Workflow preference

At every major milestone:

1. run focused tests and the relevant final benchmark;
2. update documentation and results;
3. commit with the Copilot co-author trailer;
4. push to GitHub automatically.

Avoid subagents unless strictly necessary.
