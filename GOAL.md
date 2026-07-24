# GOAL

## Ultimate goal

Build **gpugrammar**: a GPU-native constrained-decoding library that *replaces*
XGrammar as the default grammar backend in LLM serving engines.

Two hard requirements set it apart from the current `gpu-lr1` prototype:

1. **Real parser power.** The engine must handle **LALR(1)**, and preferably
   **IELR(1)**, grammars — not a bounded subset, not an acyclic schema DFA.
2. **Paper before code.** The design is written up and submitted as an academic
   paper first; the open-source release follows. The paper must present a set of
   clearly identified technical challenges solved in a way that is *elegant*,
   not merely engineered.

## Sequencing

| Milestone | Target |
|---|---|
| Technical challenges frozen, architecture decided | now |
| Core engine + evaluation complete | before submission |
| **MLSys 2027 submission** | **2026-10-30** (expected deadline, 23:59 PDT) |
| MLSys 2027 conference | 2027-05-17 – 05-22 |
| Public source release of `gpugrammar` | after acceptance/decision |

Everything in this repository is a feasibility prototype feeding that paper.
Work that does not either (a) retire a named challenge below or (b) produce a
figure/table in the paper is out of scope.

## What "replaces XGrammar" must mean concretely

A serving engine should be able to swap XGrammar for gpugrammar and get:

- **Equal or greater grammar coverage.** Full LALR(1)/IELR(1), plus a regex
  lexer layer, plus JSON Schema as a front-end — with no silent language
  truncation.
- **Bit-exact masks.** Differentially verified against a reference parser on
  every benchmark configuration, not just spot-checked.
- **Lower per-step cost at serving batch sizes**, with the constraint step
  resident inside the model's CUDA graph — no host round trip per token.
- **Serving-grade compile times.** New schemas arrive per request; compilation
  must be incremental and cached, not a per-grammar batch job.
- **Bounded memory** under heterogeneous continuous batching, competitive with
  XGrammar's kilobyte-scale compiler cache.
- **Full sampler integration:** temperature/top-k/top-p, plus rollback and fork
  for speculative decoding.

## Killer examples (measured, A100 80GB, Qwen3 151,669-token vocabulary)

Two workloads where a GPU-resident engine is not incrementally better but
categorically better. Both use XGrammar's own builtin JSON grammar, XGrammar
0.2.3 for the baseline, and FlashInfer 0.6.15 `top_k_top_p_sampling_from_logits`
as the sampler — the sampler vLLM/SGLang actually use.

### Example 1: fused constrained sampling is cheaper than *unconstrained* sampling

At a JSON number/structural position the grammar allows 396 of 151,669 tokens.
Sampling over the allowed set instead of the vocabulary (median wall, µs):

| batch | FlashInfer, **no constraint** | XGrammar mask (on GPU) + FlashInfer | XGrammar full path + FlashInfer | **gather + FlashInfer** |
|---:|---:|---:|---:|---:|
| 1 | 275.5 | 331.0 | 1,014.7 | **191.1** |
| 128 | 1,070.9 | 1,130.6 | 5,867.6 | **201.6** |
| 512 | 3,456.7 | 3,775.4 | 8,968.0 | **198.0** |
| 2,048 | 13,007.2 | 14,526.0 | 29,737.3 | **408.6** |

Constraining the grammar makes sampling **17.5× faster at batch 512 and 31.8×
faster at batch 2,048 than sampling with no constraint at all**, and 45×/73×
faster than the deployed XGrammar path. Nobody exploits this today because every
existing system hands back a full-vocabulary mask and lets the engine sample
over the whole vocabulary.

In the opposite regime — a JSON string body, 147,144 allowed tokens, 4,525
exceptions — fusion still does not lose: 3,063.8 µs versus 3,435.1 µs
unconstrained and 3,980.9 µs for XGrammar at batch 512. The constraint is
effectively free at both ends of the density spectrum.

Context: a 7B decode step at batch 512 is ~8.5 ms (weight streaming lower bound,
measured 1,648 GB/s). Today's constrained path adds 8.97 ms — it more than
doubles the step. The fused path adds 0.198 ms, or 2.3%.

### Example 2: speculative decoding breaks CPU grammar execution

A CPU matcher must fill a mask and accept a token serially for each draft
position, then roll back. Grammar cost per outer step (median wall, µs):

| batch | k | XGrammar | gpugrammar | ratio |
|---:|---:|---:|---:|---:|
| 512 | 1 | 4,155.6 | 13.3 | 312× |
| 512 | 4 | 16,500.7 | 31.9 | 518× |
| 512 | 8 | **33,366.7** | 56.2 | **594×** |

At batch 512 with 8 draft tokens XGrammar spends 33 ms of CPU per outer step —
about 4× a 7B decode step. Constrained speculative decoding at scale is not
slow today; it is infeasible.

## Grammar class decision

Canonical LR(1) — what the prototype implements today — is verifiably correct
(exhaustively cross-checked against an independent Earley recognizer, including
ε-productions, unit chains, and an LR(1)-but-not-LALR(1) grammar) but its state
count explodes on realistic grammars.

**IELR(1) is the target**: it accepts the same grammars as canonical LR(1) while
producing tables the size of LALR(1). LALR(1) is the acceptable fallback if the
IELR(1) construction cannot be made incremental. The choice must be justified in
the paper with measured state counts and table sizes on real grammars (JSON,
JSON Schema, a tool-call/DSL grammar, and a programming-language grammar).

Also required at this layer: precedence/associativity declarations for conflict
resolution, and an EBNF front-end, since practical grammars are not written in
bare BNF.

## Technical challenges (the substance of the paper)

Each challenge below is stated as a problem, not a solution. Solutions are what
the paper contributes. Challenges marked **[measured]** are already backed by
numbers from this repository.

### C1. Tokenizer–grammar impedance mismatch

Model tokens are BPE byte strings that straddle lexeme boundaries; an LR parser
consumes terminals. A token may end mid-lexeme, so the parser configuration must
carry **lexer state**, not just a parser stack. The prototype avoids this by
restricting itself to byte-terminal grammars, which is why it cannot express a
real language today.

*Why hard:* the configuration space is (lexer DFA state × parser stack), and
maximal-munch lexing means a token's terminal segmentation is not a function of
the token alone.

### C2. Stack-dependent transitions vs. GPU-friendly flat tables

A reduction pops states, and the following `goto` depends on the newly exposed
stack state. Therefore `next[state, token]` is **not** stack-independent, and no
flat per-state table is exact in general.

**[measured]** Enumerating reachable stacks up to a depth bound works but
explodes: at vocabulary 4,096 / depth 6 the compile times out (>10 s), and the
bound silently drops edges that exceed it, making the accepted language a strict
*subset* of the grammar — a correctness cliff, not just a performance one.

*Why hard:* the needed object is a sound **and** complete finite abstraction of
the stack language, compact enough to index on GPU. Prior art (Pre3, PSC) does
parser-stack classification; the open question is an abstraction that is exact
for mask computation, not merely conservative.

### C3. Wide-row mask representation — the dominant scaling wall

**[measured]** The fused CSR kernels are one-program-per-row with
`BLOCK_SIZE = next_pow2(row_nnz)`, hard-capped at 32,768 entries. A realistic
JSON *string* state over Qwen3 allows **146,924 of 151,669 tokens**, needing
`BLOCK_SIZE` 262,144 — the kernel raises rather than runs. The density sweep
already shows CSR losing to dense/bitset beyond ~8,192 allowed tokens.

**Status.** The 32,768 cap is removed and the complement path is implemented.
Streaming the allowed list was correct but unusably slow (6.2 ms at batch 1,
46.7 ms at batch 2,048, against 0.27 / 13.0 ms for unconstrained FlashInfer)
because every bisection probe re-gathered the row through its index list. The
wide kernel now reads logits **contiguously** and tests membership against an
18.5 KiB per-state bitset, and each sweep evaluates 8 candidate thresholds at
once so the search costs 8 passes instead of 32. Wide rows dropped to 1.43 ms
at batch 1 and 20.5 ms at batch 2,048 — a 4.3x improvement, and now 1.6x of
unconstrained sampling rather than 3.6x.

The remaining wide-row gap is **parallelism, not algorithm**: one program per
sequence means batch 1 occupies a single SM while sweeping 151,669 logits
eleven times. Splitting a wide row across several CTAs with a two-level
reduction is the next step and should mostly close the small-batch gap.

*Why hard, and why it is the most promising contribution:* real grammar states
are bimodal — either a handful of allowed tokens (structural positions: 396–759
tokens) or almost the entire vocabulary (string bodies: 147,144). A single
representation cannot serve both. The narrow half is now solved by gathering the
row; the wide half needs the dual: a **per-state complement** (4,525 exceptions,
18.5 KiB as a bitset) consumed by one contiguous O(V) pass, with the threshold
found from a single-pass histogram instead of repeated global probes. Cost then
becomes the unconstrained sampler's cost plus about 6% of extra traffic.

### C4. Table memory under heterogeneous continuous batching

**[measured]** Naive CSR costs ~1.1 MiB for a single JSON-string state on Qwen3;
1,000 such states is 1.09 GiB. The schema DFA backend already reaches 25 MiB for
14 schemas at a 32k vocabulary. XGrammar's compiler cache is ~52 KiB.

*Why hard:* thousands of concurrent requests with distinct grammars share GPU
memory with the model and KV cache. Needs structural sharing (interned mask
sets, a DAG over states, suffix sharing across schemas), plus paging/eviction
policy — while keeping lookups branch-free on device.

### C5. The latency floor is dispatch, not compute

**[measured]** For the direct LR(1) step at batch 1: reported 36.4 µs, CPU
dispatch 38.7 µs, CUDA-graph replay 3.41 µs, **actual kernel 3.16 µs**. The
device work is already ~10× cheaper than the framework overhead around it.

*Consequence:* the contribution cannot be "a fast kernel". It must be a
constraint step that is **resident in the serving engine's CUDA graph**, with
in-place state updates, no host synchronization, and stable launch shapes across
changing batch composition. Benchmarks must report device time and dispatch time
separately; the current `measure()` helper conflates them and must be fixed
before any number goes in the paper.

### C6. Warp divergence from unbounded reduce chains

**[measured]** A right-recursive grammar needs a reduce chain proportional to
nesting depth (depth 100 → 100 reductions in one step); with a bound of 16 the
step returns `REDUCTION_LIMIT`. On SIMT hardware the deepest lane dictates the
step cost for its whole batch.

*Why hard:* per-step work must be bounded and uniform. Candidates: precomputed
reduce-closure compression so one token = one table lookup, or work
redistribution across lanes. Any bound must not truncate the language (see C2).

### C7. Online grammar admission and compile budget

**[measured]** XGrammar compiles in ~3.2 ms; the prototype takes ~0.86 s plus
~1.1 s of launch autotuning per batch shape. In serving, grammars arrive with
requests.

*Why hard:* needs lazy/incremental table construction (build states on first
visit), a persistent cross-request cache, and possibly GPU-side construction —
while preserving the exactness guarantees of C2.

### C8. Correctness at scale, as a first-class artifact

The prototype's parser core is exhaustively verified against an independent
Earley recognizer, and its GPU kernels match the CPU reference step-for-step.
That standard must extend to the full system: bit-exact differential masks
against a reference parser across grammars, tokenizers, and batch shapes, plus
an argued **soundness and completeness** result for the stack abstraction —
over-approximation admits invalid strings, under-approximation silently removes
valid ones.

### C9. Sampler and speculative-decoding integration

Masks must compose with temperature/top-k/top-p sampling, and support rollback
and fork for speculative decoding and beam search — all batched on device. The
prototype does argmax only, which is not a usable serving interface.

### C10. Ragged fused sampling across a heterogeneous batch

**[measured]** The sampler-fusion win (Example 1) depends on processing only the
allowed set. But sequences in one batch sit in states of wildly different width
— 396 tokens at a number position, 147,144 inside a string — and 51.4% of the
tokens in a realistic JSON document are emitted from wide string states. A dense
gather buffer is sized by the widest row in the batch, so a single wide sequence
erases the advantage for all 512.

*Why hard:* the sampler must consume **ragged** rows — per-sequence widths, not a
padded rectangle — while keeping top-k/top-p semantics exact and staying inside
one CUDA graph. Candidates: width bucketing with per-bucket launches, a
per-row persistent kernel, or a two-tier design that routes narrow rows to a
gathered sampler and wide rows to a complement-masked full-width sampler. This
is the concrete engineering-and-algorithms problem that makes Example 1 real
instead of anecdotal.

**Status.** Implemented in `src/gpu_lr1/ragged_sampler.py` and verified by
`tests/test_ragged_sampler.py` (11 tests, exact against a sorted reference).
Bucketing is sync-free: both bucket kernels launch over the same grid and each
program returns immediately if its row belongs to the other bucket, so the hot
path has no host round trip and stays CUDA-graph capturable. Removing the two
`.item()` syncs that the first version hid in the dispatcher cut narrow-row
sampling from 355 µs to **86 µs**, flat from batch 1 through 512.

Measured on A100 with Qwen3 and XGrammar's builtin JSON grammar (median wall,
`results/a100-ragged-sampler.json`):

| batch | narrow: gpugrammar | FlashInfer unconstrained | XGrammar full path | speedup vs unconstrained |
|---:|---:|---:|---:|---:|
| 1 | 86.4 µs | 282.2 µs | 963.2 µs | 3.3× |
| 128 | 85.5 µs | 1,071.8 µs | 6,018.7 µs | 12.5× |
| 512 | 86.0 µs | 3,433.9 µs | 10,030.0 µs | 39.9× |
| 2,048 | 155.4 µs | 13,004.8 µs | 29,480.6 µs | **83.7×** |

With the complement path added for the wide bucket (see C3), a realistic mixed
batch — 51.4% string-body sequences, matching the measured composition of a JSON
document — now crosses over:

| batch | mixed: gpugrammar | FlashInfer unconstrained | XGrammar full path |
|---:|---:|---:|---:|
| 128 | 2,258.5 µs | 1,071.5 µs | 5,271.5 µs |
| 512 | 3,636.8 µs | 3,436.8 µs | 10,078.6 µs |
| 2,048 | **10,994.5 µs** | 12,993.2 µs | 30,829.4 µs |

At batch 2,048 constrained sampling is 1.2x faster than sampling with no
constraint at all and 2.8x faster than the deployed XGrammar path. Below batch
512 the wide bucket's single-CTA-per-sequence layout still dominates, so C10 is
retired for narrow rows and for large batches, and open for small batches.

## Decision 1: API surface for release

Verified against the actual integration points: vLLM's
`v1/structured_output/backend_types.py` (`StructuredOutputBackend` /
`StructuredOutputGrammar`) and SGLang's `constrained/base_grammar_backend.py`
(`BaseGrammarBackend` / `BaseGrammarObject`). SGLang exposes a public
`register_grammar_backend(name, init_func)` registry, so a third-party backend
needs **no upstream patch**; vLLM currently selects backends by name and needs
either a plugin hook or a small upstream registration.

Four layers, shipped in this order.

### Layer 0 — engine-agnostic core (`gpugrammar`)

```text
Compiler(tokenizer_info, cache) -> CompiledGrammar     # IELR(1) tables + token alignment
GrammarPool(compiled, max_seqs, device)                # GPU-resident state arena
  .create(seq_idx) / .clone(seq_idx) / .free(seq_idx)
  .accept(seq_idx, tokens) -> bool                     # batched
  .validate(seq_idx, tokens) -> int                    # prefix length, no advance
  .rollback(seq_idx, k)
  .is_terminated(seq_idx)
  .forced_string(seq_idx) -> bytes | None              # jump-forward
  .fill_bitmask(bitmask, indices)                      # batched, device-resident
  .sample(logits, sampling_params) -> tokens           # fused fast path
```

State is one `int32` configuration ID per sequence, so `clone` is a copy of one
integer and `rollback(k)` is an index into a per-sequence ring buffer of the
last 200 configuration IDs (800 B/sequence). Both are O(1); XGrammar must
rebuild a matcher or replay tokens.

### Layer 1 — vLLM adapter

Implement exactly: `compile_grammar(request_type, grammar_spec)`,
`allocate_token_bitmask(max_num_seqs)`, `destroy()`, and on the grammar object
`accept_tokens`, `validate_tokens`, `rollback`, `fill_bitmask(bitmask, idx)`,
`is_terminated`, `reset`.

`validate_tokens` is vLLM's speculative-decoding hook and is implemented today
as serial accept-then-rollback — this is precisely where Example 2 applies.

### Layer 2 — SGLang adapter

Implement `accept_token`, `rollback(k)`, `is_terminated`, `copy`,
`allocate_vocab_mask(vocab_size, batch_size, device)`, `fill_vocab_mask(mask, idx)`,
`move_vocab_mask(mask, device)`, `apply_vocab_mask(logits, mask)`,
`try_jump_forward`, `jump_forward_str_state`, `jump_and_retokenize`, plus
`is_support_token_filter` / `set_token_filter`. Register with
`register_grammar_backend("gpugrammar", ...)`.

**Key adapter design rule.** Both engines call `fill_*_bitmask(mask, idx)` once
per sequence in a Python loop, then apply the mask. gpugrammar must make that
loop free: `fill` only records `(state_id, idx)`, and the single batched kernel
runs at `apply_vocab_mask(logits, mask)` time. SGLang's
`allocate_vocab_mask` already receives `device`, so the mask can be allocated on
GPU and `move_vocab_mask` becomes a no-op — CPU fill and the H2D copy both
disappear **without any upstream change**.

### Layer 3 — fused sampling (upstream proposal)

Neither engine exposes a "select the token" hook, so Example 1 cannot be
delivered through the current interfaces. Propose an optional capability —
`supports_fused_sampling` plus `sample_constrained(logits, sampling_params)` —
and contribute it upstream. The paper evaluates both layers so the result stands
even if the extension is not merged.

### Drop-in requirements that are not negotiable

To be selectable as a backend the library must accept every spec type the
engines dispatch: JSON Schema, bare JSON object, EBNF (including Lark
conversion), regex, choice, and structural tags. It must handle byte-fallback
and tekken tokenizers, honor model EOS/stop-token overrides, support
`rollback` up to 200 tokens, and expose jump-forward strings. IELR(1) is the
execution engine underneath, not the user-facing contract.

## Decision 2: paper contributions

**Motivation.** The CPU-to-GPU capability gap widens every generation, so any
stage left on the host becomes the serialization point of the decode loop.
Constrained decoding is the last major CPU-resident stage, and it is not a
niche: structured output is now the default interface for tool calling and
agents. Yet it is the stage most hostile to GPU execution — parsing is
stack-dependent sequential control flow, batches are heterogeneous, and the
allowed set is bimodal. That difficulty, not indifference, is why every
deployed system still parses on the CPU.

Claimed contributions, in the order they should appear:

1. **A characterization that reframes the problem.** The cost of constrained
   decoding is not mask *generation*; it is that a full-vocabulary mask forces
   the sampler to stay O(V). Measured on an A100 with Qwen3: allowed-set size
   is bimodal (396 versus 147,144 tokens), the deployed path more than doubles
   a 7B decode step at batch 512, and under speculative decoding it exceeds the
   model step by 4x. Prior work optimizes the wrong term.

2. **GPU-executable IELR(1).** Turning stack-dependent reduce/goto closure into
   a form indexable by a single device lookup, with a soundness and
   completeness argument, plus lexer state folded into the configuration key
   (C1, C2). This is the "hard on GPU" core.

3. **Bimodal allowed-set algebra and a ragged fused sampler.** Representation
   selection per state and a variable-width sampler that keeps top-k/top-p
   exact across a heterogeneous batch (C3, C10). This is what turns constrained
   sampling into something *cheaper* than unconstrained sampling.

4. **A graph-resident constraint step with O(1) clone and rollback**, which
   makes constrained speculative decoding practical for the first time (C5, C6),
   and makes jump-forward a compile-time table lookup rather than a runtime
   search.

5. **Drop-in integration and an honest crossover evaluation.** Real vLLM and
   SGLang backends, end-to-end serving numbers, and an explicit statement of
   the regimes where a GPU-native engine does *not* matter.

**The elegance claim.** One artifact — a GPU-resident map from parser
configuration to allowed-set representation — simultaneously serves masking,
sampling, speculative validation, rollback, and jump-forward. The five features
that are five separate mechanisms in existing systems collapse into one table.

**What the paper must not claim.** That it beats XGrammar on mask generation for
narrow toy grammars; that GPU-native execution helps a 70B model at batch 1;
that IELR(1) is broader coverage than XGrammar's general CFG. Each of those is
either measured to be marginal or simply false.

## Decision 3: benchmarks

### What the incumbents actually run — and the scope limits of their claims

Verified from the sources, not from summaries.

| System | Workload | Baselines | Metrics | Scope limit to exploit |
|---|---|---|---|---|
| XGrammar (MLSys'25) | Llama-3.1-8B-Instruct, prompts from `NousResearch/json-mode-eval`; two grammars: the dataset's JSON Schema and the full ECMA-404 JSON CFG | Outlines v1.0, llama.cpp b3998; end-to-end vs vLLM v0.6.3+Outlines | per-token mask time (<40 µs), TTFT, TPOT | **Batch 1 and 16 only.** "Near-zero overhead" (TPOT 6.2→6.3 at B=1, 9.0→9.2 at B=16) is never tested past 16, and never with speculative decoding. |
| llguidance / MaskBench | MaskBench inside JSON Schema Bench: 10k schemas, 2.5M tokens | XGrammar, Outlines, llama.cpp | mask time p50 and tail; <50 µs mean, <1% over 1 ms, 0.001% over 10 ms | Claims "16 cores and a 10 ms forward pass handle batch 3200". Assumes 16 **dedicated** cores, excludes the H2D mask copy, excludes speculative decoding, and ignores that the sampler stays O(V). |
| JSONSchemaBench (arXiv 2501.10868) | ~10k real schemas: Github trivial/easy/medium/hard/ultra, Glaive function signatures, JsonSchemaStore, Kubernetes, Snowplow, WashingtonPost | Guidance, Outlines, llama.cpp, XGrammar, OpenAI, Gemini | declared/empirical coverage, compliance, efficiency, quality | Coverage collapses on hard/ultra (OpenAI 9%, Guidance ~41% on Github_hard). This is the coverage bar, not a speed bar. |
| vLLM harness | `benchmarks/benchmark_serving_structured_output.py`, datasets `json`, `json-unique`, `grammar` (SQL EBNF), `regex`, `choice`, `xgrammar_bench` | any registered backend | TTFT, TPOT, throughput | De-facto industry harness. `json-unique` appends a UUID field per request to defeat schema caching — already the heterogeneous-batch stress test. |

### What gpugrammar runs

**Tier A — mask/step microbenchmark.** MaskBench protocol on the full JSON
Schema Bench corpus, reporting p50/p99/p99.9 plus device time and dispatch time
separately. Adds the axis MaskBench lacks: a batch dimension.

**Tier B — coverage.** JSONSchemaBench declared/empirical coverage and
compliance per split, plus the official JSON Schema Test Suite. Github_hard and
Github_ultra are the bar an IELR(1) engine must clear to prove it is not
narrower than a general-CFG engine.

**Tier C — end-to-end serving.** The vLLM harness across all six datasets with
gpugrammar, XGrammar, llguidance, and Outlines, plus SGLang via
`register_grammar_backend`.

**Tier D — the axes nobody has measured.** This is where the paper's
contribution lives:

1. batch sweep 32 → 2,048, since XGrammar stops at 16;
2. speculative decoding crossed with grammars, k = 1…8 — no prior work reports it;
3. sampler-inclusive accounting: report top-k/top-p cost, not mask cost alone;
4. `json-unique` at scale, so schema caching cannot hide per-request compilation;
5. p99.9 under cold-schema arrivals into a warm continuous batch.

### Workloads beyond JSON

JSON alone does not need LR power, so a JSON-only evaluation invites the
reviewer question "why a parser at all?". Required non-JSON workloads:

- **Spider / BIRD text-to-SQL** — the natural showcase for an LR engine and a
  direct comparison point with GRID's LALR(1) SQL work;
- **BFCL** and the Glaive split for tool/function calling;
- **HumanEval / MBPP with a Python grammar**, the SynCode setting;
- Kubernetes and JsonSchemaStore configs for deep nesting and `$ref` recursion.

### Reporting rule for competitive claims

Every comparison table must first **reproduce the incumbent's own claim on the
incumbent's own axis**, then extend that axis. A number that only exists outside
the regime the baseline was tuned for is not evidence; a number that matches
inside their regime and diverges outside it is.

## Non-goals

- Beating XGrammar on a grammar with ~15 allowed tokens per state. That is the
  most favorable possible case and is already demonstrated; it is not the paper.
- Supporting ambiguous grammars, or general CFGs beyond LR(1) power.
- Training-time or model-side changes. This is a decoding-time system.
- Any claim of general unbounded tokenizer-aware LR(1) support until C1–C3 are
  actually solved.

## Reporting rules

Decision 3 fixes the workloads; these rules govern how any number derived from
them is reported.

- **Baselines:** XGrammar, llguidance, Outlines, plus the closest parser-aware
  systems (Pre3, PSC) and Gram2Token where reproducible. The sampler baseline is
  FlashInfer (`top_k_top_p_sampling_from_logits`), not a `torch.sort` reference
  implementation — a sorted-softmax sampler is 5× slower and would be a strawman.
- **Metrics:** end-to-end tokens/s inside a real serving engine (not an isolated
  microbenchmark), constraint-step device time, dispatch overhead, table memory,
  compile latency, and mask exactness rate.
- **Ablations:** per representation (CSR / bitset / complement / interval),
  per stack abstraction, graph-resident vs. launched, IELR(1) vs. LALR(1) vs.
  canonical LR(1) table sizes.
- **Reporting rule:** every reported speedup states batch size, allowed-token
  distribution, and whether model execution is included. Isolated-microbenchmark
  speedups are never headlined.

## Working agreement

- Prototype code in this repository is evidence, not product. `gpugrammar` is a
  clean implementation informed by it.
- No claim enters README, GOAL, or the paper without a measurement or a proof.
- At every major milestone: run focused tests and the relevant benchmark, update
  documentation and results, commit with the Copilot co-author trailer, push.
