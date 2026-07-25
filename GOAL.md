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

**Status.** Implemented in `src/gpu_lr1/ragged_sampler.py` and
`src/gpu_lr1/wide_sampler.py`, verified by `tests/test_ragged_sampler.py`
against a sorted reference. Three findings drove the optimisation, each
measured rather than assumed:

1. **Hidden syncs, not kernels.** Two `.item()` calls in the first dispatcher
   cost 270 µs per step. Row widths are now cached at table construction and
   bucketing uses an in-kernel early exit, so both bucket kernels launch over
   one grid and the step never touches the host.
2. **Occupancy, not bandwidth.** fp16 logits did not speed the wide kernel up
   at all, and block/warp tuning plateaued, which ruled out bandwidth and
   compute. One program per sequence left batch 128 with 128 programs for 108
   SMs. Splitting each wide row into chunks — 64 at batch 1, down to 2 at batch
   2,048 — gave 3.5x at small batch.
3. **Launch count, not work.** The split path issues about twenty launches, so
   the step became dispatch-bound again. Capturing it as a CUDA graph collapsed
   that to one replay: narrow rows went from 493 µs to 28 µs at batch 1.

Final measurement on A100 with Qwen3 and XGrammar's builtin JSON grammar,
graph-replayed, median wall (`results/a100-ragged-sampler.json`):

| profile | batch | gpugrammar | FlashInfer unconstrained | XGrammar full path | vs unconstrained |
|---|---:|---:|---:|---:|---:|
| narrow | 1 | 28.3 µs | 289.3 µs | 1,041.6 µs | 10.2× |
| narrow | 512 | 35.3 µs | 3,471.0 µs | 8,985.1 µs | 98.4× |
| narrow | 2,048 | 114.7 µs | 12,991.3 µs | 32,009.2 µs | **113.3×** |
| mixed | 1 | 88.8 µs | 282.1 µs | 1,031.9 µs | 3.2× |
| mixed | 128 | 732.4 µs | 1,086.9 µs | 5,988.7 µs | 1.5× |
| mixed | 2,048 | 8,927.5 µs | 13,027.8 µs | 28,954.4 µs | 1.5× |
| wide | 1 | 89.3 µs | 282.4 µs | 1,009.2 µs | 3.2× |
| wide | 2,048 | 16,850.9 µs | 13,009.1 µs | 26,001.3 µs | 0.8× |

A realistic mixed batch is now faster than sampling with no constraint at all
at every batch size, by 1.4–3.2×, and 3.2× faster than the deployed XGrammar
path at batch 2,048. C10 is retired.

### Measured on the real workload

Earlier profiles were synthetic in both directions and the comparison charged
each engine differently. The measurement below fixes both.

**Workload.** `gpu_lr1.generate_instances` has Llama-3-8B-Instruct produce 533
JSON values under a real XGrammar constraint — 50 schemas from each of the 11
JSONSchemaBench configs, sampled at temperature 0.8 / top-p 0.95, up to 256 new
tokens, mean 125 tokens. `gpu_lr1.replay_tokenizer` then replays that text
through each tokenizer's grammar, which is faithful because the state a matcher
reaches is determined by the bytes consumed. Content and tokenization are
separated so vocabulary effects are isolated from schema effects.

**Cost model.** Both engines are charged for everything they must do per step.
XGrammar pays mask fill, pinned H2D, mask apply, FlashInfer sampling **and**
`batch_accept_token` plus rollback — the advance alone is 4.6–22.2% of its CPU
work and was previously omitted. Its thread count is swept over 1/2/4/8/16/auto
and the fastest is reported, and its device work is CUDA-graphed, as ours is.

**Width is set by the schema, not the vocabulary.** Across three tokenizer
families the median step allows a few hundred tokens no matter how large the
vocabulary is, so the O(V)-versus-O(allowed) ratio grows with vocabulary size:

| tokenizer | vocab | steps | median allowed | wide (>8,192) | forced |
|---|---:|---:|---:|---:|---:|
| Llama 3 | 128,256 | 55,406 | 396 | 32.9% | 0.8% |
| Qwen 3.6 | 248,077 | 66,557 | 378 | 32.0% | 0.7% |
| Gemma 4 (SentencePiece) | 262,144 | 69,313 | **107** | 32.3% | 0.0% |

Per split the spread is wide: Glaive function calls are only 11–12% wide, while
WashingtonPost is 46–54%.

**Result** (A100, graph-replayed, median wall):

| tokenizer | batch | gpugrammar | FlashInfer unconstrained | XGrammar full path | gap |
|---|---:|---:|---:|---:|---:|
| Llama 3 | 32 | 222.4 µs | 299.7 µs | 1,954.8 µs | 8.8× |
| Llama 3 | 128 | 416.0 µs | 628.3 µs | 5,963.5 µs | **14.3×** |
| Llama 3 | 512 | 1,464.2 µs | 1,984.9 µs | 13,981.4 µs | 9.5× |
| Qwen 3.6 | 32 | 291.2 µs | 917.2 µs | 3,011.7 µs | 10.3× |
| Qwen 3.6 | 128 | 771.8 µs | 1,755.3 µs | 6,039.5 µs | 7.8× |
| Qwen 3.6 | 512 | 2,234.1 µs | 5,697.8 µs | 13,896.6 µs | 6.2× |
| Gemma 4 | 32 | 310.8 µs | 607.4 µs | 2,969.7 µs | 9.6× |
| Gemma 4 | 128 | 874.1 µs | 1,200.8 µs | 7,826.4 µs | 9.0× |
| Gemma 4 | 512 | 2,439.1 µs | 4,025.2 µs | 10,937.2 µs | 4.5× |

The honest range is **4.5–14.3×**, not the 12–35× an unfair thread setting and a
hand-written schema produced.

**Table memory is now reportable.** Wide rows keep only a bitset plus a default
successor and a small override list; their token lists are dropped outright,
which `tests/test_ragged_sampler.py` checks does not change a single sampled
token:

| tokenizer | resident tables | as plain CSR | reduction |
|---|---:|---:|---:|
| Llama 3 | 24.5 MiB | 935.0 MiB | 38.2× |
| Qwen 3.6 | 36.6 MiB | 1,807.7 MiB | 49.4× |
| Gemma 4 | 34.7 MiB | 1,952.6 MiB | 56.3× |

Tens of MiB is defensible next to a KiB-scale compiler cache only because it
buys a GPU-resident step; the paper must report it, not hide it.

**Threats that remain.** Table construction is still excluded — rows are
replayed, while XGrammar computes masks online from a compact automaton, so
compile time and incremental admission (C7) must be measured before any
end-to-end claim. There is one GPU, one generating model, and no serving
integration, so these are isolated-step numbers and cannot be headlined.

**What is left.** Pure-wide batches above 128 still run at 0.8× of
unconstrained because the search costs ten sweeps of the vocabulary. Replacing
the multi-probe search with a single-pass histogram would cut that to about
four sweeps; it is the only remaining algorithmic gap.

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
