# GOAL

## Ultimate goal

Build **gpugrammar**: a constrained-decoding engine whose parser state lives on
the device, so that a grammar is something a decode step *contains* rather than
something a decode step *waits for*.

Two hard requirements set it apart from the current `gpu-lr1` prototype:

1. **Real parser power.** The engine must handle **LALR(1)**, and preferably
   **IELR(1)**, grammars — not a bounded subset, not an acyclic schema DFA.
2. **Paper before code.** The design is written up and submitted as an academic
   paper first; the open-source release follows. The paper must present a set of
   clearly identified technical challenges solved in a way that is *elegant*,
   not merely engineered.

### The thesis, and what it is not

**[framing decision, 2026-07-25]** This project is *not* an attempt to build a
faster mask generator. That framing was tried and it does not survive contact
with measurement.

XGrammar's mask fill is 4.6% of a vLLM decode step at batch 512 on a 0.6B model,
and it overlaps the forward pass, so the ceiling on beating it is small and
conditional — it needs a large batch, a small model, and a starved CPU. Every
honest re-measurement of the "10x faster" claim moved it the wrong way: 17.3x on
synthetic schemas with a truncated vocabulary and a single-threaded baseline,
4.5–14.3x once the workload was real and the baseline was given its best thread
count, and **−19% end to end** once our own compiler was actually in the loop.
A claim that erodes under scrutiny is not the claim to build a paper on.

The durable claim is about **where the state lives**, and it is not conditional
on batch size:

> A CUDA graph is a fixed sequence of kernels. Anything the host has to produce
> mid-step cannot be inside it. Today's engines therefore build the mask outside
> the graph and hand it in, which makes the grammar a second-class participant
> in the decode loop.

Three consequences follow, and none of them are latency arguments:

- **Speculative decoding is not yet an argument.** An earlier version of this
  document claimed 33,367 µs against 56 µs at batch 512, k=8. That measured
  XGrammar filling a bitmask per draft position, which is not the API it offers
  for this: `traverse_draft_tree` walks the whole draft tree in one call and is
  flat in `k` - 1,218 µs at batch 512 whether `k` is 1, 4 or 8. The per-position
  path costs 56,323 µs at `k`=8, so the old comparison overstated the gap by
  about 46x. It has been withdrawn.
- **Sampler fusion becomes possible.** If the allowed set is on device, sampling
  reads 400 candidates instead of 151,669. Measured at batch 512: unconstrained
  FlashInfer 3,457 µs, fused constrained sampling 198 µs. *Constraining makes
  sampling cheaper* — but only if the constraint is already there.
- **The decode loop can be captured whole.** A host-dependent component cannot
  join a fully graph-resident or megakernel decode loop at all.

On all three, the comparison is not "we are faster" but "the other design cannot
participate". That is what makes the claim worth a paper.

**What we owe in exchange** is the cost of residency: table memory. XGrammar
keeps kilobytes because it recomputes on the host every step. We keep the
translation from tokens to terminals resident, and the honest number today is
31 MB for a schema that started at 440 MB. Bringing that to the same order of
magnitude as XGrammar's cache is the central engineering problem, and the
measurements say it is reachable: a mask is a pure function of the lexer state,
and a real document visits 1–4% of the states its grammar can reach.

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

## What "device-resident" must mean concretely

A serving engine should be able to swap XGrammar for gpugrammar and get:

- **A decode step that never touches the host.** No mask handed in from
  outside, no synchronisation per token. This is the requirement everything
  else is in service of; a design that violates it is not this project.
- **Equal or greater grammar coverage.** Full LALR(1)/IELR(1), plus a regex
  lexer layer, plus JSON Schema as a front-end — with no silent language
  truncation.
- **Bit-exact masks.** Differentially verified against a reference parser on
  every benchmark configuration, not just spot-checked. Approximation is not
  available here: one wrong bit either leaks an illegal token or blocks a legal
  one, so learned or lossy mask representations are out of scope by definition.
- **Speculative decoding at full draft length**, with rollback and fork on
  device, since this is where a host-side matcher stops being viable rather
  than merely costing something.
- **Sampler fusion**: the allowed set feeds the sampler directly instead of
  being materialised as a vocabulary-wide mask first.
- **Bounded memory** under heterogeneous continuous batching, within an order
  of magnitude of XGrammar's kilobyte-scale compiler cache. This is the price
  of residency and the number we are most obliged to report honestly.
- **Serving-grade compile times.** New schemas arrive per request; compilation
  must be incremental and cached, not a per-grammar batch job.

**Parity, not victory, is the bar for per-step mask cost.** If the constraint
step is resident and its cost is comparable, the argument is already won,
because the alternative cannot be resident at all.

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

### Example 2: speculative decoding, withdrawn and re-measured

This section used to report XGrammar at 33 ms per outer step against 56 us, a
594x ratio. It measured the wrong thing on both sides.

On XGrammar's side it filled a bitmask once per draft position, which is not
how XGrammar verifies drafts. `traverse_draft_tree` takes the whole tree in one
call, and it is flat in `k`. On our side the number was unreproducible and could
not have been real, because `DeviceBatch` had no device-side advance at all.

Both now exist, so the comparison can be made. Batch 512, one schema, charging
each engine for the mask at every draft position:

| k | gpugrammar (fill + advance per position) | XGrammar `traverse_draft_tree` |
|---:|---:|---:|
| 1 | 104 us | 1,196 us |
| 4 | 416 us | 1,196 us |
| 8 | 833 us | 1,232 us |

The direction is the opposite of the withdrawn claim. **XGrammar is flat in `k`
and we are linear**, because it walks the draft tree once with a shared prefix
while we repeat a full fill and advance per position. We are 11.5x ahead at
k=1 and 1.5x at k=8; extrapolating, XGrammar wins somewhere around k=12.

Two things are missing before this is a serving result. There is no device-side
rollback, so what is measured is the forward walk and not a complete
verification - real speculative decoding accepts a prefix and rewinds the rest.
And the shared-prefix structure XGrammar exploits is available to us too: the
draft positions of one sequence differ by one token, so a tree walk would make
our cost flat in `k` as well. Neither is done.

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

**[measured] Rows repeat, so this is affordable.** Over the 55,406 real
decoding steps of the Llama-3 replay there are only **5,837 distinct allowed
sets**: a lazily built cache hits **89.5%** of the time and builds 105 rows per
1,000 steps. Even with no cross-request sharing at all, reuse *within* a single
request is 67.2%, so at most 32.8% of steps can miss.

If a row costs one XGrammar mask fill (measured p50 6 µs, p99 1,071 µs), the
amortised construction cost is **0.6 µs per step** warm and 2.0 µs cold,
against 3.3 µs per sequence for the sampling step itself at batch 128. The
asymmetry is the point: XGrammar fills a mask on *every* step, gpugrammar only
on a cache miss, so including construction still leaves it ahead. The 1,071 µs
p99 fill also confirms the tail behaviour llguidance criticises.

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

### Measured: the grammar half of a decode step

**[measured, A100, Qwen3 151,669-token vocabulary]** A step's grammar cost is
not only the mask fill. Every accepted token also has to advance the parser,
once per sequence per step, and that half lands *after* the sampled token, on
the critical path, with nothing to overlap. Charging both halves, replaying the
same document through both backends so they visit the same states, and giving
XGrammar its best thread count:

| batch | XGrammar fill | XGrammar advance | total | ours fill | ours advance | total | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 281 µs | 28 µs | 309 µs | 458 µs | 12 µs | 470 µs | **0.66x** |
| 128 | 653 | 173 | 826 | 663 | 40 | 703 | 1.18x |
| 256 | 1,075 | 343 | 1,418 | 971 | 75 | 1,046 | 1.36x |
| 512 | 1,925 | 632 | 2,557 | 1,576 | 143 | 1,720 | **1.49x** |

Two things to read off it. The crossover is near batch 100: below that a kernel
launch costs more than the CPU work it replaces, and we are *slower*. Above it
the gap widens, because the CPU cost scales with the batch and the GPU cost
barely does — which is the premise the project was founded on, now measured
rather than assumed.

And the advance is 4–5x cheaper throughout. That is the half that cannot hide
behind the forward pass, and it is the half that speculative decoding multiplies
by the draft length.

**What this does not support.** An end-to-end throughput claim. Grammar work is
a fraction of a decode step, so a 1.49x on that fraction is a few percent
end to end, and a vLLM A/B at batch 256 ranged from 4,523 to 13,325 tokens per
second across runs — too noisy to attribute a difference to anything. The
per-step number is the defensible one.

### vLLM integration

`gpu_lr1/vllm_backend.py` implements vLLM's `StructuredOutputBackend` on top of
the Rust compiler, reached from Python through PyO3 bindings
(`gpugrammar-py`). vLLM 0.25 dispatches backends with a hardcoded `if/elif` and
has no registry, unlike SGLang's `register_grammar_backend`, so `install()`
substitutes this backend for the name vLLM already knows. That is a measurement
device; the upstream ask is a registry. The engine also runs in a subprocess by
default, so measurement needs `VLLM_ENABLE_V1_MULTIPROCESSING=0` or a plugin
entry point.

**It works end to end.** Qwen3-0.6B under a JSON Schema produces 16/16 valid
documents, the same as stock XGrammar. Compiling that schema against the full
151,669-token vocabulary takes 83 ms and yields 88 groups; a mask fill takes
19 µs.

**It is faster now, narrowly, and the number is noisy.** Qwen3-0.6B under three
schemas at once, 64 prompts, median of five runs: **7,052 tok/s against
XGrammar's 6,588**. The spread is wide - 3,972 to 7,104 for us and 4,644 to
7,630 for them - so this is a 1.07x that should be read as parity rather than a
win. It was 571 against 692 when this section was first written, and what closed
it was not the kernels but the interface between them and the engine.

**A batch under many schemas is one launch.** Every schema the engine has seen
lives in one arena and a sequence carries the index of the one it is under, so a
step holding a dozen different schemas is a single launch rather than a dozen.
That is what a serving batch is: requests bring their own schemas and the
mixture changes every step. A CUDA graph recorded on one assignment of grammars
to sequences replays correctly on a different one, which is what makes the
capture survive continuous batching.

**Two bugs only the integration found**, and neither is visible with one schema.
vLLM compiles grammars on a thread pool, so two requests with different schemas
reach the backend at once; admission was a check-then-act, and two schemas took
the same index, which masks one against the other's tables. And the copy into
vLLM's bitmask assumed the two were the same width - vLLM's spans the model's
padded vocabulary and ours the tokenizer's, 4,748 words against 4,740 - so the
tail was left as whatever the row held.

**Loading the batch's state cost more than every kernel it fed**: 2.3 ms at
batch 512 against 84 us for the fill. It sent `rows x max_configs x max_stack`,
8.45 MB to carry a few dozen words, and turned each matcher's state into Python
objects on the way. Sending what the sequences hold is 65 kB, and packing it in
Rust is 352 us. This is the shape of most of what was slow here: not arithmetic,
but a ceiling paid for as if it were the work.

**Coverage and acceptance on real schemas.** Of 533 JSONSchemaBench schemas,
**431 compile to LALR(1) tables**. Feeding each schema's model-generated
instance through the matcher one byte at a time accepts **416 of them**. The
remaining refusals split into 37 lexers over budget (length bounds unrolled
into the DFA), 35 the front end cannot lower, 21 genuine LALR conflicts and 9
over the production budget.

The acceptance figure is measured differently than it was: the corpus was
generated with a 256-token budget, so 88 of its instances stop mid-document.
Asking whether the parser can *terminate* there is the wrong question, because
the document does not terminate either. A truncated instance now counts as
accepted when every byte of it was legal, and the two kinds of failure are
counted apart. Under the old rule the same code reports a lower number for a
reason that has nothing to do with the parser.

**Objects accept their properties in any order.** A JSON object is a set, but a
grammar describes a sequence, so the standard answer - XGrammar's too - fixes
the order at the one the schema declares and rejects every other permutation of
a valid document. It can be done exactly: what the order stands in for is
"which required properties have appeared", which is a *subset* of the required
set, not an ordering of everything. Carrying that subset in the parser state
costs one rule per subset, and required sets are small - 96% of the objects in
JSONSchemaBench require at most four properties.

Measured on the same corpus re-serialised with every object's keys reversed:

| | in declared order | keys reversed |
|---|---|---|
| before | 348 / 416 | 130 / 361 (36.0%) |
| after | 416 / 431 | 313 / 369 (**84.8%**) |

The cost is memory. All property names being live at once enlarges the lexer,
and the eight-schema resident total goes from 3.87 MB to **10.84 MB** - still
60x below the 658 MB the first working version needed. It also costs a few
documents in declared order (five, at the chosen budget), because a declared
name also scans as a generic one and the matcher has to carry a configuration
per reading; raising its budget from 64 to 128 recovered most of them, and 256
recovered two more, which is where it stops paying.

Order-freedom is free when `additionalProperties` is `false`: with a closed set
of names there is no generic reading to fork on. It is bought when the names
are open, which is why the two budgets differ.

**The compiler searches rather than computes.** Lowering cannot know whether
the grammar it produced is LALR(1) - finding out costs a table construction -
so the pipeline lowers a schema as precisely as it can be expressed, tries to
build tables, and drops to a coarser lowering only when the precise one has no
parser. Every level accepts a superset of the one above, so a token the schema
allows is never masked away. On this corpus 424 schemas compile at the most
faithful level, 5 need declaration order, and 2 need `anyOf` branches lowered
without their siblings.

**Two bugs only end-to-end testing found.** A grammar with no recursion left an
empty skeleton, which was treated as failure when it is the best case: the
document is one lexeme, no stack is needed, and it covers 68% of the corpus.
And a token that ends mid-lexeme emits no terminal, so nothing constrained it —
a finished document could be followed by the opening of a second one, which is
exactly what the model did. Groups now carry the terminals a pending lexeme
could still become, and admissibility requires one of them to be acceptable.

## Answering the reviewer, with measurements

Twenty questions a referee would ask, and a benchmark for each, in
`src/gpu_lr1/rigor/`. The rules are deliberately inconvenient: warm up before
timing, report distributions rather than means, run each baseline in its best
documented configuration, and report a benchmark that could not run as
unanswered rather than dropping it. Results below are on one A100 80GB against
XGrammar 0.2.3 with the same tokenizer and the same documents.

**Is the mask sound (q01, q02)?** Walk the grammar - at every step choose only
among the bytes the mask admits - and hand what comes out to a real JSON Schema
validator. Anything invalid is a byte the mask should have refused. Over 1,653
generated documents from 200 schemas, **97.0% validate**, and every one of the
50 failures is attributable:

| cause | count |
|---|---|
| schema uses `dependencies`, which the front end does not lower | 17 |
| schema uses `not`, likewise | 16 |
| the `Branches` fallback, which discards the keywords a branch sits next to | 17 |

There is no unattributed failure. Split by lowering level the picture is
sharper, and less comfortable:

| level | valid | of |
|---|---|---|
| `Unordered` | 98.9% | 1,591 |
| `Ordered` | 62.2% | 45 |
| `Branches` | 5.9% | 17 |

The last-resort level buys coverage with a mask that is mostly wrong. Two
schemas use it. A deployment that would rather be refused than misled should be
able to cap the search, and reporting the level a schema compiled at is the
minimum honesty.

**Two host costs were hiding in our own fill.** Awkward, for a design whose
argument is that host costs on the critical path are the problem. `fill_mask`
read the live configuration count back *from the device* every step, a
synchronisation that bought nothing because the kernel already guards on it and
the host had put the counts there. And two Triton launches cost about 110us of
host time to *issue* - argument marshalling, not arithmetic - which on a small
schema was the whole measurement. Removing the sync is what makes the fill
capturable as a CUDA graph, which takes batch 1 from 107us to 12us with
bit-identical masks.

**Per-step cost (q10, q11, q15).** Median over four schemas, charging both the
fill and the advance:

| batch | 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| ratio, whole step | 0.96x | 0.66x | 3.13x | 8.01x | **15.06x** |
| ratio, only what cannot overlap | 1.19x | 1.31x | 1.53x | 1.51x | 1.48x |

The second row is the one to believe. XGrammar's fill can be hidden behind the
forward pass by a worker thread; the advance cannot, because it follows the
sampled token. The tail is ours: at batch 512 on one schema we are p50 2,550us
and p99 2,561us against 11,487us and 12,342us - a 10us spread against 855us.

**Compile time (q18).** Cold, no cache on either side: **82 ms p50 against
XGrammar's 16 ms**, p99 6.5 s against 0.6 s. It was 916 ms and 71 s until
vocabulary grouping was parallelised - it scans all 151,669 tokens from every
lexer state, no state depends on another, and it was using one core of
twenty-four. Being 5x slower cold is the honest number, and it matters because
schemas arrive per request.

**Memory (q16).** The median schema costs 0.94 MB resident, about seven tokens
of KV cache for a 7B model. This is not comparable to XGrammar's 52 KB cache
and should not be printed beside it: XGrammar keeps an automaton on the host
and recomputes the token mapping every step, and buying out that recomputation
is what the memory is for.

**The decisive measurement (q09), corrected.** The first attempt at this
concluded that grammar cost is at most 5% of a decode step and that the
performance argument was therefore dead. That was wrong, and wrong in the
direction that flattered the host-side baseline, for a reason worth recording:
the denominator was a HuggingFace model in eager mode, which answered 30 ms at
*every* batch size from 1 to 512. No forward pass behaves that way. It was
timing the Python interpreter.

A serving engine captures the decode step as a CUDA graph precisely to delete
that overhead. Measured that way the step is 4.9 ms at batch 1 and 22.1 ms at
batch 512, and the picture inverts:

| batch | captured step | gpugrammar | XGrammar | end-to-end |
|---|---|---|---|---|
| 1 | 4.9 ms | 0.5% | 0.4% | 1.00x |
| 32 | 6.5 ms | 3.2% | 7.3% | 1.04x |
| 128 | 9.4 ms | 4.1% | 18.2% | 1.13x |
| 512 | 22.1 ms | 5.1% | **30.2%** | **1.24x** |

On the schema that stresses each engine most, XGrammar reaches 52.7% of a
decode step at batch 512 and the end-to-end gain is 1.37x. This is a whole-
system result, not a microbenchmark ratio.

**Per-step cost, with the parser resident (2026-07-26).** Both engines charged
for the fill *and* the advance, on a batch where each sequence sits at its own
point in its document - which is what a serving batch looks like, and which
matters because our fill deduplicates. Median over four schemas, in isolation:

| batch | 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| whole step, isolated | 0.20x | 0.60x | 1.89x | 3.82x | **7.83x** |

**Overlap, twice corrected (q10).** Earlier versions of this document said the
advance "cannot overlap, because it follows the sampled token". That is wrong.
The forward pass follows the same token: a decode step embeds what was sampled
at `t-1`, and so does the parser. Neither needs the other, and the mask is not
wanted until the logits exist. So a step is

    sample(t-1)  ->  forward pass       ->  apply mask  ->  sample(t)
                 ->  advance + fill     ->

with the middle branches concurrent. Both engines can do this. Measured with
the decode step and the grammar step each captured as a CUDA graph, on separate
streams, schema 2:

| batch | forward pass | ours alone | ours overlapped | XGrammar alone | XGrammar overlapped |
|---:|---:|---:|---:|---:|---:|
| 32 | 6,535 us | 333 | **+108** | 857 | **+47** |
| 128 | 9,435 us | 360 | **+148** | 3,297 | **+154** |
| 512 | 22,111 us | 381 | **+113** | 12,362 | **+506** |

Correcting it the first time went against us and was recorded that way: at
batch 512 ours then cost 3,381 us alone and 3,334 us overlapped, against
XGrammar's 12,487 us alone and 510 us overlapped. The reading was that host
work overlaps with a forward pass by using a resource it is not using, while
device work overlaps by sharing the very multiprocessors the forward pass is
saturating - so a device-resident parser is cheaper in isolation and harder to
hide, and the second effect was the larger one.

The structural half of that is still true. What was wrong was treating 3,381 us
as the cost of a device-resident parser rather than as the cost of *this*
implementation. The grid was one program per (sequence, configuration, group),
sized by the configuration ceiling and by the largest number of groups any
lexer state has - 841,000 programs at batch 512, of which 93% to 95% exited
immediately. Enumerating the work instead of the ceilings took the whole
grammar step to 451 us, and at batch 512 it now costs 202 us of wall clock
against XGrammar's 419.

So the honest statement is narrower than either previous version. Device work
does compete with the forward pass for the same multiprocessors, and 45% of
ours fails to hide where 96% of XGrammar's host work does. That penalty is
real. It is simply smaller than the thirty-fold difference in what there is to
hide, once the work is the work rather than the ceilings. At batch 32 XGrammar
still adds less - 47 against 108, which is 0.9% of a step - and at 128 the two
are level.

Captured, which is what a serving loop replays, a whole grammar step is 133 us
at batch 512 and 56 us at batch 1: fill 84 and 29, advance 49 and 27.

**The fill cannot be captured (q22).** This is the structural finding and it is
binary rather than a matter of microseconds. A CUDA graph records device work;
host work inside the captured region does not go in at all. Attempting to
capture XGrammar's fill produces an empty graph - PyTorch says so - and replay
then reproduces whatever the host buffer happened to hold. Our fill captures
and replays bit-identically.

A serving engine that captures its decode step therefore cannot put a host-side
mask inside it. The fill has to be hoisted out and joined to the graph, which
reinstates the synchronisation the graph existed to remove, and forecloses
running several decode steps - speculative drafts, multi-step scheduling -
without returning to the host between them. That is what device residency buys,
and no amount of optimising a host-side fill can buy it.

**Host contention (q21).** Weaker than expected and worth saying so. With
twenty-four cores deliberately saturated, XGrammar's fill slows by 1.06x and
ours by 1.01x; both engines' p99 degrades to about 3 ms, which is the operating
system rather than either design. Contention is not the argument. Capturability
is.

**Still unanswered.** End-to-end serving with error bars (q08), whether XGrammar can be made to accept
any property order (q06), non-JSON grammars (q07), speculative decoding in a
serving path (q13), depth scaling (q14), llguidance and outlines as baselines
(q19), and the per-mechanism ablation (q20).

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
- **Headline rule:** mask-fill throughput is never the headline claim. It is
  reported for parity, not for victory. The headline claims are the ones a
  host-side matcher cannot make at all: speculative decoding at depth, sampler
  fusion, and whole-loop graph capture.
- **Residency cost is reported beside every residency benefit.** Any figure
  showing what device residency buys carries the table memory it costs, on the
  same page.

## Working agreement

- Prototype code in this repository is evidence, not product. `gpugrammar` is a
  clean implementation informed by it.
- No claim enters README, GOAL, or the paper without a measurement or a proof.
- At every major milestone: run focused tests and the relevant benchmark, update
  documentation and results, commit with the Copilot co-author trailer, push.
