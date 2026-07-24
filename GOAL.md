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

*Why hard, and why it is the most promising contribution:* real grammar states
are bimodal — either a handful of allowed tokens (structural positions: 731
tokens) or almost the entire vocabulary (string bodies: 146,924). A single
representation cannot serve both. The elegant answer is a **mask algebra** whose
cost is O(exceptions) rather than O(allowed) — complement/negative sets,
interval or run-length encoding over sorted token IDs, hierarchical bitsets —
with per-state representation selection and automatic kernel routing, proven
equivalent across representations.

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

## Non-goals

- Beating XGrammar on a grammar with ~15 allowed tokens per state. That is the
  most favorable possible case and is already demonstrated; it is not the paper.
- Supporting ambiguous grammars, or general CFGs beyond LR(1) power.
- Training-time or model-side changes. This is a decoding-time system.
- Any claim of general unbounded tokenizer-aware LR(1) support until C1–C3 are
  actually solved.

## Evaluation plan (paper-grade)

- **Baselines:** XGrammar, llguidance, Outlines, plus the closest parser-aware
  systems (Pre3, PSC) and Gram2Token where reproducible.
- **Grammars:** JSON Schema suite, a tool-call/function-call grammar, a
  programming-language grammar, and a recursive stress grammar — chosen so that
  wide-row states (C3) and deep stacks (C2, C6) are actually exercised.
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
