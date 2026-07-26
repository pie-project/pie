# Auditable Weight Loading — Paper Design Notes

> **Thesis: _Don't trust the loader. Check the plan._**
>
> Weight loading is currently a trusted, unverified step in the ML supply chain.
> We make it a verified, attestable one — not by trusting the loader, but by
> making it emit a declarative certificate (the `LoadPlan`) whose properties are
> machine-checkable, whose compilation is deterministic, and which can be signed
> and independently re-verified.

This document sketches the motivation, contributions, and evaluation for a paper
built on `pie-loader` (`loader/`). It also records, with citations, which claimed
properties **already hold in the code** and which are **still missing** — the
missing ones are precisely the paper's technical contributions.

---

## 1. Motivation

### 1.1 The loading gap

Model supply-chain security stops at the file. Sigstore / model-transparency
efforts and safetensors signing all attest the checkpoint **at rest**. Nothing
attests the step that turns those bytes into tensors resident in device memory.
That step is arbitrary glue code.

Consequently, a question that should be trivial is currently unanswerable:

> *Is the checkpoint I audited actually the one being served?*

The file hash can verify fine while the loader materializes different weights,
and nothing detects it.

### 1.2 Why this particular gap is the worst one

1. **The loader is the ideal attack surface.** It is untyped, per-model, rarely
   reviewed, and runs with full privileges before any inference happens. A
   one-line change to a name-matching rule can swap expert weights or truncate a
   shard. No test catches it, because the output is still plausible text.

2. **It breaks without an adversary.** TP-degree changes, quantization
   conversions, and fused-QKV splits are the largest source of "the model runs
   but is subtly worse" bugs. Failures are silent by construction.

3. **This repository contains a live specimen.** The ABI lowering layer is full
   of silent early-exits that skip a tensor when a shape or name predicate does
   not match, returning `Ok(())` rather than an error:

   | file | silent `Ok(())` / `Ok(None)` sites |
   |---|---|
   | `src/abi/nemotron.rs` (212 lines) | `:11, :70, :73, :80, :86, :90, :95` |
   | `src/abi/fusion.rs` (236 lines) | `:33, :71, :98, :127, :130, :192, :202, :205` |
   | `src/abi/qwen_moe.rs` (239 lines) | `:8, :79, :83` |
   | `src/abi/gpt_oss.rs` (268 lines) | `:8` |

   These are exactly the vulnerability class the paper is about. **Do not fix
   them before writing the paper** — they are the most compelling motivating
   example available, and the coverage checker (§3, C3) converts them from
   silent corruption into compile-time errors. Showing that on our own system is
   worth more than showing it on someone else's.

4. **The existing defense against exactly this is vacuous.** The C++ driver
   already contains what looks like the coverage check C3 proposes:

   ```cpp
   // driver/cuda/src/model/loaded_model.cpp:327
   if (planned_load.covered_contract_count != planned_load.runtime_tensor_count) {
       throw std::runtime_error(
           "engine: Rust loader did not cover the full RuntimeABI; covered " ...);
   }
   ```

   Both operands are assigned from a single variable, `view.tensors.len`, in
   `driver/cuda/src/loader/load_plan_bridge.hpp:131-137`. The condition is
   `if (x != x)` and cannot fire. What it checks is that the plan covers the
   tensors *the plan declares* — the compiler grading its own output.

   This is the sharpest available argument for C3 and C5. The check is not
   missing because nobody wanted it; it is missing because a coverage claim is
   meaningless unless the contract has an author independent of the compiler.
   That is the property the paper supplies, and this is the specimen that shows
   good intentions are not sufficient without it. **Also do not fix this before
   writing the paper.**

### 1.3 The intellectual move

The layout algebra is deliberately restricted: no general gather, `Select` and
`Partition` are contiguous ranges on a single axis, `Reorder` is an axis
permutation, `View` is a reshape (`src/ir.rs:44-136`). In a
performance-framed paper this restriction is a limitation to be defended. Here
it is **the enabling property**, and this reversal is the paper's spine:

> An arbitrary Python loader is Turing-complete, so auditing it is undecidable.
> This algebra is total, first-order, and strided-closed, so the loading
> transformation is a **finite object whose audit predicates are decidable**.
> The restriction is not a performance trick — it is the precise condition that
> makes auditing possible.

Without this connection the paper degenerates into "we added a checksum." With
it, the algebra and the security result are one argument.

---

## 2. Definition

A plan `P` is a **valid witness** for (checkpoint `C`, model config `M`,
deployment `D`) iff:

| predicate | meaning |
|---|---|
| **Type soundness** | every expression's declared shape / dtype / encoding / quant scheme is consistent, and matches the runtime contract |
| **Disjointness** | no two persistent buffers overlap in the arena |
| **Coverage** | every byte of every realized tensor buffer is written by exactly one instruction |
| **Provenance** | every written byte traces to a `(file_id, offset, span)` in `C`, and transitively to a content digest |
| **Semantic preservation** | optimizer rewrites preserve denotation |
| **Determinism** | compiling `(C, M, D)` twice yields byte-identical `P` |

Together these give an end-to-end chain:
**signed checkpoint bytes → plan certificate → resident device bytes.**

---

## 3. Contributions

### C1 — Problem formulation
First identification of the *loading gap*: signed checkpoints combined with
unverified loaders. Formal definition of auditable weight loading (§2).

### C2 — The algebra as an audit substrate
The decidability argument of §1.3, contrasted with Turing-complete loaders.

### C3 — The verification procedure

Honest status of each predicate in the current tree:

| predicate | current state | citation | verdict |
|---|---|---|---|
| Type soundness | full checker over all 15 expression forms | `src/typecheck.rs` (616 lines) | **exists** |
| Disjointness | arena base alignment, pairwise non-overlap, `CreateView` window containment | `src/planner/arena.rs:99-154` | **exists** |
| Determinism | all `HashMap`/`HashSet` uses are lookup-only; the arena sort key carries `buffer.id` as a total-order tiebreaker | `src/planner/arena.rs:69-71` | **holds, unproven** |
| Semantic preservation | denotational interpreter + differential tests asserting pre/post-optimization equality | `src/reference.rs`, `tests/algebra.rs` (10 tests) | **hand-written examples only** |
| **Coverage** | — | — | **MISSING — core contribution** |
| **Provenance (cryptographic)** | no `sha` / `digest` / `checksum` / `blake` / `crc` anywhere in `src/` | — | **MISSING — core contribution** |

**Coverage is the real hole.** `validate_persistent_layout` proves buffers do
not *overlap*; it never proves every byte is *written*. If part of a persistent
buffer is never written, garbage weights are served with no error raised. The
`DestExtent` records needed to decide this are already present in the emitted
plan (`src/load_plan.rs:108-113`), so this is an interval-covering check over
data the compiler already produces.

Note the deliberate layering: disjointness is *safety*, coverage is
*completeness*, provenance is *binding to actual bytes*. All three are needed;
the tree currently has one.

### C4 — Reproducible compilation as deployment identity
Deterministic compilation makes the plan a reproducible build artifact. Two
independent parties compile and obtain byte-identical output, so **the plan hash
is an attestable identity for "this deployment's weights."** The existing
content-hash `compiler_version` (`build.rs` → `PIE_LOADER_COMPILER_HASH`,
consumed at `src/load_plan.rs:22-26`) already provides compiler-side
invalidation and slots directly into this story.

### C5 — A small independent checker (translation validation)

**This is the crown jewel.** Do not verify the compiler; verify each
compilation — the CompCert verified-validator / Pnueli translation-validation
lineage.

- Compiler is ~12.9k lines; the checker should be ~500. **TCB shrinks ~25×.**
- Decisively, this **neutralizes the weakest part of the system.** The
  hardcoded per-architecture passes and silent fallbacks in `src/abi/` are not
  in the trusted base, because neither the ABI builder nor the optimizer needs
  to be trusted — only the output is checked. The reviewer objection "you still
  hand-write per-model code" stops being fatal.
- The checker can live inside the **C++ executor**, which already parses the
  plan (`driver/common/include/pie_native/load_plan.hpp`), so the executor can
  refuse to run an unchecked or unsigned plan.

---

## 4. Evaluation

### E1 — Fault injection *(headline experiment)*

Mutation operators over the loading pipeline, modeling both real bugs and real
attacks:

- swap two tensor names (expert 3 ↔ expert 7)
- off-by-one on a shard boundary (`Select` start ± 1)
- drop the final row of a fused-QKV split
- **silently skip a tensor** — i.e. the `Ok(())` fallbacks of §1.2
- omit a transpose (`Reorder` permutation altered)
- misassociate quantization scales (`Attach` wrong metadata)
- truncate `span_bytes`
- stale file offset (checkpoint updated, plan cached)

Measure detection rate for (i) type checker alone, (ii) + coverage/disjointness,
(iii) + provenance digests. Baselines: HF `transformers` / vLLM loader error
handling, loss comparison, benchmark output equality.

**Target result:** *"N of these mutations produce no error and plausible output
under existing loaders. Our checker rejects 100% of them at compile time in
X ms."*

**Report honestly:** which mutations are undetectable in principle without
content digests. This motivates the provenance contribution and demonstrates
intellectual honesty.

### E2 — Cross-deployment equivalence *(a claim nobody can currently make)*

One checkpoint; compile TP ∈ {1,2,4,8} × backend ∈ {CUDA, Metal} × quant ∈
{fp16, mxfp4} = 16 plans. Verify each, then verify that the consumed source byte
ranges, modulo the declared transformation, are identical:

> *"These 16 deployments provably serve the same model."*

Include a negative case: inject a wrong shard `Select` on one rank and show the
checker reports it.

### E3 — Cost of auditing

Checker time vs. compile time vs. actual load time, from 1B → 70B → 400B+ MoE.
Expected: checker in milliseconds, loading in seconds-to-minutes, so audit
overhead < 0.1%. Also report plan size and instruction count vs. model size.

Digest verification is *not* free (I/O bound). Report two modes:

1. **Fold hashing into the DMA path** — the bytes are read anyway, so this is
   near-free.
2. **Per-tensor Merkle tree for partial verification** — TP rank *r* verifies
   only the 1/N of the file it actually reads. Partial verification for sharded
   loads is a genuinely novel sub-result.

### E4 — Checker independence and TCB

LoC comparison. Demonstrate independence by reimplementing the checker in a
different language (inside the C++ executor). **Mutation-test the checker
itself** to prove it is not vacuous — checker mutants must be caught by the E1
corpus.

### E5 — Real-bug case study

Mine vLLM / HF `transformers` / llama.cpp history and issues for actual
weight-loading bugs (RoPE dimension splits, GQA head grouping, MoE expert
ordering, GGUF quant block misparsing). For each, classify whether the checker
would have caught it. This grounds the mutation set in reality rather than
imagination.

### E6 — Adversarial scenario

Threat model: the attacker controls loader code (malicious PR, compromised
dependency) but not the signed checkpoint.

| attack | caught by |
|---|---|
| inject backdoored weights from an unsigned side file | provenance (no source tensor for those bytes) |
| reorder experts using only signed bytes | coverage/disjointness pass, but plan hash ≠ attested hash |
| forge the plan as well | signature failure |

State the trust boundary crisply: compromise of the attestation authority or of
the model config defeats the scheme.

### E7 — Semantic preservation at scale

Replace the 10 hand-written equivalence tests with **property-based random plan
generation** differentially evaluated through `src/reference.rs`. Check every
rewrite that fires during real-model compilation. The rewrite rules under test:
`collapse_selects` (`src/optimizer.rs:721`), `cancel_partition_join` (`:756`),
`coalesce_reorders` (`:793`), `elide_identity_views` (`:825`),
`push_select_through_join` (`:492`), `push_select_through_decode` (`:548`),
`normalize_encode` (`:604`), `normalize_cast` (`:266`).

---

## 5. Engineering roadmap

Ordered by leverage:

1. **Coverage checker** — does not exist; the paper's central theorem. Interval
   covering over `DestExtent`. Unblocks the E1 headline experiment on its own.
2. **Per-tensor content digests (Merkle)** — upgrades provenance from structural
   to cryptographic.
3. **Promote determinism to a tested property** — currently true by
   construction, not by test. Compile N× across platforms and assert byte
   identity. Add a guard so the `HashMap` iteration at
   `src/planner/passes.rs:219-256` stays telemetry-only (it is inside an
   `eprintln!` diagnostic block today and does not affect the emitted plan; if
   it ever feeds emission, determinism breaks).
4. **`src/abi/` silent fallbacks** — leave them. Use as the motivating example
   (§1.2), then show the coverage checker catching them.

### Existing invariants worth citing as prior structure

- `src/planner/arena.rs:99-154` — the documented three-part persistent-layout
  invariant (alignment, non-overlap, view containment) is already well built and
  is the model to follow for the coverage checker.
- `src/typecheck.rs` — re-run after *every* optimizer iteration
  (`src/optimizer.rs:33, :62`), so type soundness is maintained as an inductive
  invariant, not just checked once.
- The fixed-point loop is capped at 32 iterations (`src/optimizer.rs:47-53`).
  For publication, either prove each rule strictly decreases a well-founded
  measure, or explicitly justify the cap. Reviewers will find this.

---

## 6. Positioning

**Venue.** This framing targets USENIX Security or NDSS. Keeping the
performance results and presenting auditability as something obtained *for free*
on top of an optimizing compiler also makes OSDI/SOSP viable. MLSys fits the
performance-first framing instead.

**Title candidates.**
- *Don't Trust the Loader: Verifiable Weight Materialization for ML Deployments*
- *Closing the Loading Gap: Attestable Weight Loading for LLM Serving*
- *Proof-Carrying Weight Loads*

**Related work to distinguish from.**
- ServerlessLLM (OSDI'24) and loading-optimized checkpoint formats — they change
  the *format*; we keep the format fixed and compile a *plan*, then verify it.
- Sigstore / model-transparency / safetensors signing — they attest the file at
  rest; we attest the transformation.
- CompCert, translation validation (Pnueli et al.) — our C5 is the direct
  methodological ancestor, applied to data movement rather than computation.
