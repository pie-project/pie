# PTIR Implementation Review Findings

Code review of the uncommitted PTIR optimizing-compiler implementation
(2026-07-12, xhigh effort, 12 finder angles + adversarial verification).
Findings were re-anchored against the current tree after the rev-4 refactor
and the Metal remediation pass on 2026-07-12 (annotations removed, exact
nucleus SSA recognized as an internal library region, and structured masks as
first-class ops). Ranked most severe first within each section.

Verdicts: CONFIRMED = trigger and wrong output identified and quoted.
PLAUSIBLE = mechanism real, trigger latent or environment-dependent.
RESOLVED = current-tree fix and regression coverage verified.

## 1. Confirmed correctness bugs

### 1.1 Grouped nucleus-library reduction data race (Metal) — RESOLVED

`driver/metal/src/pipeline/m1_codegen.cpp:878-951`.

Both canonical width-32 reductions now ping-pong between disjoint
`reduction_a` and `reduction_b` arrays, swap input/output only after a
device-memory barrier, and never read and write the same reduction level in
place. `driver/metal/tests/m1_generated_test.cpp:4330-4683` pins the emitted
two-pass structure and executes singleton, fused, grouped, and production
vocabulary paths. The V=248320 device gate passed.

### 1.2 Negative-cache poisoning on transient failures (Metal) — RESOLVED

`driver/metal/src/pipeline/m1_runtime.cpp:754-864,942-1205` and
`driver/metal/src/context.cpp:420-485`.

`compile_program` now reports `Deterministic` versus `Retryable` failure.
Only the deterministic lambda calls `remember_negative`; Metal compiler,
archive/cache-directory IO, and program/stage cache-capacity failures use
the retryable path. Driver registration returns `PIE_STATUS_DRIVER_ERROR`
for retryable failures and leaves `record.m1_error` empty, while structural
unsupported plans remain `PIE_STATUS_UNSUPPORTED` and are negative-cached.
Fault-injection, real blocked-cache-directory recovery, capacity, and
driver re-registration regressions are at
`driver/metal/tests/m1_generated_test.cpp:4711-4869,4925-4996`.

### 1.3 On-disk cache identity truncates versions to 4 bits (Metal) — RESOLVED

`driver/metal/src/pipeline/m1_runtime.cpp:25-54,168-180`.

The first 22 bytes now follow Rust `ExecutableCacheKey::encode` (including
the full u16 compiler version and exact semantic-mode byte), followed by
fixed-width full compiler/PTRP/lane-table/emitter fields. No nibble masks
remain. `driver/metal/tests/m1_generated_test.cpp:1215-1240` verifies that
changing any version from 0 to 16 changes the cache identity.

### 1.4 CUDA generated-module negative cache — OBSOLETE

The `generated_epilogue.hpp` backend and its module/negative cache were
removed. CUDA now keeps grouped Tier 0 as the permanent execution floor, so
there is no generated module load to poison and this finding no longer has a
current-tree target.

## 2. Plausible / latent correctness traps

### 2.1 Grouped effect-PSO failure cached as degraded positive entry (Metal)
— RESOLVED

`driver/metal/src/pipeline/m1_runtime.cpp:1187-1215`.

Grouped readiness and commit now compile sequentially and each failure
returns `Retryable` before `programs.emplace`; no executable with null
grouped PSOs can enter the positive cache. The same policy now covers
grouped region compiler failures (`:1111-1121`). Deterministic emitter
unavailability remains an explicit reason with the generated singleton
fallback. Fail-once recovery for both shared PSOs is pinned at
`driver/metal/tests/m1_generated_test.cpp:4792-4845`.

### 2.2 FP32 grouped logits path: physical vs logical row stride (CUDA)
— RESOLVED

`k_grouped_copy_dynamic_root` now maps each logical `(row, column)` through
the physical model row stride while keeping the compact destination
logical-vocab packed. Group admission rejects a physical stride shorter than
the logical vocabulary. `fp32_dynamic_root_stride_case` in
`driver/cuda/tests/ptir_grouped_test.cu` covers three rows with
`logical_vocab=5`, `physical_stride=8`, including hostile padding values.

### 2.3 Single-member forward never resets the PTIR logits cursor (Metal)
— RESOLVED

`driver/metal/src/batch/forward.cpp:1795-1806`.

Every `forward()` call resets `ptir_logits_next_row_` and ensures staging
capacity before either the paged or linear path, matching
`forward_batch` (`:1842-1852`). The repeated-call checkpoint regression is
at `driver/metal/tests/ptir_checkpoint_e2e_test.cpp:377-400`.

### 2.4 Symbolic extents dropped by rank-changing ops (Rust compiler)

`interface/ptir/src/compiler.rs:578` (`symbolic_result_type` default arm)
— PLAUSIBLE, latent.

The default arm propagates a symbolic extent only from an operand whose
rank equals the result rank. The structured mask ops append the `len` axis
(operand rank = result rank - 1), so a symbolic input would type the result
fully static: over-specialized cache keys (churn per step) and, if the
runtime extent diverges from the baked value, a wrong static row count.
Unreachable with today's emitters (channel reads are static); becomes real
the moment masks are driven by symbolic query/key extents.

### 2.5 C++ `Intrinsic` enum ordering diverges from wire values

`driver/common/include/pie_native/ptir/trace.hpp:93` — PLAUSIBLE, latent.

Enum order (ValueHead=3, Query=4, MtpDrafts=5, no Layer) differs from the
wire `PTIR_INTR_*` values (Query=3, ValueHead=4, Layer=5, MtpDrafts=6).
Everything currently routes through `map_intrinsic` (`bound.hpp:121`), but
any future `static_cast<Intrinsic>(op.intr)` silently swaps Query/ValueHead
and mis-tags MtpDrafts. Align the enum values with the wire constants.

## 3. Hot-path efficiency (per-fire / per-token waste)

### 3.1 Per-fire deep copy and re-hash of the registered plan (CUDA)
— RESOLVED

`PtirProgramCache` owns decoded immutable `StagePlan` objects, and staged
lanes retain `const StagePlan*` references. No `build_execution_plan` or
per-fire plan merge remains.

### 3.2 Per-token MTLBuffer allocation and residency churn (Metal)
— RESOLVED

`driver/metal/src/mtl4_context.mm:316-454` and
`driver/metal/src/pipeline/m1_runtime.cpp:1233-2965`.

M1 prepared-fire storage and M1/M2/M3 command scratch/metadata now use a
power-of-two size-classed resident pool. Reuse performs no allocation or
residency-set mutation; recycling happens only after the synchronous
completion fence/finish path. The pool is capped at 1 GiB overall and eight
cached buffers per class, evicts idle classes when needed, never CPU-clears
vocabulary scratch, and exports allocation/reuse/in-flight/resident/peak/
capacity metrics. Bounds/no-clear/reuse and M1/M2/M3 completion coverage is
at `driver/metal/tests/m1_generated_test.cpp:1242-1269,1684-1959,4874-4888`.
The latest production run reported 42 allocations, 314 reuse hits, 18,432
resident bytes, zero in-flight buffers, and a 1,073,741,824-byte cap.

### 3.3 Generated CUDA device-property/string key rebuild — OBSOLETE

This belonged to the removed `generated_epilogue.hpp` backend. The current
grouped Tier 0 path has neither that device-property query nor its string
cache key.

### 3.4 Per-fire graph-key rebuild (CUDA) — RESOLVED

Registration now precomputes each immutable stage/signature/region identity
in `PtirProgramCache`. `compiled_program_set_hash` performs one allocation-free,
order-independent fold over those integer identities plus row buckets and
exact multiplicity; it no longer copies signature bytes, performs quadratic
deduplication, sorts, or re-FNVs plans. `program_identity_test.cpp` covers
signature/region identity, bucket sensitivity, order independence, and counts.

## 4. Parity-drift surfaces (numeric-contract risk)

### 4.1 Keyed-RNG helpers transcribed four times — CONFIRMED

`driver/metal/src/kernels/ptir_m0.metal:18`,
`driver/metal/src/kernels/ptir_m1_runtime.metal:51`,
`driver/cuda/src/pipeline/tier0/tier0_kernels.cuh:46`,
`driver/metal/src/pipeline/interp.hpp:895`.

Four independent transcriptions of the splitmix64 / hash_uniform constants
and the `>>40` mantissa path (the two Metal copies differ only in name
prefix). The numeric contract requires byte-identity across backends; a
one-sided tweak silently breaks RNG parity. Share one MSL preamble string
and one canonical constant header in `driver/common`.

### 4.2 Structured-mask predicates hand-rolled three times — CONFIRMED

`interface/ptir/src/interp.rs:1444` (Rust reference), CUDA kernels, Metal
interp + generated MSL.

Causal/sliding/sink/ancestry boundary conditions (`key+window>pos`,
`key<sink`) are written independently in six sites across three backends.
Verified byte-consistent today, but a one-sided off-by-one fix breaks
attention-mask parity silently. Wanted: one spec table (or generated
predicate shared like `op_info`) plus a single golden vector set exercised
by all backends.

## 5. Pre-existing (outside this diff, worth a follow-up)

- `interface/ptir/src/container.rs:578-654`: container decode preallocates
  `Vec::with_capacity` from untrusted wire counts (`n_names`, `n_channels`,
  `n_ports`, `n_stages`, `n_ops`, `n_externs`). A crafted ~20-byte
  container with count `0xFFFFFFFF` forces a huge allocation and aborts the
  engine (reachable from `ProgramRegistry::register` on inferlet-supplied
  bytes). Predates this change; harden by clamping capacity to remaining
  input bytes.

## 6. Verified clean (checked at xhigh, no action needed)

- RNG constants and mantissa path bit-identical across generated CUDA,
  grouped Tier 0, Metal, and the reference (given 4.1's caveat).
- New op wire format (0x66-0x68 masks): encode/decode, shape/dtype inference,
  skip tables, and all consumer switches consistent across Rust/C++/CUDA/Metal.
  Nucleus sampling has no wire opcode: the SDK emits ordinary SSA and the Rust
  compiler alone recognizes its exact normalized dataflow as an internal
  library region.
- Canonical width-32 reduction adopted in all three backends; golden-file
  value changes are the intentional numeric-contract change, pinned by a
  dedicated test.
- Metal PTIR PSOs compile with fast-math disabled (MTLMathModeSafe).
- Ticket/readiness/commit semantics preserved through the grouped path; no
  double-advance of RNG or channel state on fallback; per-lane commit
  isolation holds.
- Stage signatures deterministic (BTreeMap/BTreeSet only; no HashMap
  iteration order in hashes); runtime values excluded from signatures.
- Refuted during verification: the suspected Rust-vs-C++ annotation
  template drift (the rev-4 refactor removed the annotation system; no
  second validator exists).

## Resolved during review (rev-4 refactor)

The mid-review refactor already addressed the review's altitude findings:
`beam_select` (an annotation carrying no semantic weight over `top_k`) was
removed from the SDK and the beam-search inferlet now calls `top_k` directly.
Structured masks became first-class ops; nucleus sampling remains composable
SSA and is cut into a library region only by the normalized Rust compiler, per
ptir_plan.md rev 4.
