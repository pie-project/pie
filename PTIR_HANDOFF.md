# PTIR Optimizing Compiler Handoff

Date: 2026-07-12

Repository: `pie-project/pie`

Branch / baseline: `dev` at `e0e3e7c2`

No commit has been created. The worktree is intentionally large and dirty:
approximately 251 changed/untracked files and 16.5k inserted lines. Preserve
unrelated user changes.

## Read first

1. `ptir_plan.md` is the binding architecture. It is now Revision 6.
2. `ptir_problems.md` is the adversarial review that drove the latest fixes.
3. This document describes the implementation state; do not infer completion
   from old SQL task names or earlier "done" reports.

## Binding decisions

- Rust owns normalization, symbolic types, signatures, semantic recognition,
  region partitions, cache identities, and lane ABI.
- Drivers execute serialized Rust plans. They do not recover SDK intent.
- Core PTIR opcodes stay general and composable.
- `CompositeKind` / `CompositeAnnotation` are deleted.
- `BeamSelect` and `BeamAncestryMask` are deleted.
- Beam selection is generic `TopK + DIV/REM + Gather` SSA.
- Beam ancestry/membership is generic composable SSA.
- There is no public `NucleusSample` opcode. The SDK emits a generic 13-node
  SSA graph. Rust recognizes the exact normalized graph and emits an internal
  `LibraryOp::NucleusSample` region with ABI:

  ```text
  inputs  = [logits, top_p, rng_state]
  outputs = [token]
  ```

- Recognition affects performance only. A recognition miss executes generic
  SSA with identical semantics.
- Channel cell shapes are static. Logical runtime extents remain separate and
  symbolic.
- Numeric reductions use the canonical width-32 schedule. NaN, tie, signed
  zero, integer, and keyed-RNG rules are contract-defined.
- One true folded `RsWorkingSet` is bound per resolved request. Buffered RS
  slots are never used as folded-state slots.
- Production must ultimately execute only grouped fused generated regions and
  stock libraries.
- Generated singleton partitions are test-only differential oracles.
- Handwritten CUDA Tier 0 must be deleted after generated coverage/cutover
  gates pass.
- Production registration compiles every fused/library executable before
  publishing a program ID. There is no fire-time compile or semantic fallback.

## Current implementation state

### Shared compiler / SDK / runtime

Implemented and repeatedly reviewed:

- Annotationless PTIR container v1/v2.
- PTIB v2 with PTRP v4 / compiler v3 plans.
- Typed normalized SSA, symbolic extents, canonical stage signatures.
- Fused and singleton partitions.
- Rust-only exact nucleus SSA recognition.
- TopK/Sort/Scan/MatMul/second-party library cuts.
- Lane ABI v2 and cache identity.
- Canonical numeric reductions and exact selection/RNG semantics.
- General `row_membership` SSA helper.
- Structured causal/sliding/sink mask operations.
- Strict name ordering, nonzero shapes, canonical decoding.
- Bounded Rust and C++ PTIB/PTRP/container decoders.
- Shared malformed-wire corpus and C++ decoder-limit test registered in both
  backend CMake suites.
- Multi-request folded RS preparation, atomic transaction commit/rollback,
  in-place ForwardPass RS rebinding, parent-state CoW fork, FIFO ownership,
  preemption-safe predecessor drain, and reverse-order slot retirement.
- Driver capability propagation for MTP-gated PTIR intrinsics.
- Rust-owned RNG contract with generated CUDA/C++ and MSL artifacts.

The RNG consolidation implementation passed its own Rust, Metal, and CUDA-host
tests but has not yet received a separate final code-review pass.

### Metal

Feature-complete relative to Revision 6 except that production/singleton
selection must be checked during the final generated-only cutover:

- Shared-storage channel authority.
- Strict generated singleton/fused/grouped MSL.
- Direct BF16 logits.
- Per-stage grouping, ragged lanes, readiness, effects, atomic commit.
- Exact parallel Nucleus and TopK libraries.
- Generic beam SSA and row-membership fallback.
- Structured-mask dense fallback.
- Multi-request hybrid RS with real folded slots.
- MTP row offsets in singleton/fused/grouped paths.
- Transient-safe compilation/negative caching.
- Full-width persistent cache identity.
- Transactional PSO rollback.
- Bounded fence-safe resident scratch pool.
- GPU clearing of fixed channel-cell tails.
- Correct mask pitch after pool resize.

Latest local suite:

```text
20/20 CTest targets passed
m1_generated_test: 106/106
ptir_decoder_limits_test: passed
metal_direct_stub_test: passed
```

`PIE_METAL_CKPT` is unset, so the optional checkpoint-backed test skips
internally.

Metal has been through repeated read-only review; the latest review reported no
significant issues before the RNG artifact consolidation. Re-review the small
RNG-generation diff.

### CUDA

Implemented and locally code-reviewed:

- Grouped per-signature execution and lane attribution.
- Staged Prologue / OnAttnProj / OnAttn / Epilogue execution.
- Per-layer Query and Layer hooks across supported model families.
- Cross-stage pending values and atomic finish/abort.
- Direct BF16 logits, physical/logical row-stride handling.
- Ragged extents and fixed-cell padding.
- Exact generic Gather/GatherRow/Scatter contracts.
- Rust-planned internal Nucleus region consumption; no CUDA pattern matcher.
- TopK and semantic libraries.
- Structured mask direct/fallback paths with tail-alignment checks.
- Explicit WSlot/WOff through Qwen3.5 dense/MoE and TP.
- MTP preflight before state mutation, aggregate staging, counted TP gate.
- Fold lengths, buffered write/fold replay, and request-order RS metadata.
- Graph key fixes, capture RAII, reset safety, cache bounds, and lock ordering.
- Allocation-free precomputed program-set identities.
- Hardened launch cardinality and descriptor-resolved RS validation.

Latest CUDA review reported no significant issue after the final
`forward.cpp` initializer-lambda fix.

Local CUDA-host/static reports:

```text
12/12 host executables passed
27/27 container cases passed
forbidden legacy identifier searches passed
IDE diagnostics clean
```

Important: the current CUDA production path still contains handwritten/grouped
Tier 0 per-op execution. It is correctness-hardened but does not satisfy the
Revision 6 north star.

## The primary remaining architecture gap

CUDA does not currently have a full production generic fused-region compiler.
Generic stages still execute approximately:

```text
unique ready signatures x PTIR ops
```

through grouped Tier 0 (possibly captured as a graph). The north star requires:

```text
unique ready signatures x fused/library regions
```

with no production fallback.

The old production `generated_epilogue` matcher was removed because it
recognized private sampler patterns rather than consuming arbitrary Rust
regions. The remaining `test_ptir_tier1` NVRTC code is a narrow test prototype,
not the required production compiler.

## Revision 6 execution plan

### 1. Complete CUDA generated singleton oracle (test-only)

- Generate every first-party op, intrinsic, symbolic extent, dtype, effect, and
  direct sink from the singleton plan.
- Cover staged phases and grouped lanes.
- Match Rust reference bitwise/tolerantly under the numeric contract.
- Ensure no production dispatch route can select singleton.

Exit gate: every authoritative golden and migrated inferlet compiles and passes
through generated singleton tests.

### 2. Implement generic CUDA fused-region codegen

- Consume `RegionKind::Generated` directly from PTRP.
- Support arbitrary elementwise DAGs.
- Support multiple canonical reductions and reduction consumers.
- Support multiple/mixed-domain outputs.
- Write terminal values directly to pending channel cells.
- Preserve staged phase placement and cross-region pending state.
- Respect library islands and role-ordered semantic ABIs.
- Group lanes by stage signature and schedule bucket.
- Compile at registration, not fire time.

Exit gate: target epilogues use one fused body per signature group plus required
library/commit regions; output matches singleton.

### 3. Enforce registration-time availability

- Compile all fused/library regions before returning a program ID.
- Persist bounded executables by canonical cache identity.
- Negative-cache only deterministic failures.
- Surface compiler/disk/OOM/cache-capacity failures as retryable registration
  errors.
- Reject incomplete coverage before native instance allocation.

Exit gate: first fire performs no compilation and has no alternate execution
path.

### 4. Delete CUDA Tier 0

Remove:

- `pipeline/tier0` runners, launchers, and handwritten op kernels.
- Grouped per-op production execution.
- Fire-time compile/fallback state.
- Fallback metrics and environment switches.
- Production singleton selection.
- Tests that assert fallback behavior.

Replace with Rust-reference versus generated-singleton versus fused
differentials.

Exit gate: production-source searches find no Tier 0 dispatcher or fallback
branch.

### 5. Generated-only validation

- Rust reference vs singleton vs fused parity.
- All shared goldens and migrated inferlets.
- Staged prologue/attention/epilogue programs.
- MTP, RS, structured masks, WSlot/WOff, ragged and multi-row paths.
- B=1/2/4/8 launch topology and critical-path timing.
- CUDA ASan.
- Compute Sanitizer: memcheck, racecheck, initcheck, synccheck.
- Full Metal 20-target suite.
- Final independent shared/CUDA/Metal review.

## Validation commands

Shared Rust:

```bash
cargo test -p pie-ptir --features eval
cargo test --manifest-path sdk/rust/ptir-dsl/Cargo.toml
cargo test -p pie-driver-abi
cargo test -p pie-driver-dummy
cargo test -p pie-engine --lib
cargo test -p pie-worker --lib
```

Migrated inferlets:

```bash
cargo check --manifest-path inferlets/Cargo.toml --target wasm32-wasip2 \
  -p attention-sink -p beam-search -p chat-completion \
  -p greenlist-watermarking -p json-schema-constrained-decoding \
  -p mirostat-v2-sampling -p sampling-primitives \
  -p sliding-window-attention
```

Metal:

```bash
cmake -S driver/metal -B driver/metal/build \
  -DPIE_METAL_BUILD_STUB_TESTS=ON
cmake --build driver/metal/build -j4
ctest --test-dir driver/metal/build --output-on-failure
```

CUDA worker build on the RTX machine:

```bash
PIE_COMPILER_LAUNCHER=env cargo build \
  -p pie-worker --features driver-cuda -j6
```

Do not use high build parallelism; CUDA compilation has previously OOM-killed
the machine. Avoid full-workspace/CUDA builds for Rust-only changes.

The CUDA CMake build used by prior agents was reported under `target/cuda`.
Confirm the remote worktree/build directory before running:

```bash
cmake --build target/cuda --target pie_driver_cuda
ctest --test-dir target/cuda --output-on-failure
```

Then run the repository's existing ASan and four Compute Sanitizer PTIR
targets. Do not substitute a narrow proxy for production-vocabulary or staged
execution tests.

## RTX access

`ssh workstation` is configured for user `ingim`.

Observed during this session:

- One successful connection returned `Workstation` and `NVIDIA GeForce RTX
  4090`.
- Other attempts timed out or temporarily failed DNS.

Treat connectivity as intermittent. Verify access before syncing/building.
Do not assume the remote tree contains the latest 251-file local diff.

## Worktree safety

- Do not reset, checkout, clean, or revert the worktree.
- Preserve unrelated contention, scheduler, reclaim, and preemption changes.
- `runtime/engine/src/inferlet/process/preemption.rs`,
  `runtime/engine/src/scheduler.rs`, and `runtime/engine/src/store/reclaim.rs`
  contain pre-existing/user-adjacent work.
- The worktree has no commit for this PTIR effort.
- `sdk/rust/inferlet/src/audio_frontend.md` appears deleted in status; do not
  restore or remove it without determining ownership.
- Generated CUDA/Metal golden copies must remain synchronized with
  `interface/ptir/tests/golden-ptir`.

## Review findings disposition

`ptir_problems.md` was re-anchored and addressed as follows:

- Metal grouped Nucleus race: fixed with ping-pong reduction.
- Metal transient negative-cache poisoning: fixed.
- Metal truncated cache versions: fixed.
- Metal degraded grouped PSOs: fixed.
- Metal forward cursor reset: fixed.
- Metal per-fire allocation churn: replaced with bounded resident pooling.
- CUDA removed generated-backend negative cache: obsolete.
- CUDA FP32 physical/logical stride: fixed.
- CUDA per-fire plan/graph identity rebuild: precomputed.
- C++ intrinsic enum/wire drift: aligned with static assertions.
- Symbolic structured-mask propagation: fixed, including DCE renumbering.
- Rust container count-bomb allocation: fixed.
- PTIB/PTRP Rust/C++ count bombs: fixed and tested in both CMake suites.
- RNG transcription drift: consolidated into Rust-owned generated artifacts;
  final independent review still recommended.
- Structured-mask predicate drift: authoritative shared golden is executed by
  Rust, CUDA, and Metal; no workload-specific ancestry opcode remains.

## Useful specialist histories

The prior session used GPT-5.6 Sol agents. Their results are reflected in the
tree and this document; do not rely on being able to read their conversations
from a new session.

- `nucleus-ssa-shared` / `nucleus-ssa-review`
- `annotationless-shared` / `annotationless-shared-review`
- `multi-rs-beam` / `multi-rs-review`
- `metal-problems-fix` / `annotationless-metal-review`
- `cuda-final-five` / `annotationless-cuda-review`
- `ptib-ptrp-hardening` / `ptib-ptrp-review`
- `rng-contract-dedup`

Reuse capable GPT-5.6 Sol or Opus/max agents only for genuinely independent
backend implementation/review scopes.

## Immediate next action

Do not continue polishing grouped Tier 0. Start Revision 6 at
`cuda-generated-singleton`, then `cuda-generic-fusion`.

Before deleting anything:

1. Run the current CUDA baseline on `workstation`.
2. Preserve the result as the pre-cutover checkpoint.
3. Implement full generated singleton coverage.
4. Implement fused production codegen.
5. Switch registration to fused-only.
6. Delete Tier 0.
7. Run the complete generated-only validation matrix.

