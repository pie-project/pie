# Metal PTIR Execution Plan

## Status

**Proposed (2026-07-10).** Successor increment to the Metal Phase-3 work
recorded in [direct_ffi_new_plan.md](direct_ffi_new_plan.md) (channel-plane
PTIR on the host interpreter). This plan wires the Metal forward into the
PTIR launch path so that forward-dependent programs (logits intrinsics,
descriptor-port device geometry) execute on the Metal driver, reaching parity
with what the CUDA driver's PTIR runtime supports today.

Owner-facing: each phase is independently landable and gated. Sections 9
and 10 list the acceptance gates and the open decisions that need a call
before the affected phase starts.

**Amendment (2026-07-12).** Phase 4 (§8, a handwritten MSL port of the
tier-0 kernels) is superseded by the Metal track in
[ptir_plan.md](ptir_plan.md) (Decision 7): Metal goes directly to generated
execution, with singleton and fused partitions of the shared region plan,
and the CPU interpreter is deleted once that track's M1 gates pass. Phases
0-3 of this plan (shared headers, forward-wired epilogues, device geometry,
batching/async) remain live prerequisites for that track's M2/M3.

## 1. Scope: what "PTIR support" means here

PTIR programs fall into classes by what they need from the driver:

| Class | Needs | CUDA today | Metal today |
|---|---|---|---|
| C1. Channel-plane (counter, ping-pong, extern SPSC) | channels only | ✅ GPU tier-0 | ✅ CPU host interp |
| C2. Epilogue over `Intrinsic(Logits)` / `Intrinsic(MtpLogits)` (greedy, temperature, top-k/p, mask-apply, beam epilogue, MTP verify) | forward logits | ✅ GPU tier-0 | ❌ rejected `UNSUPPORTED` |
| C3. Device-geometry (descriptor ports bound to channels, e.g. run-ahead beam) | pre-forward port reads → batch geometry | ✅ `resolve_descriptors` | ❌ rejected |
| C4. Submit-bound `HostInput` values | host tensor per fire | ⚠️ plumbing exists (`FireInputs.host_inputs`), nothing populates it | ❌ rejected |
| C5. Per-layer taps (`OnAttnProj`/`OnAttn` stages, `query`/`layer` intrinsics) | hooks inside the forward | ❌ not implemented (tier-0 runs stages once per pass, no per-layer iteration; see `driver/cuda/src/ptir/tier0_runner.hpp:356`) | ❌ rejected at classification |
| C6. `kernel_call` (second-party kernels) | named kernel registry | ❌ rejected at decode (`driver/cuda/src/ptir/bound.hpp:179`) | ❌ rejected (same shared decode) |
| C7. `sink_call` effects | forward config | ⚠️ decoded, effect ignored at tier-0 (`bound.hpp:177`) | same (shared decode) |

**The parity target of this plan is C2 + C3, plus the KV/state ABI
(`copy_kv`, `copy_state`, `resize_pool`) that forward-dependent inferlets
rely on.** C1 is done. C4–C7 are not supported by the CUDA driver either;
they stay rejected on Metal with the existing `reject_reason` strings and are
out of scope (§11). Anyone extending C4–C7 later does it for both backends
against the `interface/ptir` contract, not as Metal work.

Normative references:

- Semantics oracle: `interface/ptir/src/interp.rs` (the tier-0 golden model).
  Every observable behavior (argmax tie-break, sort order, splitmix64 RNG,
  readiness, pass-atomic commit) is pinned there and in `PTIR-CONTAINER.md`.
- ABI: `interface/driver/include/pie_driver_abi.h` (v2). The engine-side backend
  adapter for Metal is complete (`runtime/engine/src/driver/backend/metal.rs`
  marshals all eleven entry points), so **no runtime/engine changes are
  required by this plan** except test enablement.

## 2. Current state (anchors)

### 2.1 Metal driver

- `driver/metal/src/entry.cpp` (852 lines) exports the full `pie_metal_*`
  vtable. Registration, channel wire rings (§4.3 writer pull, §4.4
  publication, poison settlement, batch-aggregated availability) are done and
  tested (`tests/direct_stub_test.cpp`, ctest `metal_direct_stub_test`).
- `driver/metal/src/ptir/host_interp.hpp` (1,175 lines): CPU interpreter for
  the full op set, pinned to `interp.rs`. Programs are classified at
  registration by `build_exec_plan` (`host_interp.hpp:238`); the launch path
  rejects non-executable plans at `entry.cpp:439` with
  `PIE_STATUS_UNSUPPORTED`. The three reject reasons are exactly the C2/C4/C5
  classes (`host_interp.hpp:276-291`).
- `copy_kv` / `copy_state` / `resize_pool` return `PIE_STATUS_UNSUPPORTED`
  unconditionally (`entry.cpp:524-535`).
- Decode layer is shared with CUDA: `host_interp.hpp` includes
  `driver/cuda/src/ptir/{container,bound,trace,op_table}.hpp` (all pure
  host, CUDA-free) via the include path added in
  `driver/metal/CMakeLists.txt:126`.
- Forward compute exists in two ABI-disconnected worlds:
  - `src/raw_metal/`: MLX-free Metal-4 pipeline. `RawMetalDecoder`
    (`raw_metal/decoder.hpp`) wraps the full lifecycle (safetensors mmap →
    heap/DAG → PSOs → resident) and a per-token `step(token_id, position)`
    returning bf16 logits (`logits_bf16()`, `copy_logits_f32()`, `argmax()`).
    Shipped geometry: Qwen3.5-0.8B/qwen3.6 (3.755 ms/step path); a Gemma4
    path exists. Paged-KV pieces are in flight: `kv_append_paged.metal`,
    `sdpa_paged.metal`, `batch_schedule.hpp`, `decode_dispatch_mb.hpp`,
    `DecodeGeometry.{kv_page_size,total_pages}` (`decode_abi.hpp:97-103`),
    per-slot state reset (`decoder.hpp` `reset_state(slot)`).
  - `src/model/` + `src/ops/`: MLX lazy-graph builders for ~10 arch families,
    gated behind `PIE_METAL_WITH_MLX` (default OFF). `src/executor/` is an
    empty `.gitkeep`; nothing drives this path.

### 2.2 CUDA driver (the shape to mirror)

- `entry.cpp` delegates PTIR calls to `ptir::PtirDispatch`
  (`driver/cuda/src/ptir/ptir_dispatch.hpp`). The executor runs the model
  forward, gathers the sampling rows, converts bf16→f32, and calls
  `ptir_dispatch->run(view, logits_f32, vocab, stream, &runtime, completion)`
  (`driver/cuda/src/executor/executor.cpp:1960-1999`).
- Pre-forward, device-geometry programs resolve their descriptor-port
  channels into standard batch geometry via
  `PtirDispatch::resolve_descriptors` → `FireGeometry`
  (`descriptor_resolve.hpp`, `fire_geometry.hpp`; the latter is already
  CUDA-free). The resolver is program-agnostic: a 1:1 port→field copier with
  the CSR-prefix and KvLen→last_page_len contracts, kept in correspondence
  with the host `map_geometry` (`runtime/engine/src/ptir/ptir_geometry.rs`).
- Intrinsic resolution surface is exactly `FireInputs`
  (`tier0_runner.hpp:40-54`): `logits` base + `vocab`, optional
  `mtp_draft_row` (MtpLogits rows inside the same buffer), row seeds, and the
  (unused) `host_inputs` map. Nothing else crosses from the forward into PTIR.

Implication worth stating plainly: **the integration contract between the
forward and PTIR is one f32 logits matrix plus row indexing.** That is the
entire seam Phase 1 has to build on Metal.

## 3. Design principles

1. **CPU interpreter stays the Metal execution engine for now.** Apple
   Silicon unified memory makes logits readback a cache-coherent access, not
   a bus copy. At Metal's realistic batch sizes the epilogue math is trivial
   next to the forward. GPU tier-0 (MSL port) is a profiling-justified later
   phase (§8), not a prerequisite. This also honors the dumb-driver
   principle: the driver stays a general executor, no program-specific logic.
2. **Classification, not capability flags.** Keep the existing pattern:
   registration is lenient, `build_exec_plan` classifies what a program
   needs, launch rejects only what the driver genuinely cannot do, with a
   precise `reject_reason`.
3. **Every phase lands with its gates green** (§9) and never regresses the
   stub-test contract (put→launch→take, availability rejection, poison
   settlement, seed credit).
4. **Behavioral parity is defined against `interp.rs`, not against CUDA.**
   Cross-backend token equality is only expected where the contract
   guarantees it (identical PTIR inputs → identical PTIR outputs). Forward
   numerics (bf16 logits) legitimately differ across backends.

## 4. Phase 0: extract the shared pure-host PTIR headers

Small, mechanical, unblocks clean ownership. Do first; everything later
touches these files.

- Move `container.hpp`, `bound.hpp`, `trace.hpp`, `op_table.hpp`,
  `ptir_abi.h` from `driver/cuda/src/ptir/` to
  `driver/common/include/pie_native/ptir/` (the move the `host_interp.hpp`
  header note already declares as planned). `fire_geometry.hpp` moves too
  (it is explicitly CUDA-free and Phase 2 needs it).
- Namespace: keep `pie_cuda_driver::ptir` as an alias initially or rename to
  `pie_native::ptir` in one sweep; either is fine, but do it in this phase,
  not later. Update both drivers' includes and the Metal CMake include-path
  hack (`driver/metal/CMakeLists.txt:126` stops reaching into
  `../cuda/src`).
- No behavior change. Gate: both drivers' full test suites pass unmodified
  (CUDA ctest 30/30 on a CUDA box, `metal_direct_stub_test` on a Mac);
  golden container bytes and C3 hashes untouched.

## 5. Phase 1: forward-wired epilogue PTIR (C2), batch=1

Goal: a chat/completion inferlet using `inferlet::ptir` (greedy, temperature,
top-k/p, packed-mask constrained decode) runs end-to-end on a Mac against the
qwen3.6 checkpoint. This is the "Metal serves tokens through PTIR" milestone.

### 5.1 Forward executor seam

New: `driver/metal/src/executor/executor.{hpp,cpp}` (the empty directory
finally earns its keep). A narrow class, deliberately smaller than CUDA's:

```
class MetalExecutor {
    bool setup(const Config&, const ModelFacts&, std::string* err);
    // One member's forward: tokens/positions in, f32 logits rows out
    // (rows selected by sampling_indices/indptr from the launch view).
    bool forward(const MemberForwardDesc&, LogitsOut&, std::string* err);
    // KV/state plumbing (§5.4)
    ...
};
```

Backing: `RawMetalDecoder` (decision D1, §10). `entry.cpp` keeps zero direct
Metal dependencies; it owns a `std::unique_ptr<MetalExecutor>` created
lazily on first forward-needing launch (mirrors CUDA's lazy
`ptir_dispatch`). Prefill for M=1 iterates `decoder.step()` over the prompt
tokens; only the readout rows' logits are materialized. Slow but correct;
M>1 prefill encode is Phase 3.

### 5.2 Launch-path changes (`entry.cpp`)

Today `MetalDriver::launch` ignores the launch view's forward fields
entirely. Phase 1:

- Parse the `LaunchView` per member (tokens, positions, qo_indptr, KV CSR
  geometry, sampling_indices/indptr) using the shared
  `pie_native::LaunchView` reader, same as CUDA's entry.
- Members whose plan `needs_forward == false` (C1) run exactly as today.
- Members with `needs_logits`: run §5.1 forward, then interp `step` with
  pass inputs (§5.3). Failure of the forward poisons the member (existing
  `poison_instance` path), not the batch, matching D4 failure domains.
- Ordering guarantees stay: channel words while settling, then terminals,
  then per-channel notifies, then the batch notify once.
- Synchronous execution on the caller thread is acceptable for this phase
  (the dummy driver already settles synchronously; the notify contract does
  not require async). Moving to a command-queue completion-handler model is
  Phase 3 work, noted in §12 risks.

### 5.3 Interpreter changes (`host_interp.hpp`)

- `ExecPlan` classification splits "rejected" from "needs inputs":
  `bool needs_logits`, `bool needs_mtp_logits` (replacing the blanket
  Intrinsic reject). `HostInput`, per-layer stages, and unknown intrinsics
  (`hidden`, `query`, `value_head`, `layer`) keep `executable = false`; the
  model-gated MTP intrinsics are additionally fenced by the runtime bind
  profile, so qwen3.6 programs using them never reach the driver.
- `step(inst, plan)` gains a `PassInputs` argument mirroring `interp.rs`
  `PassInputs` / CUDA `FireInputs`: `{ const float* logits; uint32_t rows;
  uint32_t vocab; int mtp_draft_row; }`.
- `exec_stage` root resolution gets a `ValueSource::Intrinsic` case:
  materialize a `Value::f32` of the declared trace shape from the logits
  rows (`Logits` from row 0, `MtpLogits` from `mtp_draft_row`, falling back
  to row 0 when unset, exactly the CUDA fallback in
  `tier0_runner.hpp:420-427`). Shape mismatch (program vocab vs model vocab)
  is a fault → poison, same as any op fault.
- Descriptor ports: the existing consume-only loop (`host_interp.hpp:1140`)
  stays for Phase 1; port values start feeding the forward in Phase 2.

### 5.4 KV and recurrent state, batch=1

The runtime leases pages and ships CSR geometry; the driver must be honest
about what it honors:

- Capabilities: report the real `total_pages` / `kv_page_size` from config
  wired through to `DecodeGeometry`; keep `rs_cache_required = true` for
  GDN models with real `rs_cache_slots` (already partially in
  `build_caps_json`, `entry.cpp:167`).
- Phase 1a (linear sequences only): accept geometry that is a single
  contiguous run per member (validate: one KV segment, monotone pages,
  in-order positions). The decoder's resident KV ring plus per-slot GDN
  state implements it. Reject anything else (forks, shared prefixes,
  scattered pages) with `UNSUPPORTED` and a reason. This is enough for
  chat/completion inferlets.
- Phase 1b: adopt the paged kernels (`kv_append_paged`, `sdpa_paged`) so
  arbitrary page CSR geometry is honored, and implement:
  - `copy_kv` (Design-B compaction and forks copy page ranges inside the
    pool; on unified memory a blit encoder or memcpy over the heap region),
  - `copy_state` (GDN conv+recurrent slab copy between slots),
  - `resize_pool` (grow/shrink the KV region; acceptable to land as
    grow-only first, matching what the scheduler actually issues).
  Phase 1b must coordinate with [kv_refact.md](kv_refact.md) if that lands
  first (WorkingSet-relative indexes change what arrives in the launch
  descriptor). If kv_refact is still pending, build against the current ABI
  and keep the mapping localized in `MetalExecutor`.

### 5.5 Model coverage

qwen3.6 (shipped geometry) is the Phase 1 target; Gemma4 follows once green
(its raw_metal path exists but is younger). Every other arch waits for the
MLX-path decision (D1). `read_model_facts` already extracts arch/vocab from
the HF config; the executor refuses unknown arch at `setup` with a clear
error instead of caps lying about readiness.

## 6. Phase 2: device-geometry programs (C3)

Port the CUDA resolver semantics; on Metal this is simpler because channel
state already lives host-side.

- New `driver/metal/src/ptir/descriptor_resolve.hpp`: read the program's
  bound descriptor-port channels' current cells (host `ChannelState`, no
  device reads needed) and fill the shared `FireGeometry` struct (moved to
  common in Phase 0). Apply the two fixed contracts (CSR-prefix trim,
  `KvLen → ((len-1) % page) + 1`) in explicit correspondence with
  `descriptor_resolve.hpp` (CUDA) and `ptir_geometry.rs` (host). Keep it a
  1:1 port→field copier; no program-specific logic.
- NOT-READY IS AN ERROR (W1.6): a device-geometry fire whose descriptor
  channel is not full fails the fire; no dummy-run.
- `launch` resolves before the forward for members carrying a
  device-geometry program, feeds the geometry into `MetalExecutor::forward`
  instead of the wire fields, and consumes the token-family ports
  (the existing consume loop moves into the resolver, keeping take-once
  semantics per fire).
- Run-ahead ordering: CUDA leans on same-stream ordering; the Metal driver
  is synchronous per launch in this phase, which gives the same
  happens-before (fire t's epilogue puts commit before fire t+1's launch is
  processed). If/when launches go async (Phase 3), the resolver must read
  channel state on the execution thread in submission order; note it in the
  executor's threading comment now.
- The runtime-side constraint of one device-geometry program per batch
  already exists; nothing to add on Metal beyond the same validation CUDA's
  `validate_launch` does (share it via `driver/common` if not already).

Unlocks: run-ahead beam search (Design-B), speculative verify flows, and the
`cuda_beam_designb_*` / `cuda_drafts_retain` test shapes on Metal.

## 7. Phase 3: batching and async completion

- M>1 forward: build on `batch_schedule.hpp` / `decode_dispatch_mb.hpp`
  (the mac-paged-kv-bridge work). Batch members map to slots; GDN slot
  reset per NEW request is already exposed (`reset_state(slot)`).
- Prefill encode for M>1 (single command buffer over the prompt) replaces
  the per-token `step()` loop from Phase 1.
- Async launch: move member execution onto the executor's queue; publish
  completions from the command buffer completed-handler through the same
  notification lists (the callback contract in `PieRuntimeCallbacks` is
  thread-agnostic; the CUDA driver already publishes from a host-func
  callback). Registry/instance state mutation must stay on one thread or be
  locked; follow the settled decisions from direct_ffi_new_plan §Status
  (the completion-vs-scheduler race class that plan closed on CUDA).
- Respect `max_forward_tokens` / `max_forward_requests` caps truthfully.

## 8. Phase 4 (optional, profiling-gated): GPU tier-0 in MSL

Only if the CPU epilogue shows up in end-to-end profiles (most likely at
M>1 with large-vocab models or matmul/sort-heavy inferlets).

- Port the ~40 row-parallel kernels in `tier0_kernels.cuh` (~630 lines) to
  one `.metal` library plus a `launch_op` switch over
  `MTLComputeCommandEncoder`. The CUDA header explicitly frames this file
  as the entire porting surface. matmul/top_k stay naive (they are naive on
  CUDA too).
- Numerics: compile the PTIR library with fast-math OFF (Metal default is
  ON; this silently breaks NaN/inf semantics that `mask_apply`,
  `pivot_threshold`, and argmax depend on). Transcribe
  `t0_splitmix64`/`t0_hash_uniform`/`t0_gumbel` constants verbatim.
- Channel rings on device follow the CUDA layout (`channel_registry.hpp`)
  with Metal atomics; unified memory removes the pinned-mirror copy
  machinery (`sampling_ir/`'s reason to exist on Metal is much smaller).
- Acceptance: byte/ULP parity against `interp.rs` on the shared golden
  vectors (`interface/ptir/tests/golden-ptir/*.txt`), then A/B token
  equality vs the host interpreter across the Phase 1/2 e2e suite.

Do not start Phase 4 before Phases 1–3 are green; a GPU epilogue with no
batched forward optimizes the wrong term.

## 9. Testing and acceptance gates

Per phase, in addition to "previous gates stay green":

**Phase 0**
- G0.1 CUDA ctest suite unchanged (30/30) on a CUDA machine; goldens byte-identical.
- G0.2 `metal_direct_stub_test` green; grep gate: no `#include` from
  `driver/metal` into `driver/cuda` paths.

**Phase 1**
- G1.1 Unit: `host_interp` with injected logits reproduces the epilogue
  golden vectors (`greedy_argmax`, temperature/gumbel, `mtp_verify_tail`,
  `matrix_mask_apply_packed`) bit-for-bit vs `interp.rs`.
- G1.2 Driver-level ctest (checkpoint-gated, env `PIE_METAL_CKPT`): register
  a greedy epilogue program, launch with real tokens, take the sampled token
  from the reader channel; cross-check argmax against
  `RawMetalDecoder::argmax()`.
- G1.3 E2E on a Mac: `bin/pie` boot with `driver-metal`, run the migrated
  `generate` / `lowlevel-chat` inferlets (the `inferlet::ptir` guests from
  the SDK migration) to completion. Mirror of `cuda_chat_completion_e2e.rs`.
- G1.4 Dummy-vs-Metal determinism: identical program + seeds + injected
  logits → identical channel outputs (runs in CI without a GPU by faking
  the executor behind a test seam).
- G1.5 Rejection honesty: C4/C5/C6-class programs still reject with the
  right reasons; `copy_kv` on a non-1b build still reports `UNSUPPORTED`
  (no silent success).

**Phase 2**
- G2.1 Port-contract unit tests: CSR-prefix trim and last-page-len math
  identical to CUDA's resolver on shared fixture geometry.
- G2.2 E2E beam (Design-B) on Metal: mirror `cuda_beam_designb_e2e`;
  run-ahead fire pairs preserve put→resolve ordering.
- G2.3 Not-ready descriptor channel fails the fire and poisons per D4.

**Phase 3**
- G3.1 Multi-member launch with mixed C1/C2 members settles all terminals
  and notifies the batch slot exactly once (extend the stub test's
  aggregate-availability case to forward members).
- G3.2 Thread-safety: no registry mutation off the execution thread
  (TSAN run of the driver tests, mirroring the CUDA dispatch-race gate).

**Phase 4**
- G4.1 Golden parity GPU-vs-`interp.rs`; G4.2 e2e A/B GPU-vs-host-interp
  token equality; G4.3 fast-math grep/build gate on the PTIR `.metal`
  target.

## 10. Open decisions (make before the affected phase)

- **D1 (Phase 1): raw_metal vs MLX as the serving forward.**
  Recommendation: raw_metal. It runs today, is MLX-free (default build),
  loads HF safetensors, and has the paged/M>1 bridge in flight. Cost: model
  coverage is qwen3.6 (+Gemma4) until more geometries are ported. The MLX
  path buys arch breadth but has no executor and drags the MLX dependency
  into the worker link. Revisit after Phase 3 if coverage matters more than
  dependency weight.
- **D2 (Phase 1a→1b): how long linear-only KV is acceptable.** If beam
  inferlets are a near-term demo target, pull Phase 1b forward or start it
  in parallel; `copy_kv` is the piece beams cannot fake.
- **D3 (Phase 1): where bf16→f32 happens.** Executor converts (matches
  CUDA's cast-before-dispatch, keeps the interpreter f32-only). Alternative
  (interpreter reads bf16 lanes) saves a copy but forks the value model;
  not recommended.
- **D4 (Phase 3): async execution model.** One executor thread + command
  buffer completion handlers, vs fully synchronous with the engine's
  batching scheduler absorbing latency. Decide with profiles from Phase 1.
- **D5 (Phase 0): final namespace for the shared headers**
  (`pie_native::ptir` rename vs alias retention).

## 11. Non-goals

- Per-layer taps (C5), `kernel_call` (C6), `sink_call` effects (C7),
  `HostInput` (C4): not implemented on any backend; adding them is
  cross-backend contract work driven from `interface/ptir`, not Metal
  bring-up. The classification/reject strings must stay accurate so guests
  get honest errors.
- Tensor parallelism, multi-GPU, NCCL-equivalents: CUDA-only concerns.
- Tier-1 JIT fusion on Metal: revisit only after Phase 4 exists and shows
  dispatch overhead worth fusing.
- MTP intrinsics on models without MTP heads: the bind profile gates them;
  no driver work.

## 12. Risks

- **Prefill latency in Phase 1** (per-token `step()` loop): correct but
  slow for long prompts; acceptable for bring-up, fixed in Phase 3. Do not
  let it become the reason to skip G1.3.
- **Blocking the scheduler thread**: synchronous forwards inside `launch`
  hold the driver call for milliseconds. The engine's per-driver batching
  scheduler tolerates it, but watch for waker starvation under concurrent
  inferlets; this is the D4 trigger.
- **Caps honesty**: the current caps JSON reports config values for a
  forward that does not exist. Phase 1 must make caps reflect the executor's
  real limits (arch, max tokens/requests, pages), or the scheduler will
  build launches the driver rejects.
- **kv_refact churn**: Phase 1b/2 touch exactly the surfaces kv_refact
  redesigns (page vectors, WorkingSet indexes). Sequence the two plans
  explicitly when kv_refact firms up.
- **Numeric drift in a future MSL port**: fast-math and precise-function
  variants are per-translation-unit decisions on Metal; the G4.3 build gate
  exists so this cannot regress silently.
- **Guest SDK migration tail**: e2e gates depend on the `inferlet::ptir`
  guest migration (generate, lowlevel-chat, specverify, mtpverify are done
  per direct_ffi_new_plan; anything else a gate needs must be migrated
  first, not worked around).

## 13. File map (where new code goes)

```
driver/common/include/pie_native/ptir/   ← Phase 0: container/bound/trace/
                                            op_table/ptir_abi/fire_geometry
driver/metal/src/executor/executor.hpp   ← Phase 1: MetalExecutor seam
driver/metal/src/executor/executor.cpp
driver/metal/src/ptir/host_interp.hpp    ← Phase 1: PassInputs + intrinsic
                                            roots + classification split
driver/metal/src/ptir/descriptor_resolve.hpp ← Phase 2
driver/metal/src/entry.cpp               ← Phases 1-3: launch wiring,
                                            copy_kv/copy_state/resize_pool
driver/metal/src/kernels/ptir/*.metal    ← Phase 4 only
driver/metal/tests/                      ← per-phase ctest targets
bin/pie/tests/metal_*_e2e.rs             ← G1.3 / G2.2 mirrors
```
