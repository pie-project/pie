# Venus build log

Working notes for the north-star build (design: `venus-project.md`).
State: decisions closed, RETRY audit closed, ABI v14 blueprint finalized
(field-level, verified against the v13 validator). Next action: execute the
`interface/driver/src/local.rs` surgery below, then compiler-driven cascade.

## ABI v14 blueprint (verified field classification)

Verified against `validate_launch_desc`: `kv_translation`/`kv_translation_indptr`
are CSR **per instance** → hoist to frame. `context_ids` is **per resolved
row** (not per instance) → stays in the step. `kv_len_device` is 0/1 pointers.

```rust
/// One sealed frame: the v14 launch unit. Executed as a closed system with
/// a single completion (venus-project.md P1/P4).
#[repr(C)]
pub struct PieFrameDesc {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    /// Lane roster: every instance participating in any step. No duplicates.
    pub instance_ids: PieU64Slice,
    /// Frame-union WorkingSet page translation (committed mapping overlaid
    /// with ALL steps' prepared write targets), CSR per roster entry.
    pub kv_translation: PieU32Slice,
    pub kv_translation_indptr: PieU32Slice,
    /// Exclusive physical KV page high-water after the LAST step — the
    /// frame-union admission demand (replaces the lease surface).
    pub required_kv_pages: u32,
    /// Reserved; must be zero.
    pub reserved1: u32,
    pub steps: PieStepDescSlice, // { ptr: *const PieStepDesc, len }
}

/// One forward step. Trimmed from v13 `PieLaunchDesc`:
///  - REMOVED: abi_version (frame carries it), settle_defer (deleted),
///    kv_translation{,_indptr} + required_kv_pages (hoisted to frame),
///    instance_ids (→ roster_rows).
///  - ADDED: roster_rows, sub_batch_indptr, sub_batch_class (decision #3:
///    ordered homogeneous sub-batches; order resolves producer→consumer).
///  - KEPT (verified scopes): terminal_cells + logical_fire_ids per
///    roster_row; context_ids/kv_len/kv_last_page_lens per wire row;
///    ptir_* bounds + channel tickets CSR per step-instance; multimodal
///    payloads ride the step that owns them (usually step 0).
#[repr(C)]
pub struct PieStepDesc {
    /// Indices into the frame roster, one per batch member, sub-batch order.
    pub roster_rows: PieU32Slice,
    /// CSR over roster_rows; sub-batch b = rows [indptr[b], indptr[b+1]).
    pub sub_batch_indptr: PieU32Slice,
    /// PIE_GEOMETRY_CLASS_* per sub-batch.
    pub sub_batch_class: PieU32Slice,
    pub terminal_cells: PieTerminalCellPtrSlice,
    pub logical_fire_ids: PieU64Slice,
    // …then every remaining v13 PieLaunchDesc field unchanged:
    // token_ids, position_ids, qo_indptr, kv_page_indices, kv_page_indptr,
    // kv_last_page_lens, kv_len, kv_len_device, rs_slot_ids, rs_slot_flags,
    // rs_fold_lens, rs_buffer_slot_ids, rs_buffer_slot_indptr, masks,
    // sampling_indices, sampling_indptr, context_ids, single_token_mode,
    // has_user_mask, reserved_flags, image_* (9), audio_* (4), embed_* (6),
    // ptir_program_row_indptr, ptir_kv_write_{lower,upper}_bounds,
    // channel_expected_{head,tail}, channel_ticket_indptr.
}
```

Note: decode-envelope steps need no new delta encoding — the device-resolved
class already elides wire geometry (empty qo/kv indptr), so decode steps are
naturally tiny: roster rows, fire ids, terminals, tickets, write bounds.

Status/entry surface:
- `PIE_DRIVER_ABI_VERSION = 14`.
- ADD `PIE_STATUS_EXHAUSTED: i32 = -6` (admission full — engine re-posts
  later) and `PIE_STATUS_IMPOSSIBLE: i32 = -7` (can never fit): admission is
  folded into the launch call (decision: prepare surface deleted).
- DELETE: `PIE_LAUNCH_PREPARE_*`, `PieLaunchPrepareResult`,
  `validate_launch_prepare_result`, `pie_{cuda,metal}_prepare_launch`,
  `pie_{cuda,metal}_launch_prepared`, `pie_{cuda,metal}_release_launch`.
- `pie_{cuda,metal}_launch(driver, *const PieFrameDesc, PieCompletion)`.
- Validator: `validate_frame_desc` (header, roster no-dupes, translation CSR
  vs roster, step loop) + `validate_step_desc` (v13 body minus hoisted
  fields, plus sub_batch CSR/class checks). One completion per frame;
  per-fire terminal cells stay (outcomes latch per fire at frame settle:
  SUCCESS / FAILED-on-kill; whole-frame RETRY is the only retry form).

## local.rs surgery order (next action)

1. Header doc: v14 paragraph (frame unit; lease + settle_defer deleted).
2. Version bump 13→14; status constants add/delete per above.
3. `PieLaunchDesc` → `PieStepDesc` (field surgery above) + new
   `PieStepDescSlice`, `PieFrameDesc`, defaults.
4. `validate_launch_desc` → `validate_step_desc` + new `validate_frame_desc`.
5. Extern blocks: cuda + metal launch signatures; delete lease fns.
6. Tests: mechanical rename + hoisted-field moves; add frame-level cases
   (dup roster, translation CSR vs roster, step class validity). ~40 sites.
7. `cargo check -p pie-driver-interface` green, then regenerate the C header
   (cbindgen) and let the compiler drive the cascade: engine `driver/abi.rs`,
   `backend.rs`, `submission.rs`, worker lane posts; driver `abi.cpp`,
   `context.cpp`, `entry_validation.hpp`.

## Cascade map (who breaks, in order)

Engine: `driver/abi.rs`, `driver/backend.rs`, `driver/submission.rs`,
`scheduler/batch.rs` (build one frame, not per-wave; kill the double build),
`scheduler/worker.rs` (LaneRequest::LaunchFrame only; delete Prepare/
FlushSettle arms, M2 group gate, lease plumbing), `scheduler/frame.rs`
(reduce to seal policy), `driver/completion.rs` (frame completion),
`interface/driver/src/remote.rs` (edge adapter: decompose frame → per-step
remote submissions server-side; wire gains the frame form).
Driver: `abi.cpp` (entry), `entry_validation.hpp`, `context.cpp` (frame
admission folded into launch; lease registry deleted), new frame pipeline
modules per venus-project.md build order, `pipeline/dispatch.cu` (snapshot
ring + kill bit; delete M2 accumulator; sever publications_done→begin),
`batch/compose.cpp` split (FramePrepare / StepEnqueue / FrameSettle).
Metal driver: step-loop internally (same C ABI).

## Progress (updated as the build proceeds)

- DONE `interface/driver` (pie-driver-abi): v14 landed — PieFrameDesc/
  PieStepDesc/PieStepDescSlice, validators (validate_frame_desc/
  validate_step_desc(roster_len)), lease surface + settle_defer deleted,
  PIE_STATUS_EXHAUSTED(-6)/IMPOSSIBLE(-7), extern pie_{cuda,metal}_launch
  (driver, *const PieFrameDesc, completion). C header regenerated;
  layout_contract.inc + c11/cpp20 sources updated. ALL TESTS GREEN (29+2).
- DONE engine driver layer: submission.rs (FrameSubmission/StepSubmission,
  settle_defer gone), abi.rs (FrameDescBorrow), backend.rs (launch →
  Result<FrameLaunchOutcome>{Launched(completion)|Exhausted|Impossible};
  prepare/lease/flush verbs deleted), cuda.rs/metal.rs backends,
  completion.rs gained SubmissionCompletion::all (remote aggregate),
  remote.rs = edge adapter (frame → per-step RemoteLaunch posts, frame
  completion = all(step completions); wire format unchanged).
- DONE driver/dummy crate: frame-native launch(&PieFrameDesc) ->
  FrameAdmission{Launched,Exhausted,Impossible}; folded admission reuses
  prepare knobs; whole-frame RETRY knob; execute_step per step; lease
  machinery deleted. ALL 13 TESTS GREEN.
- DONE engine scheduler (ALL 351 LIB TESTS GREEN):
  - frame.rs = seal policy only: FramePlan::Dispatch(Vec<Vec<u64>>) pops the
    WHOLE sealed frame; makeup/wave_outstanding/open-wave machinery deleted;
    plan_dispatch(queued, blocked_lanes, executing, now) — blocked_lanes
    holds a frame whole for copy barriers; SealedFrame = {waves, members}.
  - worker.rs: post_frame() replaces dispatch_frame_batch (whole-frame
    pick → build_frame_submission → one LaneRequest::Launch); depth gate =
    frames in flight; AdmissionWatermark/prepare round-trip/lease/
    FlushSettle/settle_defer plumbing deleted; lane Launch arm does the
    folded-admission loop (EXHAUSTED retries IN PLACE on the FIFO lane,
    200µs × ≤25k ≈ 5s then loud failure; IMPOSSIBLE → launch failure).
  - RETRY deleted end-to-end: a RETRY terminal at frame settle → loud
    per-fire FAILED ("not a v14 outcome"); retry_count/retry_after/
    classifier/backoff/budget all gone. CONTRACT NOTE: raw prebuilt fires
    must have consumed inputs staged BEFORE submit (the §5 philosophy —
    tracked fires always did); the old retry-until-host-put behavior for
    raw fires is gone (writer_ring_backpressure test deleted).
  - batch.rs: build_frame_submission — step formation reuses
    LaunchGrouping (instance/pipeline dedup, mask/solo exclusions, budgets,
    NEW geometry-class homogeneity rule); each group = one STEP with one
    sub-batch, so every step's wire is byte-identical to the old wave
    batch. Roster first-appearance order; per-lane translation = last
    fire's overlay; required_kv_pages = frame-union high-water.
- DONE CUDA C++ driver frame entry (44/45 driver tests green; the 1 failure
  `pipeline_dispatch_bind` is PRE-EXISTING at HEAD — verified by stashing
  the Venus driver changes and re-running):
  - `pie_native::StepLaunch` (driver/common/include/pie_native/
    step_launch.hpp) = the v13 batch shape as an INTERNAL type; driver
    internals renamed to it wholesale, so the per-batch pipeline
    (validate_finalized_launch → LaunchScratch → handle_fire_batch) is
    untouched.
  - context.cpp: `launch(const PieFrameDesc&, completion)` — expand_step
    (roster→instance ids, frame translation → per-step CSR slices),
    validate ALL steps first (frame atomicity), folded admission (one
    atomic commit of frame-union kv/state targets; EXHAUSTED/IMPOSSIBLE
    return as statuses), step enqueue loop with the REAL completion only
    on the tail (host-func callbacks are stream-ordered); mid-loop
    exception → FAILED terminals for remaining steps + notify. Lease
    registry/prepare/release/flush deleted; resize_pool floors replaced by
    the quiescence gate (= horizon-empty, Venus D6).
  - abi.cpp: single `pie_cuda_launch(driver, PieFrameDesc*, completion)`;
    abi_validation.hpp gained validate_frame_desc/validate_step_desc;
    entry_validation.hpp resource validator templated over launch shape;
    driver tests updated (frame-wrap helper, roster fixtures).
- DONE metal C++ driver source-level port (same recipe: StepLaunch rename,
  frame entry + expand_step loop in context.cpp, lease surface deleted,
  abi.cpp frame entry; CANNOT compile on this Linux box — macOS
  verification deferred).
- DONE workspace green: pie-worker executor rewraps each merged wire
  launch as a single-step frame (`single_step_frame` helper); cargo check
  --workspace 0 errors. (`cargo test --workspace` link failure of the
  engine test binary vs CUDA symbols is PRE-EXISTING config, not Venus.)
- DONE live certification round 1 (real GPU, Qwen3-0.6B):
  - Oracles TOKEN-EXACT k∈{1,2,3} t=32 and k=1 t=256 vs pre-Venus dumps.
  - k=1 2048×32: 35.34–35.37k (pre-Venus band 35.3–35.5k — HELD).
  - c0-256: 28.07k (band ~28.0k ✓). 512×512@c256: 19.80–19.82k ×3
    consecutive, 512/512 (pre-Venus 19.2–19.7k ✓+). NOTE: 512×512 must
    run at concurrency 256 — c512 physically exceeds the elastic budget
    (11511 vs 10756 pages; same pre-Venus, documented in phase2).
  - k=2 2048×32: 28.0k — EXPECTED regression vs k=1 until the
    driver-internal phase lands (publications_done still couples steps).
- FIXED (regression found in cert): **ResizePool barrier × frame
  atomicity crawl.** Gen-boundary reclaim posts ResizePool controls; with
  the async-control barrier marking every fire behind the queued resize
  and frames being atomic, whole frames held per resize cycle → 512×512
  gen-2 collapsed ~45× (436 tok/s). Pre-Venus survived because partial
  waves posted around the barrier. Fix: ResizePool no longer barriers
  fires (pure capacity op; the driver's quiescence gate holds correctness
  and engine grants make live-page trims impossible). Data-dependent
  copies (CopyKv/CopyKvTracked/CopyState) still barrier. Debug method for
  the record: coarse 2s `[venus-diag]` counters (PIE_VENUS_DIAG=1, still
  in worker.rs — DELETE at final cleanup) revealed plan=Hold(500µs)
  spinning with front_kinds=[resize_pool, launch…] blocked_lanes=256.
- Admission-demand semantics fixed during cert: required_kv_pages is
  DECLARED demand + wire page maxima only (v13 semantics) — folding
  translation ids in demanded full-arena commits and broke oversubscribed
  long-context shapes (engine batch.rs, worker single_step_frame, dummy).
- uv wheel trap NOTE: python-side "Installed in <1s" = NO rebuild; verify
  with `strings _engine.so | grep <new-literal>`; anchor-asserted python
  edits only (a silent no-op replace cost a debugging hour).
- k=2 LIVE ATTRIBUTION (the driver-internal target, measured on 2048×32):
  k=2 = 27–28.7k vs k=1 35.4k. Fire-timing per run: `h2d_prepare`
  Σ19.8ms(k1) → Σ316ms(k2); `begin_pull_validate` Σ~40ms(k1) →
  Σ180ms(k2). NOT the pinned staging pool: raising kUploadStagingDepth
  4→13 (runahead.hpp, steps-in-flight sizing — kept) changed nothing, and
  PIE_SCHED_MAX_IN_FLIGHT=1 (2 steps in flight ≤ old pool) still gives
  27.5k. Root cause: within one frame call, step 2's `begin`
  (pull_validate) and h2d path HOST-BLOCK on step 1's GPU publication
  (`publications_done` is a stream wait, but the pull staging/readback is
  a host sync) — ≈1 GPU step of lane stall per frame, collapsing
  enqueue-ahead to ~synchronous. Pre-Venus k=2 (34.7k) avoided it because
  per-wave posts naturally spaced wave 2's begin after wave 1's
  publication. THIS is P1's closed-system work: step i+1 must not
  host-read step i (per-step device-side state; snapshot ring exists —
  `commit_snapshots` is already per-occurrence; the remaining host reads
  are in StagedLaunch begin pass-C/pull_validate + the h2d readback).
  Facts for the redesign: `publications_done` is cudaStreamWaitEvent (not
  host); commit snapshots are already ringed per instance_occurrence;
  k=1 at depth 1 gives 26.7k (run-ahead is worth ~9k — the two-track
  model's value confirmed).
- Wheel-build trap #2: pyproject cache-keys covered only driver `*.rs` —
  C++/CUDA edits did NOT invalidate the wheel. FIXED: added cpp/cu/cuh/
  hpp/h/CMakeLists keys to sdk/python-server/pyproject.toml.
- CHECKPOINT COMMIT: e12ca83d1 "venus: frame-native launch unit end to
  end (ABI v14, checkpoint)" — everything above committed on the golf
  branch (vesuvius-project.md operator edit left untouched, NOT pushed).
- k=2 hunt continued (inconclusive at the API level, attribution stands):
  c0-256 (zero churn): k1 28.1k vs k2 23.1k → the gap IS steady-state
  intra-frame coupling (pre-Venus k2 c0256 was 26.5–27.4k). Probes that
  did NOT move k=2 (all kept, all principled): staging depth 4→13
  (runahead.hpp), default mempool release-threshold retention
  (context.cpp initialize), PIE_SCHED_MAX_IN_FLIGHT=1 (27.5k → not pool
  exhaustion). nsys (venus-k{1,2}.sqlite in the scratchpad): BOTH k show
  369× cuMemUnmap/SetAccess/Release ≈220ms (elastic pool map↔unmap
  oscillation — a separate cleanup target, not the k differentiator);
  lane-thread steady-window API sums do not show a 2.8ms/step block →
  the fire-timing h2d/pull stalls are likely non-API host waits (pinned
  slot event syncs measured under cudaEventSynchronize at k=1 too) —
  STOP microhunting; the fix is structural (below).
- k>1 ROUND 2 (staging-ring fixes; oracle k∈{1,2,3} token-EXACT after):
  code-level trace of the per-step submit path found two structural
  host-block candidates and fixed both on principle (every per-step
  staging pool sizes from runahead.hpp):
  - AttentionWorkspace plan staging had TWO slots (attention_workspace)
    — at k=2 with 6 steps in flight, `begin_plan_update`'s
    cudaEventSynchronize waits ~4 GPU steps EVERY step. Ringed to
    kUploadStagingDepth (slot 0 eager for non-rotating users like the
    VL vision adapter; the rest pin lazily on first rotation).
    MEASURED: k=2 `h2d_prepare` Σ316ms → Σ32ms (= k=1's 36ms). CONFIRMED
    as that leg's cause.
  - `begin`'s ticket/pull-lane upload was cudaMemcpyAsync from PAGEABLE
    std::vectors + 2×cudaMallocAsync per step → new PinnedUploadRing
    (channels.hpp, 13 pinned slots) + ONE [tickets][lanes] blob and one
    H2D per step (blob doubles as the launch's device ticket array;
    settle frees it). NOTE: `begin_pull_validate` still reads ~0.8-1.1ms
    per submit at BOTH k after this (was Σ40ms k1 / Σ180ms k2, now
    Σ141/Σ188 over ~170 submits with ft on) — no longer a k
    differentiator; possible ft-perturbation artifact, worth a clean
    look during final cleanup.
  - Throughput did NOT move: k=1 35.07k, k=2 28.1-28.3k, k=3 28.5k. So
    the host legs were real but not the k bottleneck.
- k>1 ROOT CAUSE FOUND (guest-side, not driver): the bench inferlet's
  unified run-ahead window was `WINDOW_FIRES = 2` FIRES. Venus frames
  settle ATOMICALLY (P4), so results arrive only at frame boundaries:
  at k=2 the guest's 2-fire window is exactly one frame — ZERO queued
  frames while a frame runs; every frame pays a full settle→take→
  submit→seal→post round trip. Quantitatively consistent: k=2 28k ≈
  k=1@depth-1 26.7k (both are pipeline-less), and the old
  PIE_SCHED_MAX_IN_FLIGHT=1 probe (27.5k ≈ default 28k) is explained —
  k=2 was ALREADY effectively depth-1. Pre-Venus k=2 (34.7k) survived
  because per-wave settlement returned results mid-frame.
  FIX (inferlet): window measured in FRAMES — keep TWO FRAMES in
  flight, `window_fires = 2k` (k=1 reproduces the classic depth-2
  exactly); `out` ring capacity k+1 → 3k (window peaks at 3k-1 and a
  full publish ring is a terminal FAILURE under v14, so slack one).
  The venus-project.md arrival contract must say "guest window ≥ 2
  FRAMES", not "≥ 2".
- ROUND-3 RESULT (two-frame window, commit 7d80778f8): k=2 28.1 →
  34.34/34.36k ×2, k=3 28.5 → 33.6k, oracles token-EXACT k∈{1,2,3} —
  the guest window WAS the k>1 bottleneck. The compose-split /
  frame-prepare-hoisting driver refactor is NOT needed for the perf
  goal (the per-step submit path is already non-blocking after the
  staging rings; the residual k gap is not host legs).
- k=1 REGRESSION HUNT (bimodal ~32.5k vs 35.3k band, mostly slow):
  - out-ring capacity NOT causal (3k → 3k-1 = old size still slow).
  - CONTROL (15c121786 driver+inferlet rebuilt and run same day):
    35.30/35.33/35.08 — machine fine, regression IS in the round-2/3
    changes.
  - BISECT A (plan-ring only + new inferlet): 35.36/35.60/35.27 —
    plan ring AND inferlet window both innocent (35.6 is a new high).
  - CULPRIT: the PinnedUploadRing BLOB — fusing tickets+lanes into one
    settlement-freed allocation moved the per-step lane table from the
    "freeAsync on the same stream right after enqueue" fast path to the
    cross-stream (settlement-stream) free path; the stream-ordered pool
    then re-serves the next step's mallocAsync from a colder path and
    k=1 drops to a slower steady mode. LESSON (measured, general):
    with cudaMallocAsync, ALLOCATION LIFETIME IS PART OF THE FAST PATH
    — free transient device tables on the allocating stream immediately
    after the consuming enqueue; only genuinely settlement-lived data
    (the ticket array) may cross streams.
  - FIX: keep the pinned host slot ([tickets][lanes] in one slot — the
    non-blocking part), restore the device side to two allocations with
    the original lifetimes (lanes freed post-enqueue on-stream, tickets
    settlement-freed). Round-4 cert = this + plan ring + two-frame
    window.
- ROUND-4 (split-alloc pinned pull upload + 3k-1 ring): k=1 STILL mixed
  (35.40/32.96/32.63) AND k=2 collapsed to 28.0×2. Two confounds — so
  round-5 separated them.
- ROUND-5 (VERDICT, the state that ships): pull-upload staging fully
  REVERTED to the v13 pageable form (PinnedUploadRing deleted in both
  variants); plan-staging ring kept; out ring 3k. Result: k=1
  35.24/35.42/35.32 (band, stable), k=2 34.06/34.55, k=3 33.83,
  oracles token-EXACT k∈{1,2,3}. Attribution nailed by A/B:
  - k=1 slow mode ⇔ PinnedUploadRing present (blob OR split form —
    even with original device lifetimes). ~10µs/step of extra host
    work cannot explain 8%; mechanism unproven (suspect: pool/stream
    interaction of the extra per-step event record on the compute
    stream). LESSON: begin_pull_validate was NEVER the k
    differentiator (equal per-step at both k) — the change had no
    measured upside and a measured downside; reverted on principle.
  - round-4's k=2 collapse ⇔ the 3k-1 out ring (throttles to ~one
    frame in flight, no failures — engine ticket staging needs publish
    room one frame beyond the window peak before takes are observed).
    Ring restored to 3k.
- PROBE (3-frame window, 4k ring): k=2 34.40/34.22 — identical to the
  2-frame window; k=1 35.48 unharmed. Window depth is NOT the residual
  k gap (34.3 vs 35.3, ~3%); reverted to the committed 2-frame
  discipline. The residual gap is a boundary/burst effect (results
  arrive k-at-a-time; drain+resubmit bursts) — accept for now; frames'
  value case is host-cost slack (the inequality), not peak c512 tput.
- ROUND-6 (cleanup deletions, certified): M2 settle_defer apparatus
  deleted end-to-end (dispatch.cu accumulator/park/drain/arm +
  flush_deferred_settlement + NotifyContext marker + LaunchView field +
  context.cpp assignment; DeferredSettle → SettleRecord, settle path is
  one unconditional settle_wave_record per wave). venus-diag deleted
  from worker.rs. Prefill-capture lever deleted end-to-end
  (PIE_PREFILL_GRAPH_* env + prefill_graph_ready chain through
  ForwardFn/IModel/llama_like plan state/tp follower mirror +
  prefill_plan_graph_capturable ops helper + the memory planner's
  cuda_graphs carve param — replay predicate is now decode-only by
  construction; decode graphs untouched). RETRY-era engine leftovers
  deleted (RetryClassifier type, QueueEnd::Front requeue,
  BatchAccumulator slimmed to AdmissionLimits (admission shape gate),
  unused submit_pass wrapper). KEPT deliberately: the two
  admission-IMPOSSIBLE stderr lines in context.cpp (terminal-failure
  operator output, once per failing frame — not debug cruft).
  Engine tests 351/351; wheel compiles; settle_defer symbol absent;
  oracles token-EXACT k∈{1,2}; k=1 35.67k (new high), k=2 34.37k.
- S3 kill-bit reclassification (round-7, implementation): the ringed
  commit snapshot widens to TWO adjacent u32 words [pass_commit, kill].
  Pull-validate seeds both (kill=0 — ring slots carry stale kills
  otherwise); the fixed-decode and envelope compose fail-stops now set
  kill=1 alongside zeroing commit (chain-kill counters unchanged);
  settlement mirrors both words (mapped path: settle kernel
  store_system_release on word 1; unmapped: the existing D2H pair
  widens to 8 bytes); abort() zeroes both. The completion callback
  classifies killed lanes FAILED (poison path: channels poisoned,
  FAILED terminal) — RETRY is now reserved for host staging-contract
  violations, exactly what worker.rs's retire arm already asserted
  ("deterministic compose kills latch FAILED"). Kill-path exercise
  needs a fault-injection run (not in the hot cert); logic is
  compile-verified + settle-order reviewed (callback is stream-ordered
  after the settle kernel on both paths).
- Elastic map↔unmap oscillation: DEFERRED with a design sketch, not
  done in this pass. Cause: gen-boundary ResizePool controls trim the
  workspace/state arenas (`trim_bytes` → cuMemUnmap per map unit) and
  the next frames' folded admission regrows them — 369× unmap
  ≈220ms/run at both k (~0.4%). The correct fix is a reclaimable-donor
  design: a resize returns LOGICAL budget immediately (so KV can grow —
  the 512×512 shape depends on it) but keeps pages mapped, and the pool
  physically reclaims donor pages (unmap) only when try_reserve would
  otherwise fail. That touches the pool's budget accounting — a real
  subsystem with oversubscription edge cases — for 0.4%; poor ROI at
  the tail of this build, so recorded as the next standalone item.
  NOT acceptable as a quick hack: skipping the trim outright breaks
  oversubscribed shapes, and a persistence/decay heuristic would be a
  tuned magic constant (venus rules: no heuristics).
- FINAL CERTIFICATION (2026-07-24, all green, the landed state):
  - Oracles token-EXACT: k∈{1,2,3} t=32 and k=1 t=256 vs the pre-Venus
    dumps.
  - 2048×32 c512: k=1 35.68/35.13k (band 35.3–35.5 held), k=2
    34.05/34.09k, k=3 34.00k, k=4 33.43k — k>1 sits 3–6% under k=1
    (was 20%+ before the two-frame window), zero failures at every k.
  - c0-256 (512×256 c256): k=1 28.09k (band ✓), k=2 27.93k (pre-Venus
    k=2 was 26.5–27.4k → EXCEEDS).
  - 512×512 c256: 19.79k (band 19.2–19.8 ✓). 64×1536: 7.38k
    (pre-Venus 7.31k ✓).
  - Residual k gap (~3–6%) is boundary/burst quantization, not window
    depth (3-frame probe flat) and not driver host legs (measured
    collapsed); frames' value case is host-cost slack (the
    inequality), and the gap shrinks as models get larger (GPU step
    time grows, boundary cost fixed).
  macOS metal compile verification deferred (metal has no compose
  kill path to port). Elastic map↔unmap reclaimable-donor design
  recorded above as the next standalone item.
  PROJECT VENUS BUILD: COMPLETE on this branch (nothing pushed).

- SECOND LANDING round 1 (frame-prepare hoisting, item ① of the committed
  roadmap; operator: "구현 시작. backward compatibility 필요 없음").
  The `handle_fire_batch` monolith (batch/compose.cpp, deleted) dissolved
  into `batch/frame.{hpp,cpp}`: `prepare_step` (FramePrepare — begin_host
  wave admission, descriptor resolve, compose, mask decode, RS/sampling/
  pad planning, and the step parameter block STAGED into per-buffer pinned
  slots) / `enqueue_step` (StepEnqueue — begin_enqueue pull-validate +
  Prologue, staged-upload commits in the original per-fire order, pack/
  pad/envelope kernels, attention-plan hook, forward body, MTP) /
  `settle_step` (FrameSettle — Dispatch::finish; tail carries the frame
  completion). context.cpp's frame loop: prepare ALL k steps at frame
  entry, then enqueue+settle per step in order. Supporting splits:
  - `Dispatch::begin` → `begin_host` (passes A/B/C, sequence applies —
    host-only) + `begin_enqueue` (ordering waits, mallocAsync,
    pull-validate upload+launch, Prologue). `begin` remains the fused
    wrapper for `run()`.
  - `DeviceBuffer::copy_from_host` → `stage_from_host` (host memcpy into
    the existing 13-slot pinned ring; prepare-time) + `commit_staged`
    (async H2D; enqueue-time). Ring depth already covers 3 frames × 4
    steps + 1, so staging a whole frame ahead never re-claims a live slot.
  - Frame failure policy: prepare fault → nothing enqueued, abort every
    staged wave, ALL steps' terminals FAILED; enqueue fault at step i →
    steps <i live (settle normally), abort/FAIL i..k-1. Both paths keep
    today's settle_failed_launch + completion notify shape.
  - rank-0 path no longer stages RS metadata through the shared
    pi.*_host mirrors (frame steps own their spans; ForwardDispatchInputs
    gained rs_slot_flags_h). Mirrors stay for the TP-follower receive
    path.
  DELIBERATELY NOT hoisted (recorded): the attention-plan hook stays at
  StepEnqueue. FlashInfer plans fuse host compute with the H2D commit
  into the SINGLE stable int workspace (graph-replay hardcodes that
  address), and `DecodePlanCache.plan_info` is one mutable host struct the
  body reads at launch — planning k steps ahead corrupts step i's plan
  with step i+1's. The clean fix is a per-step plan snapshot through the
  model layer (own landing). Better fact found on the way: intra-frame
  decode steps have IDENTICAL static-nonsplit plans (same R; the schedule
  is KV-length-independent), so the end state is plan-ONCE-PER-FRAME, not
  plan-per-step-hoisted.
  Fire-timing legs re-scoped to the split (prepare legs vs
  h2d/forward/settle enqueue legs; host_total = sum of the two exclusive
  phase spans).

- SECOND LANDING round 1 debug: the k>1 chain kills, and THE WINDOW LAW.
  First hoisted build: k=1 green (35.4k), EVERY k≥2 run dead — the
  fixed-decode compose kernel fail-stopped whole chains, single-lane k=2
  oracle included. Root cause chain (device printf bisect: pv seeds →
  phase probes → settle probe; then a loop-fusion control run that went
  green and proved the phase SPLIT sound and the CROSS-STEP hoist at
  fault):
  1. `apply_lane_sequence_tickets` (begin pass C) is not just wave
     admission — it maintains the host mirror cursors (`host_head_/
     host_tail_`) that EXECUTION-TIME metadata builders read (the fused
     stage build's unticketed fallback, settlement prep). Hoisting all
     applies to frame entry let step X's applies move the mirrors before
     step W's Prologue/Epilogue metadata was built → W's epilogue bound
     another wave's cells → its generated readiness AND'ed pass_commit
     to 0 (silently — generated stages don't use k_stage_readiness) → W
     dummy-ran → no publishes → every successor's pull-validate found
     tail short by one → chain kill. THE LAW: host mirrors must advance
     at each wave's ENQUEUE position, in wave order — they are the
     enqueue track's clock, not admission bookkeeping. Applies now live
     at the head of `begin_enqueue`.
  2. FramePrepare-time consumers therefore must NOT read mirrors at all:
     the wave's window comes from its TICKETS. Flag-free tickets
     (read-only channels) are now kept in the lane ticket list (device
     side is entirely flag-gated — they are inert there) purely to carry
     positions; `lane_ticket_window` resolves prepare-time cell/cursor
     lookups (stage_fixed_decode / stage_decode_envelopes tables,
     resolve_descriptors' readback pack) from expected_head/tail, falling
     back to live mirrors only for engine-unsequenced channels (apply-
     invariant by definition).
  3. The composition table build itself moved to FramePrepare
     (`stage_fixed_decode`/`stage_decode_envelopes`; arena claim + compose
     kernel stay in the enqueue halves) — at enqueue time the tables read
     post-ALL-applies state, which is exactly the fleet-kill mechanism
     of the very first broken run.
  4. `prior_put_slots`/`prior_take_slots` fill at phase EXECUTION, and
     the Prologue's own metadata build must not see its own effects —
     prepare-time consumers get a separate `prologue_put_slots`
     (statically derived from the prologue stage plans at begin_host).
  5. Prologue order restored: it historically executed against the
     pre-resolution wave state; `update_launch_geometry` now runs at
     enqueue AFTER the Prologue (its old position relative to it).
  After the fix: k=2/k=3 oracles token-EXACT vs the pre-Venus base, zero
  kills. All temp diagnostics removed before certification.

- SECOND LANDING item ② closed by measurement (no code). Depth
  dose-response re-run on the hoisted build (2048×32 c512,
  PIE_SCHED_MAX_IN_FLIGHT ∈ {1,2,3}): k=1 → 26.7 / 35.5 / 35.2k; k=2 →
  28.6 / 34.4 / 34.0k. Depth 2 remains the optimum at both k — the N9
  result survives hoisting, so the admission gate is staleness-limited,
  not capacity-limited, and a capacity-derived gate would regress a
  measured optimum. Recorded in venus-project.md item 2; the standing
  relation (engine depth = measured policy; driver staging derives from
  it via runahead.hpp) is already the correct one.

- SECOND LANDING round 2: plan-once-per-frame LANDED (operator-approved
  as a pure upgrade; ④ delta-form geometry DEFERRED by operator decision
  — guest-surface exposure not wanted, no current backend needs it).
  Mechanism: prepare_step(i) content-compares EVERY host-consumed
  attention-plan input (R, N, decode/mask/window flags, qo/kvpp/kvlpl/
  kvpi arrays byte-exact) against the SAME frame's previous step; on
  identity, enqueue_step skips the plan hook — the workspace already
  holds the identical plan (intra-frame chained decode steps share R and
  plan from frame-constant envelope bounds). No heuristic: any
  difference whatsoever plans normally; frame head always plans.
  Certified: oracles k∈{1,2,3,4} token-EXACT; 2048×32 k=1 35.2k, k=2
  34.1/34.4k, k=3 33.4k, k=4 33.3/33.2k (one 13.6k reading was a
  non-reproducible anomaly — n=3 confirms the band), c0-256 k=2 27.9k.
  No tput change at today's T_gpu (expected); the win is structural —
  chained StepEnqueue is now literally kernel-launch/copy-commit only,
  completing ①'s contract.

- SECOND LANDING round 1 CERTIFIED (frame-prepare hoisting landed).
  Oracles k∈{1,2,3} t=32 + k=1 t=256: token-EXACT vs pre-Venus dumps.
  2048×32 c512: k=1 35.58/35.36k (fast band 35.3–35.5 HELD — the
  restructure costs k=1 nothing), k=2 34.06/34.21k (first-landing band
  34.05–34.09 held), k=3 33.65k (band ✓), k=4 33.16k (−0.8% vs 33.43,
  run noise). c0-256: k=1 28.18k ✓, k=2 28.00k (≥ first landing's
  27.93k). 512×512: 19.82k (top of band). 64×1536: 7.39k ✓. Zero
  failures at every shape and k. Engine untouched this round (driver
  C++ only). Net state: per-step host compute (admission passes,
  resolve, compose, masks, RS/sampling planning, pad decision,
  parameter-block staging, composition tables) runs once per frame at
  entry; StepEnqueue is applies + pull-validate + prologue + staged
  commits + kernels + the attention-plan hook (recorded follow-up:
  plan-once-per-frame via a model-layer plan snapshot).

## Done so far

- venus-project.md: decisions 1–8 closed (single path, one ABI jump,
  sub-batch steps, snapshot ring, RETRY gate, dormant-lever deletion,
  boundary cancel, k=2 const), build strategy = north star.
- RETRY audit (gate #5): `pass_commit` writers exhaustively enumerated
  (seed, k_stage_readiness need-full/need-empty, two compose kills).
  S1 vacuous by §5 arm; S2 → sub-batch ordering + whole-frame RETRY
  backstop; S3 (deterministic kills) reclassified per-lane FAILED with a
  kill bit in the snapshot-ring entry. Per-lane RETRY, makeup replay,
  retry budgets, force-retry battery all delete.
