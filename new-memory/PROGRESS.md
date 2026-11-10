# new-memory progress

Design doc: https://claude.ai/artifact/3QFaEVRYHGJrZMquXdsG2H
Shared spec for implementation agents: [SPEC.md](SPEC.md)

This branch collects every merged package of the state-chunk memory redesign plus this log.
The `new-memory/` directory is bookkeeping and is removed before anything lands on main.

## Milestones

### M0 — 2026-11-09: design fixed, batch 1 launched
- Final north star in the design doc (chunk/slot/segment, C = f(model, DeviceProfile), fixed/list/ring,
  rs buffer as fixed rows, governor/PLEX/pool roles, per-backend realization, degenerate-mode-first migration).
- Batch 1 in flight, one branch each (`sc/<package>`):
  - P1 poem `state()` handle API (pilot: qwen_3, nemotron_h)
  - P2 stream scheduler: per-row dependencies instead of per kv space
  - P3 compiler `chunk::plan` -> ChunkRecipe (server CUDA and on-device Metal presets)
  - P4 CUDA kv page stride (degenerate mode, bitwise)
  - P5 CUDA state segment table (degenerate mode, bitwise; Wan conv3d frame cache cuttable)
  - P6 Metal kernels (degenerate mode), verified on the Mac Studio
  - P7 plan: engine API, runtime allocator, governor, backend pool switches (read-only)
  - P8 verification baseline: golden logits probe + pre-existing failure list

### M1 — 2026-11-09: stage plan (P7) done, batch A launched
- P7 (read-only plan) done: [PLAN-P7.md](PLAN-P7.md). Engine `ChunkFacts` API, runtime `SlotPool`,
  six places where the runtime drops state on its own -> `tier-pressure` to PLEX, governor on segments,
  per-backend pool switches (CUDA/ROCm/Metal/Vulkan/wgpu/XLA/Cerebras), kv-space removal K1-K3, batches A-F.
- Decision: the runtime may reclaim idle cache entries on its own only when their PLEX retention key is <= 0;
  everything else goes through `tier-pressure`. R4 ships a default responder so waiters cannot deadlock
  when no worker policy is installed.
- Found while planning (existing bug): wgpu silently clamps over-limit storage bindings
  (engine-wgpu/src/encode.rs:350-355) — being turned into an error in BWk-prep.
- Batch A in flight: E1 engine chunk API, R1 runtime SlotPool, G1-core governor on segments,
  R4 tier-pressure plumbing + default responder, BWk-prep (`Removed` config keys, wgpu/vulkan binding limits).
- Batch 1 still running: P1, P2, P3, P4, P5, P6, P8.

### M2 — 2026-11-09: G1-core merged
- `sc/g1-governor-core` (d8ce7ce9c): the governor drives the pool only through a `PoolTarget` trait
  (demand, segment_bytes, floor, growth Fixed|Elastic, retarget -> Reached|NoRoom, shrink_rung/restore_rung)
  and moves in whole segments. No method can drop state. Shrink without room keeps the segment and raises
  `TierPressure` once (then `Eased`); Critical drops to the floor and a smaller rung; Fixed pools never shrink.
  Today's `Extent` implements the trait (1-byte segments), so behavior is unchanged.
- Verified: `PIE_BUILTINS=skip cargo test -p runtime --lib governor` 34/34, clippy clean; `cargo check -p runtime` on new-memory.
- Environment note: the builtins build script needs nightly-2026-10-18 with the cargo component, missing here;
  runtime tests run with `PIE_BUILTINS=skip`.
- Remaining for G1-wire: PoolTarget on R1's SlotPool, Signal -> R4 tier-pressure, real rung selection
  (then delete yield_scratch/narrow/NARROW_WAVE), replace ExtentPool/extent.rs after R2.

### M3 — 2026-11-09: E1 merged
- `sc/e1-engine-chunk-api` (b87e5f135): new `engine::chunk` (`ChunkFacts`, `SegmentFacts{growth: Fixed|Bottom|TwoEnded}`,
  `ReservedRun{Null@0, Sink, Runtimeless}`, `SlotShape{Fixed|List|Ring, slot_bytes, slots_per_chunk, sets_per_working_set,
  rows, pinned}`, `ChunkTable` logical<->physical permutation, `Evacuation`, `SlotCopy`, `fit`). Purely additive:
  `Capabilities.chunks: Option<ChunkFacts>`, `Lane.tables`/`Lane.fixed` + validation, Engine trait
  `commit_segments/release_segments/evacuate/copy_slots` defaulting to unsupported. `ChunkFacts::degenerate(&PoolFacts, &DegenerateBytes)`
  reproduces today's pools as pinned runs (null chunk 0, kv, window, state, rs buffer; fixed shape ids 0..3).
- Verified: engine 38/38 tests on new-memory after merge; `cargo check` engine, runtime, engine-cuda; the agent checked
  every dependent crate (--all-targets) on its branch.
- Accepted decisions: reserved chunks carry explicit positions; degenerate `chunk_bytes` is nominal (use per-shape
  `slot_bytes` for byte accounting there); `ChunkTable::apply` swaps (runtime must only move onto physical chunks of
  free logical chunks); `commit_segments(n)` means "back the first n segments".

### M4 — 2026-11-09: P6 Metal degenerate kernels merged
- `sc/p6-metal-degenerate` (741a6ebd6, 07335a451, 719cd4c13), verified on the Mac Studio (M1 Max, macOS 27;
  shaders compile at runtime from source, so the missing `xcrun metal` is not a blocker).
- KV: `kernels/prim/kv_page.h`; every paged kernel takes `k_page_stride, v_page_stride, k_offset, v_offset`;
  `KvPool{page_stride, plane_offset}`, `paged_densely()` = today. State: `kernels/prim/state_map.h`, a u64 table
  (slot_stride, chunk_slots, chunk_stride, seg_chunks, piece_atoms, pieces, segment bases) passed to every state
  kernel; pieces along heads (GDN/KDA/Mamba-2), channels (conv, multiple of 4 on the 4-channel path) or frames
  (frame cache); `StateMap::whole(stride)` = today. Names align with `engine::chunk` (E1).
- Bitwise: new `a_scattered_kv_pool_is_the_dense_one` (bf16 d64-512, 8-bit, packed 4-bit; decode, lse, tiled, MPP,
  prompt arbitration, split; 256B poison) and `a_scattered_state_bank_is_the_slab` (GDN, KDA, Mamba-2, conv, frame
  cache; pieces 2-4, reversed segments, NaN poison) — all identical. Suites equal to base: kernels-metal 1 pre-existing
  failure (ANE service over ssh), engine-metal 64 pass / 3 pre-existing failures.
- Not covered by scattered tests yet (default layout only): MLA, indexer, compressor, score capture, verify kernels,
  n-gram window.
- Metal segment/governor design for the pool switch: segment = run of one placement-sparse buffer (multiple of the
  16MiB heap, 256KiB tiles); commit via `elastic::map_all` on the Metal 4 mapping queue one segment ahead; return =
  evacuate the top segment then `shrink_all` after frames drain; target bytes -> segment count (round down on fall,
  one segment per calm tick up); Critical / refused map -> evacuate+release top segment; floor pinned in segment 0.
- Linux check: `cargo check` kernels-metal, engine-metal, runtime on new-memory.

### M5 — 2026-11-09: BWk-prep, R1, R4, P2, P3 merged
- **BWk-prep** (b37de3e4e, d7d729ce3): `Removed<G>` + `removed_keys!` — a removed config key refuses the file with
  "`key` was removed: <why>; delete the key" (TOML/JSON, `pie config set/get`, templates can't emit it). wgpu: a
  storage view past `max_storage_buffer_binding_size` is an error (was silently clamped) and `max_buffer` is capped
  at the binding limit; Vulkan: `max_buffer = min(maxStorageBufferRange, maxMemoryAllocationSize, maxBufferSize)`,
  over-range views refused (1GiB floor gone). Caveat: small-limit devices now refuse what used to load truncated.
  No real GPU run.
- **R1** (bbb669d1b): `runtime::store::{slots, segments}` — `SlotPool` with refcounted slots, holders
  (WorkingSet/CacheEntry/Snapshot), epoch-gated reuse, logical vs physical chunks, budget
  `reserved + state + act(rung) <= committed`, `demand()` in bytes, segments with `plan_evacuation` (whole chunks
  into lower segments), `commit_ahead`. 24 seeds x 3000-step property run. Perf vs store/pool.rs: one-at-a-time
  1.5x, batch of 1M chunks 11.7x slower (follow-up; real pools have far fewer chunks).
- **R4** (5b3fd0aab): `policy::pressure::{raise, relieve}`, bands clear/short/critical with hysteresis,
  `Edge::TierPressure` once per crossing, byte attrs `{device,host}-{total,state,live,wanted,act,short}-bytes`,
  per-working-set kv/state bytes, `may_drop(key) = key <= 0`, `on_tier_pressure` in the policy SDK, and a default
  responder (policy layer) used when no policy subscribes: discards lowest retention key first. Policy thread always
  runs. Follow-ups for R3: wire `may_drop` into reclaim_idle/reclaim_cache; waiters keep raising while parked.
  Decision: keys at `i64::MAX` are pinned and never discarded by the default responder (to implement in R3).
- **P2** (a190590c6): the stream scheduler orders regions by the cache/state row they touch, not by kv space.
  Proof over every deployment x {Cuda, Metal, Wgpu, Vulkan} x {default, dag_streams}: new edges are a subset of the
  old, 0 violations (132.9M dropped edges checked). 3640 plans: 3560 identical; under dag_streams 56 gain concurrency
  (gpt-oss-20b 72->1194, gpt-oss-120b 108->2709), 24 lose some (greedy lane assignment, e.g. qwen35-d9b-dflash
  452->383). Remaining `space` readers: IR fields (`CacheRow::Kv.space`, `Mask/Geometry{space}`), fixtures and three
  bake tests.
- **P3** (a754736ca): `poem_compiler::chunk::plan(trace, budgets, profile) -> ChunkRecipe` with List/Ring/Fixed
  layouts, atoms (Heads/Channels/Frames/Tokens/Whole), rs buffer derived as fixed `[16, width]` rows, waste
  breakdown, `per_rank(tp)`; `DeviceProfile.chunk` presets SERVER_CUDA / ON_DEVICE_METAL / TPU. All 455 deployments
  plan under both presets with invariants holding; worst waste on a real model ~0.9% server, ~5.9% Metal.
  Examples (server): qwen3-next C=17.2M, L16x384K + F36x1.3M (fixed 95.5MiB/ws incl. 2 sets); nemotron-super
  C=12.2M, F40x4.1M; gemma4-31b C=25M, L16x1.2M + R1024:16x12.5M*67; dsv4-pro C=25.2M, one list for all ~90 spaces;
  wan22 F447x9.1M (4GiB/clip). To review later: ring cap uses fire_rows=16 (bigger prefill borrows activation
  pages); both fixed sets include the rs buffer; server C lands 8-32MiB; preset numbers are placeholders.
- Verified on new-memory after merges: runtime lib 186/186 (builtins now build: nightly-2026-10-18 got its cargo
  component), worker 12/12, poem-compiler lib 28/28, `cargo check` engine-cuda/metal/xla/wgpu/vulkan, poem-exec.
- Still running: P1 (poem state() API), P4 (CUDA kv stride), P5 (CUDA state segments), P8 (golden baseline).

### M6 — 2026-11-09: BF and R3 merged; fixed-set rule corrected
- **Spec correction:** fixed state sets are ONE per working set plus ONE copy-on-write set per lane that runs in a
  fire (a global reserve that scales with max lanes), not two per working set. P3's recipe still counts 2 sets per
  working set in `chunks_per_ws` / fixed MiB per ws — follow-up to fix.
- **BF** (1ba72e08e..6ccb5edec): XLA, Cerebras, Metal, Vulkan, wgpu fill `Capabilities.chunks` with degenerate
  `ChunkFacts` from their own stores (engine-cpu reports no capabilities) and log the `chunk::plan` recipe at debug
  (Metal: ON_DEVICE_METAL, XLA: TPU, others: SERVER_CUDA). XLA/Cerebras sinks are a `Reserved::Sink` run after the
  pools; Metal sparse heaps report `Growth::Bottom`. Tests: XLA/Cerebras dry-load 4 catalog models and read counts
  back exactly; Metal on the Mac (lib 44, integration 65/69 with the 3 pre-existing failures + 1 fixture skip);
  Vulkan/wgpu unit tests only. Mapping gaps noted: per-plane watermarks on Metal, Vulkan/wgpu copy-shrink reported as
  Fixed, XLA rs-buffer plane has no sink page while padding lanes use `sink_slot = slots` (possible latent
  out-of-range, untouched), Metal/Vulkan/wgpu page sliding-window rows in full (no ring shape yet).
- **R3** (244412899, 4cb817555, fe1950603, 304f22050): `yield_checkpoints` family, checkpoint drops in
  `prepare_suspend` / `reclaim_tail`, `state_room`, `rs_seat_cap` and its depth override are gone. Suspend moves
  sharing checkpoints to host with their set; no host room -> nothing moves, host pressure raised. Reclaim drops only
  retention key <= 0; anything else -> `tier-pressure`. Waiters park, re-raise every 100ms, fail after SEAT_WAIT
  (10s) with `PlannerError::Short`. Default responder never discards key `i64::MAX` (pinned). Admission: one fixed
  set per process + a global copy-on-write reserve of `max_forward_requests` lanes (5 slots, 1 lane -> 4 processes).
- Verified on new-memory: runtime lib 191/191 (1 ignored); `cargo check` xla, cerebras, metal, vulkan, wgpu, runtime;
  XLA lib 8/8.
- Still running: P1 (poem state() API), P4/P5 (CUDA degenerate kernels), P8 (golden baseline), R2 (stores on SlotPool).

### M7 — 2026-11-09: P1 poem state() handles merged
- `sc/p1-poem-state` (abb08e8d7, 01f3bb712, 06895046c, 375739065): `state(shape, dtype)` declared in the layout like
  `weight()`; first axis `tokens` (or `tokens // r`) = per-token, else fixed. Rows are named by model-tree path
  (`layers.9.attn.kv`, `layers.0.mamba.conv_state`, `mtp.kv`). Window/head_dim come from the reading attention op,
  head split for GQA readers, conv split on channels, GDN/KDA/mamba2 on heads. Lowers to today's `CacheRow` (one kv
  space per (dtype, window, rate)), so compiler/engines are untouched. `kv_append` binds write positions itself.
  `caches()` still works for unmigrated models (a package may not mix both).
- Pilot: `qwen3_dense` (dense attention; `qwen_3.poem` is GDN+gated attention+dflash, not migrated yet) and
  `nemotron_h`. 123 deployment x tp traces: lowered caches identical except names; whole-trace canon diff identical
  for 255 traces (and 116 traces of other models sharing lib/blocks.poem).
- Costs/compromises: handle models trace twice (+1% .. +26% total trace time); an unread per-token state copies the
  layout of a read state with the same shape/dtype.
- Open for the remaining migration: unequal-width planes (MLA latent+rope), deepseek `packed()` / deployment-dependent
  split / `rows=` and `live=`, window inference for WindowSink and MLA ops, inkling k/v_state and engram/replay splits,
  lib/dflash.poem and pattern.rs drafter rows.
- Verified on new-memory: pie-poem 14+30 tests; `cargo check` poem-compiler, runtime, engine-cuda.
- Still running: P4/P5 (CUDA degenerate kernels), P8 (golden baseline), R2 (stores on SlotPool).

### M8 — 2026-11-10: weight cache decided; WC2 merged
- Design: routed experts and PLE rows become read-only weight slots in the chunk pool (design doc section
  "weight cache"; spec section "Weight cache"). `weight_streaming = auto|on|off`; on => under pressure weights shed to
  the floor (MoE: current + next layer staging; PLE: this fire's rows) before any tier-pressure; off => never evicted.
  Inside the cache: LRU, but misses stream through without insertion and membership changes only per window tick
  (prevents thrashing on the per-token layer cycle). Streamability inferred by the compiler from the reading op
  (no new poem declaration; `Weight::host()` to be replaced). Prior art compared: Rimuru (arXiv 2610.40305).
- **WC2** (3c80c7adc, ee04fb0b5, 84a9c44fc): `[model] weight_streaming` (auto: stream only if the load reports weights
  not all resident); `ChunkFacts.weights` (Expert|Rows shapes, floor slots) + `Engine::weight_residency` /
  `weight_uses` (default unsupported); runtime `store::weights` cache with `Holder::Weights{pinned}` (floor = live,
  resident above floor = wanted), 4-tick window membership, LRU, shed-to-floor before tier-pressure when on,
  refill only when no kv/state waiter is short; `WeightFirst<P, W>` pool-target wrapper; governor `ticked()` hook.
  Not live yet: needs R2's SlotPool target (wrap in WeightFirst), a planner hook to shed before raising pressure,
  engine support (WC3 kernels, PLE staging) and the compiler's weight shapes (WC1).
- Verified on new-memory: runtime lib 205/205 (1 ignored), engine lib 40/40.
- Running: P4, P5 (CUDA degenerate kernels), P8 (golden baseline), R2 (stores on SlotPool), WC1 (compiler
  streamability), WC3 (CUDA expert table).

### M9 — 2026-11-10: P5 CUDA state segments and P8 golden probe merged; speed rule
- Rule (user): no full test suites; `cargo check` + tests for the changed code only (spec "Verification rules").
- **P5** (2fd2b2c20): every CUDA state kernel (GDN step/chunked incl. sm80/sm100, KDA, Mamba-2, conv1d x8, Wan frame
  cache gather/store, replay_floor, ngram) addresses rows via `StateRows{base, segs, ids, chunk, off, seg_chunks,
  pieces, atoms}`; `Segments::ONE` = today's `base + slot*stride`. Frame cache cut along slab axis-0 rows
  (frame x spatial). Bitwise tests: slot layout vs pieces shuffled across two segments with garbage — identical,
  wrong offset fails. Not GPU-tested here: sm100 tcgen05 walk; engine-cuda suite only built. Still contiguous:
  host-side copy_slot/zero_slot/state_spans and the engine's per-lane slot tables (needs piece-aware ids).
- **P8** (e536895fe): `engine-cuda/tests/engine_cuda/golden_logits.rs` (ignored tests; `golden_fixtures` then
  `golden_logits::golden_logits` under the GPU lock; `PIE_GOLDEN_CACHE` keeps checkpoints + recorded cuBLAS picks so
  separate processes match). 8 miniatures (dense, sliding+global, GDN hybrid, mamba+MoE, compressed pools, dense,
  lightning+full, windowed/full), 12 hashed steps each; base hashes in GOLDEN-LOGITS-BASE.txt. Determinism proven for
  qwen3/qwen35 across processes; other six recorded once. Baseline failures (partial) in BASELINE-FAILURES.md.
- Running: P4 (CUDA kv stride), R2 (stores on SlotPool), WC1 (compiler streamability), WC3 (CUDA expert table).

### M10 — 2026-11-10: R2 merged (runtime stores on SlotPools)
- `sc/r2-stores-on-slots` (ba706c77c..6137440e4): kv and rs stores take device ids from SlotPools built from
  `Capabilities.chunks` (else degenerate facts); each pinned shape is a one-shape `ShapePool` (lowest free first,
  epoch recycle kept), so engines see exactly the old ids (randomized old-vs-new id test, 40 seeds x 600 ops).
  `SeatBook`, `Extent`, `ExtentPool` deleted; `store/grow.rs` (`Grow`) implements the governor's `PoolTarget`;
  `PoolPort::{chunk_stats, grow}`; `SlotMeter` exact byte attrs (now incl. rs buffer pages); fixed-set rule
  corrected in slots.rs (`COW_SETS_PER_LANE = 1`); suspended checkpoints keep their cache hold across the move.
- E2E (separate processes, CUDA): Qwen3-0.6B and Qwen3.5-0.8B — same tokens and all 132 lane lines identical
  (page ids, seats, state slots).
- Left: holders partial (kv cache entries/snapshots not per-holder yet); seat reuse order is lowest-first (was
  most-recently-freed); `Grow` still splits bytes kv/state with 1-byte segments and no evacuation; `RsUnits` and
  old stats views remain as thin views; shared-pool / pieced facts are refused at boot until engines report them.
- Verified on new-memory: `cargo check` runtime + worker; `runtime --lib store::` 63/63.
- Running: P4 (CUDA kv stride), WC1 (compiler streamability), WC3 (CUDA expert table).

### M11 — 2026-11-10: P3fix merged
- `sc/p3fix-fixed-sets` (1cdab5d05): chunk recipe uses `FIXED_SETS_PER_WS = 1` + `COW_SETS_PER_LANE = 1` counted once
  per pool (`cow_sets(&Budgets) = max_lanes`); new `fixed_sets_per_ws()`, `cow_sets()/cow_slots()/cow_chunks()`.
  Server preset: qwen3-next fixed 47.8 MiB/ws (was 95.5), nemotron-super 163.3 (was 326.7); C unchanged.
  Note: the default budget's 256 lanes make the CoW reserve large (and distort the Metal preset numbers);
  revisit whether the reserve should cover only lanes that fork/snapshot in a fire.
- Verified: `pie-poem-compiler --lib chunk`.

### M12 — 2026-11-10: GW merged (governor + weight cache wired)
- `sc/gw-governor-wiring` (683d6662b): each engine's pool target is `WeightFirst<Grow or Fixed>` (governor/steered.rs),
  shared by governor and planner; weight shapes from `ChunkFacts.weights` get their own slot pool for now
  (store::weights::slot_pool; none in degenerate mode); the governor now also runs (tick-only) for dedicated/fixed
  pools; `Signal::TierPressure/Eased` -> `pressure::raise/relieve` on a separate gauge; planner sheds the weight cache
  before raising device pressure; `weight_residency`/`weight_uses` go through the engine's scheduler lane, non-blocking,
  stop asking after "unsupported". Behavior unchanged for every engine today (streaming off, no weight shapes).
- Left: weights live in a separate slot pool until the pool switch (shedding does not yet free kv/state memory);
  nothing fills `ChunkFacts.weights` yet (needs WC1 + engines).
- Verified: `cargo check` runtime + worker; `runtime --lib governor`.
