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

### M13 — 2026-11-10: WC1 merged (weights classified by their readers)
- `sc/wc1-weight-streamability` (519f4e8a0, 3c46867ad, 611b89c5d, f30863656): `poem_ir::residency` classifies every
  weight: expert-streamable (read only by MoeSelect / MoeBiasSum / fused Moe* selects; atom = expert on axis 0),
  row-streamable (read only by token-id gathers Embed / EmbedScaleAdd*; atom = row), else resident; companion planes
  follow their main plane. `ParamPlacement` is now derived when a trace finishes; `Weight::host`, the Starlark
  binding and qwen_4's `.host()` are deleted. `ChunkRecipe.weights: Vec<WeightSlot>` (Expert slot = one expert of one
  MoE layer across the banks its selects read; Rows slot = a token id's rows across tables gathered alike), floor
  (top-k x 2 layers; rows = min(slots, max_tokens x ids/token)), Metal `cut_experts` cuts experts along each bank's
  output axis into >= 1MiB pieces. Placement equivalence: inferred Host set == old hand marking in every deployment
  (CUDA + Metal).
- Decisions: untied input embeddings classify row-streamable and get weight slots (follows the reader rule; the
  spec's "embedding stays resident" is narrowed to tied embeddings / heads). The Host placement still uses a narrow
  rule ("hashed table read outside every layer") to reproduce today's marking; unify later.
- Follow-ups: weight tail waste when C must hold a whole expert (dsv4-pro, minimax-m3 on Metal); `per_rank()` does
  not cover weight slots yet; splits checked only against an earlier rule version.
- Verified: `cargo check` poem, poem-compiler, engine-cuda, engine-metal, runtime; `poem-compiler --lib chunk`.

### M14 — 2026-11-10: MA merged (state() API complete for every blocker; 8 models on handles)
- `sc/ma-state-api` (e32910087..2cad9fe4e): unequal planes `state([tokens, [512, 64]], dt)`; row formats as dtype
  (`kv_rows.v4|mx8|nv4|fp4_keys|fp8`); pools `tokens // r` (1/r width only for r in {2,4} when it divides — replaces
  deepseek `packed()`/COMPACT_RATIOS); inferred: windows from WindowSink + MLA decode/prefill, MLA head width,
  indexer/pool key widths, per-pool spaces for PoolKvAppend rows, unsplit rows for PoolGather/PoolStateWrite and
  single-head planes (replaces split_kv), conv taps cut only when feeding attention/recurrence; `dflash.declare(kv=)`.
- Migrated + equivalence (lowered caches identical except names, tp 1/2/4/8): dots3_note (16), glm_5 (84),
  deepseek_v4 (104) identical; inkling, gemma_4, qwen_3 identical except accepted differences below.
- Decisions: DFlash block rows and inkling local-layer rows are now kept only through their window (only windowed
  attention reads them) — accepted (memory saving, same outputs expected; confirm with the golden probe);
  gemma4-e4b-mini-l1 space renumbering (no empty space 0) — accepted. mimo_v2 `live=`: infer from
  `split_qkv_padded`, no hint — goes to the migration batch.
- Migration guide for the remaining models is in the MA report (record baseline via `moved!(pkg)` +
  `PIE_RECORD_CACHES=1`, replace row names with states, share objects for shared rows, delete `caches()`, order
  fields as the old `caches()`, rename function, run `moved::<pkg>`).
- Verified: `cargo check` poem + poem-compiler; `pie-poem` state tests.

### M15 — 2026-11-10: MG-A merged; work split with the L40S session
- `sc/mg-a` (4fcc1b542..5103abc56): llama_3 (52 traces), glm_4 (112), minimax_m2 (60), qwen3_moe (240) on state
  handles, all identical except names, no exceptions; llama_3's mtp drafter has its own kv state. 12 models migrated.
- Split: the "Pie L40S" session (one L40S GPU) takes the CUDA engine half — BC pool switch (one VA, segments,
  commit-ahead, two-ended activation, evacuation, real ChunkFacts, delete CUDA slot fitting), CUDA weight cache engine
  side (weight_residency/uses, PLE staging, expert staging), golden-probe verification on L40S incl. the M14 window
  change and sm_89-only arms. Branches `sc/l40s-<package>`; New Pie merges.
  New Pie keeps poem/compiler migration (MG-B..D), old caches() API removal, K2 IR kv-space removal, runtime wiring,
  Metal. P4 and WC3 (CUDA kernels) still finishing here.

### M16 — 2026-11-10: P4 CUDA kv page stride merged
- `sc/p4-cuda-kv-stride` (401269dd4, 4e9dabc9b, 96ea7767e): `KvPool.page_stride: [u64; 2]` (bytes per page, key and
  value planes) passed to every pointer-addressed paged-KV kernel (FA2 decode/prefill, decode_gqa, appends incl.
  qkv/envelope/dequant/rope, score, selected, MLA appends+decodes+wgmma+expand, compressed decode/pools/stores,
  index top-k/block max/FP4 store, norm_rope append); per-plane base offset rides in the pointer; engine sets the
  stride via `Shape::page_stride` (also used by copy_kv and host swap/export). Kernels that keep a cell index in
  shared memory pack `page << 16 | offset` (refuse/fall back at >= 65,536 positions per page).
- Bitwise: writers (`attn::strided`, 5 tests) and readers (fa2, selected, score, mla_lse, mla_sparse, compressed,
  index) identical between packed and spread (layer-major structs, shuffled pages, garbage gaps); wrong layer offset
  changes the result. engine-cuda tests not run (speed rule).
- Still dense-only: tcgen05 TMA passes (sm100 prefill, compressed sparse prefill, MLA sparse, index logits) accept
  packed pools only and otherwise fall back to the pointer kernels; legacy scaled schemes (unused); envelope planes
  and DeepSeek compressor state slabs; engine copies treat a plane-page as one stride-sized block (layer-major layout
  will need separate stride and length). sm90/sm100 kernels compile-checked only.
- Verified: `cargo check` kernels-cuda + engine-cuda (with tests) on new-memory.

### M17 — 2026-11-10: MG-B, MG-D merged (23 models on state handles)
- `sc/mg-d` (8 commits): mimo_v2 (48 traces), muse_glimmer (20), muse_spark (44), qwen3_asr (8), qwen3_tts (12),
  gpt_oss (44) on handles; `live` inferred from `split_qkv_padded` (no hint); lib/dflash `caches()` path deleted
  (`declare(kv=)` required). Accepted differences: mimo space order; muse_glimmer/muse_spark/gpt_oss sliding rows now
  window-retained; qwen3_tts codec prenet rows get their 72-token window (latent bug fixed).
- `sc/mg-b` (6 commits): wan_2 (12), ltx_2 (12, space order), hunyuan_image_3 (11), flux_2 (10), z_image (11) on
  handles; VAE frame caches are `state([front*plane, c_in], bf16)` read by both arms; `lib/dit.held_text_encoder`
  bridges lib/qwen3_encoder until minimax_h3 (MG-C) migrates it. Needed fix (applied on new-memory): a deployment
  that declares no state traces with an empty spec whether or not `caches()` exists (crates/poem/src/star/run.rs).
- Merge note: MG-B and MG-D both added a space-order hook; unified as `window_space_first` (mimo_v2, ltx_2).
- Verified: `cargo check` poem-compiler (tests); `moved::ltx_2` and `moved::mimo_v2` after the unification.
- Remaining: MG-C (kimi_k3, glm_5_next, hy_4, minimax_m3, minimax_h3, qwen_4), then delete the old caches() API.

### M18 — 2026-11-10: MG-C merged (28 of 29 models on state handles)
- `sc/mg-c`: minimax_m3 (40 traces), hy_4 (40), kimi_k3 (24), glm_5_next (48), qwen_4 (48) on handles;
  lib/mla `FP8_ROW`/`fp8_rows`/`cache` deleted; lib/kda `kda()` declares its own conv/delta states (`kda_caches`
  deleted). Accepted differences: index keys share one space per dtype (was one space per layer); glm_5_next's
  zero-width rope plane becomes one 512-wide latent plane (MLA reads 0 rope bytes; confirm on GPU).
- Not migrated: minimax_h3 — its rows come from lib/qwen3_encoder (`encoder()` + `qwen3_caches`), shared with
  flux_2/z_image (which use the temporary `lib/dit.held_text_encoder`). Final cleanup package FC handles it.
- Verified: `cargo check` poem-compiler (tests); `moved::glm_5_next`, `moved::qwen_4`.

### M19 — 2026-11-10: all 29 models on state handles; caches() string API deleted
- `sc/fc-drop-caches` (e1f1c7314, 7e4a5ed62, 097f31684, d567e7f9e): minimax_h3 migrated (lib/qwen3_encoder
  `encoder(..., kv)` gives each layer a state; `qwen3_caches` and `lib/dit.held_text_encoder` deleted; flux_2/z_image
  use the encoder's own states). Deleted from the Starlark DSL: `caches()` support, `c.kv_space`/windowed, `c.kv`,
  `c.state`, string row lookups (`inputs.kv("x")` etc.), string fallbacks in op bindings (new `StateName` argument
  type; `bind_ops.py` ROW_NAMED). A package defining `caches()` or a string row is refused with a clear message.
  Kept: Rust `HybridSpec`/`CacheRow` (lowering target) and Rust `Input::kv/state/...`.
- The caches()-to-handles equivalence test and its 29 digest files are deleted (verification scaffolding; K2 changes
  the IR and needs its own check).
- Verified: `cargo check` poem + poem-compiler (tests), clippy, fmt; pie-poem state/package/refusal tests (18);
  moved:: minimax_h3, flux_2, z_image, qwen3_dense, deepseek_v4 and the_flux_2*/the_z_image* before the deletion.
- Next on this side: K2 (IR kv-space removal: CacheRow::Kv.space, Mask/Geometry{space}, write geometry) once the
  engines key by slot shape; Metal pool switch. CUDA side (L40S): BC + RW merge together.

### M20 — 2026-11-10: RW merged (runtime on real shared ChunkFacts)
- `sc/rw-runtime-shared-pool` (10bae1966, 010cce492, aed977a99, ac80264ca, 9e4f75977): `Lane.fixed` per piece with
  `cow == run` legal (in place); slot pool enforces the activation position rule (state below
  `total - act_chunks(rung)`, top act_max span committed from load, `plan_rung/apply_rung` evacuate before a raise);
  logical ids follow the engine's ChunkTable; `store::shared::SharedPool` (list, ring, fixed records with the rs
  buffer inside, weight shapes, any chunk to any shape); stores draw from it; `SharedGrow` governor target sends
  CommitSegments / Evacuate / ReleaseSegments through the engine lane; fire paths fill `Lane.tables`/`Lane.fixed`
  (fork/restore takes a record from the per-lane CoW reserve). Degenerate path unchanged.
- Not live yet: rung raises (the scheduler cannot yet cap a fire's tokens by rung, so fires stay at the largest rung);
  shrink only releases the top of the prefix. Contract amendments 6-12 recorded in CONTRACT-BC.md (snapshot refs per
  piece and `sets_per_working_set = 1` need both sides).
- Verified: `cargo check` runtime, worker, engine and every engine crate on new-memory; RW ran store/planner/governor/
  pipeline::fire::rs (154) and engine chunk (12) tests.

### M21 — 2026-11-10: RW2 merged
- `sc/rw2-buffer-rules` (fb35c00c8, 55a21671a): shared mode refuses a fire leaving more than 16 unfolded rs rows
  (`RsError::Unfolded`; folding is the inferlet's decision, so the runtime refuses rather than folds/splits);
  every lane of a multi-lane group names the record the group writes; `sets_per_working_set() == 1`.
- Contract amendment 12 updated; engine must fold lane 0 only after the group's other lanes read the state.
- Verified: `cargo check` runtime, engine, worker; RW2 ran store::rs / pipeline::fire::rs / store::shared (21).

### M22 — 2026-11-10: P3fix2 merged (equal fixed-record pieces)
- `sc/p3fix2-equal-pieces` (39c0b17ae): a fixed row larger than a slot is cut on its atom axis into the fewest equal
  pieces (whole grains, last padded as waste), each in its own slot at one common offset; rows that fit stay whole and
  share slots. `SlotLayout::Fixed` exposes `FixedRun { row, pieces, atoms_per_piece, offset, slots }` (fills P5
  `StateRows` / P6 `state_map` 1:1); `per_rank` gives `RankRun`. Waste: nemotron3-super 1.48% -> 0.53% server,
  4.04% -> 0.54% Metal; qwen3-next Metal 6.84% -> 1.36%. Regressions on Metal for hybrids whose C < 2 pieces
  (qwen35-397b 12.4% -> 34.1%, qwen35-122b e4m3 11.8% -> 34.2%, glm53-flash-mtp 4.0% -> 16.8%).
- Follow-up: the 1 MiB min-piece rule comes from a CUDA prefill-GDN observation; measure on Metal and move the floor
  into DeviceProfile (smaller on Metal would recover these regressions).
- Verified: `pie-poem-compiler --lib chunk`.

### M23 — 2026-11-10: IT (RW + BC on sm_120) results; contract 14-17
- IT merged sc/l40s-bc 912a75bf7 onto new-memory and ran e2e vs plain new-memory on the RTX PRO 4500: Qwen3-0.6B and
  gemma-4-E2B (list + ring) identical in tokens and top-4 logit bits across single prompt, resumed turns, fork, prefix
  reuse, long prompt + fork, sequential prompts, segment growth, and a 2-3 chunk evacuation mid-decode (3004 steps).
- Found: engine boot over-commit without a kv budget; engine `seat_pages` checks logical chunk ids against committed
  segments (must map through the table); state opened in an unbacked part of the activation's segment (contract 14);
  runtime waiters never grew the shared pool (fixed, cherry-picked as d7ad71dcf); runtime drains/recycles on a kv
  epoch that does not advance per fire (contract 17, to fix).
- L40S finished BC leftovers (pieces from P3fix2 runs, rs buffer rows inside the record at s % 16, snapshots per
  piece) at 9d9bb109e and found the multi-lane group fold race -> contract 15.

### M24 — 2026-11-10: RW3 merged (contract 15-17 runtime side)
- `sc/rw3-groups-epochs` (df6183f1c): `RsSnapshot.pieces` (serde default; `slot` = first piece) filled for
  multi-piece records; a multi-lane group with a folding lane names `[held, cow]` (cow != held) on every lane;
  per-fire epoch: `KvStore::begin_fire()` opens an epoch for every fire at reserve, closed when the engine completes
  it; drains release after the first fire following the evacuation copy retires; freed kv pages were under-fenced
  before (reused while decodes ran) and are now fenced per fire.
- Verified: `cargo check` runtime, worker, engine and every engine crate; RW3 ran store::shared/slots/rs/kv, planner,
  pipeline::fire::rs.

### M25 — 2026-11-10: RWH merged (launch hang on sparse shared pools)
- `sc/rwh-launch-hang` (7898038ac): in shared mode `KvStore::pages_out()` counted uncommitted segments as handed-out
  pages, so a sparse pool (Metal, Growth::Bottom, one segment committed at load) reported kv pressure ~250/255 and the
  gateway held every launch. New `ShapePool::out()` counts only slots actually held for shared list/ring shapes;
  also fixes reported capacity/used tokens, the server handler and policy pressure. Regression test with Metal-like
  facts. (The rs records pool's "out" has the same shape but does not feed gateway pressure — left as is.)
- Verified: `cargo check` runtime + worker; `store::shared`.

### M26 — 2026-11-10: P3fix3 merged; IT round 2 results
- `sc/p3fix3-recipe-gaps` (60eea984e state_f32, f6a56e21b, 2e35d560e): rs rows of a record share slots — whole if
  they fit, else every rs row cut identically on its tokens, piece p in slot p, chosen by waste
  (`ChunkRecipe::rs: RsLayout { pieces, tokens_per_piece, slots, rows }`); other rows may fill the room rs rows leave;
  n-gram id rows get an i32 rs row (Metal layout); KvU4 / block-scaled `rows` formats are sized; Metal `state_f32`.
  Average waste 0.47% -> 0.55% CUDA, 2.96% -> 3.04% Metal; kimik3-2.8t 4.40% CUDA, 5.75% Metal.
- IT round 2 (new-memory 65d849523 + sc/l40s-bc 131eca26d, sm_120, default config): Qwen3-0.6B and gemma-4-E2B
  identical everywhere; grow -> evacuate -> release -> keep decoding identical on Qwen3, gemma, Qwen3.5-0.8B.
  Open: (1) Qwen3.5 fires batching lanes of DIFFERENT working sets diverge by ~1 bf16 ulp from the first shared
  fire (deterministic) — engine, suspected kernel-arm choice by alignment; sent to L40S. (2) beam 3x12 on Qwen3.5
  refused by the 16-row rs limit (degenerate mode serves it) — design decision pending (proposed: overflow rows
  beyond 16 borrow shared-pool slots). (3) no mamba/Nemotron mini importable for e2e.
- BC + RW merge stays on hold until (1) and (2) are resolved.

### M27 — 2026-11-10: sliding-window kv is a trimmed list (contract 19)
- Decision (user): window kv uses the list shape kind (one window list per distinct window), slots freed by refcount
  + per-fire epoch once wholly below `committed_end - window`; trimmed table entries are ref 0. Ring shape, ring caps
  (W + draft + fire rows), wrap CoW, gcd addressing and `kv.window_copies` are removed. Reason: reuses the kv list's
  allocation / sharing / fork / prefix-cache / evacuation; prefill > W and spec rollback need no extra rules; Metal
  needs only trimming. Cost: one alloc/free per T tokens (O(1)), epoch fencing already exists (17).
- Work: compiler recipe (window list shapes), runtime trim, CUDA (BC) ring path -> list, Metal (BM) trim.

### M28 — 2026-11-10: RW4 merged (runtime rs list, contract 20)
- `sc/rw4-rs-list` (133c4f1be): shared-mode rs buffer pages are real rs-list slots (refcounted, CoW tail via new
  `Lane.slot_copies`), 16-row cap removed (`RsError::Unfolded`, `UNFOLDED_ROWS` use, `BUFFER_PAGES_PER_RECORD`
  gone), `SlotShape.role` marks the rs list, base aligned down to T_rs, pressure counts rs-list slots.
  Beam 3x12 on one working set accepted (10 slots). Engines (BC, BM) must implement `slot_copies` + rs-list pages.
- Verified: `cargo check` runtime/engine/worker; runtime `store::shared`, `store::rs`, `pipeline::fire::rs` (27).

### M29 — 2026-11-10: P3fix4 merged (recipe rs list)
- `sc/p3fix4-rs-list` (2a618acab): `SlotLayout::RsList` (T_rs in {1,2,4,8,16} chosen by waste with
  `EXPECTED_UNFOLDED` = 4 as weighting only), fixed records hold state only, 16-row reservation gone,
  `RsLayout::at(row, plane, s, base)`, `Waste::rs_tail`, per_rank cuts rs rows like their state's heads.
  Waste avg CUDA 0.55% -> 0.57%, Metal 3.04% -> 2.67%; kimik3-2.8t CUDA 4.40 -> 2.05. Worse: glm53-flash-mini CUDA
  22.84 -> 30.72 (oversized 25M record per 48M chunk, pre-existing), kimik3 Metal 5.61 -> 12.15 (extra open chunk).
- Engines must set `SlotRole::RsBuffer` on the rs list when building facts from the recipe (BC / BM).
- Verified: cargo check compiler/runtime/engine/worker after merge; chunk lib + golden tests on the branch.

### Note — 2026-11-10: IT divergence localized (sc/it-rw-bc 5483ae4fe, new-memory/it-divergence/ on that branch)
- Qwen3.5 48+48: degenerate submits two 1-lane fires, shared submits one 2-lane fire; every lane's verbs equal
  (ids/tables/fixed differ only). Lane 1 bit-identical, lane 0 ~1 ulp off -> engine batch-composition dependence in
  the hybrid prefill path, not a runtime/BC addressing bug. L40S asked to confirm and name the kernel arm.
- Open: the degenerate runtime never co-batches fresh hybrid working sets (Qwen3 does) — explains why the old build
  never showed this.

### M30 — 2026-11-10: BM merged (Metal on the shared chunk pool)
- `sc/bm-metal-pool` (3e1a9f1df, on 76b080825): window list (contract 19; stopgap: stated as `SlotKind::Ring
  { window, tokens }` until engine::chunk lists carry a window — WL replaces), rs list (18/20, T_rs from recipe),
  `Lane.slot_copies`, snapshots per piece (16), contract 15 group reads, FixedRun pieces.
- Verified on Mac Studio via `pie run text-completion`, greedy, 2 prompts x 48 tokens, tokens equal to 76b080825
  degenerate: Qwen3-0.6B, Qwen3.5-0.8B (hybrid, MTP), gemma-4-e4b-it-4bit (sliding). Engine scattered-pool tests
  (qwen3-micro-text, qwen35-tiny, gemma4-e4b-mini-l1), packed kv migration, store:: lib tests.
- Not exercised on device: trimming past the 512 window, spec decode (rs buffer), multi-piece records in a fire,
  snapshots in a fire, contract 15 group path, slot_copies.
- Open (P3): qwen38-flash-next-mini refused — Metal buffers every conv input incl. cache row 9 that feeds no
  deferred recurrence; recipe keeps no rs row for it.

### M31 — 2026-11-10: WC3 merged (CUDA expert / row tables)
- `sc/wc3-cuda-expert-table` (9e1aa09e3; forks A bf16/FP8, B FP4/affine-u4, C tcgen05 grouped + mega-MoE):
  `ExpertTable { table: u64 addresses table[e*planes+p], hits, located, window }`, `RowTable { table, block_rows }`;
  null table = contiguous bank (degenerate, computed in-kernel); misses may point at mapped host memory
  (located=false, plain-load selects only). Engine passes RESIDENT for now.
- Verified sm_120: moe_table / moe_table_fp8 / moe_table_fp4 bitwise vs contiguous under scattered pools; the timing
  test failed once under a concurrent IT run (247 vs 74 us), passes alone (table 27 us vs contiguous 44 us).
- Next: L40S builds PLE streaming on RowTable (`sc/l40s-wc-ple`).

### M32 — 2026-11-10: WL merged (window kv = trimmed list, ring deleted; contract 21)
- `sc/wl-window-list` (7fde8f1d7): `SlotKind::List { window }`, ring layout/RingCap/RING_FIRE_ROWS/small-window fold
  and `KvDelta::window_copies` deleted; runtime "window set" (one slot of every window list per kv page, refcounted,
  forks + prefix-cache checkpoints share, trims by widest window, epoch-fenced); window models may now checkpoint.
  Metal states the new form; CUDA/XLA/Cerebras read window copies from the pre-launch copy (CUDA ring kernels -> BC).
- Verified: cargo check --tests compiler/engine/runtime(+metal)/worker/cuda/xla/cerebras/metal/vulkan/wgpu; chunk
  lib+golden; engine chunk; runtime store:: + pipeline::fire::rs 92 pass incl. trim boundary, long prefill, fork /
  checkpoint sharing, draft rows.
- Waste regression (T forced to the kv page): models with windows CUDA 0.36 -> 1.19%, Metal 2.43 -> 4.47%;
  dsv41-flash-dspark 1.33 -> 19.3 / 5.9 -> 45.7, dsv4-base 0.48 -> 16.6 / 7.1 -> 22.8. -> WL2: per-window T_w and
  independent trimming.

### Note — 2026-11-10: IT round 3 (sm_120)
- Gemma-4-E2B 2-lane vs 1-lane (degenerate, 48/37, 48/48, 700/37, 1200/600, window 512): top-4 logits move up to
  ~0.5 at step 0, no token changes; lane result independent of partner/order; plans correct. vs fp32 HF reference
  alone and together are equally far (bf16 pipeline error size). Not a bug.
- Shared mode (aaefde791 + sc/l40s-bc 51a1d2e81): beam 3x12 runs; tokens identical to degenerate for beam 2x8/3x12/
  4x16, ckpt_sync, MTP k=0/2/4, resume/fork/long prompt/sequential incl. top-4 bits. Group fire dumps on
  sc/it-rw-bc 7100e78bb `new-memory/it-divergence/group/` (for the L40S group probe).
- Engine bug (pre-existing): engine-cuda serve/prepare.rs `rs_ahead` not cleared on a plain Fold, so a new working set
  reusing an rs record after MTP is refused ("lands its rows at slot 29 ... below 92"). Fix assigned to L40S (BC).
- Seen once: engine stall during MTP ("pending CloseInstance 19, frame k=2 lanes=1 pending_binds=1"), not reproduced.

### M33 — 2026-11-10: WL2 merged (per-window T_w, independent trim; contract 21 final)
- `sc/wl2-window-tw` (491557b6f): window list T_w chosen per chunk size by waste (power of two >= 16), own table in
  its own token order, one pool per window list, trimmed by its own window; window-set unit deleted; planner grants
  per shape; <= 4 window lists; Ring* names -> Window*. Metal addresses window lists with T_w (needs device check).
- Waste mean (ring / WL / WL2): CUDA windowed 0.36 / 1.19 / 0.62, all 0.57 / 0.84 / 0.65; Metal windowed
  2.43 / 4.47 / 3.01, all 2.67 / 3.26 / 2.71. dsv41-flash-dspark 19.3 -> 2.77 CUDA, 45.7 -> 11.9 Metal; dsv4-base
  16.6 -> 1.08 / 22.8 -> 7.77. Remainder = retire-slot accounting + 16-token floor (kept: table length <= ctx/16).
- Limits: host save of rs state handles one window list; old ring-format saves don't reload (clean break);
  Metal loads one window list (gemma4-26b-a4b-dflash refused there, as before).
- Verified: cargo check --tests all crates/backends; chunk lib+golden; engine chunk 13; runtime store/fire/planner
  135 incl. two window lists trimming independently.

### M34 — 2026-11-10: BC merged — CUDA on the real chunk pool (BC + RW complete)
- `sc/l40s-bc` (0237b6d78, 19 commits on M33): two-ended VA with activation at a fixed top, state segments, slot-ref
  lanes, chunk::fit (max_state_slots deleted), rs list from `Buffer.pages` + `slot_copies`, window lists
  (`SlotKind::List { window }`, ring/Kind::Ring/fold_rings deleted), rs_ahead cleared on a Fold lane (pre-existing
  MTP bug; unit test), compressor state rows in their keying row's pages (dsv4_pro expectation updated).
- Verified on L40S sm_89: golden_logits (8), by_refs, buffered, submit_buffered, pair, pair_buffered, pair_by_refs all
  bit-identical to base e536895fe; chunk unit + every_deployment; engine-cuda store:: + serve::prepare. IT round 3
  (sm_120, shared mode): beam/ckpt/MTP/fork/resume tokens identical to degenerate.
- Accepted interim limits: CUDA pages all windowed rows by one table -> one window list of the widest window
  (`one_window` profile; narrower rows keep slots longer); CUDA T_w = kv page (16). Engine-internal gcd grain stride
  stays (not ring addressing). Engine-level group probe skipped (IT covers it).
- Gemma mini jump explained: single-lane prefill(n) vs prefill(n-1)+decode(1) moves the random-weight mini by 4-6.5
  too; real gemma is bf16-noise only.
- Next: IT rerun of the MTP reused-record repro on this build; PLE rebase; K2; tiers; ROCm; other backends.

### M35 — 2026-11-10: PLE row cache merged (CUDA weight streaming, rows)
- `sc/l40s-wc-ple` (eb79b3076): `embed_tabled` + per-block hit counters; engine pins plain-Gather::Embed tables on
  host (mapped), states their row blocks as a weight shape (WEIGHTS=4); `weight_residency` h2d-fills named slots and
  rewrites the RowTable, unnamed blocks read zero-copy; `weight_uses` returns/clears per-block device counts.
- Verified L40S: golden_rows host/cached/mixed = base; golden_logits 8, by_refs, submit_buffered = base;
  golden_rows_resident (streaming off: all blocks seated before the first fire at engine level); decode one-row
  fires resident 8.98 ms / all-host 8.98 / base 9.09 (debug).
- Gaps: first residency published only at the first governor tick (fires before read host); bootstrap fallback
  "stay as loaded" = permanent host reads on CUDA; staging floor ignored (sync h2d); other embed variants resident.

### Note — 2026-11-10: IT round 4 on M34 (sm_120) — pass
- Default config runs SHARED mode (Qwen3.5: 10.3 MB chunks x 3125, segments of 104, 3 shapes; gemma-4-E2B 16.7 MB,
  list + window list).
- MTP rs_ahead repro fixed: 24 runs reusing record 2, no errors, no CloseInstance stall, tokens = degenerate.
- Tokens + top-4 bits identical to degenerate aaefde791: Qwen3.5 beam 2x8/3x12/4x16, ckpt_sync, MTP k=0/2/4,
  resume/fork/1500+fork/sequential; gemma-4-E2B same set + 3 concurrent lanes + 1500-token fork past the 512 window +
  prompt pairs 48/37, 48/48, 700/37, 1200/600 alone and concurrent. Gemma beam refused on both builds (windowed).
- Grow -> evacuate -> release mid-decode identical: Qwen3.5 (40 WS, 3 segments, two releases, fork after shrink),
  gemma (90 WS, list + window list chunks moved).
- golden_logits and golden_logits_by_refs byte-identical to GOLDEN-LOGITS-BASE; golden_submit_buffered = buffered.

### M36 — 2026-11-10: RWW merged (weight residency at boot) + row stats
- `sc/l40s-row-stats` (90af984b5): `WeightUses.held: Vec<WeightHeld { shape, resident, host }>`; CUDA fills it.
- `sc/rww-weight-boot` (c249d150e): bootstrap sends each engine its starting residency on the scheduler lane and
  waits for the answer before governors start / any fire; a streaming engine refusing it fails the load (unsupported
  = fine). `store::weights::boot_cache` replaces "stay as loaded": off + doesn't fit -> load error with bytes needed
  vs free; auto + doesn't fit -> stream at floor; floor doesn't fit -> load error in any mode. off pins every atom.
- Verified: cargo check runtime/worker; store::weights 24, governor::steered 5. Not covered by a test: the
  bootstrap wait through a real scheduler lane (IT to check with WeightUses.held host == 0 after boot).
- Closes the M35 gaps 1-2; open: staging floor ignored (sync h2d), other embed variants resident, experts engine side.

### M37 — 2026-11-10: BM2 merged (Metal window lists verified on device)
- `sc/bm2-metal-window` (cd907604e): page tables sized per window list; runtime 6da9a5d14 fixes "KV/state pool
  starved" on long prompts (dedicated shared pool may grow to its whole range; state waiters reach its walker).
- Mac: trimmed window list reads bitwise as the untrimmed one (gemma4-e4b-mini-l1 700-token prefill in 256 pieces;
  muse-glimmer-30b-mini-l8 T_w 16 beside 32-token kv pages, 2200 prompt past the 2048 window). Qwen3.5-0.8B MTP
  k=0/2/4, beam 3x12/1x12, text-completion equal to pre-BM incl. acceptance counts (rs list + slot_copies on device).
- Open: gemma-4-e4b-it-4bit long-prompt greedy differs from pre-BM (76b080825) in some runs (700-prompt at token 12,
  284-prompt at token 212 before the window); 1500/fork/resume past window equal. Started with BM (M30), not WL2.
  Logit-level check pending. Metal multi-window-list not done (untestable on device now). MTP accepts 0/127 drafts
  on Qwen3.5-0.8B in both builds (separate issue).

### M38 — 2026-11-10: ROCm option B merged
- `sc/l40s-rocm` (2f4e8ed78): kernels-rocm Rust side on the new API (page_stride, Segments, RowTable stub,
  state_stride, conv cache by &RecurrentPool), each refusing what its unmodified .hip kernel cannot address; ROCm
  engine lays the pool packed (Layout::pack), recurrent rows as one piece, no row streaming. Refused at load: mixed
  packed page sizes in one shape (DeepSeek V4, dots, MiniMax shapes — possible ROCm regression until option A),
  non-whole packed pages, multi-piece recurrent rows. new-memory/ROCM-PORT.md lists option A work.
- Verified (no AMD hardware): cargo check/clippy rocm builds; CUDA default goldens unchanged; CUDA with
  `packed-pages` diagnostic bit-identical on 5 mini models. HIP kernels not compiled or run.

### Note — 2026-11-10: BM3 — Metal gemma long-prompt divergence explained (no bug)
- Cause: kv page size. Pre-BM Metal paged gemma at 32 tokens; since BM the recipe pages kv list + window list at 16.
  `kernels-metal/kernels/attn/sdpa_paged.metal` switches from a token walk to a per-page SIMD-group walk at 32 pages,
  i.e. at 497 keys with 16-token pages vs 1009 with 32 — first logit difference always at query position 496.
- Proof: pre-BM with `engine.kv_page_size = 16` is bit-identical to M37 (prefill row + every step's top-5).
  Divergences are bf16 ties (margin 0 / 0.25 at logits ~25). Accepted under the kernel-arm rule.
- Follow-up (perf, not correctness): measure Metal decode/prefill at 16 vs 32-token pages before fixing the
  recipe's Metal page size.

### M39 — 2026-11-10: K2 merged (kv spaces leave the IR)
- `sc/k2-kv-space-removal` (d32db03dd): `CacheRow::Kv.space`, `HybridSpec`, `KvSpace`, `kv_space`,
  `windowed_kv_space`, space numbering and per-space scheduler scaffolding deleted; a row's list comes from its
  `window`; mask/plan geometry read the All axis (engines map it to the kv list, or the window list if only windowed
  rows); write-position inputs carry the row's window (rows with the same window share one input — 380 of 1819 traces
  differ only by that); engines stage one page table per list. `caches()` packages still refused.
- Verified: golden_logits + golden_logits_by_refs (sm_120) identical to GOLDEN-LOGITS-BASE for all 8 models; poem /
  poem-ir / poem-exec / compiler libs; chunk + bake tests; XLA, Cerebras libs; cargo check all engines incl. rocm.
- Open: Metal not run on device (dsv4 plan now reads the kv list's geometry; Metal prefill score capture reads
  last_page_len — check when window T_w < kv page). Vulkan/wgpu compile-only. `CacheRow` rename deferred.

### M40 — 2026-11-10: experts step 1 merged (dense bf16 routed experts in the weight cache)
- `sc/l40s-wc-experts` (8ba74d1a4): dense bf16 banks of plain MoeSelect stated as Expert weight shapes, read
  through ExpertTable; `WeightResidency.held` (runtime sets held = !streaming): a held set with every atom named makes
  bank tables `located`, re-arms recorded bodies on the fast arm, frees the pinned host image (609 MiB on the
  Nemotron mini); later re-seating / non-held after held refused. Fix: row streaming decided on the rewritten trace
  (Nemotron embed stays dense). Diagnostics `dense-weights` / `dense-experts` for the older probes.
  Write-up: new-memory/WEIGHT-CACHE-CUDA.md.
- Verified L40S: golden_experts cached+held bitwise base on 5 MoE minis (same kernels as base); all-host / mixed
  per-route GEMV, max |d| 0.05-0.61, no top-token change; golden_rows, golden_logits 8, by_refs, buffered, pair =
  base; runtime store::weights 24; rocm check.
- Open: quantized banks; engine per-expert tier (ROCm's only streaming today) is a second path for dense CUDA banks;
  streaming off arms the lattice twice at boot; runtime boot with held untested e2e (IT).

### Note — 2026-11-10: main merged into new-memory (21 commits, XLA/TPU v5e/v6e by a teammate)
- One conflict (engine-xla api.rs PoolFacts): new-memory's `shell.pool_facts()` now carries main's seat fields
  (seated_first / seat_to_end / seat_spread). engine-xla lib 7/8: the v6e tuning test fails identically on main.
- Main is actively changing engine-xla / kernels-xla; the XLA chunk-pool port must rebase often.

### M41 — 2026-11-10: weight-slot corruption fix (sc/l40s-weight-chunks ed36cd4d8)
- Bug in M35/M36/M40: before lanes name slot refs the engine seats lane pages itself in the lowest free chunks, the
  same chunks the runtime's allocator gives weight slots -> warm-up / re-arm fires overwrote cached experts / rows
  (nemotron mini cached: wrong logits once slots sat at chunk 1). Probes had hidden it (slots in segment 2).
- Fix: residency chunks are kept clear of engine seating (seated pages there reseated; refused if a live lane holds
  rows on engine-seated pages); a held set arriving after slot-ref lanes no longer re-arms (stays on per-route select,
  logged). Probes now lay weight slots from chunk 1.
- Follow-up (design): engine self-seating and the runtime allocator share one pool; warm-up lanes should not use
  runtime-owned chunks at all (e.g. a reserved scratch chunk) — revisit when degenerate mode is deleted.

### M42 — 2026-11-10: BM4 merged (K2 on Metal)
- Engine tests + e2e (Qwen3, Qwen3.5 incl. MTP, gemma long prompt / window probe) M38 vs M39 equal.
- Fix 93379cd54: a Metal score capture reads its own list's last-page lengths (window list T_w), not the plan's kv
  list's (latent since before K2; no catalog model captures over a windowed row). Kernel test 3b1f2fae0.
- OPEN BUG (pre-K2, Metal): dsv4-flash-mini's window list read BY SLOT REF gives wrong logits from the first fire
  (compressed side planes, Row::Compressed via pool_gather / pool_state_write); seat-addressed is correct. Test
  ignored at eddd0d2c7. Shared mode uses slot refs -> must fix before more Metal features (BM5a).
- OPEN: Metal gemma run-to-run nondeterminism already at M38 (1 of 3 runs differs). Source unknown.
- Metal MTP on Qwen3.5-0.8B accepts 0/127 drafts in every build (pre-existing).

### M43 — 2026-11-10: ROCm option A, host/disk tiers (contract 22), Vulkan/wgpu kernels step 1, Cerebras host pool
- `sc/amd-rocm-port` (a1dcf5886, MI300X): flat range fixes; recurrent rows via segments + pieces; kv page stride +
  compressor stride across all paged HIP kernels; RowTable embed; ExpertTable in every HIP routed select;
  concat_cols ported; FP8 tile sched_barrier fix (m33 off BASELINE-FAILURES); PACKED/pack paths deleted.
  Goldens: 6/8 = ref incl. dots + minimax (were refused); dsv4 + mimo differ ONLY with graphs on, and the ref
  (e536895fe) is wrong too -> pre-existing ROCm graph-replay bug with >= 2 MoE layers (AMD item d). golden_rows dsv4
  host/cached/mixed equal. CUDA golden_logits + by_refs on sm_120 still = GOLDEN-LOGITS-BASE after the merge.
- `sc/tier-shared` (b52b19ca9, 1a7ce3a12): host tier = second chunk pool, disk = kv list slots in the slot file;
  every shared shape parks/restores through copy_slots, holders kept; kv_swap_capable on shared pools (CUDA kv
  pressure now suspends to host). CUDA legacy host rs/window shelves and copy_state_host etc. deleted.
  Verified: runtime tiers 6/6 + store/planner/scheduler; CUDA golden_offload (dense, window, GDN) bitwise vs never
  offloaded, with a negative control. Open: Metal device check; e2e suspend/resume; tier copies on the compute stream
  (legacy overlapped on a swap stream — perf follow-up); named snapshots / taps of host refs; MTP + park e2e.
- `sc/vk-chunk-pool` step 1 (e45549767, ad0e2a3fc): Slang kernels address kv/state as base + slot*stride + offset in
  up to 4 segment bindings (WGSL has no buffer arrays); new model_logits tests; wgpu bitwise, Vulkan hybrid/window
  bitwise, dense 1 bf16 ulp (driver codegen, address math proven exact). Engines still degenerate; pool switch next.
- `sc/cb-chunk-pool` (191684a88): Cerebras state in a host chunk pool (segments really allocate/free/evacuate),
  fires stage named slots into per-fire planes and copy back; one window list (widest) interim; weights always host
  -> no weight shapes. 124 lib tests incl. host-simulation word-for-word checks. Unverified on CS hardware.
  Behavior: recurrent models' lanes per fire <= state slots. Wafer `Spatial` engine still degenerate.

### M44 — 2026-11-10: Cerebras tiers (contract 22)
- `sc/cb-chunk-pool` (97a8139ce, 1a0f80b7a): host chunks (seated only with host_kv_budget; pool is host memory) and
  a kv slot file on disk (persistent with meta/layout check); copy_slots in every direction. 127 lib tests.
- Wafer `Spatial` engine keeps degenerate facts: kv pages are fixed wafer locations, recurrent state is replayed (not
  addressable), and the unmerged `cerebras-model-support` branch owns those files. Follow-up for its owner after
  that branch merges: a kv-only shared pool for batched paged builds. Details in BACKENDS.md.

### M45 — 2026-11-10: ROCm graph-replay bug fixed (pre-existing, not memory)
- `db186bbf`: HIP `swiglu_clamp_split` hardcoded fan = 1 for split gate/up routed planes (MiMo, DeepSeek V4, dots), so
  under a seat (every captured body) only the first `live` routes landed and the down select read stale rows. Now
  takes plane_fan like CUDA's twin. CUDA unaffected. Test linear_glu_flat fails with fan 1, passes now.
- ROCm baselines recorded in new-memory/GOLDEN-LOGITS-ROCM.txt (dsv4/dots3 = eager; mimo shaped, split-kv arm).
- Note: this ROCm bug also exists on main (served wrong logits under graphs for split-plane MoE before the redesign).

### M46 — 2026-11-10: BM5 merged (Metal compressor state in pool; ANE determinism)
- BM5a: the dsv4 "slot-ref bug" was the test's stride (7 = slots per chunk); fixed b4391da5f, test un-ignored. Real
  engine bugs found on the way (418b34efa): Metal compressor state lived in side buffers indexed by kv-list page while
  pool gathers address by window-list ids; window slot copies / spills / evacuations didn't carry it; copy_kv copied
  window rows at kv-list offsets. Now compressor state lives in its row's slots (as the recipe and CUDA do).
  Trimmed-window tests bitwise on Mac: gemma mini, muse (T_w 16 / T 32), dsv4-flash-mini.
- BM5b: Metal gemma run-to-run nondeterminism = ANE compile refusals (ANECompilerService 4097) and serving before
  compiles finished. Fix d6132f473 / 87d2f7bfa: ANE compile retried up to 8x with backoff; fires that can use the ANE
  wait for their program. 4 forced-cold runs = warm output. Open: all 8 refused -> GPU keeps the work (logged).

### Note — 2026-11-10: XLA chunk pool ready on sc/xla-chunk-pool (NOT merged — needs TPU)
- One fixed u32 pool `[chunks*C/512, 128]` with per-plane bitcast views and affine slot maps; slot-ref lanes, rs list,
  slot_copies, one window list (T_w = page), fixed shapes per block size; tiers (contract 22). Degenerate pools deleted.
- Bitwise vs base on the CUDA PJRT plugin (Mosaic kernels off): qwen3-micro, qwen35-tiny, gemma mini past the window;
  park/restore bitwise.
- Held because: the Mosaic kv append is disabled for pool planes (plain HLO writes) and the bitcast views may cost a
  TPU relayout — TPU decode/prefill speed unmeasured, while main is actively tuning TPU v5e/v6e. Needs a TPU owner to
  run the bitwise gate + kernel tests listed in the agent report and measure speed. Weight streaming: design only.

### M47 — 2026-11-10: ROCm dense experts on the weight cache
- d3769d3e: streamed_experts no longer returns early on ROCm; HIP dense arms already read ExpertTable. golden_experts
  (MI300X): cached+held = baseline on nemotron/minimax/mimo/dots3/dsv4; host/mixed = per-route stream arm (max diff
  0.03-0.34; dsv4 flips a near tie at step 9). Quantized ROCm selects still RESIDENT (follow CUDA's next step).
- GOLDEN-LOGITS-ROCM.txt rewritten in probe format (usable as PIE_GOLDEN_CHECK); notes moved to .md.

### Note — 2026-11-10: ROCm tiers verified (MI300X)
- golden_offload 8/8 models (incl. mamba, dsv4, dots3, minimax, mimo), 12/12 steps bitwise vs never-offloaded,
  host + disk chunks exercised. No ROCm-only legacy tier code left (sc/tier-shared deleted the shared verbs).

### M48 — 2026-11-10: Vulkan + wgpu on the shared chunk pool
- `sc/vk-chunk-pool` (c390f42a8, 601a9c51f, c64788741): one pool in up to 4 buffer-segment bindings (<= binding
  limit, pool <= 8 GiB for 32-bit bf16 indexing), ~4 MiB chunks, bottom growth, evacuation queued before the next fire;
  slot-ref lanes, rs list pages, slot_copies, contract 15; copy_slots device/host/disk (wgpu host tier = plain host
  memory via queue); per-row slabs, pool resize, separate rs buffer and parked tiers deleted. Records in pieces (state
  kernels look pieces up per lane). Compressor side planes follow evacuation (untested: dsv4 mini never ran on Vulkan).
- Verified (RTX PRO 4500, NVIDIA Vulkan; wgpu via Vulkan): default placement = base 88a621693 (wgpu all 3, Vulkan
  hybrid + windowed; dense 1 ulp accepted); scattered + evacuate + release mid-decode bitwise; 4-piece records bitwise.
- Open: window lists not trimmed (kept whole in the kv list), snapshots, on-device test of tiers / group CoW /
  slot_copies, weight streaming. Build needs PIE_SLANGC=/root/slang/2026.17/bin/slangc on this machine.

### M49 — 2026-11-10: IT round 5 (sm_120) + auto-mode fix
- Fix `sc/it-weights-auto` 1d98747c0: engines with weight shapes report weights_resident=false (weights start on
  host), so auto always resolved to streaming; now auto starts held and boot_cache decides fit.
- Real MoE (Qwen3-30B-A3B cut to the mini shape, real weights; the full 57 GiB model is OOM-killed by the 62 GB
  container because the engine pins a full host copy of the experts at load): off = held before the first fire,
  576 MiB host freed, host=0, tokens + top-4 bits = base, 481 vs 493 tok/s. on (512 MiB): runs, residency changes
  per tick, 208-219 tok/s, sheds weights before tier-pressure.
- Qwen3.5 record in 18 pieces (chunk-bytes): beam/MTP/fork/resume/grow-evacuate-release identical to default chunks.
- Host tier e2e (M43): sequential sessions parked/restored 49/97, all identical (Qwen3.5, gemma).
- NEW PROBLEMS: (1) first boot with experts on host took >13 min arming at max_forward_tokens 5120 (every arming
  fire reads experts per route from host); (2) engine pins a full host copy of routed experts at load -> large MoE
  cannot load in limited RAM; (3) MTP k=2 under kv pressure hangs ("nothing is left to discard", short never met);
  (4) concurrent decodes at a small budget stall instead of rotating sessions through host; (5) gemma (no recurrent
  state) refused with NoRsSlots "state pool starved ... 233 free"; (6) prefix-cache pages pushed off device are
  dropped instead of demoted to host/disk (disk journal stays empty).

### M50 — 2026-11-10: Metal weight streaming, multi-window, pieces + tiers on device, ANE policy
- `sc/bmw-metal-weights` (358c26a94..8bea0076c), all on the Mac Studio (M1 Max):
  - PLE rows through a row table + hit counters (unified memory: misses read the table in place; held frees the
    per-table buffers); bitwise in 5 modes on gemma mini l1/l6 and dsv4 mini.
  - Dense bf16 routed experts through an expert table (pieces per expert as atoms); bitwise in 5 modes on minimax,
    qwen3-30b-a3b and dsv4 minis.
  - Up to 4 window lists, each own T_w / table / trim; bitwise on a two-window muse mini past both windows.
  - Records in 4 pieces in real fires (seat, slot ref, evacuated) bitwise; host/disk offload probe bitwise (3 models).
  - ANE: a load waits (<= 20 min) for its ANE programs and fails if one is refused 8 times (deterministic split).
- Left: quantized experts / quantized embedding tables not streamed (Metal, like CUDA); warm mapped loads have no
  separate buffer to free; runtime boot with real streamed weights untested on Metal.

### M51 — 2026-11-10: CUDA window lists each by their own table
- `sc/l40s-window-tables` (e5d2f9f51): up to 4 window lists, narrowest first (ids 1, then 65536 + index), each seated
  and trimmed apart; `one-window` diagnostic / `.merged()` keep the old single list. T_w = kv page (16).
- Verified L40S: golden_logits 8, by_refs 8 (each table trimmed by its own window), pair_by_refs, submit_buffered,
  golden_offload all = base. Two-window models untested (only quantized mini / 26B real model) -> needs fixture
  writer for quantized planes (comes with quantized experts).
- Also: 4 MoE minis with weight slots at chunk 1 bitwise base (M41 follow-up). CUDA graphs vs eager (golden_eager):
  not bitwise but arm-level on all minis (max 0.02-0.66), no sign of ROCm's replay bug.

### M52 — 2026-11-10: runtime pressure / tier fixes (IT round 5 problems 3-6)
- `sc/rt-pressure` (ca1079b53, fa8f91217, 5650d93b3):
  - Fire-epoch leak: retire_on spawned a second waiter on a completion that holds one waker per slot -> epochs never
    retired (MTP: ~50 s, 4533 freed pages stuck). Now `WorkItemCompletion::on_settled` hook; per-set kv txn guard.
  - Shared pool: a state (rs-list / window / record) shortfall now counts as kv pages to park (whole chunks free for
    any shape); fixes MTP-under-pressure hang and concurrent-decode stall (gemma 192 MiB, 48 concurrent: completes).
  - Starvation error names the state it lacked (was kv numbers under a NoRsSlots label).
  - Idle prefix cache demotes host -> disk (journaled) under a waiter's want before anything is dropped; restart
    takes back 750 pages, hits bitwise. Limit: pages already on host are not pushed to disk when host fills.
- Verified: runtime store/planner/completion/fire 154 pass; e2e repros on sm_120 all identical to unconstrained.

### M53 — 2026-11-10: Vulkan/wgpu window lists, on-device tiers / CoW / slot_copies, PLE rows
- `sc/vk2` (2679c5848, 67de88455, 0341ec670, 1f35b185d): up to 4 window lists with own T_w (compiler: a stated kv page
  fixes T_w only for single-table backends); trimmed = untrimmed bitwise (gemma l1/l6, two-window variant). Tiers
  (contract 22) offload probe bitwise on 3 models; contract 15 group CoW and slot_copies (rs + window tail forks)
  bitwise with negative controls. PLE rows streamed (misses: flush + read token ids + host-visible stage —
  synchronous on fires with misses; all-resident fast path, Vulkan command-buffer recording kept).
- Snapshots not served (rs_snapshots 0) -> parity gap. Experts not started. Runtime e2e not run.

### M54 — 2026-11-10: held weights keep no host copy, arm once (CUDA)
- `sc/l40s-held-weights` (213b3fcd0): `Residency.weights_held` (worker: held unless weight_streaming = "on"); a held
  load leaves streamed tables / routed banks in the serving file mapping (not device, not pinned), fires nothing until
  the first residency, which fills every slot from the file and arms once. Decoded-at-load planes fall back to a
  pinned copy (logged). Streaming loads refuse up front when the pinned miss copy exceeds the machine's free memory.
- Boot (L40S): nemotron mini 5120 tokens 282.6 s -> 6.3 s; dsv4 mini 256 tokens 102.7 s -> 15.2 s; logits = base;
  pinned host bytes 609 MiB / 4850 MiB -> 0.
- Open: auto + pool can't seat -> runtime streams but engine was told held (refuses non-held) -> must build the miss
  copy lazily (assigned). Peak RSS on a real-size model, HMM branch, refusal message, real imported files: untested.

### Note — 2026-11-10: IT round 6 on M54 (sm_120)
- Full Qwen3-30B-A3B bf16: off -> clean boot_cache error (58.1 GB needed, 25.6 GB free), no OOM kill; auto -> runtime
  streams, engine refuses (held mismatch, L40S fixing). Peak RSS ~55-56 GiB (likely file-backed mapping pages; at the
  container cap but not killed). Experts left in the file mapping (no pinned fallback lines).
- qwen3-30b-a3b-mini at max_forward_tokens 5120: boot ~2-3 s (round 5: >13 min), tokens + top-4 bits = base,
  450 vs 433 tok/s. Open: RSS +1.3 GB vs base; log "bodies armed 0 of 0 compositions" to be explained.

### M55 — 2026-11-10: host-tier cache spills to disk (RT2)
- `sc/rt2-host-to-disk` (e2841c770): when a waiter wants device room and the host lacks room above its 1/8 reserve,
  the host's lowest-key idle cache moves to disk first (HostPinned -> LocalDisk via copy_slots, kv list only),
  journaled, whole groups on a persistent tier; suspends short of host room spill to disk too. TierMove gains
  `source`. Host `live` excludes parked idle cache (spillable vs real state visible to policy).
- Verified: runtime store/planner/policy::pressure 151 pass; e2e Qwen3 (640 MiB device, 512/256 MiB host, persistent
  disk): restart takes back 750 / 875 pages, P now hits after restart (M52: cold); suspend regression identical.

### M56 — 2026-11-10: held as a hint; load seats weights and arms (fixes "bodies armed 0 of 0")
- `sc/l40s-held-hint` (1ae41d5ca): M54's deferred arming made the pool fit take the whole card (no room left for
  bodies -> real models decoded eager). Now a held load seats every atom in pool slots at load (from the serving file,
  chunks from 1, shape by shape) and arms at load; `WeightShape.seated: Option<u32>` states where (atom a -> slot
  first + a). The first residency re-points (refill from file, logged) if it names other slots; a non-held / partial
  first residency pins a miss copy from the file (free-memory refusal) and re-arms. MADV_DONTNEED after file copies.
- L40S: nemotron mini told held + adopted seating: boot 5.2 s / 5.4 s (256 / 5120 tokens) vs 19.6 / 266 s, logits =
  base; off-by-one-chunk and auto-undersized cases = untold results bitwise.
- Next: runtime adopts `seated` at boot (RWS). Known: engine-wgpu without features fails to build (VK2 rows.rs
  `write_shared` on the stub buffer) — VK3 to fix.

### M57 — 2026-11-10: runtime adopts the engine's weight seating (RWS)
- `sc/rws-adopt-seated` (8a7bddb09): `SlotPool::claim` takes exact slot refs (fails cleanly on any conflict);
  WeightCache claims every seated shape before staging / other shapes; held residency names first + a; auto-undersized
  keeps whole seated chunks as cache.
- sm_120, qwen3-30b-a3b-mini at 5120 tokens: off and auto both show no re-seat warning and "bodies armed 131 of 131"
  (the M54 "0 of 0" bug is gone); tokens + top-4 bits = base; 405 / 399 tok/s (shared GPU; earlier rounds 450-481,
  so re-measure on a quiet GPU).

### M58 — 2026-11-10: CUDA plan seats no expert itself (plan A)
- `sc/l40s-cuda-plan-a` (e1e7c9a83): dense routed banks the weight cache takes are off every plan tier; uncached dense
  banks stay whole on device; device_weight_budget now only decides which DENSE planes spill to host. On CUDA the
  engine's per-expert slab / vote / promote are unreachable (kept only for ROCm quantized banks until their cache).
- BEHAVIOR CHANGE: a CUDA MoE that used device_weight_budget to keep some experts on device no longer does; experts
  live in the pool (held, or streamed by the runtime's cache).
- Verified: routed_experts_stream plan tests, golden_submit_buffered, golden_offload = base. Untested: a real MoE
  boot under a device_weight_budget below the dense planes (IT); budget refusal text still says "REGISTERED planes".

### M59 — 2026-11-10: Vulkan/wgpu experts, snapshots, PLE staging (VK3)
- `sc/vk3` (1883bcbe9, ad470f203, 018adf1c8, 2eb5abef8): every routed bank (bf16, affine 64/8, 64/4, {32,64,128}/2,
  MXFP4, +/- bias) is an expert weight shape read through tables on every kernel path; misses staged per layer from the
  mmap (no second host copy); experts no longer cut into pieces (chunk >= one expert); the engines' own expert tier
  DELETED. bf16 MoE minis (qwen3-30b-a3b, glm45-air, qwen36-35b-a3b): held = base bitwise, none/half resident = held
  bitwise. Quantized: kernel-level bitwise only (fixture writer is bf16-only).
- Snapshots per piece (contract 16): rs_snapshots = 4 for conv / GDN / KDA scans; bitwise vs a cut pass (1 and 4 pieces,
  beside a buffering lane). KDA end to end untested (no KDA mini).
- PLE misses staged in frame order without a flush (Vulkan 199.6 -> 209.6 tok/s all-resident; half 174 -> 185);
  remaining gap is fires with misses not being recorded (proposed: ring of stages selected by an input).
- engine-wgpu / engine-vulkan build without features again.

### Note — 2026-11-10: IT round 7 on M58 (sm_120, quiet GPU)
- Decode: no regression (base 451/461, M54 off 484/476, M58 off 476/453, auto 471/484 tok/s); tokens + top-4 = base.
- BUG (engine, M58): device_weight_budget below the whole model is refused at every size; the routed expert planes
  (576 MiB) are still charged to the device weight tier, the refusal says "0 routed banks", and the stated demand grows
  with the budget (608 MB at 16 MiB ... 2.0 GB at 1400 MiB). Serves only at >= 2 GiB. Assigned to L40S.
- RSS: M58 2,552 MiB vs M54 2,779 vs base 1,810 (expert mapping pages released; ~0.75 GB above base, likely dense
  planes read in place from the mapping — to break down).

### M60 — 2026-11-10: device_weight_budget fixed for cached experts
- `sc/l40s-budget-fix` (ca8acc446): banks the weight cache takes are on no tier from the cut on (`Plan::pooled`); demand
  excludes them; refusal text names what cannot leave the device. Mapping pages of dense planes copied out of the
  serving file released after the load's stream syncs.
- golden_budget (engine API, minimax-m3-mini, 2.7 GB experts): no budget = base bitwise; 512 MiB and 128 MiB load
  and serve (dense planes on the pinned host tier, experts in the pool), max 0.125 vs unbudgeted, no top-token change.
- Known (pre-existing): a budgeted load arms a dense rotor and walks every fire eagerly (no recorded bodies) -> perf
  follow-up. RSS re-check pending (IT).

### M61 — 2026-11-10: CUDA tier copies on the swap stream
- `sc/l40s-tier-swap` (28aa8371b): device<->host copy_slots run on the swap stream behind compute with an event; fires
  naming a moving kv slot wait for it; window / record / rs-list moves are "wide" (every fire waits); slot-touching
  ops without names wait for all moves. `tier-inline` diagnostic restores the old path. golden_offload with a 0.5 GB
  bulk move beside a step: all steps same.
- Gain small (step beside the move 26.8 -> 22.5 ms dsv4 mini). Real cost found: copy_slots blocks 265-430 ms pinning
  host-tier chunks on first use -> pre-pin the host tier (assigned).
- Rotor (budgeted loads walk eagerly) scoped: recordable in principle (static per-fire copy plan; copy stream joins the
  capture via events); medium, risk in aligning body cuts with rotor regions. Scheduled after quantized banks.

### M62 — 2026-11-10: host tier pre-pinned in the background (CUDA)
- `sc/l40s-tier-prepin` (d747f46a0): `Shelf::pin_ahead` pins every host-tier chunk from seat_tier on (thread
  "pie-host-tier-pin", lowest first); copies wait only for chunks not pinned yet; on failure fall back to inline.
  copy_slots of 0.5 GB device->host: 265-430 ms -> 6.3-6.6 ms; golden_offload all same.
- Behaviour change: the whole host_kv_budget is page-locked shortly after boot. Open: a budget larger than lockable
  memory surfaces at the first copy that needs it (to be reported at boot instead).

### M63 — 2026-11-10: XLA chunk pool merged (TPU verification by the user's team)
- `sc/xla-chunk-pool` (b067ca70e..c1794d466, rebased on main #1594-#1602 without conflicts): per-plane PageMap; one
  u32 pool with bitcast views; Mosaic kv append only on unmapped planes (pool kv writes via HLO); slot-ref lanes,
  rs list, slot_copies, one window list (T_w = page), tiers (contract 22).
- CUDA PJRT plugin, Mosaic off: bitwise vs base 0596e5176 (qwen3-micro, qwen35-tiny, gemma past window, park/restore);
  kernels_mla/index/pool/replay/ple all pass; kernels_attn 21/24 and kernels_ssm 10/22 (failures reported as matching
  base on the plugin; ssm not re-run on base this round).
- TPU: new-memory/XLA-TPU-CHECKLIST.md (bitwise gate commands, TPU-only kernel tests, v5e/v6e speed vs main incl.
  Mosaic kv append and bitcast relayout cost). Weight streaming on XLA: design only.

### M64 — 2026-11-10: quantized fixture writer (all backends)
- `sc/l40s-wc-quant` (c29148c53): `checkpoint::testkit::artifact::import(source, contract, stamp, tiled, out)` runs
  what `pie model import` does into a serving artifact; feed packed weights as bf16 at the unpacked shape (scales made
  by the encode), plain E4m3 / E8m0 / U8 planes as served bytes. Tested: dsv41-flash-mini-dspark (Mxfp4 banks, block
  FP8, NVFP4 kv) imports and loads on CUDA. Untested: affine U4/U2, tiled banks, other platforms.
- Fix: under `one-window`, rows go to the list the recipe lays them in; write tables found by the declared window.
  golden_logits_by_refs = base in both layouts on 4 single-window minis.
- Two-window mini loads (2 window lists) but cannot fire on L40S (sm_89 kernel gap for its fp8 compressed rows) ->
  bitwise two-lists vs one-window comparison to run on sm_120.

### M65 — 2026-11-10: host tier bounded at seat (CUDA)
- `sc/l40s-tier-bounds` (7b544a593): seat_tier cuts host_kv_budget to the machine's available memory (MemAvailable +
  cgroup limit; RLIMIT_MEMLOCK does not govern cudaHostAlloc) with a warning, so the stated host_chunks can be pinned;
  the pin thread warns on a failed chunk (locked vs asked). golden_offload unchanged; cut / warning paths untested.

### Note — 2026-11-10: IT round 8 (sm_120, M64) — REGRESSION from M62
- Default config RSS 27,178 MiB at ready (base 1,810; M58 2,551): the default host tier equals the device pool
  (1,442 chunks x 18.9 MB) and M62 pins ALL of it at boot. Would OOM the full 30B in the 59 GB container. With
  host_kv_budget = 1GiB: 2,228 MiB. Assigned to L40S: pin a bounded lookahead, not the whole budget.
- dsv41 dspark two-window mini: refused on sm_120 too (fused.lse_sink: fp8 compressed rows only read by
  attn::compressed::decode::decode, not this shape / gathered seats) -> kernel gap, two-window bitwise still unverified.
- device_weight_budget 512 / 128 MiB on the MoE mini: serves, tokens = base, but decode ~38 tok/s (~10x slower;
  eager rotor + host plane reads) -> recordable rotor task.

### M66 — 2026-11-10: wait slots wake every waiter (serving hang fix, all backends)
- `sc/amd-waker` a7c301013 (AMD) + `sc/hang-fix` 0fd90731b: a WakerTable slot held ONE waker and `register` replaced
  it, so of several lanes awaiting the same front fire only the last woke; the others slept on a settled completion
  and the scheduler quorum waited forever (MI300X: 20-30% of batch-8 runs hung; CUDA hid it by settling first).
  Now a slot keeps every waiter (deduped per task); a future that takes back a shared registration wakes its task so
  sibling futures re-poll; teardown sweeps. M52's on_settled hook kept (no longer needed for correctness).
- Tests: waker unit + 2 loom tests (every interleaving), runtime engine/pipeline/scheduler/inferlet; deterministic
  `every_lane_waiting_on_the_front_fire_wakes_when_it_settles` fails before ([false,false,false,true]), passes after.
  MI300X 0/40 hangs (was 20-30%); CUDA 20/20 clean.

### M67 — 2026-11-10: host tier locked lazily again; MXFP4 expert banks in the weight cache (CUDA)
- `sc/l40s-tier-ahead` (4c57bf934): fixes the M62 regression (27 GB RSS at boot). The pin thread locks 512 MiB ahead
  of use in 32 MiB pieces, paced (locking stalls every device-runtime call); a copy past the lookahead locks its own
  pieces. golden_offload: RSS +530 MiB after seating with a 27 GB budget (was all of it); 0.5 GB offload enqueue
  23.7-26.6 ms (M62 whole-pinned 30-32 ms). Host chunks unlock only at drop -> RSS tracks the host high-water mark.
  RSS at ready on a real server: IT to confirm.
- `sc/l40s-wc-quant-engine` (40568ae3c): MXFP4 W4A16 routed banks (plain quantized MoeSelect, no bias / act quant)
  stated as Expert weight shapes on CUDA off tcgen05 (ROCm, sm_100: none); one table per bank (codes, scales);
  use counts via note_routes while the set may change; held MXFP4 sets report zero uses.
  golden_experts dsv4-flash-mini-bf16-mxfp4: cached/held/moved bitwise the dense serve; host/mixed max 0.51, no top
  token change. Untested: sm_90 wgmma arms, real MXFP4 models (gpt-oss), budgets, speed. NVFP4 / affine / FP8 /
  W4A8 / biased selects remain.
- Merge fix: `Run::bank_table` built only with `cuda`; ungated (featureless build).

### M68 — 2026-11-10: Vulkan/wgpu quantized MoE e2e, held weights, recordable PLE misses (VK4)
- `sc/vk4` (bf6130a45, 75259a509, 4d4c3cbfb): quantized MoE minis serve end to end through the weight cache on both
  engines via the M64 fixture writer: qwen3-30b-a3b (u4 g64, u2 g32/64/128, MXFP4), gptoss-20b (MXFP4 + bias),
  glm45-air (u4), qwen36-35b-a3b (u4, Vulkan) — cached+held = base (old expert tier) bitwise; none/half = held.
  Writer fix: affine scale layout used group 64 regardless of the stated group (u2 g32 overran, g128 doubled).
- Held weights: load seats every atom from the serving mapping, states `seated`, records once; mapping pages released
  (0 MiB resident vs 756 MiB without). Runtime e2e adoption not run on Vulkan.
- Vulkan PLE fires with misses record (ring of stages per arm); throughput too noisy on the shared GPU to show gain.
- Merge fix: the deviceless Frame stub lost its `impl Frame {` line (featureless build).

### Note — 2026-11-10: IT round 9 on M67 (sm_120)
- RSS at ready (qwen3-30b-a3b-mini, off, default): 1,744 MiB (base 1,810; M64 27,178) -> M62 regression fixed.
  Tokens + top-4 = base; 491 tok/s. Offload-heavy (1 GiB device kv): all 98 entries identical, RSS grows by what is
  parked. gemma-4-E2B / Qwen3.5 sanity: tokens = base, 10x batch-8 rounds with no stalls.
- BUG (engine): off + device_kv_budget 256 MiB -> CUDA illegal address (700) at boot instead of the clean "weights do
  not fit" refusal (604 MB of held experts vs a 252 MiB pool). Assigned to L40S.
- BUG (runtime): 768 MiB device kv -> "pool starved with no reclaim path ... NoEligibleVictim" without parking idle
  sessions first (1 GiB parks and passes). Assigned to RT3.
- To look at: gemma decode 123 vs 136 tok/s (measured once); gemma boot peak 7.2 GB on both builds.

### M69 — 2026-11-10: planner counts only the room weights and activation leave (RT3)
- `sc/rt3-starved`: on a shared pool the planner's kv total counted chunks the held weights pin, so an ask that could
  never fit waited and failed as NoEligibleVictim (IT r9 at 768 MiB: 67 chunks = 40 weights + 25 activation + 1 null,
  1 left = 144 pages; one 3,000-token session needs 188). `SharedPool::room()` caps totals (hog check, starvation,
  admission, diagnostics); new `PlannerError::PastPool` names the split and the shortfall.
- e2e: 768 MiB refuses s0 with the clear message; 1 GiB identical to unconstrained.
- Follow-up: the weight cache's 8-chunk staging floor stays pinned under held (nothing streams) -> release it.

### M70 — 2026-11-10: held weights pin no staging floor
- `sc/rt3-staging` (d63f730c9): under held (off / auto that fits) the weight cache takes no staging slots and sends
  an empty `WeightResidency.staging` (no engine needed changes). Gives back 8 chunks (144 MiB) on the MoE mini:
  768 MiB now passes the 48 x 3,000-token script (parks 69 to host, 54 to disk), identical to unconstrained.

### M71 — 2026-11-10: XLA pool planes are lazy row views (TPU session)
- `sc/xla-tpu` (be7276b4a, 6c4de8a67): M63's whole-pool bitcast views were padded ~64x on TPU (192 GiB of temporaries
  at +3 GiB). Now a pool view emits nothing; gathers / slices read single 512-byte pool rows and unpack only those;
  scatters pack into rows (mask + or, last write wins, read-splice-write for unaligned blocks); while/case carry the
  pool; anything else materializes and is logged (PIE_XLA_VIEW_STRICT=1 refuses). "Packed rows" element order matches
  a Mosaic bitcast of a u32 block; GDN step/chunk Mosaic kernels bitcast their bank ref in VMEM.
- v6e, Mosaic on, strict: qwen3-micro, qwen35-tiny, gemma mini bitwise vs base at default, +3, +20/+24 GiB pools;
  fire HBM at a 24 GiB pool = pool in place + < 300 MB. engine-xla 14 + kernels-xla 18 lib tests pass.
- Next (TPU): Mosaic decode page copy, flash fresh rows, kv append on the pool; speed vs main. CUDA-plugin chunk-count
  re-check pending here.

### M72 — 2026-11-10: Metal held weights, quantized weight streaming, mamba conv fix (BMX)
- `sc/bmx-metal` (c16362372, f42e97dd9, a1466b196, 6d6e93226), all on the Mac Studio:
  - Held weights: the load seats atoms (after the seats' runs, else from chunk 1), states `seated`, warm loads pread
    streamed weights out of the mapped windows; runtime adopts. Bitwise in 3 new + 5 old modes on 5 minis.
  - Quantized streaming: every quantized expert kernel path (affine 64/{2,4,8}, 32/2, 128/2, MXFP4 +/- bias; qmv,
    batched, MPP, register tiles) and 4/8-bit embedding gathers read through tables; compiler: Metal expert pieces
    hold multiples of 128 output rows. Kernel-level and e2e bitwise (u4 experts, MXFP4 in 9 pieces, 4-bit PLE).
  - Real models via `pie run`: tokens = base. Qwen3.6-35B-A3B-4bit boot 32-37 s -> 10.8-12.2 s (off); gemma held
    +0.5 GB vs base (copies all PLE rows).
  - Fix: Metal buffered every conv input in the rs list; non-recurrence convs (mamba2) now commit as they run ->
    nemotron3-nano-mini and qwen38-flash-next-mini load and match.
  - kv page 16 vs 32 on Metal measured: equal or 16 slightly better -> stays 16.
- Open: runtime boot fit check counts only committed segments (Metal: one ~65 MB) -> must count growable segments
  (engine works around by backing weights + floors at load). Writer refuses qwen38 u2 (ple.table 160 not a multiple
  of 64). Affine 2-bit / biased MXFP4 untested on real models.

### M73 — 2026-11-10: boot weight fit counts the growable pool (RT3, PR-READINESS A3)
- `sc/rt3-fit` (b3d9b646a): `SlotPool::growable_chunks()` (free chunks below activation across every allowed
  segment); `SlotPool::commit_for` commits the lowest uncommitted segments the held weights (or a streaming floor)
  need before claiming; Steered::boot asks the engine to back that prefix before the starting residency;
  `SharedGrow::hold_mapped` keeps the governor from giving those segments back. Metal-like facts tested; CUDA e2e
  (768 MiB and default) unchanged. Metal device run pending; engine-metal's back-everything-at-load workaround can
  shrink later.

### M74 — 2026-11-10: Metal on the M73 runtime fit (BMX2)
- `sc/bmx2-metal` (bc2fd6d12, ff47d82a3): engine-metal backs only the chunks it seats (whole segments); the runtime's
  growable fit (M73) passes on device. `pie run`: gemma-4-e4b-4bit and Qwen3.6-35B-A3B-4bit tokens = base in off /
  auto / on; Qwen3.6 boot ~11 s vs base 32-37 s; footprint ~0.1 GB below M72. qwen38 u2 imports (writer fixed in M68);
  new affine 2-bit (g128) + 4-bit + 8-bit weight-cache test bitwise in all eight modes.

### M75 — 2026-11-10: held load with a small pool no longer crashes (PR-READINESS A2)
- `sc/l40s-held-small` (2da2eaa12): preseat_plan ignored a stated device_kv_budget and seated weights into chunks the
  fit then cut off -> arming wrote unbacked memory (CUDA 700). Now it stops at the budget (seated = None -> pinned
  fallback), and pin_weight_hosts rewrites the tables to the pinned copy before arming. golden_held_small: nemotron
  mini held at 256 MiB boots, arms 10/10, equals the all-host result; dense at 256 MiB = base. Runtime refusal message
  under off to be confirmed by IT.

### Note — 2026-11-10: main #1604/#1605 merged; build fix
- The merge of main (#1604 gemma-4 shared tail readouts, #1605 tall prefill kv-append parts) into new-memory broke
  kernels-xla (pushed briefly as 68a2b38a3): #1604's new `flash::readouts` calls `older()` without the M63 page maps.
  Fixed by passing `[keys_map, values_map]` through `attend_readouts`. engine-xla 14 / kernels-xla 18 lib tests pass.
  TPU session to verify the readouts path bitwise on v6e.

### M76 — 2026-11-10: dense weights always on device (A10, B2)
- `sc/l40s-dense-on-device` (0cef5ef5d) + `sc/bmx3-metal` (731361b76, d1187db11): `device_weight_budget`,
  `host_weight_budget` and `max_state_slots` removed (config/removed.rs refuses each with why; six backend option
  structs + embedded Settings; bindings / template / Python / tests updated). CUDA rotate.rs and the dense rotor,
  Budgets, admit tiers deleted; Plan::cut puts every dense plane on the device; refusal up front when weights exceed
  the share. Vulkan/wgpu dense pump + host ring deleted, same refusal. Metal: budget plumbing gone plus its own
  unreachable expert tier / gathered-row slab / host source (~3,900 lines); its fit counts streamable weights at the
  cache floor.
- Verified: L40S golden_logits / by_refs (8) / rows / offload / experts = base; Vulkan + wgpu logits = base bitwise
  (VK5, 16 + 14 dumps) and the refusal fires; Mac: weight-cache tests, pie run gemma-4-e4b + Qwen3.6-35B tokens = base;
  full cargo check matrix after merge.
- Left (follow-up commit, L40S): engine-cuda's now-unreachable legacy expert tier and Residency.deferred_tier /
  host_weight_deferred -> delete. tests/handles/e2e.py --slots runs need new sizing.

### Note — 2026-11-10: e2e handles run on the shared pool (sc/e2e-handles-sizing) — BUGS FOUND
- tests/handles/e2e.py sized by --kv-budget only (176 / 136 / 116 MiB runs). Default run 56/60.
- BUG (engine, correctness): device_kv_budget <= 2 GiB -> "bodies armed 0 of 112", and Qwen3.5 resume (second process
  right after the first) produces wrong tokens from its 3rd token, deterministic. Armed (>= 4 GiB) is correct. -> EAGER.
- BUG (runtime): host tier full -> launch fails NoEligibleVictim instead of quota; a parked resume then waits forever,
  shortfall 2.78 GB for a 2-token turn (matches the engine's "fire wants 2776596480 bytes of the state pool").
- GAP: named snapshots (t_snapshot, t_long_snapshot) fail on the shared pool: take_host returns None (byte reads of
  host refs not stated, contract 22) -> must work before the PR.
- GAP (policy): keep-handles `Retain::upto(head)` covers kv tokens only; the record keeps key -1 and goes first.
- t_quota / t_close: launch-time state quota removed (4cb817555); host limit needs host_kv_budget in the script.

### M77 — 2026-11-10: XLA pool blocks by Mosaic DMA; kv append onto the pool (TPU session)
- `sc/xla-tpu` (4d40b8cd9..2975a122b): pool blocks move HBM->HBM by Mosaic DMA through tpu.memref_bitcast (PIE_XLA_POOL_DMA=0
  off, for the CUDA plugin); token-row appends and per-lane state slices as DMA blocks; the Mosaic kv append writes a
  pool plane directly.
- v6e speed vs base (tok/s): qwen3-0.6b decode c1 615->711, c8 4981->4928, c64 17.4k->16.4-17.5k, prefill 512
  124k->100k, 2048 169k->149k; qwen35 c64 15.2k->13.2k, c256 18.1k->17.4k; qwen3-30b-a3b mini c64 31.2k->31.5k,
  c256 65k->57k, prefill 2048 262k->237k. Remaining: prefill (append = 2 calls + 2 cases vs main's one pass),
  hybrid/MoE c64/c256.
- OPEN: with DMA reads, MoE and gemma minis differ by up to 0.03 (3-7 of 16 readouts) at pools +64 MiB and larger;
  dense/hybrid bitwise at every size; reads proven exact by unit tests; with DMA reads off, bitwise at every size but
  c64 12.3k. Suspected XLA fusion change around DMA reads (unproven).

### Note — 2026-11-10: XLA DMA-read pool-size dependence proven compiler-only (accepted)
- sc/xla-tpu 234036561: PIE_XLA_POOL_DMA_CHECK=1 cross-checks every DMA block read against XLA's gather (any
  difference -> NaN; =2 self-test NaNs 16/16). At +1040 MiB: 0/16 MoE and 0/29 gemma readouts NaN -> DMA bytes exact.
  Optimized HLO diff +0 vs +1040: same ops / kernels, different memory-space assignment (VMEM vs HBM) and fusion
  grouping around the opaque DMA custom calls -> moved bf16 roundings in routed MLP / attention. Accepted under the
  kernel-arm rule (max 0.03, no top-token change). TPU now on the perf-parity gaps.

### M78 — 2026-11-10: engine-cuda legacy expert tier deleted
- `sc/l40s-legacy-tier` (745055eb4, -1,564 / +195): Plan.banks/seated, Tier seats/berths/votes/promote/cells,
  deferred fill and T2 counters, WeightRow::Streamed, the hosts_experts recording gate, Residency/Boot.deferred_tier
  ([model] host_weight_deferred refused). Tier keeps only pinned images of plan-pinned planes and file-mapped planes.
  Goldens / rows / held_small / experts = base; build matrix clean.
- Left: kernels-cuda / kernels-rocm still take a GroupSeat argument always passed RESIDENT -> remove (CUDA side L40S,
  HIP side AMD) before the quantized formats.

### M79 — 2026-11-10: ROCm generated region kernels by default (AMD A3)
- `sc/amd-rocm-port` (2c74009b4, e6799ade4, 78bcf5591, d3f60e9c5, 60da5cb9f): ROCm JIT compiles generated units; the
  wave64 row skeleton at parity with kernels-cuda synth (22 CUDA synth tests pass on MI300X); ROCm serves synthesized
  regions and its replaced hand row fusions are deleted; ROCm goldens re-baselined with per-step traces.
  Parity: dense / hybrid within noise; MoE b8 within noise over five prompts. CUDA golden_logits unchanged after merge.
- `sc/amd-envelope-test` (e526a6a5): stale kv envelope test (missing page-stride arg) fixed; not a serving bug.
- Recorded pre-existing: two models on one ROCm device (flat arenas on main too).
- OPEN (blocker): ROCm at DEFAULT config arms 0 bodies on new-memory (main 89/89): the flat chunk range takes all the
  spare before bodies -> EAGER agent fixing in shared code.

### M80 — 2026-11-11: dead-code sweep (~5,000 lines)
- `sc/dead-code` (21 commits): engine-metal ICB plane / recording walk / descriptor ABI survey (~2,700 lines),
  kernels-metal icb_rebind, engine-xla serve::dry, FireCost on every engine, runtime EngineConfig fields and dead
  counters, `Capabilities::chunks` no longer optional (worker's degenerate fallback gone), `copy_state` removed from
  shared-pool engines (only the Cerebras wafer engine keeps it), engine-cuda one_frame_when_short decision, etc.
- Merge fixes: kernels-xla `plane_append` dropped beside the TPU session's `rebase`; runtime `engine::load::sequence`
  (the import's artifact order by the now-deleted tier ranking, M78) removed with its `merge_order` parameter; three
  stale imports in engine-cuda dispatch/custom.rs.
- Verified: cargo check --all-targets for the workspace, engine-cuda cuda / rocm, runtime all engines (0 warnings);
  lib tests runtime 278, engine 40, worker 12, kernels-xla 18, engine-xla 14; DC's CUDA goldens byte-identical.
- Listed, not deleted: pinned-pool host tiers (TierKind::Rs / RsPages / Window) that only the wafer engine could
  reach; host/disk branches in shared-pool copy_kv; Cerebras Unified; Vulkan/wgpu profiling hooks; worker
  set_draining; test-only APIs (incl. the runtime's rung API: SharedPool::raise_rung / lower_rung / steer — no
  production caller -> check whether the governor's rung shrink is wired).

### Note — 2026-11-11: serving order restored (sc/l40s-serving-order 741c07c01)
- M78 broke `pie --features cuda` (runtime engine::load::sequence called the deleted Ranking::images); M80 removed
  sequence() outright, which changed artifact plane order. L40S's fix adds engine_cuda::experts::serving_order (same
  order as before: dense planes by first read, then packed routed groups, then dense routed banks) and the import uses
  it again, so serving-file layout is unchanged vs main. Merge check matrix now includes `pie --features cuda / rocm`.

### M81 — 2026-11-11: kernels-rocm drops the dead GroupSeat argument
- `sc/amd-groupseat` (111c8dfe): HIP quantized selects / gathers / dequant lose GroupSeat (always RESIDENT since M78);
  three shared engine-cuda call sites temporarily duplicated per backend (CUDA copy deleted with L40S's kernels-cuda
  GroupSeat removal). ROCm goldens all 8 + buffered bitwise; build matrix incl. pie cuda / rocm clean.

### M82 — 2026-11-11: bodies always arm; capture fire runs once; unrecorded serving is loud (A11)
- `sc/eager-fix` (97fc96139): arming's synthetic lanes took a record + fresh pages each (~269 chunks on Qwen3.5-0.8B)
  and failed inside a budgeted pool -> they now share records/pages (Pools::synthetic_room); fit_the_card subtracts
  unbacked activation (likely the ROCm 0/0 cause, AMD verifying); a load that arms nothing is refused. Wrong tokens:
  on the fire that captured a body during serving, the step ran AND the new body launched -> recurrent rows folded
  twice; extra launch removed. Unrecorded serving fires log WARNING (1st, 2nd, 4th, ...) and count eager_unbodied.
  New probe golden_eager_resume. t_resume correct and 112/112 armed at 216 MiB / 1 GiB / 2 GiB / none; CUDA goldens
  = base (re-run after merge).
- Merge fix: enqueue.rs eager_walk (M78 removed hosts_experts) + eager_unbodied reason; stale deferred_tier in a probe.

### M83 — 2026-11-11: dead-code sweep 2 (~1,100 lines)
- `sc/dead-code-2` (4970e5a76, 31c158fa1, ef1a08ced, 0dda02e7f): Cerebras Unified; shared-pool copy_kv host/disk
  branches (CUDA spill_kv path, Metal disk branch, KvCopy::tier_move, Metal copy_pages); CUDA Pinning::uninit;
  pinned-pool host tiers TierKind::Rs / RsPages / Window, StateCopy::direction, degenerate window / host-state runs.
  Matrix 0 warnings incl. pie cuda / rocm; runtime 279 lib tests; CUDA goldens byte-identical.
- FOUND (A14): the governor's rung shrink under critical pressure is not wired: SharedGrow::shrink_rung / restore_rung
  are no-ops ("the scheduler does not cap a fire's tokens by rung yet"); raise_rung / lower_rung / steer exist unused.
  SPEC decision 6's activation boundary never moves.

### M84 — 2026-11-11: shared-pool snapshots, host-full handling, head key covers state (A12)
- `sc/rt4` (aaf68fdab, abda894ba): tier-pressure shortfall measured in the shared pool's own slot sizes (was ~24x:
  the 2.78 GB for a 2-token turn); host-short heads wait on policy and fail `Short{Host}` ("quota: the host tier is
  full (host_kv_budget) …"); restores fail after SEAT_WAIT instead of hanging; a launch leaves one set's state room
  on the host. Snapshots: new `Engine::read_tier_slots` / `write_tier_slots` (host-tier slots by shape and ref, as
  copy_slots lays them); CUDA implemented (TP too); others answer unsupported (A15). A set at rest ranks by its
  highest key (keep-handles head protects record / rs list / window). tests/handles/e2e.py --host-budget.
- e2e sm_120: t_snapshot / t_long_snapshot pass, t_quota / t_close pass, t_keep head-key check passes (9 pages, was
  19). Runtime lib 279+; build matrix clean.

### M85 — 2026-11-11: CUDA host tier is pageable memory behind a pinned staging ring
- `sc/l40s-host-pageable` (36517dc0c): HostTier = one anonymous MAP_NORESERVE mapping (MADV_HUGEPAGE; RSS = data
  written) + a 512 MiB pinned staging ring + landing threads (big batches split over 4). D2H into the ring on the swap
  stream, landed into the tier after the event; H2D reads tier->ring in a stream host function, then ring->device.
  Pin thread / lookahead / SHELF_AHEAD / TIER_PIN deleted. Contract 22 unchanged (fence_swaps).
- Root cause it fixes: demoted cache grows the held host data monotonically (3,004 of 3,584 chunks first-touched in
  order), so page-locking fresh memory could not keep pace -> Qwen3-8B prefill 2048 492 -> 151 ms (main 145).
- Merge fix: rt4's read/write_tier_slots go through HostTier::slice (waits for landings); HostTier::chunk_bytes.
- Verified: golden_offload + golden_logits (sm_120) = base; build matrix clean.

### M86 — 2026-11-11: XLA pool parity work (TPU session)
- `sc/xla-tpu` (8977ec8ce..65a075593): a prompt's kv append lands both pool planes in one write; DMA gathers unrolled
  with one wait; warm-ladder seal (fires outside the ladder counted; PIE_XLA_WARM_STRICT=1 fails); pool-native Mosaic
  conv step for decode.
- v6e (in-process medians): qwen35 c64 13.7k -> 15.3k (base 15.1k, parity); prefill 512 above base; MoE prefill above
  base. Conv kernel: state bitwise vs base, output ulp-level (base XLA clones the in-projection matmul into the conv
  fusion) -> explained matmul-arm difference (PIE_XLA_CONV_KERNEL=0 restores bitwise). Accepted.
- Open: qwen3 c256 -4% (pool-native decode page copy next), prefill 2048 -7.5%, MoE c1 -3%; sc/xla2 rebased
  (snapshots bitwise on TPU, PLE rows bitwise; streamed experts open).

### M87 — 2026-11-11: main's Cerebras wafer-program backend (#1435) merged
- `sc/merge-main-cerebras` (b64cc6509): main deleted the host-traced Cerebras engine and replaced it with a
  wafer-program backend; new-memory's host chunk pool / tiers for the old engine (M43/M44) were dropped with it. The
  new backend states `ChunkFacts::degenerate` (kv pages = host token lists replayed into the program; state slots),
  refuses copy_state / copy_slots / tier verbs / disk_kv_dir, weights placed whole. Worker keeps M76's removals and
  adds main's new Cerebras options. Build matrix clean; worker / engine-cerebras lib tests pass.
- Cerebras north-star follow-up (A8 replaced): multi-sequence on-wafer kv as a real shared pool (segments, evacuate),
  host / disk tiers, record copies, weight streaming. Needs the backend's owner / hardware.

### M88 — 2026-11-11: XLA decode at or above base (TPU session)
- `sc/xla-tpu` (6822f26f5, 9c0a12282, d63a5cc97): main's decode page-copy kernel pool-native; the prefill's older
  pages by one DMA call with PageMap / sink / clamp in the kernel; decode kv rows for both planes in one block
  scatter. qwen3-0.6b c64 18.0k (base 16.5k), c256 22.6k (base 20.8k). New gate a_wide_decode_of_many_lanes bitwise.
- Open: qwen3 prefill 2048 -5% (pooled append kernel's in-VMEM [16,1024]<->[128,128] relayout, inherent to packed
  rows for w > 128); MoE c1 -3% (device time equal or lower; host runs ahead less — donated-pool wait suspected).
- Merge fix: kernels-xla test Shape initializers gain `pooled: None`.

### M89 — 2026-11-11: held weights claim the pool before the bodies (A16)
- `sc/eager-fix-2` (b774f872c, bf34a80e8): M82 subtracted unbacked activation and the bodies' allowance before
  counting held weights, so the pool shrank under them (MI300X Qwen3-30B-A3B flipped to streaming and stalled). Now the
  fit reserves one working set plus every held weight first when the card holds them beside programs and activation;
  preseating stops at the fitted pool; arming margin capped at a quarter of the allowance. CUDA qwen3-30b-a3b-mini:
  shares 0.90 / 0.14 / 0.13 hold weights and arm fully; 0.11 streams. CUDA goldens = base after merge.
- Open: arming over the host copy after a flip to streaming is very slow (8-28 s per prefill body) -> arm after the
  starting residency (EAGER, next); MoE mini t_resume diverges from token 5 (pre-existing) -> EAGER to investigate.
  MI300X verification pending (AMD).

### M90 — 2026-11-11: idle cache demoted only when keyed above the floor (A18, user decision)
- `sc/rt6-keyed-demote` (1082c3799): `retention::FLOOR` (= 0, the key of uncovered KV) and `above_floor`;
  `pressure::may_drop` = not above the floor; prepare_demote and prepare_spill skip floor-keyed groups, which
  reclaim_idle drops as on main. Suspend / sweep / preempt moves unchanged.
- Verified: runtime store/planner/policy 192; e2e t_keep 10/10, default 63/65 (t_clients unbuilt); keyed prefix
  (keep-session) survives a restart via disk (125 pages), unkeyed control dropped; Qwen3-8B 4 x 256 concurrent waves
  1810-2097 tok/s flat, 0 idle demotion, RSS 9.2 GB (only preemption traffic remains).

### M91 — 2026-11-11: no infinite restart when a process's own demand exceeds the pool (A19)
- `sc/rt7` (1e0fc42ed, d0cd87bdf): check_overcommit (held kv + parked asks + state counted in kv pages vs the pool)
  fails the process's asks with PlannerError::Overcommit instead of restarting forever; ShortRs releases pages
  gathered past the head's need on a shared pool; departures count state; starved-ask errors name missing state.
- Probes (sm_120, 128 / 96 MiB): restart loops -> clear errors in 0.01-0.12 s; c2 batches still serve. Runtime
  planner/store/inferlet/policy 208 pass. Open: ~1,600 restarts in 2 s before the error in one 96 MiB case
  (fleet_stalled marks a process stalled while other lanes have fires in flight).

### M92 — 2026-11-11: the governor's rung shrink moves the activation boundary (A14)
- `sc/rt5-rung` (670f591d6): poem_compiler::arena_rung carves the arena for fires up to 256 / 1024 / 4096 tokens,
  each carve at the top of the activation span (fixed addresses: bodies stay valid at every rung their fire fits);
  CUDA states rungs in ChunkFacts and re-reserves the range to the fit when rungs exist. Runtime RungWalk: down = cap
  fires, wait for wider fires to retire, lower_rung; up = evacuate state out of the larger span, apply_rung, lift the
  cap; governor shrink_rung / restore_rung step it. PIE_PRESSURE_FILE test hook; tests/handles/rung.py.
- Verified: runtime 214, poem-compiler 34, e2e rung.py 6/6 (Qwen3.5, induced critical pressure), CUDA golden_logits /
  by_refs / offload = base after merge. Limits: raise evacuation proven by unit test only; one decode differs at the
  smaller rung (finer prefill cuts, unconfirmed); only CUDA states rungs (A21: Metal / Vulkan / wgpu / XLA).

### M93 — 2026-11-11: combined fit (pool-fit + flat-fit) and no churn for stalled sole holders
- `sc/l40s-pool-fit`: the fit after arming (settled) subtracts only the activation span a fire backed
  (activation_unheld_bytes), giving the rest to cache rows -> recovers the c64 -19% left by eager-fix.
- `sc/flat-fit`: the fit counts a flat range's whole backing as room (ROCm Qwen3-30B no longer flips to streaming
  on that miscount). Combined in fit_the_card: live = spare + backed - (settled ? act - unheld : act).
- `sc/rt7b`: a process with fires in flight is not stalled; a stalled sole holder fails instead of restarting
  (closes the A19 residual churn).
- Verified: workspace / engine-cuda cuda+rocm+featureless / pie cuda+rocm / runtime all-features check clean;
  runtime 235 pass; CUDA golden_logits / by_refs / eager_resume / offload = base. Pending: L40S Qwen3-8B
  c64/c256 confirm on the combined fit; AMD MI300X 30B verify.

### M94 — 2026-11-11: Vulkan/wgpu speed pass 1 (A15 for Vulkan/wgpu)
- `sc/vk-perf` (fb898ea56): per-row segment lookup in Vulkan sdpa/gdn/conv; record-key fix (more replays);
  expert-miss fires recorded via segmented command buffers; read/write_tier_slots on Vulkan and wgpu. Logits
  bitwise-unchanged. Checks: engine-vulkan (vulkan), engine-wgpu, pie vulkan+wgpu clean; engine-vulkan lib 18 pass.
- Open (VK6): split_rows=32 keeps >=32-row fires unrecorded on desktop; descriptor indexing; wgpu c1 host work;
  gemma c64/boot; Qwen3-30B 6.57 GB embed view past 4 GiB; trimmed-window model_logits test.

### M95 — 2026-11-11: XLA tier slot copies (A15 on XLA); A21 not applicable on XLA
- `sc/xla-tier` (0a109def3, TPU): read/write_tier_slots at SlotShape::tier_offset per shape; refuses wrong byte
  counts / overruns whole; park test restores qwen3-micro and qwen35-tiny through host-slot bytes, bitwise on v6e.
  engine-xla 14 + kernels-xla 18 lib pass; pie --features xla checks clean.
- A21 on XLA: activation is PJRT scratch outside the single fixed pool -> no rungs stated, walk stays a no-op.

### M96 — 2026-11-11: dead rung APIs removed
- `sc/dead-code-3` (06ffda2ec): SharedPool::raise_rung, ::steer, SegmentPort, RungError::Engine deleted (only
  tests called them since M92's RungWalk). Tests now drive the live walks (walk_segments, walk_raise); the raise
  evacuation test runs against RungWalk::step + apply_rung. runtime store:: 124 pass; runtime rocm / default check clean.
- MI300X: M93 verified by AMD (30B holds weights, 89/89 armed on 4 models, 9 ROCm goldens equal).

### M97 — 2026-11-11: streaming loads boot in seconds (A16 eager-fix-3)
- `sc/eager-fix-3` (fc75cb91f, EAGER): a streaming load arms at its starting weight residency (deferred arm,
  stand-in tables); streamed bf16 expert banks take the tiled mma.sync select (ordinary loads read host-mapped
  experts; tcgen05 / Hopper wsgemm stay off streamed tables). qwen3-30b-a3b-mini share 0.11 on sm_120: b2048 body
  48 s -> 466 ms, boot 2m45s -> ~21 s, 104/104 armed; held share 0.9 unchanged (139/139, 4.5 s). Flat range: the
  settled fit no longer counts unheld activation as room (activation_unheld_bytes = 0 on a flat range).
- Verified: new streamed-table bitwise test + 7 moe_table pass; CUDA golden_logits / by_refs = base, golden_held_small,
  t_resume 4/4 112/112 armed; engine-cuda cuda/rocm/featureless + pie cuda/rocm check clean. MI300X (AMD): 30B holds
  weights, 106728 pages (= M93).
- Open (AMD, MI300X fill test): on a flat range the fit still counts held weight (expert) chunks as cache room —
  30B declares 106728 pages but only ~70k fit (38 x 31.4k seqs starve; main sizes 69120). Fix in progress.

### M98 — 2026-11-11: runtime per-fire epoch hooks batched (CUDA c64 runtime overhead)
- `sc/rt-perf` (b4c7f7fbe, RTP): each fire's FireEpoch release spawned its own tokio task (64 spawns per decode step
  at c64, ~150 us); settle_batched runs a landing's hooks in order from one task. Qwen3.5-0.8B c64 (sm_120): main 11756,
  new-memory 11236 (-4.4%), with fix 11825 (+0.6%); c256 all within 1%. runtime lib 297 pass.
- Found: gemma-4-E2B fails to boot on new-memory (CUDA 700 preparing the last wide bucket 4096) since between M67 and
  M92 (rt5); M67 build boots. Owner assigned (bisect).

### M99 — 2026-11-11: window models reuse prefixes at any length (A20)
- `sc/rt8-window-prefix` (0aaf75a29, RT8): automatic checkpoint at a page-aligned pass end (eac8052c6); a pass
  ending inside a page checkpoints at the last page boundary via RsStore::checkpoint_at, holding the window slots
  [b - w, b) (WindowAdvance::keep stops the fire trimming them); resumes copy shared tails via slot_copies.
- gpt-oss-20b (sm_120) p2048 after a shared 512 prefix: 93-97 ms vs main 96-97 ms (was 117 ms, -25%); cold 2048
  ~4% faster than main; no unrecorded fires. gemma-4-E2B (pre-merge): 67.5 -> 55 ms with prefix, main reuses none.
  runtime store 126 / planner 51 / pipeline 16 pass.
- Closed: gpt-oss resumed tokens differ from cold in 3-4/6 seeds — main's own reuse diverges on the same seeds at the same
  tokens (split-prefill MoE near-ties), cold outputs identical to main.

### M100 — 2026-11-11: Metal tier slot copies (A15) and first perf fixes; A21 n/a on Metal
- `sc/metal-gate` (f6b70f094, e1b93e4da): read/write_tier_slots on Metal's host tier (one memcpy per slot, unwritten
  chunks read as zeros); parked-lane bytes mode bitwise for dense / hybrid / windowed. Weight-use reads no longer
  drain in-flight frames (counters report growth); tiled prefill attention does page lookup once per token
  (641.6 -> 574.6 ms device, main 595.4). A21: activation is a sparse buffer outside the pool -> not applicable.
- M1 Max vs main: Qwen3-0.6B p2048 +2.8% (was -2.4%), p512 +6%, c1 -0.8%; Qwen3.5-0.8B prefill +6..23%, c1 -0.9%;
  gemma-4-e4b c1 -2.4%, p512 -4.2%, p2048 -2.0%; Qwen3.6-35B-A3B boots in 14 s (main: not within 600 s).
  Greedy tokens identical to pre-change on 3 models x 3 prompts. Open: gemma decode/prefill, Qwen c1 small gap.

### M101 — 2026-11-11: Metal held-weight seating 2.4x faster (gemma boot / prefill parity)
- `sc/metal-gate` (5e9f63f7d): PLE row seating at load described as runs, chunk addresses looked up once, windowed
  reads over 8 threads, slot positions computed once (1440 -> 590 ms). gemma-4-e4b (M1 Max) vs main: boot 1.95 s vs
  3.27, p512 +1.8%, p2048 +0.3%, c1 -2.4% (open: +0.38 ms GPU per fire; suspect per-buffer hazard tracking on the
  single pool buffer). Greedy tokens unchanged; weight-cache engine tests pass on the Mac.

### M102 — 2026-11-11: XLA speed pass, Vulkan/wgpu pass 2, Metal token table in place
- `sc/xla-tier` (67e146592, TPU): pooled kv append builds u32 rows, sink pages from a zeroed VMEM block, prefill's
  older pages by XLA gather, narrow decodes without kernel calls, block gather only up to 512 KiB. vs main:
  qwen3-0.6b c8 +57%, c64 +7.7%, p512 +21%, p2048 -1.8% (was -5.5%); MoE c1 +1.3% (was -3%), c256 -3.8%;
  qwen35-d0.8b p2048 +7.4%, c64 -4.1% (both c256 / c64 gaps present at the tip). Bitwise gate passes.
- `sc/vk6` (82b3a35ff): desktop GPUs submit steps whole (c64 / prefill record; Adreno keeps split 32); a view in a
  slab past 4 GiB binds the range when its tensor fits (30B first fire); threaded PLE fill (gemma boot 7.2 s vs main
  9.0); one VkInstance per process (driver refuses the 27th: trimmed-window test fixed); WGSL rows find their segment
  once. Vulkan 16/16 + wgpu 14/14 dumps bitwise. vs main: Qwen3-0.6B c64 862 vs 785; gemma c64 >= main; open:
  Vulkan p2048 -1.8%, Qwen3.5 wgpu c1 (noisy, GPU time per fire at parity).
- `sc/metal-gate` (e43834384): a held token-id table reads in place from the mapped weights (gemma boot 1.07 s vs
  main 3.42); hazard tracking ruled out for gemma c1 -2.1% (bisect next). Greedy tokens unchanged.
- Held back: sc/rocm-fit and sc/e2b-fix wait for L40S's fix of the M93 combined fit (Qwen3-8B on L40S: 7196 pages
  vs pool-fit 9275 / main 9216, c64 -19%).

### M103 — 2026-11-11: CUDA fit with rungs, gemma-4-E2B boots, golden_rows, Metal c1, Vulkan prefill
- `sc/l40s-fit-rungs` (53d9f8850, L40S): a mapped range's stage reshape (rt5-rung's fit_range_whole) reserves the
  stage fit + unheld activation + bodies' allowance, so the settled fit can take the unheld span (Qwen3-8B on L40S:
  7196 -> 9303 pages, c64 644.9 / c256 678.8 vs main 649.2 / 687.8). Flat range unchanged (EF3). Includes
  `sc/rocm-fit` (b9174fb11, AMD): a held load's weight chunks are out of the kv pages (MI300X 30B 106728 -> 69864
  pages, main 69120; 34 x 31.4k fills complete; no-op on CUDA models tested).
- `sc/e2b-fix` (b55cfba59): a captured fire's readout padding capped at its bucket (M92 put the forked carve at the
  reservation's end; mixed bucket-256 padded 512 readout rows past a 256-row carve -> CUDA 700). gemma-4-E2B boots
  136/136 armed.
- `sc/rows-fix` (a08357e53): arming zeroes the weight-use reads its synthetic fires made (EF3 arms streaming loads
  inside weight_residency; golden_rows read 1570 vs 546).
- `sc/metal-gate` (0b853430e/dfd8f524e, ac4c6e2e2): paged planes bind at their offset into a page; in-page cell math
  in 32-bit. Metal c1 vs main: Qwen3-0.6B -0.1%, gemma -0.7..-1.7%, Qwen3.5 -0.8%.
- `sc/vk6` (2a7c24da8): one-segment twins of Vulkan prefill attention (serving binds one segment): Qwen3-0.6B
  p2048 9712 vs main 9510 (+2.1%), p512 +5.6%. Vulkan 16/16 + wgpu 14/14 bitwise. A21 not applicable on
  Vulkan / wgpu (activation arena outside the pool).
- Verified (sm_120, release): golden_logits / by_refs / held_small / offload = base; golden_rows host/cached/mixed
  546 reads = base; golden_buffered / submit_buffered = buffered reference; engine-cuda cuda/rocm + pie cuda +
  workspace check clean.

### M104 — 2026-11-11: CUDA lane seats = max_lanes; XLA kv budget and program key
- `sc/prefill-mix` (a9250e860): with nothing stated, a CUDA load seats boot.budget.max_lanes slots (the chunk pool
  had dropped engine.state_slots_per_lane and fallen back to 256). gemma-4-E2B arms 144/144 = main's key set; c512
  fires reach 512 lanes; c256 11368 vs main 11171. golden_logits / by_refs = base; runtime 299 pass.
- `sc/xla-tier` (945ea9b67, 414cb19b4): kv budget buys pages (null + sink past it; MoE decode:8 no longer panics);
  full program key only on a shape-key miss.
- Open: gemma-4-E2B c64 -1.5..2.5% — not prefill packing (identical scheduler and fire mix); per-fire GPU time jumps
  ~165 us mid-run and disappears with a 3 GiB pool -> likely kv page placement (test against L40S's slot padding).

### M105 — 2026-11-11: slot stride skew (DRAM channel aliasing); Metal c1 pass 3
- `sc/l40s-slot-skew` (daea97605, 686a8a8fa, L40S): poem_compiler::chunk::plan makes every token-slot stride an odd
  number of max(1 KiB, align) (3·2^k strides alias on L40S's 12-channel bus: up to 1.49x slower page reads past
  L2). No config, kernels unchanged; XLA lays its own slots (unaffected). L40S c64 / c256 before -> after (main):
  gemma-4-E2B 6325 / 6465 -> 6567 / 6829 (6580 / 6879); Qwen3-8B, Qwen3.5 unchanged; 0.08-0.8% fewer pages.
  nemotron-super fixed waste 0.53% -> 1.91% (bound widened to 0.025, user decision (a)). CUDA goldens bitwise;
  poem-compiler chunk 11 + every_deployment 13, runtime store 126 pass; all backends check clean.
- `sc/metal-gate` (ab5e7172b, 693c5321a): state record lookups divide by multiply-high (exact for every u32);
  decode attention page strides as Metal function constants. M1 Max c1 vs main: Qwen3-0.6B +0.3%, gemma -0.3%,
  Qwen3.5 -0.5% (c8 +1.1%). Rest of Qwen3.5's ~45 us/fire is inside the state kernels (needs a GPU timeline).

### M106 — 2026-11-11: window models cut long passes
- `sc/window-cut` (0783d49e2, RT8): window-list models go through the pass cutter (page-boundary pieces, each its
  own fire writing every window list and trimming behind its committed end; a piece short of window slots settles
  the tail first; checkpoints via checkpoint_at/keep). gpt-oss 1 x 31.4k 2.48 s (main 2.55), 8 x 31.4k 20.3 s
  (main 20.6), p2048 122.4 ms (main 126.0); gemma 31.4k now serves (main refuses: window ring). runtime 194 pass.
- Re-measured: 24 x 31.4k completes at main's speed (Qwen3-0.6B 33.2-36.7 s vs main 32.8; gpt-oss 61.8-62.5 vs 62.2-62.7),
  no stalls. The earlier 730 s / 297 s runs had accepted launch batches + a tracked kv copy uncompleted for 10-190 s
  from the first fires (not planner pressure) -- likely environmental or the rare engine hang also seen on main;
  looping under a stack probe to catch a recurrence.
- gemma-4-E2B c64 on sm_120 (after M105): decode GPU work per fire identical to main per kernel (time-weighted ratio
  0.998, same launches); ~1% mean gap (main ~6283 / 7 runs, new ~6215 / 5 runs; single runs overlap) comes from the
  per-round prefill bucket mix (same armed key set as main) and per-run clock level under the 200 W cap. Closed as
  noise-level; no kernel fix.

### M107 — 2026-11-12: rs-buffered fires recorded (A13, a main bug); MTP fixes
- `sc/rs-record` (466da2ba7): per-lane rs destinations / replay counts / rs page addresses as per-fire input data
  at load-fixed addresses (one body serves any slot assignment); table-driven rs copy kernels (CUDA + ROCm); three
  body variants (fold / buffer / replay; folding buffer into replay changes bits). Buffering variants armed only for
  a drafter's classes (an output name no plain class writes, e.g. mtp.drafts) on loads that buffer recurrent rows,
  at 14 buckets (`rs-bodies` diagnostic arms all). Fixes: each guest stage places the outputs it reads at its own
  slots (MTP past ~64 tokens failed on main too); planes over readout / lane rows fan out by one (panic at 16+
  concurrent MTP).
- sm_120: 0 eager fires with MTP; buffered goldens bitwise vs the eager walk; golden_logits / by_refs = base.
  Boot Qwen3.5-0.8B 20.1 s (main 22.1-22.5), Qwen3-0.6B 7.9 s (main 10.1); c64 / c256 unchanged vs base; MTP
  64 concurrent 5357 tok/s (main fails a second MTP process).

### M108 — 2026-11-12: per-fire host work at high lane counts
- `sc/cuda-residual` (95d24bd79): the scheduler moves lanes (with their tables / fixed / slot_copies) into the frame
  instead of cloning; FireInFlight counts skip the planner's O(P) with_inner walk (was O(P^2) per step under the
  planner lock, ~131k process visits per frame at c256); state_tables / seat_pages without per-lane copies. Decode
  frame host phases at c64 now at or below main's. golden_logits / by_refs = base; runtime scheduler / planner /
  pipeline / store pass. sm_120 is GPU/power-bound, so throughput there is unchanged (c64 12221 vs main 12301).
- Found: Qwen3.5 gdn_step / conv1d +7% per launch in serving: not codegen (same speed cold) but L2 placement --
  a layer's 64 lane states sit one record (~10.37 MB) apart at the same offset (likely L2 set/slice aliasing);
  +16% in the partly-L2-resident regime vs main's contiguous bank. Stride/offset sweep next. A/B tip: share gemm-algos picks across builds (each binary
  times and caches its own; ~13% of picks differ -> +/-0.3-1% bias).

### M109 — 2026-11-12: dead code and non-essential tests removed (user instruction)
- `sc/dead-code-4` (53850520c): phase 2 deletes the unreachable pre-shared-pool elastic path (runtime store::grow,
  PoolOp commit/release verbs, KvStore/RsStore extent methods, the tail-past-limit machinery, Engine commit/release
  verbs, PoolFacts elastic fields, engine-cuda commit_pool & co, Metal release_scratch & co, poem-compiler prefetch
  slotting) ~2,100 lines. Phase 3 deletes tests new-memory added except the essential gates (~23,850 lines): CUDA /
  ROCm golden probes (fixtures, logits, by_refs, offload, buffered x3, rows, held_small, experts, eager_resume),
  Metal / Vulkan / wgpu / XLA logits gates, one runtime test per store mechanism, kernels moe_table(_fp8/_fp4) /
  ROCm linear_moe_table, poem-compiler every_deployment_packs_into_chunks. Main's tests untouched.
- Merge: +72 / -25,947. Workspace --all-targets, engine-cuda cuda/rocm --tests, pie cuda check clean; runtime 150 pass.

### M110 — 2026-11-12: Mamba2 rs-buffered fires recorded (A13 remainder)
- `sc/rs-record-2` (830db112c): ChunkProfile::rs_mamba2 (CUDA) buffers each Mamba2 layer's conv input and dt;
  mamba2_extended replays them through recorded bodies; mamba2_chunked's tail pass normalises once (it re-normed
  first-pass rows: 28-41% of logit spread vs fold before the fix). Drafter classes read out up to lane_ceiling x
  readouts_per_lane (conc-64 MTP eager fires 9-11 -> 0); replay ceiling checked per window.
- sm_120: Nemotron mini buffered = eager bitwise, 0 warnings; Qwen buffered = M107; golden_logits / by_refs pass;
  boot / c64 / MTP unchanged vs M107. Known gap: mixed verify+prompt multi-class fires (2-4 per conc-64 run) still
  walk eagerly; Metal / Vulkan / wgpu don't buffer Mamba2.

### M111 — 2026-11-12: host-tier ring, MTP class split, XLA kv row order
- `sc/oversub` (439c01108, dc7993cd0): CUDA host tier's pinned staging ring sized to a quarter of the device kv pool
  (512 MiB-4 GiB) so a suspend only enqueues (engine thread blocked in tier moves 7-8 s -> ~2 s per run); contention
  trace times eviction phases / tier copies / restores. 24 x 31.4k: Qwen3-0.6B 32.3 s vs main 33.8; gpt-oss 61-62 s
  vs 62-64; boot 40.7-45.6 s vs main 40.4-49.4. Open: Qwen3-8B c256 on 32 GB -8.5% (planner evicts in overshooting
  bursts, restores trickle back ~20 / 2 s; quiesce up to 3.3 s per eviction).
- `sc/rs-record-3` (0f025adc7): a verify (drafts-fact) lane fires only beside lanes of its own class -> no eager
  mixed drafter fires; MTP / c64 unchanged.
- `sc/xla-tier` (2d0071387..5a6e585cb, TPU): every XLA kv plane in (row-pair, token) order (identity for rows <= one
  pool row), packed kernel branches deleted, non-fitting planes refused at load. vs main: MoE decode 256 x 80 steps
  73.6k vs 73.3k (was -8%); qwen3-0.6b c256 +9.1%, c1 +7.3%; open: MoE decode 8 -3.1%, qwen3-0.6b prefill 2048 -2.7%,
  d0.8b c64 -2.6% (layer-major state).

### M112 — 2026-11-12: XLA in-place pool writers stay in HBM
- `sc/xla-tier` (7ea2aeff3, TPU): Compiled::call_aliased colors in-place outputs and their aliased operands HBM
  (XLA had parked the whole MoE-mini pool in VMEM between Mosaic calls, ~52 us / step). vs main: MoE d8 +2.6% (was
  -3.1%), d1 / d256 / p2048 +0.8 / +0.9 / +3.9%; qwen3-0.6b d1 / d64 / d256 +7.2 / +4.2 / +9.4%, p2048 -1.8%;
  d0.8b d64 -2.6% (layer-major). Gates bitwise.
