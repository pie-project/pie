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
