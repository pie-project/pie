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
