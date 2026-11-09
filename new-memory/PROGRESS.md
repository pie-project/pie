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
