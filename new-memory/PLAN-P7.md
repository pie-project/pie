# P7 plan: engine chunk API, runtime slot allocator, governor, pool switches, kv-space removal
(Planner output, citations checked on state-chunk f2f1353df. Coordinator decisions are marked DECIDED.)

## Facts
1. The engine already numbers kv pages in one id space across all non-windowed kv rows: `KvDelta.pages` (engine/src/fire.rs:169-185); engines only distinguish windowed vs not (engine-cuda/src/serve/prepare.rs:1269-1276). Degenerate mapping: list shape = today's kv page; ring = windowed page (page 0 null); fixed = state slot; fixed [P,width] = rs buffer page.
2. Four allocators today: KvStore.pool (runtime/src/store/kv.rs:106, 108-112); RsStore pools/refs/checkpoints/StateExtent (store/rs.rs:221-256); SeatBook (store/seat.rs:27-90; pipeline/fire.rs:1709-1722; store/kv/working_set.rs:57,242); Extent::split (store/extent.rs:166-177).
3. Runtime drops state on its own in six places (all become tier-pressure to PLEX):
   RsStore::yield_checkpoints (rs.rs:668-671) via PoolPort::yield_checkpoints (planner/pool.rs:60-62,281-283), yield_own/yield_head_checkpoints (planner/eviction.rs:280-320), planner/acquire.rs:150-160, planner/plan.rs:132;
   prepare_suspend dropping checkpoints (rs.rs:1499-1504); Extent::release_state -> rs.reclaim_tail (extent.rs:331-334, rs.rs:438-447);
   Extent::vacate -> kv.reclaim_tail(epoch) (extent.rs:355, kv.rs:2502); state_room quota (handle/table.rs:391-437,500-518; server/handler.rs:491; policy/service.rs:395-404);
   rs_seat_cap admission cap + dispatch-depth override (bootstrap.rs:278-353).
4. Worker never raises tier-pressure today; WIT edge exists (crates/policy/wit/types.wit:120-121; crates/policy/src/lib.rs:199,275; runtime/policy/program.rs:408). Worker attrs are slot-counted (policy/service.rs ~1003-1020).
5. CUDA reserves a VA range per plane (engine-cuda/src/device/elastic.rs:346-400, called from store.rs:744-873). ROCm = same code, mapped_on_demand()==false (elastic.rs:23-25).
6. Existing bug: wgpu silently clamps an over-limit binding (engine-wgpu/src/encode.rs:310,350-355).
7. XLA per-row arrays cannot share one chunk space (each row sized for all chunks); only a fixed split at load works = degenerate. Real sharing needs the tiled pool.

## 1. Engine API — package E1 (new crates/engine/src/chunk.rs; additive edits to caps.rs, fire.rs, engine.rs, load.rs)
```rust
pub struct ChunkFacts {
    pub chunk_bytes: u64, pub total_chunks: u32, pub segment_chunks: u32,
    pub segments: SegmentFacts,          // {committed, max, growth: Fixed | Bottom | TwoEnded}
    pub reserved: Vec<Reserved>,         // chunk 0 = Null; Sink (padding lanes); Runtimeless{lanes}
    pub shapes: Vec<SlotShape>,          // {id, kind: Fixed{pieces}|List{tokens,rate}|Ring{window,tokens},
                                         //  slot_bytes, slots_per_chunk, sets_per_working_set (2 for fixed), rows}
    pub act_chunks: Vec<(u32, u32)>,     // (rung tokens, activation chunks), ascending
    pub host_chunks: u32, pub disk_chunks: u32, pub disk_persistent: bool, pub disk_kept: bool,
}
```
- One chunk-id space for kv and state; slot ref = chunk * slots_per_chunk + i (u32). Runtime hands out LOGICAL chunk ids; engine keeps logical->physical table (evacuation never renames a held slot).
- Reserved: chunk 0 null (replaces window page 0, caps.rs:66-70); a Sink chunk replaces XLA/Cerebras one-past sink page/slot (engine-xla/src/store.rs:6-8,399-430; engine-cerebras/src/serve/fire.rs:228-280); Runtimeless chunks for lanes the runtime doesn't seat (runtime/pipeline/fire/float.rs:182-190), replacing "state rides the seat" (fire.rs:207-209).
- Per-fire Lane inputs (optional during migration): `tables: Vec<(ShapeId, Vec<u32>)>` (replace kv.pages, kv.translation, kv.window, RsVerb::Buffer.pages); `fixed: Vec<(ShapeId, [u32;2])>` (running, cow) replacing slot, rs_slot, RsSnapshot.slot (fire.rs:195,209,213,387).
- Engine trait additions (old ones kept until F): commit_segments(n)/release_segments(n) off the fire path via PoolOp (runtime/scheduler/worker.rs:294-300); evacuate(&Evacuation{moves: Vec<(logical,new_physical)>}) stream-ordered before next fire; copy_slots(&SlotCopy{shape,src,dst,moves}) replacing KvCopy/StateCopy (transfer.rs:129-280).
- Capabilities gains `chunks: Option<ChunkFacts>` beside PoolFacts; `ChunkFacts::degenerate(&PoolFacts, ..)` = identity. New `chunk::fit(room, C, act_chunks(max rung), least_chunks)` replaces fit.rs (pages_for, least_state_slots, state_slots_within, balanced_state_slots, pool_within, pool_for) — deleted in F with Budgets.slots (load.rs:20), Residency.kv_seats/device_kv_budget (load.rs:69,79).
- Verify: serde round-trips, Lane::validate_for with new fields, degenerate ChunkFacts reproduce every PoolFacts count, clippy.

## 2. Runtime
### R1 slot allocator (new runtime/src/store/slots.rs, store/segments.rs; owns nothing else)
SlotPool: free-chunk bitmap (lowest first, like store/pool.rs:17-18); per chunk {shape, used, free-slot bitmap}; per-shape partial-chunk lists; per-slot refcounts; Holder::{WorkingSet, CacheEntry(Hash256), Snapshot}; freed slot recycled after epoch of last fire that touched it (reuse pool.rs recycle_after_epoch / retire_through semantics); empty chunk -> free list. Budget: state_chunks + act_chunks(rung) <= committed <= total. Segments: emptiest_segment(), plan_evacuation(seg) -> Option<Evacuation> (whole chunks into free chunks of lower segments), commit_ahead(bytes) (replaces COMMIT_AHEAD_BYTES/STATE_AHEAD, extent.rs:23-28). demand() -> {live, wanted} at slot granularity.
Verify: property tests (refcount conservation, no double hand-out, epoch reuse, evacuation empties a segment), degenerate perf with 1M single-page chunks.
### R2 wire stores onto SlotPool, degenerate (store/kv.rs, store/kv/*, store/rs.rs, store/rs/*, store/registry.rs, delete store/seat.rs & store/extent.rs, pipeline/fire.rs slot stamping, planner/pool.rs, grant.rs, holdings.rs)
KvStore pool fields -> SlotPool list shape (page_table keeps logical ids); RsStore slots/pages/host pools/refs/extent/pages_at -> fixed shapes; ring store (rs.rs:278-288) -> ring shape; checkpoints (rs.rs:238) and KvStore.cache (kv.rs:118) -> CacheEntry holders. Delete SeatBook + notify (registry.rs:42-43,127-128; pipeline/fire.rs:1709-1760). Extent -> segments.rs driven by governor. PoolPort (planner/pool.rs:11-72): device_stats/rs_stats/grow_rs -> chunk_stats/grow. RsUnits (rs.rs:101-143) -> per-shape slot demand -> chunks.
Verify: store/kv/tests.rs, store/rs/tests.rs, planner/starvation_race_tests.rs green; e2e bitwise same-process on Qwen3 and a GDN hybrid: same tokens and same page/slot ids as main. Risk: kv.rs 2572 lines + page_table.rs 2219 — swap only the id source.
### R3 retention to PLEX (planner/eviction.rs, acquire.rs, plan.rs, handle/table.rs, handle/mod.rs, server/handler.rs, bootstrap.rs:278-353, planner/error.rs:87)
Delete the six self-drop paths, state_room, rs_seat_cap cap/override. Short -> waiter parks and raises tier_pressure(Device); released by PLEX discard / install-retention, or governor growth. Suspend moves only private slots (shared stay).
DECIDED reclaim_idle (planner/pool.rs:162-169, residency_planner.rs:130): runtime may reclaim on its own ONLY entries whose PLEX retention key <= 0 ("may drop"); anything else -> tier-pressure.
Verify: deleted symbols gone; one tier-pressure per band crossing, waiter served after fake PLEX discard; hybrid suspend keeps checkpoints. Over-admission risk without rs_seat_cap: planner charges fixed-shape chunks (2 sets per ws) at admission.
### R4 tier-pressure plumbing (runtime/policy/service.rs, manifest.rs, new runtime/policy/pressure.rs)
Event::TierPressure(tier, band) -> Edge::TierPressure(WORKER) with band hysteresis (like controller/src/state.rs:161). Byte attrs on Subject::Worker: {device,host}-{total,state,live,wanted,act}-bytes, replacing slot attrs (service.rs ~1003-1020). No new verbs. MUST ship a built-in default responder when no worker policy is installed (drop lowest retention key), else parked waiters deadlock.

## 3. Governor — G1 (governor.rs, governor/run.rs; macos.rs/psi.rs unchanged)
run.rs:42-55 extent.asked()/split() -> segments.demand()/retarget(target): target bytes -> ceil(target/segment_bytes), floor = one ws at declared context + act_chunks(smallest rung). Demand{live,wanted} (governor.rs:109-114) from SlotPool: live = ws + snapshot slots; wanted = live + cache holders + waiters' shortfall. Limits.shrink_step (run.rs:26-30) = max(segment_bytes, ..) or u64::MAX for Fixed backends; constructor asserts shrink_step >= segment_bytes. Shrink = evacuate emptiest segment then release; no room -> tier-pressure, keep segment, never drop. Critical (governor.rs:203-206) -> floor target + smaller activation rung (replaces yield_scratch/narrow, extent.rs:236-243, scheduler.rs:193, frame.rs:3 NARROW_WAVE).
Verify: adapt governor tests (governor.rs:316-566); new: shrink invariant, Critical lowers rung, no-room shrink emits pressure without dropping (fake SlotPool).

## 4. Backend pool switches (after degenerate kernels verify; one package per crate)
- CUDA (BC): one VA reservation at load; state segments from the bottom (cuMemCreate/Map, multiple of C at 2MiB granularity); activation at a fixed top sized for largest rung. Commits via PoolOp::CommitSegment between frames. Evacuation = batched D2D (cudaMemcpyBatchAsync on 12.8+ or a copy kernel) + stream-ordered table update; graph-key assert on pool base; activation graphs per rung. Delete after verify: store.rs:14 re-export, :423, kept_slots/keep_slots (:556-558,908-910), fit_the_card slot cutting (:1186-1233), max_state_slots message (:1241-1248); serve/load.rs:115-125,352,895,1281-1286,1696; chip.rs:81-84,279,314 + profile/src/search.rs; api.rs:603; Paging.slots.
- ROCm: same crate, feature rocm, one flat allocation, one segment, Dedicated governor, no evacuation.
- Metal (BM): one sparse buffer VA for the pool replacing per-plane elastic::create (engine-metal/src/store.rs:286-325); segments via resource-state updates on a separate command buffer; blit evacuation; no-sparse devices stay flat. Delete api.rs:357-363,429; serve.rs:638 (pool_for); commit_state/release_state (api.rs:669-675; serve.rs:1277-1300).
- Vulkan (BV): segment buffers up to min(maxStorageBufferRange, maxBufferSize); address by buffer device address (core 1.2) = no binding-count limit; fallback bounded segment-array binding. Today rows are Buffer::zeroed (vulkan store.rs:126-215), limits at ctx.rs:924. Delete serve.rs:448,529,814; api.rs:350.
- wgpu (BW): segment = binding limit; request adapter max required_limits; kernels bind fixed N_seg segments within the 8/stage budget; attention splits per segment with LSE merge when needed; fixed shapes in low segments. FIRST turn the silent clamp (encode.rs:350-355) into an error. Today rows store.rs:123-212, growth copies whole slabs store.rs:698-760, limits ctx.rs:486,590. Delete serve.rs:479,571,842; api.rs:397,465.
- XLA/TPU (BX): today one donated buffer per plane + sink (store.rs:1-13,235-390; take/put :461-487). Tiled [chunks, rows, 128] u32 pool feasible only via Pallas (scalar-prefetched indices kernels-xla/src/mosaic.rs:17-23; VMEM u32 (8,128) tile reinterprets as bf16 (16,128); rows multiple of 16 bf16 / 8 f32; state_block store.rs:959-966 and lanes() :221-227 already pad to 128). Plain HLO bitcast_convert (hlo.rs:1049-1070) pads -> every state reader/writer must be Pallas. GATE: GDN decode state microbench tiled-Pallas vs per-row + HLO alias check that the donated pool is never defensively copied. Until then XLA stays degenerate fixed split by shape. Delete api.rs:215,352,394; serve.rs:416-417,542-545; sink_slot.
- Cerebras (BCe): host chunks, flat, fixed at load. Delete api.rs:363, serve.rs:471, spatial/engine/mod.rs:232; sink -> reserved chunk.
- Worker (BWk): remove max_state_slots (worker/src/config/backend.rs:12,40,58,94,154,186; backend.rs:142,180,631,674,726,770,871; embedded.rs:24,47,138), device_kv_budget, kv_seats; removed keys fail with a clear message via a `Removed` serde type (deny_unknown_fields only says "unknown field"); translate.rs:164-183 maps ChunkFacts.
- Verify per backend: degenerate bitwise first; then pool-switch e2e with forced growth, forced shrink+evacuation mid-decode (tokens identical), CUDA graph replay across evacuation; GPU under flock.

## 5. kv space removal — K1-K3
Reads: poem-ir trace.rs:202, value.rs:233-240 (Mask{space}, Geometry{space,kind}), value.rs:174-175 (WritePage/WriteOffset), check/classes.rs:595; poem forward.rs:26-54, :266-267, :372-392 (space 0 via kv_space() :582-588), :516-521 (space_of :590-595), record.rs; compiler stream.rs:398-490 (P2), fixture.rs; poem-exec feeds.rs:77,118, fire/fallback.rs:145-149, fire/fixture.rs; engine-cuda run.rs:69,1487-1488,1584-1592,1761-1765,1842-1877,2180-2224,2378-2399,2451,2526-2533, serve/prepare.rs:1269-1300, store.rs:483-484,929-958,1337, inputs.rs:168,262,517-704,807-816,995,1193-1297,1382, window.rs, dispatch/{custom,attn}.rs; metal/vulkan/wgpu/xla/cerebras run.rs; xla store.rs:411; cpu forward.rs; tests (poem-compiler the_{wan_2,ltx_2,z_image,flux_2}_rows_bake.rs; poem a_modulate/a_ragged/a_joint; metal a_copied_window).
- K1 (after P4-P6): engines key geometry/attention plans by slot shape (list|ring) not space index; bitwise.
- K2 (after P1+P2): poem emits Mask without space (All axis) and Geometry{shape,kind}; IR checker, poem-exec feeds/fallback; drop CacheRow::Kv.space; re-bake IR tests.
- K3 (after degenerate kernels address writes from slot table + position): delete GeomKind::WritePage/WriteOffset, write-geometry rewrite prepare.rs:1278-1300, KvWrites.pages/offsets (engine.rs:29-36).

## 6. Schedule
| Batch | Starts | Packages |
|---|---|---|
| A now | independent | E1, R1, G1-core (governor tests/invariant, run.rs behind a trait), R4, BWk-prep (Removed serde type) [+ wgpu clamp->error] |
| B after P1+P3 | ChunkRecipe exists | per-backend degenerate ChunkFacts report (api.rs/serve*.rs); R2; R3 (after R4) |
| C after P2 | per-row stream deps | K2 |
| D after P4/P5/P6 | new addressing | K1, then K3; Lane inputs -> tables/fixed (E1 follow-up + runtime/pipeline/fire*); G1-wire |
| E pool switches | after D per backend | BC, ROCm, BM, BV, BW, BCe, BX (after gate); each deletes its slot fitting after verify |
| F scaffolding removal | all verified | delete PoolFacts, fit.rs, commit_pool/commit_state, KvCopy/StateCopy, Budgets.slots, Residency.kv_seats/device_kv_budget, SeatBook remnants, slot attrs; BWk final |

Risks: R2 size; set_keeps (registry.rs:93) and prefix sharing semantics across shapes; evacuation under in-flight fires needs epoch fencing + stream-ordered table updates; no-PLEX deadlock (R4 default responder); over-admission without rs_seat_cap; wgpu binding budget; XLA defensive copy / HLO bitcast padding; CUDA graphs keyed per rung, top reservation covers largest rung.
