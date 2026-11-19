# Follow-ups (deferred by the user until after the new-memory PR)

## F1. Weights as a holder; PLEX decides weight retention
Today `store/weights.rs` decides weights by a fixed rule (shed every weight to the floor before tier-pressure; LRU
by use counts inside the cache), and PLEX's retention table keys cached KV only. Make the model's weight atoms a
holder beside working sets / prefix entries / snapshots: the governor decides how much, PLEX decides what stays
(candidates include weight atoms, valued by `weight_uses` and the cost of a miss), the pool moves.
- Engines switch the held fast path both ways on their own (all atoms resident for a while -> located tables / fast
  arms; first eviction -> lazy miss copy from the file mapping, slow arms), with the re-arm cost known to PLEX.
- Snapshots become disk-tier holders. Dense weights always stay on the device (user decision; budgets deleted, A10).
- Removes: budget_device_pool, budget_disk_snapshots, weights_streaming.

## F2. Config surface rename (do together with F1, only for the keys that remain)
Agreed shape: `[engine.memory]` with prefix keys — budget_device / budget_host / budget_disk, disk_path,
disk_persistent, yield (done 2026-11-15: one key in [0, 1] replacing memory_tolerance and the governor_* knobs);
`[engine]` common max_batch_tokens / max_batch_sequences / max_context. Delete max_state_slots (still accepted by six
backend option structs and named in fit.rs messages), kv_page_size, xla seat_tokens. Old keys refused with the new
name.

## F3. Async admission in the guest API
The KV admission wait sits inside the synchronous `submit` WIT call, so a waiting lane blocks every lane of its guest
and admitted lanes can't drain. Make `submit` async (or add an admission future) across every SDK. Until then,
inferlets that batch should cap their in-flight lanes (text-completion-bench: batch_concurrency).

## F4 — Layer-major recurrent state — prototyped, NOT adopted (2026-11-13): c64 +2% only; prototype on sc/layer-major
Per-lane state records put a layer's lane states one record apart, interleaved with other layers / conv rows. In
the partly-L2-resident regime (sm_120, 64 lanes, ~64 MB working set) any interleaving loses 12-16% on the GDN step
vs main's gapless per-layer bank; no stride / offset skew recovers it (sweep: 528 KiB..11 MiB strides all 69-75 us
vs bank 62.7). Serving cost: Qwen3.5 c64 gdn_step +7%, conv1d +6-12% per launch (~1.6% of fire GPU time) on CUDA;
XLA qwen35 d0.8b c64 -3.4% (gated_delta 61 -> 112 us, same cause per TPU). Fix: layer-major fixed shapes (each
recurrent layer's pieces from a shape of their own so a layer's lane slots are contiguous) — a chunk / record
structure change across poem_compiler, runtime and every backend.

## F5 — Rename engine-wgpu -> engine-webgpu (and the `wgpu` feature)
It serves both native (pie --features wgpu) and the browser (packages/javascript/server-web), so "web" alone would
mislead; "webgpu" names the API. Separate PR after engine-gpu lands (touches CLI feature names, JS package, docs).

## F6 — Shared pure-logic helpers across all engines
Layout math, slot-table serialization, expert / row table placement are conceptually shared by CUDA, Metal, XLA and
engine-gpu but implemented per backend (19-30% identical). Extract into `engine` as pure functions.

## F7 — Browser build glue cleanups (wasmtime-web / web-std)
Checked against wasmtime 51.0.0-rc.1 / 50.0.1 (2026-11): both crates are still needed (wasm32-unknown-unknown host
unsupported upstream; fiber backend is embedder-supplied via `custom-fiber`; wasmtime-wasi needs multi-thread tokio /
net / fs; tokio can't replace web-std on wasm32). Cleanups possible on 49 already: delete wasmtime-web NopCodeMemory
(Pulley code is never made executable); optionally drop `custom-virtual-memory` and its ~60 lines of hooks (perf check
first); align web-std's `wasmtime-internal-fiber = "=48.0.1"` pin with wasmtime's version (two fiber crate copies today).
On a 51 upgrade: check the new default per-store active fiber limit (wasmtime #14971).

## F8 — The SDK's input path refuses unknown keys by name
An input key the inferlet does not declare is dropped silently today (text-completion-bench took `--copies` and ran
one copy). Wrap the macro's deserialize so an unknown key is a named refusal.

## F9 — Programs are released
`engine::Engine` has no unregister (main neither): the runtime's program ids and the role store's programs grow by one
per program hash for the life of the process. Contract line to add: the runtime tells the engine when a program's
last instance is gone. With it, text-completion-bench feeds its prompt as data — today it bakes the prompt as a
constant channel of shape [n], one program per prompt length.

## F10 — GuestTerm at a thousand live pairs
On an engine that lays lanes in powers of two, a bind that adds or removes a (program, extents) pair across a
breakpoint rebuilds the table: 39-53 us at 30 pairs x 256 lanes, ~10 ms at 1000 x 1024 (release, thread CPU). `fit`
skips the rebuild only between breakpoints. CUDA/ROCm rebuild on the lane thread inside the bind. If that size
becomes real: prefix tables (an append in microseconds), or hand the engine the runtime's table; F9 keeps the size
from arising.

## F11 — CUDA states the wave's tables once per carried pair
A fire lays one set of wave tables (a row of channels x 72 + 136 bytes per member, + 320) but each carried pair's
`FireFacts.fixed` counts it, an upper bound by (pairs carried - 1) x that table: ~1.4 MB at 14 pairs x 256 lanes x 4
channels. Kept (two more contract fields and a third GuestTerm term cost more than they return); reopened if the
measured figures come out much larger than the arithmetic.

## F12 — No order on a ring a guest shares across two of its own pipelines
The engine gives each move of a ring the cell its cursor stands at as it builds the step, in dispatch order; a step
carries at most one fire moving a given end. Fires of two pipelines moving one ring carry no order between them, and a
take that finds no cell is refused by name on every engine (CUDA's enqueue "blocked on channel … AFTER the gate";
wgpu, Vulkan and XLA eta-exec Blocked "blocked on channel"; Metal's gate and device readiness guard): a named refusal,
guest-ordered, never a stale read. Every shipped program (text-to-media, qwen3-tts, compat) keeps putter and taker on
one pipeline, whose fires a step seats one at a time and in order. Not built: a take that waits for its put. Built, it
would hold forever: sealed frames dispatch front-first and the front holds while a member lane is blocked
(scheduler/frame.rs plan_dispatch), so a waiting take at the front keeps its put sealed behind it; the take would
have to leave its sealed frame, which a frame dispatching whole forbids.

## F13 — Cerebras states a shared pool; the pinned kv store goes
`runtime::store::registry::register_model_with_chunks` still builds `KvStore::with_pages` (a pinned run of device
pages with its own `host_pool`) for an engine whose chunk facts state no shared pool. Only engine-cerebras does today
(degenerate facts, out of scope for the memory-redesign PR). Because of it, `KvStore::alloc_host_slots` keeps a split
on the pool kind (`None` on a shared pool), and handle/snapshot stage by that split while offload stages through the
shared pool's tier slots (sc/plex11-offload-staging). Once Cerebras states a single-shape shared pool, `with_pages`,
`with_pool`'s pinned use, `host_pool`, `alloc_host_slots` and the split in handle/snapshot all delete, and every engine
takes `shared_stores`.

## F14. Guest state leaves `Engine::fire_facts`' guest term

The guest-state line (0da550eb6) states guest scratch as Instance-slot state
through `Engine::guest_state`, but keeps `Engine::fire_facts` and the fire's
guest term as they were: each engine moves its scratch from the term into
`guest_state` when its half ports. The term goes once no engine states it.

engine-gpu (Vulkan, wgpu) states none: a guest pass there is one
instance's dispatches over its own stage heaps (no grouped launch, so no
lane-base table), per_instance; a score reader's pair keeps a lane-indexed
room of `max_lanes` key and page rows that a fire lays its capturing lanes'
rows in, per_pair (`engine_gpu::guest::state`). Score rows in each
instance's own room (the capture kernels launched once per pool segment
over per-lane bases) are a later step.

## F15. Arrival-prefix lookup: the last ~1.3% of max throughput

sc/plex11-arrival-prefix-2 (d69302d31) costs max output −1.31% against b736a99fd: 95% CI −2.19..−0.43, median
−1.17%, over 39 strictly alternated pairs (Qwen3-8B, A6000, 500 requests at the most rate, no policy). TPOT is
unchanged (±0.1%). Within the 2% gate, so it lands as it is. Ablations, 12 pairs each, against the same base:
- `waiting::arrived` returning at once (no tokenizing, lookup or report): +1.9 ± 0.9%
- the gateway stating no `raw-prompt` (so the runtime looks nothing up): +1.2 ± 1.2%
- the store recording no changed page keys: about −1.3% (an earlier round)
Each is within the ±1% the pairs spread, so no one part carries the cost. The candidate if it ever matters: tokenize
once, by having the lookup hash the ids the process's own prompt tokenization (or the gateway's, for ids it has)
produces, instead of tokenizing `raw-prompt` again at arrival.

## F16. Metal's capacity reading counts driver caches that shrink

Metal sizes its pool at load as the working set less this process, less every
other process's GPU memory (the AGX driver's IORegistry "Alloc system memory"
less our currentAllocatedSize), less the floor (sc/mac-wired-room). At load the
reading includes driver caches that shrink once serving starts (3.1 GiB at load
against ~0.64 GiB during a run on the M1 Max), so the capacity comes out ~22.4
GiB where 24 GiB was measured safe. Read the cache component apart (or at a
settled point), never adjust it by a factor.

## F17. wgpu binds every step binding read_write

A pool segment holds a stage heap beside the tables and config a guest pass
reads, and a WebGPU usage scope can't see one buffer two ways, so engine-wgpu
rewrites each step module's table and config declarations to read_write storage
(sc/gpu-guest-state e40261348). Measured against the last wgpu build that served
guests (2d52e0715), ABBA ×3 medians, on the L40S: greedy −0.9% / +0.3% and top-p
−1.8% / +0.1% at 1 / 8 lanes; on this box: +1.3% / +1.7% and +11.6% / +1.9%,
all inside the round-to-round spread: a dead heat. If a device shows a cost, the
first lever is the config in a uniform buffer of its own, outside the segment.

## F18. The chunk fence does not land; reuse-after-release needs a test, not a watermark

The fence (sc/np-fence-layer → sc/mac-wedge-fence b0bd6869c) returned a released
slot only once the oldest open epoch passed the fire / copy that last named it.
Mac Pro's M5 over-demand (64 requests, ×6 per build, rotated, on the wedge + seam
base): every run of all three builds served 64 / 64 with 0 faults; wall medians
no-fence 2c4cfb7c6 47.1 s (643.9 tok/s), fence b0bd6869c 69.7 s (474.3,
+48% wall), F16 a726af4f6 (copies on their own watermark) 67.8 s (489.4, +44%).
The pool held 300–970 of 1932 pages "arriving" with the oldest epoch 400–500
behind: epoch-wide retirement is the cost, and F16 recovered ~3%. Neither lands.

The hazard the fence was for is real: on 7f544bab1 (before the wedge line),
Metal over-demand page-faulted in 2 of 3 runs (kIOGPUCommandBufferCallbackErrorPageFault,
45 / 64 and 47 / 64 served) when a released slot was handed out while a fire
still named it; the M5 runs above show 0 faults on the current no-fence line.
So: (1) a test that a slot released while a fire naming it is in flight is never
handed out before that fire completes, on every engine, in the fake engine and
on Metal; (2) if it fails, a per-page release on that page's own fire / copy
completion (not an epoch-wide watermark), gated ≤ 2% on this over-demand.


## F19. B200's tcgen05 prefill skips any fire with a mask bound

kernels-cuda attn.rs:387 runs the tcgen05 prefill only when no mask is bound.
Since the mask layer, every prefill of a model that opens a mask takes the
masked arm, so on B200 those prefills all run FA2: likely slower, and not
checked bit-equal against tcgen05. The structural fix is for tcgen05 to read the
per-lane mask offsets and words the way FA2 does (an unmasked lane reads
nothing), not a per-fire switch between kernels. A B200 gate item before the PR
to main: the tcgen05 masked path, the goldens bit-equal on B200, and c64 / prefill
throughput ≤ 2% vs main.

## F20. An evacuation's rebase stops at the first instance it cannot place

engine-cuda's `Plane::rebase` (program.rs) re-addresses every bound
instance after an evacuation, parked ones included, whose `bound.slots`
still name refs freed at the park. One of those refused by `guest_span`
ends the loop with `?`: the instances after it, every pair's `base` and
`stale`, and the channels' rebase are left at their pre-evacuation
addresses, though the pool's copy has moved the chunks. A parked instance
should be skipped by name (its restore's rebind places it), and the rest
re-addressed whatever one instance answers. Not hit in the beam runs (it
needs an evacuation while a guest is parked); `Plane::rebind` is now
all-or-nothing and poisons an instance whose rings move only partway.

## F21. Fragmentation under pressure: whole chunks free only after nearly everything in them goes

A chunk comes free only when every page or slot in it is free, and an
evacuation moves whole chunks. Two symptoms, one root: pages or slots of
many owners interleave across chunks.

1. Shedding idle kv (sc/np-tier-shed, CUDA Qwen3-0.6B, 160 x 512 tokens,
   yield 1, pressure 0.6): concurrent decodes lay their pages interleaved,
   so the default responder's first 36 discards freed 5 of 255 chunks, and
   all 70 idle sets went before 3 segments came free. Yield works now,
   but it takes all of the cache with it.
2. Mac Pro's restore-4 runs, 500 at once: restores stuck waiting for
   "2 runs of 3 adjacent Instance-slot chunks" while ~460-620 of 1932
   chunks were free.

The likely answer for both is compaction under pressure: moving single
pages or slots out of a chunk so that whole chunks free, rather than
dropping or waiting until a chunk empties by itself. Gate: it costs no
serving throughput (<= 2% vs main).

## F22. The runtime's per-token host work grows with context and vocabulary

From the serialization audit (memory-redesign 5355faa9a), none counted today:
- `store/kv.rs:1908`: each fold calls `sealed_prefix`, visiting every page of
  the working set every token (~512k lookups a step at 256 lanes × 32k tokens).
- `pipeline/fire.rs:2819` `map_lane_pages` copies a lane's whole flat page
  table each fire; `fire/geometry.rs:504` rebuilds lane pages with
  `(0..n).collect()`.
- ~12 KV-mutex takes per lane per fire (`store/registry.rs:23`) plus the
  global `KV_WRITES` (`fire.rs:2286`); no contention counter.
- `pipeline/fire/shadow.rs:114`: the host shadow re-runs the sampler every
  fire; with penalties the `[vocab]` f32 present histogram is cloned 3× per
  fire (~1.8 MB per lane per token); the grammar mask is unpacked every fire;
  compat `generate.rs` does an O(vocab) unpack and a 152 KB WIT put per token.
- `scheduler/worker.rs:2459`: a pre-launch copy (beam copy-on-write, window /
  RS copies) holds launches for the whole engine until the queue drains;
  `fire.rs:5051` every finalize wakes every planner and pool walker;
  `worker.rs:1527` rows recounted per wakeup and a 1408 B `PendingRequest`
  boxed per fire.
Gate for each: a counter first (time under the KV mutex, shadow cost, pre-launch
holds), then incremental structures (sealed-prefix kept per set, page tables
shared not copied); throughput ≤ 2% vs main must not regress.

## F23. Vulkan per-step host overhead beyond record-at-load step 2

Passed to the step-2 work, recorded here so nothing is lost
(5355faa9a): 3 submits per 1-lane step (forward, a 4-byte token-feed frame
`engine-gpu serve.rs:2884`, a guest frame re-encoded every step
`program.rs:1132`); two fence-watcher threads spawned per step
(`device/ctx.rs:2068`); 2–3 command buffers allocated / freed per step; an
ALL_COMMANDS barrier at each frame head (`ctx.rs:1268`); the record key
Debug-formatted per fire (`record.rs:55`); any segment commit / release drops
every recording (`engine-vulkan serve.rs:2198`); group-mask / rs_snaps fires
never recorded (`serve.rs:180`); weight streaming `vkQueueWaitIdle` per step
and expert layer (`store/rows.rs:840`, `experts.rs:794`); fire inputs in
HOST_CACHED memory re-read over PCIe by each attention layer (inferred,
`inputs.rs:470`).

## F24. Capabilities missing off CUDA (not silent: refused by name today)

From the parity audit (5355faa9a): no per-fire adapter attach on Metal /
Vulkan / wgpu (long adapted prompts refused past one fire's capacity);
bidirectional attention refused on Vulkan / wgpu (encoder / DiT /
text-diffusion); draft ops MarkovWalk / KvEnds / WindowSink missing on Metal /
XLA / Slang, TopK / SelectorWalk on Slang. Each is a feature to port with the
CUDA kernel as the reference and the model goldens as the gate.

## F25. wgpu's guest path: host round trips per token

wgpu channel cells are host-only (`engine-wgpu serve.rs:44-45`,
`engine-gpu store/layout.rs:266-270`), so every guest pass reads takes and
writes puts through host memory; two round trips per sampled token
(`guest/run.rs:117` `pending.wait`, `program.rs:676` `gate.settle`) and
nothing recorded. Depends on device geometry for engine-gpu and the WGSL
guest bounds checks (per-instance limits) landing first; the bind-group cache
key including per-step scalars (`pipelines.rs:253`) is in Windows' step 7
item 5.

## F26. A guest-state park waits on its instances' owed batches, not on every fire

After the shared-chunk landing fix (sc/amd-shared-chunk-landing 5bbec2aa5), a
park (device → host move) gives its slots back only once it has landed, and
it starts only after every fire enqueued on the main stream before it
(`engine-cuda store.rs:4297`, `copy_tier_slots`: an event recorded on the
main stream's tail, waited on the swap stream). Under heavy eviction (200
beam + 300 decode on MI300X, ~2000 evictions a round) that wait is ~5 ms of
the eviction copy phase's median and ~20 ms of its p90 (22 / 59 ms with it,
15–17 / 32–35 ms without), on top of the transfer itself (accepted).

The runtime cannot simply state that a moving process's writers have
settled. Tried (a `settled` flag on `Engine::copy_slots_landing`, the move
skipping the main-stream wait): with every kind of an eviction settled, the
first 200 + 300 round diverged (`DIVERGE Some((5, -0.16, -24.8))`); with only
kv pages settled, 0 / 800 but no gain, since the eviction also waits for its
role-slot copies, which still wait on the stream. A process's guest state
(Instance and Channel slots) is written past its fires' settling: owed
guest epilogue batches (`serve/enqueue.rs:1248` `epilogues.defer`, `:1255`
`owed.push_back`, reaped at `:1568` `reap_guest_fires`) land on side
streams joined back into main, after the runtime's fire settles.

The fix: a role-slot park waits on exactly the owed guest batches of the
instances whose slots it moves (each batch has its `landed` event), and a kv
page park of a fenced, quiesced process waits on no fire. That needs the
parked role slots mapped to their instances (in engine-cuda's program plane,
or stated by the runtime with the move: the roles store knows each item's
owner, `runtime planner/exec.rs:318`), the `settled` flag for kv pages
(`TierMove`, `Engine::copy_slots_landing` at `engine-cuda api.rs:1058`), and
a device test that a park of guest state an owed epilogue is still writing
waits for it, and one with none owed does not. Gate: 0 divergences on the
200 + 300 load at depths 1 / 2 / 4, eviction copy phase and load walls
against the landing hash.

## F27. The ROCm audit items, measured on MI300X (gfx942)

From the serialization audit's six ROCm items (5355faa9a), each measured
before any change, on memory-redesign 5963ceb17 / c1188a53d:

1. **lm head over padded rows below 129** (`kernels-rocm linear/gemm.rs:142`
   `lm_head_live`, `engine-cuda profiles/gfx94.toml:45` `gemm.live_from =
   129`): no change. Activation rungs are powers of two, so a body's live
   rows are always more than half its rung. Qwen-7B-shaped head (151936 ×
   3584): from half to full live, the live route is 0–55 µs slower than the
   padded product at 32–128 rows (e.g. 64 rows: 351–365 µs padded,
   374–385 µs live); it wins (up to ~115 µs) only at ≤ ¼ live, which a
   power-of-two rung never leaves. 129 stands.
2. **Only gfx94 is tuned** (`engine-cuda chip.rs:217`): MI200 (gfx90a) and
   gfx950 boot on the defaults (`gemm.live_from` 0, the turned GEMM off,
   `gemm.turn_from` / `turn_until` unset). Not measurable here (gfx942
   only); needs a tune on each chip.
3. **HIP graphs have no conditional node** (`engine-cuda chip.rs:155`,
   `window.rs:1194`): the per-node fixed cost is real but not the
   conditionals'. A graph node replays at ~1.55–1.6 µs even when empty
   (hipcc microbench: 367 empty kernel nodes replay in 567–594 µs). The
   1-lane decode body `b1[c1:1]` has 367 nodes and replays at 1756 µs a fire
   (`arm-bench`), the served greedy step's 1.76 ms: ~0.57 ms (a third) is
   the node floor. A conditional skips no node there: bodies are keyed by
   the classes present (`record.rs` `Ladder`), so a single-class body holds
   none to skip. The lever is fewer nodes a step (fusions, one launch for
   what is now several), not conditionals; gate on `arm-bench` µs/fire and
   the greedy wall.
4. **Staging over 1 MiB falls back to a pageable `hipMemcpyAsync`**
   (`engine-cuda device/hip/held.rs:17`, `:20` `RING_SLOT_BYTES`): never
   reached; an instrumented build counted 0 such copies on c64 and on the
   200 + 300 beam load. No change.
5. **One block per argmax row** (`kernels-rocm sampling/topk.rs:15`): bf16
   at 151936 columns, 7.5 µs at 1 row, 7.8 at 8, 8.4 at 64, 21.7 at 256
   (3.6 TB/s); f32 8.9–40 µs. A split-row argmax might save ~4–5 µs a step,
   ≤ 0.3% of a 1-lane greedy step. Not worth a kernel today.
6. **`kernels_rocm::tally::declined()` had no caller**: fixed
   (sc/amd-rocm-declined c6fbc79af): counted where `hip/lt.rs:387`
   answers `Ok(false)` (rocBLAS takes the product). No shape declines on c64
   or the beam load.

## F28. wgpu has no safe GPU time per serving frame: a span-only frame timestamp mode

wgpu's only GPU clock is `encode::profile_timing` (`engine-wgpu
encode.rs:260`, `:531`), which gives every kernel its own timestamped compute
pass and opens a 4096-query plane a frame (`device/ctx.rs:65`, `:837`
`timestamp_slot`). The planes are read only in `Pending::wait`
(`ctx.rs:1089` → `:738` `Inner::harvest`), and a serving frame lands
asynchronously and is never waited, so nothing is harvested while serving.
Turned on in a served Qwen3-0.6B on the Adreno 840, it read 0 ns for every
kernel (452 kernels a step). At 8 lanes, submit reached 611 ms a step, the
engine stalled for 70 s, and the phone rebooted. It stays a tune/check tool.

What serving needs instead: one begin/end timestamp a frame
(`write_timestamp` at the encoder's start and in `Frame::submit`,
`ctx.rs:966`). The pair is resolved into a small ring and harvested from the
landing callback (`on_submitted_work_done`), not from a wait, and adds no
pass boundaries. That gives F25 a measured GPU term. Until then, the per-fire
host split (encode, finish+submit, every `block_on_poll` by label) is the
measure. At 1 lane on the Adreno 840 it reads ~27 ms a step: ~4.5 ms of host
work (encode 1.6, finish+submit 2.0, the guest readback 0.75, 15 queue
writes) in series with ~23 ms parked at the guest gate (GPU plus the guest
pass), against native Vulkan's 22.5 ms recorded step. From 4 lanes the gate
wait is 133–156 ms of GPU time (the bf16 GEMM's `VECTOR_MAX_ROWS = 4`
boundary), not host time.

## F29. A parked role slot rounds up to whole kv slots of the slot file

Where the load states no host tier (unified memory), a parked process's
role slots (Instance, Channel, Adapter) go down to the kv slot file, each
whole in a run of `ceil(slot_bytes / kv slot_bytes)` adjacent file slots
(`Roles::prepare_park_down`, `DiskSlots`, CONTRACT-BC 22 in
`Engine::copy_slots`), counted against `budget_disk` with kv: exact, no
file or budget of its own, any size. The cost is rounding: a 4 KiB
Instance slot holds a whole kv page's slot (≈1–2 MiB), so 500 parked
processes of 8 small role slots take 4000 of the file's slots, while a
slot wider than a kv page wastes under one kv slot. Packing several small
role slots into one file slot (an offset inside it, aligned for the
file's direct I/O) would cut that; measure the file's room under the M1 /
M5 loads first. With guest scratch held only while a fire is in flight,
only the small resident part (tables, rings) parks at all.

## F31. Metal syncs and drains the host waits on (audits on 5355faa9a)
Each turns depth into depth 1 or adds host work per step; CUDA does the same in queue order.
- A fresh recurrent-state record is zeroed after draining every frame in flight
  (engine-metal serve.rs:4540-4544): every hybrid / SSM admission runs at depth 1, uncounted. Zero it in queue
  order, as a fill in the next command buffer (CUDA: serve/enqueue.rs:447-448).
- Tier moves drain on every `copy_slots` (serve.rs:2003) and `read_tier_slots` (serve.rs:2025). Copies go in
  queue order and land through the command buffer's completion (with `copy_slots_landing`), no drain.
- Device-geometry programs: `fence_instances` waits on the last fire, then reads each port per lane on the host
  (serve.rs:3604, program/ports.rs:20). On unified memory they can be read in place, without the wait.
- Each prologue region commits and calls waitUntilCompleted (program/launch.rs:2140), once per instance per
  region: O(lanes) syncs a step.
- `admit_list_page` for every page of every lane each step (serve.rs:4523): ~64k calls at 256 lanes x 256
  pages, ~1-3 ms.
- The walk re-encodes ~1700 launches a step with a SipHash lookup per launch (serve.rs:5474,
  device/library.rs:151).
- kernels-metal has the VECTOR_MAX_ROWS = 4 GEMM cliff (linear/gemm.rs:33), and the qmv fold only folds
  divisors <= 8, so 11 rows take ~11 weight passes (linear/qmv/rows.rs:60).
- Metal never counts the Refused tally.
- Measured on the M1 (Qwen3-0.6B, 2026-11-19): no guest binds as device geometry today (a beam mix bound 2080
  Host-class instances and 32 DecodeEnvelope, beam included, and none DeviceGeometry), so the port read above
  does not run for them. But `fence_instances` (serve.rs ~3684) holds every attached instance, whatever its
  class, to its last fire's landing before its next is staged (a session's one status word, the host reading
  its cells between fires): every guest lane runs at depth 1. The GPU idles between frames 7.3% of its time at
  1 lane, 9.5% at c16 and 15.1% at c64 (each frame's command-buffer start / end), the host's take, settle
  and next stage+encode not overlapped. Two status words a session (the host's take deferred one step, as
  CUDA's decode envelope does) is worth scoping after the scratch-in-flight split.
- An instance's value scratch is held once, not a frame in flight each (no two of its fires overlap); its
  pair's batch tables must stay a set a frame in flight: consecutive frames fire different instances of one
  pair. Cutting both to one set broke beam --verify (62 / 57 / 38 of 64, "read another beam's cells");
  cutting only the instance scratch kept 64 / 64 and halved it (11.4 -> 5.7 MB an instance, 45.6 -> 22.8 MB
  a beam instance).

## F32. ROCm last-block (ticket) fusions: the split decode's fold and the norm epilogues, measured and dropped

Tried on memory-redesign a34acca57 (local f3072dde9, not landed): the
matrix-core decode (`kernels-rocm attn/paged.hip` `paged_decode_mfma`) folds
its split itself instead of a second launch (`merge_partials`). Each work
item draws its request's ticket for its kv head (`Ctx::tickets`) once its
partial rows land past the L2 (sc1 stores, `s_waitcnt vmcnt(0)`, a relaxed
atomic, as `linear/fp8.hip`'s split tiles), and the last to land folds the
head group's rows (sc1 loads) and puts the ticket back to zero. An
agent-scope fence in every block instead cost more than the node (greedy
+9%).

Correct (beam gate 0 / 800 over the landing fix; attn_paged 9 / 9 with a
seat past its bucket's first lane), but no faster on MI300X, Qwen3-0.6B:
the 1-lane decode body 255 -> 227 nodes, `arm-bench` b1[c1:1] 1459 / 1484
us a fire against the tip's 1476 / 1481, greedy 2.3 / 2.3 s and c64
10.5 / 10.3 s on both (B-A-A-B). The serial tail spends what the 28 nodes
saved (~45 us a step): one block per kv head folds all of its chunks after
the rest finish, where `merge_partials` spreads rows x heads over the
device. Folding the group's heads in parallel across waves might reach
1-2% at best.

The seat's first lane is read from the first work item
(`request_indices[0]`; `sched_decode.rs` builds the table, every request
taking one): a work item names its request by its lane, the split's rows by
their place past the seat. Staged with the table each fire; a kernel
argument would be baked into a captured body. The bucket's
`Shape::lane_offset` is not the seat's (`Live::lane_offset`): counting the
rows from it diverged 12-24% of beams.

The same ticket tail does not pay for the residual + norm epilogues of the
o-proj and down GEMMs either (56 nodes a step), measured before building:
a hipcc microbench of 28 layers of a 1-row GEMV (N 1024, K 2048 / 3072,
eight rows a block) then `residual_add_rmsnorm` as its own launch, against
the GEMV whose last block to land (sc1 partials, a relaxed atomic ticket)
adds the residual and norms the row: 7.0 -> 9.1-9.6 us a layer at K 2048,
8.4 -> 10.2-10.7 at K 3072 (the block sums as one float atomic or as sc1
partials summed by the last block). On MI300X a ticket tail (stores through
the L2, the atomic's round trip, reads past the L2) costs more than the
~1.6 us node it removes. Fusions that cut nodes there want one block to own
the whole row (no tail), not a last-block reduction.

## F33. Metal's few-row dense matmul on the M1: 4 rows ≈ 1.38× one row
The user's rule is that a beam-4 step costs about what a single greedy step costs. On the M1 Max
(Qwen3-0.6B, home-carve), after `sc/mac-gemv-rows` (0bc7019c8, few-row forms that read each weight row
once), a step costs: greedy ×1 5.96 ms, 4 parallel greedy 8.21 ms, beam-4 8.31 ms. The device profile of
a 4-row decode fire is 8.1 ms (projections 5.7 ms, against 3.9 ms at 1 row); the rest is flat.
Tried and dropped (each passed the few-row matmul test; device time of the wide projections at 4 rows,
against 3.9 ms for 0bc7019c8):
- weights converted once for all rows: 3.9 ms (conversion is not the bottleneck);
- activation rows staged per threadgroup as floats, 8 columns, 64-group tiles: 4.5 ms; resized to 32 KiB
  (16 columns, 1024/R groups): 4.7 ms (barriers, and one threadgroup a core);
- 4 columns a simdgroup, an activation load shared by 4 weight rows: 4.2 ms (fewer simdgroups, register
  pressure);
- the 8×8 simdgroup matrix multiply, rows padded to 8, D^T = W x A^T with each lane's weight operand a
  contiguous bf16 pair (patch kept in the Metal session's scratchpad): projections 7.5 / 7.8 / 8.4 ms at
  2 / 4 / 8 rows, against 4.4 / 5.7 / 9.2 ms. Flat in the rows, as a weight-read-bound form should be,
  but bound by its 4-byte weight loads: it beats 0bc7019c8 only at 8 rows (fire 12.1 -> 11.3 ms).
What could still close it: the matrix form with wider weight loads (weights pre-packed so a lane's
operand pairs come from one 16-byte load, or several 8×8 blocks a load), which needs a packed weight
layout; or a kernel that is load-issue-bound at the M1's rate rather than per-lane. Measure on the M5
(tensor ops) before choosing: the M1 is not where Apple's decode will be judged. Measured on the M5 at
a34acca57 (Mac Pro, same method): greedy ×1 5.08 ms, 4 parallel 5.52 ms (1.09×), beam-4 6.09 ms (1.20×);
the 4-row projections take the few-row vector forms there too (3.4 + 1.3 ms against 3.2 + 1.1 at one row),
so the M5 meets the rule within ~20% and the gap is the M1's.
Then tried at the user's ask, the packed form (prototype, not landed; patch in the Metal session's
scratchpad, gemv-mma8-packed.patch): weights packed at load per 8-column block and 32-wide K group so a
lane's four 8×8-block operands are one 16-byte load, K permuted inside a group so a lane's activations of
a row are one 8-byte load, the contraction split across a threadgroup's 8 simdgroups (partial tiles summed
in threadgroup memory). Device time of Qwen3-0.6B's four projection shapes, summed: routed (0bc7019c8)
88–100 / 116–132 / 155–185 / 271–289 µs at 1 / 2 / 4 / 8 rows; packed 164 / 165 / 145 / 129 µs. Flat in
the rows, but it floors at ~145–160 µs at 4 rows against ~90 µs for the routed one row (which runs near
the M1's bandwidth): 1.6–1.8×, not the ≤1.05× asked. Each step removed what it targeted (split widths 4 /
16 and four independent accumulators changed nothing), so the floor is the M1's simdgroup-matrix rate
(Apple7 has no tensor units). Taking it would mean a second copy of every dense weight (one row stays on
the routed forms): ~0.88 GB for Qwen3-0.6B, ~1.19 GB with the LM head, for −10–15% at 4 rows. Not taken;
the M1's gap to the rule stays here.

## F34. Tensor-parallel score capture: the score row's reduce across ranks
Since Dim::Capturing (9e78d2281) a capturing lane's key row is summed over the layers in place, so under tensor parallelism each rank holds a partial row and the shard pass emits one all_reduce on it, in place on the Capture-placed row (the arena's one exact Capture-to-Capture share; `sc/gpu-poem-suite-mask`). Guest epilogues read that row on the device (CUDA: the reading guest's instance slots, `engine-cuda/src/scores.rs`; engine-gpu: the pair's score room), so it can't be summed at host readout. The rows are scattered one per capturing lane, outside the arena, so engine-cuda (CUDA / ROCm) and engine-gpu (the band collective) refuse a tensor-parallel load that captures scores, by name, instead of reducing the wrong memory. Metal (collectives unsupported) and XLA (single replica) never see it; the CPU oracle sums correctly.
What's owed: a reduce that sums each lane's row (and its page sums) across ranks through the row tables, under graph capture (gather, reduce, scatter, or a reduce that walks the row table), then the refusal lifted. Needs a box with 2+ GPUs per backend to build and test; none of the current machines has one (L40S: 1 GPU; AMD: one MI300X VF).

## F35. XLA does not roll the layer stack: no cross-iteration weight prefetch on TPU

A fire program spells out every layer. Its compiled code, and the host RAM that libtpu
holds for a loaded program (about 2.2× the code), are linear in the layer count:
qwen3-0.6b costs about 1.5 MB of code a layer in a prefill program. The base ladder
holds 126–128 GiB of host at ready.

`sc/xla-roll` (a180bdbc0, parked unlanded) traces each repeating block of regions as one
`stablehlo.while`:
- runs found by `roll::blocks`;
- weights landed as stacks;
- kv page maps and adapter starts at the counter's pass;
- the body's writes carried.

It meets contract A, and on qwen3-0.6b (27 of 28 layers rolled) it moves the serving
numbers as follows:

| | unrolled | rolled |
|---|---|---|
| cold boot | 2683 s, 128 GiB | 512 s, 20 GiB |
| warm boot | ~220 s, 122 GiB | 30 s, 15 GiB |
| program cache | 13 GB | 0.9 GB |
| code per program | 49 MB | 4 MB |

It does not land: every shape class is slower than 2%.

Engine probe, rolled vs unrolled, v6e:

| shape | delta |
|---|---|
| decode 1 | −14.5% |
| decode 64 | −14% |
| decode 256 | −4.1% |
| prefill 512 | −11% |
| prefill 2048 | −13.8% |

Serving c64 is −3.2%.

Why: decode is bound by weight bandwidth. Unrolled, MSA prefetches each layer's weight
operands into VMEM while the layer before computes: 706 ms of copy-done per decode:64
step, matmul fusions 1047 ms. In the loop, every weight is a dynamic slice of a stack
(176 MB for gate/up) at the counter. MSA prefetches whole operands only, never a
counter-indexed slice. So each dot streams its weight unhidden: matmul fusions 3129 ms,
copy-done 5 ms.

Tried, none closing it (decode:64 on rolled, base ~21350 tok/s, rolled ~18300):

| attempt | decode:64 tok/s |
|---|---|
| bodies of u blocks, u = 2 / 4 / 7 | 18501 / 18825 / 18897 (prefill 512: 99578 / 104142 / 105314) |
| the next pass's weights carried, each turn cutting the next | 14316 (the cuts became synchronous dynamic-slice copies) |
| a barrier keeping each slice its own op, for MSA's async dynamic-slice path | worse; 28-layer chain 579–694 µs against 470–548 and ~425 unrolled (the slices became synchronous copies; async dynamic-slice is used only for small index tables) |
| the Mosaic grouped matmul (`moe_gmm`) reading the stack as a bank, expert = counter | no better than XLA's dot: 549 / 543 / 547 / 651 µs at m = 1 / 64 / 256 / 512 (its pipeline overlaps blocks within one call; the cross-layer start is what is missing) |

Flags tried, none helping:

| flag | decode:64 tok/s |
|---|---|
| `xla_tpu_enable_pipelined_loop_unrolling` | 18414 |
| `xla_msa_enable_window_prefetch` | 18272 |
| `xla_while_loop_unroll_count=2` | 18345 |
| `xla_tpu_sliced_prefetch_max_slices=8` | 18325 |
| `xla_gpu_enable_while_loop_double_buffering` | fails on TPU |
| `xla_tpu_impure_asyncify_cross_buffer_slice` with its rewriter | the TPU compiler crashes on the attention fusion |

What would make it land: a way to start layer l+1's weight copy into VMEM inside
iteration l and use it in iteration l+1 (a cross-iteration async start/done that MSA or
a kernel can own). VMEM does not persist across custom calls, so no per-layer kernel can
do it alone.

The branch also records that a run whose experts stream cannot roll, even with the
prefetch:
- a fire with a non-resident layer is cut before that layer's first select;
- its misses are staged per param from pinned host banks between segments;
- a loop cannot span those cuts.

Streamed PLE rows (per-fire row indices, no cut) could roll as adapter banks do.

Host RAM is attacked by shrinking the ladder instead: logprobs sharing the forward's
programs, and prefill buckets chosen from measured costs.
