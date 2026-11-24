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

## F36. XLA prefill buckets: which rungs a model can drop is measured per model, not per device

The token rungs come from `Pad::ladder` (`engine-xla chip.rs`). They are the powers of
two up to 512, then `engine.pad.row_steps` steps between powers of two from 512 up (v6e:
4, i.e. 640, 768, 896, 1024, …). Each rung brings its prefill and mixed ladder shapes.

The rule (Leader): a rung goes if padding every prompt it serves up to the next remaining
rung costs at most 2%. Measured as t(next) / t(rung) on single-lane prefill fires (engine
probe), and on batched fires (several lanes beside decodes, a c64 step); both must stay
within 2%.

qwen3-0.6b on v6e, single-lane prefill, ms per fire:

| rung | ms | dropping it pads to | cost |
|---|---|---|---|
| 64 | 3.40 | 128 | +3.2% |
| 128 | 3.51 | 256 | +10.9% |
| 256 | 3.89 | 512 | +16.5% |
| 512 | 4.53 | 640 | +10.6% |
| 640 | 5.01 | 768 | +0.6% |
| 768 | 5.04 | 896 | +17.2% |
| 896 | 5.91 | 1024 | −1.0% |
| 1024 | 5.78 | 1280 | +17% |
| 1280 | 6.77 | 1536 | +16.1% |
| 1536 | 7.86 | | |
| 1792 | 9.04 | 2048 | +15.5% |
| 2048 | 10.44 | | |
| 2560 | 13.29 | | |

Only 640 and 896 pass. The model is weight-bound up to about 1024 rows and compute-bound
past that.

Why per model:
- `row_steps = 4` is itself a measured device decision. It came from Gemma-4-31B, where a
  512-token prompt plus its template ran a 1024-row fire, and the 128-row steps took cut
  Raw prefill toward vLLM-TPU (8.8k vs 12.8k tok/s).
- A compute-bound model pays nearly in proportion to its rows for the same padding.
- So the kept set is a property of (device, model).

What it would need:
- the kept set derived at load from a measured cost curve: the device's measured rates (bandwidth, MXU throughput) against the model's weight bytes and FLOPs per row, or a per-model probe;
- the batched-fire check beside it.

What it would buy on qwen3-0.6b: 2 of 21 rungs, of the order of 50 programs, about 8–10
GB of host at ready, against 126 GiB (F37). Not built.

## F37. XLA host RAM at ready is libtpu's live per-executable data, about 200 MB a program

qwen3-0.6b's ladder (652 programs, tip e88370f2c, census config) holds 126 GiB of host RSS
at ready. 129 GiB peaks on a cold boot. A ladder twice the size froze the 172 GB host
(readout iii).

smaps at ready: 132.8 GB RSS, all anonymous (61 GB in THP, 0.15 GB file-backed).

| mapping class | maps | RSS |
|---|---|---|
| exactly 64 MiB (glibc's per-thread arena heaps) | 992 | 49.9 GB |
| variable 16 MiB – 1 GiB large mmaps | 319 | 49.9 GB |
| 128 MiB | 147 | 19.1 GB |
| 1–2 GiB | 21 | 13.5 GB |

It is live: `malloc_trim(0)` after the warm ladder returned nothing (126 GiB before and
after).

Per executable, in a bare PJRT client, the 28-layer qwen3 decode program (22.5 MB of code,
38.5 MB serialized, 1.8 MB of StableHLO) holds:
- 177 MB after `DeserializeAndLoad` (about 7.6× its code);
- 380 MB after a compile.

Pie keeps nothing besides the loaded executable: `Executable::adopt` destroys the
`PJRT_Executable` it reads, and the cache bytes are dropped.

libtpu flags tried, deserialize per program:

| flag | deserialize | compile |
|---|---|---|
| `xla_xprof_register_llo_debug_info=false` | 174 MB | |
| `xla_tpu_impure_track_debug_metadata=false` | 177 MB | 306 MB |
| `xla_tpu_keep_hlo_proto_literals_up_to=0` | 176 MB | |

The PJRT C API has no call that drops a loaded executable's host-side data.

Levers:
- fewer programs (the ladder: logprobs share the forward's programs since 6123377f2; buckets, F36);
- smaller code a program (rolling, F35, does not land under 2%);
- a libtpu change: release or make configurable the host-side program data after load (an upstream request);
- evicting cold executables to the disk cache (parked: a reload at fire time is a fire built after ready).

Draft of the upstream request (text only; filing it is the user's call):

> **libtpu: release (or make configurable) a loaded executable's host-side data after it is
> on the device.** A serving engine that loads many fixed-shape programs at startup
> (hundreds, through `PJRT_Executable_DeserializeAndLoad`) holds about 180 MB of live host
> memory per loaded executable on v6e (libtpu 0.0.48). That is about 7.6× the program's
> generated code (22.5 MB of code, 38.5 MB serialized), and 380 MB when the program is
> compiled in process instead. 652 programs hold 126 GiB of host RAM; a ladder of 1178 has
> exhausted a 172 GB host. The engine keeps no references beyond the
> `PJRT_LoadedExecutable` (it destroys the `PJRT_Executable` it reads), `malloc_trim`
> returns nothing, and the flags `xla_xprof_register_llo_debug_info`,
> `xla_tpu_impure_track_debug_metadata` and `xla_tpu_keep_hlo_proto_literals_up_to` do not
> change it. Request:
> - document what a loaded executable keeps on the host after its first execute;
> - an option (a compile or load option, or a flag) to release whatever is not needed to
>   execute the program (the optimized HLO module, buffer assignment, debug or profiling
>   metadata, serialized form), keeping only what a re-execute needs;
> - if the program binary itself must stay host-resident (for instruction overlays or a
>   reload into HBM), a way to share or page it rather than hold a private copy per
>   executable.

## F38. ROCm 1-lane greedy moves +0.6% when gemv.hip's code object changes, its 1-row kernels unchanged

The turned norm fold (the norm prologue in `skinnyt_bf16`) leaves every 1-row kernel
untouched, yet 1-lane greedy on MI300X (qwen3-0.6b) is 0.6% slower. Over 10 ABAB runs of
1,536 tokens the wall is 2.122 vs 2.107 s, and the step cost (32 → 544 tokens) is 1.345 vs
1.338 ms. c64, 4-parallel, beam-4 and prefill are flat or faster.

Bisect:

| build | g1 ms/step, 32 → 544 |
|---|---|
| tip before the fold | 1.334 – 1.338 |
| the fold's kernels with the old routing (the fold never taken) | 1.343 – 1.344 |
| the same, the skinny kernel's prologue the tip's verbatim | 1.344 – 1.345 |
| the fold | 1.343 – 1.348 |

The 1-row decode runs only `skinny_bf16` instantiations. Their summed device time per step
is equal within 2 µs (rocprofv3, 1,024 tokens), and their registers, LDS and occupancy
match (no scratch). What moved is the module the kernels share, so the suspicion is code
placement (instruction-cache or alignment), not the kernels' work.

Next:
- dump the code object's kernel offsets on both builds; place or align the hot skinny entry
  points and see whether the 7 µs follows;
- if it does, a stable placement for the decode kernels, rather than chasing each module
  edit.

## F39. ROCm `kernel.gemm.turn_min_n` (2048) stands in for the turned kernel's narrow-bank idle

The turned kernel (`skinnyt_bf16`) walks one 16-row tile a block, so a bank of n rows is
n / 16 blocks. Below 2048 rows the route keeps the skinny kernel, and that constant is
the narrow-bank idle, not a measured choice.

Removing it costs MI300X 4-parallel decode +28 µs a step (qwen3-0.6b: 1.475 → 1.510 ms;
beam-4 2.155 → 2.188). o-proj (1024 x 2048) moves to the turned kernel: 64 blocks on 304
compute units, two serial 256-element slices a wavefront. It takes 7.2 µs a call against the
skinny kernel's 6.2 (rocprofv3 device durations).

Stream-timed microbenchmarks said the opposite: turn ahead from 128 columns up, cold weights
too. Their back-to-back launches were capped by the dispatch rate (about 5 µs) and hid kernel
duration. Kernel choices are compared on device durations.

What replaces it: turn vs skinny priced per (n, k, m) from device durations. Either a probe
that times kernels (not dispatches) in the profiler's rates step, or a tiles-per-CU /
slices-per-wavefront model fitted on rocprof durations. Then the decision goes.

Resolved: the decision is gone. Inside the turn rows, a bank both kernels cover takes the kernel
its decode body is measured faster on. At the first boot that meets a shape, the lattice is
armed with that bank on each kernel and the m-lane decode body replayed (9 rounds of 20 fires).
The turned kernel is kept where it is faster past the rounds' interquartile spread, else the
skinny kernel. The step times are saved in the device's user profile (`[rates] product_ns`,
keyed by kind, bank rows, K and m), so later boots measure nothing.

On qwen3-0.6b / MI300X the first boot chose 5 shapes in 7.5 s:
- o-proj at m = 4: skinny;
- gate-up at m = 4, the lm head at m = 4, and o-proj at m = 8: turned;
- qkv at m = 4: skinny (22 µs within its 26 µs spread).

The second boot: 4-parallel 1.463 ms a step, beam-4 2.138 (against 1.475 / 2.15 with the
constant); greedy, c64, prefill flat.

Per-product graph bursts were tried first and mispredict. One product back to back over its
layers' weights put gate-up at m = 4 on skinny (8.74 vs 8.89 µs); the step has turn 20 µs
faster. A product's cost is measured in its real neighbourhood (F40).

Open: the first boot's 7.5 s re-arms the whole lattice once per choice. Arming only the
buckets a shape's rows name would cut it to the decode bodies'.

## F40. ROCm 4-row decode: where the 4-parallel step's ~170 µs over 1 greedy goes (MI300X, qwen3-0.6b)

At short context 1 greedy is 1.31 ms a step and 4 parallel 1.48 (after the turned norm
fold). The attach kernel trace attributed the gap mostly to o-proj + down (+78 µs). That
was an artifact: under `rocprofv3 --attach --kernel-trace` every kernel starts after a
~290 µs idle gap, so its durations describe a cold, down-clocked device, not graph replay.
The serialized PMC run gives o-proj + down only about +1 µs a layer, and LDS bank conflicts
are 0.

**Correction (2026-11-20): the product rows below are warm-cache lower bounds; do not plan from
them.** A duplicated launch re-reads the weights its first launch just pulled into the 256 MB
Infinity Cache (8–13 MB a product here): it times a warm second run. At the kernel level the
folded qkv at m = 4 duplicated read +0 µs. The cold-weight ledger below replaces them.

The instrument that measures in context is ablation by duplication. Launch one product
twice (idempotent) under a scratch switch, and difference the step cost (32 → 160 tokens,
median of 5, two rounds):

| product (per call) | m = 1 | m = 4 | 4-row extra a step |
|---|---|---|---|
| o-proj 1024 x 2048, skinny | 3.7 µs | 3.7 µs | 0 |
| down 1024 x 3072, skinny | 4.6 µs | 4.8 µs | +6 |
| qkv 4096 x 1024, folded norm (skinny / turned), 27 layers | 6.0 µs | 7.1 µs | +30 |
| gate-up 3072 x 1024 gated, folded norm (skinny / turned), 28 layers | 6.6 µs | 9.0 µs | +67 |

The other kernels, ablated the same way (one extra launch of each, Δ step µs at g1 → g4):

| kernel | g1 | g4 | 4-row extra |
|---|---|---|---|
| attention decode (`paged_decode_mfma`) | +204 | +202 | 0 |
| `merge_partials` | +125 | +128 | 0 |
| qkv / rope / KV write | +137 | +162 | +25 |
| lm head (one call a step) | +71 | +81 | +10 |

The ledger: gated turned gate-up +67, folded qkv on turn +30, qkv/rope +25, lm head +10,
down +6. o-proj, attention and merge add nothing. That is ≈138 µs; the remaining ~30 is
host / channel / sampling work, not kernel-ablated.

The routes are already the faster ones here. Gate-up on the folded skinny kernel instead of
turn costs 4-parallel +20 µs (1.497 vs 1.476 ms); o-proj on turn costs +28 (F39). The gated
turned product's cost is its own: 9.0 µs a call against skinny's 6.6 at m = 1, about
1.4 TB/s against 1.9. More tiles a block (one norm stage walking them) does not help: two a
block 1.488 ms (the same), three 1.524, four 1.573. A 16 + 16-column gated tile is the same
work a block as two tiles, so it is not expected to help either.

Tried, none kept (4-parallel ms a step, against 1.475–1.484):
- skinny at m = 4: staging the activation four loads at a time (worse); reading it from
  global, not LDS (no change); 16-wave blocks over 64 (worse); a grid trimmed to the blocks
  with rows (noise); the quad kernel on narrow banks (much worse, 402 vs 306 µs traced);
- turn_min_n removed (o-proj on turn): 1.510 (F39);
- the turned kernel's tiles spread evenly (192 two-tile blocks for gate-up's 384 tiles over
  304 CUs): 1.498–1.505.

Instruments, ranked by fidelity, for a kernel's cost in a decode step:
- the step's own replay, the body armed each way (F39), or duplication of the kernel inside it;
- graph bursts of one product (a different neighbourhood: they reversed gate-up's order);
- the attach kernel trace (kernels ~290 µs apart);
- stream-timed microbenchmarks (dispatch-capped near 5 µs).

Next:
- gated turned product at m = 4: why its walk streams at 1.4 TB/s (latency per tile under
  one wave a SIMD), with the PMC counters inside a duplication run rather than the attach
  trace;
- the ~30 µs host / channel / sampling share at four lanes;
- the duplication switch is a scratch patch: a profile probe that times a kernel in place
  (in a captured body) would make it a standing instrument.

### Cold-weight ledger (2026-11-20, tip a7052acfc + the padded turn rows)

Each product is duplicated over the next layer's weight, last read about a step (≈ 1 GB of other
weights) earlier, so the duplicate pays a real HBM stream. The lm head's one 311 MB weight is
past the cache, so its plain duplicate is already cold. Δ step (µs, all layers), 32 → 160
tokens, median of 5:

| product (its kernel at 1 / 4 rows) | 1 row | 4 rows | 4-row extra |
|---|---|---|---|
| gate-up, folded (skinny / turned) | +207 (7.4 a call) | +274 (9.8) | +67 |
| o-proj + down (skinny / skinny) | +237 (4.2 a call) | +275 (4.9) | +38 |
| qkv, folded (skinny / skinny, F39's choice) | +171 (6.3 a call) | +195 (7.2) | +24 |
| lm head (skinny / turned) | +71 | +72 | +1 |

That is ≈ 130 µs of the 152 µs 4-parallel-over-greedy gap (1.456 vs 1.304 ms): the products
are most of it.

Counters on the gated turned product at m = 4:
- 2 waves per SIMD (4-wave blocks, 160 AGPRs + 16 VGPRs; 2 blocks per CU by the 32 KB turn);
- waves wait 61% of their cycles, mostly on memory;
- LDS bank conflicts were 147 k a dispatch, fixed by padding the staged rows (sc/amd-gated-turn).

Occupancy does not bind. 384 tiles × 4 waves is 1,536 waves, about 5 per CU against 8
resident. Every wave's whole weight share is requested at once: 12.6 MB in flight, a floor of
≈ 1 µs latency + 12.6 MB / 5.3 TB/s ≈ 3.4 µs. The rest is each block's serial tail after its
data lands (the fold's dependent loads, the MFMAs, the cross-wave LDS reduction, the epilogue),
longer at four rows. So split-K (no loads added in flight) and 8-wave blocks (no residency
added) are not expected to help.

Tried, none kept: the norm weight fetched under the row loads (greedy +12 µs); the first tile's
weight issued after the norm (no change).

In-wave pipelining, checked (2026-11-20):
- The skinny kernel at m = 4 already computes each chunk as it lands: its ISA waits vmcnt(3),
  (2), (1), (0), each before its group of MFMAs.
- The turned folded kernel issued its weight loads, then each norm item's row loads followed by
  vmcnt(0). The counter retires in issue order, so the norm waited for the whole weight share. A
  two-phase norm (rows and the norm's weight loaded first from clamped places, then the weight,
  then the stage's arithmetic in its order; bit-exact; ISA vmcnt(14…8)) overlaps them. It
  measured slower: 4 parallel 1.482 / 1.480 vs 1.465 / 1.469 ms, beam-4 2.153 / 2.154 vs
  2.140 / 2.145. Not kept. Issuing the weight after the norm made no difference either.

So the norm's serial chain is not what the four-row products pay. Ruled out: occupancy, split-K,
load ordering, in-wave pipelining; LDS conflicts were real (fixed, −5 µs). What remains is the
per-byte work at four rows (each weight byte feeds four rows' MFMAs and reductions).

Next: an MFMA shape that covers four rows without waste. The turned kernel's 16x16x16 spends
three quarters of each product on padding rows at m = 4; skinny4's 4x4x4-per-block layout lost on
narrow banks. Each candidate is judged by the cold-weight A/B.

### Restated against region-merge-2 (2026-11-20, tip bcbc1b430)

With the split epilogue (region-merge-2 and the split partials' agent-scope stores), MI300X,
Qwen3-0.6B, per step (32 → 160 tokens, median of 5, two rounds):

| | ms a step | over greedy |
|---|---|---|
| greedy | 1.306 / 1.310 | 1 |
| 4 parallel | 1.490 / 1.497 | 1.14 |
| beam-4 | 1.696 / 1.701 | 1.30 (1.14 over 4 parallel) |

Beam verify passes at w2 / w4 / w8 × 256. So beam-4 over greedy is ≈ 390 µs: about half is the
four-row forward (4 parallel over greedy, ≈ 185 µs, the products' ≈ 130 of it by the cold
ledger above) and about half beam-4 over 4 parallel (≈ 205 µs: the epilogue and the page
lists). The user's beam cost rule (beam-4 ≈ one greedy step) needs both halves.

## F41. Windows NVIDIA (RTX Spark): a one-time submit stall early in an engine's life

`runtime::register::a_cold_registration_compiles_while_a_lane_fires` fails on the Spark
(Windows on Arm, NVIDIA driver) in about half its runs, on wgpu-Vulkan and on native Vulkan,
on memory-redesign (04df99f24) as on the branches after it. It passes on Linux (RTX PRO 4500).
One fire stalls 100 to 600 ms; the lane fires once inside the compile window instead of
~100 times. Measured in-process (no nsys on the Spark):

- The time is inside the submit: wgpu's `queue.submit`, or the native submit. The fire's
  settle (its fence wait) is ~1.5 ms.
- It is not the compile and not a lock in pie. `program::adopt` is host-only and
  single-threaded, ~46 ms, and the stall outlasts it by up to 470 ms. With the off-lane thread
  replaced by a pure spin, a spin that yields every millisecond, or a thread that only
  sleeps, the stall still shows (1 of 3, 3 of 5, 3 of 5 runs).
- It is not native Vulkan's pipeline-cache write either. In the test the cache has no file,
  so `PipelineCache::persist` never reads the driver's data, and a run still stalled 601 ms.
- It lands on one submit early in the engine's life. Three or more fires between building
  the registration and the compile window avoid it in every variant run (16 of 16), and the
  fires after the stalled one take ~1 ms.

Read: a one-time host-side stall in the Windows NVIDIA driver's submit path, landing on
whichever submit is in flight; the test catches it when that submit overlaps the compile.
The test stays strict, known on this device. Owed: driver-level tracing on that box (not
nsys, which bluescreens it), or a driver update that moves it.

## F42. The first request after boot runs slow; the first served fire of each XLA shape too

Two one-time costs after ready, measured on v6e qwen3-0.6b (census config), tip and the
mixed decode capacity alike:

- The server's first request of an inferlet: 117 ms against 22 ms for the next ones (a
  12 ms install, guest registrations cache hits of ~1 ms). A burst that arrives first
  piles up behind it: the first staggered load after boot runs 1508 tok/s against 2189 on
  the next, and one tiny request beforehand brings the first staggered load to 2335
  against 2345. It sits in the runtime's first launch of an inferlet program (not the
  engine's programs); naming it is the runtime's.
- XLA: the first served fire of each fire shape runs 2.5-6 ms longer on the device than
  its median (PIE_XLA_TIMING: a 1536-row mixed fire 23.6 against 17.4 ms, an 8-lane
  decode 5.6 against 2.2 ms; about 45-50 ms over the 20 shapes a first load touches; host
  stage unchanged, nothing compiled or loaded after ready). Executing every laid program
  once at load, on scratch, does not remove it (2.36 against 2.47 ms mean excess, at 17 s
  of boot), so it is not an executable's first execution: something a served fire of a
  shape first touches (its staged inputs' device buffers, its readout seats) that a
  scratch fire does not. Unnamed; small next to the first item.

## F43. A mixed fire with no canonical shape compiles after ready (the runtime composes it)

`canonical_pads` (engine-xla serve) pads a fire that prefills to rungs so a laid program
holds it, and returns None where no laid shape exists; the fire then goes out at its exact
lane counts and compiles mid-run. One composition the runtime forms: 35 three-row prompts
beside 17 decodes (128 rows, 53 lanes, v6e qwen3-0.6b c64). Its prompts pad to the 64-lane
rung, which leaves 16 decode lanes within max_lanes (80); 17 decodes fit neither 16 nor
their rung 32 (96 lanes). Base's rule has the same gap (the decodes' rung 32 > the 16 left
and 17 > 16); with the mixed decode capacity (decodes beside prompts pad to the largest
decode rung their prompts' rung leaves) it hit 2 of 4 serving runs (a c64 pass at 642
tok/s), base 0 of ~10.

A contract bug: a served fire with no laid shape is a compile after ready. Fix on the
cutter (runtime, every backend): never compose a mixed fire the engine has no canonical
shape for; seat the decodes and the prompts that fit beside them, and defer the rest to the
next fire. The ladder stays as it is (no half rungs, no extra programs). Test case: the
composition above.

Also after ready, and not a fire's: a built-in inferlet's guest stages compile at its first
registration on a machine (v6e, beam_search at widths 1, 4, 8: six guest programs, a prompt
and a decode epilogue a width, x 8 batch rungs = 48 compiles of ~23 KB texts, ~20 s for each
new width's first request). They persist: the next boots' same registrations took 0.017 s
and 0.032 s, 0 compiles after ready, and the beam fires themselves were on the ladder (0
misses). The width is a constant of the inferlet's trace, so no ladder derives them; laying
built-ins' guest programs before ready is GPU's queued item.

## F44. engine-cuda reads its cgroup room itself; move it onto engine::host_memory

`engine::host_memory::available` now leaves a capped cgroup `memory.max` less `memory.current` less
`memory.stat inactive_file` (v6e: 150 GiB user.slice, 115 GiB charged, 96 GiB of it file cache,
read 2 GiB free before and ~95 GiB after; the XLA guest compile ledger had refused a 2.3 GiB stage
by name). engine-cuda nets the page cache in its own reader (`weights.rs` ~372-407), counting
`inactive_file + active_file`. Move engine-cuda onto the shared reader and settle one rule for both:
inactive only (what the kernel reclaims first, before a charge fails: the conservative room) or
inactive + active (all file cache, which reclaim reaches under pressure).

## F63. Beam page lists: per-group reserve terms against a pool-wide reserve by fires in flight

A beam group's pool (`new-memory/designs/BEAM-PAGE-LISTS.md` §3) grows a
column at a time to `B (c + 2) + B F + 1` pages: each live column's pages,
plus `B` pages one fire may hand out and `B F` pages freed in the `F`
frames in flight, terms every group carries whatever its length. At short
lengths they are most of the reserve: 64 tokens, B = 4, F = 2 reserves up
to 33 pages against the earlier shared pool's 18, and the 9 GiB-capped
200-beam + 300-decode gate on MI300X ran 25-28 s against the tip's 21-22 s
(uncapped 16 s both: speed parity, a capacity cost). Measure whether the
`B` handed and `B F` pending terms can come from one pool-wide reserve
sized by the fires in flight across groups (a fire hands out at most `B`
pages per group it runs, and frees wait `F` fires whichever group freed
them) without weakening the per-step guarantee that a fire never opens a
column it cannot back: the capped gate's walls against the tip, the pages
the planner counts per group, and the refusals under over-demand.
## F47. General per-lane split of a pass whose whole demand exceeds the device

A pass whose whole demand (every lane's kv and state) is past the device while each lane fits
alone is refused today (only a unit past the whole device alone is refused at submit, bc9354e5a).
The general split by lanes into sub-fires that each fit needs part outputs held and joined by the
engine across backends, and per-set parking of finished groups in the planner. The beam example's
per-copy prefill is fixed by forking instead; not a runtime gate.

## F45. Remove the 600 s silence timeout

The frame gate no longer drops a lane on a timer (the leash and `submit-deadline-us` are gone):
a lane whose guest waits on outside input is parked by the host (`process::away_while`: session
receives, `wasi:clocks` waits, an HTTP request until its response head), and a computing lane
stays awaited. `runtime.silence_timeout` still terminates an awaited lane silent that long.
Remove it once each case it still catches has its own mechanism:
- (a) poll-mixed waits are classified: `wasi:io/poll` in componentize-js / py language
  components (one poll can mix the guest's own pie futures with outside ones, so parking on
  every poll would un-await own-fire waits), HTTP response-body streams after the head, and
  sockets;
- (b) a guest spinning in wasm without submitting;
- (c) a completion that never arrives (a lane in flight for twice the timeout is terminated
  through the same path today).

Resolved on sc/mac-no-silence: the timeout, its clock, `owes_forever`, `FramePlan::Terminate`
and `runtime.silence_timeout` are gone (an old config naming the key is refused as an unknown
field). A blocking lane whose guest computes, (a) and (b) alike, is held while a frame is in
flight, or until every other member has submitted since the last retire; past that, with the
device idle, the frame seals past it and its next submit rejoins: never held for good, never
terminated. (c) is a named wait ("its fire in flight" / "its bind"); a completion lost is the
engine's or the store's to refuse by name where it is lost. The stall dump names each lane's
wait. Metal Qwen3-0.6B, M1, ABBA vs 0b2bd5386: c1 180.1 → 182.9, c8 668.0 → 669.4, c64
1745.4 → 1744.0, c8 at frame_size 1 683.9 → 683.3 tok/s.

## F46. Something writes into the guest lane arenas' split lines while a meeting runs, under overlap (ROCm, found 2026-11-20)

While building self-resetting meet counters (each split row's meeting leaving its word's count
zero, the split select's merge zeroing its count, so no memset before each split launch), the
engine-cuda ROCm integration set (every_op / epilogue / suite / beam / guest, 18 tests) hung at
every_op_agrees_with_the_host_interpreter when run in one process. It needed the accumulation
of the earlier tests: each test alone, each paired with every_op, and either half of the set
passed. The tip passes that point.

The check: before every split row or split select launch, sync the stream, read every split
lane line of the arenas (16 words each, `lane_span(1)` apart) and report any non-zero count,
naming the launch before it. With it on, the whole set passes (152 s) and no count is ever
found dirty. So no launch leaves a count behind. Something disturbs a meeting while it runs,
and only when work overlaps: the sync before each split launch removes the overlap and the
hang.

Meeting groups are forked onto main (`fan.fork(&meet, main)`), so two meeting regions do not
run side by side. The writer is not found. Candidates: a run-ahead frame's staging next to the
arenas in the activation span, or another run of that span (`Shell::staged_runs`) laid over
them.

The tip's per-launch memset (`zero_barriers` before each split launch) re-zeroes the counters,
so its meetings are immune. But a concurrent writer into the guest arenas could as well corrupt
the partials and candidates the split regions publish there: silent wrong results, not a hang.

Tool for whoever picks it up: the sync-and-read check above, on a build without the memset
(the self-resetting branch, `sc/amd-meet-reset-dropped`, local on the MI300X host), plus the
fire's stream map at the stalled launch.

Resolved (2026-11-21): the counters the meetings disturbed are gone or sized out. Not the
writer found: the dropped branch's last work points at the sampler itself (a sampler lays its
lanes at `lane_span(16)` apart, its blocks' arrival words and data over the lines split rows
meet in at `lane_span(1)`, where a self-resetting count read them as arrivals), not proven by a
repro. What closes it:
- A region whose rows split runs a launch a phase (`ptir_phase`): its blocks fold every block's
  partials in the next launch, so no split row counts arrivals or waits. The spin meeting also
  had a race of its own: a block past a meeting wrote its next reduction's partials into the
  slot a slower block still folded; a phase's partials now alternate between the slot's halves,
  an argmax's candidates between two banks.
- A sampler lane's blocks still meet at `M1Lane::barrier`, laid over no more units than this
  process's queues run on (`device_units`, CU masks counted) at the kernel's occupancy, checked
  at launch. A split select's last block merges; none waits.
- `blocks_that_meet_run_on_a_few_units` runs the epilogue's cases (split rows) and the lane
  sampler's in a process `HSA_CU_MASK` keeps to four units; both hung there before.

## F51. `resident_workgroups` is not reproducible across boots: its consumers must tolerate +-40%

The load's rates step measures the workgroups a device holds at once (`[rates]
resident_workgroups`, `engine_gpu::rates::resident_workgroups`): one workgroup of the matmul's
peak kernel timed against the peak of the curve. Across boots of one device it moves
20-40% (RTX Spark, 2026-11-21: wgpu 154, 156, 196, and a saved 299; native Vulkan 30, 45, 28;
Android Adreno homes 8.8-9.0 and 10.9). Timing the peak shape and the one workgroup back
to back over 5 rounds, median, floored, did not settle it: wgpu 319, 262, 271; Vulkan 109,
114, 136. That spreads ~20%, adds 35-75 s to each boot, and moves Vulkan ~3x, so it changes
what is measured instead of fixing it (the patch is not kept).

So whatever reads it must give the same answer for a value 40% off: a guest step's grid
(`step_jobs`), Vulkan's `cores` where the driver reports none (the qmm ladder's `fill`, the
decode split per core). Or derive it from device facts instead of a timing. Note that the
measured count is saved at a home's first boot and kept, so two homes of one device can
size differently forever.

## F48. A batched top-k (more than one row) still sorts each row whole

A one-row `top_k` (a flattened beam's candidates) selects its `k`-th key and sorts only
what lies past it (`steps::Ladder::Select`, `sel_top_k`): on wgpu Qwen3 the beam-4 step
went from 1.11-1.20x a greedy-4 step to 1.08-1.15x. A `top_k` over `[rows, vocab]` (a batched
top-k sampler, a per-beam pre-cut) takes the full merge sort per row (~43 us a merge round,
~780 us a step at Qwen3's vocab). If a guest program uses one on the hot path, carry the
select per row: its state (`SELECT_WORDS`) is one row's today.

## F49. XLA's in-place write fence costs hybrid decode 2.6–4.3% (v6e, found 2026-11-21)

`kernels_xla::hlo::Func::fence` puts an `optimization_barrier` over a parameter's buffer and
every read of the version an in-place write overwrites, so XLA cannot schedule a read after the
write and keep the old version as a copy (a second 22 GiB pool left 15 Qwen3.5-0.8B hybrid warm
shapes out of HBM). It costs hybrid decode: v6e Qwen3.5-0.8B c1 251 → 244 tok/s (−2.6%), c4
756 → 723 (−4.3%), the decode fire's device time +4.6%; dense Qwen3-0.6B is flat. A decode fire
has ~6 barriers holding ~16 small pool reads (i32 scalars and small tiles), all on the pool:
not bytes, but those reads taken out of the consumers they were fused into and run before the
write. Measured alternatives that did not help: fencing only the reads the write is not computed
from (c1 244.4, c4 723), and passing the barrier through the write's update instead of the pool
(c1 243, c4 723). To revisit with an ordering that keeps the reads fused: the pre-write values
read inside the fused op that writes, or a fence on only the overwritten version's last read.

## F50. XLA's pooled expert matmul runs 1.5–3% slower in a MoE decode fire than held banks, though at parity alone (v6e, found 2026-11-22)

With streamed expert banks the grouped matmul reads each expert from the pool (`Gmm::pooled`, a
table of expert rows and the pool in the bank's place). Alone (`time_a_pooled_select`, the MoE
mini's shapes: 16 experts, n 768 k 2048 and n 2048 k 768, 512 and 2048 routed rows) it is at
parity with the dense bank (e.g. 95.2 vs 94.5 us). In the served decode fire it is not:
qwen3-30b-a3b-mini decode, held vs streamed, c64 42.9k vs 41.7k tok/s (−2.8%), c256 107.5k vs
106.0k (−1.6%); vs main (which holds them) c64 −1.3%, c256 −3.1%. xprof at c256: held calls run
gate ~29, up ~59, down ~28 us; pooled calls a uniform ~45 us (+75 us a step). Suspected: the
expert DMA's overlap with neighbouring ops in the fire (the held up call's 59 us looks like a
wait on its weight copy that the pooled form spreads), or reads off the 4.5M-row pool. Next: a
profile of the pooled call's DMA waits in the fire, and the same fire's schedule held vs pooled.

## F52. The declared-context cut's "one sequence at most half the room" is a tuned constant

`store::declared_context_within` (engine-cuda) serves a declared context at
the longest one sequence of which takes no more than half the live room past
the weights, "the activations and a second sequence having the other half".
The half was moved from the old pre-stage cut, not set by any ruling. Derive
it (the lanes the load promises must seat at the cut context, beside one
rung's activation span) or drop it in favour of the capacity gate's finisher
guarantee; the fit's own shortening (`Pools::declare_context`) already
derives what one working set needs.

## F53. An unplaced served load hangs the GPU (slang backends, 2026-11-21)
A direct fire on a served load before the engine places its programs (`Programs::load`)
hangs the device instead of being refused: found when the dense settle fired from inside
`Shell::load` (rows 1, its first fire never landed; the same fire placed after
`Programs::load` lands in 6 ms). Suite and check loads place no programs and fire fine, so
it is something a served load expects `Programs::load` to have set up (its readout / shelf
placement, a table it writes, or a submit it makes), not yet pinned. Today a served load is
guarded by phase (`Boot::places_programs`, refused by name until `Shell::programs_placed`).
Find the resource and guard on it instead.

## F54. Every consumer of a device's scalar rates handles an unmeasured one (slang backends, 2026-11-21)
A device rate read below its own launches (a physically impossible frame) is repeated
(`engine_gpu::rates::repeated`, REPEATS times); a curve point every repeat of which reads so is
left unmeasured and the load goes on, but a single rate (the copy and launch, the per-row ops, the
scan) every repeat of which reads so still refuses the load by name. A load measures a rate only
where its profile lacks it, so no saved value stands behind it then. Make every consumer of those
rates (the estimate's roofline, the top rung, the submit bound's step estimate, the dense choice's
launch) handle an unmeasured rate, stated at load, so a load never refuses on timing.

## F55. The dense settle skips a model whose classes read a guest input (slang backends, 2026-11-22)
A settle fire seats plain lanes; on a model whose served classes read a guest input channel,
`serve::feed_inputs` refuses them ("feeds no channel for it"), and the settle now stops there,
said, the curves' kernels serving every range it did not reach. Such a model gets no settle at
all, so a wrong or stale curve can still serve a kernel several times slower on it: the case the
settle exists for. Fix: stage a feed for those channels in the settle's fires (zeros of the
declared shape; the settle only times), or settle through a class that reads none, instead of
stopping.

## F61. Metal streamed guest passes run far below bandwidth (needs Metal System Trace counters)
Qwen3.5-0.8B beam-4 on the M1 Max, the beam epilogue's log-softmax region (4 rows x vocab ≈ 1M
elements), after sc/mac-beam-hybrid 83cbd18f9 / ce84536d1 / 0f6db38c4 (register handover, dropped
stores, broadcast index once a pass, a pipeline per streamed step, broadcast sinking, length
classes; guest time 0.81 -> 0.54 ms a fire). The region now runs only the passes it needs (max
reduce, exponentials, sum reduce, one output pass) and still takes 0.32 ms: each full pass ~70 µs
and each reduction ~60 µs, against ~10-15 µs of memory traffic each.
Measured (streamed-repeat / streamed-nop / streamed-groups diagnostics):
- a dispatch's fixed cost is ~4 µs (streamed-nop), so launches are not it;
- the group cap is at its optimum: r2 0.68 / 0.45 / 0.33 / 0.32 / 0.42 / 0.87 ms at 8 / 16 / 32 /
  64 / 256 / 1024 groups of 512 threads, slower both ways;
- pass count, materialized broadcasts and the per-element broadcast index are gone and moved it
  little (0.55 -> 0.54 ms for the last two passes removed).
Not yet separated: memory latency against ALU (the precise exp / log the bit-exact results need,
the bf16 roundings), and the reductions' three dependent levels. That needs the GPU's counters
(Metal System Trace / Instruments: occupancy, ALU and memory utilization per dispatch), which a
headless session cannot capture. The M5 meets the beam rule already (beam-4 / 4-lane 1.023x), so
this is the M1's and a general guest-codegen item, not a gate.

## F62. Chunked lane scratch through the eta scratch ABI (post-PR)

A guest instance's value scratch lies whole in one run of adjacent chunks inside one segment: the eta
scratch ABI (`eta-compiler` codegen/lane_scratch.rs) has a grouped kernel address every value of a lane
from its one base (`scratch = lane_scratch[l]`, value `v` at `scratch + offsets[v]`), so a lane's stage
scratch is one contiguous run, and a place whose run is past a segment is refused by name (engine-metal
`Plane::guest_state`). A beam group's scratch grows with its beams (Qwen3-0.6B at 4K context on the M5:
113 MB at w17, 215 MB at w32, ~6.7 MB a beam of peak-live log-softmax-wide values; liveness packing is
already on), so its width is bounded by the segment: Metal's 64 MiB (35 chunks of 1.9 MB on the M5) holds
about w17-w19 at 4K context, CUDA's 1 GiB w96.

Raising Metal's segment to CUDA's 1 GiB (ChunkProfile::ON_DEVICE_METAL.segment_bytes, which
engine-metal's pool reads) was tried and kept out of the PR: Metal's segment is the elastic pool's grow
and release step (the governor would yield in 1 GiB steps on unified memory), and the activation span is
placed against segment boundaries (pool::top_first), so a small pool's span landed on unmapped top heaps
("mapped bytes of an elastic buffer's top"); two integration tests also assume several segments. Doing it
means placing the span at heap granularity whatever the segment, with the governor's step decoupled.

`eta_exec::lay_out_chunked` already packs a stage's values into pieces of at most a segment (lifetimes
shared, each item whole in one piece); no kernel reads that layout. Removing the bound for every engine:
- the lane table carries a base a (lane, piece), or a per-lane piece table;
- each value's place is (piece, offset), not one offset;
- every emitter's value addressing follows: Metal's grouped, streamed and fused forms, CUDA / HIP's SIMT
  (one ABI), the emitter versions bumped;
- each engine lays `lay_out_chunked`'s pieces in the place the runtime took (`GuestState::run` the
  piece run, pieces of a segment at most), and `FireFacts` follow.
Estimate for Metal and the shared eta change: ~2 days, plus CUDA / HIP's emitter side.


## F60. Hybrid beam replay grows with lineage divergence

On a hybrid model each beam lane replays its buffered rows past its group's
common ancestor every step (`new-memory/designs/BEAM-PAGE-LISTS.md` §11.3,
`[c, s_t]`). Where two lineages stay apart, the rows grow with the steps
since they parted: Vulkan Qwen3.5 beam-4 on one prompt carried 4 × 257 rows
at step 257 and up to 4 × 1018 rows later in the search. engine-gpu serves
it (a beam lane's carried rows are a span whose pages resolve through its
list, its set's rs entries in its group's state; the per-fire tables stay
O(lanes + widest bucket)), but the cost per step grows with the divergence
(O(steps) replay, O(steps²) over a search). CUDA, Metal and XLA move to the
same shape. A deep group's resident state (its lists and set entries) lies
first in its instance's one scratch place (`GuestState::resident`); past a
chunk that place is whole runs of adjacent chunks, so binding or restoring it
relies on the planner naming a compaction for a run the free chunks' total
covers, else it waits on scattered room that never forms a run. A bound by
design is a spec question for AMD / Leader: per-beam state checkpoints (each
beam's recurrent state kept at page boundaries, so a beam replays at most a
page past its last checkpoint) would bound the replay by a page at the cost
of a state per beam per page.

## F56. Every engine's load-time measurements through the one keyed resolve path (found 2026-11-22)
sc/gpu-committed-profiles put engine-gpu (Vulkan, wgpu) behind one path: each stored entry is taken from the user's
file, else a committed `profiles/<device>.measured.toml`, else measured — or, under `[engine] measure = false` /
`pie serve --no-measure`, skipped and served by the estimate, by name (`engine::rates::resolve_at_load`,
`profile::Rates::sift_as` / `whence`, `Shelf` `.measured.toml` layer, `pie tune export`). Each other engine still
measures outside it; until it ports, `measure = false` is refused by name for it (worker `EngineConfig::validate`).
What each still times at load or boot, for its owner (paths under `crates/`):
- **engine-cuda (CUDA, ROCm)**: ported on ROCm (sc/amd-keyed-measurements); CUDA's keys open.
  - `engine-cuda/src/stored.rs` `Keys: Now`: on ROCm the device's name, arch and HIP driver / runtime; a kernel's
    code is the carried kernel library's text and options (`kernels_rocm::stored::library_code`: any `.hip` change
    measures again what fired it) and hipBLASLt's version. Method versions `RATES_METHOD` / `PRODUCT_METHOD`.
  - Roofline `[rates]`: `chip::resolve_keyed` (committed `<device>.measured.toml` layer, each stored entry sifted),
    `rates::ensure` -> `engine::rates::resolve_at_load` under `Knobs::measuring`, what it measures stamped
    (`Keys::stamp`). `chip::resolve` stays unkeyed for `pie tune` / `pie profile`.
  - Decode products: `product_ns` saved by `save_rates`, keyed (`measured_by["product_ns"]`: library + hipBLASLt),
    read back through the resolved profile; route codes 0 / 1 stay tags. Skip: the stated kernel, by name.
  - Bodies record (`serve/measured.rs`): keyed by the code that shapes it (`build.rs`: engine-cuda's sources and
    its kernel crate's Rust and kernel sources, a compile-time constant: nothing hashed at boot, a pie version
    that touches none of it keeps every record) and by the checkpoint's own identity (`zt::artifact_identity`,
    its manifest's digests: a copied or moved file keeps its records). Committed beside the measured profile
    (`profiles/bodies/<key>`, `pie tune export --engine cuda`), so a committed clean boot holds the figure.
  - `measure = false`: a one-device cuda engine; group / lockstep boots still measure, and are refused by name.
  - Open, CUDA: `stored::backend` for `not(rocm)` keys by device name and method alone (no driver, cuBLASLt or
    cubin codes): the L40S fills it. Committed: `amd-instinct-mi300x-vf.measured.toml` (`cached_gbps`,
    `product_ns`).
- **engine-metal**: ported (sc/macpro-committed-profiles). `chip::resolve_keyed` resolves through
  `Shelf::resolve_keyed` under `stored::Keys` (`on` = `metal <device> macOS <major>`; each `[rates]` key and dense curve
  keyed by its method and the resolved Metal source of every file its measurement fired, recorded by
  `encode::record_fired`); `rates::at_load` takes `Measuring` through `resolve_at_load`; `tune::dense::at_load` keeps
  the per-model crossover as `kernel_tflops_by_shape_keyed` curves (vector forms `1/8`, simdgroup tiles `2/0`,
  tensor-op tile `3/32`, `KernelCurve::MPP`) in the user file, the ladder fitted from them each load (no
  `dense_settled`: the fit is the settle, recomputed from the curves); `[metal] measure = false` skips both (the fire
  top a decode fire of the lanes where no roofline is stated); `cover_fingerprint` keys on `on` and every decision's
  resolved rungs. Committed: `apple-m5-pro.measured.toml` (Qwen3-0.6B, Qwen3.5-0.8B, Gemma 4 E4B shapes). The
  runtime host cover is committed beside a device's profile on sc/gpu-committed-profiles-2 (below).
- **engine-xla**: nothing timed at load and stored (the guest compile ledger `guest/run.rs:198` is in-process RSS).
  No `cover_fingerprint`, so the runtime re-times its host cover every boot (below).
- **runtime host cover**: `host_cover` `runtime/src/bootstrap.rs:1062` (called :1021), timing `measure_host_cover`
  :1125 (warm `compat-openai` runs at depth 1), cached in `<cache>/cover/<fnv>.json` keyed by every engine's
  `cover_fingerprint` + deployment + sampler hash + frame + depth + context. Committed (sc/gpu-committed-profiles-2):
  `<device>.covers.json` beside the device's measured profile, written by `pie tune export` from the home's cache
  (entries whose key names the device's `measured_on`), listed in the backend's `COVERS`
  (`Engine::committed_covers`); looked up home cache > committed > measured. Under `measure = false` (every engine)
  the dispatch depth is taken by name in place of the warm runs (a cover too large costs window, never a result).
  engine-cuda states one too (the device and its kernel libraries' code, the guest emitter's version, the decode
  products' routes table, a fire's lanes and tokens) and lists its committed covers in `chip::COVERS`; a CUDA
  group or lockstep rank states none (measured every boot). Open: engine-xla / engine-cerebras state no
  fingerprint (`Engine::cover_fingerprint` default None): measured every boot, nothing kept.
- **engine-cuda, open (accepted)**: an evacuation is all or nothing on one shell (sc/l40s-evac-atomic: every check
  a dry run before the copy, nothing past it fails but the device), but a tensor-parallel group (`group.rs`) and
  the expert-parallel lockstep (`lockstep.rs`) evacuate rank by rank. Every logical check is the same on each rank,
  so a refusal comes from the first rank before anything moves; only a device fault on a later rank could leave
  the ranks split, and a device fault poisons the context anyway.
- **engine-gpu, closed on sc/gpu-committed-profiles-2**: under `measure = false` on a platform stating a submit
  bound (Android, Windows TDR) with no roofline stated or committed, the load times only what the bound's estimate
  reads (`engine_gpu::rates::for_bound`: a short launch chain, a one-row and a widest-rung matmul over the model's
  widest layer plane, the attention of the widest rung against the context, the scan on a hybrid; single fires,
  "timed for the submit bound despite measure = false"), kept in the bound alone and stored nowhere; steps are
  held and cut as under a measured load. Without curves the dense kernel is estimated from the work's shape
  (`engine_gpu::dense::by_shape`; by the roofline where one is stated); with curves but no roofline, the curves
  weigh it (`dense::choose_unrated`). Still open: the Adreno 840's committed profile (`adreno-tm-840`, a
  commented slot in each backend's `SEEDED`) awaits the phone's export.

## F57. Metal's kernel.dense ladder from agreeing curves (found 2026-11-22; decided 2026-11-22)
The load's ladder was `tune::dense::fit_at` over the per-shape curves (`profile::tune::fit`, `min_gain` 1% of the
floor): a rung that fixes the 8-row pick alone saves under 1%, so noise decided whether it was kept. Two clean
measuring boots of Gemma 4 E4B on an M5 Pro measured curves that agree (rows 6-12, 153 points: median 0.25%, max
1.88% apart; the fastest kernel the same at every row count) and fitted ladders serving c8 at 177.8 and 199.8 tok/s.
Decided (Leader): the load takes each measured point's winner (`tune::dense::winners`, sc/macpro-dense-winners),
no fit and no threshold; ties within a point's timed rounds go to the neighbouring row count's winner, else the
profile's pick. Two clean measuring boots now serve Gemma c8 at 199.0 and 200.0. Still open:
- Their ladders are not byte-identical (50 and 51 rungs): at one point (2560x2048, 16 rows) the two runs' winners
  differ by more than either run's spread (1.4% one way, 2.5% the other), and at 13 points one run calls two
  kernels tied and the other does not. Run-to-run variation exceeds a run's own spread (the fastest of 7 rounds);
  a spread taken from two passes apart in time would state it, measured, at twice the curve pass's time.
- An M1 Max served c8 7.5% under 8ed8f8679 on a per-point ladder whose 8-row fires were faster on the device:
  Mac's whole-step capture is to show which row counts or frames the curves mispredict.
- `pie tune metal`'s cross-model fit (the checked-in device file's generic ladder) still fits under `min_gain`.

## F58. The capacity gate does not charge the room a park to disk opens (planner sim, found 2026-11-23)
The field mix as it now runs (beam pools bound beside the prefill's instance, frames owing rung raises, two-ended
pools with the span backed on demand) fails 4 of the 751 seeds without the wide shape: 401 and 645 (liveness: every
tier full, the finisher a page short), 589 (a victim evicted twice for one head) and 603 (a raise owed to its
owner's frame never planned: `rung_short` reads the raise as fitting while `plan_rung` finds no free run below for
a run in the span, so the planner funds nothing and the walk has nothing to move). Seed 401's cause: a park to
disk lays its guest parts in whole disk chunks per slot class, which raises the census's slack; the gate's margin
(`CapacityRead::margin`, the largest other resident's move) counts the move's own pages only, so a move under a
finisher whose spare was 0 left every tier full (free 72 -> 39 in one step, slack +32). Charging a chunk per slot
class of the mover's held parts in the margin passes 589 and 645 but breaks main seed 134 (a thrash) and leaves 401
failing: the margin must be the exact granularity a park opens (the classes whose chunks on the tier below have no
room for its parts), not a bound.

## F59. The safe order turns short with no ask past it: a gate accounting gap, not admission (planner sim, found 2026-11-23)

The field mix's seed 401 (45 processes arriving at once on a 15-chunk device) and the 401 family (589, 645;
the Metal w17 mix's thrash and liveness seeds, 31 of its first 200: 18, 43, 47, 54, 67, 70, 74, 76, 77, 84,
88, ...) end with every process short in the banker's order. `PIE_SIM_SHORT=1` says each step the order turns
short (`ResidencyPlanner::order_least`, the raw least room in the order, negative when some process cannot
finish in turn) with the gate's read. On 401 it turns short at step 7 already (least -6, free 99, `moving` 9,
unit 29), again at 10 (-6) and 17 (-21, `moving` 21): before any process grew far, while moves were under way.

Admission at the last place does not explain it: a newcomer appended last needs its end with every other
process's holdings given back, which fits whenever its end fits the pool, so a banker's check there refuses
nothing. The order goes short because a step's reads move under the gate: the second slots of moves under way
(`moving`) and the margin (`unit`, the largest other resident's move) grow with grants the gate let through as
safe at their own read. A grant's cost (`CapacityGate::cost`) charges the margin for the asker's own move only;
the read after the grant also counts the moving copies the grant's own eviction or restore opened. The fix is
in the gate's accounting (charge what a grant will put in flight, or read the order after the moves it starts),
then admission priced by the same order. Repro: `PIE_SIM_SHORT=1 PIE_SIM_FIELD=1 PIE_SIM_SEED=401` on
`replay_one_simulated_seed`, and `PIE_SIM_METAL=1` sweeps.


## F66. XLA states a replay bound its prepare does not hold to (found 2026-11-24)

sc/rt-rung-carried (6fd1ea355) has each engine state per rung what its replay scratch holds
(`engine::RungReplay { own_max, carried_max, total_max }` in `ChunkFacts::act_replay`) and the
runtime picks a rung by `fire::rung_holds` against it. XLA's is stated as own T / carried T / no
total, but its prepare check counts every lane's carried rows, decode lanes included: stricter
than the stated floor, so a rung the runtime picks may be refused at prepare. engine-xla should
state the bound its static scratch shape holds (`rs_compact = false` where it lays every carried
row; `carried_max` / `total_max` as laid) and check by `rung_holds` too, so nothing the runtime
chose is refused after. Predates that branch (the runtime never counted decode lanes' replay for
XLA); after the beam-line merge.

## F65. libtpu miscompiles the affine offset as a dot in gpt-oss's multi-lane decode (found 2026-11-23)

Served gptoss-20b-u4g64 answered garbage to concurrent requests and sense to one at a time. The
u4g64 head's affine offset (`Σ_g bias[n, g] · Σ_{k∈g} x[m, k]`, a `dot_general` f32 [m, 45] ·
[201088, 45]ᵀ at HIGHEST) came out wrong in every decode of 2+ lanes: XLA fuses it as a
convolution over the bf16 biases with the add, the bf16 round trip and the in-place
dynamic-update-slice into the aliased f32 [32, 201088] logits output. The same op is right
standalone, in the 1-lane decode (where XLA makes it a reduce) and in the 2-lane prefill; the
same program returning the offset's operands as extra outputs computes it right. kernels-xla
`linear::quant::grouped_dot` now adds each group's offset in the partials' own per-group sum, with
no dot of its own (a workaround, named there; as fast as the dot it replaces, ABBA); the engine test `a_batched_decode_answers_what_each_lane_answers_alone` holds every
u4g64 mini's multi-lane decode to its single-lane runs. Repro (StableHLO only, two programs one op
apart, a JAX runner; libtpu 0.0.48, jax 0.11.2): v6e host `/workspace/tpu-tools/repro`. Filing
upstream is the user's call. Drop the workaround once a libtpu computes `decode2_dot.mlir` as it
computes `decode2_reduce.mlir`.


## F67. An XLA program is the decoder's per-layer code repeated: lower the layer stack as a loop (v6e, found 2026-11-23)

A hybrid Qwen3.5-0.8B ladder's executables are 4–65 MB packed (median prefill 34 MB, 168 MB raw;
the ladder 8.6 GB packed, 40.4 GB raw over 266 programs). No weights or tables are folded in
(every plane a parameter; 2 dense literals, 1.5 KB). The HLO text (median 8.9 MB) is 76% Mosaic
kernel bodies: 102 `tpu_custom_call`s carry 6.8 MB of `backend_config`, 12 of them distinct
(conv_step, gdn_step, gdn_chunks, pool_gather, pool_scatter 18 times each, a layer each). The
compiled form repeats at long range: zstd with a 128 MB window takes the 168 MB to 4.4 MB (38x;
the cache's level 1 34 MB), 57% of its 512-byte blocks duplicates. Loading it is deserializing
it: 856 thread-s of a record boot's loads against 78 s of reads and 56 s of unpacking. Lowering
the decoder's layers as one `stablehlo.while` over stacked per-layer weights would hold one
layer's code a program: an order of magnitude off each executable, its compile, its load and
the cache. A big lowering change with a steady-state risk on the critical path: after the PR, a
prototype on one dense model with steady-state numbers against main.

## F68. The XLA loader's width is measured each boot, not kept in the device's profile (v6e, found 2026-11-23)

Loading behind serving grows its width while the fastest of three samples of its load rate
rises (v6e: 4 or 5 of 40 cores, a first request at 9.4–9.9 s, all 266 of Qwen3.5's programs
at 74–84 s). The width is a property of the device and its PJRT plugin and belongs in the
device profile's keyed `[rates]` (device, plugin, method), measured once and committed with the
v6e profile, not found again each boot; it waits on the keyed profile path (`profile::fire::
Rates`, `measured_by`) landing with the committed profiles. Beside it: a parked program (the
host held no room for it) whose disk file is corrupt is built by its first fire, a build on the
serving path counted as `BuiltInFire`.

## F69. A fresh host cover is timed while programs load behind serving (v6e, found 2026-11-23)

The runtime times the host cover with warm runs as the load becomes ready, while XLA loads its
cached programs behind serving. With 44 loads at once the cover's frame read 21 ms against 3.7
ms quiet; at the loader's measured width (4–5) and with fires' loads ahead of it, it reads 3.7
ms. A cover the cache or a committed file holds is not timed at all. A fresh one should wait for
the loader (an engine hook the runtime calls before it times the cover); not built, as the
measured width leaves it unpolluted on v6e.

## F64

The harmony chat decoding leaves channel headers in an OpenAI chat completion's `content`, on main (ecfd6ed57)
as on the windowed-state line: gpt-oss-20b served greedy through `/v1/chat/completions` answers `content`
`"<|channel|>analysis"` beside its `reasoning_content`, and, once it reaches its final channel,
`"<|channel|>analysis<|start|>assistant<|channel|>final<|message|>**17 × 23** …"`. The reasoning is split
out (`ThinkingDecoder` on `analysis_prefix`, `chat-template/src/harmony.rs:236`), but the tokens before its
`<|message|>` stay in the content, and nothing strips the `<|end|><|start|>assistant<|channel|>final<|message|>`
header that opens the final channel: the chat decoder is the generic one (`GenericChatDecoder`, :229), which
knows only the stop ids. Open: a harmony chat decoder that drops every header a channel opens with (and keeps
only the final channel's text as content), checked through compat-openai on gpt-oss-20b. Formatting only: the
tokens drawn are the model's.

## F70. A device with no measured profile serves Qwen3.5 slower than its family's measured device (found 2026-11-23)
With no profile of its own (`--no-measure`, a clean home, the Adreno 840's committed profile unlisted), the phone
serves `qualcomm.toml`, whose numbers are not measured on any device. Every `[rates]` key is skipped and served by
the estimate. The load is safe: the submit bound holds every step, prefills ramped to ~2000 tokens, and gpufaults did
not change over 12 boots. It is slower, though. Measured ABBA on 05d724218 against the committed profile, ms/step:
- Qwen3.5: 27% slower at c1 (40.1 vs 31.5) and 4.9% slower at c4 (56.7 vs 54.1). A 1-token request takes 0.35 s vs
  0.10 s.
- Qwen3 is unaffected (c1 1% faster, c4 1.2% slower).

Cause: with no rates, the dense rungs pick a kernel by the work's shape alone. Rung 1 takes gemv x1 ("reads no more
weight … so no bandwidth or flops makes another faster"), where the measured rates pick gemv x8 for 1-64 rows on
Qwen3.5's shapes. Fix shape: a family-level committed entry, with no hard-coded default. The vendor family file
(`qualcomm.toml`) carries a measured device's crossover as an estimate, keyed so an exact-device entry (the device's
own committed or stored profile) wins over it.

## F72. ROCm's causal masked prefill: split or merged, chosen per bucket, picks wrong where live rows decide (frozen, ~1%; MI300X, 2026-11-23)
A causal masked prefill serves its requests that read no bit either split (`paged_prefill_unmasked_of`
then the skipping `paged_prefill_masked`, landed in 385b52884) or merged (one launch, the plain body
behind a branch). Split gains Qwen3-30B-A3B about 4.3 ms at pf2048; merged is about 0.75 ms faster
on Qwen3-8B served (forced, same build and home: split 71.4 / 71.9 ms, merged 70.9 / 70.9).
- A per-shape choice keyed and timed per body bucket (prototype: `profile::routes`
  `Kind::PrefillMasked`, the bucket's prefill body timed on lanes stating no mask, local
  sc/amd-prefill-split-measured, not pushed) picks split for 8B: equal or faster at every bucket
  from 64 rows (2048: 56.07 vs 56.35 ms; 2234: 58.07 vs 58.11). Its timing lane fills its bucket;
  a served 2061-row prompt runs in the 2234 bucket with about 8% of its tiles empty (block_valid).
  The tradeoff turns on how many query tiles fill 304 CUs at one block a CU against two, which a
  bucket key cannot see.
- If reopened (the bar: more than 2%): choose per fire by live tiles against a measured crossover
  in tiles per CU (stored keyed, measured once by sweeping live rows on one bucket's body), the
  pick staged as a per-fire word both kernels read, as the mask word is, so one recorded body
  serves both ways. No constant and no residency guess.

## F73. A product pick decided inside its own measured spread lands either way from boot to boot, and moved served c64 ~2% (MI300X, 2026-11-23)
Of a kind with F51 (a measured quantity that does not reproduce across boots), on the choosing side:
`profile::routes` keeps a decode product's stated kernel unless another is faster past the two
measurements' spread, but where the two kernels sit within a few microseconds of each other the
measured times land on either side of that line from one measuring boot to the next. Qwen3-30B-A3B,
two fresh homes of one build family: Normed 128x2048 m8 took the turned kernel in one (5691.4 vs
5672.4 us) and the skinny kernel in the other (5667.9 vs 5670.4 us). Served c64 between the two
homes moved about 2% (mean of 10, 6006 vs 5859 tok/s); with both builds on one shared home the
difference went away (5921 vs 5981, sd 132 / 216), so the profile's pick, not the code, carried it.
- Bounded mostly by committed profiles: a device's committed `<device>.measured.toml` fixes the
  pick once per device, so every boot of a shipped build serves the same choice. What stays open
  is a user's own measuring boot (a home with no committed entry for its key, or a code change that
  retires the committed one), which can land the other way and serve ~2% differently.
- If it matters: a pick inside the noise could keep the stated kernel unless the margin holds
  across repeated rounds (re-time the near-ties, or widen the spread to the rounds' range), so a
  near-tie stays stated instead of a coin flip; or the export can mark near-ties so a reviewer sees
  which committed picks are noise-decided.
