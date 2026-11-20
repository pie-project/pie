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

## F42. Lanes carry page deltas; each engine keeps a per-slot page table resident

Every fire the runtime hands each lane its full page list (`Lane.tables`), so every
backend walks every lane's pages every fire to lay its page tables. On XLA the paged
walks read a dense `[lanes x max_pages]` table the device gathers from the CSR list each
fire, at 6-9 ns an entry (24-64 µs a fire at today's shapes, ~200 µs at a fixed lane
capacity of 128 lanes x 256 pages). Laying that table on the host instead, beside the CSR
walk serve already does, was measured and withdrawn: about 4 ns an entry on the host at
serving shapes (host stage of a prefill-sized fire 0.187 -> 0.253 ms), the device run no
shorter (19.17 -> 19.37 ms), and mixed serving 2-4% slower (v6e qwen3-0.6b: staggered
mixed load -3.6%, decode beside long prompts -4.0%). Without a resident table every
layout of it is O(pages) a fire on one side or the other.

Shape (GPU's):
- each Lane carries its slot index and its page deltas: the pages granted and freed
  since the slot's last fire, in order;
- each engine keeps a per-slot device table, laid at load and sized by the pool's max
  pages per slot (no allocation after load), and applies the deltas in the fire's own
  command stream;
- a slot reused by another set gets a reset delta;
- the host walk becomes O(deltas) a fire.

A coordinated change (the CUDA, engine-gpu, Metal and XLA halves), with a correctness
test: a table built from deltas equals one built from the full list after arbitrary
grant / free / reuse sequences. AMD's beam page lists already keep page state resident
for beams; the two should converge. Paired with XLA's fixed-capacity lowering, where the
table grows with the lane capacity.
