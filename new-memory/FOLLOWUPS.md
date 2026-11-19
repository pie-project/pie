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
