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
disk_persistent, governor_backoff / governor_patience / governor_release_idle (replacing memory_tolerance profiles:
dedicated = 1.0 / never / false, aggressive = 0.9 / 4 / true, balanced = 0.8 / 2 / true, polite = 0.7 / 1 / true);
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
