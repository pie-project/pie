# Device memory: redesign direction

Draft, 2026-11-13. Based on `memory-followups` at dbb5e043d and the inventory in `DEVICE-MEMORY-INVENTORY.md`. Covers CUDA, ROCm, Metal, Vulkan, wgpu and XLA. Cerebras is out of scope.

## North star

> An operator loads any model on any device with no memory setting, and the engine uses the whole device: every byte is in one ledger, and how the bytes are split between weights, state and fires follows demand at run time.

How we know we are there:
1. **One ledger.** On every backend the ledger's sum equals what the device reports as used, at any time, asserted in tests.
2. **Nothing to set.** A load with an empty memory config works and is the best configuration; the only keys are optional caps for sharing a machine.
3. **Load decides one thing:** whether the floor fits. Every other split is made at run time and can be undone.
4. **No number without a derivation.** Each remaining constant is listed in section 3 with the hardware or structural fact it comes from.
5. **One implementation.** The ledger, the layouts and the fit are shared code; a backend supplies only how to reserve and back memory.
6. **No loss.** Throughput and capacity are at least main's on every backend.

## 0. The one rule

> Every device byte we allocate belongs to exactly one of three classes, is counted in chunks in one ledger, and the ledger holds `model + state + fire(rung) <= total` at all times. Nothing is decided at load except whether the floor fits.

The invariant is the **ledger**, not an address range. Five backends realise it as one address range; XLA realises it by measurement. That is what lets one design cover all of them.

What already exists and stays: `engine::chunk::ChunkFacts` (the shared contract), slots / chunks / segments, the logical-to-physical `ChunkTable`, evacuation, `copy_slots`, demand-driven rungs in the runtime (M118), weight atoms in pool slots.

## 1. Where the code stands

| Concern | Shared today | Per backend today |
|---|---|---|
| Contract | `engine/src/chunk.rs` (788 lines) | each backend fills `ChunkFacts`; `act_chunks` is empty on Metal, Vulkan/wgpu, XLA |
| Pool sizing | `engine::chunk::fit` (20 lines, clean) **and** `engine/src/fit.rs` (341 lines of page/slot balancing for the old pools) | CUDA `admit_the_card` + `refit` + `fit_the_card`; Metal `Accounting`; Vulkan/wgpu "whatever is left"; XLA `share × HBM − held − weights − headroom` |
| Device allocation | none | CUDA `elastic.rs` 1208 + `held.rs` 853 + slab map; Metal `elastic.rs` 1360; Vulkan / wgpu one buffer per owner |
| Arena layout | `poem_compiler::arena::carve` by liveness; `arena_rung` re-carves for a narrower ceiling | — |
| Kernel scratch | none | CUDA / ROCm: asked at launch through `Ctx::scratch*` (104 and 48 call sites, 25 and 20 names) from a grow-only slab map. Metal / Vulkan / wgpu: `scratch.rs` (679 / 626 lines) computes "rooms" from (trace, budgets, compiled) at load. XLA: inside the executable |
| Staged inputs | none | `inputs.rs` four times (1553 / 1126 / 1037 / 265 lines), the same layout formula |
| Rungs | `ACTIVATION_RUNGS = [256, 1024, 4096]`, plus a second ladder `ARENA_RUNG_TOKENS × ARENA_RUNG_LANES` | CUDA only |
| Policy | runtime `store/shared.rs`, `store/weights.rs` (fixed rule for weights, FOLLOWUPS F1) | — |

Two observations drive the design:

1. **Metal and Vulkan/wgpu already compute kernel scratch statically** from the compiled model; CUDA computes the same thing at launch time. The information is the same; only CUDA's form forces growth, retirement, warm-up and the "unwarmed" refusal inside a capture.
2. **The compiler already lays out the arena by liveness for any ceiling.** Scratch and staged inputs are the same kind of object (bytes that are a function of the fire's rows and lanes) and are simply not handed to it.

## 2. The design

### 2.1 `engine::memory`: the ledger (new, pure logic)

```
enum Class { Model, State, Fire }

struct Ledger {            // one per device
    chunk_bytes: u64,
    total: u32,            // measured, see 2.2
    model: u32,            // floor + resident cache atoms
    state: u32,            // chunks the runtime has handed out
    fire:  u32,            // fire(rung) of the current rung
}
```

- Every backend allocation goes through one entry that takes `(Class, owner, bytes)`. In debug and in tests the ledger's sum is compared with what the device reports as used.
- `ChunkFacts` becomes the serialised view of the ledger. `act_chunks` is stated by every backend. `PoolFacts`, `ChunkFacts::degenerate` and `engine/src/fit.rs` go.
- This is FOLLOWUPS F6 (shared pure helpers) with a purpose.

### 2.2 `total` is measured, never configured

`total = (device free after context, handles and modules are up − outside) / chunk_bytes`, capped by `[engine.memory] budget_device` when the operator states one (the single legitimate knob: sharing a device with another process cannot be derived).

`outside` is what the driver allocates beside our span after that point (AMD review, MI300X): graph body instantiation (Qwen3.5-9B: 845 MiB for 161 bodies, about 11.8 KB a node), code objects of modules loaded on first use, BLAS handle workspaces we do not set, collective buffers under TP / EP. It is a fourth ledger term, **priced by measurement, not guessed**: bodies are `nodes × price`, the node count known from the compiled regions and the price measured on the first instantiation (the code already measures it); a module's bytes are the free-memory delta of its load. A flat span cannot give physical memory back, so on flat backends the span is reserved after the first body has been priced, and a rung's bodies are priced before the rung is offered.

Removes: `device_memory_share`, `device_kv_budget`, the `min(128 MiB, total/10)` floor, XLA `headroom`, `[wgpu] device_memory` default of 8 GiB.

### 2.3 One backend trait for memory

```
trait Span {
    fn capacity(&self) -> u64;                 // measured free bytes
    fn bindings(&self) -> u64;                 // largest single buffer (Vulkan, wgpu, Metal)
    fn reserve(&mut self, chunks: u32);        // the whole span, once, in as many ranges as the device needs
    fn back(&mut self, chunks: Range<u32>);    // optional: default no-op
    fn unback(&mut self, chunks: Range<u32>);  // optional: default no-op
}
```

| Backend | `reserve` | `back` / `unback` |
|---|---|---|
| CUDA | `cuMemAddressReserve` | `cuMemCreate` + `cuMemMap` / unmap |
| ROCm | one `hipMalloc` | no-op |
| Metal | N placement-sparse buffers under one budget (today: two buffers, two budgets) | heap map / unmap |
| Vulkan, wgpu | N buffers of at most `bindings()` | no-op |
| XLA | the pool tensor | no-op |

**Flat is the base case.** Class boundaries are ledger numbers, so moving the boundary between state and fire costs nothing on a flat span. Today the ROCm range is allocated at its first commit, sized by the fit made before arming, so everything that fit held for arming and arming did not use (the bodies stretch, about 7 GiB on Qwen3-8B, the arena-sized scratch guess) is lost to the cache for the life of the process: the 4.6% fewer pages than main. With boundaries as ledger numbers there is nothing to lose. Backing on demand only matters for giving memory back to other processes; it is an optimisation of two backends, not part of the model. This removes the present ROCm oddity where lowering a rung frees nothing: state simply takes the chunks.

**The span is one ledger, not necessarily one address range** (Mac review). A Metal buffer is capped at `maxBufferLength` (18.7 GiB on an M1 Max with a 25 GiB working set), so one range would put about 6 GiB out of reach; Vulkan and wgpu bind at most four segments addressed by a u32 element index. On these the span is several buffers that draw on one budget: chunk ids are global, a chunk never straddles a buffer, and the fire carve sits in the top buffer. Metal's `elastic` then needs one change: buffers that share a budget may be mapped in one batch (refused today), and tiles may be mapped at any range, not only as a prefix.

Deleted: CUDA device heap, recycle shelf, endpoint pool, kernel slab map, raw `cudaMalloc` sites; Metal's second sparse buffer and `Accounting`; per-owner buffers on Vulkan / wgpu.

### 2.4 Fire: the compiler states the whole footprint

`fire(r) = frames × FrameMap(r).bytes + ArenaMap(r).bytes`, in chunks. Both maps come from the compiler.

- **Workspace placements.** Each kernel family exports what it needs as data: `workspace(op, rows, lanes, tuning) -> [(name, rows, width, dtype, scope)]` (`tuning` is the device's kernel tuning: Metal's split-K bound and MoE register rooms depend on it), scope one of `launch` (dead when the op returns), `stream` (read by the next launch on the same stream, e.g. the decoded weight tile), `fire`. `arena::carve` lays them with the values. `Ctx::scratch(name)` becomes a lookup of an address the carve fixed.
  - CUDA / ROCm: 25 / 20 names to convert. Removes the slab map, both grains, growth and retirement, the warm-up fires, `Unwarmed`, `scratch_growth`, `decoded_weight_reserve`, `gemm_workspace_mb` (cuBLAS takes the workspace by pointer already).
  - Metal / Vulkan / wgpu: `scratch.rs` rooms move into the same exported form; the two files go.
- **`FrameMap`.** The staged-input layout (tokens, positions, page tables, masks, rs tables, attention plan workspaces) and the readout rows, as one shared function of `(rows, lanes, pages per lane, context)`. Replaces four `inputs.rs` layouts. A ring of `frames` of them sits in the fire carve.
- **Data-dependent tables** (group masks, mask and inject runs: sized by a fire's composition, today regrown and the old buffer kept) get a bound from the rung in the `FrameMap`; no growth after load.
- **Model bytes inside fire buffers** are split out first: the n-gram hash constants in the Metal and Vulkan/wgpu scratch plane are model-class, and a re-carve would overwrite them.
- **Guest program scratch** is a `fire`-scope workspace of the program's stage, sized by its own plan; `program_scratch_reserve` goes.
- Lanes are not a second axis: a rung of `r` tokens allows `r` lanes (as `arena_rung` already does). `ARENA_RUNG_LANES` goes.

### 2.5 Rungs: powers of two, the largest derived

- Rungs are every power of two from the smallest useful fire up to the largest `r` with `floor + one working set + fire(r) <= total`. `chunk::fit` already checks this inequality; it is solved for `r` instead of halving a configured token count.
- Carves for all rungs are computed at load (they are layouts, cheap). Every rung's bodies are captured in the background after load (user decision, 2026-11-13), smallest rung first; the model serves at the rungs already armed meanwhile, and a raise to a rung still being captured waits on it. Every serving fire replays a captured body, never runs eagerly (existing user decision; Metal encodes live and is exempt). Bodies are therefore a fixed model-class cost, known once arming ends, not a cache entry.
- Because every rung is armed, the number of rungs multiplies arming time and body memory: the rung ratio (2 here) is to be set by measuring both, not assumed.
- Removes: `ACTIVATION_RUNGS`, `max_forward_tokens` as a sizing input, `tokens_within`, `TOKENS_FLOOR`, `working_set_percent`, `walk_fire_rows`, the lane/token refit.
- The runtime side (M118: rest at the smallest rung, raise on demand as a planner waiter) is unchanged.

### 2.6 State: no exceptions outside the pool

New slot roles beside `State` and `RsBuffer`:

| Today | Becomes |
|---|---|
| Channel rings, endpoint cells | fixed shape, role `Channel`, held by an instance |
| Guest per-instance tables | fixed shape, role `Instance` |
| Adapter banks (written after load, not in the model file) | fixed shape, role `Adapter` |
| Compressor side planes keyed by physical page (Vulkan/wgpu) | rows of the list slot they describe |
| Weight-cache tables and hit counters | reserved run, sized by the weight shapes |

One attribute on a role: **replayable** (kv, rs: may be dropped and rebuilt) or not (channel, adapter: may only be parked on the host tier). The existing rule "the runtime never drops state on its own; PLEX decides" covers both.

### 2.7 Model: a floor and a cache

- **Floor**: what one fire cannot run without. Dense weights stay resident (existing user decision). Load places the floor and refuses if it does not fit. That is all load decides.
- **Cache**: streamable atoms (as today), optional derived copies (fp8 shadow, prescaled banks, tied-embedding copy, decoded planes). Each is a `WeightShape`-like entry with a size and a cost to rebuild. This is FOLLOWUPS F1 extended from atoms to copies and bodies.
- Removes: fp8 shadow "under free/8", XLA "under 5/8 of HBM", `bodies_mem` with its quarter rule, 512 MiB floor and 1/20 stretch, `weights_streaming`.

### 2.7a The weight cache without a mode or a threshold

Today (`runtime/src/store/weights.rs`): `weight_streaming = auto | on | off`; `on` sheds the whole cache to its floor before anything else gives way; refill only into room no waiter is short of; order by the use record; membership changes once a window of `WINDOW_TICKS = 4` governor ticks. Rimuru (arXiv 2610.40305) has the same principle, attention state first and experts into what is free, realised with thresholds: shrink below 2 free pages, expand above 9, at most 5 experts a pass, an offline routing profile, a configured floor; it reports slow recovery after a burst.

The principle both share is the operating system's page cache: a weight atom is a clean page (read-only, a copy on the host), so dropping it loses no work, only future misses. Three rules, no mode and no threshold:

1. **Live demand before cache.** Running working sets and the fire's rung are never refused for the sake of a cache entry. The governor's `Demand { wanted, live }` already draws this line. `weight_streaming` goes: when everything fits, every atom is resident (today's `off`); when live demand needs chunks, atoms go (today's `on`). The engines switch their held fast path on their own (F1).
2. **Among cache entries, lowest value per byte goes first**, value = use rate × miss time, both measured: for an atom, uses a second from `weight_uses` × the time to stream it from the host; for idle kv, the reuse the policy's retention key stands for × the time to replay its tokens. This is the cost-aware cache (GreedyDual-Size), the standard rule for entries of different sizes and miss costs, with a competitive bound; it replaces "weights always first". The engine publishes the facts; PLEX may reorder by key (2.8).
3. **An atom comes back at break-even**: when the miss time it has cost since it left reaches the time to load it (the ski-rental rule, at most twice the cost of the best choice in hindsight). This replaces the window constant, and gives recovery after a burst an order: the atoms that have cost most come back first.

Decided by the user (2026-11-13): rules 1 and 3 first; rule 2 after measurement. Rule 2 across weights and idle kv needs a reuse estimate for idle kv where no policy sets a key; whether the runtime records prefix-cache hits has not been checked. Rules 1 and 3 need nothing new and already remove the mode key and the window constant; rule 2 can follow.

### 2.8 Governor, PLEX and the pool: unchanged roles, fed by the ledger

SPEC decision 7 stands: the governor decides how much, PLEX decides what stays, the pool decides how it moves, and the runtime never drops state on its own. The redesign adds no role and no PLEX verb; it changes where their numbers come from.

| Role | Interface today | What it reads after the redesign |
|---|---|---|
| Governor (`runtime/src/governor.rs`) | `PoolTarget`: `floor`, `hard_limit`, `held`, `demand {wanted, live}`, `growth`, `retarget(segments)` | `hard_limit` = measured capacity − `outside` − model floor (2.2), no longer a share of the card. `floor` = one working set + the smallest rung's `fire` (as its doc already says). `held` and `demand` are ledger sums. `growth` is `Fixed` on a span that cannot be unbacked (ROCm, Vulkan, wgpu, XLA) and `Elastic` on CUDA and Metal, as today. |
| Ledger | — | `total` at any moment is `min(capacity, governor target)`. The invariant `model + state + fire(rung) <= total` is checked against that. |
| PLEX (`plex:policy`, `runtime/src/policy/pressure.rs`) | one `tier-pressure` edge a band crossing; worker attrs `{device,host}-{total,state,live,wanted,act,short}-bytes`; answers with `discard` and retention keys | The same attrs, now exact because every byte is in the ledger (`act` is `fire(rung)` whole, not the arena alone). Two attrs added, `model` (cache bytes) and `outside`; attrs are not verbs. |
| Pool | segments, evacuation, `copy_slots` | unchanged |
| Rung | demand-driven; a raise is a planner waiter (M118) | unchanged; a raise short of room raises `tier-pressure` like any waiter |

What PLEX may be asked to give up:
- **Replayable state** (kv, rs, prefix entries, snapshots): as today, by retention key; the default responder discards lowest key first.
- **Model cache** (FOLLOWUPS F1): weight atoms and optional derived copies become holders with retention keys. The engine publishes facts for each (bytes, uses from `weight_uses`, refill cost); how to weigh them is the policy's, exactly as for kv. The fixed rule in `store/weights.rs` (shed every weight to the floor first) goes.
- **Never offered**: the floor, bodies, and state that cannot be replayed (channels, adapters). Unreplayable state may only be parked on the host tier.

So the redesign defines no valuation of its own. An earlier draft proposed a common unit ("expected time lost"); that is a policy's business and is dropped.

Not addressed here: the governor's own loop constants (`FIRST_HOLD`, `IDLE_AFTER`, `KEEP_CACHE`, `CUT_SHARE`, and the tolerance table), which its header says were measured on one 16 GB machine. They act only when the process is not dedicated, and they size nothing; FOLLOWUPS F2 turns the tolerance into `governor_*` keys.

### 2.9 Metal: ANE

IOSurfaces cannot be placed in our heaps, so ANE memory stays outside the span, but we size every surface and staging buffer ourselves: the ledger counts weight and scale surfaces as model and input, partial and staging surfaces as fire, exactly. Only the compiled ANE program is invisible (owned by the system daemon, not in our footprint): `outside`.

### 2.10 XLA

Same ledger, different realisation (measured on a v6e by the TPU session, 2026-11-13):
- `state` and `model` are counted exactly as elsewhere. Pool donation updates in place: `alias_size` is the pool's size and `bytes_in_use` moves only by code and non-aliased outputs, so there is no transient pool-sized copy (XLA-TPU-CHECKLIST 3.2 closed).
- `fire` is known exactly **at compile time**, from `PJRT_Executable_GetCompiledMemoryStats`, before any run:
  - executable temporaries live in one scratch region the runtime reserves at the bottom of memory, sized to the largest `temp_size` over loaded programs; it only grows, and it is invisible to `bytes_in_use` and its peak (reported as `bytes_reserved`). So `fire = max over loaded programs of temp_size` + the non-aliased outputs (`output − alias`: readout planes, gather results, carried guest cells).
  - generated code joins `bytes_in_use` at a program's first execute and stays: Σ `generated_code_size` over loaded programs is a model-class cost, like bodies.
  - predicted against measured change in `bytes_in_use`, four programs: within 0.1 MiB.
- This replaces `headroom` (HBM/16, 2 GiB on v6e, where the largest temp on the two models measured was 140 MiB). An earlier draft proposed sampling the peak around one execute; that cannot work (the peak is a lifetime high-water mark and does not see temps).
- What cannot hold on XLA: the pool tensor's size is in the program text, so the state / fire boundary is decided at load (moving it means a new pool and recompiling every program); the scratch region never shrinks, so lowering a rung frees nothing and the ledger charges `fire` at the largest rung loaded; a rung must be compiled off the fire path and offered only once ready (which the background arming of 2.5 already does).

## 3. Configuration

Memory sizing reads **no key by default**. Everything below is optional; an empty `[engine.memory]` is the best configuration for a dedicated machine.

| Key (`[engine.memory]`, FOLLOWUPS F2 names) | Default | Why it cannot be derived |
|---|---|---|
| `budget_device` | all the device has | sharing a device is the operator's intent |
| `budget_host` | what the governor allows | same, for host memory |
| `budget_disk` | none without `disk_path` | same, for disk |
| `disk_path`, `disk_persistent` | no disk tier | where files go is the operator's |
| `governor_backoff`, `governor_release_idle` | dedicated: `1.0`, `false` | how much to yield when the system is congested, and whether idle memory goes back |

Seven keys, none required. `governor_patience` (F2) goes (user decision, 2026-11-13): in `Governor::tick` it is the count of congested ticks before the backoff acts, a debounce of the pressure signal rather than an intent, so it becomes one fixed value ( today's profiles tie it to the backoff: 0.9 → 4 ticks, 0.8 → 2, 0.7 → 1). Limits that are product decisions rather than memory (`max_context`, and F2's `max_batch_tokens` / `max_batch_sequences`) stay outside `[engine.memory]` as optional caps; when absent the context is the model's and the batch is the largest rung.

Removed, each refused with a message naming what replaced it:

| Key | Replaced by |
|---|---|
| `device_memory_share`, `[wgpu] device_memory` | measured capacity; `budget_device` to cap |
| `device_kv_budget`, `host_kv_budget`, `disk_kv_budget` | the ledger; `budget_*` to cap a tier |
| `disk_kv_dir`, `disk_kv_persistent` | `disk_path`, `disk_persistent` |
| `memory_tolerance` | `governor_*` |
| `weight_streaming` | weights are holders (2.7, F1) |
| `snapshot_budget` | snapshots are disk-tier holders (F1) |
| `max_forward_tokens`, `max_forward_requests` | the derived largest rung (2.5) |
| `max_model_len` | `max_context` |
| `max_state_slots`, `kv_page_size`, `[xla] seat_tokens` | the compiler's chunk plan |
| `bodies_mem` | bodies priced by measurement (2.2) |
| `working_set_percent`, `window_spare` | `fire(rung)` stated by the compiler (2.4) |
| `gemm_workspace_mb` | the workspace declared by the kernel (2.4) |
| `[xla] headroom` | measured `fire(r)` (2.10) |
| `adapters.seats` | adapters are state slots taken on registration (2.6) |
| `max_patches`, `max_images`, `max_voxels`, `max_clips` | rungs on the patch and voxel axes, derived like the token rung (user decision: no product limit; a request is refused only when it is past the largest rung the device can hold) |

Structural constants that stay, each with its reason: chunk size (the compiler's plan of the model and the device's binding limit), alignment 256 B (kernel requirement), frames in flight (pipeline depth, a runtime setting, not a memory one), rung ratio (to be measured, 2.5).

## 4. Order of work

Each step is mergeable alone. Steps 1-5 are verified bit-for-bit with the golden logits; step 6 by throughput.

1. **Defects** from the inventory (Vulkan counter, wgpu uniform arena, `device_kv_budget` check, recycle shelf before the fit).
2. **Ledger, no behaviour change.** `engine::memory`, class tags on every allocation in every backend, the sum check in tests. This also finishes the inventory.
3. **Fire, CUDA first, then the rest.**
   a. workspace placements in the compiler; convert the 25 CUDA names; delete the slab map.
   b. `FrameMap`; inputs and readout into the carve.
   c. ROCm (20 names), then Metal and Vulkan/wgpu (`scratch.rs` → exported workspaces, arena into the span, `act_chunks` stated).
4. **State stragglers** into the pool (channels, instance tables, adapters, side planes).
5. **Derived rungs and measured total**; delete the load-time fit, `engine/src/fit.rs`, and the config keys, each refused with a message.
6. **Model cache under PLEX** (F1 extended) and the config surface (F2, section 3).
7. **XLA**: measured `fire(r)`.

Hardware: steps 1-5 for CUDA and Vulkan/wgpu can be done on the L40S; ROCm, Metal and XLA need their own machines.

## 4a. How we work on this (user, 2026-11-13)

1. No full test suites and no verification done only to feel safer.
2. An experiment needed during implementation is set up to run fast; no end-to-end runs for it.
3. Clean implementation first; refactoring is approved where it is needed; verification may be done once per package, at its end.

## 5. Open questions

1. Settled by the user: every rung's bodies are captured in the background after load (2.5). Open in its place: the rung ratio, by measuring arming time and body memory per rung.
2. Settled by the Mac review: one ledger over several buffers (2.3), for Metal as well as Vulkan / wgpu.
3. Settled: no valuation in the engine; PLEX weighs the facts the ledger publishes (2.8).
4. Settled by the user: `max_patches` / `max_images` / `max_voxels` / `max_clips` go; derived rungs, only abnormal requests refused.
5. Settled by the user: weight cache rules 1 and 3 now, rule 2 later (2.7a); governor is two keys (section 3).

## 6. Decisions during implementation (2026-11-13)

- **Workspaces one size.** Kernel workspaces stay one set sized for the top rung (`fire(r) = carve_r + W + rs prefix`); per-rung workspaces would overlap slab addresses that must persist, for about 1% of state on one family (L40S, measured W: 11 / 40 MiB MoE, 429 MiB GDN hybrid).
- **Rungs derived from 1.** Every power of two of tokens from 1 up to the top; no `ACTIVATION_RUNGS`, no max_lanes floor. A load arms every rung through the widest decode fire; only wider prefill rungs arm in the background.
- **Background arming contract** (`engine::live`): `Engine::arming_room() -> ArmingRoom { pages, records }`, `Engine::arm_step(&ArmingLease) -> Result<bool>`. The runtime lends free pool slot refs per step (never chunk 0, never past the allocator ceiling, records named by their pieces' refs). One composition per step; smallest rung first, monotone `Live::arm`; a refused composition is `Err`, and `Live` reaches `MAX` only when every planned composition holds a body. A pass never waits on arming: it fires at the widest armed rung.
- **Held.** `Live::set_held(bytes)` is the part of the stated range the engine's fit keeps back while arming (`[fitted, stated)`, directly below the activation span). Governor hard limit = range − held; the allocator refuses chunks in the held region. `Live::outside` is the ledger's outside class, published to PLEX only (the stated range already excludes it). On flat ROCm, held is 0.
- **Guest state (step 4).** Option A with D for items past a slot (user): `Instance` 4 KiB and `Channel` 1 KiB fixed shapes keyed by role; `Footprint { bytes, slots, contiguous }`; contiguous items get adjacent slots of one chunk; an item past a chunk is refused. Non-replayable slots park to the host tier only; `Engine::rebind_slots(Rebind)` after unpark; the runtime holds and replays host writes (publish / take) to a parked owner. Measured on Metal: paging a 594 KiB cell through 1 KiB slots costs +100-160%, hence D.
- **XLA expert streaming (A7).** No cut fires: streamed banks are pinned-host PJRT buffers; resident atoms in u32 pool slots as K/128 panels read by a pooled gmm arm; misses staged host->pool inside the fire (zero misses, zero transfer). Held keeps the Slice route at decode 1 unless gmm matches it.
- **No legacy paths** (user): a replaced path is deleted in the same package, not kept as a fallback.

## 7. North star on every backend (redesign matrix, 2026-11-14)

A step is done only when it holds on every backend (user). ✅ verified on hardware · 🟡 in progress · 🔴 not started · — not applicable (reason given) · ? to be confirmed by the owner.

| Step | CUDA (L40S) | ROCm (AMD) | Metal (Mac) | Vulkan (New Pie) | wgpu (New Pie) | XLA (TPU) |
|---|---|---|---|---|---|---|
| 2 Ledger, audit gap ~0 | ✅ | ✅ 0 MiB | ✅ 88 KiB | ✅ 0 | ✅ 0 | 🔴 |
| 2.2 Measured capacity, no 0.9 share, floor 64 MiB | 🟡 share still read | 🟡 via CUDA path | ✅ (floor ?) | ✅ | ✅ (heap budget where reported) | 🔴 memory_share × bytes_limit − held − weights − HBM/16 headroom; both go |
| 2.2 Outside priced, nothing allocated after load + arming | 🟡 guest Prepared scratch left | 🟡 same | ? | 🟡 pipelines priced; guest scratch ? | 🟡 same | 🟡 temps live in one shared reservation (bytes_reserved) that grows at program load and never shrinks; check bytes_reserved and bytes_in_use equal after warm-up and after serving (WARM_STRICT) |
| 3 Fire footprint declared at load | ✅ size-only walk | ✅ walk | 🟡 scratch.rs static, not in span | 🟡 scratch.rs static, not in span | 🟡 same | ✅ compile stats |
| 3c Activation in the span, `act_chunks` stated, rungs lend to state | ✅ | ✅ flat | 🔴 | 🔴 | 🔴 | 🟡 reopened (TPU, real 30B): fire temp 254 MiB at 4K tokens, ~1 GiB at 16K, mostly MoE intermediates in kernels we own; plan: the gmm gathers its rows and writes outputs into pool chunks named by a table, stated as act_chunks per rung (bitwise; only 15–30 MiB stays XLA's) |
| 3 Guest program fire footprint (ProgramFacts) | 🟡 contract in work | 🟡 | 🔴 | 🔴 | 🔴 | 🔴 guest executables' compiled stats at registration (with measured fire) |
| 4 State stragglers in slots (Instance / Channel) | 🟡 b46d16059 | 🟡 gate | ✅ | ✅ | ✅ (no device channel state) | — proposed none |
| 5 Derived rungs, background arming | ✅ | ✅ | — encodes live (no capture) | 🔴 rungs; recorded frames per shape ? | 🔴 | 🟡 warm ladder |
| One budget over N buffers (Span) | ✅ one range | ✅ one range | 🔴 two buffers | ? | ? | ✅ one pool |
| Legacy paths deleted | 🟡 | 🟡 slab growth | ? | ? | ? | 🟡 cut path gone; A7 debug switches, host expert fill, host token-table misses go with A7; kernel-off switches (MOE_GMM, KV, FLASH_PREFILL, OLDER_DMA, POOL_DMA, NORM, ROPE, GDN, CONV, GUESTS_ON_HOST) go after A7 |
