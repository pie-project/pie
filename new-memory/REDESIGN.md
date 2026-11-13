# Device memory: redesign direction

Draft, 2026-11-13. Based on `memory-followups` at dbb5e043d and the inventory in `DEVICE-MEMORY-INVENTORY.md`. Covers CUDA, ROCm, Metal, Vulkan, wgpu and XLA. Cerebras is out of scope.

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
- Carves for all rungs are computed at load (they are layouts, cheap). A rung's bodies are captured when a pass first asks for it, and the raise waits on the capture (it is already a planner waiter): every serving fire replays a captured body, never runs eagerly (existing user decision; Metal encodes live and is exempt). A body is a model-class cache entry (2.7).
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
- **Cache**: streamable atoms (as today), optional derived copies (fp8 shadow, prescaled banks, tied-embedding copy, decoded planes), compiled bodies per rung. Each is a `WeightShape`-like entry with a size and a cost to rebuild. This is FOLLOWUPS F1 extended from atoms to copies and bodies.
- Removes: fp8 shadow "under free/8", XLA "under 5/8 of HBM", `bodies_mem` with its quarter rule, 512 MiB floor and 1/20 stretch, `weights_streaming`.

### 2.8 One policy

The governor says how many chunks are needed; PLEX picks what gives them up among idle kv, cache atoms, derived copies, bodies. The rung is not a candidate: it follows demand (M118) and its chunks count as state room while lowered.

Unit of comparison: expected time lost (replay time × reuse likelihood for kv; refill time × use rate for a cache atom, from `weight_uses`; re-arm time for a body). All are rates the system already measures. **Open:** this unit is the least settled part of the design and needs measurement before it replaces the fixed weight rule.

### 2.9 Metal: ANE

IOSurfaces cannot be placed in our heaps, so ANE memory stays outside the span, but we size every surface and staging buffer ourselves: the ledger counts weight and scale surfaces as model and input, partial and staging surfaces as fire, exactly. Only the compiled ANE program is invisible (owned by the system daemon, not in our footprint): `outside`.

### 2.10 XLA

Same ledger, different realisation:
- `state` and `model` are counted exactly as elsewhere.
- `fire(r)` cannot be laid out by us. It is **measured**: run rung `r`'s executable once at load or on first raise and read the device's peak (`PJRT_Device_MemoryStats` is already used at open). The measurement replaces `headroom`.
- Open risk, already in XLA-TPU-CHECKLIST 3.2: whether donating the pool updates in place. If it copies, `fire(r)` includes a pool-sized term and the ledger will show it.

## 3. What remains configurable

| Key | Why it cannot be derived |
|---|---|
| `[engine.memory] budget_device` / `budget_host` / `budget_disk` (optional caps, F2 names) | sharing a machine is the operator's intent |
| `governor_*` (F2) | how politely to share is intent too |
| `max_context` (optional cap) | a product limit, not a memory one |

`max_batch_tokens` and `max_batch_sequences` (F2) become optional caps on the rung, defaulting to derived.

Structural constants that stay, each with a reason: chunk size (from the compiler's plan of the model and the device binding limit), alignment 256 B, frames in flight (pipeline depth), rung ratio 2.

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
6. **Model cache under PLEX** (F1 extended).
7. **XLA**: measured `fire(r)`.

Hardware: steps 1-5 for CUDA and Vulkan/wgpu can be done on the L40S; ROCm, Metal and XLA need their own machines.

## 5. Open questions

1. When a rung's bodies are captured: on the first pass that asks for the rung (the raise waits; unused rungs cost nothing), or all rungs in the background after load (no wait, but bodies for rungs never used, and capture shares the device with serving). Either way capture precedes the rung's first serving fire. Recommended: on first demand. Not decided.
2. Settled by the Mac review: one ledger over several buffers (2.3), for Metal as well as Vulkan / wgpu.
3. The policy unit in 2.8.
