# Storage Refactor and Metal Weight Loading

> **The runtime compiles every checkpoint into a `LoadPlan`; every driver
> only executes one.** Drivers state device facts, the runtime decides policy,
> and bulk weight bytes never cross the boundary.

This plan finishes what the load-planner migration started
(`runtime/load-planner/src/inproc.rs`, `worker/src/embedded_driver.rs`): the
runtime-side compile becomes the *only* compile path, driver creation is split
from model load so the compiler can see device facts, the C++ compile bridge
and the crate's FFI surface are deleted, and Metal is ported from its own
hand-rolled loader to the plan-executor model. It deliberately supersedes the
"Metal keeps its own loader; no Rust load-planner adoption in this plan"
carve-out in `cpp-refact-metal.md` — that plan scoped adoption out of the file
moves; this plan is the adoption.

Companion documents: [boundary.md](boundary.md) (steady-state runtime–driver
contract), [cpp-refact.md](cpp-refact.md) (CUDA source layout),
[cpp-refact-metal.md](cpp-refact-metal.md) (Metal source layout).

---

## 1. Where we are

One Rust planner (`runtime/load-planner`), two invocation paths, three
backends in three different states:

| Backend | Compile | Execute | State |
|---------|---------|---------|-------|
| CUDA | Runtime in-proc planning when possible, else C++ parse + legacy FFI in-driver | `load_plan_executor.hpp` interprets the plan (file IO, staging, H2D, transcode) | Variant A live, with fallbacks |
| Metal | none — no `LoadPlan` involvement | `heap_bind` / `heap_layout` / `safetensors_view`: mmap → memcpy into one resident `MTLHeap`, per-model name registry baked into `heap_bind_names.cpp` | independent loader |
| Dummy | none | none (no weights) | no coverage of the plan path |

The two CUDA paths run the *same* Rust compiler; the difference is
compile-via-FFI vs deserialize-from-file (`load_plan_bridge.hpp`, and the
doc fix already recorded in `cpp-refact.md` Phase 6). The locality switch is
legacy plan-path key presence in the boot TOML.

**Why the fallback path still exists.** the old CUDA compile helper
(`worker/src/embedded_driver.rs:265`) bails to the in-driver compile in three
cases, all rooted in one ordering defect: the plan is compiled *before* the
driver exists, so the runtime cannot ask the device anything.

1. `runtime_quant="fp8"` — needs the `fp8_native` device query.
2. `mxfp4_moe="native"` — needs a Blackwell (`sm_100+`) query.
3. Rust `parse_model_config` failure — silently masked by the C++ `HfConfig`
   parse feeding the same compiler.

A secondary smell with the same root: the worker mirrors driver constants it
cannot query (`CUDA_MAX_TILE_BYTES`, `CUDA_PREFERRED_ALIGNMENT` copied from
`loaded_model.cpp`; Metal's `heap_layout.hpp` hardcodes 256-byte alignment).

**Why Metal's loader must go.** `heap_bind_names.cpp` hardcodes checkpoint
facts for one model family (the `.language_model.` infix, the tied 4-bit
`lm_head` triplet bound to both Embed and QmvLmHead, the GDN
`in_proj_a/in_proj_b` dense-bf16 exceptions, the absent `conv1d.bias`, the
`model.visual.*`/`mtp.*` skips). That is exactly the per-model knowledge the
Rust semantic graph already owns for CUDA. Every new model on Metal currently
means editing a C++ name registry.

---

## 2. Target architecture

```mermaid
sequenceDiagram
    participant W as worker (boot)
    participant C as load-planner (runtime crate)
    participant D as driver

    W->>D: create(context config)         %% device select, CUDA ctx / MTL4 ctx, NCCL
    D-->>W: device facts (JSON)           %% fp8_native, sm_100, alignment, tile, UMA
    W->>C: compile(snapshot, config, StorageTarget from facts)
    C-->>W: serialized LoadPlan           %% placement + transforms; no payloads
    W->>D: load_model(plan bytes)         %% driver executes: reads payloads, places
    D-->>W: model capabilities (JSON)     %% arch_name, total_pages, rs_cache_*, ...
```

Division of labor (the dumb-driver principle applied to storage):

| Concern | Owner |
|---------|-------|
| Device facts (`fp8_native`, `sm_100+`, alignment, tile bound, UMA, page size) | driver, reported at create |
| Policy (quant fallback, MXFP4 lowering, layout, sharding, packing, tiling) | runtime compiler |
| Mechanism (payload IO, staging, H2D or UMA memcpy, transcode kernels, residency) | driver executor |
| Checkpoint *header* parsing (safetensors/GGUF metadata, HF config) | runtime compiler |
| Checkpoint *payload* reading | driver executor, from its own copy |
| KV/state/scratch pools, page counts | driver at load (unchanged) |

The value invariant is unchanged from Variant A: the plan records where
each tensor lives (file id + offset + span) and how to place and transform it;
**bulk weight bytes never cross the boundary in either direction.**

---

## 3. Locked decisions

- **S1 — One compiler, runtime-owned.** The crate moves to
  `runtime/load-planner` (consumers: `worker`, tests). The in-driver compile
  path is deleted, not deprecated. Layering becomes honest: today a `driver/`
  crate is linked by `worker/`.
- **S2 — Create and load are separate boot calls.** `create` brings up the
  device context (device select, CUDA ctx / MTL4 ctx, NCCL) and loads no
  model. `load_model` takes serialized plan bytes and executes them. Both
  are blocking boot-time calls, explicitly *outside* the B1 steady-state
  contract of boundary.md (B1 governs bounded per-step verbs; model load is
  seconds long, as `create` already is today).
- **S3 — Capabilities split with the calls.** Create returns *device facts*
  (new, small JSON). `load_model` returns the existing `DriverCapabilities`
  payload unchanged in shape — its fields (`arch_name`, `total_pages`,
  `rs_cache_*`, `max_forward_*`) are model-derived and cannot exist before a
  model is loaded.
- **S4 — Facts, not policy.** The driver never decides a lowering. Today the
  CUDA driver *clears* `runtime_quant="fp8"` on non-FP8 devices; after S4 the
  runtime reads `fp8_native=false` and chooses the fallback itself. Mirrored
  constants die: `StorageTarget.max_tile_bytes` / `preferred_alignment` come
  from the facts payload, never from copied `#define`s.
- **S5 — Bulk bytes never cross.** Unchanged invariant, restated because it
  shapes S7/S8: the plan is placement + transforms, payloads are read
  driver-side.
- **S6 — LoadPlan-present is the only switch, then no switch.** During
  migration, `load_model` with plan bytes executes them and without bytes
  runs the backend's legacy loader (generalizing today's
  legacy path-key switch). End state: bytes are mandatory,
  the legacy branch is deleted.
- **S7 — Metal is an arena backend.** The single resident heap stays. The
  *weights region's internal layout and size* become plan-owned
  (arena-relative offsets = heap offsets); KV/State/Scratch/IO regions remain
  driver-owned (weights are model facts; the rest is runtime state). This is
  the same split `cpp-refact-metal.md` already chose for `heap_layout.hpp`
  (weights half → `loader/`, cache halves → `store/`).
- **S8 — Copy-once into a wired resident heap; no file-backed GPU access.**
  The UMA zero-copy temptation (`newBufferWithBytesNoCopy` over the mmap) is
  rejected: safetensors payloads are not 16 KB page-aligned per tensor, and
  file-backed pages fault on first GPU touch and evict under memory pressure,
  making decode latency hostage to the pager. The existing resident-once
  invariant (I2, `mtl4_context.mm`) exists precisely to prevent this. mmap
  stays as the zero-copy *read* side during load only.
- **S9 — Mock-first.** The dummy driver executes `LoadPlan`s against a
  host-memory arena before either GPU backend changes. One conformance suite,
  three backends (boundary.md §10 house rule).
- **S10 — The LoadPlan format is versioned at deserialize.** The serialized
  plan carries the `compiler_version` source hash; the executor rejects a
  mismatch. Embedded drivers
  ship in the same binary so skew is impossible today; the check is cheap
  insurance for any future split.

---

## 4. Boundary changes

This is an ABI change and is versioned as one (`PIE_DRIVER_ABI_VERSION`
bump). The freeze recorded in `cpp-refact.md` applied to that refactor's
moves; this plan is a deliberate, planned change to the frozen surface.

### 4.1 Create

`pie_cuda_create` / `pie_metal_create` keep their shape (config blob in, caps
out — the channel already exists at `runtime/engine/src/driver/backend/cuda.rs:39`)
but the semantics narrow: no model load. The boot TOML's `[model]` section
loses the legacy plan-path key and every load-policy knob the planner now owns
(`runtime_quant`, `mxfp4_moe` move to compile inputs); `snapshot_dir` stays
(the executor reads payloads from it).

### 4.2 Device facts payload (new, returned by create)

```jsonc
{
  "abi_version": 1,
  "backend": "cuda" | "metal" | "dummy",
  "unified_memory": false,
  "fp8_native": true,          // CUDA: device FP8 support
  "native_mxfp4_moe": false,   // CUDA: sm_100+ (Blackwell)
  "storage_alignment": 256,    // CUDA: arena alignment; Metal: heapBufferSizeAndAlign
  "storage_max_tile_bytes": 67108864,
  "page_size": 16384           // Metal: host/GPU page granularity
}
```

Facts only. If a field encodes a decision rather than a queryable property,
it does not belong here (S4).

### 4.3 load_model

```rust
// Typed boot call, blocking, one per rank. Returns the existing
// DriverCapabilities JSON (arch_name, total_pages, rs_cache_*, ...).
pub fn load_model(&mut self, desc: &ModelLoadDesc) -> Result<DriverCapabilities>;

pub struct ModelLoadDesc {
    pub load_plan_bytes: Vec<u8>, // serialized LoadPlan (mandatory, end state)
    pub snapshot_dir: PathBuf,    // payload source, driver-local
    pub compiler_version: u64,    // expected source hash; executor rejects skew
}
```

Delivery is bytes through the call, not a temp file beside the TOML; the
The temporary serialized-plan file and its path key die with the migration. A
possible later evolution (dynamic model load/unload with a `PieCompletion`)
is explicitly out of scope.

### 4.4 Tensor parallel

`cuda_group_create` returns per-rank device facts; the runtime compiles one
plan per rank (`StorageTarget` already carries `tp_rank`/`tp_size`) and
issues one `load_model` per rank. NCCL init stays in create, where it is
today.

### 4.5 Dummy

`create` returns synthetic facts (`unified_memory=true`, host alignment);
`load_model` executes the plan against a host arena and returns the caps
it currently fabricates from options. This is what makes the conformance
suite (S9) run in CI without a GPU.

---

## 5. Compiler work

1. **`BackendKind::Metal`** (`types.rs:65` — currently `Cuda | Unknown`).
   Classified as an arena backend so the existing coalescing of per-buffer
   `ExtentWrite`s into arena-relative `SlabPlacement`s applies as-is. This is
   the load-bearing luck of the whole plan: Metal's single-heap fixed-offset
   layout *is* the arena model.
2. **MLX quantization encoding.** New `QuantScheme` variant (working name
   `MlxAffineU4`): u32-packed 4-bit, group size 64, `scales` + `biases`
   sibling tensors. `QuantSpec` is already parametric
   (`bits_per_element`, `group_size`, `scale_dtype`, `zero_point_dtype`), so
   this is a variant plus lowering rules, not a schema rework. The triplet
   stays packed end-to-end — Metal kernels consume packed+scales+biases
   directly, so most placements lower to raw strided copies.
3. **Schema port from `heap_bind_names.cpp`.** The Qwen3.5/GDN checkpoint
   facts (`.language_model.` infix, tied `lm_head` triplet doubling as embed,
   dense-bf16 `in_proj_a/in_proj_b` exceptions, absent `conv1d.bias` → zeroed
   slot, `model.visual.*`/`mtp.*` skips) move into the semantic
   graph/schemas. This is the majority of the real porting effort and the
   majority of the payoff: model knowledge stops living in a C++ registry.
4. **`StorageTarget` from facts.** Constructed from the create-time facts
   payload; the mirrored constants in `embedded_driver.rs` and the hardcoded
   256 in `heap_layout.hpp` are deleted (S4).
5. **`RuntimeAbi::default_for_target`** gains the `"pie-metal"` contract set
   (CUDA's is `"pie-cuda"`, v1).

---

## 6. CUDA migration

### 6.1 Kill the fallbacks

With facts available at compile time, all three `return None` branches in
the CUDA compile fallback branches are deleted; runtime planning is
unconditional. The config-parse fallback needs a parity audit first: today a
Rust `parse_model_config` failure is silently masked by the C++ `HfConfig`
parse. Gate: every checkpoint in the test/bench fleet compiles through
`parse_model_config` (use `plan_dump` for goldens) *before* the mask is
removed, so no model regresses from "loads via C++ parse" to "hard error".

### 6.2 Delete the compile side, keep the executor

Stays (execution machinery): `load_plan_executor.hpp`,
`transcode_engine.hpp` (1,176), `weight_copy_engine.hpp` (458),
`staged_h2d.hpp` (172), `strided_copy.hpp` (95), `weight_store_codec.hpp`
(529), `safetensors_manifest.*` (94; file-id → path order must keep matching
`inproc.rs::discover_safetensors_files`), `shard_plan.hpp`,
`buffer_resolver.hpp`, `checkpoint_source.hpp`, `backend_target.hpp`,
`loader_config.hpp`, `phase_timer.hpp`.

Dies (compile input machinery):

| File | Lines | Note |
|------|-------|------|
| `loader/rust_loader_input.hpp` | 438 | FFI input construction |
| `loader/load_plan_bridge.hpp` | shrinks to deserialize-only wrapper |
| `loader/safetensors.{cpp,hpp}` | 785 | header parse fed the compile; audit executor for residual uses before deleting |
| `loader/dtype_map.hpp`, `tensor_spec.hpp` | 136 | audit: compile-input-only? |
| crate `ffi.rs`, `ffi_arena.rs`, `ffi_types.rs` | 1,583 | whole FFI surface |
| crate `build.rs` cbindgen half, `cbindgen.toml`, `include/` | ~200 | generated headers |
| crate tests `cxx_compat.rs`, `ffi_layout.rs` | — | FFI layout contracts |

Roughly 3.5–4k lines net-deleted, in the deletion-first spirit of
`cpp-refact.md`.

---

## 7. Metal executor

### 7.1 Executor semantics on UMA

Much simpler than CUDA's: no pinned staging, no H2D, no streams, no
load-time GPU compute in v1. Interpret the plan's placements as CPU writes
into the shared-storage heap: raw or strided `memcpy` from the mmap
(`SafetensorsView` is reused as the payload source) into
`weights_region_base + arena_offset`; any dtype transforms run on the CPU
during load. If a transform ever becomes load-time-critical, a Metal compute
pass can be added later without changing the LoadPlan format.

### 7.2 Arena mapping and binding

The plan's arena is the weights region of the single heap. Because the
compiler emits offsets, binding can move from thousands of placed
per-tensor `MTLBuffer` objects to **one region + offsets** —
`MTL4ArgumentTable` binds `gpuAddress + offset` anyway. Fewer API objects,
denser packing, and the residency story stays O(1) (one heap in one
`MTLResidencySet`, requested once after staging completes; I2 preserved).

### 7.3 Metal-specific considerations

- **TLB pressure is a packing problem.** Every decode step streams the whole
  weight arena once, so the GPU TLB working set is proportional to the
  arena's page count. The compiler's dense arena packing (already done for
  CUDA) is the TLB optimization; padding and fragmentation inflate the
  working set directly. One contiguous VA range beats scattered per-tensor
  allocations, and both beat scattered file-backed mappings — which is the
  second half of why S8 rejects zero-copy.
- **Load-time 2× memory.** During load, the page cache (mmap) and the heap
  hold the model simultaneously; on a 16–32 GB machine a large model can
  push into swap mid-load. Execute placements in tiles
  (`storage_max_tile_bytes` regains meaning on UMA as the memcpy chunk
  bound) and `madvise(MADV_DONTNEED)` / unmap consumed mmap ranges behind
  the copy so peak overhead is one tile, not one model.
- **Region alignment.** Heap region boundaries align to the 16 KB host/GPU
  page size (facts payload `page_size`), not the current 256. Placement
  alignment inside the arena comes from `heapBufferSizeAndAlign` via facts,
  not a literal.
- **Residency timing.** Stage everything, then make resident once. No
  incremental residency churn during load; first GPU touch after
  `requestResidency` faults nothing because the CPU writes already
  materialized the pages.

### 7.4 Parity gates

- The compiler-emitted weights layout is diffed offline against the current
  `heap_layout` weights half for the Qwen3.5 checkpoint (the
  `heap_bind_probe` / `rawmetal` assets exist for exactly this kind of
  check) before the executor lands.
- The `bind_decode_dag` ordinal contract must survive: the first executor
  test binds compiler-placed weights through the existing ordinals and runs
  the existing decode parity check (`tests/parity/parity_check.py`)
  token-for-token.
- Steady-state decode latency is unchanged (same resident heap, same
  kernels; only who computed the offsets changed).

---

## 8. Phases

Each phase is independently shippable; gates are blocking.

**Phase 0 — Boundary split.** `create`/`load_model` verbs, capability split
(§4), LoadPlan delivery as bytes. All backends keep their current load
behavior behind S6's plan-absent branch (Metal and dummy take no plan
yet; CUDA accepts bytes where it read a file). ABI version bump.
*Gate:* every existing model boots through the two-call sequence on
cuda/metal/dummy; TP group boots per-rank.

**Phase 1 — Mock-first executor.** Dummy executes LoadPlans against a host
arena; the conformance suite (place, strided copy, transform, tile bounds,
version rejection) runs in CI.
*Gate:* suite green with no GPU.

**Phase 2 — CUDA unconditional runtime compile.** `StorageTarget` from
facts; delete the three fallbacks after the config-parity audit (§6.1).
*Gate:* fp8 and mxfp4-native models load via runtime compile on real
hardware; `cuda_plain_gen` and the loader bench are unchanged.

**Phase 3 — Deletion + crate move.** Delete the C++ compile bridge and the
crate FFI surface (§6.2); move the planner crate to `runtime/load-planner`.
*Gate:* build + ctest; `grep -rn "pie_loader_compile" driver/` is empty;
no `driver/`-path dependency remains in `worker/Cargo.toml`.

**Phase 4 — Compiler learns Metal.** `BackendKind::Metal`, `MlxAffineU4`,
schema port from `heap_bind_names.cpp`, `"pie-metal"` ABI contracts (§5).
*Gate:* `plan_dump` golden for the Qwen3.5 checkpoint; offline layout
diff vs the current heap plan (§7.4).

**Phase 5 — Metal executor.** Execute LoadPlans into the heap's weights
region (§7); delete `heap_bind_names.cpp` and the weights half of
`heap_layout.hpp`; shrink `heap_bind.cpp` to region alloc + ordinal binding.
*Gate:* decode parity token-for-token; latency unchanged; S6's
plan-absent branch deleted for Metal.

**Phase 6 — Sweep.** Make `load_plan_bytes` mandatory everywhere; supersede
the `cpp-refact-metal.md` carve-out and stale comments; update boundary.md's
source map and §7 status line for Metal.
*Gate:* the legacy plan-path key is absent; docs point here.

---

## 9. Deletion ledger

| Area | What | Lines (approx) |
|------|------|-------|
| CUDA loader | `rust_loader_input.hpp`, bridge shrink, `safetensors.{cpp,hpp}`, input-only headers | ~2,000 |
| load-planner crate | `ffi.rs`, `ffi_arena.rs`, `ffi_types.rs`, cbindgen/build/include, FFI tests | ~1,800 |
| Metal loader | `heap_bind_names.cpp` (173), `heap_layout.hpp` weights half, `heap_bind.cpp` staging shrink | ~350 |
| Worker | three fallback branches, mirrored constants, plan temp-file plumbing, dead `mtp_assistant_snapshot_dir` key | ~130 |

New code: device-facts payload + `load_model` verb (small), dummy executor
(small, host memcpy interpreter), Metal executor (§7.1, small by design),
compiler Metal knowledge (§5 — the one genuinely new component).

---

## 10. Risks and open questions

- **Config-parse parity (highest).** The silent C++-parse mask may be hiding
  checkpoints Rust can't parse today. The Phase 2 audit is load-bearing; do
  not delete the mask on green CI alone, run the model fleet.
- **Metal binding churn.** Moving from per-tensor buffers to region+offset
  binding touches the argtable path shared with scratch/KV binding ordinals.
  The parity gate (§7.4) exists for this; land it as its own commit.
- **MLX group-size variants.** g32/g128 checkpoints exist in the wild;
  `MlxAffineU4` must key off `QuantSpec.group_size`, not assume 64.
- **GGUF remains deferred.** The planner can inspect direct GGUF metadata, but
  worker boot rejects `.gguf` explicitly until model-config planning and native
  payload execution support it end to end.
- **`mtp_assistant_snapshot_dir` is dead plumbing, not an open question.**
  The worker writes the key into the boot TOML (`config.rs:651`,
  `embedded_driver.rs:397`) but no driver reads it; the CUDA config parses
  only `mtp_num_drafts`. MTP weights ride the main snapshot (`mtp.*`
  tensors; `registry.hpp` derives `has_mtp` from the bound weights), which
  matches how current checkpoints ship. The verb stays one snapshot → one
  plan → one `load_model`; the dead key is deleted in the Phase 6 sweep.
- **Alignment assumptions.** Do not assume 256/16 KB on future devices;
  everything flows from the facts payload by construction (S4). The risk is
  a hardcode sneaking back in — the Phase 6 sweep greps for the literals.

---

## 11. Source map

| Concern | File |
|---------|------|
| Runtime-side planning entry | `runtime/load-planner/src/inproc.rs` |
| Planner core | `runtime/load-planner/src/{planner,frontend,semantic,schema,optimizer,typecheck}.rs` |
| Worker boot compile + fallbacks | `worker/src/embedded_driver.rs:265-368` |
| Driver create + caps channel | `runtime/engine/src/driver/backend/{cuda,metal,dummy}.rs` |
| Capabilities payload | `interface/driver/src/capabilities.rs` |
| ABI types / generated header | `interface/driver/src/local.rs`, `interface/driver/include/pie_driver_abi.h` |
| CUDA compile bridge (dies) | `driver/cuda/src/loader/rust_loader_{bridge,input}.hpp` |
| CUDA executor (stays) | `driver/cuda/src/loader/load_plan_executor.hpp` + copy/transcode engines |
| Metal loader (replaced) | `driver/metal/src/loader/{heap_bind*,heap_layout,safetensors_view}.*` |
| Metal context / residency | `driver/metal/src/mtl4_context.{hpp,mm}` |

---

## Provenance

Authored 2026-07-11 from the create/load-split and Metal-adoption design
discussion. Supersedes the load-planner non-goals in `cpp-refact-metal.md`
(§ non-goals, "Rust load-planner adoption") and completes the migration
described in `runtime/load-planner/src/inproc.rs` and
`worker/src/embedded_driver.rs`.
