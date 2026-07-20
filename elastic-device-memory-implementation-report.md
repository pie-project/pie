# Elastic Device Memory Implementation Report

Date: 2026-07-20

Implementation commit: `faa0e9d4` (`Implement elastic device memory`)

Status: **partial implementation; not complete against the E0-E5 plan**

## 1. Executive summary

The landed implementation establishes elastic backing primitives in both
drivers:

- CUDA allocations can use stable VMM addresses backed by a shared physical
  accounting pool.
- Metal KV, state, and scratch allocations use placement-sparse private buffers
  backed by shared placement-heap chunks.
- The runtime periodically derives a safe KV tail high-water from the
  epoch-aware store and requests KV/workspace trim.

The CUDA implementation was built and extensively tested on the local RTX 4090.
It is performance-neutral and returns committed memory after idle. The Metal
implementation was intentionally not built or tested.

This is **not yet the complete architecture described in the plan**. In
particular, scheduler admission does not lease shared physical pages, the CUDA
memory planner still freezes logical pool capacities, and the Metal idle-release
path has an unresolved final-release issue.

## 2. Implemented

### 2.1 Shared contract

Added `driver/common/include/pie_driver/elastic.hpp` with:

- `PhysicalPool::try_reserve`
- `PhysicalPool::unreserve`
- `Arena::ensure_committed`
- `Arena::trim_committed`
- A common 2 MiB logical accounting page

Driver capabilities now report:

- `elastic_page_bytes`
- `elastic_budget_pages`

The existing `resize_pool` ABI names three pools:

- `PIE_ELASTIC_POOL_KV`
- `PIE_ELASTIC_POOL_STATE`
- `PIE_ELASTIC_POOL_WORKSPACE`

### 2.2 CUDA

Implemented a CUDA VMM backend using:

- `cuMemGetAllocationGranularity`
- `cuMemAddressReserve`
- `cuMemCreate`
- `cuMemMap`
- `cuMemSetAccess`
- `cuMemUnmap`
- `cuMemRelease`

`DeviceTensor` and `DeviceBuffer` support scoped arena allocation. The following
surfaces were moved behind VMM addresses:

- Main `KvCache`
- Main forward workspace and its logits buffers
- Attention workspaces
- Qwen 3.5 linear-attention and MoE workspaces
- Nemotron-H workspace
- DeepSeek V4, Kimi, GLM5, and Gemma 4 auxiliary workspaces
- Recurrent-state cache, verification stash, and buffered-state pool

CUDA graph capture sees immutable virtual addresses. The hidden graph-padding KV
page is committed before replay so partially filled graph buckets cannot access
an unmapped tail.

The physical accounting budget is derived from free VRAM after weights, with a
reserved floor. A 32 MiB default VMM handle size is used, reduced to allocation
granularity for smaller arenas.

### 2.3 Metal

Implemented placement-sparse backing in `RawMetalContext`:

- Private sparse buffers created with
  `newBufferWithLength:options:placementSparsePageSize:`
- 16 KiB sparse tiles
- 256 MiB Shared placement-heap chunks
- Per-chunk Shared alias buffers for CPU zero/copy operations
- A dedicated MTL4 mapping queue
- Shared-event ordering between mapping and compute queues
- Stable sparse GPU addresses across grow and trim

Converted:

- Legacy M=1 KV storage
- Paged KV storage
- GDN convolution/recurrent state
- Decode and paged scratch pools

Paged-KV resize no longer allocates replacement buffers, copies live pages, or
rebinds argument tables.

### 2.4 Runtime trim policy

Every 10 seconds the runtime:

1. Reads the KV pool's epoch-aware physical high-water.
2. Attests the free tail through `unmap_ranges`.
3. Issues a KV trim to that high-water.
4. Trims transient workspace backing to zero.

Logical KV capacity and device virtual addresses remain unchanged. A later fire
recommits backing before dispatch.

## 3. CUDA validation

### 3.1 Build and unit coverage

- Full `pie-worker --features driver-cuda` build passed.
- New CUDA VMM grow/trim/remap smoke passed.
- KV quantized-cache test passed.
- Swap-pool test passed.
- Runtime engine: 330/330 passed.
- Driver ABI: 28/28 passed.
- Worker library: 60 passed, 1 ignored.

### 3.2 Kernel and driver parity

Passed:

- Masked-attention parity
- Mixed-length decode batch parity
- Explicit fused `WSlot`/`WOff` parity
- Multi-step co-batch K/V and attention parity
- CUDA entry/resource validation
- Real CUDA boot
- Real plain generation
- Real 256-request concurrent generation

Two hardware fixtures fail before entering the driver:

- PTIR prefill
- Runahead

Both report the existing host-geometry error:

```text
EmbedTokens is not host-derivable: channel 0 has no host-known value
```

These failures are not attributed to elastic memory.

### 3.3 Throughput

Canonical c0/256 shape:

- Qwen3-0.6B
- 256 requests, unlimited concurrency
- 128 output tokens
- warmup 2
- runahead depth 2
- n=3 interleaved control/candidate

Results:

| Arm | Median output tok/s |
|---|---:|
| Control | 32,056.20 |
| Elastic candidate | 32,091.50 |
| Delta | **+0.11%** |

All candidate runs completed 256/256 with zero demotions.

Serial oracle:

- 256/256 requests completed
- 256/256 output token hashes matched control
- The run crossed repeated 10-second trim intervals

### 3.4 CUDA footprint lifecycle

Persistent-server c0/256:

| Point | VRAM |
|---|---:|
| Boot idle | 3,944 MiB |
| Immediately after workload | 23,050 MiB |
| After idle trim | 3,982 MiB |
| Returned | **19,068 MiB** |

## 4. Not implemented or incomplete

### 4.1 Admission lease invariant

The most important missing item is the E4 lease ABI.

The scheduler does **not** convert wave KV/RS/workspace demand into elastic
pages and reserve them all-or-nothing at admission. `try_reserve` currently
lives inside driver-side commit. Therefore:

- Late commit is still theoretically failable.
- The plan's "admission is the only failable point" invariant is not proven.
- Wait-all cannot yet reason directly about shared physical-page demand.

### 4.2 Logical pool boundaries remain

The CUDA memory planner still computes and freezes:

- `kv_pages`
- `state_slots`
- workspace shape limits
- `arena_bytes`

Physical commits share one accounting pool, but logical capacity cannot move
freely between KV, state, and workspace. The planner lattice and per-pool
capacity fields have not been deleted.

### 4.3 CUDA KV is not proportional while graphs are active

The CUDA graph-padding page is located at the KV tail. Tail-only VMM commit must
therefore commit through that page before graph replay.

Consequences:

- Idle decommit works.
- Active inference recommits the full planned KV tail.
- CUDA does not yet achieve demand-proportional KV physical usage.

Fixing this requires either:

- A separately mapped graph-padding page/range, or
- Non-tail sparse range mapping inside the CUDA arena.

### 4.4 CUDA conversion gaps

These planned allocations remain outside elastic arenas:

- Runtime-quant GEMM scratch (`GrowScratch`, raw `cudaMalloc`)
- MLA cache
- DSA cache

`PersistentInputs` also remains directly allocated; it is mostly metadata and
was not part of the primary >90% conversion target, but the planner currently
counts it inside `arena_bytes`.

### 4.5 Metal final release bug

Metal unmap operations enqueue heap/alias objects in
`pending_elastic_releases`. Those objects are collected only when another
mapping operation begins.

If the system becomes completely idle immediately after trim:

- The unmap can complete.
- No later mapping call collects the pending objects.
- The placement heap may remain retained/resident.
- Immediate return to the OS is not guaranteed.

This must be fixed before claiming the macOS idle-footprint goal. Collection
needs an event completion callback, timer, or other post-unmap retirement path
that runs without requiring another mapping operation.

### 4.6 Metal memory pressure

No `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` hook exists. Critical/warning pressure
does not currently:

- Bypass trim hysteresis
- Force immediate safe-tail trim
- Reduce admission

### 4.7 E0 probes and Metal validation

The planned probe sources were not landed:

- `sparse_probe.mm`
- `sparse_probe2.mm`
- `sparse_probe3.mm`
- CUDA VMM latency/granularity microbench

Metal was not built or tested, per operator request. The Objective-C++ code uses
the measured Metal 4 API shape, but compilation and behavior remain unverified.

### 4.8 Remaining cleanup and knobs

Not completed:

- E5 deletion of obsolete planner fields and paths
- Configurable CUDA handle size and 2-64 MiB A/B
- First-touch background warmup after deep trim
- State high-water trim policy
- CUDA handle-cache hysteresis policy
- On-device/jetsam integration

## 5. Claims that are currently valid

It is valid to claim:

- CUDA and Metal elastic backing implementations exist.
- CUDA VMM virtual addresses remain stable across remap.
- CUDA is performance-neutral on c0/256.
- CUDA idle trim returns committed VRAM.
- Metal paged-KV no longer uses realloc-copy growth.

It is **not** yet valid to claim:

- The complete E0-E5 plan is implemented.
- Reserve-then-commit cannot fail after admission.
- Logical KV/RS/workspace boundaries are removed.
- Metal idle trim reliably returns memory to the OS.
- Metal behavior or performance has passed validation.

## 6. Recommended completion order

1. Fix Metal post-unmap heap retirement.
2. Build and run the Metal implementation and land the probe assets.
3. Add shared-budget lease/reservation to scheduler admission.
4. Add non-tail CUDA mapping for the graph-padding page.
5. Convert runtime-quant scratch and MLA/DSA storage.
6. Add Metal memory-pressure handling.
7. Delete the old planner capacity lattice and obsolete fields.
8. Run CUDA handle-size and first-touch latency A/B.
