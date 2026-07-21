# Elastic Device Memory Implementation Report

Date: 2026-07-21

Baseline commits: `faa0e9d4`, `ef314886`, `555b2d35`; C2/E4 changes are
currently uncommitted.

Status: **local CUDA TP1 admission implemented; overall E0-E5 plan remains
partial**

## 1. Executive summary

The implementation now establishes:

- CUDA allocations can use stable VMM addresses backed by a shared physical
  accounting pool.
- Local, single-rank CUDA launches use ABI-v12 prepare/lease admission. Prepare
  validates the finalized descriptor, atomically commits independently rounded
  KV/state/workspace/attention deltas, and returns either a one-shot lease,
  transient exhaustion, or explicit impossible demand.
- Wait-all treats the driver verdict as the commit point. Exhausted proposals
  remain queued and do not consume credits, depth, demotion state, wave
  statistics, tickets, or ordinary retry budget.
- Metal KV, state, and scratch allocations use placement-sparse private buffers
  backed by shared placement-heap chunks.
- The runtime periodically derives safe KV and RS high-waters and requests
  KV/state/workspace trim.

The CUDA implementation was built and extensively tested on the local RTX 4090.
It is performance-neutral and returns committed memory after idle. The Metal
implementation was intentionally not built or tested.

This is **not yet the complete architecture described in the plan**. Remote,
Metal, and CUDA TP admission are deliberately capability-gated rather than
claimed correct. The CUDA planner still freezes logical pool capacities, and the
Metal idle-release path retains its pre-existing final-release issue.

## 2. Implemented

### 2.1 Shared contract

Added `driver/common/include/pie_driver/elastic.hpp` with:

- `PhysicalPool::try_reserve`
- `PhysicalPool::unreserve`
- `Arena::ensure_committed`
- `Arena::trim_committed`
- A common 2 MiB logical accounting page

Local ABI v12 adds:

- `*_prepare_launch`
- `*_launch_prepared`
- `*_release_launch`
- `PieLaunchPrepareResult` with ready/exhausted/impossible outcomes and the
  observed budget generation

The remote wire remains v8 because leases are intentionally not serialized.

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

CUDA graph capture sees immutable virtual addresses. Graph padding aliases KV
page 0, and each launch carries the physical KV translation high-water that must
be committed before replay.

The physical accounting budget is recalibrated after fixed allocation/graph
capture and before every preparation, using one safety floor and never dropping
below charged pages. `charged = committed + held <= budget`; cached generic
handles were removed so unmapped memory cannot escape accounting. Budget and
release changes advance a generation. A 32 MiB default VMM handle size is used,
reduced to allocation granularity for smaller arenas.

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
4. Converts the exact RS slot high-water to elastic pages and trims state.
5. Trims transient workspace backing to zero.

Logical KV capacity and device virtual addresses remain unchanged. A later fire
recommits backing before dispatch.

## 3. CUDA validation

### 3.1 Build and unit coverage

- Full `pie-worker --features driver-cuda` build passed.
- CUDA VMM grow/trim/remap plus injected second-arena rollback passed.
- KV quantized-cache test passed.
- Swap-pool test passed.
- Runtime engine library: 334/334 passed (including deterministic
  exhausted/impossible admission and unchanged-proposal tests).
- Driver ABI: 28/28 passed.
- Dummy one-shot lease consume/release regression passed.
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

### 4.1 Admission scope and gates

The E4 invariant is implemented for local CUDA TP1:

- Preparation performs full launch/resource/PTIR validation.
- KV, state, workspace, and attention targets are rounded per CUDA arena.
- One pool hold covers the complete multi-arena transaction.
- A mapping or access failure unmaps every new mapping and restores pool
  counters.
- A successful prepare leaves no elastic allocation on the prepared launch
  path.
- The lease is consumed by one launch or one release; in-flight floors remain
  until stream retirement.

This claim does **not** extend to Remote, Metal, or TP. Remote post-scheduler
coalescing would invalidate a worker-side lease, Metal has no C2 implementation,
and TP needs all-rank atomic prepare/abort. All three use the existing unprepared
path and report prepare unsupported.

### 4.2 Logical pool boundaries remain

The CUDA memory planner still computes and freezes:

- `kv_pages`
- `state_slots`
- workspace shape limits
- `arena_bytes`

Physical commits share one accounting pool, but logical capacity cannot move
freely between KV, state, and workspace. The planner lattice and per-pool
capacity fields have not been deleted.

This boundary was intentionally not changed speculatively. Correct deletion
needs the active planner lifecycle work to define bounded logical KV/RS ceilings
while preserving c0 capacity, the hidden non-page-zero graph padding allocation,
and recurrent-state initialization. The C2 patch therefore keeps virtual/index
shape ceilings and explicit `total_pages` behavior unchanged.

### 4.3 CUDA graph padding is demand-proportional

The gated page-0 alias landed on 2026-07-20.

Stage 1 made invalid-row KV write suppression a checked model capability:

- Gemma 4's packed fused writer and fallback generic writer now consume
  `row_valid`.
- Nemotron-H's generic writer now consumes `row_valid`.
- Every graph-safe model must declare the checked KV-write capability or model
  attachment fails.
- Quantized KV writers remain excluded because graph safety is gated on native
  BF16.
- CUDA-graph canaries cover Llama-like, Qwen3-VL, Qwen3.5 dense/MoE, Gemma 4
  fused/fallback, and Nemotron-H. Each replays an off-lattice invalid row
  against a dedicated canary page and verifies that K/V remain untouched.

Stage 2 removed the dedicated tail page from the page-0-safe path:

- Native graph-safe families use `graph_pad_page = 0`; models/formats whose
  non-graph dummy-lane writers are not row-validity-safe retain a hidden
  sacrificial tail page and keep its backing pinned.
- Physical KV allocation no longer adds `kv_pages + 1` on the page-0-safe
  path.
- Upfront graph-lattice capture maps page 0 explicitly and uses page 0 for every
  synthetic request instead of fully precommitting KV.
- Launch commit demand no longer has a graph-pad tail floor.

The first c0/256 gate exposed an important vantage error: the ABI wire CSR for
a device-resolved fire contains logical pages, not its translated physical
page IDs. Using that CSR max committed three pages and faulted the first wide
graph replay. The final implementation computes the exact physical high-water
from each fire's existing `kv_translation`, takes the batch maximum, and
carries it in `PieLaunchDesc::required_kv_pages`. The driver commits that prefix
before descriptor resolution or model execution. The field entered in ABI v11
(the current local ABI is v12) and remote wire v8; remote launch merging
recomputes the batch
high-water, and TP broadcasts it so every follower commits the same prefix.

Final gates:

- Graph canaries: all graph-safe model families pass.
- Serial oracle: 256/256 output hashes match the pre-alias control.
- Upfront lattice: 51 decode graphs capture with minimal KV commit.
- c0/256 interleaved n=3: control 32,073.86 tok/s, page-0 alias
  32,116.08 tok/s, **+0.13%**; all runs completed 256/256 with zero failures
  and demotions.
- Active c0/256 proportionality: peak required 2,816 / 10,407 planned pages;
  peak committed KV 5.25 / 17.79 GiB (29.5%).

The weights-floor idle assertion is no longer a valid c0/256 oracle after the
operator-ratified always-on KV index change: cached index roots intentionally
retain physical pages until explicit removal or pressure eviction. A
persistent-server cross-check measured 23,002 MiB after 27 seconds on the
pre-alias control and 8,698 MiB on the alias. Reaching the weights floor while
preserving those roots would require eviction or swap, so this change does not
silently introduce either policy.

`graph_pad_slot` was not aliased. RS graph padding is currently disabled by the
host-reset eligibility predicate, while the state allocator is fully committed
at load, so the sacrificial slot does not add a separate physical-commit floor
today. Future proportional state commit must still retain a sacrificial slot
unless every recurrent-state write is proven to skip invalid rows.

### 4.4 CUDA residual conversion

The C1 residual conversion landed after the page-0 alias:

- Runtime-quant cuBLASLt workspace and growable FP8/INT8/MXFP4 scratch are
  Context-owned and allocated from the workspace arena. Load, launch, encode,
  and TP follower threads bind the owning context explicitly.
- Kimi/GLM5 MLA cache storage is allocated from the KV arena and grows with the
  translated physical page high-water. Invalid MLA rows are `row_valid`-gated,
  so these families also use the page-0 dummy alias.
- DeepSeek V4, Kimi, and GLM5 auxiliary workspace allocations are included
  exactly in planner candidates.
- `PersistentInputs` remains a direct allocation intentionally, but is now
  reported as fixed `persistent_input_bytes` instead of elastic `arena_bytes`.

`DsaCache` currently has no backing allocation; it is an empty GLM5 interface
stub. There is therefore no DSA device pool to convert until the indexer cache
itself is implemented.

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
- Full logical planner-boundary deletion
- On-device/jetsam integration

## 5. Claims that are currently valid

It is valid to claim:

- CUDA and Metal elastic backing implementations exist.
- CUDA VMM virtual addresses remain stable across remap.
- CUDA is performance-neutral on c0/256.
- CUDA idle trim returns committed VRAM.
- Local CUDA TP1 prepare is the only elastic-allocation failure point before a
  prepared launch, and its one-shot lease prevents unsafe trim.
- Exhausted wait-all proposals are deferred without consuming wave state.
- Metal paged-KV no longer uses realloc-copy growth.

It is **not** yet valid to claim:

- The complete E0-E5 plan is implemented.
- Prepare/lease correctness for Remote, Metal, or CUDA TP.
- Logical KV/RS/workspace boundaries are removed.
- Metal idle trim reliably returns memory to the OS.
- Metal behavior or performance has passed validation.

## 6. Recommended completion order

1. Fix Metal post-unmap heap retirement.
2. Build and run the Metal implementation and land the probe assets.
3. Define and implement bounded logical KV/RS ceilings, then delete the old
   planner partition coupling.
4. Keep the landed page-0 graph-padding canaries and physical high-water ABI
   contract in CUDA CI.
5. Add Metal memory-pressure handling.
6. Add all-rank prepare/abort before enabling CUDA TP admission.
7. Design worker-side preparation that survives remote coalescing before
   enabling Remote.
8. Run CUDA handle-size and first-touch latency A/B.
