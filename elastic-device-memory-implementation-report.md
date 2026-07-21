# Elastic Device Memory Implementation Report

Date: 2026-07-21

Implementation commits through `b4db0531`; this report includes the final
planner, pressure, trim-order, and E5 validation delta.

Status: **local CUDA TP1 and Metal elastic admission implemented and
validated; Remote and CUDA TP admission remain capability-gated**

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
- Metal uses the same ABI-v12 prepare/lease contract, with atomic multi-buffer
  growth, rollback, trim exclusion while leases exist, and owner-thread
  retirement.
- The runtime periodically derives safe KV and RS high-waters and requests
  KV/state/workspace trim.

The CUDA implementation was built and extensively tested on the local RTX 4090.
It is performance-neutral and returns committed memory after idle. The Metal
implementation and embedded worker were built on the Mac Studio; the complete
Metal CTest suite, sparse lifecycle, pressure gates, and raw probes passed on
the live AGX device.

Remote and CUDA TP admission remain deliberately capability-gated rather than
claimed correct. Remote needs a lease contract that survives post-scheduler
coalescing, and TP needs all-rank atomic prepare/abort.

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
below charged pages. `charged = committed + held <= budget`; every retained
arena-local hysteresis handle remains in `committed`, so unmapped memory cannot
escape accounting. Partial trims may retain one handle only when at least two
handles remain mapped; deep/idle trim releases the cache. Budget and release
changes advance a generation. `PIE_CUDA_VMM_HANDLE_MB` selects 2--64 MiB, with
32 MiB as the validated default.

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

Metal launch preparation computes exact KV/state/token-row demand, atomically
commits every required sparse buffer, and returns a one-shot ABI-v12 lease.
Failure restores all buffers to their prior commitments. Trim is rejected while
any prepared or in-flight lease exists.

### 2.4 Runtime trim policy

Every 10 seconds the runtime:

1. Reads the KV pool's epoch-aware physical high-water.
2. Attests the free tail through `unmap_ranges`.
3. Issues a KV trim to that high-water.
4. Converts the exact RS slot high-water to elastic pages and trims state.
5. Trims transient workspace backing to zero.

Logical KV capacity and device virtual addresses remain unchanged. A later fire
recommits backing before dispatch. Resize commands share the launch/prepare FIFO
rather than the ordinary control queue; this prevents a later prepare from
overtaking a pending trim and republishing a stale admission watermark.

On macOS, a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source updates an atomic
admission level. Warning pressure lowers the growth budget to 50%; critical
pressure lowers it to `PIE_METAL_PRESSURE_FLOOR_BYTES` (zero by default).
Existing commitments are grandfathered, but no new physical growth can exceed
the pressured budget.

## 3. CUDA validation

### 3.1 Build and unit coverage

- Full `pie-worker --features driver-cuda` build passed.
- Complete CUDA CTest suite: 45/45 passed.
- CUDA VMM grow/trim/remap, arena-local cache reuse, and injected rollback
  after cached-handle reuse passed.
- KV quantized-cache test passed.
- Swap-pool test passed.
- Runtime engine library: 359/359 passed (including deterministic
  exhausted/impossible admission and unchanged-proposal tests).
- Driver ABI: 28 unit checks plus 2 C/C++ layout checks passed.
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

The complete CTest gate also exposed two inherited PTIR lifecycle defects:
asynchronous instance retirement made immediate channel close fail permanently,
and standalone tier-0 tests did not settle grouped channel initialization.
Deferred channel retirement now uses one per-channel pending-close bit without
blocking the driver lane, and standalone runners use the production settlement
boundary. The c0 close warnings disappeared and the affected race/bind/runner
tests pass.

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

The former FlashInfer 0.6.15 + planner-boundary + E5 headline of 47,117.60
tok/s is invalid. Its 256/256 serial oracle compared two Pie builds that shared
the same decode-channel corruption and therefore did not establish model
accuracy. The corrected result and root cause are recorded in section 4.1.

### 3.4 CUDA footprint lifecycle

Persistent-server c0/256:

| Point | VRAM |
|---|---:|
| Boot idle | 3,944 MiB |
| Immediately after workload | 23,050 MiB |
| After idle trim | 3,982 MiB |
| Returned | **19,068 MiB** |

That table is the original no-live-root lifecycle gate. With the current
text-completion lifecycle, the final persistent cross-check was 3,960 MiB at
boot and 8,728 MiB both immediately after c0/256 and after 22 seconds. The
remaining pages are still live according to the runtime high-water; the trim
path correctly does not invent eviction. The independently measured deep-trim
first-wave gate below uses no retained wide c0 working set.

### 3.5 E5 handle and first-touch gates

Same-binary c0/256, interleaved n=3:

| VMM handle | Median output tok/s |
|---|---:|
| 2 MiB | 44,766.00 |
| 32 MiB | 46,813.44 |
| 64 MiB | 46,853.66 |

All nine runs completed 256/256 with 133 batches. 64 MiB was only +0.09%
versus 32 MiB, while 2 MiB was -4.37%; 32 MiB remains the default.

On one persistent server, a one-request first wave measured 41.36 ms TTFT
before deep trim and 41.19 ms after the 10-second trim boundary. No
background warmup was added because no first-touch regression was observed.

## 4. Remaining scope and resolved audit items

### 4.1 Admission scope and gates

The E4 invariant is implemented for local CUDA TP1 and Metal:

- Preparation performs full launch/resource/PTIR validation.
- KV, state, workspace, and attention targets are rounded per CUDA arena.
- One pool hold covers the complete multi-arena transaction.
- A mapping or access failure unmaps every new mapping and restores pool
  counters.
- A successful prepare leaves no elastic allocation on the prepared launch
  path.
- The lease is consumed by one launch or one release; in-flight floors remain
  until stream retirement.
- The scheduler caches the highest driver-prepared KV/state demand. Waves below
  that watermark use the already-committed mappings without a second
  driver-lane round trip; pool resize invalidates the watermark and is ordered
  in the same FIFO as prepare/launch.

The inferlet/attention lifecycle merge channelized `EmbedIndptr`, `Readout`, and
flat single-lane `Pages`. CUDA now verifies those channel shapes and preserves
device-side decode composition for both all-decode and mixed host/envelope
waves instead of attempting premature host descriptor readback.

The original post-merge performance gates compared two equally corrupted
decode paths and are not accuracy evidence. Long-output cross-validation found
that every generation matched vLLM only through `<think>\n`, then became
prompt-independent gibberish.

The corruption was in grouped asynchronous channel initialization:

- `seed_cell_async` copied a seed into device cell 0 and published host tail 1,
  but unlike `seed_cell` it did not advance `pulled_tail` for a host-writer
  channel.
- The first fixed-decode `pull_writer_inputs` therefore treated sequence 0 as a
  new host input and copied the zero-filled host mirror over the committed
  seed.
- For `Pages`, the verified bind-time value `[0,1,...]` became `[0,0,...]`.
  Translation then mapped every logical page to one physical page. Prefill K/V
  remained correct, but the first decode collapsed the page table and every
  later token was wrong.

The fix makes asynchronous writer seeding establish the same pulled-tail
baseline as synchronous seeding. A CUDA regression now seeds a writer
asynchronously, performs the first ring pull, and asserts that no copy occurs
and the committed canary remains unchanged.

Corrected gates:

- The known Raft prompt matches vLLM exactly for the first 64/64 greedy tokens.
- Four independent 1,024-token prompts (fiction, Raft, Python, incident report)
  retain their requested entities and facts and produce coherent,
  prompt-specific text. Pie/vLLM common prefixes are 38--67 tokens; forced
  long-form continuation diverges numerically after that point.
- Concurrent four-prompt outputs complete 1,024/1,024 each and match their
  corresponding serial Pie token sequences exactly.
- Correct c0/256 n=3 is 34,256.63 tok/s median
  (34,256.63 / 34,489.34 / 34,184.02), 132 batches and 256/256 completions.
  The former 47,117.60 tok/s was a 27.3% false uplift from repeatedly reading
  one aliased physical KV page, not usable model throughput.
- CUDA CTest passes 45/45, including the new seed-pull regression; engine unit
  tests pass 359/359.

This claim does **not** extend to Remote or CUDA TP. Remote post-scheduler
coalescing would invalidate a worker-side lease, and TP needs all-rank atomic
prepare/abort. Both report prepare unsupported.

### 4.2 Planner physical pool boundaries removed

The CUDA planner lattice now selects only:

- KV page size
- Forward token/request/page-reference shape limits
- Object-to-byte coefficients such as `kv_page_bytes`

`CudaMemoryPlan::kv_pages`, `state_slots`, `arena_bytes`, `kv_bytes`, and
`state_bytes` are deleted. Context derives the KV logical/VA ceiling from the
shared elastic budget divided by `kv_page_bytes`; an explicit `total_pages`
configuration remains a hard operator clamp. State logical capacity equals the
selected request shape when recurrent state exists. Workspace virtual capacity
comes from the selected forward shape. ABI-v12 admission is the only physical
concurrency partition.

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

### 4.5 Metal final release resolved

Metal unmap operations enqueue heap/alias objects in
`pending_elastic_releases`. Trim now drains the mapping event on the owner
thread, removes completed heaps from the residency set, commits the residency
change, and drops both heap and alias retention before returning. Final context
destruction also drains the last event. Live grow/trim/regrow/final-release
coverage observes zero pending releases.

### 4.6 Metal memory pressure integrated

A dispatch memory-pressure source reduces admission as described in section
2.4. The driver does not invent a safe-tail high-water under pressure; physical
release remains runtime-owned and occurs at the next ordered trim boundary.
Forcing immediate trim would require a new driver-to-runtime callback or
duplicated liveness state, so the simpler correctness-preserving policy is
growth denial plus the existing 10-second trim.

### 4.7 E0 probes and Metal validation

The planned sources are landed under `driver/metal/tools/rawmetal/`:

- `sparse_probe.mm`
- `sparse_probe2.mm`
- `sparse_probe3.mm`

They cover stable sparse VA grow/trim/final release, atomic over-budget
rollback, and alias copy integrity across trim/regrow. All three compile and
run on the Mac Studio. The full Metal CTest suite passes 20/20; extended sparse
lifecycle and pressure coverage passes 23/23; `pie-worker --features
driver-metal` builds with Rust 1.97.1.

### 4.8 Completed cleanup and knobs

- Obsolete planner capacity/byte fields are deleted.
- CUDA handle size is configurable and the 2/32/64 MiB gate is recorded.
- Arena-local handle hysteresis is budget-accounted and deep-trim-safe.
- Deep-trim first-touch was measured; background warmup was unnecessary.

## 5. Claims that are currently valid

It is valid to claim:

- CUDA and Metal elastic backing and local ABI-v12 admission are implemented.
- CUDA VMM virtual addresses remain stable across remap.
- CUDA is performance-neutral on c0/256.
- CUDA idle trim returns committed VRAM.
- Local CUDA TP1 prepare is the only elastic-allocation failure point before a
  prepared launch, and its one-shot lease prevents unsafe trim.
- Exhausted wait-all proposals are deferred without consuming wave state.
- Metal paged-KV no longer uses realloc-copy growth.
- Metal sparse retirement, rollback, pressure admission, and live build/test
  gates pass on macOS.
- CUDA planner physical pool partitions and obsolete capacity fields are
  removed.

It is **not** yet valid to claim:

- Prepare/lease correctness for Remote or CUDA TP.
- Immediate memory-pressure trim before the next runtime trim boundary.
- End-to-end Metal model throughput parity; the validated Metal scope is build,
  driver tests, sparse lifecycle, and ABI admission.

## 6. Remaining distributed work

1. Keep the landed page-0 graph-padding canaries, physical high-water ABI, and
   elastic lifecycle assertions in CI.
2. Add all-rank prepare/abort before enabling CUDA TP admission.
3. Design worker-side preparation that survives remote coalescing before
   enabling Remote.
