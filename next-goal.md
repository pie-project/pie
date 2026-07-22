# Next Goal: Extreme Short-Request Scalability

Date: 2026-07-21  
Status: Operator-requested follow-up  
Baseline: `48b0f1d5` on `tasks/ptir-fusion/agents/charlie`

## 1. Objective

Make Pie competitive with vLLM when thousands of short requests arrive at
once, without regressing correctness, ordinary c0/256 throughput, long decode,
memory proportionality, or pipeline lifecycle safety.

The primary target is:

- Qwen3-0.6B, CUDA TP1, RTX 4090
- 2,048 logical requests submitted together
- At most 512 active requests
- Approximately 35--40 prompt tokens per request
- Exactly 32 greedy output tokens
- Prefix caching disabled
- Identical model snapshot and chat template

The ultimate performance target is parity with the measured vLLM rate of
approximately 30.8k output tok/s. The first acceptance gate is to close at least
half of the remaining gap without regressing any existing production shape.

## 2. Current Baseline

### 2.1 End-to-end performance

| Shape | Pie | vLLM | Result |
|---|---:|---:|---:|
| 2,048 x 32, active 512 | 19.08k tok/s | 30.81k tok/s | Pie -38.1% |
| Mixed 2,048, active 256 | 15.08k tok/s | 17.49k tok/s | Pie -13.8% |
| 512 x 512, active 256 | 18.51k tok/s | 18.31k tok/s | Pie +1.1% |
| 64 x 1,536 | 7.34k tok/s | 7.17k tok/s | Pie +2.4% |
| 16 x 1,900 | 3.72k tok/s | 3.57k tok/s | Pie +4.3% |

Pie remains strong when decode is long enough to amortize process setup. The
deficit is specific to fleets of short-lived inferlets.

### 2.2 GPU evidence

The model kernels are not slower:

| Profiled item | Pie | vLLM |
|---|---:|---:|
| Profiled wall | 3.523 s | 2.169 s |
| GPU kernel sum | 0.741 s | 0.863 s |
| GPU kernel span | 3.517 s | 1.917 s |
| Idle inside kernel span | 2.775 s | 1.054 s |
| GPU kernel instances | 28,475 | 6,053 |

Pie performs about 14% less GPU kernel work but takes about 62% more wall time.
Only about 21% of Pie's GPU activity span contains kernels. The problem is
feeding and retiring work, not executing the transformer.

### 2.3 CUDA API fan-out

Measured during the same Pie window:

| CUDA API | Pie calls | vLLM calls |
|---|---:|---:|
| `cudaMalloc` | 41,126 | 1 |
| `cudaHostAlloc` | 32,175 | 12 |
| `cudaMemcpyAsync` | 75,490 | 604 |
| `cudaFree` | 26,523 | 0 |
| `cudaLaunchKernel` | 27,657 | 3,318 |
| `cudaGraphLaunch` | 165 | 121 |

This is allocation-count bound, not capacity bound.

## 3. Measured Limitations

### 3.1 Tiny per-request allocations

A prefill-only request registers exactly 10 channels. Its total payload is only:

- 1,247 bytes of device-cell storage
- 1,247 bytes of pinned host-mirror storage
- 320 bytes of host control words

Despite that small payload, the first 512-request cohort performs:

- 16 `cudaMalloc` calls per request
- approximately 21 `cudaHostAlloc` calls per request

The allocations come from:

- one device cell per channel;
- one pinned mirror per host-visible channel;
- one pinned host-word block per channel;
- per-instance commit flags;
- per-stage channel readiness/commit lists;
- per-fire or per-instance snapshot storage.

Across 2,048 requests, only 1.36 MiB of new device-cell payload and 1.36 MiB
of new host-mirror payload are needed. They are split into:

- 8,316 device-cell growths;
- 8,316 pinned-mirror growths;
- 5,790 separate 32-byte host-word allocations.

Only 12 allocation size classes appear, ranging from 8 to 296 bytes.

The existing channel registry reserves metadata arrays at model load, but it
does not reserve the channel payload, pinned mirrors, host words, or
per-instance stage lists. Metadata pre-sizing therefore does not solve the
measured allocator pressure.

### 3.2 WASM instantiate/link cost

For one 2,048-request first-token fleet:

- client launch fan-out completes in 88 ms;
- process instantiate/admit spans 770 ms;
- final driver bind completes at 849 ms;
- aggregate Wasmtime instantiate work is 7.30 CPU-seconds;
- aggregate component/link work is 7.79 CPU-seconds.

Worker threads overlap this work, but the total remains large enough to delay
cohort readiness.

### 3.3 Serialized register/bind control lane

The driver bind RPC has:

- p50 latency: 70.9 ms
- p95 latency: 104.7 ms
- actual register+bind control occupancy: 426 ms total

The latency is mostly queueing. A single scheduler control lane serializes
2,048 per-instance channel registration and bind operations.

The four 512-process cohorts leave measured GPU gaps of approximately:

- 69 ms
- 101 ms
- 81 ms

### 3.4 Wave and readiness inflation

The useful work is:

- 2,048 requests
- 1 prefill fire plus 31 decode fires each
- 65,536 useful fire lanes
- roughly 134 expected waves

The profiled depth-2 run executes:

- 184 waves
- 76,308 processed fire lanes
- 19 prefill waves
- 165 decode waves
- average active width 454
- average missing pipelines 29

This is approximately 37% wave inflation and 16.4% fire-lane inflation.

The principal mechanism is successor readiness:

- the guest submits depth-2 run-ahead;
- a successor may reach composition before predecessor channel publication;
- the fixed-decode composer observes incomplete readiness;
- the launch is retried or falls through a narrow/dummy composition;
- the fleet fragments into additional waves.

Forcing `PIE_SCHED_MAX_IN_FLIGHT=1` proves the attribution:

- waves: 174 -> 136;
- average missing pipelines: 88.5 -> 0;
- all measured waves have at least 128 requests;
- throughput: 17.83k -> 18.83k, +5.6%.

Depth 1 is not the complete solution. It removes readiness inflation but leaves
the allocation and lifecycle costs.

### 3.5 Per-wave host cost

The measured 174-wave run spends 571 ms in CUDA submit host work.

Largest measured lines:

- settlement enqueue: 321 ms;
- epilogue assembly/execution: 164 ms;
- settle preparation: 156 ms;
- H2D preparation: 81 ms;
- dispatch begin/ticket handling: 64 ms.

Some lines overlap and must not be summed as independent wall time. They still
identify the dominant per-wave host payers.

### 3.6 Mixed-length admission is process-count based

Mixed 8/32/128/512-token fleets cannot safely use the same active-process cap
as uniform short requests:

- active 512 can accumulate enough long requests to exceed the 10,723-page
  driver ceiling;
- cap 300 can still hit allocator high-water and stall;
- cap 256 is the measured safe point.

The current cap is explicit and workload-specific. Pie does not yet have
automatic KV-weighted execution admission comparable to vLLM's token/KV-aware
scheduling.

### 3.7 Persistent-fleet retirement is not fully closed

The first two repeated 2,048-request fleets complete, but a third immediate
fleet can stall on retained page/channel high-water. Reuse must therefore
include provable stream retirement and resource release. An unbounded cache or
a pool that never returns pressure would only hide the leak.

## 4. North-Star Design

The preferred design has four layers.

### 4.1 One packed PTIR-instance device allocation

Replace many per-instance CUDA allocations with one packed device arena
containing:

- pass commit flag;
- static stage readiness lists;
- static stage commit lists;
- per-instance channel-slot remap tables;
- fixed-size snapshot/control words;
- other small immutable or fire-local device metadata.

The layout is computed once from the decoded program and instantiated by
offset. Compiled program metadata remains shared.

Target:

- one device allocation per PTIR instance initially;
- eventually suballocate instances from a bounded fleet slab;
- no pointer movement while a driver launch may reference the instance.

### 4.2 One packed pinned-host allocation

Pack into one pinned block per instance, or a fleet-level slab:

- host channel mirrors;
- head/tail/poison/wait words;
- bool wire staging where required;
- settlement/callback metadata.

Do not issue one `cudaHostAlloc` per channel. The measured payload is small and
the size classes are highly regular.

### 4.3 Bounded size-class reuse

Use a small bounded allocator for the measured 8--296 byte channel classes,
rounded to stable power-of-two buckets.

Requirements:

- per-device ownership;
- explicit maximum retained bytes;
- stream/completion retirement before reuse;
- no reuse while native work can still reference a cell;
- idle/pressure trim integration;
- counters for live, retired, reusable, and retained blocks;
- no silent fallback to unbounded allocations.

### 4.4 Readiness-aware successor deferral

Do not submit a successor to fixed decode merely to discover on GPU that its
predecessor publication is incomplete.

The scheduler should defer a successor when:

- its preceding fire has not published the required channel epoch;
- its pipeline already has the configured in-flight depth;
- a device channel readiness dependency is known incomplete.

The first implementation may select depth 1 automatically for very large
fleets. The final implementation should retain depth 2 where it is profitable
without generating retry waves.

## 5. Proposed Milestones

### M0: Stable benchmark and conservation ledger

Before changing allocation:

- run 2,048 x 32 n=5;
- run cold and repeated-fleet variants separately;
- preserve the vLLM active-512 reference;
- record client wall, kernel sum/span, allocation API counts, waves, useful
  lanes, missing pipelines, bind span, and resource high-water;
- reject any comparison with different prompt rendering, output count, or
  active width.

Deliverable: one machine-readable baseline artifact plus a concise report
table.

### M1: Pack per-instance device metadata

Move commit flags and stage lists into one device allocation per PTIR instance.

Gates:

- exact token parity on serial and output-boundary oracles;
- CUDA CTest and engine suites pass;
- no pointer instability under graph capture/replay;
- `cudaMalloc` count falls materially before touching channel mirrors;
- no c0/256 regression above 1%.

### M2: Pack pinned channel metadata

Move mirrors and host words into one pinned allocation per instance.

Gates:

- `cudaHostAlloc` count falls from approximately 21 per request to at most 2;
- host reader/writer rings preserve ABI word ordering;
- poison, close, and post-close drain tests pass;
- concurrent multi-client output remains correct;
- pinned bytes remain bounded and visible in status.

### M3: Add bounded fleet size classes

Recycle packed device and pinned blocks across completed inferlets.

Gates:

- three consecutive 2,048-request fleets complete;
- no third-run high-water stall;
- no ABA reuse before stream retirement;
- pool retained bytes return under pressure;
- no allocation-count rebound after the first fleet.

### M4: Remove readiness retry waves

Introduce readiness-aware successor deferral or an evidence-based adaptive
depth policy.

Gates:

- 2,048 x 32 waves <= 140;
- average missing pipelines approximately zero;
- processed fire lanes <= 68,000;
- no chain-kill growth from retryable readiness;
- ordinary c0/256 retains the depth-2 win.

### M5: KV-weighted execution admission

Replace workload-specific process caps with explicit per-process KV demand.

Possible demand inputs:

- currently committed pages;
- declared writable frontier;
- next-fire delta;
- bounded output/context reservation.

Gates:

- mixed 2,048 runs safely without a hand-selected process cap;
- no launch can request a physical prefix above the driver ceiling;
- long requests do not starve short requests;
- no hidden swap/preemption policy is introduced.

### M6: Consider inferlet reuse or multiplexing

Only pursue this if M1--M5 do not close the remaining gap.

This is the largest architectural change because it alters the assumption that
one logical request owns one WASM instance and one channel graph. It must not
be introduced merely to hide allocator inefficiency.

## 6. Acceptance Targets

### 6.1 Performance

Initial gate:

- Pie 2,048 x 32 >= 25k output tok/s.

Final gate:

- Pie within 5% of, then at parity with, the matched vLLM 30.8k result.

Supporting gates:

- profiled wall <= 2.4 s;
- GPU idle inside activity span <= 1.3 s;
- waves <= 140;
- normal c0/256 >= 99% of the 34.27k baseline;
- no regression on 512 x 512, 64 x 1,536, or 16 x 1,900.

### 6.2 Allocation

- `cudaMalloc` <= 4 per request after M1, then <= 1 amortized after M3;
- `cudaHostAlloc` <= 2 per request after M2, then <= 1 amortized after M3;
- no per-stage device allocation;
- no per-channel pinned-word allocation;
- bounded retained bytes with pressure release.

### 6.3 Correctness

- fixed-shape serial oracle exact;
- output-boundary oracle exact for
  1/2/15/16/17/31/32/33/63/64/65/127/128/129;
- four long serial outputs exact against the corrected control;
- concurrent outputs coherent and complete;
- 2,048-request run has zero failures;
- no readiness chain kills classified as fatal geometry corruption;
- lifecycle close/terminate tests cover late bind, late fire, orphan pipeline,
  and repeated fleet teardown.

### 6.4 Memory and lifecycle

- no driver-ceiling request;
- no stale scheduler quorum member;
- no leaked bind placeholder;
- no third-fleet stall;
- idle trim and explicit pressure trim remain correct;
- TP and Remote remain disabled for any new admission mode until separately
  validated.

## 7. Explicit Non-Goals

- Do not rewrite model kernels: they are already faster than vLLM in aggregate.
- Do not increase unbounded cache capacities to hide lifecycle leaks.
- Do not globally force scheduler depth 1.
- Do not precommit the full KV pool.
- Do not add automatic swap or eviction policy as a side effect.
- Do not pool WASM instances before proving that packed metadata and channel
  reuse are insufficient.
- Do not weaken channel readiness, poison, or close ordering.

## 8. Complexity Tradeoffs

Simplicity remains the governing principle.

The smallest useful change is a packed per-instance allocation. A fleet-level
allocator is more powerful but adds:

- free-list state;
- stream-retirement state;
- size-class policy;
- pressure interaction;
- shutdown ordering;
- new ABA and lifetime failure modes.

Therefore:

1. pack first;
2. measure allocation-count reduction;
3. add bounded reuse only if the remaining gap justifies it;
4. keep KV-weighted admission separate from storage pooling;
5. stop and report if correctness requires cross-subsystem state that cannot
   be explained by one ownership model.

## 9. Required Deliverables

- Implementation commits with no temporary diagnostics.
- Updated allocation and wave telemetry.
- Cold/warm/repeated-fleet profile artifacts.
- Pie/vLLM matched performance table.
- Accuracy comparison artifacts.
- CUDA, engine, ABI, and relevant Metal/runtime validation.
- Updated implementation report stating both recovered performance and any
  remaining explicit tradeoffs.
