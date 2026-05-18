# Optimization History

This file records the CUDA memory-planning and throughput debugging work from
the `agents/alpha` branch. It is intentionally measurement-oriented: each item
captures the hypothesis, what was tested, and what survived.

## Goals

- Driver capabilities should be derived by the driver, not configured by users.
- CUDA should derive KV page size, KV page count, forward token capacity, request
  capacity, and page-reference capacity from model shape and device memory.
- Runtime scheduling should use driver-reported limits and avoid model-specific
  knobs.
- Pie should be correct under stress and competitive with standalone vLLM on
  latency and throughput for TP=1 and TP=2 where hardware allows.
- Benchmark tiers now include Qwen3-0.6B, Qwen3-8B, and Qwen3-14B. The 14B
  target is the larger cross-device model; 32B is intentionally not the default
  matrix model because it turns 48-80GB devices into memory-fit/admission tests.

## Evaluation Matrix

The optimization loop is intentionally parallel and iterative. A candidate
heuristic is tested across representative GPUs and workload shapes before it is
accepted; device-only hacks are rejected even if they improve one benchmark row.

### Devices

- L40 local pair: TP=2 correctness, TP=2 throughput, and lower-bandwidth Ada
  behavior.
- A40: lower-SM Ampere guardrail.
- A100-SXM4-80GB: high-end Ampere guardrail.
- H100-SXM-80GB: primary Hopper baseline.
- H200: high-bandwidth Hopper and current tight vLLM comparison.
- RTX Pro 6000: Blackwell guardrail; vLLM/SGLang comparison should be retried
  with the installed stack before treating it as Pie-only.

### Model Tiers

- Small/narrow: `Qwen/Qwen3-0.6B`.
- Standard: `Qwen/Qwen3-8B`.
- Large cross-device: `Qwen/Qwen3-14B`.
- Extra-large large-GPU-only: `Qwen/Qwen3-32B`.

`Qwen/Qwen3-32B` should fit weights on 80GB GPUs in BF16, but not the heavy
`512x512` residency workload. It is used only on A100/H100/H200/RTX Pro 6000
with smaller shapes.

### Workloads

- Common matrix for 0.6B/8B/14B: `1x128`, `16x128`, `128x128`, `512x128`,
  `128x512`, `512x512`.
- 32B large-GPU matrix: `1x128`, `16x128`, `32x128`, `64x128`, `16x512`,
  `32x512`, promoted to larger request counts only when the driver admits them
  with healthy KV headroom.
- TP matrix: local L40 TP=2 for Qwen3-8B and Qwen3-14B where memory allows;
  additional TP machines are used if available.

## Capability and Config Cleanup

- Folded the separate forward-capacity concept into `DriverCapability`.
- Removed public CUDA sizing knobs such as manual forward tokens, request count,
  KV pages, and KV page size.
- Portable, dummy, vLLM, and SGLang-facing drivers now report resolved
  capabilities instead of asking runtime/users to provide raw capacity knobs.
- Runtime scheduling is capacity-driven:
  - `total_tokens <= max_forward_tokens`
  - `request_count <= max_forward_requests`
  - `page_refs <= max_page_refs`
- Legacy `max_num_kv_pages` and user-facing `kv_page_size` style knobs were
  removed from normal config paths.

## Correctness Findings

- The earlier `<think>!<think>!` text-completion output was a real forward-path
  issue, not just sampling noise.
- The important correctness fix was CUDA graph capture safety around the decode
  plan:
  - planning happens outside graph capture,
  - captured graph bodies only read stable buffers,
  - graph layout is keyed by the decode/prefill-decode layout class.
- In-proc values were inspected: request metadata, speculation inputs, and next
  speculation data were flowing through the channel correctly.
- After the graph fix, inferlet outputs for text completion, sampler paths,
  constrained decoding, parallel generation, and speculative text completion were
  manually inspected and were coherent.

## Runtime/Scheduler Findings

- Token budget is admission control, not a forward-pass capacity limit. It
  prevents over-admitting work that would immediately exceed the KV/page budget.
- Over-admitting and evicting later can improve early GPU occupancy, but it also
  increases churn. The current policy is conservative and predictable.
- Market scheduling is conceptually aligned with agentic inference because
  contexts can pause, resume, and compete for long-lived KV residency. For simple
  text-completion serving, it must reduce to efficient batch formation and avoid
  adding per-request overhead.
- The default bid strategy is acceptable as a simple serving baseline, but
  inferlet-level default bids remain the cleaner long-term design.
- Removed over-optimization/debug paths that did not justify complexity, such as
  response fanout worker heuristics and related flags.

## TP Findings

- A TP=2 hang pattern where one GPU sat at 100% and the other at 0% was not a
  scheduling issue. It was a rank-divergence/collective-path issue.
- Restoring the fused residual-plus-RMSNorm custom all-reduce path fixed the
  local TP2 hang and improved throughput.
- Local L40 TP=2 Qwen3-8B 512x512 completed 512/512 requests after the fix.

## Memory Planner Findings

- The planner must load weights first, then compute budget from actual device
  memory:
  - `usable = total_vram * gpu_mem_utilization`
  - `budget = usable - current_used - safety_headroom`
  - `safety_headroom = max(512 MiB, 5% total_vram)`
- `N` (forward tokens) is the primary activation dimension; request count is a
  decode occupancy dimension.
- The robust base policy is a candidate search over concrete profiles, scored
  by prefill saturation, decode saturation, KV residency, page size, and memory
  pressure.
- Auto profile is not a fifth concrete profile. It searches the concrete policy
  families and scores them under a unified objective.
- For narrow models on lower-SM GPUs, auto should collapse to the latency-shaped
  policy. This fixed A40 Qwen3-0.6B regressions.
- Capacity and latency profiles should use a smaller decode target than balanced
  and throughput. This reduced unnecessary decode/KV pressure.

## KV Page Size Findings

- vLLM generally uses 16-token blocks by default.
- Pie tested both 16-token and 32-token KV pages through the planner.
- On H200 Qwen3-8B, forcing planner preference toward 16-token pages was worse:
  - 512x128 dropped to about 21.3k output tok/s versus about 22.2k with 32-token
    pages.
  - 512x512 also remained below the best 32-token-page runs.
- Current conclusion: 32-token pages are still the better Pie default for
  throughput-oriented CUDA serving, while latency profile can keep 16-token
  pages for finer granularity.

## CUDA Graph Policy

- Pie captures an upfront lattice of decode graphs rather than capturing every
  exact batch count variant on demand.
- vLLM uses a capture-size lattice as well; its H200 logs showed capture sizes:
  `512, 504, 496, ..., 16, 8, 4, 2, 1`.
- Pie H200 Qwen3-8B startup captured 83 decode graphs, using about 142 MiB graph
  memory and under 1 second capture time in the measured runs.

## FlashInfer Attention Findings

- Dedicated decode is faster for shallow KV histories on Hopper/Blackwell.
- FlashInfer prefill-decode becomes useful after the average KV history is deep
  enough.
- A single threshold is too blunt:
  - Earlier prefill-decode improves long-output batches.
  - Too-early prefill-decode hurts short-output, high-request batches such as
    512x128.
- Full-attention prefill-decode is a second, separate decision:
  - Using the full-attention variant as soon as prefill-decode starts hurt
    512x128 badly.
  - It helped the 512x512 tail.
- The cleaner policy is two-stage:
  - start prefill-decode after a KV-page threshold,
  - switch to the full-attention prefill-decode variant only for wide cohorts
    with deeper KV histories.

## Device Matrix Notes

### A100-SXM4-80GB

- Full TP=1 matrix passed with Pie ahead of vLLM for Qwen3-0.6B and Qwen3-8B.
- Qwen3-8B representative results:
  - 1x128: Pie 90.32 vs vLLM 84.97
  - 128x128: Pie 6592.91 vs vLLM 6453.75
  - 512x512: Pie 8788.30 vs vLLM 8573.68

### A40

- Full TP=1 matrix passed with Pie ahead of vLLM after the narrow-auto latency
  rule.
- Qwen3-8B representative results:
  - 1x128: Pie 33.57 vs vLLM 32.13
  - 128x512: Pie 2612.12 vs vLLM 2500.96
  - 512x512: Pie 3252.45 vs vLLM 2921.75

### H200

- H200 is the most sensitive device in the current matrix.
- The memory planner is not the primary bottleneck on H200:
  - Qwen3-8B plan reported 4096 forward tokens, 1024 request capacity,
    524288 page refs, 32-token pages, about 716k KV tokens, and about
    4756 MiB forward arena.
- vLLM on the same H200 reports chunked prefill enabled with
  `max_num_batched_tokens=16384`, 512 max seqs, and about 783k KV tokens.
- H200 Qwen3-0.6B is comfortably ahead of vLLM across the tested TP=1 shapes.
- H200 Qwen3-8B is near parity and sensitive to attention path thresholds:
  - `prefill_decode_min_kv_pages=7` helped 512x128 and 128x512 but had a small
    512x512 miss in one full run.
  - `prefill_decode_min_kv_pages=6` helped a focused 512x512 run but could lose
    512x128 in a full run.
  - Full-attention at 512 requests without a KV-depth guard hurt 512x128.
  - Two-stage full-attention with a KV-depth guard brought 512x128 and 128x512
    back above vLLM, while 512x512 remained within measurement noise.

### RTX Pro 6000

- Pie runs successfully on the Blackwell RTX Pro 6000 host.
- Standalone vLLM/SGLang comparison was blocked by the installed CUDA/FlashInfer
  stack for SM12/Blackwell, so this host currently has Pie-only numbers.
- Pie-only final probe with the Hopper/Blackwell threshold work:
  - Qwen3-0.6B 512x512: about 25.4k output tok/s
  - Qwen3-8B 512x128: about 11.8k output tok/s
  - Qwen3-8B 512x512: about 10.4k output tok/s

## Rejected or Deprioritized Ideas

- Async channels for low-latency IPC were rejected because the serving target
  requires roughly 1 us latency. The spin should move closer to the blocking
  boundary rather than be replaced by async wakeups.
- Manual user-facing knobs for KV pages, page size, forward tokens, and request
  count were rejected. These are driver capabilities.
- Large response-fanout worker heuristics were removed. For token-only large
  batches, direct synchronous response emission was simpler and faster.
- Forcing 16-token KV pages on H200 was rejected for Pie throughput.

## Current Work in Progress

- The serving path already used fused QKV and fused gate/up projections through
  `llama_like_forward_paged`; a later parity-path cleanup wired the same fused
  dispatch into `qwen3_forward_paged`, but that did not affect serving
  throughput.
- Large-model long-output losses are primarily KV-residency losses, not raw
  matmul speed:
  - A100 Qwen3-14B 512x512: Pie reported about 230k KV tokens while vLLM
    reported about 268k.
  - H200 Qwen3-32B 512x512: Pie reported about 214k KV tokens while vLLM
    reported about 252k.
  - In both cases Pie was evicting/running waves where vLLM kept a larger
    resident set.
- Two general over-reservations explain most of that delta:
  - output logits/probability scratch was sized by `max_forward_tokens`, even
    though serving needs rows for sampler/logit outputs;
  - the planner subtracted a fixed 5% VRAM reserve on top of
    `gpu_mem_utilization`, making `0.90` behave more like `0.85` on 80GB+
    devices.
- Active candidate: size logits/probability scratch by output-row capacity and
  bound the graph/runtime reserve to `max(1 GiB, 1% VRAM)` capped at 2 GiB.
  This should convert the freed memory into KV pages while preserving a real
  post-planning reserve for CUDA graphs, allocator slack, and runtime buffers.
- H100 Qwen3-8B remains a distinct latency-throughput gap against vLLM. It
  already has enough KV for 512x512, so fixes that only increase resident KV are
  not expected to close that row by themselves.
- A high-SM Hopper/Blackwell 8192-token prefill arena was tested and rejected.
  It reduced H200 Qwen3-8B 512x512 throughput to about 20.2k output tok/s,
  below the 4096-token arena runs near 21.1k.
- Planning prefill-decode CUDA graphs with a synthetic 16,384-token row budget
  was also rejected. It allowed FlashInfer to request more split-KV workspace
  than the fixed attention arena had reserved and produced 0 completed requests
  in the H200 512x512 probe.
- The active hypothesis is back on attention path selection: when to use
  dedicated decode, sliding-capable prefill-decode, and full-attention
  prefill-decode for wide Hopper/Blackwell decode cohorts.

## 2026-05-18 Continued Tuning Notes

- Output-row arena compaction plus a smaller runtime reserve fixed the raw KV
  deficit on A100/H200, but it did not by itself close throughput:
  - H100 Qwen3-8B 512x512 stayed around 17.6k-18.1k tok/s while vLLM was about
    19.9k-21.3k in nearby runs.
  - A100 Qwen3-14B 512x512 improved from about 3.6k at util=0.90 to about
    5.5k at util=0.95, enough to beat the earlier vLLM 0.90 baseline.
  - H200 Qwen3-32B 512x512 improved only from about 4.1k to about 4.35k at
    util=0.95, still behind vLLM's 0.90 result of about 6.1k.
- Runtime now enforces `max_logit_rows` and `max_prob_rows` from
  `DriverCapabilities`. The CUDA arena may size logits/probability rows by
  output-row capacity, and scheduler admission now preserves correctness for
  rich samplers/probes that need dense rows.
- Enabling the prefill-decode path on Ampere did not materially improve A100
  Qwen3-14B at util=0.90. The real A100 win came from KV residency at higher
  memory utilization, not from the attention-path change.
- Lowering the Hopper full-attention prefill-decode threshold from 512 to 384
  was neutral on H200 Qwen3-32B 512x512. It does not explain the remaining gap.
- A general BF16 cuBLASLt path was smoke-tested and benchmarked:
  - H100 Qwen3-8B 512x512: about 17.65k tok/s.
  - A100 Qwen3-14B 512x512 util=0.95: about 5.45k tok/s.
  - H200 Qwen3-32B 512x512 util=0.95: about 4.34k tok/s.
  The candidate is neutral/slightly worse than classic cuBLAS for these
  shapes, so it should be reverted unless a later workload proves otherwise.
- The H200 32B long-output gap now looks more like admission behavior than raw
  memory planning: Pie starts the whole generation as two cohorts
  (roughly 478 active + 34 tail) because the token budget reserves the full
  608-token horizon per request. vLLM has similar KV capacity but appears to
  admit the larger cohort for longer and pays capacity pressure near the end.
  Active experiment: sweep `admission_oversubscription_factor` on H200 32B.
