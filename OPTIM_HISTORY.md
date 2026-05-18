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

`Qwen/Qwen3-32B` fits weights on 80GB GPUs in BF16, but the heavy `512x512`
workload is an admission and KV-residency stress test on A100/H100. It is now
part of the large-GPU validation set because it exposes planner mistakes that
8B hides.

### Workloads

- Common matrix for 0.6B/8B/14B: `1x128`, `16x128`, `128x128`, `512x128`,
  `128x512`, `512x512`.
- 32B large-GPU matrix: `1x128`, `16x128`, `32x128`, `64x128`, `16x512`,
  `32x512`, plus `512x512` as a stress row on A100/H100/H200/RTX.
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
- Direct embedded `tests/inferlets` runs still exposed a teardown hang in the
  Python embedded server path. CLI-server based correctness probes completed and
  produced coherent text, so this is tracked separately from CUDA forward
  correctness.

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
- One concrete TP startup bug was device parsing: `--device 0,1` wrote bare
  rank-local device strings into the generated config, while the CUDA driver only
  parsed `cuda:N`. Rank 1 could silently bind device 0. The driver now parses
  both forms.
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
- The final cross-device evidence favored 16-token pages as the automatic
  default:
  - L40 TP=2 Qwen3-8B 512x512 moved into the 7.3k output tok/s range.
  - A40 Qwen3-8B 512x512 stayed above standalone vLLM, around 4.1k-4.2k output
    tok/s.
  - H200 Qwen3-8B 512x512 moved from a small loss to a small win over vLLM,
    around 21.3k output tok/s.
  - H200 Qwen3-32B 512x512 stayed near parity but remained slightly below vLLM,
    so page size was not the remaining bottleneck there.
- Current conclusion: auto and latency score 16-token pages. Throughput/manual
  concrete profiles may still consider 32-token pages, but the public config no
  longer exposes page size as a knob.

## CUDA Graph Policy

- Pie captures an upfront lattice of decode graphs rather than capturing every
  exact batch count variant on demand.
- vLLM uses a capture-size lattice as well; its H200 logs showed capture sizes:
  `512, 504, 496, ..., 16, 8, 4, 2, 1`.
- Pie captures the decode lattice upfront. Recent rows captured 51 decode graphs
  for R=512 plans and 35 decode graphs for R=256 plans; occasional tail variants
  are still captured on demand when a non-lattice layout appears.

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

- Qwen3-8B and Qwen3-14B rows are competitive after the residency fixes.
- Qwen3-32B 512x512 remains a stress row. Admission overbooking improved the row
  from roughly 687 output tok/s to 766 output tok/s, and raising restore pause
  utilization after the FlashInfer workspace fix reached roughly 897 output
  tok/s before the KV-heavy R=256 retest.
- The A100 FlashInfer scratch estimate was wrong when it used the base 80 MiB
  attention workspace on SM80. Restoring to higher utilization exposed a
  `batch_prefill_tmp_v` overflow. The planner now sizes that float workspace for
  Ampere as well as Hopper/Blackwell.

### A40

- Full TP=1 matrix passed with Pie ahead of vLLM after the narrow-auto latency
  rule.
- Qwen3-8B representative results:
  - 1x128: Pie 33.57 vs vLLM 32.13
  - 128x512: Pie 2612.12 vs vLLM 2500.96
  - 512x512: Pie 3252.45 vs vLLM 2921.75

### H200

- H200 is the most sensitive device in the current matrix.
- H200 Qwen3-8B is now slightly ahead of standalone vLLM on the 512x128 and
  512x512 rows with 16-token pages and the current attention policy.
- H200 Qwen3-32B 512x512 is stable and close but still below the best vLLM row:
  Pie measured about 6.50k output tok/s versus a fresh vLLM result around
  6.68k output tok/s.
- Since H200 32B has enough resident KV for the full 512x512 row, the remaining
  gap is more likely attention-path overhead or scheduler/admission cadence than
  raw page capacity.

### RTX Pro 6000

- Pie runs successfully on the Blackwell RTX Pro 6000 host.
- Standalone vLLM/SGLang comparison was blocked by the installed CUDA/FlashInfer
  stack for SM12/Blackwell, so this host currently has Pie-only numbers.
- Blackwell is the only architecture where the 16k prefill candidate survived:
  - Qwen3-8B 512x512: about 10.3k-10.4k output tok/s.
  - Qwen3-32B 512x512: about 1.25k output tok/s before the latest planner
    retest.

## Rejected or Deprioritized Ideas

- Async channels for low-latency IPC were rejected because the serving target
  requires roughly 1 us latency. The spin should move closer to the blocking
  boundary rather than be replaced by async wakeups.
- Manual user-facing knobs for KV pages, page size, forward tokens, and request
  count were rejected. These are driver capabilities.
- Large response-fanout worker heuristics were removed. For token-only large
  batches, direct synchronous response emission was simpler and faster.
- Host KV swap was rejected for the default policy. On H100 Qwen3-32B 512x512,
  `swap_pool_size=16384` dropped throughput from about 2.0k to about 1.68k
  output tok/s.
- Capacity profile was rejected for long-output default serving. On H100
  Qwen3-32B 512x512 it chose R=64 and failed 298/512 requests.
- 16k prefill was rejected on Hopper because it regressed or stayed flat on
  H100/H200 8B rows. It remains Blackwell-only.

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

## 2026-05-18 Active Accepted Changes

- Default benchmark/server admission overbooking is now 4.0. This is not a
  forward-capacity override; it lets the market scheduler start more contexts
  and lets eviction/admission account for real page pressure. It materially
  improved H100 and A100 Qwen3-32B 512x512 while not affecting rows that already
  fit.
- Auto memory planning now prefers 16-token KV pages. This aligns with vLLM's
  default block size and improved the most important long-output rows.
- The auto planner uses a 128-token minimum KV horizon for KV-heavy models only
  on Blackwell or very-high-memory GPUs. Generalizing that low horizon to 80GB
  H100 was rejected: it changed Qwen3-32B 512x512 from R=128 to R=256 and
  dropped throughput from about 2.02k to 1.72k output tok/s. The 80GB rule uses
  a 256-token minimum horizon, which rejects R=256 but keeps the useful R=128
  plan.
- Attention scratch is sized from model shape and request capacity on Ampere as
  well as Hopper/Blackwell. This fixed the A100 restore-utilization crash.
- Throughput prefill candidates are capped at 8192 tokens on Ampere/Hopper and
  16384 tokens only on Blackwell.

## 2026-05-18 Matrix Continuation

- Clean-source rebuilds after the failed SM90 experiment reproduced the prior
  large-model picture:
  - Local L40 TP=2 Qwen3-8B 512x512: 7384.52 output tok/s.
  - H100 Qwen3-32B 512x512 auto: 2031.77 output tok/s, R=128,
    about 49k resident KV tokens.
  - H200 Qwen3-32B 512x512 auto: 6494.92 output tok/s, R=512,
    about 274k resident KV tokens.
  - A40 Qwen3-8B 1024x128 auto: 4540.95 output tok/s.
  - RTX Pro 6000 Qwen3-32B 256x1024 auto did not finish the row within the
    300s request timeout: 150/256 completed, 511.98 output tok/s over the
    partial run. The planner selected R=512 but only about 107k resident KV
    tokens, so this row is capacity-limited and needs either a smaller shape or
    a larger timeout to be a valid throughput comparison.
- H200 profile sweeps showed the auto scorer is the best of the current public
  profiles for Qwen3-32B 512x512:
  - auto: 6494.92 output tok/s.
  - throughput: 6136.35 output tok/s. It chose R=1024, page=32, N=4096 and
    paid extra attention workspace/graph pressure.
  - balanced: 5938.08 output tok/s. It chose R=256, page=32, N=2048 and
    doubled the number of steady-state batches.
  - eager scheduler with auto layout: 6507.24 output tok/s, only a tiny gain
    over adaptive. Adaptive is not the main H200 gap.
- A100 Qwen3-32B 512x512 with the cleaned source completed at 772.59 output
  tok/s with R=128 and about 50k resident KV tokens. This is still far behind
  the earlier vLLM baseline near 1322 output tok/s, so the A100 gap is not
  solved by capacity/profile tuning alone.
- Decode PDL launch handling was audited. FlashInfer's Python side enables PDL
  only for compute capability >= 9. Pie's decode path had always passed
  `enable_pdl=true`, while prefill already used the Hopper+ guard. The code now
  stores `enable_pdl` in the decode plan using the same device-capability
  policy as prefill. Early measurements:
  - Local L40 Qwen3-8B 512x512: 4664.78 output tok/s in a single-GPU run.
  - Local L40 Qwen3-8B 1024x128: 4747.96 output tok/s.
  - A40 Qwen3-8B 512x512: 4164.25 output tok/s, effectively neutral against
    earlier A40 runs.
  - A100 Qwen3-32B 512x512: 772.59 output tok/s, effectively neutral against
    the earlier clean-source A100 512x512 result.
  The PDL cleanup is retained as a correctness/consistency fix, not as a
  proven throughput win.
- FlashInfer contains XQA decode kernels under `csrc/xqa/` for Ampere, Hopper,
  and Blackwell, plus Python tests for `xqa_batch_decode_with_kv_cache`.
  These kernels take a dense per-request page table and VLLM-style KV layout.
  Pie currently uses FlashInfer generic paged decode/prefill with a ragged page
  index list and separate K/V page buffers. The next plausible kernel-level
  optimization is a narrow XQA integration for BF16, head_dim=128, page=16,
  GQA group sizes used by Qwen-like models, with a standalone parity/perf test
  before wiring it into serving.

## 2026-05-18 vLLM 0.21 and Negative Kernel Results

- The latest vLLM stack tested here is 0.21.0, not 0.10. Its default attention
  backend was not always the fastest on Hopper:
  - H100 Qwen3-14B 512x512: vLLM 0.21 default was about 14.02k output tok/s;
    forcing FlashInfer with `attention_config={"backend": "FLASHINFER"}` reached
    14.33k output tok/s.
  - H200 Qwen3-32B 256x1024: vLLM 0.21 default was about 5.74k-5.75k output
    tok/s; forced FlashInfer reached 5.88k output tok/s.
  - A100 Qwen3-14B 512x512: vLLM 0.21 default was about 5.61k output tok/s.
- Clean Pie after the reverted experiments measured:
  - H100 Qwen3-14B 512x512: about 13.62k output tok/s.
  - H200 Qwen3-32B 256x1024: about 5.53k output tok/s.
  The remaining gap on these rows is about 5%-6%, with KV residency already
  close to vLLM. That points away from the memory planner as the sole cause.
- Rejected kernel/path experiments from this continuation:
  - Widening FlashInfer decode GQA dispatch to include group sizes 5/6/7 made
    H100 Qwen3-14B 512x512 worse, about 13.05k output tok/s. The generic prefill
    fallback remains better for Qwen3-14B's GQA=5.
  - A separate no-sliding-window full-attention variant for prefill-decode was
    neutral/slightly worse on H200 Qwen3-32B 256x1024, about 5.50k output tok/s.
  - Replacing TP=1 cuBLAS `beta=1` residual GEMMs with scratch GEMM plus fused
    residual-RMSNorm regressed both H100 and H200. Extra scratch traffic cost
    more than the fusion saved.
  - Preparing a prefill-decode plan for forced-prefill GQA fallback caused a
    startup/load hang and was reverted.
  - A general BF16 cuBLASLt route, pre-initialized for graph capture safety,
    regressed H100 Qwen3-14B 512x512 to about 13.61k output tok/s and H200
    Qwen3-32B 256x1024 to about 5.48k output tok/s. Classic `cublasGemmEx`
    remains the BF16 path for now.
- Pass-level speculation is not the throughput regression:
  - H100 Qwen3-14B 512x512 with `speculation_depth=0` fell to about 10.76k
    output tok/s.
  - H200 Qwen3-32B 256x1024 with `speculation_depth=0` fell to about 5.32k
    output tok/s.
  - `speculation_depth=2` did not improve the no-WASM benchmark; H100
    Qwen3-14B 512x512 was about 13.51k output tok/s. Depth 1 remains the
    right default for the current simple serving path.
- Explicit `memory_profile="throughput"` is not a better auto target for these
  latest rows:
  - H100 Qwen3-14B 512x512 was about 13.56k output tok/s, below auto.
  - H200 Qwen3-32B 256x1024 selected R=1024/page=32 and fell to about 2.88k
    output tok/s. The larger request/page metadata shape hurt cadence badly.
- Retesting the force-prefill planned path for unsupported FlashInfer decode
  GQA groups reproduced the previous startup stall after KV-cache logging.
  The candidate would be the right architectural direction for Qwen3-14B
  GQA=5, but the current FlashInfer prefill-plan + upfront graph-capture
  interaction is not safe enough to keep.
- Local L40 TP=2 was rechecked after reverting the failed candidates:
  - Pie Qwen3-8B 512x512 adaptive: 7260.05 output tok/s.
  - Pie Qwen3-8B 512x512 eager: 7243.21 output tok/s.
  - vLLM 0.20.2 Qwen3-8B 512x512: 7262.88 output tok/s.
  Direct Pie startup confirmed the custom all-reduce path is active. The local
  L40 pair is `SYS` topology across CPU/NUMA domains, so this row is effectively
  TP parity rather than a clean interconnect win/loss signal.
- After cleaning up the custom all-reduce topology flag and rebuilding, the same
  local L40 TP=2 Qwen3-8B 512x512 row measured 7338.53 output tok/s. No planner
  change was involved; this is within the normal local-run variance but confirms
  the cleanup did not regress TP2.
- A vLLM-like long-KV upfront graph-capture experiment was rejected:
  - Local L40 TP=2 rerun was healthy at 7371.28 output tok/s, but one run
    showed a transient admission drop to 500 requests.
  - H200 Qwen3-32B 256x1024 captured 102 graphs instead of 51 but stayed flat at
    5513.85 output tok/s.
  - H100 Qwen3-14B 512x512 regressed to 13548.56 output tok/s.
  The result says Pie's remaining large-model gap is not explained by missing
  upfront capture of the long-KV layout alone.
- A narrower GQA=5 FlashInfer decode experiment was also rejected. It added only
  group size 5 and allowed split-KV on Hopper for that group, but H100
  Qwen3-14B reproduced the same startup stall pattern seen in the earlier
  force-prefill planning attempt: startup reached KV-cache logging, never
  reached graph-capture logging, and GPU memory fell back to about 529 MiB. The
  fallback prefill path remains the only safe Qwen3-14B path for now.
- vLLM 0.21 forced-FlashInfer eager runs showed the remaining Hopper gap is not
  primarily vLLM's Inductor/CUDA-graph model-body capture:
  - H100 Qwen3-14B 512x512 compiled FI: 14386.97 output tok/s.
  - H100 Qwen3-14B 512x512 eager FI: 14250.83 output tok/s.
  - H200 Qwen3-32B 256x1024 compiled FI: 5884.05 output tok/s.
  - H200 Qwen3-32B 256x1024 eager FI: 5788.74 output tok/s.
  The vLLM eager path keeps almost all of the lead, pointing back to kernel
  choices and model-body implementation rather than scheduler/capture policy.
- A Hopper `N=16384` auto-prefill candidate was rejected. It matched vLLM's
  max batched-token scale but traded too much KV capacity for the larger arena:
  - H100 Qwen3-14B 512x512 improved only slightly, 13662.10 -> 13745.20 output
    tok/s.
  - H200 Qwen3-32B 256x1024 fell badly, 5507.43 -> 2878.59 output tok/s, with
    KV residency dropping to about 265k tokens for a workload that needs roughly
    the full 256x1024 tail plus prompts.
  The Hopper auto prefill cap stays at 8192; Blackwell keeps the larger cap.

## 2026-05-18 Current Clean-State Validation

- PyPI and the H100/H200 venvs now agree that the current vLLM baseline is
  0.21.0. H100 was already on `/root/Workspace/vllm021-venv`; H200 was updated
  to a separate `/root/Workspace/vllm021-venv` so the large-model rows are not
  compared against the older system vLLM 0.10.2.
- Correctness/coherence checks on local L40 with Qwen3-8B and `cuda_native`
  passed:
  - `test_text_completion.py`: coherent answer naming Paris. The duplicated
    text in the output file is the harness concatenating stdout and return.
  - `test_parallel_generation.py`: both sampled responses had coherent
    Qwen-style reasoning prefixes.
  - `test_json_schema_validation.py`: generated valid, semantically plausible
    structured JSON for Alice in San Francisco.
  - `test_output_validation.py`: probability mass correctly favored
    `David Chen`.
  - `test_constrained_decoding.py`: valid constrained JSON, but the constrained
    sampler produced semantically odd names (`":"`). This looks grammar/sampler
    related rather than forward-pass corruption because the unconstrained and
    schema paths are coherent.
- Focused Rust validation after the cleanup:
  - `cargo build -p pie-server --release --no-default-features --features driver-cuda`
    passes locally.
  - `cargo test -p pie-server --no-default-features --features driver-cuda config`
    passes.
  - `cargo test -p pie --lib scheduler`, `adaptive_policy`, and `sched` pass.
- H100 Qwen3-14B, 512 requests x 512 output tokens, `gpu_mem_util=0.90`,
  standalone vLLM 0.21 default versus Pie:
  - vLLM default: 13,407.42 output tok/s.
  - Pie auto before the auto page-size scoring tweak: 13,450.59 output tok/s.
  - Pie explicit throughput: 13,682.86 output tok/s.
  - Pie auto after the auto page-size scoring tweak: 13,499.87 output tok/s.
  The default Pie path still beats default standalone vLLM on this row, but the
  small spread says this workload is noisy enough that page-size scoring should
  not be overinterpreted from a single run.
- H100 Qwen3-14B, sequential latency, 16 requests x 128 output tokens:
  - vLLM 0.21 default: mean 1415.3 ms, 90.44 output tok/s.
  - Pie auto: mean 1438.6 ms, 88.97 output tok/s.
  - Pie explicit latency profile: mean 1444.4 ms, 88.62 output tok/s.
  - Pie adaptive policy instead of the latency benchmark's greedy policy:
    mean 1446.2 ms, 88.50 output tok/s.
  Pie is close but still slightly behind for pure single-request latency; the
  gap is about 20-30 ms per 128-token request and is mostly not fixed by memory
  profile or scheduler policy.
- H200 Qwen3-32B, 256 requests x 512 output tokens, standalone vLLM 0.21 default
  versus Pie:
  - vLLM default: 6,211.66 output tok/s.
  - Pie auto: 5,894.63 output tok/s.
  - Pie throughput: 5,887.65 output tok/s.
  - Pie latency profile: 5,896.88 output tok/s.
  - Pie speculation depth 0: 5,757.28 output tok/s.
  - Pie greedy policy: 4,400.70 output tok/s.
  - Raising the full-attention prefill-decode threshold for KV-heavy models to
    512 requests was neutral at 5,898.08 output tok/s and was reverted.
  The 32B gap is not a planner profile problem. KV residency is comparable
  (`vLLM 249k tokens`, `Pie 246k-253k tokens` depending on profile), and the
  best Pie rows all cluster around 5.9k. The remaining gap is likely in the
  GQA=5 attention/model-body path and a smaller amount of runtime overhead.
- Attempting to force vLLM FlashInfer via `VLLM_ATTENTION_BACKEND=FLASHINFER`
  on vLLM 0.21 is not valid anymore; vLLM logs it as an unknown environment
  variable and still chooses `FLASH_ATTN` for attention. Future vLLM backend
  A/Bs should use the current vLLM 0.21 configuration API rather than the old
  environment variable.
