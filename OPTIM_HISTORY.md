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

## TP1 Unlimited vLLM Chase Log

This section records the current TP=1, unlimited-concurrency comparison loop.
Do not retry an item here unless the implementation or benchmark harness has
changed materially. Pie runs use `--concurrency 0 --ipc-profile latency`.
Every new candidate should record the source/config delta, benchmark artifact,
throughput, keep/reject decision, and the reason. Failed rows are valuable data;
do not repeat them by changing only the output file name.

### Baselines

- vLLM Qwen3-0.6B 512x128: 28,141 output tok/s
  (`/root/e2e_bench/experiments/pie_dense_spec/vllm_06b_tp1_unlimited_512_flashinfer_current_rerun.json`).
- vLLM Qwen3-1.7B 512x128: 17,851 output tok/s
  (`/root/e2e_bench/experiments/pie_dense_spec/vllm_17b_tp1_unlimited_512_flashinfer_current_rerun.json`).
- vLLM Qwen3-4B 512x128: 8,075 output tok/s
  (`/root/e2e_bench/experiments/pie_dense_spec/vllm_4b_tp1_unlimited_512_flashinfer_rerun.json`).

### Current Best Rows

| Model | Best Pie row | vLLM row | Decision |
| --- | ---: | ---: | --- |
| Qwen3-0.6B | 28,159 observed with shared page spans; 27,865 rerun | 28,141 | Not reliable/large enough |
| Qwen3-1.7B | 17,186 | 17,851 | Improved, still below |
| Qwen3-4B | 8,148 observed best; 8,145 shared-page-span validation | 8,075 | Kept, small win |

- 0.6B observed best:
  `/root/e2e_bench/experiments/pie_dense_spec/pie_06b_shared_pagespan_512x128.json`.
- 0.6B shared-page-span rerun:
  `/root/e2e_bench/experiments/pie_dense_spec/pie_06b_shared_pagespan_rerun_512x128.json`.
- 1.7B current best:
  `/root/e2e_bench/experiments/pie_dense_spec/pie_17b_shared_pagespan_512x128.json`.
- 4B current best:
  `/root/e2e_bench/experiments/pie_dense_spec/pie_4b_hnd_depth1_512x128.json`.
- 4B shared-page-span validation:
  `/root/e2e_bench/experiments/pie_dense_spec/pie_4b_shared_pagespan_512x128.json`.

### Successful or Kept

- 4B default TP1 already edges vLLM: Pie 8,101 output tok/s vs vLLM 8,075.
- 0.6B prefers HND KV layout with speculation depth 3. Best current-code rerun
  reached 28,056 output tok/s, but this is not yet a reliable win over vLLM.
- 1.7B default NHD/page16 with speculation depth 3 reached about 16,849 output
  tok/s after host and mask fixes, but was superseded by the 6144-token
  workspace/depth-1 HND row at 17,098 output tok/s.
- 1.7B benefits from a smaller CUDA workspace and shallower pass speculation:
  `PIE_CUDA_PREFILL_TOKENS=6144` with speculation depth 1 measured 17,040
  output tok/s, versus 16,644 for the 20,480-token/depth-3 current row. The
  same 6144-token workspace regressed 0.6B HND/depth3 to 26,431 output tok/s,
  so this is model-specific, not a global default.
- Combining the 1.7B 6144-token workspace/depth-1 shape with HND KV layout
  improved the best 1.7B row to 17,098 output tok/s. It is still below the
  vLLM baseline at 17,851, but it supersedes the prior NHD 6144/depth-1 row
  for this model.
- Scheduler/speculator physical-page lists now use shared page spans for cold
  and pre-staged decode submits instead of cloning the active page-prefix
  vector at every chain stage. This reduced token-response fanout and is kept:
  0.6B measured 28,159 output tok/s once and 27,865 on rerun; 1.7B improved
  the kept row from 17,098 to 17,186 output tok/s; 4B stayed above vLLM at
  8,145 output tok/s. The 0.6B result is not stable enough to count as a
  large-margin vLLM win.
- Cached prefill planning now reuses host `qo_indptr` / `kv_indptr` buffers.
- Cached prefill dispatch now keys and dispatches the causal-mask mode
  correctly. Real prefill uses causal masking; decode-as-prefill remains
  non-causal to match the FlashInfer decode wrapper semantics.
- Scheduler batch construction now does one pre-scan for exact vector reserves
  and decode-mask elision. 0.6B improved from about 27,549 to 27,657 output
  tok/s in the measured run.
- Token-only direct fanout bypasses `PendingRequest::send_result` for direct
  completions. 0.6B improved to about 27,840 output tok/s in the measured run.
- XQA ratio-2 decode support was added, but it is only useful when the model
  also satisfies the compiled page-size/head-dim shape.
- SDK generation now refreshes market bids at page cadence instead of on every
  token. On the current 0.6B HND/depth3 row this improved a restored rerun from
  27,360 to 27,611 output tok/s and reduced e2e response fanout from about
  1.85 ms to about 1.49 ms per scheduler batch.
- 4B remains ahead after the SDK page-cadence change. Current verification
  measured 8,142 output tok/s versus the vLLM baseline at 8,075 output tok/s.
- 4B with HND KV layout and depth 1 slightly improves the kept TP1 row to
  8,148 output tok/s, versus the vLLM baseline at 8,075. This is a small
  model-specific improvement, not a large-margin win.
- A clean 0.6B rerun after reverting the CUDA prefill-template experiment
  measured 27,513 output tok/s. This confirms the reverted build is back in
  the prior current-code range, but still below the vLLM 28,141 baseline.

### Failed or Reverted

- Static non-split FlashInfer decode plan bypass regressed 0.6B and is now
  default-off behind `PIE_CUDA_STATIC_DECODE_PLAN`.
- FlashInfer prefill-decode for the 0.6B short-output row regressed; dedicated
  decode remains the default there.
- 1.7B page32 to enable XQA ratio-2 decode regressed from about 16,849 to
  16,226 output tok/s. Keep page16/default for 1.7B.
- 1.7B HND KV layout at the default 20,480-token workspace/depth-3 shape
  regressed versus default NHD/page16. The later 6144-token/depth-1 HND row is
  a separate, kept model-specific setting.
- 1.7B decode full-attention off regressed.
- 1.7B speculation depth 4 regressed versus depth 3.
- 0.6B speculation depth 2 and depth 4 both regressed versus depth 3.
- 0.6B cohort waits of 32 ms and 128 ms both regressed versus 64 ms.
- Replacing the staged-chain map with `DashMap` regressed and was reverted.
- Removing the per-token registry lock in the chain extender regressed and was
  reverted.
- Pass-level staged-hit aggregation regressed as a default path. 0.6B measured
  27,814 output tok/s and 1.7B measured 16,728 output tok/s; both were below
  the prior best rows. The code remains opt-in behind
  `PIE_SPEC_HIT_AGG_MAX_EXTRA` for future workloads where successors are ready
  often enough to amortize the extra bookkeeping.
- API-side staged-hit `try_recv` before awaiting the staged oneshot did not
  beat the best rows: 0.6B measured 27,215 output tok/s and 1.7B measured
  16,711 output tok/s. It was reverted to the straight await path.
- Page16 XQA for GQA2 turned `xqa_decode=on` for 1.7B/default page16 but
  regressed to 15,709 output tok/s. The p16 specialization remains opt-in
  behind `PIE_CUDA_XQA_GQA2_P16`; keep it off for the TP1 unlimited row.
- Awaited staged-hit aggregation (`PIE_SPEC_HIT_AGG_WAIT_EXTRA=1`) broke batch
  accounting during 1.7B warmup (`expected 190, got 0`) and was removed rather
  than left as a broken opt-in path.
- Forced full-attention prefill-decode on 1.7B regressed slightly to 16,684
  output tok/s, so the default dedicated decode path remains better for the
  512x128 row.
- Decode-as-prefill with FlashInfer's internal graph plan disabled did not make
  the prefill-decode path viable on the TP1 unlimited 1.7B row. With
  `PIE_CUDA_PREFILL_DECODE_PLAN=1` and min pages forced to zero, throughput was
  16,591 output tok/s, below the dedicated-decode baseline.
- Token-only response view trimming did not improve the tight 0.6B row. The
  HND/depth3 run measured 27,595 output tok/s with worse e2e fanout/queue than
  the best kept rows, so the response view was restored.
- Zero-price market tick fast path regressed the 0.6B HND/depth3 TP1 row to
  27,006 output tok/s, despite skipping rent/dividend work when clearing price
  is zero. It was reverted.
- 0.6B with `PIE_CUDA_DECODE_FULL_ATTENTION=0` regressed to 26,741 output tok/s.
  Keep the full-attention decode variant enabled for the HND/depth3 row.
- 0.6B `--batch-policy eager` and `--batch-policy greedy` regressed to 26,567
  and 26,663 output tok/s respectively. Keep adaptive scheduling for unlimited
  throughput.
- Nsight Systems profiling works on this host, but `nsys profile` automatic
  import may leave only `.qdstrm`. Manual import with
  `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`, followed by
  `nsys export -t sqlite`, produced usable SQLite traces.
- 0.6B Nsight/e2e breakdown showed the measured batch path is dominated by
  driver/GPU wait, not scheduler construction: in
  `/root/e2e_bench/experiments/pie_dense_spec/pie_06b_nsys_runtime_512.json`,
  measured driver forward averaged about 14.84 ms/batch, batch build about
  0.27 ms/batch, and response fanout about 1.34 ms/batch. CUDA API launch/copy
  overhead in the measured window was small; `cudaStreamSynchronize` dominated
  host-visible CUDA time.
- Opt-in response-fanout sub-timers were added behind
  `PIE_SCHED_FANOUT_BREAKDOWN=1`. The profiling run
  `/root/e2e_bench/experiments/pie_dense_spec/pie_06b_fanout_breakdown_512x128.json`
  measured 27,687 output tok/s with instrumentation overhead. It attributed
  only about 0.15 ms/batch to direct `oneshot` sends and near-zero time to token
  output construction/classification; most of the fanout bucket is destructor /
  deallocation work outside those substeps.
- `PIE_FLASHINFER_FORCE_SPLIT_KV_SMALL=1` regressed the 0.6B HND/depth3 TP1
  unlimited row to 27,609 output tok/s. Keep Pie's small-batch non-split
  override for this workload.
- Partitioning the single-GPU greedy argmax did not close the 0.6B gap.
  `PIE_ARGMAX_PARTS=4` regressed to 27,471 output tok/s, and
  `PIE_ARGMAX_PARTS=2` reached 27,930 output tok/s but still trailed the best
  default HND/depth3 row. Keep the default one-block argmax for this shape.
- Forcing alternate cuBLASLt BF16 heuristic indices for the 0.6B HND/depth3
  row regressed. `PIE_CUBLASLT_BF16_ALGO_INDEX=3` reached 26,519 output tok/s,
  index 1 reached 27,253 output tok/s, and index 0 reached 25,010 output tok/s.
  Keep the current shape-aware default, which selects index 2 for the
  H=1024 wide `lm_head` shape.
- Scheduler dense-token fanout detection plus API-side staged-hit clone deferral
  regressed the 0.6B HND/depth3 row to 27,641 output tok/s. The e2e response
  fanout counter rose to about 1.87 ms per scheduler batch, so the dense indptr
  scan was removed.
- Keeping only the API-side sampler/lineage clone deferral still regressed the
  same row to 27,151 output tok/s, with e2e response fanout rising to about
  1.97 ms per scheduler batch. The API path was restored to the prior clone
  behavior.
- SDK-side buffer reuse plus stack single-token position slices regressed the
  0.6B HND/depth3 row to 27,115 output tok/s and raised e2e fanout to about
  2.07 ms. It was reverted.
- Refreshing generation bids only once per 512-token reservation window
  regressed the 0.6B HND/depth3 row to 26,664 output tok/s. Keep page-cadence
  bid refreshes; the less frequent update appears to hurt scheduling/chain
  timing despite fewer WIT calls.
- Rechecking current-code 1.7B with speculation depth 1 did not reproduce the
  older faster rows. It measured 16,615 output tok/s, below the current depth-3
  comparison row, so depth 1 is not a current fix for the 1.7B vLLM gap.
- Caching the lowered SDK WIT sampler inside `Generator` did not improve the
  tight 0.6B row. It measured 27,573 output tok/s versus 27,611 for the
  page-cadence baseline, with e2e fanout rising from about 1.49 ms to about
  1.56 ms per scheduler batch, so the change was reverted.
- Retrying opt-in staged-hit aggregation after the SDK page-cadence change was
  worse than both the old hit-aggregation row and the current baseline.
  `PIE_SPEC_HIT_AGG_MAX_EXTRA=1` measured 26,279 output tok/s on 0.6B, with
  e2e queue/fanout both rising, so hit aggregation remains off.
- Retrying the old `PIE_ARGMAX_PARTS=2` candidate together with SDK
  page-cadence bids also regressed the 0.6B row, measuring 27,502 output
  tok/s. Keep the default single-block argmax for this shape.
- Deferring direct request drops to Tokio's blocking pool reduced the reported
  scheduler fanout bucket from about 1.7 ms/batch to about 0.55 ms/batch, but
  regressed real throughput to 27,479 output tok/s
  (`/root/e2e_bench/experiments/pie_dense_spec/pie_06b_defer_direct_drop_512x128.json`).
  The background deallocation work appears to contend with the benchmark; the
  opt-in code was removed.
- Replacing the cold-submit `ForwardRequest` clone with a compact speculator
  chain-state seed regressed 0.6B to 27,243 output tok/s
  (`/root/e2e_bench/experiments/pie_dense_spec/pie_06b_chainstate_clone_elision_512x128.json`).
  The clone removal disturbed chain timing more than it saved host work, so it
  was reverted.
- A vectorized bf16x2 greedy argmax scan regressed 0.6B to 27,302 output tok/s
  (`/root/e2e_bench/experiments/pie_dense_spec/pie_06b_argmax_bf162_512x128.json`).
  Keep the scalar argmax kernel; the bottleneck is not simply scalar bf16 load
  count.
- A compact internal staged-decode request representation, avoiding the full
  one-token `ForwardRequest` wire shape for speculator stages, regressed 0.6B
  to 27,411 output tok/s
  (`/root/e2e_bench/experiments/pie_dense_spec/pie_06b_compact_staged_decode_512x128.json`).
  It was reverted; reducing Vec churn alone hurt staging cadence enough to lose
  throughput.
- Intermediate dense-cohort waits around the 64x baseline did not help after
  SDK page-cadence bids. `PIE_DENSE_COHORT_WAIT_MULT=96` measured 27,586
  output tok/s, and `=48` measured 27,477 output tok/s on 0.6B. Keep 64 for
  the TP1 unlimited row.
- Changing FlashInfer prefill dispatch to match vLLM's tensor-core decode
  `USE_FP16_QK_REDUCTION=false` was not a cross-model win. The all-false
  build improved the normal 0.6B HND/dedicated-decode row to 27,955 output
  tok/s and made forced prefill-decode less bad at 26,667 output tok/s, but
  it regressed 1.7B NHD to 16,530 output tok/s. A layout-conditional build
  then regressed 0.6B badly, measuring 26,369 and 25,990 output tok/s on two
  runs with much higher e2e fanout/queue. The prefill template flag was
  restored to the original single `true` instantiation.
- `--batch-policy eager` on the improved 1.7B 6144/depth1 shape regressed to
  16,745 output tok/s and fragmented the batch histogram. Keep adaptive
  scheduling for the TP1 unlimited rows.
- Exact-match caching for persistent control-buffer uploads regressed the
  0.6B HND/depth3 TP1 unlimited row and was reverted. The candidate skipped
  H2D copies for unchanged `qo_indptr`, `kv_page_indices`, and
  `kv_page_indptr`, but measured only 26,755 output tok/s. E2E batch build
  rose to about 398 us and response fanout to about 2.46 ms, so the CPU-side
  compare/cache bookkeeping outweighed the smaller driver-forward time.
- Reusing the prior FlashInfer non-split decode plan across fires was also
  rejected. The candidate was narrowly gated to the TP1 shape class where Pie
  already forces split-KV off, but the default-on run measured only 27,462
  output tok/s on 0.6B. A same-build kill-switch run
  (`PIE_CUDA_REUSE_NONSPLIT_DECODE_PLAN=0`) measured 27,807 output tok/s, so
  skipping the per-fire plan refresh is not safe/beneficial even when the
  schedule appears shape-only. The patch was reverted.
- Refreshing SDK generation bids every two KV pages instead of every page
  regressed the 0.6B HND/depth3 TP1 unlimited row to 27,165 output tok/s.
  This confirms the old 512-token bid-window failure was not just too large;
  the scheduler depends on page-cadence bid updates for this benchmark. The
  interval was restored to one page.
- 1.7B workspace/depth follow-up did not beat the 6144-token/depth-1 row:
  `PIE_CUDA_PREFILL_TOKENS=4096` with depth 1 measured 16,869 output tok/s,
  `8192` with depth 1 measured 16,628 output tok/s, and `6144` with
  speculation depth 0 fell to 15,251 output tok/s. Keep the current
  6144/depth-1 setting for this model-specific row.
- Forcing the vLLM-like NHD prefill-decode path on 0.6B did not help.
  `PIE_CUDA_PREFILL_DECODE_PLAN=1` with min KV pages forced to zero and
  default NHD layout measured 26,368 output tok/s, below the HND dedicated
  decode row. The dedicated FlashInfer decode path remains better for 0.6B.
- `PIE_ARGMAX_PARTS=8` regressed the 0.6B HND/depth3 row to 27,227 output
  tok/s. The extra partition/select pass is too expensive; the previous
  parts-2/parts-4 failures were not just the wrong partition count.
- Increasing the single-GPU greedy argmax block size from 256 to 512 threads
  regressed 0.6B to 26,547 output tok/s. Keep the 256-thread one-block
  argmax kernel.
- Vectorizing the 256-thread argmax kernel to load `bf16x2` pairs also
  regressed 0.6B, measuring 25,761 output tok/s. The scalar bf16 load loop
  was restored.
- Opting 0.6B into the page16 XQA GQA2 decode path with default NHD layout
  regressed badly to 24,323 output tok/s. This matches the earlier 1.7B
  page16-XQA failure; keep `PIE_CUDA_XQA_GQA2_P16` off for TP1 unlimited.
- 0.6B HND/depth3 workspace tuning around the default 20,480-token arena did
  not help. `PIE_CUDA_PREFILL_TOKENS=24576` measured 26,783 output tok/s and
  `=16384` measured 26,473 output tok/s. Keep 20,480 for the 0.6B row.
- Sparse per-request indptrs for pre-staged decode requests were rejected and
  reverted. The hypothesis was that scheduler-owned single-token requests do
  not need populated `kv_page_indptr`, `qo_indptr`, mask/logit/sampling/sampler
  indptrs, or `spec_indptr` because the batch builder reconstructs them. In
  practice the 0.6B shared-page-span row regressed to 26,473 output tok/s, so
  staged requests keep canonical single-request indptrs.
- Retesting 0.6B speculation depth 4 after the shared-page-span change still
  regressed badly, measuring 26,329 output tok/s. Depth 3 remains the 0.6B
  setting.
- Retesting the 0.6B dense cohort wait at `PIE_DENSE_COHORT_WAIT_MULT=32` after
  the shared-page-span change also regressed, measuring 26,207 output tok/s.
  Keep wait multiplier 64 for this TP1 unlimited row.
- A clean rebuild after reverting the hot-path clone-elision experiments ruled
  out stale native objects as the cause of the later 0.6B variance. The clean
  row measured 27,483 output tok/s
  (`pie_06b_revert_clone_elision_clean_512x128.json`), still below the vLLM
  28,141 output tok/s baseline.
- Current vLLM could not be rerun on this host: the only found vLLM env
  (`/root/Workspace/.local-vllm-venv`) fails at import/runtime with the current
  NVIDIA driver (`found version 12040`) being too old for that PyTorch/vLLM
  wheel. Existing vLLM artifacts remain the comparison point until the env is
  fixed.
- Nsight Systems works when invoked through the real binary
  `/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/nsys`; using bare
  `nsys` or `env` without an absolute path can fail with `Not executable`.
  Auto-import still fails, so keep using manual `QdstrmImporter` plus
  `nsys export -t sqlite`. The clean 0.6B capture
  (`nsys_06b_clean_revert.sqlite`) again showed CUDA API launch/copy overhead
  is not the main gap: graph launch total was about 8.7 ms, async memcpy about
  12.6 ms, while `cudaStreamSynchronize`/GPU wait dominated. The user-visible
  regression is mostly runtime cadence/host jitter, not missing Nsight usage.
- Opt-in API-level e2e timers were added behind
  `PIE_API_E2E_BREAKDOWN=1` and surfaced in `model_status` / `pie_bench.py`.
  In `pie_06b_api_sched_breakdown_512x128.json`, measured `execute()` averaged
  about 17.3 ms: staged-hit wait about 16.4 ms, `try_hit` about 11 us, cold
  submit averaged about 0.89 ms across all executes, append/unpin/output build
  were sub-microsecond average. This confirms the hot loop is mostly waiting
  for staged GPU output; request admission and WIT output construction are not
  the large 0.6B gap.
- `PIE_SCHED_FANOUT_BREAKDOWN=1` remains useful as profiling-only
  instrumentation. The latest breakdown again attributed only about 0.24
  ms/batch to direct oneshot sends; token output construction and response
  classification are near-zero. The rest of the fanout bucket is drop/destructor
  work and host scheduling noise.
- Replacing the staged-batch map mutex with `parking_lot::Mutex` regressed
  0.6B to 25,370 output tok/s. Despite `try_hit` averaging about 11 us under
  API instrumentation, this lock swap disturbed chain/runtime cadence and was
  reverted.
- Making the singleton linker spawn each WASM instantiation concurrently
  reduced reported per-process instantiate time from roughly 80-95 ms to about
  4.4 ms, but throughput regressed or stayed below baseline: 27,403 output
  tok/s on the first run and 26,130 output tok/s on rerun. The saved startup
  time shifted into worse WASM/GPU overlap and longer driver-forward time, so
  the linker remained sequential for this row.
- `PIE_SPEC_SEND_BEFORE_EXTEND=1`, which published the next staged receiver and
  returned the current token before submitting the next speculative stage,
  regressed to 26,645 output tok/s. The original order, submit/queue next stage
  before releasing the current token, preserves better chain cadence.
- Bounded latency IPC spinning did not help. Keeping `ipc_profile=latency` but
  setting `--spin-budget-us 1000` measured 27,292 output tok/s, below the
  default unbounded latency-spin row.
- Tokio worker-thread tuning did not help the 0.6B row. `--worker-threads 64`
  measured 27,288 output tok/s and `--worker-threads 16` measured 27,523 output
  tok/s, both below the best current rows. Keep the runtime default.
- 0.6B speculation depth 1 was tested for the current HND/depth3 shape and
  measured 27,027 output tok/s. Depth 3 remains the best setting; depth 1/2/4
  are now all rejected for this row.
- Extra warmup did not explain the 0.6B variance. Raising warmup from 4 to 16
  measured 27,443 output tok/s, still below the vLLM baseline and below Pie's
  best observed rows.
- `--auto-token-budget` regressed the 0.6B row to 26,855 output tok/s. Smaller
  per-process KV reservation bookkeeping did not improve this throughput shape.
- Additional cuBLASLt BF16 heuristic overrides were rejected. Algo index 4
  measured 25,571 output tok/s and index 5 measured 26,660 output tok/s. Keep
  the current shape-aware default, which selects index 2 for the 0.6B lm-head
  shape.
- Benchmark-script audit for the TP1 unlimited comparison found no fixed
  concurrency cap in the active Pie/vLLM throughput rows: Pie maps
  `--concurrency 0` to no client-side semaphore, and vLLM maps it to
  `max_num_seqs=num_requests` for the row. Prompt/output accounting also
  matched the saved artifacts at 18,330 prompt tokens and 65,536 output
  tokens. A harness-only bug where Pie throughput-mode latencies were all set
  to the full bulk wall time was fixed; it did not affect throughput.
- A fresh current-source 0.6B HND/depth3 TP1 unlimited rerun after the
  benchmark cleanup measured 27,187 output tok/s
  (`pie_06b_current_fresh_512x128.json`), below the vLLM 28,141 output tok/s
  baseline. The run completed 131 scheduler batches with average driver
  forward about 14.98 ms, response fanout about 1.74 ms, and batch queue about
  4.40 ms.
- `PIE_DENSE_COHORT_SLACK=8` regressed the 0.6B row to 27,037 output tok/s
  (`pie_06b_dense_slack8_512x128.json`). Letting dense resident cohorts fire
  short of the full target increased batch count and queue jitter, so the
  default slack remains zero.
- Opt-in Wasmtime component pre-instantiation caching
  (`PIE_LINKER_PREINSTANTIATE=1`) reduced average process instantiate time
  from about 87 ms to about 82 ms but did not improve throughput: the 0.6B
  row measured 27,161 output tok/s
  (`pie_06b_linker_pre_512x128.json`). The cache remains off by default.
- Additional API/fanout instrumentation
  (`pie_06b_api_fanout_breakdown_512x128.json`) confirmed the hot request path
  is still dominated by waiting for staged GPU output. Average API `execute()`
  was about 17.6 ms, with staged await about 16.9 ms, submit await about
  0.71 ms, `try_hit` about 16 us, and output construction/pin/unpin
  sub-microsecond to low-microsecond. Scheduler fanout rose under
  instrumentation, so treat that row as profiling-only.
- Capturing the single-GPU greedy argmax inside the decode CUDA graph
  (`PIE_CUDA_GRAPH_ARGMAX=1`) regressed the 0.6B row to 26,793 output tok/s
  (`pie_06b_graph_argmax_512x128.json`) and raised average driver forward to
  about 15.39 ms. Keep argmax outside the graph for this shape.

### Pending

- Remaining 0.6B TP1 work should target steady runtime cadence rather than
  blind GPU knob swaps. The best current evidence is: driver forward is usually
  14.8-15.2 ms/batch, graph launch/copy overhead is small, response fanout is
  noisy but mostly destructor/drop work, and API execution is dominated by
  staged-output wait. The reliable win likely needs reducing host scheduling /
  response-drop jitter or a real model-body/logits kernel improvement, not more
  prefill-decode/XQA/page-size retests already listed above.

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
- Inspecting vLLM 0.21's Hopper attention backend explains the remaining
  large-model gap more concretely:
  - `vllm/v1/attention/backends/flash_attn.py` builds paged metadata with a
    dense `block_table`, per-request `seq_lens`, and optional FA3 scheduler
    metadata from `get_scheduler_metadata`.
  - `vllm/vllm_flash_attn/flash_attn_interface.py` then calls
    `torch.ops._vllm_fa3_C.fwd(q, k_cache, v_cache, ..., seqused_k,
    block_table, scheduler_metadata, num_splits, ...)`.
  - This supports GQA generally because FA3 accepts Q heads and KV heads
    directly; Pie's FlashInfer decode kernel only supports GQA groups
    `{1,2,3,4,8}`, so Qwen's GQA=5 decode is forced through Pie's FlashInfer
    prefill fallback.
The next real optimization is therefore not another planner heuristic. It is
either a native FA3 paged-attention integration for Pie's KV layout, or a
deliberate KV-layout adapter plus graph-safe metadata cache for the vLLM/FA3
ABI. The previous generic XQA attempt was not that path and was slower.

## 2026-05-19 XQA/FA3 Follow-Up

- The profiler/cache path was rechecked on H200 Qwen3-32B, 256 requests x 512
  output tokens. With the current rule selector (`N=8192`, `R=256`, page=32),
  the profiler did not find a better memory layout:
  - rule: 6173.68 output tok/s in the best clean sweep run.
  - profiled `8192/512`: 6156-6169 output tok/s.
  - profiled `4096/256` and `4096/512`: about 6151-6152 output tok/s.
  The right behavior here is to avoid installing a profile when measurement
  says the rule plan is already best.
- Forcing Hopper FA3 prefill-decode ahead of XQA for pure decode was tested and
  rejected on the same H200 32B row. It fell to 5910.80 output tok/s. The
  conclusion is narrower than "FA3 is bad": Pie's current FlashInfer Hopper
  prefill wrapper is useful for real prefill/mixed batches, but it is not a
  faster replacement for the GQA=8 XQA decode path on this shape.
- XQA metadata construction was hoisted from every layer's attention call to
  once per forward fire. This removes repeated page-table/sequence-length
  metadata kernels and reduced H200 32B decode graph memory from about 124 MiB
  to about 118 MiB. Throughput was noise-neutral on the 32B row, but the code
  now better matches the real lifetime of that metadata.
- Fresh H200 Qwen3-32B 256x512 comparison after these changes:
  - standalone vLLM 0.21: 6173.99 output tok/s.
  - Pie best clean sweep run: 6173.68 output tok/s.
  - Pie individual reruns varied from about 6102 to 6174 output tok/s.
  This should be treated as parity, not a robust Pie win. The earlier 6194
  output tok/s Pie run was not reproduced consistently.
- Local L40 TP=2 Qwen3-8B 512x512 remains a clear Pie win:
  - Pie: 7455.62 output tok/s.
  - standalone vLLM 0.20.2, used because the local driver cannot run the vLLM
    0.21 torch/CUDA wheel: 7298.62 output tok/s.
- MLA primitives do exist in FlashInfer (`paged_kv_mla_t`, batch MLA plan/run
  files, SM90 and SM120 MLA kernels), but Pie does not yet have an MLA model
  path or MLA KV-cache layout. Wiring those kernels would be model support work,
  not a small attention dispatch swap. No fake MLA integration was kept.

## 2026-05-22 0.6B TP1 Unlimited Debugging

- The benchmark scripts were audited for the "unlimited concurrency" rows.
  Pie `tput --concurrency 0` maps to no process-admission semaphore, and vLLM
  `tput --concurrency 0` maps `max_num_seqs` to the request count. A harness
  artifact that assigned the full wall time as every bulk request latency was
  fixed for reporting only; throughput accounting was already based on wall
  time and token counts.
- Current vLLM comparison artifact:
  `vllm_06b_tp1_unlimited_512_flashinfer_current_rerun.json`, 28,141 output
  tok/s. Current Pie clean reruns clustered around 27,187-27,365 output tok/s.
  The remaining wall gap is roughly 60-70 ms on a 512x128 row, not a fixed
  concurrency or prompt/output accounting issue.
- E2E breakdown for Pie shows request admission and WIT output construction are
  not the bottleneck. API execute is dominated by the staged GPU wait; scheduler
  fanout is roughly 1.7-1.9 ms/batch and driver forward is roughly 14.8-15.1
  ms/batch. The best old Pie run was not materially faster in driver forward;
  it mostly had lower fanout/queue timing.
- `ipc_profile=latency` was confirmed to use the in-process polling channel with
  effectively unlimited spin budget by default.
- `PIE_DENSE_COHORT_SLACK=8` was rejected on 0.6B TP1 unlimited:
  `pie_06b_dense_slack8_512x128.json` measured 27,037 output tok/s, with higher
  queueing and no throughput gain.
- Wasmtime component linker pre-instantiation was rejected as a default path.
  `PIE_LINKER_PREINSTANTIATE=1` reduced average instantiation time but measured
  27,161 output tok/s, so startup got slightly cheaper without improving the
  measured throughput row.
- Scheduler/API fanout instrumentation confirmed that direct WIT output build is
  sub-microsecond per request and not the gap. The instrumentation row itself
  measured 26,718 output tok/s due to overhead, so it should not be used as a
  performance baseline.
- Capturing the single-GPU greedy argmax in the CUDA graph was rejected:
  `PIE_CUDA_GRAPH_ARGMAX=1` measured 26,793 output tok/s and increased driver
  forward time.
- Publishing context working-token cache only at page boundaries was rejected.
  `PIE_CONTEXT_APPEND_PUBLISH_PAGE_ONLY=1` measured 27,315 output tok/s. It
  lowered the fanout counter but increased driver-forward/total time, so the
  hot-path branch was removed.
- Forcing a cached memory profile with R=256 was rejected:
  `pie_06b_profile_r256_512x128.json` measured 26,105 output tok/s and split the
  row into 389 total batches. An attempted R=384 cache did not actually select
  R=384 because that shape is not in the planner candidate list; it fell back to
  the rule-selected R=512 and should not be interpreted as an R=384 result.
- Nsight Systems was used through the real binary path
  `/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/nsys`. Pie's clean
  0.6B trace is dominated by FlashInfer decode kernels plus GEMMs; host-visible
  CUDA time is mostly stream synchronization, not launch overhead. This matches
  the failed argmax-graph result: the remaining gap is not just one extra CUDA
  launch.
- vLLM's current FlashInfer decode wrapper uses
  `BatchDecodeWithPagedKVCacheWrapper(..., use_tensor_cores=True)`, which routes
  decode planning through FlashInfer's BatchPrefill module with `causal=False`,
  split-KV enabled, and `use_fp16_qk_reduction=False`. Pie's C++ prefill-decode
  path already matched most of this shape, but the QK reduction mode remained a
  concrete API mismatch worth testing as an opt-in rather than a default.
- Retesting that mismatch as an opt-in on the current code was rejected.
  `PIE_FLASHINFER_PREFILL_FP16_QK=0` measured 27,102 output tok/s on the 0.6B
  TP1 unlimited row, below the same-build default at 27,303 output tok/s. The
  extra template branch was removed after the test to avoid keeping the compile
  and binary-size cost.
- Moving request/page drops before direct response sends was rejected as a
  scheduler fanout optimization. It measured 26,311 output tok/s on 0.6B TP1
  unlimited, far below the same-build default. The send-before-drop ordering is
  kept despite the fanout counter because it gives better end-to-end cadence.

## 2026-05-22 Local L40 TP1 Continuation

- The old Pie regression on the tight 0.6B row was mainly host/runtime cadence,
  not a single missing GPU kernel. Current-source reruns had driver-forward
  times close to the old best, but response fanout and queue timing were worse.
- A correctness fix widened the CUDA graph layout key from 8 bits to 32 bits.
  Prefill-decode graph keys now include the actual FlashInfer launch layout
  class, padded batch size, HND/NHD, split-KV, causal/full variant, and SM90
  marker. This prevents graph replay across incompatible FlashInfer plan
  offsets/topologies.
- The unsafe no-split prefill-decode graph experiment was removed. FlashInfer
  only materializes `block_valid_mask` for split-KV prefill plans, so non-split
  graph replay with padded CTAs can read invalid request indices.
- Restored the direct token-only response fast path. The normal non-breakdown
  path now uses the direct `oneshot` send bypass instead of routing direct
  completions through `PendingRequest::send_result`.
- Added default-on all-direct pre-extraction in the scheduler. For batches where
  every completion is direct, the scheduler extracts response senders and drops
  heavy request/page payloads on Tokio's blocking pool while the GPU forward is
  in flight. Chunked continuations keep the old path. Set
  `PIE_SCHED_PREEXTRACT_DIRECT=0` to disable. A two-phase variant that sent the
  response targets back before dropping payloads regressed badly and was
  reverted (`pie_06b_preextract_twophase_512x128.json`, 26,491 output tok/s).
- Benchmark harness CPU locality was a real measurement issue. Both Pie and
  vLLM throughput scripts now default to `--cpu-affinity auto`, which pins the
  benchmark process and child engine/server to the GPU-local CPU mask reported
  by `nvidia-smi topo -m`; `--cpu-affinity none` disables it. This does not
  impose any request concurrency cap. On local GPU0 the mask is
  `80-87,208-215`.
- With equal auto-affinity, 0.6B TP1 unlimited now beats vLLM by a large margin:
  - Pie default pre-extract:
    `pie_06b_default_preextract_autoaffinity_rerun_512x128.json`,
    29,764.59 output tok/s.
  - vLLM FlashInfer:
    `vllm_06b_tp1_unlimited_512_flashinfer_autoaffinity_20260522.json`,
    28,123.34 output tok/s.
- Manual `taskset` confirmed the same locality effect before it was added to
  the scripts: Pie 0.6B reached 29,633.50 output tok/s on `80-87,208-215`;
  vLLM with the same mask measured 28,223.66 output tok/s.
- 1.7B TP1 unlimited improved substantially but is not a robust win on GPU0
  yet. Best GPU0 Pie row so far is
  `pie_17b_preextract_autoaffinity_prefill8192_512x128.json` at 17,688.17
  output tok/s, versus vLLM FlashInfer
  `vllm_17b_tp1_unlimited_512_flashinfer_autoaffinity_20260522.json` at
  17,730.47 output tok/s. GPU1 produced a near-tie/slight Pie win once
  (`pie_17b_preextract_autoaffinity_gpu1_prefill8192_512x128.json`,
  17,745.21 vs
  `vllm_17b_tp1_unlimited_512_flashinfer_autoaffinity_gpu1_20260522.json`,
  17,743.53), but a rerun fell to 17,421.29, so this is not accepted as a
  stable win.
- Rejected 1.7B attempts under the pinned harness:
  - Prefill workspace sweep: 6144, 7168, 7680, 8192, 9216, 10240, 12288, and
    20480 tokens. 8192 remains the best observed shape.
  - Speculation depths 0, 2, and 3; depth 1 remains best for 1.7B.
  - `PIE_CUDA_PREFILL_DECODE_PLAN=1` with internal FlashInfer graph mode and
    min-KV-pages 0 regressed to 17,159 output tok/s.
  - `PIE_CUDA_DECODE_FULL_ATTENTION=0` regressed to 17,461 output tok/s.
  - NHD layout, XQA GQA2 page16, and XQA page32 all regressed.
  - `PIE_CUDA_GRAPH_ARGMAX=1` regressed to 17,477 output tok/s.
  - Dense wait multipliers 16/32/48 and eager/greedy policies regressed.
- Current next target for 1.7B is GPU-side forward time. Under auto-affinity,
  e2e response fanout is already about 80-90 us/batch; the vLLM gap tracks
  driver forward, not request admission or response fanout.
- The benchmark-script audit for the refreshed 1.7B rows again found no fixed
  concurrency cap: Pie uses `--concurrency 0` with no client semaphore, and
  vLLM maps `--concurrency 0` to `max_num_seqs=num_requests`. The current
  comparison rows all use unlimited concurrency and Pie uses
  `--ipc-profile latency`.
- Capturing graph-safe single-GPU argmax became useful after the later 1.7B
  GPU-side changes, but only with `PIE_ARGMAX_PARTS=2`. The parts-2 row with
  static decode planning, lm-head-only Lt, decode-full disabled, and an
  8192-token workspace reached 17,756.55 output tok/s before the qkv
  postprocess optimization. `PIE_ARGMAX_PARTS=1` was retested after qkv
  postprocess and regressed to 17,700.81 output tok/s
  (`pie_17b_qkvpost_warp_argmaxparts1_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
- A warp-level q/k RMSNorm+RoPE+KV-write decode postprocess is kept. Nsight on
  `pie_17b_qkvpost_warp_nsys_node_512x128.json` showed the qkv decode
  postprocess falling from roughly 60 ms to about 30 ms over the traced row,
  and the best untraced run reached 17,968.91 output tok/s
  (`pie_17b_qkvpost_warp_graph_argmax_parts2_staticdecode_lmheadlt_decodefull0_autoaffinity_prefill8192_512x128.json`).
  The win is not yet stable or large: clean reruns ranged from 17,648.59 to
  17,898.26 output tok/s, roughly tied with the refreshed vLLM row at
  17,898.96 output tok/s.
- Reducing the fallback qkv postprocess CTA block from 128 to 64 did not help
  the 1.7B row. It measured 17,613.75 output tok/s and was reverted.
- Forcing the decode-as-prefill path to run non-split, to mimic the visible
  vLLM eager grid shape, was rejected. The run saturated the GPU for minutes
  and was killed; the source was restored to split only when FlashInfer's
  head-dim support requires it. Do not retry this without a deeper FlashInfer
  plan/layout change.
- A row-wise/vectorized SwiGLU candidate regressed 1.7B to 17,670.96 output
  tok/s (`pie_17b_swiglu_vec2row_qkvpost_warp_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`)
  and was reverted to the original chunked kernel.
- Nsight after the qkv postprocess optimization shows the duplicate sampler is
  gone from the captured decode graph path: `sample_temp` totals about 0.6 ms
  over a few warmup calls, while the hot path is now FlashInfer decode
  (~1.17 s traced), GEMMs, SwiGLU (~90 ms), RMSNorm (~67 ms), and captured
  argmax (~37 ms). The next viable optimization target is still GPU-side
  forward time, not request admission.
- TP1 residual-add/RMSNorm fusion was rejected for the 1.7B row. The candidate
  routed o/down projection GEMMs through a scratch buffer with `beta=0` and
  reused Pie's existing fused residual-add/RMSNorm kernel for the following
  MLP/next-layer norm. It measured 17,876.33 output tok/s
  (`pie_17b_tp1_fused_resid_rms_qkvpost_warp_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  below the best qkv-postprocess rows, so the source change was reverted.
- Increasing `chunked_swiglu_bf16` from 128 to 256 threads per CTA was
  rejected. The 1.7B row measured 17,839.81 output tok/s
  (`pie_17b_swiglu_block256_qkvpost_warp_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  worse than the kept 128-thread chunked kernel, so the block size was
  restored.
- Reducing the warp-level qkv decode postprocess launcher from 8 warps/CTA to
  4 warps/CTA was rejected. The 1.7B row measured 17,649.73 output tok/s
  (`pie_17b_qkvpost_warpblock128_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  well below the kept 8-warps/CTA shape.
- Increasing the same qkv postprocess launcher from 8 warps/CTA to
  16 warps/CTA was also rejected. It measured 17,716.11 output tok/s
  (`pie_17b_qkvpost_warpblock512_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  Keep the original 8-warps/CTA warp-level postprocess launch.
- `PIE_CUBLASLT_BF16_ALGO_INDEX=1` was rejected for the 1.7B lm-head-only Lt
  path. It measured 17,667.77 output tok/s
  (`pie_17b_lmheadlt_algo1_qkvpost_warp_graphargmax_parts2_staticdecode_decodefull0_prefill8192_512x128.json`).
  The previous sweep already rejected indices 0 and 2-7, so keep the default
  heuristic index for this row.
- Prefill-backed decode with a fixed FlashInfer split size of 8 pages was
  rejected. The candidate added an env-gated
  `PIE_FLASHINFER_PREFILL_FIXED_SPLIT_SIZE=8` and ran with
  `PIE_CUDA_PREFILL_DECODE_PLAN=1`,
  `PIE_CUDA_PREFILL_DECODE_MIN_KV_PAGES=0`, and
  `PIE_CUDA_PREFILL_DECODE_FULL_MIN_KV_PAGES=0`. It saturated the GPU for
  more than 90 seconds on the 1.7B 512x128 row, versus about 4 seconds for the
  dedicated-decode baseline, and was killed before producing a JSON artifact.
  The source change was reverted; fixed split does not rescue Pie's
  prefill-decode path.
- Retesting 1.7B speculation depth 2 after the qkv postprocess optimization
  still regressed. It measured 17,618.01 output tok/s
  (`pie_17b_depth2_qkvpost_warp_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  Keep speculation depth 1 for the 1.7B TP1 unlimited row.
- Retesting default cuBLASLt coverage after the qkv postprocess optimization
  still regressed. Removing `PIE_CUBLASLT_BF16_MIN_N=100000` measured
  17,414.94 output tok/s
  (`pie_17b_defaultlt_qkvpost_warp_graphargmax_parts2_staticdecode_decodefull0_prefill8192_512x128.json`).
  Keep lm-head-only Lt for this row.
- Simplifying the fused qkv decode postprocess K/V destination to the current
  decode token's last page/last offset is kept for now. The first run measured
  17,707.95 output tok/s, but the confirmation run reached 18,015.20 output
  tok/s
  (`pie_17b_qkvpost_directkvslot_rerun2_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  slightly above the refreshed vLLM FlashInfer baseline at 17,898.96 output
  tok/s. This is not a large enough margin by itself.
- Retesting the prefill-backed decode route on the current qkv/direct-slot
  build, without the previously rejected fixed split-size knob, still
  regressed. With `PIE_CUDA_PREFILL_DECODE_PLAN=1`,
  `PIE_CUDA_PREFILL_DECODE_MIN_KV_PAGES=0`, and
  `PIE_CUDA_PREFILL_DECODE_FULL_MIN_KV_PAGES=0`, the 1.7B row measured
  17,729.10 output tok/s
  (`pie_17b_prefilldecode_current_qkvdirect_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  Dedicated FlashInfer decode remains the accepted path unless a deeper API
  mismatch is fixed.
- Matching vLLM's tensor-core decode QK reduction flag on the prefill-backed
  route was also rejected for 1.7B. A narrow opt-in
  `PIE_FLASHINFER_PREFILL_FP16_QK=0` improved that rejected path to 17,794.67
  output tok/s
  (`pie_17b_prefilldecode_qkfp32_qkvdirect_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  but it remained below both the kept dedicated-decode Pie row and vLLM. The
  extra FlashInfer template specialization was removed again to avoid carrying
  compile and binary-size cost for a non-winning path.
- A vec8 BF16 plain-RMSNorm kernel, modeled after vLLM's vectorized
  load/store shape for hidden sizes divisible by 8, was rejected. The 1.7B row
  measured 17,818.39 output tok/s
  (`pie_17b_rmsnorm_vec8_qkvdirect_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  below the kept qkv/direct-slot dedicated-decode row. The source change was
  reverted.
- Attempting to stabilize the noisy 1.7B rows by locking GPU0 to its max
  supported application clocks was blocked by the environment. `nvidia-smi -i 0
  -lgc 2490,2490` reported that the current user does not have permission to
  change clocks. The comparison remains under the default boost behavior.
- Current-source qkv/direct-slot reruns after reverting rejected experiments
  did not reproduce the 18,015 output tok/s outlier. Confirmation rows measured
  17,691.70 and 17,746.07 output tok/s
  (`pie_17b_qkvdirect_rebuild_confirm_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`,
  `pie_17b_qkvdirect_rebuild_confirm_rerun2_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  A same-period vLLM FlashInfer rerun measured 17,754.75 output tok/s
  (`vllm_17b_tp1_unlimited_512_flashinfer_autoaffinity_current_afterpie_20260522.json`),
  so the current median is still a tie/slight Pie loss, not an accepted win.
- Retesting `PIE_CUDA_DECODE_FULL_ATTENTION=1` on the current qkv/direct-slot
  build was neutral and still below the target, at 17,750.13 output tok/s
  (`pie_17b_qkvdirect_decodefull1_retest_graphargmax_parts2_staticdecode_lmheadlt_prefill8192_512x128.json`).
  Keep `PIE_CUDA_DECODE_FULL_ATTENTION=0` for the accepted command.
- Runtime-side retries did not help the current 1.7B row. `--wasm-warm-slots
  512` regressed to 17,548.82 output tok/s
  (`pie_17b_wasmwarmslots512_qkvdirect_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  removing `PIE_SPEC_STRICT_DEPTH=1` measured 17,742.87 output tok/s
  (`pie_17b_nostrictdepth_qkvdirect_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  and `PIE_SPEC_HIT_AGG_MAX_EXTRA=1` measured 17,677.94 output tok/s
  (`pie_17b_hitagg1_qkvdirect_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  Keep the strict-depth/default hit path and no explicit WASM warm-slot
  override.
- Retesting TP1 residual-add/RMSNorm fusion as an opt-in combined with the
  current qkv/direct-slot path regressed to 17,644.85 output tok/s
  (`pie_17b_tp1_residrms_optin_qkvdirect_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  This is worse than the earlier residual-fusion attempt and below the current
  vLLM rerun, so the opt-in branch was removed again.
- The direct K/V slot simplification was reverted after it failed to reproduce
  reliably. Restoring the general page/offset calculation in the qkv decode
  postprocess recovered the no-direct current row to 17,905.01 output tok/s
  (`pie_17b_qkvpost_warp_revert_directkv_current_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  Treat the direct-slot shortcut as unstable unless a correctness proof and
  stronger row-specific evidence are added.
- An invalid argmax-parts comparison was launched with parts=3 and parts=4 in
  parallel on the same GPU and was killed. Do not use artifacts from that run.
  Re-running serially rejected both variants: parts=3 measured 17,888.36
  output tok/s and parts=4 measured 17,874.74 output tok/s
  (`pie_17b_qkvpost_warp_revert_directkv_argmaxparts3_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`,
  `pie_17b_qkvpost_warp_revert_directkv_argmaxparts4_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  Keep `PIE_ARGMAX_PARTS=2`.
- Node-level Nsight on the restored no-direct Pie row showed the remaining GPU
  time is mostly GEMM and attention, not request admission. The grouped kernel
  totals were roughly GEMM 2747.8 ms (65.5%), FlashInfer decode 1188.7 ms
  (28.4%), SwiGLU 89.5 ms, RMSNorm 53.7 ms, argmax 36.8 ms, warp qkv
  postprocess 29.7 ms, and prefill/misc qkv postprocess 25.2 ms over the node
  trace. For vLLM, node-level tracing required
  `--trace-fork-before-exec=true`; otherwise the child engine kernels are
  mostly missed.
- HND KV layout was useful but noisy on 1.7B. HND with dedicated decode measured
  17,973.80 output tok/s and confirmed at 17,929.08 output tok/s
  (`pie_17b_hnd_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`,
  `pie_17b_hnd_qkvpost_warp_revert_directkv_confirm_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  Enabling `PIE_CUDA_DECODE_FULL_ATTENTION=1` on HND improved the accepted row
  to 18,031.87 output tok/s
  (`pie_17b_hnd_decodefull1_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_lmheadlt_prefill8192_512x128.json`).
- XQA-style FlashInfer routes did not beat the kept HND dedicated-decode row.
  HND plus `PIE_CUDA_XQA_DECODE=1` printed `xqa_decode=off` and measured
  18,035.07 output tok/s, so it is only an HND/no-XQA noisy row. NHD with
  `PIE_CUDA_XQA_DECODE=1` but without the GQA2 page16 gate also printed
  `xqa_decode=off` and measured 17,607.15 output tok/s. Explicit GQA2 page16
  XQA (`PIE_CUDA_XQA_DECODE=1 PIE_CUDA_XQA_GQA2_P16=1`) printed
  `xqa_decode=on` but regressed to 17,214.19 output tok/s
  (`pie_17b_nhd_xqa_gqa2p16_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`).
  Do not retry this exact XQA path without changing the FlashInfer API mapping.
- HND prefill-backed decode was rejected for the current build. With
  `PIE_CUDA_PREFILL_DECODE_PLAN=1`, it measured 17,790.72 output tok/s
  (`pie_17b_hnd_prefilldecode_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_lmheadlt_decodefull0_prefill8192_512x128.json`),
  below HND dedicated decode.
- cuBLASLt scope remains lm-head-only for 1.7B. Disabling Lt regressed HND/full
  attention to 16,849.67 output tok/s, and the broader default Lt coverage
  regressed to 17,822.74 output tok/s
  (`pie_17b_hnd_decodefull1_nolt_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_prefill8192_512x128.json`,
  `pie_17b_hnd_decodefull1_defaultlt_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_prefill8192_512x128.json`).
  Keep `PIE_CUBLASLT_BF16_MIN_N=100000`.
- A BF16 vectorized argmax path was successful and is now default-on unless
  `PIE_ARGMAX_VEC2=0` is set. Under HND + full-attention + graph argmax +
  static decode + lm-head-only Lt + parts=2, the explicit vec2 run measured
  18,153.43 output tok/s and confirmed at 18,169.11 output tok/s
  (`pie_17b_hnd_decodefull1_argmaxvec2_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_lmheadlt_prefill8192_512x128.json`,
  `pie_17b_hnd_decodefull1_argmaxvec2_confirm_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_lmheadlt_prefill8192_512x128.json`).
  The scalar control measured 17,969.07 output tok/s. After rebuilding with
  vec2 default-on, the same command without `PIE_ARGMAX_VEC2=1` measured
  18,195.10 output tok/s
  (`pie_17b_hnd_decodefull1_argmaxvec2_defaulton_confirm_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_lmheadlt_prefill8192_512x128.json`).
- Argmax vec2 is not enough to loosen other rejected knobs. Vec2 with
  `PIE_CUDA_DECODE_FULL_ATTENTION=0` measured 17,883.05 output tok/s, and vec2
  with `PIE_ARGMAX_PARTS=1` measured 17,731.85 output tok/s
  (`pie_17b_hnd_decodefull0_argmaxvec2_qkvpost_warp_revert_directkv_graphargmax_parts2_staticdecode_lmheadlt_prefill8192_512x128.json`,
  `pie_17b_hnd_decodefull1_argmaxvec2_parts1_qkvpost_warp_revert_directkv_graphargmax_staticdecode_lmheadlt_prefill8192_512x128.json`).
  Keep full attention and parts=2.
- Same-session vLLM FlashInfer TP1 unlimited-concurrency baseline after the
  argmax vec2 work measured 17,797.37 output tok/s
  (`vllm_17b_tp1_unlimited_512_flashinfer_current_after_argmaxvec2_20260522.json`).
  Pie is now about 2.2% faster on this 1.7B row, but this is not yet a large
  margin, so the next target must come from GEMM/attention rather than runtime
  queueing.
- Node-level Nsight on the current HND + full-attention + vec2 row measured
  roughly GEMM 2750.4 ms (66.6%), FlashInfer decode 1118.8 ms (27.1%),
  SwiGLU 89.6 ms, RMSNorm 53.6 ms, argmax 35.6 ms, qkv decode postprocess
  28.9 ms, and q/k RoPE prefill 14.6 ms over the traced kernels. The profile
  artifact is `/tmp/pie17_hnd_full1_vec2_current_node_nsys.nsys-rep` and the
  benchmark JSON is `pie_17b_hnd_full1_vec2_current_node_nsys_512x128.json`.
  This keeps the next target on GEMM/attention; argmax is now too small to
  carry a large margin alone.
- Caching the static non-split FlashInfer decode-plan upload was rejected and
  reverted. The candidate skipped the host rebuild/device copy of
  request_indices, kv_tile_indices, o_indptr, and kv_chunk_size when batch
  size/page size/workspace pointer were unchanged. It was neutral on 1.7B
  (18,185.76 output tok/s,
  `pie_17b_staticplan_uploadcache_hnd_full1_vec2_512x128.json`) and did not
  recover the tight 0.6B cadence row (28,690.00 and 28,622.62 output tok/s,
  `pie_06b_staticplan_uploadcache_default_depth3_512x128.json`,
  `pie_06b_staticplan_uploadcache_default_depth3_rerun_512x128.json`). A
  0.6B scalar-argmax control with `PIE_ARGMAX_VEC2=0` was lower at 28,457.61
  output tok/s, so the 0.6B drop was not caused by the new vec2 default. After
  reverting the upload cache and rebuilding, 1.7B reran at 17,948.04 and
  18,030.67 output tok/s
  (`pie_17b_post_uploadcache_revert_hnd_full1_vec2_512x128.json`,
  `pie_17b_post_uploadcache_revert_hnd_full1_vec2_rerun_512x128.json`).

## 2026-05-22 fused dense projection replacement

- Benchmark-script audit: `--concurrency 0` remains symmetric. Pie drops
  `max_concurrent_processes` (`None`, serialized as unlimited) and vLLM maps
  the same flag to `max_num_seqs=num_requests`; current 8B vLLM JSON confirms
  `max_num_seqs=512`. Both paths use GPU-local CPU affinity
  `80-87,208-215`. Pie rows in this section use `--ipc-profile latency`.
- The 8B regression source was confirmed as duplicated fused dense projection
  weights, not fixed client concurrency. With the old duplicate-fused path, 8B
  used `used_after_weights=24,700 MiB`, reduced the forward cap to R=256, and
  measured 4,453.58 output tok/s
  (`pie_8b_hnd_full1_vec2_unlimited_current_512x128.json`). Disabling fused
  projection duplicates recovered R=512 and measured 4,829.13 / 4,791.64
  output tok/s
  (`pie_8b_nofusedproj_hnd_full1_vec2_unlimited_512x128.json`,
  `pie_8b_nofusedproj_hnd_full1_vec2_unlimited_confirm_512x128.json`).
- Accepted source fix: dense fused QKV / gate-up contracts now consume the
  original raw tensors, and the Qwen/Llama-like binder creates non-owning
  per-projection `DeviceTensor::view` aliases into the fused buffer when the
  canonical names are absent. This preserves the unfused fallback without
  keeping duplicate weights resident. The default fused-projection threshold
  is now 10 GiB: all fused through 8B-class Qwen models, QKV-only above that.
- 8B memory behavior is fixed. All-fused replacement runs at R=512 with
  `used_after_weights=16,060 MiB` and measured 4,835.54 output tok/s
  (`pie_8b_fusedreplace_hnd_full1_vec2_unlimited_512x128.json`). The final
  default-10GiB gated build measured 4,807.34 output tok/s
  (`pie_8b_default10g_fusedreplace_swigluvec2gated_hnd_full1_vec2_unlimited_512x128.json`)
  against refreshed vLLM 4,805.45
  (`vllm_8b_tp1_unlimited_512_flashinfer_current_rerun_after_fusedreplace_20260522.json`).
  This is only a narrow win, not a large margin.
- 8B rejected follow-ups on the replacement build:
  qkv-only replacement with 6 GiB budget was consistently low at 4,755.27 /
  4,733.45 output tok/s; no-fusion final control was 4,778.17; default
  cuBLASLt threshold was 4,370.23; Lt disabled was 4,703.16; forced Lt algo
  indices 0-7 were all worse (best 4,729.93); depth 2/3 were 4,745.13 /
  4,728.34; page size 32/XQA was 4,769.06; prefill N=16,384 and 32,768 were
  4,805.13 and 4,771.16; worker threads 16 was 4,742.68; disabling the static
  decode-plan path was 4,771.78. Forced prefill-decode was noisy and not
  accepted: 4,830.06 before the SwiGLU experiment, then 4,795.26 on the final
  gated build.
- Nsight for 8B confirmed the remaining Pie cost is CUDA/GEMM-heavy. The Pie
  qkv-only profile grouped roughly 14.5 s GEMM/CUTLASS (87.9%), 1.5 s
  FlashInfer attention (9.1%), and sub-2% activation/norm. The vLLM 8B fork
  trace required `/root/Workspace/vllm/.venv/bin/python` plus
  `--trace-fork-before-exec=true`; the profiled vLLM run measured 4,771.42
  output tok/s and the plain rerun measured 4,805.45. The vLLM trace is
  dominated by compiled GEMM/Triton kernels, so a large 8B margin likely
  requires better GEMM/kernel fusion rather than request admission changes.
- 14B benefits from QKV-only replacement but not gate/up fusion. Default 10 GiB
  (QKV-only) measured 2,022.08 output tok/s
  (`pie_14b_default10g_fusedreplace_hnd_decodefull0_vec2_unlimited_512x128.json`)
  versus vLLM 1,933.72
  (`vllm_14b_tp1_unlimited_512_flashinfer_current_20260522.json`). All-fused
  replacement regressed to 1,981.13, full attention was 1,999.78, and a forced
  prefill-decode attempt did not enable that path and measured 2,016.13.
- 4B benefits from replacement and the gated vec2 chunked SwiGLU helper.
  Default replacement measured 8,154.63 output tok/s; with vec2 enabled for
  intermediate widths <= 10,000, 4B measured 8,231.71 output tok/s
  (`pie_4b_default10g_fusedreplace_swigluvec2_hnd_simple_depth1_unlimited_512x128.json`)
  versus vLLM 8,074.90
  (`vllm_4b_tp1_unlimited_512_flashinfer_rerun.json`). Weight residency fell
  from the old all-fused duplicate 12,770 MiB to 8,162 MiB.
- 1.7B remains noisy but above the refreshed vLLM baseline when rerun. Default
  replacement measured 17,815.84; vec2 SwiGLU first measured 17,723.74, then
  reran at 18,159.67
  (`pie_17b_default10g_fusedreplace_swigluvec2_hnd_full1_vec2_unlimited_rerun_512x128.json`)
  versus vLLM 17,797.37
  (`vllm_17b_tp1_unlimited_512_flashinfer_current_after_argmaxvec2_20260522.json`).
- 0.6B after the fused replacement build measured 28,325.92 output tok/s
  (`pie_06b_default10g_fusedreplace_depth3_unlimited_512x128.json`) versus
  vLLM 28,123.34
  (`vllm_06b_tp1_unlimited_512_flashinfer_autoaffinity_20260522.json`). This
  still beats the refreshed vLLM row but remains below the older 29.8k Pie
  outliers, so the small-model cadence variance is not fully solved.
- Failed/successful policy summary: keep HND, N=8192, graph argmax parts=2,
  static decode plan, lm-head-only Lt (`PIE_CUBLASLT_BF16_MIN_N=100000`) for
  1.7B/8B/14B knob bundles; keep 14B `PIE_CUDA_DECODE_FULL_ATTENTION=0`; keep
  8B page size 16 and depth 1. Keep fused projection replacement and default
  10 GiB threshold. Keep chunked SwiGLU vec2 only for smaller fused gate/up
  widths (`I <= 10000`); do not use it on 8B-class gate/up.

## 2026-05-22 scoped 8B prefill-shape planner

- Accepted source fix: the CUDA memory planner now exposes and prefers a
  12,288-token workspace only for TP1 Qwen3-8B-class dense models on
  L40/H100-class GPUs (`model_type=qwen3`, `hidden_size=4096`,
  `sm_count>=100`, sm8x/sm9x). This keeps the benchmark at unlimited
  client concurrency while changing the server workspace shape. The first
  no-env 8B runs selected `N=12288 R=512` and measured 4,889.38 and
  4,868.88 output tok/s
  (`pie_8b_auto12288_scoped_fusedreplace_hnd_full1_vec2_unlimited_512x128.json`,
  `pie_8b_auto12288_scoped2_fusedreplace_hnd_full1_vec2_unlimited_512x128.json`)
  versus refreshed vLLM 4,805.45. The 8B command should no longer force
  `PIE_CUDA_PREFILL_TOKENS=8192`; let the planner pick 12,288.
- Rejected 8B prefill-shape rows: forced 20,000 tokens was unstable
  (4,863.43 then 4,765.06), 18,432 measured 4,821.90, 24,576 measured
  4,779.73, 10,240 measured 4,838.58, 11,264 measured 4,813.40, 13,312
  measured 4,805.14, and 14,336 measured 4,806.03. Keep the scoped 12,288
  planner preference only.
- Nsight on the final all-fused 8B path (`/tmp/pie8_final_fused_hnd_full1_vec2_node_nsys.nsys-rep`)
  captured the graph-build/body kernels and grouped roughly 2.33 s
  GEMM/CUTLASS (92.4%), 79 ms SwiGLU, 61 ms RMSNorm, 32 ms QKV post/rope/KV,
  and 20 ms attention. The capture does not expand every CUDA-graph replay,
  but it confirms the non-GEMM work is too small to carry a large 8B margin.
- A broader auto-planner attempt was rejected. Raising the large-GPU
  prefill cap globally and adding an unscoped prefill-underfill penalty made
  14B choose `N=4096 R=256`, regressing to 1,783.69 / 1,757.00 output tok/s.
  Restoring the accepted 14B command with `PIE_CUDA_PREFILL_TOKENS=8192`
  selected `N=8192 R=128` and measured 2,010.55 output tok/s
  (`pie_14b_scoped2_prefill8192_default10g_fusedreplace_hnd_decodefull0_vec2_unlimited_512x128.json`).
- Regression checks after scoping: 4B measured 8,298.85 output tok/s
  (`pie_4b_scopedplanner_default10g_fusedreplace_hnd_simple_depth1_unlimited_512x128.json`),
  0.6B measured 28,691.16
  (`pie_06b_scopedplanner_default10g_fusedreplace_depth3_unlimited_512x128.json`),
  and 1.7B reran at 17,929.78 after a low/noisy 17,627.54 first sample
  (`pie_17b_scopedplanner_default10g_fusedreplace_hnd_full1_vec2_unlimited_rerun_512x128.json`).

## 2026-05-22 simplification pass

- Removed the rejected staged-hit aggregation opt-in
  (`PIE_SPEC_HIT_AGG_MAX_EXTRA`). Earlier default and opt-in attempts
  regressed throughput, and the accepted speculative path only needs to claim
  one ready staged result per inferlet call. The API hit path now awaits and
  returns the claimed staged result directly.
- Removed the rejected dense resident cohort slack hook
  (`PIE_DENSE_COHORT_SLACK`). The documented slack experiment regressed the
  0.6B unlimited row, so dense cohorts again fire at the exact tracked target.
- Left the larger or accepted tuning surfaces in place: fused dense projection
  replacement, the scoped 8B planner preference, graph argmax/argmax-parts,
  CUBLASLt thresholds, static decode planning, and the prefill/decode knobs
  used by accepted benchmark rows.

## 2026-05-22 fresh TP1 unlimited advantage check

- Fresh 512x128, TP1, unlimited-concurrency matrix after the simplification
  pass used `--concurrency 0 --ipc-profile latency` for Pie and
  `max_num_seqs=512` / FlashInfer for vLLM. First-pass paired rows:
  - 0.6B: Pie 29,607.42 vs vLLM 28,219.35 output tok/s (+4.92%).
  - 1.7B: Pie 18,228.98 vs vLLM 17,767.86 (+2.60%).
  - 4B: Pie 8,247.50 vs vLLM 8,193.03 (+0.66%).
  - 8B: Pie 4,807.20 vs vLLM 4,779.81 (+0.57%).
  - 14B with the previous accepted graph/static row: Pie 1,988.11 vs vLLM
    2,001.84 (-0.69%). This row is no longer accepted.
- Repeat samples for the tight rows:
  - 4B Pie [8,247.50, 8,277.06, 8,226.37] vs vLLM
    [8,193.03, 8,212.66, 8,187.69]. All paired samples won, but the
    worst-Pie/best-vLLM margin was only +0.17%.
  - 8B Pie [4,807.20, 4,776.94, 4,780.29] vs vLLM
    [4,779.81, 4,752.00, 4,754.60]. All paired samples won, but the
    worst-Pie/best-vLLM margin was -0.06%, so this is a narrow/noisy edge.
  - 14B previous graph/static row Pie [1,988.11, 1,981.33, 1,979.33] vs vLLM
    [2,001.84, 1,997.12, 1,993.21]. Reject graph/static as the 14B default.
- 14B variant sweep showed the old microopts were hurting this build:
  - HND + `PIE_CUDA_PREFILL_TOKENS=8192` + lm-head-only Lt
    (`PIE_CUBLASLT_BF16_MIN_N=100000`) + `PIE_CUDA_DECODE_FULL_ATTENTION=0`,
    with graph argmax and static decode plan unset: 2,031.90 output tok/s.
  - Static decode only: 2,009.13.
  - Graph argmax parts=2 only: 1,999.48.
  - Current graph/static with full attention on: 1,996.52.
  - Default Lt coverage instead of lm-head-only Lt: 1,916.43.
  Decision: do not use `PIE_CUDA_STATIC_DECODE_PLAN=1` or
  `PIE_CUDA_GRAPH_ARGMAX=1` for the 14B TP1 unlimited row.
- 14B simple prefill sweep, keeping HND + lm-head-only Lt + decode-full off and
  leaving graph/static unset: N=6144 measured 2,025.33, N=7168 measured
  2,008.75, N=9216 measured 2,000.11, N=10240 measured 1,998.25, and N=12288
  measured 1,995.17 output tok/s. N=6144 is the best new 14B candidate, but
  repeats were noisy: 1,993.49 and 1,984.63, followed by a paired late-session
  run of Pie 1,996.53 after vLLM had dropped to 1,975.28. This is a paired win
  under the same thermal drift, not a strict every-sample win over the best
  vLLM sample.
- Conclusion from this run: Pie currently has a fresh paired advantage across
  the tested TP1 0.6B/1.7B/4B/8B rows and can recover a paired 14B win with the
  simpler N=6144/no-graph/no-static command. It does not yet have a large,
  noise-proof margin on 4B/8B/14B; 14B in particular needs more GPU-side work
  before claiming "always faster" against vLLM.
