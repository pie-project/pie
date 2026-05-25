# Nemotron-H Omni CUDA Native Notes

Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`

Date: 2026-05-25

## Current status

- CUDA-native TP2 serving is implemented for the text LLM path: hybrid Mamba2,
  attention, routed MoE, shared experts, lm head, recurrent state cache, and
  TP collectives.
- TP1 is not viable for this BF16 checkpoint on the current GPUs. The native
  TP1 plan needs about 63.2 GB of persistent weights before runtime cache.
- Standalone vLLM default AOT still fails startup for this model in this
  environment with a `NoneType.size` dummy/profile-path error. The fair
  runnable target is standalone vLLM V1 with `VLLM_USE_AOT_COMPILE=0`,
  `enforce_eager=False`, CUDA graphs enabled, Triton Mamba SSU/chunk-scan, and
  Triton MoE. This is **not** Pie's vLLM driver.
- Pie CUDA-native TP2 does not yet beat that optimized standalone-vLLM target.
  Pie still beats the older eager baseline, but the current target has moved to
  standalone vLLM no-AOT/non-eager.
- Pie CUDA-native TP2 exactly matches standalone vLLM for the deterministic
  1x8 token check:
  `[4268, 2534, 1317, 7418, 1261, 4958, 8303, 2314]`.
- Latest Pie CUDA-native 1x16 smoke after reverting direct deterministic
  lookahead and the failed FlashInfer decode-SSM attempt is deterministic and
  coherent:
  `[12, 1010, 4268, 2534, 1317, 7418, 1261, 4958, 8303, 2314, 1261, 23057, 1046, 1531, 3330, 25747]`
  with sha256 prefix `b21e81e0e029222a`, text
  `<think>\nWe need to write a short story about a robot. The user asks`.
- Longer deterministic Pie samples are coherent, not gibberish. The 1x64
  sample produces the same reasoning-style preamble and `</think>` behavior as
  standalone vLLM, then starts the story. The 16-token benchmark sample remains
  the exact token sequence above.
- After the decode-conv optimization and cleanup, the latest 1x64 Pie sample is
  still coherent, with sha256 prefix `18d3645073450f00`. It produces a normal
  reasoning preamble and begins the story with `</think>\nThe`; no gibberish was
  observed.

## Current benchmark numbers

All commands used `--tp-size 2`, text-only multimodal, `--gpu-mem-util 0.92`,
temperature 0, ignore EOS, and unlimited throughput concurrency
(`--concurrency 0`) where applicable.

Current fair target: standalone vLLM no-AOT/non-eager via `benches/vllm_bench.py`
with `VLLM_USE_AOT_COMPILE=0`, not Pie's vLLM driver.

| workload | Pie CUDA-native TP2 | standalone vLLM TP2 no-AOT | status |
| --- | ---: | ---: | --- |
| 64x16 warm64 | 2095.97 tok/s | 2820.12 tok/s | Pie 0.74x |
| 128x16 warm128 | 2950.52 tok/s | 3341.09 tok/s | Pie 0.88x |
| 256x16 warm256 | 3361.33 tok/s | 3885.76 tok/s | Pie 0.87x |

Pie numbers above use the working grouped-up MoE device-pointer fix, the direct
grouped-output sum variant, and default-on exact device bucketing for decode
MoE. Baseline files:
`.tmp/benches/nemotron_cuda_native_tp2_exact_decode_moe_tput_*x16*.json`,
`.tmp/benches/nemotron_cuda_native_tp2_default_exactdecode_tput_128x16_warm128.json`,
`.tmp/benches/nemotron_cuda_native_tp2_final_default_tput_256x16_warm256_rerun.json`,
and `.tmp/benches/nemotron_vllm_standalone_tp2_noaot_rerun_tput_*x16*.json`.

After adding packed Nemotron expert backings in the loader, adjacent 64x16
unlimited runs measured 2108.28 tok/s and 2107.38 tok/s after removing the
failed CUTLASS wrapper. This is neutral/slightly up versus the restored
2103.35 tok/s adjacent baseline, so it should be treated as layout foundation
for a future fused MoE kernel rather than a throughput unlock by itself.

Historical eager-vLLM comparison, retained only as context:

- Standalone vLLM eager, latency 4x8 warmed:
  - mean 361.5 ms, p50 361.5 ms
  - output tok/s 22.13
- Pie CUDA-native, latency 4x8 warmed:
  - mean 141.6 ms, p50 139.8 ms
  - output tok/s 56.48
  - speedup: 2.55x mean latency / 2.55x output tok/s
- Standalone vLLM eager, throughput 64x16 unlimited:
  - output tok/s 939.90
- Pie CUDA-native, throughput 64x16 unlimited:
  - output tok/s 1685.31 after larger custom all-reduce, aligned prefill MoE,
    fused aligned-MoE combine, 512-thread prefill SSM, and prefill dt/dA
    precompute
  - speedup: 1.79x
- Standalone vLLM eager, throughput 128x16 unlimited:
  - output tok/s 1337.52
- Pie CUDA-native, throughput 128x16 unlimited:
  - output tok/s 2246.40 after larger custom all-reduce and aligned prefill MoE
  - speedup: 1.68x
- Standalone vLLM eager, throughput 256x16 unlimited, warmup=256:
  - output tok/s 2790.31
- Pie CUDA-native, throughput 256x16 unlimited, warmup=256:
  - latest default run after fused aligned-MoE combine, 512-thread prefill SSM,
    and prefill dt/dA precompute: 2450.63 tok/s
  - best pre-fusion same-path observation: 2437.36 tok/s
  - status: still behind warmed vLLM for this larger steady-state workload
- Intermediate/default-regression numbers on the way here:
  - old Pie default 64x16 with unused system speculation: 443.84 tok/s
  - Pie with `--no-system-speculation`: 584.01 tok/s
  - Pie with no system speculation plus 64 ms prefill grace: 784.07 tok/s
  - Pie after warp SSM, before 8k auto workspace, 64x16: 1148.20 tok/s
  - Pie after warp SSM, 128x16 with 4k workspace: 1330.25 tok/s
  - Pie after forced/default 8k workspace, 128x16: 1424.64 tok/s
  - Pie after B/C cache in Mamba SSM, 256x16 warmup=256: 1838.33 tok/s
  - Pie after decode conv update, 256x16 warmup=256: 2197.28 tok/s
  - Pie cleaned code before Mamba parameter precompute, 256x16 warmup=256:
    2099.58 tok/s
  - Pie with Mamba parameter precompute, 256x16 warmup=256: 2154.52 tok/s
  - Pie with prefill register-state SSM, 64x16 warmup=64: 1503.42 tok/s
  - Pie with prefill register-state SSM, 128x16 warmup=128: 2019.40 tok/s
  - Pie with prefill register-state SSM, 256x16 warmup=256: 2158.09 tok/s
  - Pie after packing host-routed MoE metadata, 128x16: 2043.58 tok/s
  - Pie after packing host-routed MoE metadata, 256x16: 2174.59 tok/s
  - Pie with `PIE_CUDA_CUSTOM_ALL_REDUCE_MAX_MIB=64`, 128x16: 2237.92 tok/s
  - Pie with `PIE_CUDA_CUSTOM_ALL_REDUCE_MAX_MIB=64`, 256x16: 2379.67 tok/s
  - Pie with aligned prefill MoE block size 64 plus 64 MiB custom all-reduce,
    64x16: 1636.39 tok/s, 128x16: 2246.40 tok/s, 256x16: 2437.36 tok/s
  - Pie after fusing aligned MoE reorder+weighted-sum, 64x16: 1670.16 tok/s,
    256x16: 2404.68 tok/s
  - Pie after 512-thread prefill SSM launch on top of fused combine, 64x16:
    1671.28 tok/s, 256x16: 2426.25 tok/s
  - Pie after precomputing prefill Mamba dt/dA once per token/head, 64x16:
    1685.31 tok/s, 256x16: 2450.63 tok/s
- Pie CUDA-native cold 1x8 smoke after cleanup:
  - exact tokens, 539.4 ms wall in the last run. Earlier same-build-shape cold
    runs ranged around 450-540 ms, still below the vLLM eager 1x8 cold baseline
    observed at about 629.5 ms.

## Successful changes

- Added native `nemotron_h` binding and forward implementation for the LLM
  component.
- Added TP2 native loader sharding for Nemotron-H and skipped unused multimodal
  tensors for text-only native serving.
- Added a persistent weight arena for direct-copy storage programs. TP2 load
  high-water now matches the plan: about 31,644 MiB planned per rank and about
  31,646 MiB observed CUDA memory delta.
- Enabled custom all-reduce by default when peer access is available, with
  explicit opt-out via `PIE_CUDA_DISABLE_CUSTOM_ALL_REDUCE=1`. The previous
  environment gate left Nemotron on NCCL when `NCCL_P2P_DISABLE=1`.
- Registered the native workspaces needed by custom all-reduce and made setup
  fall back cleanly to NCCL if peer setup or registration fails.
- Custom all-reduce was the main latency unlock: decode MoE all-reduce dropped
  from about 25.5 ms/token across 23 MoE layers to about 0.3 ms/token.
- Matched vLLM/HF router semantics by computing Nemotron router logits into
  FP32 and running sigmoid/top-k over FP32 logits. The previous BF16 router
  could flip close expert choices.
- Matched vLLM's `dt_limit=(0, inf)` behavior by removing the `time_step_min`
  clamp from the native Mamba SSM call.
- Restored Nemotron recurrent state storage to BF16. This matches vLLM eager's
  auto cache dtype behavior and keeps max recurrent slots at 256 instead of 128.
- Split MoE decode routing policy:
  - N=1 decode uses the all-device route-sized batched GEMM path, which is best
    for latency.
- Batched decode (N>1) uses grouped expert GEMMs, which cut R=16 decode MoE
  from about 24-25 ms/token-step to about 12 ms/token-step.
- Added exact device-side MoE route bucketing for multi-token decode. It keeps
  sorted routes and the route-to-row map on device, copies only the per-expert
  counts back to build cuBLAS grouped shapes, and avoids the previous full
  top-k route/weight D2H sync. Warmed throughput moved 4x16 from 251.68 to
  254.97 tok/s, 16x16 from 868.15 to 879.10 tok/s, 32x16 from 1294.82 to
  1304.43 tok/s, 64x16 from about 2058 to 2095.97 tok/s, 128x16 from 2897.03
  to 2950.52 tok/s, and current 256x16 reruns measured 3353.65-3361.33 tok/s
  with exact decode enabled versus 3194.24 tok/s with it disabled in the same
  build. It is default-on with `PIE_NEMOTRON_DISABLE_EXACT_DECODE_MOE=1` as the
  escape hatch.
- Added packed Nemotron expert backings in the weight-loader ABI:
  `experts.up_proj.packed.weight` and `experts.down_proj.packed.weight` are
  materialized once per MoE layer, and the original per-expert tensor names are
  exposed as views. A Join/Partition optimizer rewrite keeps TP2 down-projection
  packing sharded before materialization, so the real layout dump showed no
  duplicate persistent storage. This is a clean foundation for a true fused MoE
  kernel, not a large runtime win by itself; adjacent 64x16 runs measured
  2108.28 tok/s and 2107.38 tok/s.
- Fixed the opt-in Nemotron grouped-up MoE path by staging cuBLAS grouped GEMM
  pointer arrays on device instead of passing host pointer arrays. The host
  pointer version caused CUDA illegal memory access; the device-pointer version
  is stable and improved 64x16 from about 1967 tok/s to about 2070 tok/s and
  256x16 from about 3268 tok/s to about 3370 tok/s range depending on noise.
- Added a grouped-output direct-sum variant that skips the route-order reorder
  pass and combines from sorted expert rows via a route-to-row map. It is
  slightly worse at 64x16 (2057.81 vs 2070.41 tok/s in adjacent runs), roughly
  neutral at 128x16, and better at 256x16 (3376.20 vs 3268.49 tok/s in
  adjacent rebuild runs). Keep treating this as a small large-batch win, not a
  main unlock.
- Disabled benchmark system speculation by default and defensively clear
  `output_spec_flags` for rs-cache models. The side channel currently returns
  no drafts for this path; leaving the flag set made the scheduler treat prompt
  forwards as dense-logit requests and split 64x16 prefill at 6 requests because
  `max_logit_rows=256`.
- Replaced the Nemotron-H Mamba SSM shared-memory `atomicAdd` accumulation with
  a warp-reduced state-axis accumulation. This preserved the 1x16 token sample
  and moved 64x16 from about 780 tok/s to about 1148 tok/s under the already
  packed scheduler shape.
- Cached the per-token Mamba B/C vectors in shared memory inside the warp SSM
  kernel. This preserved the deterministic output and improved the warmed
  256x16 run from about 1467 tok/s to about 1838 tok/s.
- Switched Nemotron-H pure decode from the batched prefill conv1d kernel to the
  batched update conv1d kernel already used by the Qwen Mamba path. This
  preserved the deterministic token sequence and reduced R=256 decode conv from
  double-digit milliseconds to about 0.55 ms across all Mamba layers; warmed
  256x16 improved to about 2.1-2.2k tok/s.
- Precomputed Mamba `A=-exp(A_log)`, `D`, and `dt_bias` into tiny FP32 buffers
  once during binding, with a single post-loop device sync. This preserved the
  1x16 token sequence. The full 256x16 result moved from the latest 2099.58
  tok/s run to 2154.52 tok/s, but the profile did not show a large SSM-only
  win; treat it as a small/neutral cleanup unless repeated runs confirm more.
- Added a prefill-only Mamba SSM kernel that keeps each lane's state slice in
  registers across the whole sequence and writes the recurrent cache only once
  at the end of prefill. This preserved the deterministic 1x16 token sequence
  and improved 64x16 from 1159.32 to 1503.42 tok/s and 128x16 from 1424.64 to
  2019.40 tok/s. The larger 256x16 run stayed around 2.16k tok/s, so decode and
  large packed-prefill bottlenecks remain.
- Packed CPU-routed MoE token IDs and route weights into one contiguous upload
  per MoE layer. Nsight showed the previous path issuing about 17k tiny H2D
  async copies in a 64x2 forward window. This preserved the deterministic
  1x16 output and moved 256x16 from 2158.09 to 2174.59 tok/s; useful but not
  the main unlock.
- Raised the default custom all-reduce cap from 8 MiB to 64 MiB, with
  `PIE_CUDA_CUSTOM_ALL_REDUCE_MAX_MIB` as an override. Large prefill MoE
  reductions were falling back to NCCL and taking about 150-175 ms per large
  profiled forward; the larger cap brought that down to about 50-58 ms and
  moved 256x16 to about 2.38k tok/s.
- Added a default-on GPU-aligned prefill MoE path for Nemotron-H. It aligns
  routes by expert on device, gathers padded expert blocks, runs fixed-size
  batched BF16 GEMMs, and avoids the CPU routing sync for prefill. Block size
  64 is the best tested setting; 256x16 reached 2437.36 tok/s in the best
  pre-fusion run.
- Fused the aligned prefill MoE output combine by writing a route-to-aligned-row
  inverse map during alignment, then accumulating top-k weights directly from
  aligned down-projection output. This removes the full route-order reorder
  write/read pass while preserving the 1x16 token sequence. It improved 64x16
  from 1525.41 to 1670.16 tok/s in the immediate cleanup comparison and measured
  2404.68 tok/s on 256x16.
- Increased the prefill register-state SSM launch from 256 to 512 threads per
  block. This preserves the per-warp math order and the 1x16 token sequence.
  It was neutral on 64x16 versus the fused-combine path but improved the latest
  256x16 run from 2404.68 to 2426.25 tok/s.
- Precomputed prefill Mamba `dt=softplus(dt+dt_bias)` and `dA=exp(dt*A)` once
  per token/head before the register-state SSM kernel. This preserved the
  deterministic 1x16 token sequence and moved 64x16 from 1671.28 to
  1685.31 tok/s and 256x16 from 2426.25 to 2450.63 tok/s. It is a small,
  general cleanup because it removes repeated scalar work across head-dim
  warps without changing the recurrence.
- Adjusted the auto memory planner for Nemotron-H TP2 on L40/Ada to select the
  8192-token workspace. The previous 4096-token workspace split 128 prompt
  tokens (about 4884 prompt tokens total) into multiple prefill batches. The
  8192-token plan still keeps 256 recurrent state slots and improves both 64x16
  and 128x16.
- Made prefill cohort grace adaptive: small prompt batches keep the 16 ms grace,
  while already-forming cohorts of at least 32 requests get 64 ms. This preserves
  serial latency behavior while avoiding premature first-burst fires in
  throughput mode.

## Profiling notes

- Latency decode profile after custom all-reduce:
  - total forward about 13.5 ms for N=1
  - Mamba about 5.5 ms
  - MoE about 4.1 ms
  - lm head about 1.0 ms
  - uncategorized runtime/driver overhead about 2.1 ms
- Batched decode profile before grouped decode:
  - N=16 total about 38-39 ms/token-step
  - MoE about 24-25 ms, routed MoE about 22 ms
- Batched decode profile after grouped decode:
  - N=16 total about 25-29 ms/token-step
  - MoE about 12-15 ms, routed MoE about 10-14 ms
  - Mamba about 9.4-9.7 ms
- Large packed 64x16 profile before warp SSM was intrusive but directional:
  - prefill N=2422/R=64 total about 979 ms under profiling
  - Mamba about 637 ms, with SSM about 433 ms
  - MoE about 223 ms, attention about 68 ms, lm head about 41 ms
  - R=64 decode showed Mamba and MoE as the remaining main contributors
- Prefill remains the main deeper optimization target. For chunks around
  220 tokens, native Mamba SSM is about 66-68 ms across 23 Mamba layers. vLLM
  uses Triton chunk-scan/SSU kernels for this class of work.
- After the decode-conv fix, a 256x2 profile shows the remaining R=256 decode
  hot spots as Mamba SSM (~19 ms across 23 Mamba layers) and MoE (~16-17 ms
  across 23 MoE layers). Prefill remains split into two chunks at the default
  8192-token workspace and is dominated by Mamba plus MoE/all-reduce.
- After the prefill register-state SSM change, an intrusive 256x2 profile split
  prefill into N=7179/R=197 and N=2200/R=76 chunks, then decoded at R=239. The
  first chunk spent about 455 ms in Mamba (197 ms inproj, 183 ms SSM) and about
  416 ms in MoE (186 ms routed, 168 ms all-reduce). The decode step spent about
  23 ms in Mamba (18 ms SSM) and about 21 ms in MoE (14 ms routed).
- vLLM and SGLang Mamba2 prefill paths use an SSD/chunk-scan pipeline instead
  of a pure recurrent scan: chunk cumsum, chunk state construction, state
  passing, chunk bmm, and chunk scan. Their decode paths dispatch selective
  state update kernels from Triton or FlashInfer. TensorRT-LLM has similar C++
  selective-scan/chunk-scan kernels plus Mamba2 cache-update kernels.
- vLLM, SGLang, and TensorRT-LLM MoE paths avoid host-routed per-expert work in
  the hot path. They fuse routing/alignment and use grouped GEMM/CUTLASS,
  FlashInfer, DeepGEMM, or TRT-LLM kernels. Pie's current Nemotron multi-token
  route still has CPU routing and per-expert GEMM launches.
- Nsight Systems tracing works if the generated `.qdstrm` is manually imported
  with `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`; the wrapper
  failed to auto-import. The 64x2 forward window showed the main CUDA-level
  issues: thousands of per-expert kernels, about 17k small H2D metadata copies,
  and NCCL all-reduce kernels dominating large prefill before the 64 MiB custom
  all-reduce cap.
- With the 64 MiB custom all-reduce cap and aligned prefill MoE enabled, a
  256x2 profile still shows large prefill dominated by Mamba plus routed MoE:
  first chunk N=6846/R=188 spent about 443 ms in Mamba and 247 ms in MoE;
  second large chunk N=7844/R=212 spent about 366 ms in Mamba and 240 ms in
  MoE. Decode at R=201 spent about 20 ms in Mamba and 16 ms in MoE.
- After precomputing prefill dt/dA, a 256x2 profile still shows the same main
  shape. The first prefill chunk N=6920/R=190 spent about 444 ms in Mamba
  including 181.6 ms in SSM and 225.7 ms in MoE; the second N=2458/R=82 chunk
  spent about 56 ms in SSM and 102 ms in MoE. Decode in that intrusive run was
  about 18 ms Mamba SSM and 31 ms MoE at R=240.
- Latest grouped-up profile after the TP-sharded Mamba and grouped pointer fix
  still points at the same work: large prefill N=6994/R=192 spent about
  313 ms in Mamba and 226 ms in MoE; decode N=202/R=202 spent about 11.7 ms in
  Mamba and 32.8 ms in MoE. The largest fair gap versus standalone vLLM is not
  request admission; it is GPU-side Mamba chunk-scan/SSU and fused/grouped MoE.
- Refreshed direct-sum profile before exact decode bucketing:
  256x2 split into prefill N=7142/R=196 and N=2262/R=102, then decode
  N=163/R=163 and N=51/R=51. The first prefill chunk spent 302.8 ms in Mamba
  and 229.3 ms in MoE; decode remained split between Mamba SSM and routed MoE.
- With exact decode bucketing enabled, the intrusive 256x2 profile still shows
  the same high-level gap: large prefill dominates, and the decode step at
  N=214/R=214 spent about 27.7 ms in Mamba and 32.9 ms in MoE across all
  layers. The exact bucketer improves steady throughput but is not a substitute
  for a fused MoE kernel.

### Nsight Systems CUDA-only hotspot comparison (2026-05-25)

Captured with `nsys profile --trace=cuda --sample=none
--capture-range=cudaProfilerApi --capture-range-end=stop` wrapping the bench;
both engines bracket only the timed `tput` window through `--cuda-profiler-range`
so the trace excludes model load and warmup. TP2 worker subprocesses are
included because the cudaProfilerApi capture is process-tree wide. Numbers
below are kernel-name percentages of total CUDA kernel time across both GPUs
in the captured window. Raw nsys reports live under `.tmp/nsys/`:

- Pie 64x16 warm64: `nemo_pie_64x16_warm64_cudaonly.nsys-rep` (wall 0.483 s, 2122 tok/s under nsys)
- vLLM 64x16 warm64: `nemo_vllm_64x16_warm64_cudaonly.nsys-rep` (wall 0.374 s, 2740 tok/s under nsys)
- Pie 256x16 warm256: `nemo_pie_256x16_warm256_cudaonly.nsys-rep` (wall 1.218 s, 3362 tok/s under nsys)
- vLLM 256x16 warm256: `nemo_vllm_256x16_warm256_cudaonly.nsys-rep` (wall 1.370 s, 2989 tok/s under nsys)

Top kernels at 64x16 warm64:

| bucket | Pie | vLLM |
| --- | --- | --- |
| MoE expert GEMMs | 33.5% (Cutlass MoeFCGemm x2) | 29.1% (Triton fused_moe_kernel x2) |
| TP all-reduce | 17.0% (vllm::cross_device_reduce_1stage, in-tree custom AR) | 25.4% (ncclDevKernel_AllReduce_Sum_bf16) + 5.3% AllGather |
| Mamba SSM core | 12.1% (mamba_ssm_batched_prefill_reg + warp decode) | 5.2% (chunk_scan + chunk_state + state_passing + bmm_chunk + selective_scan_update + causal_conv1d) |
| Mamba in-proj GEMM | 5.8% (cutlass bf16 64x64 prefill) | 6.8% (ampere bf16 64x128) |
| Decode workhorse GEMM | 5.5% (ampere bf16 64x64) | 6.3% (ampere bf16 128x64) + 6.2% (128x64 sliced) |

Top kernels at 256x16 warm256:

| bucket | Pie | vLLM |
| --- | --- | --- |
| TP all-reduce | 22.4% (custom AR) | 27.8% (NCCL AllReduce) + 5.1% AllGather |
| MoE expert GEMMs | 26.3% (Cutlass MoeFCGemm x2) | 25.2% (Triton fused_moe_kernel, 6 distinct shapes) |
| Mamba SSM core | 23.5% (warp decode 12.6% + prefill_reg 8.1% + prefill_reg alt 2.1% + warp variant 0.7%) | 6.2% (selective_scan_update x2 = 2.1% + state_passing x2 = 2.0% + chunk_state x2 = 1.8% + chunk_scan x2 = 1.2% + causal_conv1d) |
| Mamba in-proj GEMM | 3.2% prefill 64x128 + 2.8% decode 64x128 | 3.0% prefill 64x128 + 1.4% decode 64x64 |
| Misc GEMMs/finalize | ~6% (ampere bf16 GEMMs, finalizeMoeRoutingKernel) | ~7% (ampere bf16 GEMMs) |

Takeaways:

1. **Mamba SSM is the dominant remaining GPU-side gap.** At 256x16, Pie spends
   23.5% of GPU time in custom recurrent SSM kernels while vLLM spends 6.2%
   in the Triton chunk-scan / state-passing / selective-scan-update family.
   The ratio at 64x16 is similar (12.1% vs 5.2%). This is the largest
   single-bucket gap and is consistent with the recurrent vs. SSU/chunk-scan
   structural difference. Item 3 of REMAINING.md remains the highest-leverage
   next change.
2. **MoE expert GEMMs are roughly comparable.** Cutlass MoeFCGemm (Pie) and
   Triton fused_moe_kernel (vLLM) consume similar fractions of GPU time at
   both 64x16 and 256x16. A fused/grouped MoE rewrite (REMAINING.md item 4)
   will still help, but the headroom is smaller than the Mamba gap.
3. **Pie's custom all-reduce remains a real win on collectives.** At 256x16
   Pie spends 22.4% in custom AR; vLLM spends 27.8% in NCCL AllReduce plus
   another 5.1% in NCCL AllGather. Keep custom AR default-on with peer access.
4. **Throughput at 256x16 is noisy.** This single nsys run had Pie at 3362
   tok/s and vLLM at 2989 tok/s — i.e. Pie wins — which contradicts the
   recent fair-bench snapshot in `REMAINING.md`. The variance argues for
   three-run medians (REMAINING.md item 1) before any further claim about
   the 256x16 gap.

### FlashInfer SSU dispatch scaffolding (2026-05-25)

`flashinfer_mamba_ssu_enabled()` is now SM-aware: defaults on for sm_90+,
off for sm_89 and below. Env override `PIE_NEMOTRON_FLASHINFER_SSU=0|1`
still wins when set.

Rationale (microbenched on L40 the same day):

- FlashInfer SSU only exposes the `simple` algorithm on sm_89; the
  `vertical` and `horizontal` algorithms that beat Pie's `warp_kernel` at
  steady-state R are compute-capability ≥ 9. On L40, `simple` is faster
  than `warp_kernel` up to R≈144 (FI 41 µs vs Pie 53 µs at R=64) but
  trails it past R=160 (FI 132–897 µs vs Pie ~371 µs at R=256). Defaulting
  on for L40 would hurt the 256x16/512x64 workloads we need to win.
- On H100/B100 the better algorithms unlock, so SM-default-on is the
  right call there. Validated on H100/B100 the moment hardware appears.

End-to-end correctness validated on L40 by forcing `PIE_NEMOTRON_FLASHINFER_SSU=1`:

- 1x8 deterministic: `[4268, 2534, 1317, 7418, 1261, 4958, 8303, 2314]` ✓ identical to legacy (sha256[:16]=`7ff374afdf996ced`)
- 1x16 deterministic: 16 tokens identical to legacy (sha256[:16]=`c5765b793b7abbfb`)

This means strides, slot mapping, A/B/C layouts, and state-cache update
are correct end-to-end in the FlashInfer path on this codebase. The H100/B100
ship path is the same code, just a different FlashInfer algorithm.

L40 non-regression (legacy path still default):

- 64x16 warm64: 2210.90 tok/s (-0.46% vs 2221.11 historical, within noise)
- 256x16 warm256: 3370–3483 tok/s across 3 runs (median 3382, identical to
  force-off run at 3372; the 3548 historical was top-of-band)
- 1x8 / 1x16 deterministic: unchanged

## Failed or neutral attempts

- Standalone vLLM default AOT failed startup for this model with
  `AttributeError: 'NoneType' object has no attribute 'size'` in the
  Nemotron-H profile/compile path. Do not compare against Pie's vLLM driver.
  The current standalone target is `VLLM_USE_AOT_COMPILE=0`, non-eager.
- Disabling custom all-reduce did not change the 16-token correctness drift,
  but it regressed 1x16 latency from about 534 ms to about 931 ms.
- Storing Nemotron recurrent SSM state as FP32 did not fix 16-token parity and
  reduced planned recurrent slots from 256 to 128. Removed from the default
  Nemotron path.
- Removing the Mamba dt floor was semantically correct versus vLLM, but by
  itself did not fix 16-token parity.
- Forcing grouped MoE for N=1 preserved tokens but regressed 1x8 latency from
  about 450 ms to about 477 ms, so the route-sized device path remains the
  N=1 latency path.
- Forcing the all-device route-sized MoE path for prefill regressed 16x16
  throughput from about 296 tok/s to about 145 tok/s. The CPU-routed grouped
  prefill path is not elegant, but it is faster for this shape until a real
  grouped/fused MoE kernel is wired.
- `memory-profile=throughput` regressed 16x16 throughput to about 177 tok/s for
  this model/shape. The auto-selected latency profile is currently best.
- Single-process batch mode was slower than launching concurrent benchmark
  processes for this workload: about 238 tok/s versus about 296 tok/s.
- `--system-speculation` was harmful for this rs-cache model: it requested
  dense logits but produced no usable speculative drafts. Scheduler trace showed
  the concrete split reason:
  `logit_rows 259>256` after only 6 prompt requests.
- Increasing prefill grace while system speculation was still enabled did not
  help; the dense-logit row limit still dominated.
- `--memory-profile throughput` did not help Nemotron-H. It kept
  `max_forward_tokens=4096`, reduced state slots/max requests to 128, and
  regressed 128x16 to about 1016 tok/s after the warp SSM change.
- `PIE_CUDA_PREFILL_DECODE_PLAN=1` had no effect for this native path in the
  tested build; logs still reported `prefill_decode_plan=off`.
- Single-process batch, defer-start, eager scheduling, and greedy scheduling did
  not address the dense-logit/system-speculation regression.
- On the larger warmed 256x16 workload, `--defer-start` and
  `--single-process-batch` are still worse: about 1744 tok/s and 1778 tok/s,
  respectively, versus about 2.1k-2.2k tok/s for the normal unlimited
  concurrency path.
- Forcing `PIE_CUDA_PREFILL_TOKENS=8192` was successful and became the
  Nemotron-H TP2/Ada auto-planner preference after confirming it did not hurt
  64x16.
- A decode-only SSM kernel that removed prompt-path loop/indptr work preserved
  the 1x16 token sequence but did not improve warmed 256x16 (about 2185 tok/s
  versus about 2197 tok/s before it). Removed.
- 128-thread SSM blocks via a temporary launcher knob regressed profiled 256x2
  behavior; the 256-thread warp SSM launch remains best among tested shapes.
  Removed the knob.
- `PIE_CUDA_PREFILL_TOKENS=10240` with `--gpu-mem-util 0.98` fit
  N=10240/R=256 but regressed warmed 256x16 to about 2133 tok/s. Larger
  workspace/memory pressure is not a win here.
- Setting both prefill grace env vars to 0 regressed warmed 256x16 to about
  2086 tok/s. The adaptive grace remains useful even after the system-spec
  dense-logit bug was fixed.
- Nemotron-specific aligned MoE decode using the existing vLLM/SGL-style route
  alignment was tested and removed. Block size 16 made profiled 256x2 decode
  MoE worse (~24 ms), and block size 4 was worse still (~30 ms), likely due to
  padding/GEMM-shape tradeoffs for this ReLU2 expert layout.
- Forcing the all-device M=1 batched-GEMM MoE decode path for N>1 was much
  worse at R=256; routed decode ballooned to hundreds of milliseconds in the
  profile. The CPU-routed grouped path is still the best tested batched decode
  path for Nemotron-H.
- Forcing the new register-state SSM kernel for decode preserved the 1x16 token
  sequence but regressed 128x16 from 2019.40 to 1966.22 tok/s. Decode remains
  on the older warp/shared-B-C SSM kernel.
- Retesting a 10240-token prefill workspace with `--gpu-mem-util 0.98` after
  the prefill register-state SSM change regressed 256x16 to 2080.77 tok/s.
  The larger workspace still increases pressure more than it helps batching.
- Disabling custom all-reduce after the prefill register-state SSM change
  regressed 128x16 from 2019.40 to 1479.95 tok/s. The custom all-reduce path
  remains necessary.
- The first opt-in cuBLAS grouped-BF16 MoE attempt passed host pointer arrays
  into `cublasGemmGroupedBatchedEx` and failed later with CUDA illegal memory
  access. The useful part was fixed by copying pointer arrays into device
  buffers before calling cuBLAS grouped GEMM. Full grouped-down projection was
  also tried and removed: it did not finish after several minutes on 64x16.
- Forcing cuBLASLt for all BF16 projections with `PIE_CUBLASLT_BF16_MIN_N=0`
  made a 128x16 warmup run exceed five minutes without producing a result. The
  default cuBLAS/cuBLASLt heuristic should not be replaced by this blanket knob.
- Lowering the cuBLASLt BF16 threshold to 6144 to catch only the wide Mamba
  in-projection shape also exceeded two minutes on 128x16 without a result.
  The Mamba projection shape does not want the current Lt heuristic on L40.
- Aligned prefill MoE block size 128 regressed 256x16 to 2340.90 tok/s and
  block size 32 regressed to 2200.16 tok/s. Block size 64 is the best tested
  tradeoff.
- After the fused aligned-MoE combine, block size 96 still regressed 256x16 to
  2345.49 tok/s and block size 48 regressed to 2387.50 tok/s. Keep block size
  64 as the default.
- A 1024-thread prefill register-state SSM launch preserved the 1x16 token
  sequence but regressed 64x16 to 1658.53 tok/s versus 1671.28 tok/s for the
  512-thread launch. Reverted to 512 threads.
- Increasing large-cohort prefill grace to 128 ms after the fused-combine and
  512-thread SSM changes regressed 256x16 to 2415.30 tok/s versus 2426.25 tok/s
  for the current adaptive 64 ms large-cohort grace.
- Retesting `PIE_CUDA_PREFILL_TOKENS=10240` with the 64 MiB custom all-reduce
  cap and aligned prefill MoE still regressed 256x16 to 2365.26 tok/s.
- Setting both prefill grace env vars to 0 after the larger custom all-reduce
  and aligned prefill MoE changes still regressed 256x16 to 2332.77 tok/s.
- A prefill register-state SSM variant that cached B/C in shared memory changed
  the deterministic 1x16 token sequence (`... user asks` became
  `... user just` at the last token), so the experimental path was removed.
- A CUTLASS `GemmGrouped` MoE wrapper modeled after TensorRT-LLM built and
  produced the correct 1x16 deterministic output only after making its scratch
  buffers per-device. It is not a viable integration point as wired through
  host-routed expert metadata: the 64x16 warmup run exceeded four minutes with
  one GPU saturated and no result. Removed rather than keeping a slow opt-in
  path.
- A decode SSM kernel that shared each token's B/C vectors across all heads in
  a Mamba group preserved the deterministic 1x16 output but regressed warmed
  64x16 to 1650.40 tok/s and warmed 256x16 to 2351.92 tok/s. Rechecking the
  old warp SSM path in the same build produced 2442.15 tok/s on 256x16. The
  shared-B/C decode kernel was removed; it saves redundant loads but loses too
  much parallelism.
- A FlashInfer-style simple STP decode kernel with 4 warps and vectorized
  contiguous state loads/stores also preserved the deterministic 1x16 output,
  but regressed warmed 64x16 to 1576.42 tok/s and warmed 256x16 to 1957.68
  tok/s. Removed; Pie's existing 8-warp strided update is faster for this
  bf16-state Ada workload.
- A lighter bf16-pair variant of the existing 8-warp decode SSM kernel also
  preserved the deterministic 1x16 output, but regressed warmed 64x16 to
  1659.43 tok/s and warmed 256x16 to 2386.47 tok/s. Removed; pairing the state
  axis reduces loop overhead but does not beat the current scalar lane mapping.
- A TensorRT-LLM-style one-thread-per-output-channel decode SSM update
  preserved the deterministic 1x16 output, but regressed warmed 64x16 to
  1135.08 tok/s and warmed 256x16 to 1555.72 tok/s. Removed; serializing the
  128-state loop per channel loses too much B/C reuse and state-axis
  parallelism in Pie's layout.
- Retesting the opt-in cuBLAS grouped-BF16 MoE path after the dt/dA precompute
  initially still failed because the pointer arrays were host-side. After the
  device-pointer fix, grouped-up/per-expert-down is valid and measurable; the
  earlier failing note should not be repeated.
- Aligned all-device decode MoE was retested for Nemotron with block sizes 16
  and 8. It regressed 64x16 to 1665.31 and 1699.09 tok/s, respectively, versus
  about 2070 tok/s for host-routed grouped-up. Padding and batched-GEMM shape
  costs outweighed the removed CPU sync.
- Forcing the route-sized all-device M=1 batched-GEMM MoE path for N>1 decode
  regressed 64x16 to 627.79 tok/s. Removing the CPU routing sync is not enough
  if the replacement launches one tiny GEMM per route.
- Skipping the D2H top-k weight copy in the grouped-up path was neutral at
  64x16 but regressed 256x16 to 3238.51 tok/s. Explicit stream-bound pageable
  pointer-array copies were also neutral at 64x16 and regressed 256x16 to
  2684.05 tok/s. Reverted to the measured-good pointer staging path.
- Retesting `PIE_CUDA_PREFILL_TOKENS=10240` with grouped-up MoE regressed
  256x16 to 3221.65 tok/s. Keep the 8192-token workspace.
- A direct CUTLASS `GemmGrouped` grouped-up wrapper was retried with a narrow
  CUDA translation unit to avoid the earlier `toml++`/nvcc include failure.
  It built and preserved the deterministic 1x16 output. The 32x128 tile was
  only neutral at 64x16 (2076.13 tok/s) and regressed 256x16 to 3249.04 tok/s;
  the 16x128 tile was neutral at 64x16 (2073.54 tok/s) and stalled the 256x16
  run for more than four minutes. Removed again; this direct CUTLASS wrapper is
  not the missing fused MoE path.
- Direct integration of FlashInfer's BF16 selective-state-update decode header
  was retried with `FLASHINFER_ENABLE_BF16` and the generated JIT config aliases
  defined in a narrow CUDA TU. It built and preserved the exact 1x16 output, but
  was neutral/slower: 64x16 measured 2054.52 tok/s and 256x16 measured
  3359.44 tok/s versus 3376.20+ for the default path at the time. Removed; Pie's
  existing 8-warp decode SSM remains better on Ada/BF16 state.
- Exact device bucketing for prefill MoE helped smaller single-chunk prefill
  shapes when forced (`PIE_NEMOTRON_EXACT_PREFILL_MOE_MAX_TOKENS=...`):
  64x16 reached 2118.91 tok/s and 128x16 reached 3088.51 tok/s. It regressed
  256x16 to 3362.24 tok/s, and a 5000-token threshold still regressed 256x16 to
  3389.29 tok/s. Default is disabled (`0`) until there is a safe whole-workload
  selector or a true fused MoE kernel.
- Grouped down-projection for the exact bucket path no longer hangs, but it is
  not a default candidate: it changed the exact 4-request deterministic output,
  improved 64x16 only slightly (2113.89 tok/s), and regressed 256x16 to
  3418.84 tok/s versus 3437.44 tok/s for exact bucket plus per-expert down
  GEMMs. Removed from active code.
- Retesting `PIE_CUDA_PREFILL_TOKENS=10240` after exact decode bucketing still
  regressed 256x16 to 3248.69 tok/s. Keep the 8192-token workspace.
- A FlashInfer/CUTLASS segment-GEMM wrapper for grouped-up MoE also failed at
  build on this toolchain because including the FlashInfer group-GEMM header in
  a CUDA translation unit pulled in `toml++` and triggered an nvcc template
  error. Removed; revisit only with a narrow wrapper that avoids those headers
  or a prebuilt FlashInfer extension boundary.
- A later packed-expert foundation pass succeeded: the weight loader now
  creates per-layer `experts.up_proj.packed.weight` and
  `experts.down_proj.packed.weight` backings and exposes the legacy per-expert
  names as views. The real Nemotron TP2 layout dump had 46 extra contracts
  (23 layers x 2 packed backings) but no duplicate persistent storage; the
  deterministic 1x16 output stayed
  `[12, 1010, 4268, 2534, 1317, 7418, 1261, 4958, 8303, 2314, 1261, 23057, 1046, 1531, 3330, 25747]`
  and adjacent 64x16 runs measured 2108.28 tok/s and 2107.38 tok/s. Keep this
  as a clean prerequisite for a true fused/grouped MoE path.
- Retesting a grouped down-projection path after the packed-layout work
  preserved the 1x16 deterministic output but the 64x16 timed run did not
  finish after more than three minutes and required killing the server. Removed
  again; do not repeat this path without changing the down-projection kernel
  strategy.
- Directly wrapping FlashInfer/TensorRT-LLM `CutlassMoeFCRunner` against the
  packed expert tensors was attempted as an opt-in `PIE_NEMOTRON_CUTLASS_MOE`
  path. The first build needed TensorRT-LLM internal includes, then failed link
  on `MoeGemmRunner`, TMA workspace helpers, sanitized memcpy, and LoRA stubs.
  Adding the minimal runner/runtime sources still expanded into `TllmException`
  and many SM80 fused-GEMM launcher instantiations. Removed rather than
  coupling Pie to a partial TensorRT-LLM runtime; revisit only as a proper
  vendored library boundary or a smaller fixed-shape kernel.
- Source scan after that attempt: vLLM's Mamba2 prefill advantage is primarily
  in Python/Triton SSD chunk kernels (`ssd_combined`, `ssd_chunk_state`,
  `ssd_chunk_scan`, `ssd_state_passing`) plus Triton/FlashInfer SSU dispatch.
  FlashInfer also has C++/CuTe SSD examples, but the available path is
  Hopper/Blackwell-oriented and not exposed as a narrow stable C++ ABI for
  Pie's Ada CUDA driver. Do not try to solve this by including random
  FlashInfer internals in `driver/cuda`; build a native Pie chunk-scan kernel
  or add a proper vendored library boundary.
- FlashInfer/SGLang/vLLM MoE integration scan: FlashInfer exposes two different
  MoE families. `trtllm_bf16_moe` is the TensorRT-LLM-Gen monolithic path and
  is gated to SM100+ in FlashInfer/vLLM; it also expects shuffled/block-layout
  weights, so it is not the right L40/Ada path. `cutlass_fused_moe` is the
  cleaner backend boundary for Pie: it has an SM89 JIT module, accepts canonical
  3D expert tensors `[E, I, H]` and `[E, H, I]`, supports `ActivationType.Relu2`
  for non-gated experts, and is what SGLang calls directly for unquantized
  FlashInfer CUTLASS. If Pie leverages FlashInfer, do it as a separate vendored
  FlashInfer CUTLASS MoE library plus a thin Pie C ABI, not by including
  TensorRT-LLM internal headers in `driver/cuda`.
- Profiling the cleaned default path shows the next real bottleneck: at
  256-request decode, Mamba SSM is about 19.2 ms per forward on rank 0 while
  routed MoE is about 9.7-10.4 ms. At batch-16 decode, routed MoE is the
  larger repeated cost (~6.6-7.2 ms), but the high-throughput vLLM gap is now
  mostly the Mamba recurrent scan.
- Nsight Systems is installed, but the current install can only capture
  `.qdstrm` for fresh runs; import failed because the importer binary is
  missing. An older imported `.tmp/nsys/nemotron_pie_tp2_64x2.sqlite` confirms
  the same CUDA-level shape for that short run: prefill register SSM is a top
  kernel group (~191 ms total), decode warp SSM is visible (~11.5 ms total),
  and NCCL all-reduce is also large in the traced workload (~282 ms total).

## 2026-05-25 follow-up attempts

- FlashInfer BF16 selective-state-update was integrated behind
  `PIE_NEMOTRON_FLASHINFER_SSU=1` and rechecked. It preserved the deterministic
  1x16 output
  `[12, 1010, 4268, 2534, 1317, 7418, 1261, 4958, 8303, 2314, 1261, 23057, 1046, 1531, 3330, 25747]`,
  but warmed 64x16 reached only 2084.89 tok/s. Keep it off by default; Pie's
  current decode SSM kernel is still faster on this Ada/BF16 shape.
- Forcing the existing register-state SSM launch for decode
  (`PIE_NEMOTRON_FORCE_PREFILL_REG_SSM=1`) regressed warmed 64x16 to
  2074.65 tok/s. A narrower tile-4 decode SSM variant changed token 16
  (`25747 -> 2586`) and was also slower at 2075.71 tok/s. Do not repeat that
  tiling without a different accumulation strategy and a correctness fix.
- The all-device aligned decode MoE path was retested with block sizes 64, 16,
  8, and 1. It stayed token-exact on the 4x16 smoke but regressed warmed 64x16
  to 1522.51, 1558.48, 1415.32, and 423.66 tok/s respectively. Padding and
  tiny batched-GEMM costs dominate the removed CPU sync.
- Nemotron CUDA graph capture works mechanically when the forward body is
  graph-safe: with aligned decode MoE it captured 35 decode graphs (~362-364
  MiB graph memory, ~0.7 s upfront capture). The graph-safe aligned MoE body is
  too slow, though: warmed 64x16 with graphs plus block-16 aligned decode MoE
  measured 1094.63 tok/s.
- The route-wise all-device M=1 batched-GEMM decode MoE path for N>1 was
  retried because it is graph-safe and matches the Qwen3.5/3.6 decode shape.
  It was token-exact on 4x16, but warmed 64x16 regressed to 629.05 tok/s; with
  CUDA graphs it regressed further to 566.48 tok/s. The exact per-expert
  grouped-up path remains the best tested decode MoE path despite its host
  counts sync.
- Grouping the exact path's down-projection again preserved the small 4x16
  deterministic output, but the warmed 64x16 run did not finish after more than
  two minutes and had to be killed. This reproduces the earlier sustained-run
  cuBLAS grouped-down instability.
- FlashInfer/CUTLASS MoE prefill-only still completes, but enabling it for
  decode still fails. The default decode tactic (`GEMM2=auto`) failed all 4x4
  requests with CUDA illegal memory access at the next stream sync; forcing
  `PIE_NEMOTRON_FLASHINFER_MOE_GEMM2=finalize` hung the 4x4 smoke until killed.
  Forcing `PIE_NEMOTRON_FLASHINFER_MOE_GEMM2=none` also failed the 4x4 decode
  smoke with the same illegal memory access, so the issue is not only the fused
  finalize epilogue. Do not enable FlashInfer MoE decode in serving until there
  is a standalone runner harness that isolates the invalid decode shape/tactic.
- cuBLASLt for the narrow Mamba in-projection shape was retried with
  `PIE_CUBLASLT_BF16_MIN_N=5000 PIE_CUBLASLT_BF16_MAX_N=5200`; the 64x16 run
  exceeded two minutes and was killed. This matches earlier broad Lt failures.
- A follow-up added BF16 cuBLASLt plan caching in `ops/gemm.cpp` so repeated
  shapes reuse descriptors and the selected tactic. The default 64x16 path was
  neutral at 2107.25 tok/s, but re-enabling cuBLASLt for the Mamba
  in-projection shape still made the 64x16 run stall for ~2.5 minutes before
  it was killed. The issue is the chosen Lt kernel/shape behavior, not merely
  descriptor or heuristic setup overhead.
- Tuning that same Mamba in-projection Lt path with
  `PIE_CUBLASLT_BF16_ALGO_INDEX=0` completed and once reduced prefill
  `mamba_inproj` by ~9 ms, but it also slowed decode in-projection badly when
  used at small M. Adding `PIE_CUBLASLT_BF16_MIN_M=512` avoided the decode hit,
  but the repeated profile was neutral/slightly slower (`mamba_inproj` 146.6 ms
  vs 148.0 ms baseline, total 380.2 ms vs 377.4 ms). Do not default this.
- Forcing cuBLASLt only on the prefill router shape (`N=128`, `K=2688`,
  `M>=512`, algo index 0) was also worse: 64x2 profile total rose to 390.0 ms
  and MoE routed time rose to 67.5 ms. Keep the default cuBLAS path for router
  GEMMs.
- Forcing cuBLASLt on the `N=2688` projection family (expert down/shared
  down/Mamba out-style shapes) with `PIE_CUBLASLT_BF16_MIN_K=0` and algo index
  0 did not complete the 64x2 profile after startup and was killed. Do not
  broaden Lt to this family.
- Making fused custom all-reduce plus residual RMSNorm default was tested and
  rejected for correctness: the first 15 deterministic tokens matched, but
  token 16 changed from `25747` to `2586`. Keep `PIE_NEMOTRON_FUSED_AR_NORM=1`
  opt-in only unless we can preserve the current BF16-rounding semantics.
- A fresh intrusive stage profile on 64x2 (`PIE_NEMOTRON_PROFILE=1`) showed:
  prefill total 377.6 ms with Mamba 191.3 ms (`mamba_inproj` 147.7 ms,
  `mamba_ssm` 20.4 ms) and MoE 136.3 ms (`router` 53.5 ms, `routed` 58.7 ms,
  all-reduce 16.2 ms). Decode at N=57 spent 18.5 ms total, with attention only
  0.8 ms, Mamba 4.3 ms, and MoE 9.7 ms. The gap is GPU-side Mamba/MoE, not
  request admission or benchmark concurrency.

## Next high-value work

- Replace the native Mamba SSM recurrence with a chunk-scan/SSU path
  comparable to vLLM/SGLang for Mamba2. This is now the largest remaining
  high-throughput hotspot, especially at 256-request decode.
- Replace the current padded-block prefill MoE with a real fused/grouped MoE
  kernel (CUTLASS/FlashInfer/TRT-LLM style). The default aligned path removes
  host routing sync, but it still pays padding and generic batched-GEMM costs.
- Investigate the 16-token numerical drift by comparing layer outputs or final
  logits against vLLM at the first divergent step. The transport and recurrent
  dtype experiments rule out custom all-reduce and FP32-vs-BF16 state as the
  sole cause.
