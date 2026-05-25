# Remaining Work

Date: 2026-05-25

Target: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` on Pie CUDA native TP2.

Comparator: standalone vLLM V1 TP2 with `VLLM_USE_AOT_COMPILE=0`,
`enforce_eager=False`, CUDA graphs enabled, text-only multimodal path. This is
not Pie's vLLM driver.

## Current Status

Pie CUDA native TP2 is functional for the Nemotron-H text LLM path:

- Hybrid attention/Mamba2/MoE forward is implemented.
- TP2 weight loading and sharding work.
- Recurrent state cache is implemented.
- FlashInfer MoE is usable for the routed expert path.
- The Nemotron chat-template boundary is now fixed in runtime: Pie appends the
  HF/vLLM generation suffix `<think>\n` before generation, so Pie and vLLM are
  compared from the same prompt boundary.
- Direct deterministic decode lookahead was reverted because it fought Pie's
  programming model.
- Synthetic upfront CUDA graph capture is disabled for `nemotron_h`; first-use
  capture with real metadata is correct, but the current graph replay path is
  slower for this model.

Correctness status:

- Pie matches standalone vLLM for the deterministic 1x8 check:
  `[4268, 2534, 1317, 7418, 1261, 4958, 8303, 2314]`.
- With the corrected template, Pie 1x16 starts from the same prompt boundary and
  generates coherent tokens:
  `[4268, 2534, 1317, 7418, 1261, 4958, 8303, 2314, 1261, 23057, 1046, 1531, 3330, 25747, 1058, 1429]`.
- Longer samples are coherent, not gibberish. Pie and vLLM can diverge after the
  first several deterministic tokens because the native path is numerically not
  bit-identical to vLLM's Triton kernels.

## Latest Fair Benchmark Snapshot

All numbers below use unlimited benchmark concurrency (`--concurrency 0` for
Pie), TP2, `--gpu-mem-util 0.92`, temperature 0, ignore EOS, text-only
multimodal inputs, and corrected Nemotron prompt templating.

Pie command environment:

```bash
PIE_NEMOTRON_FLASHINFER_MOE=1 \
PIE_NEMOTRON_FLASHINFER_MOE_DECODE=1 \
python benches/pie_bench.py tput \
  --driver cuda_native \
  --device cuda:0,cuda:1 \
  --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --tp-size 2 \
  --gpu-mem-util 0.92 \
  --max-model-len 2048 \
  --concurrency 0 \
  --max-tokens 16 \
  --temperature 0 \
  --ignore-eos \
  --trust-remote-code \
  --ipc-profile latency
```

vLLM command environment:

```bash
VLLM_USE_AOT_COMPILE=0 \
/root/Workspace/pie/driver/vllm/.venv/bin/python benches/vllm_bench.py ...
```

| workload | Pie current | standalone vLLM recent | status |
| --- | ---: | ---: | --- |
| 64x16 warm64 | 2221.11 tok/s | 2795.88 tok/s fresh rerun | Pie behind, about 0.79x |
| 128x16 warm128 | 3127.69 tok/s | 3094.65 to 3341.09 tok/s observed | not a robust win; rerun matrix needed |
| 256x16 warm256 | 3548.78 tok/s | 3885.76 to 3953.99 tok/s observed | Pie behind, about 0.90x |

Current benchmark files:

- Pie 64x16:
  `.tmp/benches/nemotron_default_template_ipclat_tput_64x16_warm64.json`
- Pie 128x16:
  `.tmp/benches/nemotron_default_template_ipclat_tput_128x16_warm128.json`
- Pie 256x16:
  `.tmp/benches/nemotron_default_template_ipclat_tput_256x16_warm256.json`
- vLLM 64x16 fresh:
  `.tmp/benches/nemotron_vllm_standalone_tp2_noaot_refresh_64x16_warm64.json`
- vLLM 128x16 references:
  `.tmp/benches/nemotron_vllm_standalone_tp2_noaot_tput_128x16_warm128_recheck.json`,
  `.tmp/benches/nemotron_vllm_standalone_tp2_noaot_rerun_tput_128x16_warm128.json`
- vLLM 256x16 references:
  `.tmp/benches/nemotron_vllm_standalone_tp2_noaot_rerun_tput_256x16_warm256.json`,
  `.tmp/benches/nemotron_vllm_standalone_tp2_noaot_tput_256x16_warm256_recheck2.json`

## What Worked

- Native `nemotron_h` CUDA model binding and forward implementation.
- TP2 loader support and skipping unused multimodal weights for text-only
  native serving.
- Persistent weight arena/direct-copy loading, fixing the earlier memory/load
  behavior.
- Custom all-reduce by default when peer access is available. This was the main
  earlier latency unlock, replacing expensive NCCL reductions in the decode
  path.
- FP32 router logits and vLLM-compatible Mamba `dt_limit=(0, inf)` behavior.
- BF16 recurrent state cache, matching vLLM's effective cache dtype.
- Decode conv update path instead of reusing the prefill conv kernel.
- Device-side exact MoE route bucketing for decode, avoiding full route/weight
  device-to-host synchronization.
- Packed expert backing tensors in the loader. This is mostly foundation for a
  true fused MoE kernel, not a large standalone throughput win.
- FlashInfer MoE integration for the routed expert path.
- Correct Nemotron chat-template suffix in runtime, which fixed the benchmark
  prompt boundary.
- CUDA profiler range support in the benchmark scripts.

## What Did Not Work

Do not repeat these without a new hypothesis:

- Prefill SSM thread-count sweeps above/below the current 512-thread setting:
  384, 640, and 768 threads were slower.
- The multidimensional one-block-per-request/head prefill SSM kernel corrupted
  output and was removed.
- Synthetic upfront CUDA graph capture for `nemotron_h` is unsafe because it
  captures with fake metadata. First-use capture with real metadata is correct,
  but current graph replay is slower on 64x16 and 256x16.
- `--single-process-batch` did not improve the corrected-template 64x16 run.
- FlashInfer MoE raw/tactic selector experiments have not produced a durable
  win. Clean up unused selector code before commit unless a new profiling pass
  proves it useful.
- Small micro-optimizations around Mamba precompute and grouped-output summing
  were at best small or workload-dependent. They do not close the vLLM gap.

## Main Remaining Gap

The remaining gap is GPU-side, not request admission or fixed benchmark
concurrency.

Pie still lacks the two major kernel paths that standalone vLLM uses well for
this model:

1. A vLLM/SGLang/TensorRT-LLM style Mamba2 chunk-scan/SSU path for prefill and
   mixed prefill/decode work.
2. A true fused/grouped MoE path with persistent planning and library-grade
   kernel selection.

The current Pie path still spends too much time in native Mamba SSM and MoE
launch/GEMM/combine structure. Runtime scheduling overhead is visible in bench
stats, but the dominant time is still driver/GPU execution.

## Recommended Next Work

1. Rebaseline with three-run medians for Pie and standalone vLLM on 64x16,
   128x16, 256x16, and at least one larger workload. Keep unlimited concurrency
   on both sides and record prompt tokens so template mismatches are caught.
2. Profile one representative losing shape with Nsight Systems CUDA-only trace:
   64x16 for small-batch overhead and 256x16 for steady state. Avoid NVTX with
   vLLM if it triggers NCCL instability.
   - **2026-05-25:** Done. Pie and vLLM both traced under
     `nsys profile --trace=cuda --capture-range=cudaProfilerApi` at 64x16 and
     256x16. Reports under `.tmp/nsys/nemo_{pie,vllm}_{64,256}x16_warm{64,256}_cudaonly.nsys-rep`.
     Hotspot comparison written up in
     `benches/nemotron_h_omni_perf.md` under
     "Nsight Systems CUDA-only hotspot comparison (2026-05-25)". Headline:
     Mamba SSM is the dominant remaining GPU-side gap (Pie 23.5% vs vLLM
     6.2% at 256x16); MoE GEMM is roughly comparable; custom all-reduce is
     a Pie win. The single 256x16 run had Pie ahead of vLLM, contradicting
     the fair-bench snapshot — argues for item 1's three-run medians before
     re-stating the steady-state gap.
3. Implement the Mamba2 chunk-scan/SSU path in a maintainable way. Preferred
   direction is a narrow adapter to an existing proven implementation
   (FlashInfer, SGLang/vLLM Triton logic, or TensorRT-LLM-style kernels), with
   explicit layout conversion only where unavoidable.
4. Implement true fused/grouped MoE through a real library integration instead
   of copying a large kernel stack as header-only code. FlashInfer or
   TensorRT-LLM style MoE needs compiled kernels, descriptors, workspace/tactic
   selection, and stable packed expert layouts.
5. Remove or hide unused experimental knobs before committing:
   raw FlashInfer MoE selector paths, failed graph toggles, and any temporary
   SSM tuning hooks that are no longer used.
6. Keep the benchmark profiler-range helper and the Nemotron template fix.
7. After any kernel change, run:
   - 1x8 deterministic token check against vLLM.
   - 1x16 corrected-template smoke for coherence.
   - 64x16, 128x16, and 256x16 throughput with unlimited concurrency.

## Cleanup Before Commit

- Review `driver/cuda/src/ops/flashinfer_moe.cu` and remove unused raw selector
  code if it remains unproven.
- Keep the explicit `nemotron_h` instruct template case and
  `generation_suffix` plumbing.
- Keep the `nemotron_h` upfront graph skip unless a real-metadata warmup graph
  path is implemented and benchmarked.
- Make sure `benches/nemotron_h_omni_perf.md` and this file agree on the final
  numbers before committing.
- Check for unrelated dirty worktree changes before staging. The repository has
  many pre-existing modified files; do not revert unrelated changes.
