# Gemma4 MTP Remaining Work

Status as of 2026-05-25 on branch `agents/lima`.

## Implemented

- Gemma4 native MTP is wired through the system speculator path for greedy decoding when the assistant checkpoint is available and `mtp_num_drafts > 0`.
- Gemma4 MTP assistant auto-discovery works for `google/gemma-4-E4B-it` via the paired `google/gemma-4-E4B-it-assistant` snapshot.
- Gemma4 and Gemma3 instruct templates were split into their own implementations, with Gemma4 emitting BOS exactly once for first-user and system-user paths.
- System speculation and pass speculation remain separate concepts. Current benchmark runs keep pass speculation enabled via `--speculation-depth 1` or `2`.
- The runtime/generator path now handles rejected draft truncation and repaired speculative tails correctly.
- Batch benchmark harness supports pretokenized HF prompts and batch concurrency so Pie and vLLM can be compared on the same token IDs.
- CUDA build was verified with:

```bash
cargo build -p pie-server --release --no-default-features --features driver-cuda
```

- Graph-captured greedy argmax is now the default (previously gated behind `PIE_CUDA_GRAPH_ARGMAX=1`). Opt-out with `PIE_CUDA_GRAPH_ARGMAX=0`.
- Fused tiled lm_head-argmax path for Gemma4 greedy verification. When `logits_argmax_only`, the forward tiles the vocab into 8 chunks, runs cuBLAS GEMM + per-tile argmax into packed pairs, and selects the global winner — never materializing the full `[rows, vocab]` logits matrix. Activates automatically for single-GPU greedy decode. New kernel: `launch_argmax_bf16_tile_pair` in `argmax.cu`. New ForwardFn fields: `supports_fused_lmhead_argmax`, `set_fused_argmax_output`, `fused_argmax_done`.

## Correctness Verified

Benchmark prompt:

```text
system: You are a helpful benchmarking assistant.
user: Write a short story about a robot. (Request #0)
```

HF-pretokenized prompt length is 34 tokens.

- Pie no-spec, 64 generated tokens: output hash `c2526a7a39534fdd`.
- Pie Gemma4 MTP d3, 64 generated tokens: output hash `c2526a7a39534fdd`.
- vLLM no-spec with the same tokenized prompt produced the same 64-token text.

This confirms the target Gemma4 path and the MTP speculative path preserve greedy output for the benchmark prompt. An earlier mismatch was caused by comparing against a different vLLM prompt/template, not by a target-model bug.

## Current Performance

All throughput numbers below use 64 requests, max 128 output tokens, temperature 0, top_p 1, ignore_eos, HF-pretokenized prompts, MTP d3, and `google/gemma-4-E4B-it`.

| Shape | Engine/config | Output tok/s | Notes |
| --- | --- | ---: | --- |
| c8 | vLLM MTP d3 (v0.21.0) | 1149.75 | accepted 3722 / proposed 13389, per-pos [2337,1002,383] |
| c8 | Pie MTP d3 (current, 5-run avg) | 973.4 | accepted ~24.4%, graph argmax + per-step positions. Range: 963-982 |
| c8 | Pie MTP d3 (prev baseline) | 938.19 | accepted 3521 / proposed 14703 |
| c8 | Pie MTP d2 | 988.64 | accepted 3212 / proposed 10424, sweet spot for c8 batch economics |
| c16 | vLLM MTP d3 (v0.21.0) | 2114.34 | vLLM scales better at higher concurrency |
| c16 | Pie MTP d3 | 1594.12 | acceptance 25.4%, gap widens at c16 |
| single request | Pie no-spec | 103.43 | 64-token correctness run |
| single request | Pie MTP d3 | 157.74 | same output hash, accepted 31 / proposed 96 |

Profiling summary (A100 80GB PCIe):

- Steady c8 verifier shape is `N=32, R=8` and costs about 10.3 ms with CUDA graph.
- Forward breakdown (profiled, no graph, N=32 R=8): MLP 5.77ms (42%), attn_prep 1.36ms (10%), attn_out 1.33ms (10%), PLE residual 1.60ms (12%), attention 0.68ms (5%), lm_head 0.86ms (6%), other_gpu 2.01ms (profiling overhead).
- Gemma4 MTP drafter for `R=8` costs about 1.2 ms total (GPU 1.0ms + host 0.2ms).
- MTP GPU breakdown (profiled, no graph): MLP 0.46ms, score+argmax 0.41ms, q+attention+attn_out 0.80ms, pre/post_projection 0.10ms, other 0.2ms.
- c16 roughly doubles verifier rows for only modestly more time, which is why c16 is competitive while c8 is not.
- Per-batch time is comparable between Pie (12.4ms) and vLLM (~12.8ms). The throughput gap is ~10% from acceptance rate (Pie 24.4% vs vLLM 27.8%) and ~5% from per-batch host overhead.
- If Pie matched vLLM's acceptance rate, projected throughput would be ~1100+ tok/s (close to or beating vLLM).

Draft count sweep at c8:

| Drafts | tok/s | batch_ms | batches | acceptance |
| --- | ---: | ---: | ---: | --- |
| d1 | 930.9 | 11.4 | 752 | 45.2% |
| d2 | 988.6 | 11.8 | 676 | 30.8% |
| d3 | 996.8 | 12.3 | 640 | 23.6% |
| d4 | 695.6 | 17.7 | 636 | 18.7% |

Experiments tried with no significant improvement:
- Fused tiled lm_head-argmax: 8 small cuBLAS GEMMs slower than 1 large GEMM for M=32 (gated behind `PIE_FUSED_LMHEAD_ARGMAX=1`, off by default).
- cuBLAS GEMMEx for row-decode O/Q/K/V/down projections: same speed as cuBLASLt.
- `PIE_GEMMA4_MTP_TARGET_RESIDUAL_HIDDEN=1` (pre-norm hidden state): acceptance drops to 10.8%, confirming post-norm is correct.
- speculation-depth 2: no improvement when MTP is active.
- `PIE_GEMMA4_SPEC_ROW_DECODE=0` (naive paged attention instead of row-decode): same acceptance (24.5%), confirms row-decode attention is not the acceptance gap cause.
- Position offset=0 (matching HF reference's `input_ids.shape[1]-1`): worse throughput (959 vs 1006 tok/s). offset=1 with per-step increment is the best.
- Adaptive drafts (`PIE_GEMMA4_MTP_ADAPTIVE_DRAFTS=1`): slower (930 tok/s), reduced verify batch efficiency.
- HF reference analysis: confirmed embed scaling (`sqrt(backbone_hidden_size)`), concat order (`[embed, hidden]`), and shared embedding all match.
- Greedy scheduler (`--batch-policy greedy`): 568 tok/s — much worse, serializes requests.
- MTP CUDA graph disabled (`PIE_GEMMA4_MTP_CUDA_GRAPH=0`): same acceptance, ~40 tok/s slower.
- vLLM upgraded to v0.21.0: c16 jumped from 1654 to 2114 tok/s. vLLM scales better at higher concurrency.
- FP32 lm_head output (BF16 weights → FP32 logits → FP32 argmax): acceptance DROPPED 2pp (22.7% vs 24.7%). Mismatching target and MTP precision hurts consistency.
- `PIE_CUBLAS_PRECISE=1` (CUBLAS_COMPUTE_32F instead of CUBLAS_COMPUTE_32F_FAST_16BF): acceptance unchanged (+0.1pp noise), throughput -5.7%. The gap is NOT from TF32 vs FP32 accumulation.
- Root cause identified: different cuBLAS/CUDA versions between Pie (12.7/cuBLAS 12.8.4) and vLLM (CUDA 12.9 via PyTorch 2.11.0). Different cuBLAS versions select different GEMM algorithms for the same shapes → different accumulation order → different BF16 rounding → different argmax for near-tie logits at ~7% of pos-0 verifications.

## Remaining Work

1. ~~Make graph-captured greedy argmax production-ready.~~ **Done.**

   Graph-captured greedy argmax is now the default (`PIE_CUDA_GRAPH_ARGMAX` defaults to enabled; set `PIE_CUDA_GRAPH_ARGMAX=0` to disable). Correctness sweep should be re-run across no-spec, MTP d1/d2/d3, c1/c4/c8/c16, and non-greedy/rich sampler cases to confirm.

2. ~~Avoid full-vocab logits materialization for greedy Gemma4 verification.~~ **Done.**

   The Gemma4 target forward now has a fused tiled lm_head-argmax path. When `logits_argmax_only` is true and the executor provides a fused argmax output pointer (`supports_fused_lmhead_argmax`), the forward tiles the vocab dimension into 8 chunks, computes a small GEMM per tile (keeping tile logits in L2), runs per-tile argmax into packed pairs via `launch_argmax_bf16_tile_pair`, and selects the global winner via `launch_select_global_argmax_pairs`. The full `[rows, vocab]` logits are never materialized. The path activates automatically for single-GPU greedy decode. Benchmark at c8 to measure the improvement.

3. **[HIGH PRIORITY] Close the acceptance rate gap against vLLM.**

   The c8 throughput gap is almost entirely from acceptance rate (Pie 23.6% vs vLLM 27.8%), not per-batch speed. Per-position data from a trace sample:
   - Pie: pos0=46%, pos1=29%, pos2=16%, mean=0.91
   - vLLM: pos0=52.4%, pos1=22.5%, pos2=8.6%, mean=0.83

   Pie's position 0 acceptance is lower (46% vs 52.4%), but positions 1-2 are actually higher. The aggregate still favors vLLM due to more total verify steps. Root cause is likely BF16 numerical precision differences in cuBLAS GEMM dispatch between the MTP drafter and target model. Matching vLLM's acceptance would yield ~1194 tok/s.

   Investigation completed: the gap is from cuBLAS version differences (Pie CUDA 12.7/cuBLAS 12.8.4 vs vLLM CUDA 12.9). Different cuBLAS versions select different GEMM algorithms. FP32 accumulation and FP32 lm_head output were both tried and don't help — the gap is algorithm-selection-based, not precision-based.

   Next steps:
   - Upgrade Pie's CUDA toolkit to 12.9 to match vLLM's cuBLAS version
   - Or: implement CUTLASS GEMMs to bypass cuBLAS algorithm selection entirely
   - Or: add cuBLASLt algorithm autotuning to select the most self-consistent algorithm

4. Reduce host/device round trips in `system_speculator()`.

   The system_speculator adds ~1.2ms per batch (1.0ms GPU + 0.2ms host). The GPU work is already CUDA graph captured. The host overhead is from FlashInfer plan building (~0.01ms), metadata setup (~0.002ms), upload (~0.03ms), and D2H sync. Consider migrating to the `forward_fn.mtp` path (like Qwen3.5) for tighter executor integration.

5. Improve small-batch verifier utilization.

   c8 is dominated by target verification at `N=32,R=8`. The MLP at 42% of forward time reads ~6.3GB of weights (150MB/layer × 42 layers) at M=32. Theoretical HBM bandwidth minimum is ~3.15ms; measured is ~5ms. The gap is from cuBLAS kernel efficiency at M=32 and inter-kernel overhead. c16 (M=64) reaches near-bandwidth efficiency.

6. Keep benchmarking apples-to-apples.

   Use HF-pretokenized prompts for both engines, same `mtp_num_drafts`, same max tokens, same ignore_eos behavior. Pass speculation (`--speculation-depth`) has negligible impact when MTP is active.

## Useful Commands

Pie c8 MTP d3:

```bash
env PYTHONPATH=/root/Workspace/pie/sdk/python-server/python:/root/Workspace/pie/client/python/src \
  PIE_BENCH_SERVER_LOG=.tmp/bench-pie-mtp-d3-c8.server.log \
  python benches/pie_bench.py tput \
  --model google/gemma-4-E4B-it --driver cuda_native \
  --num-requests 64 --concurrency 8 \
  --warmup 8 --warmup-max-tokens 32 \
  --max-tokens 128 --temperature 0 --top-p 1 --ignore-eos \
  --gpu-mem-util 0.85 --max-model-len 2048 \
  --mtp-num-drafts 3 --speculation-depth 1 --pretokenized-prompts \
  --json-out .tmp/bench-pie-mtp-d3-c8.json
```

Pie c8 with verifier graph argmax experiment:

```bash
env PIE_CUDA_GRAPH_ARGMAX=1 \
  PYTHONPATH=/root/Workspace/pie/sdk/python-server/python:/root/Workspace/pie/client/python/src \
  PIE_BENCH_SERVER_LOG=.tmp/bench-pie-mtp-d3-c8-graphargmax.server.log \
  python benches/pie_bench.py tput \
  --model google/gemma-4-E4B-it --driver cuda_native \
  --num-requests 64 --concurrency 8 \
  --warmup 8 --warmup-max-tokens 32 \
  --max-tokens 128 --temperature 0 --top-p 1 --ignore-eos \
  --gpu-mem-util 0.85 --max-model-len 2048 \
  --mtp-num-drafts 3 --speculation-depth 1 --pretokenized-prompts \
  --json-out .tmp/bench-pie-mtp-d3-c8-graphargmax.json
```

vLLM c8 MTP d3:

```bash
/root/Workspace/pie/driver/vllm/.venv/bin/python benches/vllm_bench.py tput \
  --model google/gemma-4-E4B-it \
  --num-requests 64 --concurrency 8 \
  --warmup 8 --warmup-max-tokens 32 \
  --max-tokens 128 --temperature 0 --top-p 1 --ignore-eos \
  --gpu-mem-util 0.85 --max-model-len 2048 \
  --mtp-assistant-model google/gemma-4-E4B-it-assistant \
  --mtp-method gemma4_mtp --mtp-num-drafts 3 \
  --json-out .tmp/bench-vllm-mtp-d3-c8.json
```
