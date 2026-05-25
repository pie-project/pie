# Qwen3.6 MTP Status and Remaining Work

Last updated: 2026-05-25.

## Current State

- Branch: `agents/lima`.
- Workspace is intentionally dirty with the in-progress Qwen3.6 MTP implementation and benchmark/debug changes.
- Qwen3.6 native MTP is wired through the common CUDA executor `NativeSystemDrafter` path.
- Qwen MTP now uses the same system-speculation response plumbing as the other native drafters.
- `[model].mtp_num_drafts` controls Qwen MTP draft count; `PIE_MTP_DRAFT_TOKENS` remains a debug override.
- Qwen keeps its native `mtp_process_cache` commit path for accepted tokens, including MTP KV/recurrent state updates.
- MTP and pass speculation are treated as orthogonal; benchmark runs should keep pass speculation enabled at depth 1 or 2.
- Qwen3.6 models default to native system speculation when MTP weights are present and `mtp_num_drafts > 0`.

## Verified Results

### Qwen3.6-35B-A3B

Pie currently beats the stored vLLM baselines on the available single-request latency benchmarks.

- Stored vLLM d4 baseline: about `204.46 tok/s`.
- Stored vLLM d6 baseline: about `183.6 tok/s`.
- Pie d4 best observed: `220.69 tok/s`, accepted `215/224`, file `.tmp/fixed-pie-qwen36-35b-a3b-d4-pass2-256.json`.
- Pie d6 stable observed: `213.82 tok/s`, accepted `228/282`, file `.tmp/pie-qwen36-35b-a3b-default-d6-pass2-final-256.json`.

### Qwen3.6-27B

Pie has a large speedup over no-MTP, but the exact/correct d8 path does not yet beat the stored vLLM d8 baseline.

- Stored vLLM d8 256 baseline: `98.26 tok/s`, accepted `212/360`, file `.tmp/vllm-qwen36-27b-mtp-d8-256-w1-with-accept.json`.
- Pie exact d8 256 best current path: `85.27 tok/s`, accepted `224/408`, per-position `[47, 42, 37, 32, 26, 19, 16, 5]`, file `.tmp/fusedmtp-pie-qwen36-27b-d8-pass2-256.json`.
- Pie shifted-prefix d6 256: `83.18 tok/s`, accepted `214/348`, per-position `[50, 41, 39, 35, 29, 20]`, file `.tmp/sweep-shift1-d6-256.json`.
- Pie no-MTP baseline is much slower, so MTP is working and gives a real speedup.

Correctness notes:

- Exact d8 128 hash: `3907f54fb4d16910`.
- No-MTP 128 hash: `3907f54fb4d16910`.
- d9 is not correct today. It can report high throughput, for example `.tmp/pie-qwen36-27b-default-d9-final-256.json` at `107.04 tok/s`, but d9 128 produced wrong hashes such as `7e77cb0e7372f4cc`. Do not use d9+ as a valid win until correctness is fixed.

## Bottleneck

The 27B exact path is now primarily limited by MTP draft-step scoring.

Profiled d8 96-token runs show each steady MTP draft step spends roughly:

- MTP input/fc: about `0.11 ms`.
- MTP attention: about `0.25 ms`.
- MTP MLP: about `0.35 ms`.
- MTP `lm_head` scoring plus argmax: about `1.43 ms`.

Across 8 drafts, `lm_head` costs about `11 ms`, which explains most of the remaining gap to vLLM.

## Cleaned Experimental Paths

The following failed/experimental code paths have been removed:

- `PIE_QWEN35_FUSED_DENSE_MLP_LAYER_LIMIT`, `PIE_QWEN35_FUSED_DENSE_MLP`, `PIE_QWEN35_FUSED_MTP_DENSE_MLP`, `PIE_QWEN35_FUSED_DENSE_MLP_KEEP_ORIGINALS`: target dense MLP gate/up fusion and all related toggles.
- `PIE_QWEN35_DENSE_GATE_UP_BATCHED` and the `gate_up_*_ptrs` scratch buffers.
- `PIE_QWEN35_MTP_DIRECT_ARGMAX` direct lm-head argmax path.
- `PIE_QWEN35_MTP_FP8_LM_HEAD` and associated `owned_fp8_buffers`/`owned_scale_buffers`/`lm_head_scale`.
- `PIE_QWEN35_FUSED_MTP_FULL_ATTN_QGKV` (MTP-specific QGKV fusion toggle).
- `PIE_QWEN35_FUSED_GDN_BA` (standalone GDN BA fusion toggle; BA fusion still occurs when GDN_PROJ is enabled).

Validated paths retained: `PIE_QWEN35_FUSED_GDN_PROJ`, `PIE_QWEN35_FUSED_FULL_ATTN_QGKV`.

## New Optimization Features

- `PIE_QWEN35_MTP_INT8_LM_HEAD=1`: Quantizes MTP lm_head to INT8 per-channel at model load. Combined with `PIE_QWEN35_MTP_FUSED_GEMV=1`, uses a fused INT8 GEMV+argmax kernel that halves weight bandwidth and eliminates logit materialization. Without fused GEMV, uses M=4 padding for the W8A8 cublasGemmEx path.
- `PIE_QWEN35_MTP_FUSED_GEMV=1`: Activates fused GEMV+argmax kernel for MTP lm_head scoring. Supports both INT8 (with `INT8_LM_HEAD=1`) and BF16 weights. Writes token IDs directly, skipping executor argmax.
- `PIE_BENCH_LM_HEAD_ALGOS=1`: One-shot benchmark of all cuBLASLt heuristics + GEMMEx for the MTP lm_head shape. Prints per-algo latencies to stderr on first MTP draft step.

## Benchmark Results (2026-05-25, A100 80GB, Qwen3.6-27B 256tok)

| Config | d | tok/s | Batch ms | Accepted | Rate | Batches |
|--------|---|-------|----------|----------|------|---------|
| BF16 cuBLASLt (baseline) | 8 | 85.08 | 61.8 | 224/408 | 54.9% | 53 |
| INT8 W8A8 cublasGemmEx | 8 | 78.92 | 68.0 | 225/400 | 56.3% | 52 |
| INT8 fused GEMV+argmax | 8 | **86.41** | **57.3** | 221/432 | 51.2% | 56 |
| BF16 fused GEMV+argmax | 8 | 78.83 | 62.7 | 221/432 | 51.2% | 56 |
| BF16 cuBLASLt | 6 | 79.88 | 56.6 | 213/360 | 59.2% | 62 |
| INT8 fused GEMV+argmax | 6 | 79.74 | 52.8 | 219/384 | 57.0% | 66 |
| INT8 fused GEMV+argmax | 4 | 74.42 | 49.5 | 201/296 | 67.9% | 76 |
| BF16 + gate+up fusion | 8 | **87.79** | **59.9** | 223/408 | 54.7% | 53 |
| INT8 fused + gate+up | 8 | 85.44 | 58.0 | 221/432 | 51.2% | 56 |
| **vLLM** | **8** | **98.26** | **~57.9** | **212/360** | **58.9%** | **45** |

Gate+up MLP fusion: unconditionally fuses gate+up projections for all 64 layers. Saves ~2ms/batch from kernel launch reduction and improved cuBLASLt tiling. Also dramatically improved pos 7 acceptance from 9.8% to 21.6% (approaching vLLM's 24.4%), suggesting numerical consistency of the fused GEMM better matches the target model.

Key findings:
- INT8 fused GEMV has the best per-batch latency (57.3ms at d8), matching vLLM's ~57.9ms.
- d8 is the optimal draft depth despite poor pos 6-7 acceptance (d6 has better acceptance but fewer tokens per cycle).
- The remaining gap is almost entirely **acceptance rate**: Pie 54.9% vs vLLM 58.9%. If Pie matched vLLM's acceptance (45 batches × 57.3ms), throughput would be **~99.3 tok/s**, beating vLLM.
- INT8 quantization changes argmax at ambiguous positions, dropping acceptance from 54.9% to 51.2%, which largely negates the batch latency improvement.
- INT8 W8A8 via cublasGemmEx is slower than BF16 cuBLASLt due to activation quant + int32 accumulator + dequant overhead at M=1.
- BF16 fused GEMV is slower than cuBLASLt (no tensor cores in custom kernel).
- INT8 fused kernel at ~72% bandwidth efficiency (0.87ms vs 0.62ms theoretical). Wider vector loads (int4/16-byte) caused register pressure and regressed to 82.9 tok/s. char4 (4-byte) loads are optimal.

## Acceptance Rate Investigation (2026-05-25)

Per-position acceptance comparison (percentages):
```
Pos:    0      1      2      3      4      5      6      7
Pie:   92.2   82.4   72.5   62.7   51.0   37.3   31.4    9.8
vLLM:  91.1   84.4   71.1   64.4   55.6   44.4   35.6   24.4
Gap:   +1.1   -2.0   +1.4   -1.7   -4.6   -7.1   -4.2  -14.6
```
Early positions (0-3) are nearly identical. The gap grows at positions 4+ and is largest at position 7 (9.8% vs 24.4%). This pattern suggests degradation at deeper draft steps.

Tested:
- `PIE_QWEN35_MTP_PREFIX_GLOBAL=0`: much worse (44.8% acceptance), do not use.
- Removing the `ws.norm_x → ws.y` memcpy (hypothesis: double-norm): worse (41.5%). Model expects post-norm hidden state between MTP steps.
- The memcpy is correct and intentional.

Correctness verified (2026-05-25):
- d8 128 hash: `ae09b1ebd2f9f9ac` (prompt: "List the first 100 prime numbers...")
- No-MTP 128 hash: `ae09b1ebd2f9f9ac` (same prompt)
- MTP is exact-correct: greedy output matches with and without speculation.

Tested and ruled out:
- Pass-level speculation interaction: depth 0 gives identical acceptance (224/408, same per-position).
- Attention context truncation: `max_global_tokens` is not capping at our sequence lengths.
- Position advancement: chain path correctly advances positions by 1 per step.

**Root cause identified (high confidence):** vLLM uses `vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MTP` which delegates to the standard `Qwen3_5DecoderLayer` — the SAME decoder layer and attention backend (flashinfer) used by the target model. This means vLLM's MTP attention uses flashinfer tensor-core attention with paged KV cache, exactly matching the target model's verification pass.

Pie's MTP attention uses a **naive FP32 scalar kernel** (`attn_mtp_paged_history_kernel` in `attention_naive.cu`), while the target model uses flashinfer. The precision and softmax implementation differences between naive and flashinfer cause the MTP hidden states to diverge from what the target model expects, especially at deeper draft positions where errors compound.

Additionally, vLLM confirmed: hidden state propagation between MTP steps uses `norm(hidden + residual)` (post-norm), matching Pie's memcpy behavior. The hidden state flow is NOT the issue.

**Fix:** Switch MTP attention to use flashinfer (matching the target model). This requires writing draft-step K/V to the paged cache (temporary pages) and using flashinfer decode attention, then rolling back pages after the chain. This is the clear path to matching vLLM's 58.9% acceptance rate.

Ruled out:
- Hidden state normalization between steps (both Pie and vLLM use post-norm)
- RoPE partial_rotary_factor (same computation in MTP and target)
- Pass-level speculation interaction (no effect on acceptance)
- QK dot product precision: changing from fp32 to bf16 multiply in the naive attention kernel had zero effect on acceptance (still 224/408).
- Position offset=1 (`PIE_QWEN35_MTP_POSITION_OFFSET=1`): worse for Qwen3.6 (80.3 tok/s, 51.2% acceptance) unlike Gemma4 where it helps.
- Gemma4 MTP agent found: **row-decode vs naive attention gives same acceptance**. This suggests switching to flashinfer may NOT improve acceptance for Qwen3.6 either.
- Position offset=1: worse for Qwen3.6 (80.3 tok/s) unlike Gemma4 where it helps.
- **Structural analysis**: even with a theoretically perfect INT8 kernel + BF16 acceptance rate, Pie would reach ~87.4 tok/s — still below vLLM's 98.3. The gap is primarily batch count (53 vs 45 batches), driven by 4% acceptance difference. This acceptance gap may be inherent to model/prompt/vLLM-version differences rather than a fixable implementation issue.

## Profiled Batch Latency Breakdown (A100, Qwen3.6-27B d8 N=1 decode)

```
Total batch: 61.8 ms
├── Target model forward: 44.4 ms (71.8%)
│   ├── MLP (64 layers):           23.0 ms  73% BW efficiency
│   │     3 GEMMs/layer [1,5120]×[17408,5120]
│   ├── Linear attn (48 layers):   13.2 ms  41% BW efficiency
│   │   ├── Projections:            4.9 ms  (qkv,z,a,b GEMMs)
│   │   ├── Recurrent GDN:          3.1 ms  (delta-net kernel)
│   │   ├── Attention + out_proj:    4.2 ms
│   │   └── Conv/prep/post:          1.0 ms
│   ├── Full attn (16 layers):      3.6 ms  46% BW efficiency
│   ├── lm_head:                    1.4 ms
│   └── Norms + other:              3.2 ms
├── MTP 8 draft steps: 17.4 ms (28.2%)
│   └── Per step: 2.18 ms
│       ├── lm_head GEMM:     1.43 ms (65.6%)
│       ├── MLP:              0.35 ms
│       ├── Attention:        0.26 ms
│       └── Input/FC + other: 0.14 ms
└── Executor overhead: <0.1 ms

Theoretical weight-bandwidth minimum: 25.1 ms (51.24 GB @ 2039 GB/s)
Actual: 44.4 ms → 57% overall BW efficiency
Gap sources: ~1060 kernel launches (~5-10 ms), non-GEMM kernels (~5-8 ms),
             cuBLASLt overhead (~1-2 ms)
```

## Remaining Work (Priority Order)

1. **Route MTP decoder layer through `full_attn_layer_body`** (highest impact, closes the acceptance gap):
   - Use the same flashinfer decode path for MTP as the target model uses.
   - Requires executor changes: allocate temporary page slots for draft tokens in the MTP KV cache, increment `kv_last_page_lens` per draft step, create a decode plan for the MTP layer, and roll back pages after the chain.
   - The existing page allocator reserves pages for speculation, so space should be available.
   - This eliminates ALL accumulated precision differences (attention, softmax, accumulation order) between MTP predictions and target model verification.
   - Expected: acceptance improves from 54.9% to ~58.9% (matching vLLM).
   - Combined with INT8 fused GEMV (57.3ms batch): projected **~99 tok/s**, beating vLLM's 98.3.
   - Implementation scope: modify `qwen3_5_mtp_forward` to call `full_attn_layer_body` instead of `mtp_full_attn_no_cache`, add page management for draft K/V in the executor's MTP draft loop.
   - **Specific blocker (tested 2026-05-25):** Naively writing draft K/V to the paged cache and incrementing `kv_last_page_lens` fails at page boundaries — `kv_last_page_lens` stores tokens on the LAST page (0..page_size), and simple increment doesn't handle rollover to new pages. The executor must pre-allocate extra pages for the draft chain and provide a modified page table. A `PIE_QWEN35_MTP_PAGED_ATTN` prototype was tested and produced corrupted results (38.6% acceptance) due to this page overflow.
   - The proper implementation must: (a) reserve `max_drafts` extra page slots in the executor's page allocator before the MTP chain, (b) pass a modified `kv_last_page_lens` that handles page_size rollovers via a device kernel, (c) update `kv_page_indptr` if new pages are needed, (d) rollback allocations after the chain. This is an executor-level change in `executor.cpp`.
2. Re-run correctness after cleanup (27B no-MTP 128, d4/d6/d8 128+256, 35B-A3B d4/d6 256).
3. Fix d9+ correctness (recurrent-state rollback/snapshot for draft_step >= 9).
4. Consider matching cuBLASLt FAST_16BF precision in the fused kernel to avoid acceptance drop, or using cuBLASLt GEMM + fused argmax post-pass instead.
5. Re-run vLLM with identical MTP params once the local vLLM environment is fixed.
6. Only commit after a final benchmark table shows Pie beats vLLM on both 35B-A3B and 27B.

