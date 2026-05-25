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

## Known Bad or Experimental Paths

Clean these before committing unless they are fixed and re-verified:

- `PIE_QWEN35_FUSED_DENSE_MLP_LAYER_LIMIT` in `driver/cuda/src/model/qwen3_5.cpp`.
- `PIE_QWEN35_DENSE_GATE_UP_BATCHED` and the `gate_up_*_ptrs` scratch buffers in `driver/cuda/src/model/qwen3_forward.*`.
- `PIE_QWEN35_MTP_DIRECT_ARGMAX` direct lm-head argmax path. It is off by default and was slower than cuBLASLt GEMM plus argmax.
- `PIE_QWEN35_MTP_FP8_LM_HEAD`. On A100 it hit the FP8 fallback path and failed CUDA graph capture because the fallback synchronizes during stream capture.
- Unsafe fusion toggles should remain off unless reworked: MTP full-attention QGKV fusion, GDN BA fusion, and target dense MLP fusion.

## Remaining Work

1. Clean the failed experimental code paths listed above so the final diff contains only the validated implementation.
2. Re-run correctness after cleanup:
   - 27B no-MTP 128.
   - 27B d4/d6/d8 128 and 256.
   - 35B-A3B d4/d6 256.
3. Fix d9+ correctness before treating draft sizes greater than 8 as supported. The likely area is speculative recurrent-state rollback/snapshot or MTP-state repair after partial acceptance.
4. Optimize the 27B MTP `lm_head` scoring path. The current direct argmax kernel launches too many tiny blocks and is slower than full GEMM. A useful replacement needs to beat cuBLASLt for `M=1`, `N=~248k`, `K=5120`, and return only greedy ids.
5. After scorer optimization, re-measure acceptance and speed for draft sizes 4, 6, 8, and 9+ with pass speculation enabled.
6. Re-run vLLM with identical MTP params once the local vLLM environment is fixed. Existing vLLM runs are stored baselines because the current vLLM venv fails CUDA init with the installed driver/CUDA combination.
7. Only commit after the cleanup and a final benchmark table show:
   - Qwen3.6-35B-A3B still beats vLLM.
   - Qwen3.6-27B exact path beats the stored or re-run vLLM baseline.
   - MTP remains default for Qwen3.6.
   - pass speculation remains enabled and independent of MTP.

