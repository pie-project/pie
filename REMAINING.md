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
| c8 | vLLM MTP d3 | 1136.56 | accepted 3678 / proposed 13518 |
| c8 | Pie MTP d3 | 938.19 | accepted 3521 / proposed 14703 |
| c8 | Pie MTP d3 + `PIE_CUDA_GRAPH_ARGMAX=1` | 984.84 to 998.35 | env-gated verifier argmax capture helps, still behind vLLM |
| c16 | vLLM MTP d3 | 1654.34 | previous reference run |
| c16 | Pie MTP d3 | 1668.00 | slightly ahead of vLLM on this shape |
| single request | Pie no-spec | 103.43 | 64-token correctness run |
| single request | Pie MTP d3 | 157.74 | same output hash, accepted 31 / proposed 96 |

Profiling summary:

- Steady c8 verifier shape is `N=32, R=8` and costs about 9.7 to 10.0 ms.
- Gemma4 MTP drafter for `R=8` costs about 1.18 to 1.25 ms with CUDA graph replay.
- c16 roughly doubles verifier rows for only modestly more time, which is why c16 is competitive while c8 is not.
- Ordered/sparse MTP embeddings were tested and are currently much slower for this model/shape.
- Disabling row-decode verification is slower. Disabling packed QKV or gate/up fusion did not produce a reliable improvement.

## Remaining Work

1. Make graph-captured greedy argmax production-ready.

   `PIE_CUDA_GRAPH_ARGMAX=1` improved c8 from about 938 tok/s to about 985 to 998 tok/s. It should be made the default only after a correctness sweep across no-spec, MTP d1/d2/d3, c1/c4/c8/c16, and non-greedy/rich sampler cases.

2. Avoid full-vocab logits materialization for greedy Gemma4 verification.

   The Gemma4 target forward currently skips final softcap in argmax-only mode but still computes the full `[rows, vocab]` lm_head matrix, then runs argmax. For speculative verification, the caller only needs greedy token IDs. A direct lm_head-argmax path is the most likely large c8 win.

3. Reduce host/device round trips in `system_speculator()`.

   Gemma4 MTP still builds host request metadata, uploads it, runs drafts, copies draft tokens back to host, then builds response spec buffers. This is semantically correct but overhead-heavy. A persistent device-side draft output layout, or direct emission into common response-builder buffers, should reduce per-step overhead.

4. Improve small-batch verifier utilization.

   c8 is dominated by target verification at `N=32,R=8`. c16 shows the GPU has much better utilization at larger verifier batches. Future work should focus on the row-decode verifier path, fixed dense verifier shapes, and vLLM-style attention/kernel behavior for these small speculative blocks.

5. Track the acceptance delta against vLLM.

   Pie and vLLM are close but not identical on acceptance for the same d3 workload: vLLM c8 accepted 3678 drafts, while current Pie accepted about 3496 to 3521. This is not a correctness issue, but improving numerical alignment or exact MTP hidden-state semantics could recover some throughput.

6. Keep benchmarking apples-to-apples.

   Use HF-pretokenized prompts for both engines, same `mtp_num_drafts`, same max tokens, same ignore_eos behavior, and pass speculation enabled for Pie. Re-run at least c1/c4/c8/c16 after every kernel or scheduler change.

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
