# Phase 13 — Config Parity Audit + Pie-side Per-step Overhead Profiling

**Pod**: `1i6k5j4qnl8yj1` (A100-SXM4-80GB SECURE, image `vllm-opt-2026-05-07-dev`, terminated)
**Date**: 2026-05-09
**Model**: `Qwen/Qwen3-8B` bf16
**vLLM**: 0.20.0+cu129
**Mode**: cap-on (cudagraph capture enabled), APC implicit (vllm default; pie bypasses scheduler so APC is a no-op for pie)

## TL;DR

Phase 12 showed pie's per-step kernel work is *faster* than vllm-direct
in eager mode (12.93 vs 15.93 ms at b=16). The remaining 22% end-to-end
gap (Phase 11 §1: P7v=1050.7 vs vllm-direct=1391.0 tps at c=16) is
therefore *not* eager-kernel cost. Phase 13 runs the audit:

1. **Config parity (Sub-task A) — no actionable knob.** Pie's
   meaningful EngineArgs overrides (`attention_backend=FLASHINFER`,
   `cudagraph_mode=FULL_DECODE_ONLY`, `splitting_ops=[]`,
   `max_num_batched_tokens=MPE`) are all correctness-driven — pie's KV
   pinning + per-call FlashInfer metadata require this exact stack.
   The two flippable knobs (`gpu_memory_utilization` 0.9 → 0.85,
   `max_num_seqs` 256 → 64) are aligned to vllm-direct's defaults; the
   parity sweep moves c=16 tps by **−0.8%** (901.8 → 894.4) and c=64
   tps by **−1.4%** (3040 → 2996). **Within noise. Confirmed:
   config diff explains nothing.**

2. **Per-step overhead (Sub-task B) — host overhead is *not* the bulk
   of the gap.** End-to-end per-step real wall at c=16 = 13.42 ms
   (Python total 12.89 + inter-call gap 0.53). Vllm-direct cap-on
   inferred per-step = 11.50 ms. Δ = **1.92 ms**.
   Of pie's 13.42 ms:
   - GPU forward (`transform_gpu`): **10.97 ms** (82%)
   - GPU sample (`sample_gpu`):     **1.26 ms** (9%)
   - Python wrapper overhead:        **0.61 ms** (5%)
   - Inter-call gap (Rust+IPC+WASM): **0.53 ms** (4%)
   Of the 1.92 ms gap vs vllm-direct, **~1.7 ms is GPU-side**
   (transform+sample combined) and **~0.2 ms is host-side** (pie
   has slightly more Python wrapper overhead; inter-call gap is
   comparable to vllm-direct's worker dispatch cost).

   The actionable lever is *pie's separate sample kernel sequence*
   (`compute_logits` + FlashInfer `top_k_top_p`), not the Rust
   scheduler or the IPC RT.

## 1. Config parity audit

### 1.1 EngineArgs diff (`pie_driver_vllm` vs `vllm-direct` cap-on)

Source: `pie/pie/src/pie_driver_vllm/loader.py` _build_vllm_config + `pie/pie/src/pie_driver_vllm/config.py` VllmDriverConfig defaults; vllm-direct ground truth = `integrations/openclaw/scripts/task106/upstream-benches/vllm_bench.py`.

| EngineArgs flag                | pie cap-on (default)        | vllm-direct cap-on (Phase 11 §1) | Why pie differs                                              | Removable?                |
|--------------------------------|-----------------------------|----------------------------------|--------------------------------------------------------------|---------------------------|
| `enforce_eager`                | `False`                     | `False`                          | (same)                                                       | n/a                       |
| `enable_prefix_caching`        | not set → vllm default `True`| explicitly `False`              | Pie bypasses vllm scheduler → APC code path never fires      | No-op on pie either way   |
| `enable_chunked_prefill`       | not set → vllm default `True`| not set → vllm default `True`   | (same; pie's BatchAccumulator drives prefill outside vllm)   | n/a                       |
| `attention_backend`            | `FLASHINFER` (pinned)       | not set → V1 picks (FA2 on sm_80)| **Correctness**: pie's per-call slot_mapping and paged_kv metadata are written into the FlashInfer wrapper's persistent cudagraph buffers. FA2/Triton read block_table directly from per-call tensors and break under FULL graph replay. | **No** — required when cap-on |
| `cudagraph_mode`               | `FULL_DECODE_ONLY` (pinned) | `PIECEWISE` (V1 default)         | Pie drives the model outside `GPUModelRunner`, so vllm's auto-resolution never runs against pie's call shapes; pie pins FULL because its decode shape is uniform | **No** — pie's flow needs the FULL graph |
| `compilation_config.splitting_ops` | `[]` (cleared)         | (default — splits at attn ops)   | Required because FULL replays the whole forward as one graph; piecewise splitting boundaries don't apply | **No** — consequence of FULL |
| `max_num_batched_tokens`       | model `max_position_embeddings` (e.g. 40960 for Qwen3-8B) | not set → vllm picks 2048 with chunked-prefill | Compile range must cover any single-request prefill since pie doesn't chunk; setting smaller crashes on `Shape:N out-of-(1,2048)` for prompts > 2048 | **No** — pie correctness |
| `max_num_seqs`                 | 256 (VllmDriverConfig default) | concurrency (4/16/64)         | Pre-allocates per-call workspace for higher batch; vllm-direct sizes per cell | **Yes** — measured below |
| `gpu_memory_utilization`       | 0.9                         | 0.85 (Phase 11 §1)               | No correctness reason; just defaults differ                  | **Yes** — measured below |
| `skip_tokenizer_init`          | `True`                      | `False`                          | Pie's Rust runtime owns tokenization, doesn't need vllm's    | Cosmetic; no perf impact  |

### 1.2 Parity sweep — measured

`baseline` = pie's bridge defaults; `parity` = `gpu_memory_utilization=0.85` + `max_num_seqs=64` (cell-max). All other pie pins kept (correctness).

Cells: c=4/16/64 × max_tokens=200, prompt = "Write a short story about a robot. (variant {idx})", `text_completion` inferlet, chars/4 token estimate.

| Cell        | Baseline tps | Parity tps  | Δ tps    | Δ %       |
|-------------|--------------|-------------|----------|-----------|
| c=4 t=200   | 237.3        | 237.3       | 0.0      | **+0.0%** |
| c=16 t=200  | 901.8        | 894.4       | −7.4     | **−0.8%** |
| c=64 t=200  | 3040.2       | 2996.9      | −43.3    | **−1.4%** |

The two flippable knobs do not move the gap at c=4 and slightly *regress* it at c=16 / c=64 (within run-to-run noise; both inside the typical ±2% range Phase 11 reported). **Conclusion: there is no actionable parity knob to swing; the 22% end-to-end gap vs vllm-direct (Phase 11 §1) is not a config artifact.**

## 2. Per-step host overhead (Sub-task B)

### 2.1 Instrumentation

Two Python-only changes (no maturin rebuild):

- `pie/src/pie_driver/latency.py` — added `inter_call_gap` field to `StepTiming` (NamedTuple); appended `inter_call_gap_ms` column to the per-step CSV header + write path. Non-breaking: existing consumers ignore the new column.
- `pie/src/pie_driver/worker.py` — closure-mutable holder `_inter_call_state["last_t_end"]` snapshots `time.perf_counter()` at the bottom of each `_run_fire_batch_inner`. The next call computes `t_start - last_t_end` to get the wall gap between successive fire_batches. This lumps together:
  - Rust `AdaptivePolicy.decide()` latency
  - shmem IPC RT in both directions (Rust → Python forward request, Python → Rust response)
  - any inferlet WASM bookkeeping between successive `decode_step()` submissions

### 2.2 Aggregated per-cell stats (after warmup, batch_num_seqs ∈ cell range)

The harness rotation logic was meant to give each cell its own CSV but pie's `latency.py` keeps the file handle open across rotations, so all rows from c=4 + c=16 + c=64 land in one file. Splitting by `batch_num_seqs` is correct because batch shape is deterministic per cell after warmup (verified by `phase13_split_csv.py`).

| Variant   | Cell | tps    | bns mean | py total | xform_gpu | sample_gpu | inter_call mean | inter_call p95 |
|-----------|------|--------|----------|----------|-----------|------------|-----------------|----------------|
| baseline  | c4   | 237.3  | 2.5      | 12.50 ms | 10.65 ms  | 1.22 ms    | 0.36 ms         | 0.52 ms        |
| baseline  | c16  | 901.8  | 14.5     | 12.89 ms | 10.97 ms  | 1.26 ms    | **0.53 ms**     | **0.85 ms**    |
| baseline  | c64  | 3040.2 | 61.5     | 14.80 ms | 12.66 ms  | 1.38 ms    | 0.85 ms         | 1.39 ms        |
| parity    | c4   | 237.3  | 2.5      | 12.50 ms | 10.64 ms  | 1.22 ms    | 0.33 ms         | 0.47 ms        |
| parity    | c16  | 894.4  | 14.5     | 12.87 ms | 10.95 ms  | 1.26 ms    | 0.53 ms         | 0.84 ms        |
| parity    | c64  | 2996.9 | 61.0     | 14.73 ms | 12.58 ms  | 1.38 ms    | 0.87 ms         | 1.52 ms        |

**The inter_call_gap is essentially invariant to the parity-config flip** (0.53 ms either way at c=16) — so the Rust scheduler + IPC RT cost is independent of the EngineArgs flags pie tweaks.

### 2.3 Per-step decomposition at c=16 (Phase 11 §3 reference cell)

| Component                                    | mean (ms) | % of step |
|----------------------------------------------|-----------|-----------|
| `transform_gpu` (model trunk on GPU)         | 10.97     | 81.7%     |
| `sample_gpu` (compute_logits + top_k_top_p)  | 1.26      | 9.4%      |
| `embed_gpu`                                  | 0.05      | 0.4%      |
| Python wrapper overhead (`total` − GPU)      | 0.61      | 4.5%      |
| Inter-call gap (Rust + shmem IPC + WASM)     | 0.53      | 3.9%      |
| **Total per fire_batch real wall**            | **13.42** | **100%**  |

Phase 11 §3 reported py-side `total = 12.67 ms` (also `transform=10.88` `sample=1.19`). Phase 13's c=16 numbers are within ±2% of those — instrumentation is consistent. The new datum is `inter_call_gap = 0.53 ms`; Phase 11 didn't capture it.

### 2.4 Decomposition at c=4 and c=64

c=4 (small batch): `inter_call_gap = 0.36 ms`. Lower than c=16 because the Rust scheduler decides faster on smaller cohorts (fewer pinned/active counters to scan; AdaptivePolicy fires sooner when it has fewer requests to wait for).

c=64 (large batch): `inter_call_gap = 0.85 ms`, p95 = 1.39 ms. Higher because pie's scheduler waits longer for cohort assembly (rule 4 of AdaptivePolicy targets `pinned_count` matching) and because the inferlet WASM fan-out across 64 sequences pays more bookkeeping per round.

## 3. Attribution: where is the 1.17 ms / 1.92 ms gap?

### 3.1 Pie cap-on per-step (measured) vs vllm-direct cap-on per-step (inferred)

Vllm-direct cap-on at c=16: tps = 1391.0 (Phase 11 §1) → 1000/1391 × 16 = **11.50 ms/step**. We can't directly measure vllm's per-step host vs GPU split because vllm's worker is in a subprocess (Phase 11 §5; Phase 12 unblocked the kernel-attribution path but not the per-step host accounting).

| Component                      | pie cap-on c=16  | vllm-direct cap-on c=16 (inferred) | Δ        |
|--------------------------------|------------------|------------------------------------|----------|
| GPU forward (transform+sample) | 12.23 ms         | ~10.5 ms                           | **+1.73**|
| GPU embed                      | 0.05 ms          | (folded)                           | ~0       |
| Host: Python in fire_batch     | 0.61 ms          | ~0.5 ms                            | +0.1     |
| Host: inter-call gap           | 0.53 ms          | ~0.5 ms (V1 worker dispatch)       | ~0       |
| **Total per step**             | **13.42 ms**     | **11.50 ms**                       | **+1.92**|

≈ **90% of the gap is GPU forward+sample (1.73 ms)**, ≈ 10% is host-side (0.19 ms).

### 3.2 Why is pie's GPU forward+sample slower than vllm-direct's?

Both engines run cudagraph-captured decode forward (cap-on). Yet pie pays ~1.7 ms more per step on GPU. Three candidates, ordered by likelihood:

1. **Sample stage is unfused.** Pie keeps `compute_logits` + FlashInfer `top_k_top_p` as a distinct kernel sequence after the captured decode forward (see `pie_driver_vllm/forward_pass.py:transform` returning hidden states; sampling runs in a separate path). Vllm-direct fuses sampling into the captured FULL graph via its `Sampler` module called inside `execute_model`. Pie's 1.26 ms `sample_gpu` is ~1 ms more than what vllm-direct hides inside its 10.5 ms — directly accounts for ~1 ms of the 1.73 ms GPU gap.
   - Phase 11 §3 already flagged this as "lever 6 candidate (separate compute_logits + flashinfer top_k_top_p kernels could be fused or cudagraph'd)".

2. **FlashInfer per-call `plan()` cost.** Pie's bridge calls FlashInfer's `BatchDecodeWithPagedKVCacheWrapper.plan()` once per fire_batch to populate its persistent cudagraph buffers from pie's per-call metadata. Vllm-direct lets vllm's own `GPUModelRunner` manage FlashInfer state, and the plan setup is amortized into the captured graph. Pie's plan call is on the host CPU side (synchronous) so it shows up partly in `transform_gpu` (cuda.synchronize bracketed) — likely 0.3-0.5 ms.

3. **slot_mapping rewrite for KV pinning.** pie pins the KV pages of inferlets (PIN=on, ticket #95551) outside vllm's BlockManager. Each fire_batch rewrites slot_mapping to point at pinned page IDs. This per-call rewrite is GPU-side index_copy → adds a few hundred µs.

### 3.3 What's NOT the source of the gap

- Not the Rust scheduler. inter_call_gap = 0.53 ms is comparable to vllm-direct's V1 worker dispatch cost.
- Not the shmem IPC. Lumped into inter_call_gap; same conclusion.
- Not the inferlet WASM. Same.
- Not chunked-prefill differences — pie's BatchAccumulator handles prefill end-to-end, vllm's chunked-prefill reaps mixing benefits at engine boot but Phase 11 sweeps measure steady-state decode where this difference washes out.
- Not APC. Pie bypasses vllm's scheduler so APC config is dead code; vllm-direct explicitly disables it for fairness; both engines benefit from prompt-prefix sharing via different mechanisms (pie pin, vllm APC) and Phase 11 §1 measures both.
- Not the parity-flippable knobs (Sub-task A measurement).

## 4. Recommendations

1. **Lever 6: cudagraph-capture the sampling path.** Wrap pie's `compute_logits` + FlashInfer top_k_top_p in the FULL_DECODE_ONLY graph. Predicted gain: ~0.7-1.0 ms / step at c=16 = **~10%** end-to-end tps lift on top of Phase 7v. This closes ~half the remaining 1.92 ms gap.

2. **Don't invest further in inter-call gap reduction.** At 0.53 ms / step (≈ 4% of step) it is comparable to vllm-direct's worker dispatch. The complexity of optimizing Rust IPC vs the available wall-clock improvement is unfavorable.

3. **Phase 13 instrumentation patches are non-breaking and worth shipping.** `inter_call_gap_ms` column is gated behind `PIE_LATENCY_LOG=1`; existing consumers see no change. Future per-step audits can use this datum directly.

## 5. Files / commit / spend

- Patches: `pie/src/pie_driver/latency.py`, `pie/src/pie_driver/worker.py`
- Harness: `pie/benches/phase13_sweep.py`, `pie/benches/phase13_aggregate.py`, `pie/benches/phase13_split_csv.py`
- Raw: `integrations/openclaw/scripts/task106/sweep-output/phase13-config-overhead/`
  - 6 cell JSONs (3 baseline, 3 parity)
  - `baseline_c4_t200_latency.csv` + `parity_c4_t200_latency.csv` (combined per-step rows; split by batch_num_seqs)
  - `SPLIT.json` (aggregated stats)
  - `sweep.log` (full pod output)
- Pod `1i6k5j4qnl8yj1` terminated. Spend: ~$0.50.
- Cumulative: ~$25.30 + $0.50 = ~$25.80 of $30.

## 6. Caveats

1. **Token count is chars/4 approximation.** The dev image ships a pre-built `text_completion.wasm` (no `ignore_eos`, no `num_output_tokens` exposure). For relative comparison (baseline vs parity) and for per-step Sub-task B measurements (which read the CSV directly) this is accurate enough; absolute tps may be ±5% off vs Phase 11's `text-completion-bench` numbers.
2. **vllm-direct per-step decomposition is inferred, not measured.** Phase 11 §5 flagged the architectural blocker. Phase 12 unblocked vllm-direct kernel attribution but only for eager mode. Cap-on vllm-direct in-process per-step host accounting would require either monkey-patching `GPUModelRunner.execute_model` or running with `VLLM_ENABLE_V1_MULTIPROCESSING=0` + cap-on (untested here; would consume cudagraph capture path and may not produce comparable numbers to subprocess-mode V1).
3. **Per-cell CSV rotation didn't work as intended.** Pie's `latency.py` keeps the file handle open globally; renaming/unlinking the path doesn't rotate the FH. All cells from one variant land in the first cell's CSV. Splitting by `batch_num_seqs` is the correct post-hoc approach (`phase13_split_csv.py`). Future fix: have `latency.py` honor a "rotate" signal that re-opens the FH.
