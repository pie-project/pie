# Multimodal benchmark — Pie vs vLLM (image input)

Head-to-head of Pie's `cuda_native` driver against vLLM on **image question
answering**, the one multimodal task with a true parity model on both engines.

## Why image / Qwen3-VL only

A vLLM comparison only means something on a model **both** engines run:

| Pie capability            | vLLM equivalent? |
|---------------------------|------------------|
| `gemma-4-E4B` (image/audio)| ✗ not a vLLM model |
| CSM-1B audio **output**    | ✗ vLLM has no TTS |
| gemma audio **input**      | ✗ no vLLM peer |
| **Qwen3-VL-2B image input**| ✓ `Qwen3VLForConditionalGeneration` |

So the head-to-head is **Qwen3-VL-2B-Instruct, image input**. Pie's `cuda_native`
driver implements this model in full (`driver/cuda/src/model/qwen3_vl*`).

## What's measured

Both engines get the **same local image** (`assets/bench_image.png`, 896×896) and
the **same prompt scaffolding** (system + "Here is an image:" + image + question),
run greedy (temp 0), `ignore_eos`, to a fixed `max_tokens`. Prompt-token counts
match within ~0.6% (text tokens + ~784 image soft tokens). The timed path on both
sides is: image preprocess (resize/patchify) → vision encode → text prefill →
decode. No network fetch is in the measured window.

- **latency** — one request at a time (concurrency 1). The unamortized,
  single-stream cost.
- **tput** — `N` requests under a concurrency cap. The batched-serving regime.

## Results (2× NVIDIA L40, Qwen3-VL-2B-Instruct, 128 output tokens)

The only meaningful number is the **gap to vLLM**. (Pie's as-shipped state ran the
vision tower on naive non-tensor-core kernels — a correctness-over-speed
placeholder, not a real baseline — so "×-over-as-shipped" figures would just
measure fixing that bug; they're omitted.) vLLM is unchanged throughout; parity
(identical image description) held at every step.

### Latency — single stream (n=16, warmup=4)

| metric           |  vLLM | **Pie** | gap |
|------------------|------:|--------:|----:|
| latency p50 (ms) |   764 | **874** | 1.14× |
| output tok/s     |   167 | **146** | — |

### Throughput — concurrency 32 (n=128, warmup=4)

| metric       |   vLLM |    **Pie** | gap |
|--------------|-------:|-----------:|----:|
| wall (s)     |   8.57 |   **14.4** | 1.7× |
| requests/s   |  14.93 |   **8.87** | 1.7× |
| output tok/s | 1910.8 | **1135.6** | 1.7× |

**Gap to vLLM: 1.14× (latency), 1.7× (throughput).** The sections below are the
engineering record of closing it: §"Optimization arc" (fix the naive vision
kernels → cuBLAS/flash), §"Latency stage 2" (decode split-KV + host cache).

## Root cause (profiled, not guessed)

Decomposing single-stream latency with `--max-tokens 1` (vision+prefill only) vs
128 showed **decode was never the problem** — 6.4 ms/token, on par with vLLM's
~6 ms. The entire gap was the **vision encoder**: ~2240 ms for one 896×896 image
(784 patches × merge). Its source file said it outright — *"first-cut draft …
naive kernel patterns … correctness over speed."* Every projection ran a
**thread-per-output-element `k_matmul`** (no tiling, no tensor cores) and
attention was a naive per-head O(N²) `k_qk`/`k_av` loop. The main LLM is fast
because it uses cuBLAS; the vision tower never got that far.

## Optimization arc

Every change preserves the math (bf16 in/out, fp32 accumulate), verified by
identical image descriptions. The vision encode dropped ~2240 ms → ~50 ms.

| # | Change | Effect |
|---|--------|--------|
| 1 | **Projections → cuBLAS.** `k_matmul` (patch/QKV/o/fc1/fc2/mergers) → `ops::gemm_act_x_wt_bf16` (tensor cores) + bias; thread cuBLAS handle from `llama_like`. | vision+prefill 2240→668 ms |
| 2 | **Attention → cuBLAS.** Per-head `k_qk`/`k_av` → two `cublasGemmEx` (QKᵀ fp32 for softmax, then P·V). | 668→224 ms |
| 3 | **Fused softmax→bf16.** One shared-memory kernel reads scores once, writes bf16 probs directly (drops a separate f32→bf16 pass). | attn 41→26 ms |
| 4 | **Host overhead.** Cache the constant pos-embed table (was re-copied D2H every pass); `cudaMallocAsync`/`cudaFreeAsync` (stream-ordered — `cudaMalloc`/`cudaFree` synchronize the device). | table 5.7→0 ms; removes per-image device syncs |
| 5 | **Batched attention (attempted, reverted).** Strided-batched cuBLAS over all heads *regressed* (attn 26→44 ms): the per-head [N,N] score buffer (~39 MB) stays resident in the L40's ~48 MB L2 across QK→softmax→AV; a batched [NH,N,N] (~945 MB) blows L2. Per-head kept. | — (finding: keep per-head) |

## Latency stage 2 (single-stream focus)

After the vision-encoder work, single-stream latency decomposes as ~166 ms
vision+prefill + ~810 ms decode. Profiling each:

| # | Change | Effect |
|---|--------|--------|
| 6 | **Vision attention → flashinfer flash.** Replace the per-head cuBLAS QK/softmax/AV with the same flashinfer prefill kernel the LLM uses (non-causal, `causal_mask=false`, head_dim 64, q/k/v as a single-seq paged view) — no [N,N] gmem scores at all. | attn 28→**5.7 ms** |
| 7 | **Decode split-KV (flash-decoding).** The driver's flashinfer planner force-disabled `split_kv` for batch≤512 on sm≥8 (`PIE_FLASHINFER_FORCE_SPLIT_KV_SMALL`). That makes **long-KV** batch-1 decode read the KV serially → linear-with-context. Re-enabling split-KV (env) makes it flat, like vLLM/SGLang flash-decoding. **A real perf bug for all long-context decode, not just multimodal.** | decode 6.75→6.21 ms/tok |
| 8 | **Grid-keyed host cache.** rope positions + interpolated pos-embed are a pure function of the grid; cache the device buffers by `(t,h,w)` so the per-image CPU interp + bf16-convert + H2D (~12 ms) runs once, not per image. | scatter 46→33 ms; host CPU 12.5→0 |
| 9 | **Cross-request vision batching (attempted, reverted).** Encode all images in a prefill batch in one tower pass (concatenated rows, multi-seq flashinfer attention, per-image mergers). *Regressed.* Re-confirmed at mt128 scale where the scheduler genuinely coalesces up to **10 images/fire**: same-session back-to-back **1065 (batched) vs 1148 (per-image) tok/s**. The per-image GEMMs at M=3136 are already past the MFU knee, so batching to M=Σn_patch gives almost no GEMM win while multi-seq attention + per-fire plan rebuilds + ~0.9 GB scratch add overhead. Kept env-gated (`PIE_VIS_BATCH_IMAGES`). | — (finding: per-image already near floor) |
| 10 | **Vision per-layer epilogue fusion.** Fold the qkv bias into the split kernel, fc1 bias into the gelu, fuse the two residual-adds via cuBLAS `beta=1` (o- and fc2-projections write the residual in place), and do q+k RoPE in one launch. Drops ~5 kernels/layer (×24) + 2 tmp round-trips/layer. Also removed a redundant mid-forward `cudaStreamSynchronize` in the scatter (stream ordering + the fire's final sync already cover it). | vision run 27.5→**24.5 ms**; latency mt1 97→**93 ms**. Throughput within noise (GPU-bound; 3 ms/img ≈ 0.4 s of a 15 s run). Parity preserved. |

The decode finding is the important one and is isolated by a controlled test
(same model, text, varying KV length):

| decode/tok | short KV (~256) | long KV (~900) | Δ |
|------------|----------------:|---------------:|--:|
| Pie (before split-KV) | 5.46 ms | 6.29 ms | +0.83 |
| Pie (split-KV on)     | — | 6.21 ms | — |
| vLLM | 5.65 ms | 5.74 ms | +0.09 |

i.e. Pie's decode was at parity at short context but scaled ~9× worse with KV
length; flash-decoding closes most of that.

Touched files: `driver/cuda/src/model/qwen3_vl_vision_forward.cu` (kernels→GEMM,
flashinfer attention, async alloc, grid host cache, env-gated `PIE_VIS_TIMING`,
**stage-3 epilogue fusion: `k_split_qkv_bias`/`k_gelu_bias`/`k_rope_qk` + β=1
residuals + removed mid-forward sync**), `qwen3_vl_vision_adapter.cpp` (cuBLAS
bridge w/ `beta` + flashinfer vision-attention helper with its own workspace/plan),
`qwen3_vl_vision_forward.hpp` + `llama_like.cpp` (handle plumbing);
`driver/cuda/src/executor/executor.cpp` (env-gated `PIE_FIRE_TIMING` per-fire log);
`benches/pie_mm_bench.py` (bakes `PIE_FLASHINFER_FORCE_SPLIT_KV_SMALL=1`, surfaces
`[fire]`).

### Where the remaining gap is (current)

Pie core engine is at **parity** with vLLM — verified controlled, including long,
prefill-heavy prompts at 2B-class sizes (text, max_tokens=8, 911-tok prompt:
Pie 83.6/33.9 req/s vs vLLM 84.3/33.8 at 0.6B/1.7B). So the residual is the
multimodal-specific vision+prefill plus a small decode delta.

**Latency** 874 ms vs vLLM 764 ms (1.14×). Remaining ~110 ms ≈:
- **vision+prefill ~+58 ms** — vision tower now ~26 ms (attn 5.7 + GEMMs/norms ~20)
  vs vLLM's whole ~22 ms, plus the single-stream LLM prefill of the 833-tok
  (mostly image) sequence, which is slower at **batch 1** than vLLM's (the same
  prefill is at parity when *batched* — it's a single-stream GPU-utilization gap).
- **decode ~+51 ms** — 6.21 vs 5.81 ms/tok after split-KV; the residual is harder
  (decode-attention micro-efficiency over long KV).

**Throughput** ~1100 tok/s vs vLLM 1911 (1.7×) — GPU-bound (~99% busy), and the
cost is **distributed** across vision (~26 ms/img) + LLM prefill (~56 ms/req) +
decode, with each Pie kernel somewhat less efficient than vLLM's fused/CUDA-graphed
equivalents. It is **not** a single addressable bottleneck: the two structural
shortcuts — batched attention (#5) and cross-request vision batching (#9) — both
**regressed**, because the per-image vision is already near its compute floor at
M=n_patch. So closing the remaining 1.7× is a **broad kernel-maturity effort**
(match vLLM's fused layernorm+bias+gelu, op-level CUDA graphs, and prefill kernels
across the whole multimodal forward), not an incremental tweak. The same naive
pattern also lives in `gemma4_vision_forward.cu` (the gemma VLM would benefit from
the cuBLAS + flashinfer treatment applied here).

## Stage 3: where the 1.7× actually lives (direct decomposition)

The earlier "distributed across everything" framing was directionally right but
imprecise. Per-fire timing (`PIE_FIRE_TIMING=1`) + single-stream isolation
(`--max-tokens 1`, concurrency 1) pin it down. The two regimes diverge sharply:

| vision+prefill (no decode) | Pie | vLLM | gap |
|----------------------------|----:|-----:|----:|
| single stream (latency, 1 req) | ~25.8 ms (GPU) / 65 ms wall | 27.5 ms | **~parity** |
| batched (tput, 64 req, c32) | 98 ms/req | 24.5 ms/req | **4×** |

Per-request kernel speed is **at parity single-stream**; the gap is entirely a
**batching/utilization** effect. Confirmed by the latency-vs-throughput split:
single-stream latency 1.14× but throughput 1.7×.

**Per-fire decomposition (single-stream, mt2 → 4 fires/request):**

| fire | tokens | imgs | time | what |
|------|-------:|-----:|-----:|------|
| A | 24 | 0 | 6.0 ms | preceding-text flush (`append_image` calls `flush()` first) |
| B | 784 | 1 | 52 ms | **vision encode (~30 ms) + image-token LLM prefill (~22 ms)** |
| C | 25 | 0 | 7.1 ms | question + cue prefill + first sample |
| D | 1 | 0 | 6.0 ms | decode step |

So one prompt vLLM prefills in **one** forward, Pie splits into **three** (A/B/C) —
the inferlet's `append_image` does its own forward. Fires A/C are ~6 ms of mostly
fixed per-fire overhead each. **But in throughput these coalesce** (the scheduler
batches small text prefills into decode fires — `batch_size_hist` is dominated by
the 32-bucket), so **fragmentation is a latency cost, not a throughput one.**

What the throughput gap *is* made of (all GPU-bound, gap/batch ≈ 0):
- **Vision tower ~24.5 ms/img — its GEMMs and attention are already at the
  hardware peak.** A GEMM-only ablation (`PIE_VIS_GEMM_ONLY=1`, skips the
  elementwise) splits the tower run cleanly: **GEMMs ≈ 14 ms = 135 TFLOP/s**,
  attention ≈ 5.9 ms (flashinfer, ~90 %), elementwise ≈ 4.7 ms. The 135 TFLOP/s
  is **not 66 % of some headroom** — it equals the rate the *at-parity* LLM
  prefill hits, i.e. it's the achievable bf16-with-FP32-accumulate ceiling on the
  L40. (The 181 TFLOP/s spec is the FP16-accumulate / 2:1-sparsity number, not
  reachable for this math; bigger cuBLASLt workspace and alternate algos/GEMMEx
  all give the same ~14 ms.) Qwen3-VL's ViT is **full attention every layer** (it
  dropped Qwen2.5-VL's window attention — `hf_config.hpp` has no `window_size`/
  `fullatt_block_indexes`), so there's no windowing FLOP to save either.
  **So the only vision headroom is the ~4.7 ms of memory-bound elementwise
  (layernorm/split/rope/gelu/bias) that idles the tensor cores between GEMMs** —
  the same work vLLM fuses away (torch.compile / fused kernels), which is exactly
  why its vision (~GEMM+attn ≈ 20 ms) beats Pie's (24.5 ms). Closing it is
  fusion, not faster GEMMs.
  - *Done this round:* fused the qkv-bias into the split, both residual-adds into
    the GEMM (β=1), and **split-qkv + q/k-RoPE into one kernel** — tower
    **27.5 → 23.5 ms**, parity preserved. The remaining ~3.5 ms (the gelu's
    51 MB round-trip, the two layernorms, the o/fc2 bias) needs cuBLASLt bias
    epilogues (≈0.7 ms, but can force a worse algo) and — for the gelu, since
    tanh-GELU isn't a cuBLASLt epilogue — a custom fused-MLP kernel. Each piece
    is sub-noise on end-to-end throughput (≈4 ms/img ≈ 0.5 s of a 15 s run); the
    payoff is reaching vLLM-vision-parity (~20 ms), a fused-ViT-kernel project.
- **Decode is weight-bandwidth-bound**, not slow: batch-32 step = 7.5 ms (≈ 4.6 ms
  just to stream the ~4 GB of bf16 weights once + 32× compute) → ~4300 tok/s during
  pure decode. ~15 % behind vLLM at most; near the HBM floor.
- **LLM prefill** ~at parity (text-confirmed); image-prefill carries M-RoPE +
  deepstack adds (inherent to Qwen3-VL).

**The one structural big lever: overlap vision with decode on separate streams.**
Decode leaves the SMs idle while it streams weights from HBM; vision saturates the
tensor cores while barely touching HBM. They are complementary, but Pie runs
**one fire at a time on one stream**, so vision never fills decode's idle compute.
Encoding upcoming requests' images on a side stream concurrent with the running
batch's decode (vLLM-style encoder-ahead + embedding cache) could hide most of the
~4 s of vision behind the ~4 s of decode — plausibly 1.7× → ~1.3×. This is a
**core execution-model change** (second stream, encoder-ahead scheduling, embedding
buffering, keeping the decode CUDA-graph path intact), not an incremental kernel
edit — hence flagged as a deliberate next step rather than landed here.

## Files

| file | role |
|------|------|
| `vllm_mm_bench.py` | vLLM driver (offline `LLM` + `multi_modal_data`) |
| `pie_mm_bench.py`  | Pie driver (`pie serve` + `image-qa-bench` over PieClient) |
| `run_mm_compare.py`| runs both, sequentially, prints the side-by-side table |
| `assets/bench_image.png` | the shared 896×896 test image |
| `../inferlets/image-qa-bench/` | the Pie bench inferlet (base64 image in, exact token counts out) |
| `out/mm/*.json` | saved per-run summaries + per-request detail |

## Reproduce

```bash
cd benches
# build the bench inferlet once
( cd ../inferlets/image-qa-bench && cargo build --release --target wasm32-wasip2 )

# full head-to-head (Pie under system python, vLLM under its venv)
python3 run_mm_compare.py \
    --model Qwen/Qwen3-VL-2B-Instruct --image assets/bench_image.png \
    --max-tokens 128 --latency-requests 16 --tput-requests 128 \
    --concurrency 32 --warmup 4 --out-dir out/mm

# isolate prefill + vision encode only (no decode):  --max-tokens 1
# single engine, e.g. just Pie latency:
python3 pie_mm_bench.py latency --requests 16 --max-tokens 128 --dump-first-text
/root/.venv/vllm/bin/python vllm_mm_bench.py tput --num-requests 128 --concurrency 32
```
