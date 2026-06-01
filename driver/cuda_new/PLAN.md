# driver/cuda_new — clean-rewrite implementation plan

> Canonical architecture overview: [`/NEW_DRIVER.md`](../../NEW_DRIVER.md).
> This file is the in-tree working checklist (live per-phase status).

Scope: **frontier-only** (legacy archs dropped). Targets: MLA (DeepSeek-V4/
Kimi/GLM), modern MoE (Qwen3.5-MoE/GPT-OSS), Gemma-4, Nemotron-H, + dense
Qwen3.5/Llama-3/4 on `llama_like`. Strategy: implement all frontiers upfront,
then validate/fix/refactor. Frontier kernel foundation lifted + standalone-
validated (banked, not yet wired): MLA paged attention (CPU-parity), MoE
dispatch, Gemma-4 rope_partial+altup, Mamba causal_conv1d, int4 dequant.

Quant-GEMM lane decided (see NEW_DRIVER §10): a W4A16 spike confirmed CUTLASS
4.1's SM90 mixed-input *builder* can't produce a working int4 kernel (needs
example-55's manual smem reorder; only fallback is dequant-then-bf16, which we
ban) → the fused quant GEMM is an **internalized, de-branded Marlin** (`qgemm`,
int4 slice in flight), cutlass scoped to grouped-MoE + attention.

Quant GEMM DONE: the internalized de-branded **int4 (u4b8) Marlin** is wired
(ABI v21, `qgemm/`: `w4a16_bf16_gemm` + `w4a16_repack` + workspace query) and a
Rust parity test confirms the §10 contract (fused ≡ dequant-then-matmul) vs the
`dequant_wna16` oracle. Bug found en route: the launcher must apply
`marlin_permute_scales` (scales pre-permuted along N into the MMA fragment
layout) — a `scale_permute_kernel` now does this. Next: fan out fp8/mxfp4/int8
as new dequant specializations + instantiations over this frozen template.

Status: **91 tests green on H100.** Frontier breadth landed via successive
parallel-agent fan-outs (each disjoint files + standalone nvcc selftest; central
ABI wiring serialized by the integrator):
- **Sparse MoE block** (ABI v22, `forward/moe_sparse.cu`) — dispatch-scatter →
  grouped GEMM → weighted combine; Rust test asserts it == the dense
  `moe_mlp_block` (bit-exact in the selftest).
- **Nemotron-H Mamba-2 mixer block** (ABI v23, `forward/nemotron_block.cu` +
  `kernels/mamba_proj.cu`) — in_proj → split → causal conv → SSD scan → gated
  RMSNorm → out_proj; Rust out_proj_w=0 passthrough smoke test.
- **MoE forward** (ABI v19, `forward/moe_forward.cu`) — Qwen3.5-MoE/GPT-OSS shape
  (llama attention + top-K MoE FFN per layer); standalone selftest exact, Rust
  ABI-marshalling smoke test.
- **MLA forward** (ABI v18, `forward/mla_forward.cu`) — DeepSeek-V4/Kimi/GLM
  (embed → N×mla_block → norm → lm_head → argmax, per-layer latent cache slice);
  standalone exact, Rust smoke.
- **Grouped per-expert GEMM** (ABI v17, `ops/grouped_gemm.cu`) — sparse-MoE GEMM
  after dispatch scatter (per-group cuBLAS, empty-group skip); CPU-parity test.
- **Mamba-2/SSD selective scan** (ABI v20, `kernels/ssm_scan.cu`) — Nemotron-H
  recurrence (verbatim de-branded lift of nemotron_h.cu's warp scan); standalone
  bit-exact + stress, Rust N=1 parity test.
**AltUp** predict/correct (Gemma-3n/4
alternating residual streams) wired (ABI v16, `kernels/altup.cu`) with CPU-parity
tests — on the Gemma-4 critical path. **MLA attention block** wired
(`forward/mla_block.cu`, DeepSeek absorbed form over banked `mla_naive_paged` +
`mla_write`): ABI v15 (`PieMlaLayerWeights` + `pie_cuda_mla_block_bf16`),
standalone selftest = exact bf16 match, Rust ABI-marshalling smoke test (W_o=0
exact passthrough + real-W_o finite/changed). Dense **MoE MLP block** wired
(`forward/moe_mlp.cu`: router → top-K softmax → per-expert gate_up/swiglu/down →
weighted combine), ABI v14, CPU-parity test. General paged KV wired into the forward
(scatter + `num_kv_pages`); loader handles BF16/F16/F32 checkpoints (cast on
load); `Model::prefill_sample` does temperature sampling. Lifted & wired via
parallel agents: Gemma kernels, FP8 dequant, YaRN RoPE, MoE (topk_softmax +
chunked_swiglu). Phase 1 complete; phase 2
construction end-to-end. `builder.rs::build` runs `config.json` → `arch::detect` →
`mem::plan` → `loader` (safetensors) → alloc → `Model::prefill_greedy`,
no `is_*_arch` cascade. General paged KV is now wired into the forward
(scatter + `num_kv_pages`). Primitives lifted + ABI-wired and ready: temp
sampling, dtype casts (F16/F32→BF16), gather_rows, Gemma kernels, FP8 dequant.
Remaining *wiring*: loader F16/F32 via casts, `sampler`→`sample_temp`,
compact logits, a Gemma forward. Larger: loader completion
(quant/sharding/GGUF), transport routing (phase 3), flashinfer two-phase. Phases 2–4 (cold control-plane logic) advanced in
parallel: the pure-Rust cores of `mem` / `arch` / `sampler` / `spec` are
ported from their `driver/cuda` sources and unit-tested (~3.2K LOC), not
yet wired to the executor (waits on the hot-path ABI). This tree is a
parallel rewrite of `driver/cuda`; the current driver keeps running
unchanged and is replaced via a strangler-fig migration (see *Migration*).

Build/test the slice:
```bash
cd driver/cuda_new/control
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_CUDA_ARCHITECTURES=90 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 cargo test
```

## Remaining roadmap (to cutover)

The device-lib frontier *compute* is largely in-tree + tested; the gap to the
north star is **integration**: detect frontier archs, load their weights, drive
their forwards from a real executor loop, then cut over. ★ = critical path.

**A. Device-lib frontier completeness (compute)**
- [x] A1. **Gemma-3/4 forward DONE** (`forward/gemma_forward.cu`, ABI v25): sandwich norms + sliding/full alternation + attn & final logit soft-caps + √H embed scale; standalone exact, Rust smoke test. Deferred (documented): per-head qk-norm, AltUp, dual global/local RoPE theta.
- [~] A2. Quant fan-out. **fp8 (fe4m3fn) DONE** (ABI v27, `qgemm` `w8a16_fp8_*`: gemm + repack + workspace; launcher folds the fp8 exponent bias into the permuted scale; int4 unchanged); Rust parity test vs an e4m3fn-decode oracle. Remaining: **mxfp4** (fe2m1f / GPT-OSS — needs the e8m0 1-byte block-scale permute + `group_blocks==2` instantiations; dequant specializations already in place, path documented in the agent report) and int8.
- [ ] A3. `qk-norm` kernel (Qwen3.5 / Gemma).
- [ ] A4. Forward refinements: MoE shared-expert, GPT-OSS sliding window, Gemma embed-scale/softcap.
- [x] A5. **Nemotron-H whole-model forward DONE** (`forward/nemotron_forward.cu`, ABI v26): hybrid `M`/`A`/`F` per-layer schedule (Mamba mixer | GQA attention | FFN); standalone exact, Rust smoke test. Deferred (documented): FFN is a dense-SwiGLU stand-in for the upstream relu² MoE (localized to `ffn_layer`).
- [ ] A6. Quantized forwards (linears call `qgemm`, not bf16 gemm).
- [~] A7. Decode path. **Transformer decode DONE**: `Model::{step_single, generate_greedy}` — prefill then 1-token decode steps reading cached KV (the KV-append kernel writes at the cached offset `pre = total − new`); tested as **incremental decode ≡ re-prefilling the grown sequence** (bit-identical). Remaining: multi-request batched decode (per-request page growth/eviction in the executor), and SSM/conv decode-state variants (Mamba single-token; left behind by the scan lift).

**B. Control-plane integration — the god-function kill**
- [x] **B1. ★ DONE** `arch::detect` frontier archs (DeepSeek-MLA·Kimi·GLM, Qwen3.5-MoE, GPT-OSS, Gemma-3/4, Nemotron-H) + dropped legacy; data-driven `ArchSpec` each (MLA/Mamba dim sub-structs, MoE sizing, softcap/altup, hybrid `LayerKind` schedule from `hybrid_override_pattern`). 7 new detect/spec tests. The `is_*_arch` cascade is gone.
- [~] B2. ★ `builder.rs` routes `ArchId` → forward + cache. **DeepSeek DONE**: `build()` is now backend-polymorphic (`Backend::{Llama paged-KV+ws, Deepseek MLA-latent ckv/kpe}`); `build_backend` matches `spec.id`, derives `DeepseekConfig` from the `ArchSpec`+`HfConfig`, loads `LoadedDeepseek`, allocs the MLA cache; arch-aware `kv_bytes_per_token` in `mem::plan`; unified `run_forward` dispatches prefill. `build_and_prefill_deepseek` runs config.json→`build`→prefill end-to-end. **Qwen3.5-MoE also DONE** (`Backend::Moe` + `LoadedMoe` + `build_and_prefill_moe`) — the pattern now spans 3 backends (dense-llama, MLA, dense-MoE). Remaining: Gemma/Nemotron builder arms (each a `build_backend` match arm + its `LoadedX`); qk-norm refinement for the MoE path.
- [~] B3. ★ `loader.rs` frontier weight layouts. **DeepSeek-MLA DONE**: `LoadedDeepseek::load` does the `kv_b_proj` split/transpose (`split_kv_b_proj`, pure-tested) + MoE expert stacking (`stack_moe_experts`, pure-tested); `load_and_forward_deepseek` runs a synthetic checkpoint **end-to-end** (load→`deepseek_forward`→tokens) + checks on-device `W_uk` == host split. Remaining: Nemotron/Gemma weight layouts, quant repack→`qgemm`, GGUF/TP/real-checkpoint.
- [~] B4. ★ hot-path loop. **Batched-forward core DONE**: `Model::fire_batch` lays R requests across the paged KV cache (per-request pages/CSR) → ONE forward → per-request greedy next token, **bit-equivalent to single runs** (tested). Un-stubbed the full paged-KV cache (builder allocs `num_kv_pages` pages/layer, arch-aware, all 3 backends; `run_forward`+`dispatch_forward` share it). Remaining: decode step (A7, single-token continuation reading cached KV), per-request sampling in the batch, and `Executor::fire` as the transport/serving wrapper around the core.
- [ ] B5. Wire `sampler.rs` + `spec.rs` (MTP) into the executor.
- [ ] B6. `tp.rs` rank-0 broadcast loop (multi-GPU).

**C. Hot path & perf parity**
- [~] C1. **Fast tensor-core paged attention WIRED (llama path).** De-branded lift
  of driver/cuda's flashinfer wrapper → `ops/attention_paged.{cu,hpp}` (raw-ptr
  bf16; plan/dispatch split; prefill + decode kernels; SM90-FA3 + KvCacheLayerView
  dropped). `forward/attn_runtime.{cu,cuh}` is the glue: lazy plan-scratch alloc,
  a tiny D2H of the index arrays (so **zero Rust signature churn**), decode-vs-
  prefill gate (GQA∈{1,2,3,4,8} + head_dim∈{64,128,256,512}, else prefill kernel,
  else naive fallback). Llama forward plans once/fire + dispatches per layer.
  Device lib + 110 control tests green (argmax-stable parity holds). FlashInfer
  consumed header-only via CPM v0.6.9 (cached tree, no codegen/csrc).
  **PERF A/B MEASURED (Qwen2-0.5B, H100, same-session, vs old driver/cuda):**
  tput 13%→**40%** of old (naive 3244 → fi 10048 → old 25397 tok/s; **3.1×** over
  naive); latency 33%→**61%** of old (naive 194 → fi 360 → old 594 tok/s; **1.85×**
  over naive). Remaining gap (2.53× tput / 1.65× lat) is dominated by per-token
  kernel-launch overhead (~300 un-graphed launches/token on this 24-layer toy) →
  that's the C2 (CUDA-graph) target. REMAINING: fan out to gemma/moe forwards;
  prepare/body ABI split (needed for C2).
  **GEMMA-4 IS THE PERF TARGET (user pivot).** gemma-4-E4B now loads+runs through
  cuda_native (3 multimodal-loader fixes: prefix, nested text_config, strip-nulls;
  simplified forward = `<pad>` garbage but real gemma-4 shape, perf-valid). Gemma
  forward wired to flashinfer + moved to persistent PieWorkspace (killed 12
  per-fire cudaMalloc). **A/B (gemma-4-E4B, H100, vs old driver/cuda — which also
  serves gemma-4):** LATENCY old 124 / **fi 109** tok/s = **88% of old (1.13× gap,
  near parity!)**; TPUT old 10485 / **fi 2756** tok/s = 26% of old (3.8× gap). So
  the per-token attention path is competitive; the gap is now BATCHED THROUGHPUT.
- [x] C2a. **Per-fire D2H killed (decode hot path).** `attn_runtime::plan_attention_for_fire`
  no longer does a synchronous `cudaMemcpy` D2H of the index arrays for decode
  (R<=512): the static-nonsplit decode schedule is R-only and the raw-bf16
  dispatch never reads `num_pages_in_batch`, so a zeroed host array suffices —
  no per-fire stream stall. Prefill (one-shot) keeps the D2H.
- [~] C2b. **CUDA-graph capture — DESIGNED, thin-C++/fat-Rust (not started in code).**
  IMPORTANT ARCHITECTURE NOTE: a first attempt put the graph cache + persistent
  input buffers + capture/replay decision INSIDE C++ (decode_graph.cu, ws fields)
  — that rebuilt the C++ monolith and was REVERTED. The correct split:
    • C++ stays THIN — only mechanics: `prepare` (plan→attn_ws; = plan_attention_
      for_fire), a `body` entry (embed→layers→lm_head→argmax, NO plan / NO sync,
      reads caller pointers + reconstructs the lightweight AttnPlan from ws), and
      `graph_begin`/`graph_end(&exec)`/`graph_launch(exec)`/`graph_destroy` (pure
      cudaStreamBeginCapture/EndCapture/Instantiate/Launch). The `pie_prepare`/
      `pie_body`/`pie_graph_capture`/`pie_graph_launch` ABI STUBS already exist.
    • RUST owns ALL control (executor.rs already has graph_request_bucket): the
      persistent pointer-stable input DeviceBuffers (refilled per fire), the
      HashMap<(R,N)→PieGraphExec> cache, and the per-fire decision (refill →
      prepare → cache-hit?launch:capture → sample → one sync).
  Gated on the decode-kernel path (GQA∈{1,2,3,4,8} ⇒ static R-only schedule ⇒
  graph-safe; gemma-4 GQA=4 qualifies). Targets the gemma-4 TPUT gap (26%→).
- [ ] C3. `qgemm` M≤8 grouped path fix + dispatch perf tuning.
- [ ] C4. Fused grouped-MoE GEMM (replace per-group cuBLAS loop).
- [ ] C5. MoE block stream-async (drop per-layer sync).

**D. Cutover (the finish line)**
- [x] **D1. ★ SERVER WIRING DONE.** `control/` is linked into `pie-server` as a Cargo **rlib** dep (NOT via `server/build.rs` CMake — the control crate's own `build.rs` runs the `cuda_new/device` CMake and emits the link directives, which propagate to the final binary). Gated by a new `driver-cuda-native` Cargo feature; the crate stays in the workspace `exclude` and is pulled only as an optional path-dep. `driver_ffi.rs` gained `Flavor::CudaNative` + `use pie_driver_cuda_native_lib::{…run_inproc,…request_stop}` (rlib symbol-GC dodge, like dummy) + dispatch arms; `from_kind(CudaNative)` prefers the new crate over the legacy C++ driver (strangler-fig). The `cuda_native` *plumbing* (DriverOptions/options/template/topology in embedded_driver.rs, serve.rs, topology.rs, template.rs, driver_cmd.rs) was made backend-agnostic via `any(driver-cuda, driver-cuda-native)`; the C++ build/NCCL/TP stays `driver-cuda`-only. **Build pin:** `cuda_new/device/CMakeLists.txt` now defaults `CMAKE_CUDA_ARCHITECTURES=90` (`native` mis-detected sm_75 here, which `#if`-strips the bf16 qgemm conversion members → build break).
- [x] **D2/D3 (driver side) DONE** — `lib.rs::run_inproc` + serve loop, **no IPC** (the in-proc path is the `pie-bridge` function-pointer `InProcVTable`, not shmem): parse `--config` → `builder::build` → READY caps JSON → loop `recv` `PieFrameDesc` → `__pie_frame_from_desc` → `Model::serve_forward` (wire `ForwardRequest` → forward → greedy tokens → `ForwardResponse`) → build `PieResponseFrameDesc` → `send_response`; `request_stop` flips an atomic. End-to-end test `run_inproc_serve_loop_forward` drives the whole loop through a mock vtable and the reply == greedy prefill. Remaining: the **server-side selection glue** (un-exclude the crate from the workspace; `driver-cuda-native` Cargo feature + `Flavor::CudaNative` + `driver_ffi` dispatch) so `pie serve` routes to it via the runtime's own (identical) vtable; temperature/top-k/top-p **sampling** (greedy only today); Copy/Adapter payloads.
- [ ] D3. Transport routing via `pie-bridge` (rkyv/shmem/inproc).
- [~] D4. ★ E2E + **REAL-MODEL ACCURACY PARITY vs driver/cuda — ACHIEVED**. cuda_new loads + forwards the real **Qwen3-4B** (sharded bf16, 36 layers, qk-norm, tied embeddings) on the H100 (prefill 5 tok ≈ 7.6 ms). vs `driver/cuda`'s parity harness (`pie_driver_cuda --parity-tokens/--parity-out`, built from the in-tree CMake) on identical tokens/config: **same argmax (264), cosine 0.999973, top-10 overlap 10/10, max-abs 0.125 (bf16-level)** — the rewrite is numerically equivalent to the native driver. (Required closing in-scope gaps, all tested: **sharded safetensors** loader, **qk-norm** kernel+wiring, **tied embeddings**.) Synthetic-checkpoint e2e (`build → generate_greedy`, decode≡re-prefill) also green for llama/MLA/MoE. REMAINING: **perf** parity (cuda_new uses the correctness-first naive paged attention + cuBLAS, not flashinfer/CUDA-graph — the C-bucket), and the **`run_inproc` serve loop** (D2/D3 — vtable wire-schema marshalling + server Flavor/build wiring) to actually serve.
- [x] **★★ FULL `pie run text-completion` E2E DONE** (Qwen2-0.5B, real CLI). `./pie run -c <toml> --path text_completion_bench.wasm …` → engine boot → cuda_new driver (READY caps handshake) → runtime tokenizer + Qwen2 chat template → cuda_new forward (multi-page prefill + autoregressive decode) → detokenize → `"The capital of France is Paris."` (stops at EOS naturally; coherent). Required a **real bug fix**: the Llama forward **workspace was sized to `page_size` (16) tokens** but the runtime sends a single **multi-page prefill** (e.g. 28 tokens / 2 pages) — every per-token scratch buffer overflowed (garbage on tiny models, CUDA-717 fault on Qwen2). Fixed by sizing the workspace to `num_pages * page_size` (= the reported `max_forward_tokens`); regression test `build_and_prefill_from_model_dir` now asserts a 28-token single-shot prefill == token-by-token decode. Also added qkv-bias (`add_bias` kernel) for Qwen2 + re-allowed `qwen2`→LlamaLike. **Non-greedy SAMPLING now wired (ABI v30):** extended the Gumbel-max `sample_temp_bf16` kernel with **top-p + top-k** (both = a per-row logit cutoff found by binary search; min-p already there; kept set = intersection of enabled filters), and `serve_forward` now reads the per-row wire `Sampler` (Multinomial/TopK/TopP/MinP/TopKTopP → temperature/top_p/top_k/min_p/seed; greedy when temp≤0) and samples the sampling rows (gather→compact→sampler) instead of always argmax. Unseeded samplers draw a fresh per-process seed (clock+pid base). Verified live: `--temperature 0.8 --top_p 0.95` gives coherent, run-to-run *varied* output ("...a princess who lived in a castle..." vs "...a young boy named Peter..."); `--temperature 0` is deterministic greedy. Kernel correctness test (no PRNG replication): top_k=1 / tiny top_p ⇒ argmax; top_k=K ⇒ drawn token always within the top-K set across seeds. **Deferred:** the non-token Sampler variants (RawLogits/Dist/Logprob(s)/Entropy/Embedding — need ForwardResponse side channels, fall back to argmax), Adapter payloads, perf parity (naive paged attn vs flashinfer/CUDA-graph).

**SERVING ROBUSTNESS (ABI v31):** (1) **Multi-request batched serve** validated — `serve_forward`'s CSR path handles R requests of mixed length in one fire (unit: 2-request batch each == single-run; live: N concurrent `Context::fork()` streams in demo-parallel-fork). (2) **Copy (D2D) KV page copy** implemented + wired: new `pie_cuda_memcpy_d2d` ABI + `DeviceBuffer::copy_within_d2d` + `Model::copy_pages` (copies each (src,dst) page across all layers in K+V, or ckv+kpe for MLA) + serve-loop `RequestPayload::Copy` arm. Only `Kv`+`D2D` (context fork / prefix share); host swap (`D2H`/`H2D`/`H2H`) stays gated off by the advertised `swap_pool_size:0`, and recurrent-state copy isn't needed by the transformer backends. Unit test: decode over a D2D-copied page == decode over the original (faithful replication). **Live**: demo-parallel-fork with a partial (non-page-aligned) prefill fires `copy_d2d` per fork (traced `copy ok: Kv D2D 1 page(s)`), forks produce identical coherent greedy output, 0 failures, and the demo's prefill-sharing win lands (2.09× less prefill, ~6× wall-time). **Deferred:** host-swap pool (D2H/H2D/H2H + CPU page pool) for memory-pressure eviction/restore — needed under load when KV exceeds capacity.
- [ ] D5. Retire the old frontier path (god-functions disappear).

**PARITY SCORECARD — toward 100% coverage of driver/cuda (frontier-only; legacy archs dropped by directive).**
Mapped the old driver's full non-arch surface (Explore agent). DONE + tested this push (ABI v31, 110 control tests, pie-server relinks):
`[x]` Adapter ops → no-op status 0 (exact parity). `[x]` Gemma builder arm + LoadedGemma (4th backend). `[x]` Non-token
sampler outputs RawLogits/Logprob(s)/Entropy/Dist → ForwardResponse side channels (host-computed; CSR matches the runtime
splitter). `[x]` Host-swap pool D2H/H2D/H2H + CPU page pool + `swap_pool_size` caps. `[x]` (earlier) token sampling
top-p/top-k/min-p; Copy D2D. REMAINING: `[ ]` BRLE custom + logit masks (#7, constrained decoding); `[ ]` KV cache dtypes
fp8/int8/fp4 (#8 — load_kv_scalar already decodes; need quant alloc + quantizing append); `[ ]` caps JSON enrichment
(#12, easy); `[ ]` GPT-OSS builder + mxfp4 (#5); `[ ]` Nemotron builder (#4, DEFERRED — forward is prefill-only, needs
rs_cache/decode rewrite); `[ ]` spec-decode/MTP (#9, large); `[ ]` flashinfer + CUDA-graph perf (#10, gates default);
`[ ]` TP/NCCL (#11, large).

**E. Validation & cleanup**
- [ ] E1. Integration tests (real-checkpoint e2e).
- [ ] E2. Minor: pre-existing lint; bound the `qgemm` build-time if the fan-out grows it.

Shortest path: **B1 → B3 → B2 → B4 → D1/D2 → D4**. C (perf) gates *flipping the
default*, not a correctness-parity demo. Much of A lands in parallel.

## North star

Maintainability. The current driver works and is fast, but two files
hold most of the accidental complexity:

- `driver/cuda/src/entry.cpp::run_impl` — a **1,286-line construction
  god-function** with **55 `is_*_arch` branches** over 10 arch
  booleans. Execution was already made polymorphic (`IModel`);
  construction was not.
- `driver/cuda/src/executor/executor.cpp` — **3,363 lines, 264
  file-local helpers**. The "run one forward + sample" path is buried
  under speculative-decode/MTP, CUDA-graph capture, and sampling
  concerns all tangled in one TU.

## The decision (redrawing the Rust/C++ line)

> **The control plane (every *decision*) moves to Rust. The device
> library (every *kernel sequence*) stays C++. The hot forward body
> stays a single FFI call**, so the boundary is coarse-grained and FFI
> overhead is irrelevant (~5 calls per token-step, each 100s of µs–ms).

This matches Pie's existing split (Rust runtime already owns
scheduling) and reuses the established Rust↔C++ machinery:
`driver/bridge` (`pie-bridge`: rkyv wire schema + shmem/inproc
transport), `driver/weight_loader` (`pie-weight-loader`, already called
*from* C++ today), and `server/build.rs` (CMake → static lib → link).

```
            BEFORE (driver/cuda)                  AFTER (driver/cuda_new)
  ┌─────────────────────────────┐      ┌────────────────────────────────────┐
  │ Rust runtime: schedule+shmem │      │ Rust control plane  (control/)      │
  └──────────────┬──────────────┘      │  builder · mem · executor           │
                 │ InProcVTable         │  sampler · spec · tp · arch registry│
  ┌──────────────▼──────────────┐      └──────────────┬─────────────────────┘
  │ C++ entry.cpp run_impl       │                     │ flat C ABI (coarse)
  │ executor.cpp (3.3k)          │      ┌──────────────▼─────────────────────┐
  │ sampling · MTP · graph       │      │ C++ device library  (device/)        │
  │ + kernels/ops/caches         │      │  kernels · ops · per-arch body       │
  └─────────────────────────────┘      │  caches · graph mechanics · context  │
                                        └──────────────────────────────────────┘
```

The codebase is already ~70% shaped for this line: `ForwardFn::Forward
Inputs` (`executor.hpp:81`) is already a POD bundle of device pointers +
ints — it crosses the FFI boundary almost verbatim as `PieForwardInputs`.
`IModel` is already the per-arch dispatch vtable. This rewrite
*formalizes a seam that is half-built* rather than inventing one.

## Layout

```
driver/cuda_new/
  PLAN.md                       ← this file
  device/                       ← C++/CUDA "thin device library" (libpie_cuda_device)
    CMakeLists.txt
    include/pie_cuda_device.h    ← the flat C ABI — the seam (load-bearing artifact)
    src/
      context.{hpp,cpp}          ← PieDevCtx: device, stream, cuBLAS handle
      abi.cpp                    ← ABI entry points → C++ impl
      forward/   (ported)        ← per-arch body() + prepare(), no host decisions
      cache/     (ported)        ← KvCache / MlaCache / RecurrentStateCache / SwapPool
      kernels/ ops/ (ported)     ← lifted from driver/cuda unchanged
  control/                      ← Rust "fat control plane" crate (pie-driver-cuda-native)
    Cargo.toml  build.rs
    src/
      lib.rs                     ← C-ABI entry; wires builder → executor → transport
      ffi.rs                     ← raw extern "C" over pie_cuda_device.h
      device.rs                  ← safe RAII wrappers over ffi (Drop = destroy)
      arch/mod.rs                ← trait Arch + ArchSpec + registry (replaces is_*_arch)
      builder.rs                 ← replaces entry.cpp::run_impl
      mem.rs                     ← replaces cuda_memory_planner.cpp
      executor.rs                ← replaces handle_fire_batch core
      sampler.rs                 ← replaces sampling_dispatch.cpp + seed helpers
      spec.rs                    ← replaces the MTP tangle in executor.cpp
      tp.rs                      ← replaces rank-0 broadcast loop
```

## The ABI seam (the contract)

See `device/include/pie_cuda_device.h`. Shape:

- **Opaque handles**: `PieDevCtx`, `PieWeights`, `PieKvCache`,
  `PieWorkspace`, `PieGraphExec`. Rust holds these as newtypes with
  `Drop` calling the destroy entry.
- **POD structs** cross by value/pointer: `PieForwardInputs` (≈ the
  existing `ForwardInputs`), `PiePrepareInputs`, `PieSampleParams`,
  `PieKvLayout`, `PieWorkspaceDims`.
- **Construction primitives** Rust drives: `pie_kv_alloc`,
  `pie_ws_alloc`, `pie_weights_bind` — replaces run_impl's alloc cascade.
- **Hot path** (one call each): `pie_upload_inputs`, `pie_prepare`,
  `pie_body`, `pie_sample`.
- **Graph**: `pie_graph_capture` / `pie_graph_launch` — Rust owns the
  *policy* (which bucket, when), C++ owns the `cudaGraph` *mechanics*.

The prepare/body two-phase split (the graph-safety constraint, see
`driver/cuda/src/executor/forward_graph.hpp:14-31`) **survives intact**:
it becomes two ABI calls instead of two `std::function`s.

## What replaces what

| New (Rust, `control/`)            | Old (C++)                                   |
|-----------------------------------|---------------------------------------------|
| `arch/` — `trait Arch` + `ArchSpec` | `bound_model.cpp` + 55 `is_*_arch` branches |
| `builder.rs`                      | `entry.cpp::run_impl` (1,286 lines)         |
| `mem.rs`                          | `cuda_memory_planner.cpp`                   |
| `executor.rs`                     | `handle_fire_batch` core                    |
| `sampler.rs`                      | `sampling_dispatch.cpp` + seed helpers      |
| `spec.rs`                         | MTP/spec-decode helpers in `executor.cpp`   |
| `tp.rs`                           | rank-0 broadcast loop in `executor.cpp`     |

Adding an arch becomes a **directory-local change**: one `impl Arch` in
Rust + one `device/src/forward/<arch>` TU. No central file grows.

## The `Arch` contract

Arch-specific knowledge becomes *data + a small trait impl*, not 1,286
lines of branches:

```rust
trait Arch {
    fn spec(&self, cfg: &HfConfig) -> ArchSpec;        // dims, layer schedule, kv layout, quant
    fn workspace_dims(&self, spec: &ArchSpec, max_tokens: usize) -> WorkspaceDims;
    fn id(&self) -> ArchId;                            // selects the C++ body
    fn caps(&self) -> ModelCapabilities;               // graph_safe, fused_argmax, ...
    fn drafter(&self) -> Option<Box<dyn Drafter>> { None }   // spec-decode opt-in
}
```

## Migration — strangler-fig, not big-bang

Each phase ships independently and keeps all archs working.

- **Phase 0 (this scaffold).** Tree, ABI header, crate skeleton, build
  wiring. `control/` is workspace-`exclude`d; nothing in the real build
  changes. *← we are here.*
- **Phase 1 — carve the seam.** Implement `libpie_cuda_device` by
  *lifting* `driver/cuda/src/{kernels,ops,cache}` behind the ABI, and
  make the *current* C++ executor call through it. Zero behavior change;
  proves FFI grain + graph capture across the boundary.
  - [x] device-memory ABI (`pie_cuda_malloc`/`free`/`memcpy_*`/`stream_sync`)
        + safe `DeviceBuffer` RAII + bf16 host helpers.
  - [x] full single-layer primitive set lifted verbatim with bf16 parity
        tests: `rmsnorm`, `residual_add`, `swiglu`, `rope`, bf16 `gemm`,
        `embed`, `argmax`, and naive paged attention
        (`attention_naive_paged_bf16`). Enough for a `llama_like` decoder
        layer + greedy decode without flashinfer.
  - [x] composed `llama_like` decoder layer (`forward/llama_layer.cu`).
  - [x] **complete forward** `pie_cuda_llama_forward_bf16`
        (`forward/llama_forward.cu`): embed → N layers → final norm →
        lm_head → argmax → next-token ids; validated vs an independent CPU
        f32 reference (logits parity + greedy-token validity).
  - [x] real `pie_ws_alloc` — pre-allocated `PieWorkspace` scratch reused
        across layers (replaces the layer slice's per-call scratch).
  - [ ] real KV-append scatter (multi-request / decode / multi-page); the
        slice uses a contiguous copy for single-request prefill. Then
        phase 2 (loader → `pie_weights_bind`), phase 3 (transport / executor
        routing), and flashinfer + the prepare/body two-phase (perf gate).
- **Phase 2 — move construction + load weights.** Port `run_impl` →
  `builder.rs`, one arch at a time.
  - ✅ `mem.rs`, `arch/` cores ported (parallel agents); `loader.rs` loads
    a real safetensors checkpoint → device → full forward (`LoadedLlama`,
    BF16 slice); `builder.rs::build` ties config → arch → mem → loader →
    alloc → `Model::prefill_greedy` (verified `config dir → tokens`).
  - ◻ completion: loader dtype/quant/sharding/GGUF + real-checkpoint test
    (reuse `pie-weight-loader`); full paged KV cache from `plan.num_pages`.
- **Phase 3 — move the loop + sampling.** `handle_fire_batch` core →
  `executor.rs` + `sampler.rs`. MTP stays C++ temporarily, called as a
  unit.
- **Phase 4 — move spec-decode.** Reimplement the MTP state machine in
  `spec.rs`; the `executor.cpp` tangle finally dissolves.
- **Phase 5 (optional) — forward dedup** in C++ (see *Conservative*).

### Build integration (at cutover)

- `server/build.rs::build_cuda()` points CMake at `driver/cuda_new/device`
  and links `libpie_cuda_device`; `control/` is added to the workspace
  `members` and linked like `driver/dummy`.
- `server/src/driver_ffi.rs` keeps the `pie_driver_cuda_run_inproc`
  symbol; during dev the new crate exports
  `pie_driver_cuda_native_run_inproc` to coexist with the old driver
  behind a `driver-cuda-new` Cargo feature.

## Conservative: the forward dedup is *not* a framework

`driver/cuda/src/model/` is 23.5K LOC of hand-orchestrated forwards
(411 → 2,138 lines each) over a shared primitive vocabulary (`embed`,
`rmsnorm`, `rope`, `residual_add`, `swiglu`, `attention_*`, `gemm`,
`moe_dispatch`). Resist a grand "forward framework" — ML forwards are
full of per-arch quirks (Gemma pre/post-norm, qk-norm, softcap, altup,
sliding window, MLA, Mamba recurrence). The realistic win is a shared
`decoder_forward` skeleton (embed → N×block → final-norm → lm_head, with
arch-provided block hooks) and *separate* skeletons for the genuinely
different paradigms (MLA, Mamba). Explicit over clever. v1 leaves most
forward bodies as a direct lift.

## Non-goals
- Not a kernel rewrite. Not a flashinfer/cutlass replacement.
- Not a forward framework (see above).
- Not a perf project — behavior- and perf-preserving by construction
  (the hot kernel sequences are lifted verbatim).
