# driver/ggml

Portable ggml driver for Pie. Sibling to `driver/cuda` (native CUDA +
flashinfer), `runtime/` (Rust), and `pie/` (the Python driver). Replaces the
Python forward-pass executor with a C++ binary built on raw `libggml`
(no `libllama`) that loads HuggingFace safetensors directly — same
checkpoint format Pie's reference Python driver uses. Talks to the runtime
over the shmem fast path defined in `runtime/src/shmem_ipc.rs`.

The binary built here (`pie_driver_portable`) is intended to be spawned and
managed by the Python wrapper module `pie_driver_portable` (forthcoming).

> **Status:** M2 — paged KV pool that honors Pie's runtime-supplied page
> table (`kv_page_indices` / `kv_page_indptr` / `kv_last_page_lens` from
> the BPIQ wire format). Per-request `ggml_flash_attn_ext` with
> `ggml_get_rows`-based KV gather; concurrent contexts share a single
> global page pool (`total_pages × page_size` slots) — no per-context
> slab cap. Multi-request graph builder, `BPIS` flat response. Smoke-tested
> on Qwen3-0.6B BF16 CPU with 3 concurrent contexts. M4 (full sampler
> suite) next. Roadmap below.

## Architecture

```
┌────────────┐  shmem fast path   ┌──────────────────────┐
│ runtime/   │ ◄────────────────► │ driver/ggml (this)   │
│ (Rust,     │   PIE3 control +   │  - decode batch      │
│  client)   │   BPIQ schema      │  - run forward       │
└────────────┘                    │  - sample            │
                                  └──────────────────────┘
```

The Rust runtime owns scheduling, KV-cache page tables, and inferlet
execution. `driver/ggml` owns model weights and per-batch forward-pass
execution via ggml. They share state only through the shmem channel (no
sockets, no RPC).

## Why ggml as a second backend?

`driver/cuda` targets NVIDIA-only paths via flashinfer. `driver/ggml` covers
everything else ggml can run on:

- CPU (no GPU required)
- CUDA (via `-DGGML_CUDA=ON`)
- Metal (via `-DGGML_METAL=ON`, on macOS)
- Vulkan (via `-DGGML_VULKAN=ON`)
- HIP/ROCm (via `-DGGML_HIPBLAS=ON`)

The shmem protocol and binary contract are identical across drivers, so the
Rust runtime treats them interchangeably.

## Dependencies

System packages (Debian/Ubuntu):

```bash
sudo apt-get install -y cmake ninja-build
```

CMake-managed via CPM:

- `ggerganov/llama.cpp@b6993` — vendored for the bundled `ggml` library
  only. The `llama` lib (and its ~70 model TUs) is excluded from the build
  via `EXCLUDE_FROM_ALL`. The llama.cpp source tree remains available as a
  porting reference for per-arch graph builders (`src/models/*.cpp`) and
  the per-stream KV cache layout (`src/llama-kv-cache.cpp`).
- `marzer/tomlplusplus@3.4.0` — config parsing
- `CLIUtils/CLI11@2.5.0` — CLI args
- `nlohmann/json@3.12.0` — capability handshake + safetensors header parse

## Build

```bash
cd driver/ggml
cmake -S . -B build -G Ninja
cmake --build build
```

To enable a GPU backend, pass the matching `GGML_*` flag:

```bash
cmake -S . -B build -G Ninja -DGGML_CUDA=ON      # NVIDIA
cmake -S . -B build -G Ninja -DGGML_METAL=ON     # macOS
cmake -S . -B build -G Ninja -DGGML_VULKAN=ON    # Vulkan
```

The binary lands at `build/bin/pie_driver_portable`.

## Run (scaffold)

```bash
./build/bin/pie_driver_portable --config dev.toml
```

Edit `dev.toml` first to point `[model].hf_path` at a local HuggingFace
snapshot directory (the one containing `config.json` + `model.safetensors`,
typically under `~/.cache/huggingface/hub/<repo>/snapshots/<sha>/`). For
example, after `hf download Qwen/Qwen3-0.6B`, the snapshot lives at
`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<sha>/`.

Then start the Rust runtime pointed at the same shmem name. The runtime
will be the client; `pie_driver_portable` decodes incoming batches and runs
the forward pass.

For an offline smoke test (no Pie runtime needed) you can pass a prompt
directly. Single-context:

```bash
./build/bin/pie_driver_portable --config dev.toml \
    --test-prompt-tokens "785,6722,315,9625,374" \
    --test-max-new 16
# GENERATED 12095 13 576 6722 315 15344 374 ...   ("Paris. The capital of Italy is Rome ...")
```

Multi-context (exercises the multi-stream KV path) — pipe-separated prompts:

```bash
./build/bin/pie_driver_portable --config dev.toml \
    --test-multi-prompt-tokens "785,6722,315,9625,374|9707,11,847,829,374|12522,5193,264,882" \
    --test-max-new 12
# GENERATED[0] 12095 13 576 6722 315 ...   ("Paris. The capital of...")
# GENERATED[1] 444 2210 13 358 2776 ...    ("Lina. I'm...")
# GENERATED[2] 11 1052 1033 220 18 ...     (", there were 3000...")
```

The token IDs come from the model's tokenizer (`tokenizer.json` in the
snapshot dir). A small Python helper in your editor can encode/decode for
sanity checks during development.

## Roadmap

- [x] **M1.1** CMake + CPM dependency wiring
- [x] **M1.1** Shmem IPC server (mirrors `runtime/src/shmem_ipc.rs`)
- [x] **M1.1** BPIQ flat schema decoder
- [x] **M1.2** HF safetensors mmap reader (`src/safetensors.{hpp,cpp}`)
- [x] **M1.2** HF `config.json` parser → `Hparams` (`src/hf_config.{hpp,cpp}`)
- [x] **M1.2** Model wrapper binding HF tensors to ggml backend (Qwen3 only)
- [x] **M1.2** Real capabilities handshake (`READY <json>`)
- [x] **M1.3** Qwen3 graph builder + greedy single-context forward (`src/forward.{hpp,cpp}`, `src/kv_cache.{hpp,cpp}`)
- [x] **M1.4** `BPIS` flat response writer (M8 adds msgpack mode)
- [x] **M2** Paged KV pool honoring runtime page table + multi-request graph (`src/kv_cache.{hpp,cpp}` `KvCachePaged`, `src/forward.cpp` per-request `flash_attn_ext` over `ggml_get_rows` gather)
- [x] **M4** Sampler suite — greedy / multinomial / top-k / top-p / min-p / top-k-then-top-p with per-row seeded PCG32 (`src/sampler.{hpp,cpp}`); host-side after `ggml_backend_tensor_get`
- [x] **M5** Logit masks — BRLE-encoded per-request vocab masks (`apply_brle_logit_mask` in `src/sampler.cpp`); applied pre-sample to filter the vocab. Verified end-to-end via `inferlets/constrained-decoding` (Lark-grammar JSON).
- [x] **M6** Custom attention masks — per-token BRLE override of the default causal mask (`build_attn_mask_f16` in `src/forward.cpp`). Verified via `inferlets/attention-sink` (sink + sliding window). Sliding-window attention reuses the same builder (M12 archs that use SWA plug in here).
- [x] **M7** KV swap + page-copy RPCs. Aux IPC unix socket between Python wrapper and binary (`src/aux_ipc.hpp` wire format, `src/aux_server.{hpp,cpp}` listener thread, `pie_driver_portable/worker.py::_AuxClient`). Binary owns optional `HostSwapPool` (`cpu_pages × page_size`); per-layer copies via `ggml_backend_tensor_get/set` for D2H/H2D, host-side `memcpy` for H2H, get→set round-trip for D2D. Verified end-to-end: `copy_d2d` works for forking inferlets (best-of-n produced 3 independent candidates from a single prompt context), `swap_pool_size` advertised in capabilities when `cpu_pages > 0`.
- [x] **M8** Speculative decoding — verifier (linear drafts). Driver parses `spec_token_ids` / `spec_position_ids` / `spec_indptr` from BPIQ, expands per-request sampler slots to `1 + n_drafts`, runs sampling at the last-pending position + each draft position, and walks the resulting predictions against the drafts: accept matching prefix, then 1 bonus token (replacing the first rejected draft, or appended after all drafts on a clean run). Variable-length per-request token output flows through both the flat and msgpack response writers. The runtime's NGRAM drafter handles draft generation upstream — no driver-side drafter in v1; output `spec_tokens`/`spec_positions` are empty. Verifier path is regression-tested end-to-end via `inferlets/text-completion-spec` (runs cleanly; runtime emits 0-length drafts for non-repetitive prompts so the walk doesn't fire, but the spec arrays do flow through and the dense fallback emits identical greedy output).
- [x] **M9** Static LoRA — adapter loader + graph integration. New aux IPC method `LoadAdapter` (path payload). Wrapper writes LoRA bytes to temp file (`~/.pie/portable-lora-<id>.safetensors`) and forwards the path; binary parses HF-PEFT tensor naming (`base_model.model.model.layers.{N}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight`) and registers in `AdapterPool` keyed by `adapter_ptr`. Plan_() reads `adapter_indices` from BPIQ, validates single-adapter-per-batch, looks up the adapter in the pool. Graph applies LoRA delta `y += scale * (B @ (A @ x))` at q/k/v/o matmuls when active. CMAES (init/update/save) returns `not_implemented`.
- [x] **M10** Logprob / Logprobs / Entropy / RawLogits / Distribution samplers + msgpack response writer (`src/msgpack.hpp` hand-rolled writer, `src/sampler.cpp::sample()`). Verified end-to-end on `inferlets/raw-logits-demo`: 50 iterations, full 151,936-vocab Qwen3 logits, 100% argmax-match with greedy sampler, +3 ms per call from msgpack-mode encode.
- [x] **M11** GPU backend (`-DGGML_CUDA=ON`) + multi-stream attention packing — pure-decode batches (every request `n_tokens=1`, no custom masks) get a single `ggml_flash_attn_ext` per layer with `ne3 = n_request` instead of N×L per-request calls. Throughput at 512×100: **621 → 1419 tok/s aggregate** (2.3×). GPU-side sampler port remains future work.
- [x] **M12** Architecture expansion. **Verified end-to-end on real HF checkpoints** (binary direct, decoded outputs match expected greedy continuations):
  - **qwen3** (Qwen3-0.6B) — Q/K-norm
  - **qwen2** (Qwen2.5-0.5B) — QKV biases
  - **llama3** (Llama-3.2-1B) — incl. 3.1+ NTK-by-parts RoPE via precomputed `freq_factors`
  - **mistral** (Mistral-7B-v0.3) — same backbone as llama3, optional SWA
  - **gemma2** (gemma-2-2b-it) — `(1+w)` RMSNorm, attn + final softcap (50/30), GeGLU, embed scale, alternating SWA layers, custom Q scaling, pre/post-attn + pre/post-FFN norms
  - **gemma3** (gemma-3-1b-it) — gemma2 features plus per-layer Q/K-norm + dual RoPE base (1e6 global / 1e4 sliding) on a 6-layer iSWA pattern
  - **phi3** (Phi-3-mini-4k-instruct) — fused QKV (`qkv_proj.weight` → q/k/v slices) + fused gate_up (`gate_up_proj.weight` → gate/up slices) at load time
- [ ] **M12 (post-v1)** **gemma4** stub raises a clear error — Per-Layer Embeddings + dual head_dim per layer are structurally distinct.
- [x] **M12 MoE infrastructure** — the full machinery is implemented (untested due to model size):
  - `Hparams` parses `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `norm_topk_prob`, `num_local_experts` (Mixtral alias)
  - `Model::declare_stacked_experts_(...)` allocates one `[hidden, ff, n_experts]` tensor per layer per expert kind (gate / up / down) and populates each slot from the corresponding HF safetensor at load time
  - Per-arch loaders for **qwen3_moe** (Qwen3-MoE: Qwen3-30B-A3B etc.), **mixtral** (Mixtral-8x7B w/ `block_sparse_moe.experts.E.{w1,w2,w3}` naming), **gpt_oss** (reuses qwen3_moe layout — sinks via existing `flash_attn_ext_add_sinks`). **qwen3_5** / **qwen3_5_moe** (Qwen 3.5 / 3.6) are supported via a dedicated graph builder — Gated Delta Networks (gated-delta-rule linear attention with per-request recurrent state in `StateCache`), mrope (IMROPE), and `attn_output_gate`.
  - `build_moe_ffn()` graph helper: `softmax(router @ cur)` → `top_k` → optional renormalize → `mul_mat_id` × 3 (gate/up/down) → weighted sum across selected experts. Mirrors `src/llama-graph.cpp::build_moe_ffn`.
  - `ArchSpec::n_experts > 0` switches the layer FFN dispatch to MoE.
  - **Untested**: smallest pure-MoE checkpoints (Qwen3-MoE-30B, Mixtral-8x7B 47B, gpt-oss 120B+) exceed 46 GB on our A40. Qwen2-MoE checkpoints (Qwen1.5-MoE-A2.7B, 14 GB) fit but use additional **shared-expert** machinery (qwen2_moe-specific) not yet wired in graph. **First test** will need either smaller-dtype MoE weights (FP8/INT4) or a smaller checkpoint.

Single shared graph builder driven by `ArchSpec` (qkv_bias / qk_norm / pre_ffn_norm / post_attn_norm / post_ffn_norm / scale_embed_by_sqrt_d / attn_softcap / final_softcap / sliding_window / layer_pattern / query_pre_attn_scalar / ffn_use_gelu / norm_weight_plus_one) + optional per-arch `rope_local_base_freq`. Per-arch loader chooses tensor names; the graph builder branches on flags.
- [ ] **Wrapper** `pie_driver_portable` Python module (spawn + shmem handshake)
