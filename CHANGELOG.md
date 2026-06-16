# Changelog

All notable changes to Pie are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-15

A large release centered on **multimodal serving**, **native speculative
decoding**, **weight/KV quantization**, a rewritten **scheduler and weight
loader**, and a broad set of **new model architectures**.

### Added

#### Multimodal (vision, video, audio)
- Model-agnostic multimodal API for **vision, video, and audio** — both input
  and output. Inferlets hand the host raw encoded bytes; the host owns
  decoding and preprocessing, so no model-specific constants leak into inferlet
  code.
- `Context` bindings: `append_image`, `append_audio`, `append_video`, plus
  `Image`/`Audio`/`Video::from_bytes` with metadata accessors
  (`token_count()`, `grid()`).
- Text-to-speech output: `model.speak(text).speaker(id).generate()` returning
  self-describing speech (`to_wav()` with `sample_rate`/`channels`).
- `inferlet::http::fetch` for pulling remote media inside an inferlet, and
  simplified `media` bindings across the Rust/Python/JS SDKs.
- Example inferlets: `image-qa`, `audio-qa`, `video-qa`, `tts`
  (+ `image-qa-bench`), and a multimodal image Pie-vs-vLLM benchmark.

#### Speculative decoding
- **Native Multi-Token-Prediction (MTP) speculation** — lossless, in-driver
  draft generation for hybrid GDN models. Opt-in via
  `enable_system_speculation` with `mtp_num_drafts` and an optional
  `mtp_assistant_snapshot_dir`.
- MTP system speculators wired for **Gemma-4**, **Qwen3.5**, and
  **Qwen3.5-MoE**, with fused GEMV+argmax, graph argmax, and per-step position
  handling on the draft path.

#### Quantization
- Runtime weight quantization: **fp8, int8, fp4/mxfp4**, with a native MXFP4
  loader repack path.
- **KV-cache quantization** formats (fp8_e4m3 / fp8_e5m2 / … / nvfp4),
  selectable via `kv_cache_dtype`.
- Dense pre-quantized **block-FP8** weights (e.g. Qwen3-FP8) in the loader.

#### New model architectures (CUDA native)
- **GLM-5.1** (MLA on Blackwell, RoPE, FP4 quant, DSA indexer).
- **DeepSeek V4** and **Kimi** (MLA, fused q_a/kv_a and gate+up projections).
- **Nemotron-H**.
- **Qwen3-MoE** (Qwen3-30B-A3B), **Qwen3.5 / Qwen3.6** (incl. MTP drafters),
  and **Qwen3-VL** vision.
- **Gemma-4** (vision + audio in) and **CSM** (audio out).

#### Drivers & runtime
- New **TensorRT-LLM driver** backend.
- `runtime::launch` API for inferlet-to-inferlet invocation.
- Runtime-managed `rs_cache` support and a low-latency IPC profile with a
  polling channel.
- Automatic driver **capacity planning** (SM-aware decode capacity, throughput
  arena selection, auto admission).
- Portable driver: **sharded GGUF** support, MoE experts from stacked GGUF
  tensors, single-`.gguf` `hf_repo` targets, and a tokenizer minted from GGUF
  KV metadata.
- Startup banner for `pie serve`.

#### Examples & SDK
- New example inferlets: **AsyncLM** (async function calling), **Reflexion**
  reasoning, hierarchical attention + MCTS, modular cache (Rust/Python/JS),
  self-correct, and custom-speculator demos.
- E2E tests for raw-completion inferlets and a per-arch SHA smoke gate
  (`benches/smoke_deterministic`).

### Changed

#### Performance — scheduler & runtime rewrite
- Main scheduler loop now runs on a **dedicated synchronous OS thread** with
  crossbeam channels, firing batches synchronously (closes the batch-coalescing
  gap that caused singleton-decode fires under load).
- **Chain-extender worker pool** replaces per-context tasks; pure-decode fast
  path extended to chain continuations.
- Switched the global allocator to **mimalloc**; capped tokio worker threads;
  `dashmap` for the staged batch; deferred dispatch drops to shrink the
  inter-fire gap.
- **Chunked prefill** in the scheduler; numerous CUDA attention, TP1/TP2
  throughput, and GDN/FLA kernel-fusion optimizations.
- Result: pie now matches or exceeds vLLM throughput on the measured Qwen3
  configurations.

#### Weight loader
- **Rust is now the canonical weight loader** — an algebraic, profile-driven
  load-plan pipeline (physical byte-plan layer, streamed checkpoint tensors,
  metadata-backed FP8 decode) replacing the legacy C++ storage compiler.

#### Build & packaging
- **Vendored FlashInfer CUTLASS MoE launchers**, removing the build-time Python
  codegen and its incidental `tvm_ffi` / `pynvml` / `tqdm` dependencies.
- **CUDA 12.8 / 13 dual-build** support and `sm120` builds. **CUDA 12.8 is the
  minimum toolkit** — FlashInfer's Hopper (Mamba SSU) and FP4 kernels require 12.8+.
- Slim CUDA Docker image for `pie-server` (driver-cuda).
- Version bumped to **0.4.0** across all crates, packages, and install scripts.

### Removed
- The **`dev` reference driver** (use `cuda_native` or the portable driver).
- The C++ storage compiler / layout planner and C++ storage-compiler
  compatibility shims (superseded by the Rust loader).

### Fixed
- Graceful error instead of a panic on compact wasm import sections.
- Windows portability: `getpid()` in the portable driver entry; explicit
  unsupported-driver errors for dev/vllm/sglang on Windows.
- Graph-cache invalidation on page-index growth; lazy-delete of stale
  restore-queue entries.
- Paged-attention decode planner `padded_batch_size` / tile-count mismatch.
- XQA `gqa=5` graph-capture failure under TP>1.
- GGUF portable driver build failure; first-shard validation in
  `open_sharded`.
- Serialized forked-branch generation; CUDA quantized generation paths.
- GPU-local CPU-affinity parsing in the benchmarks (NIC columns in the
  `nvidia-smi` topology), now also applied to `pie_bench`.

[0.4.0]: https://github.com/pie-project/pie/compare/0.3.0...0.4.0
