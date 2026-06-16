# Changelog

All notable changes to Pie are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-15

- **Multimodal**: vision/video/audio input and audio output via a model-agnostic media API (`append_image`/`append_audio`/`append_video`; TTS via `model.speak`).
- **Native MTP speculative decoding**: lossless, in-driver draft generation (Gemma-4, Qwen3.5, Qwen3.5-MoE).
- **Quantization**: runtime weight quant (fp8/int8/fp4/mxfp4) and KV-cache quant (`kv_cache_dtype`).
- **New model architectures**: GLM-5.1, Nemotron-H, Kimi/DeepSeek, Qwen3-MoE, Qwen3.5/3.6, Qwen3-VL, Gemma-4, CSM.
- **TensorRT-LLM driver**.
- **Scheduler & weight-loader rewrite**: pie now matches or exceeds vLLM throughput on Qwen3; Rust is the canonical weight loader.
- **Removed**: the `dev` reference driver (use `cuda_native` or the portable driver).
- **Build**: CUDA 12.8 is the minimum toolkit; CUDA 12.8/13 dual-build with sm120.

[0.4.0]: https://github.com/pie-project/pie/compare/0.3.0...0.4.0
