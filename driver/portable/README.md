# driver/portable

Embedded ggml-backed driver for Pie. It loads HuggingFace safetensors directly
and runs through the same shared-memory driver ABI as the CUDA and dummy
drivers.

Use this driver for CPU inference, local development, or non-NVIDIA backends
supported by ggml.

Multimodal support (Qwen3-VL / Gemma-4 vision, Gemma-4 audio input, CSM-1B TTS)
is documented in [MULTIMODAL.md](MULTIMODAL.md).

## Build

The normal path is through the `pie` binary:

```bash
cargo build -p pie-server --release
```

For driver-only development:

```bash
cd driver/portable
cmake -S . -B build -G Ninja
cmake --build build
```

Enable a ggml backend with the matching CMake flag:

```bash
cmake -S . -B build -G Ninja -DGGML_CUDA=ON
cmake -S . -B build -G Ninja -DGGML_METAL=ON
cmake -S . -B build -G Ninja -DGGML_VULKAN=ON
```

The development binary lands at `build/bin/pie_driver_portable`.

## Config

```toml
[model.driver]
type = "portable"
device = ["cpu"]
activation_dtype = "bfloat16"
```

The driver prefers the best compiled ggml backend and falls back to CPU. KV
pages, page size, and forward limits are derived at startup and reported in
`DriverCapabilities`.

Weight loading is compiled by the shared Rust storage-program loader for every
portable backend. CPU, CUDA, Metal, and Vulkan builds therefore use the same
typed load plan and executor; backend-specific differences are handled by
ggml's `ggml_backend_tensor_set`/`get` APIs after the storage program has
chosen the exact byte ranges and transforms.
