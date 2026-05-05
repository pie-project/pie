# driver/portable

Embedded ggml-backed driver for Pie. It loads HuggingFace safetensors directly
and runs through the same shared-memory driver ABI as the CUDA and dummy
drivers.

Use this driver for CPU inference, local development, or non-NVIDIA backends
supported by ggml.

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

[model.driver.options]
max_batch_tokens = 10240
max_batch_size = 512
max_num_kv_pages = 1024
```

The driver prefers the best compiled ggml backend and falls back to CPU.
