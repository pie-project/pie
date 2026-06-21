# driver/cuda

Embedded C++/CUDA driver for Pie. It owns model weights and CUDA forward
passes; the Rust runtime owns scheduling, inferlet execution, and the
shared-memory control path.

## Build

Normally this driver is built into `pie`:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
  cargo build -p pie-worker --release --features driver-portable,driver-cuda
```

For driver-only development:

```bash
cd driver/cuda
cmake -S . -B build -G Ninja
cmake --build build
```

System requirements: CMake, Ninja, a CUDA toolkit, and an NVIDIA GPU with
SM 8.0 or newer.

## Config

```toml
[model.driver]
type = "cuda_native"
device = ["cuda:0"]
tensor_parallel_size = 1
activation_dtype = "bfloat16"

[model.driver.options]
gpu_mem_utilization = 0.90
memory_profile = "auto"
```

`memory_profile` may be `auto`, `latency`, `balanced`, `throughput`, or `capacity`.
The driver derives forward capacity, Qwen3.5 state slots, and KV pages after
weights load.

Run `pie driver cuda-native doctor` to confirm the installed `pie` binary has
the driver compiled in and can see the GPU.
