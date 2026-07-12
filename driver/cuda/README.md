# CUDA driver

The native C++/CUDA backend owns model weights, device stores, and forward
execution. Rust owns scheduling, inferlet execution, and the control plane.

## ABI

`interface/driver/include/pie_driver_abi.h` is the frozen boundary. The driver
implements the shared direct exports, including the two boot calls:

`pie_cuda_create`, `pie_cuda_load_model`, `pie_cuda_register_program`, `pie_cuda_register_channel`,
`pie_cuda_bind_instance`, `pie_cuda_launch`, `pie_cuda_copy_kv`,
`pie_cuda_copy_state`, `pie_cuda_resize_pool`, `pie_cuda_close_instance`,
`pie_cuda_close_channel`, and `pie_cuda_destroy`.

`src/abi.cpp` only validates and converts ABI handles. `src/context.*` is the
composition root behind that boundary.

## Modules

| Path | Responsibility |
| --- | --- |
| `src/batch/` | Compose and execute one submitted batch |
| `src/pipeline/` | Own PTIR programs, instances, channels, and dispatch |
| `src/model/` | Architecture registry, family weights, and forwards |
| `src/ops/` | Reusable attention, GEMM, MoE, and state-space wrappers |
| `src/kernels/` | Leaf CUDA kernels; no higher-layer includes |
| `src/store/` | Long-lived KV/MLA/DSA/state/swap storage and memory planning |
| `src/loader/` | Execute runtime-compiled storage programs into `WeightStore` |

`tensor.*` and `distributed.*` are shared core types.

## Build

Build the driver as part of the Rust `pie` binary (see `worker/README.md`):

```sh
CUDACXX=/usr/local/cuda/bin/nvcc \
  cargo build -p pie-worker --release --features driver-cuda
```

The binary lands at `target/release/pie`.

For the standalone diagnostic shim:

```sh
cmake -S driver/cuda -B target/cuda -G Ninja
cmake --build target/cuda --target pie_driver_cuda
target/cuda/bin/pie_driver_cuda --config driver/cuda/dev.toml
```

The standalone target validates device creation. Full model boot is composed by
`pie-worker`, which compiles and supplies the mandatory storage program before
serving.

To run the same commands on a GPU workstation:

```sh
ssh workstation 'cd /path/to/pie && \
  cmake -S driver/cuda -B target/cuda -G Ninja && \
  cmake --build target/cuda'
```

## Structure and tests

The text-only structural gate needs neither CUDA nor a GPU:

```sh
./driver/cuda/check-structure.sh
```

After building all test targets on a CUDA host:

```sh
cmake --build target/cuda
ctest --test-dir target/cuda --output-on-failure
```

Some parity harnesses are build-only because they require external reference
dumps; they are intentionally not registered with CTest.

## Configuration boundary

`driver/cuda/dev.toml` is only the C++ `pie_driver_cuda` shim's direct config
and uses `[model]`, `[batching]`, `[distributed]`, and `[runtime]`.

The Rust `pie` and `pie-worker` processes use a separate user-facing
configuration and translate it into the frozen driver ABI. That service
configuration is not accepted by `pie_driver_cuda`.
