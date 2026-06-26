# driver/metal

Embedded C++ driver for Pie targeting Apple Silicon (macOS, arm64). It uses
[MLX](https://github.com/ml-explore/mlx) for compute and native Metal shaders
where MLX lacks coverage, and runs through the same shared-memory driver ABI as
the CUDA and dummy drivers.

This is the macOS backend for Apple Silicon systems.

## Status

Foundation skeleton: builds, links, and registers with `pie-worker`, and serves
a Health + (stubbed) Forward round-trip through the in-process `InProcServer`.
MLX compute (ops/executor/kernels), model graphs, and the weight loader are
landed incrementally on top of this seam.

## Build

The normal path is through the `pie` binary:

```bash
cargo build -p pie-worker --no-default-features --features driver-metal
```

For driver-only development:

```bash
cd driver/metal
cmake -S . -B build -G Ninja
cmake --build build
```

`PIE_METAL_WITH_MLX=ON` (CMake option) fetches and links MLX. It is `OFF` by
default for the bare skeleton; the compute layer turns it on.

The development binary lands at `build/bin/pie_driver_metal`.

System requirements: CMake, Ninja, the Xcode command-line tools (Metal /
Foundation / Accelerate frameworks), and an Apple Silicon GPU.

## Config

```toml
[model.driver]
type = "metal"
device = ["metal:0"]
activation_dtype = "bfloat16"
```

KV pages, page size, and forward limits are derived at startup and reported in
`DriverCapabilities`.
