# driver/metal

Embedded C++ driver for Pie targeting Apple Silicon (macOS, arm64). It uses
[MLX](https://github.com/ml-explore/mlx) for compute and native Metal shaders
where MLX lacks coverage, and runs through the same shared-memory driver ABI as
the CUDA and dummy drivers.

This is the macOS backend for Apple Silicon systems.

## Status

Direct-ABI surface: builds, links, and registers with `pie-worker`. The Linux
stub validates bind/layout/callback semantics; the Apple path layers MLX/Metal
compute on the same entry surface.

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
default for the direct surface; the compute layer turns it on.

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
