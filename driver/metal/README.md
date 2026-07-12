# driver/metal

Embedded C++ driver for Pie on Apple Silicon. The shipped path is a native
Metal 4 decoder with one live model family, Qwen3.5 hybrid attention. It uses
the same 11-function shared-memory ABI as the CUDA and dummy drivers.

## Status

The production path is:

```text
abi.cpp -> Context -> batch/ -> model/qwen3_5/ -> kernels/
                    -> pipeline/ (CPU PTIR interpreter)
                    -> loader/ + store/ + mtl4_context.mm
```

`src/` contains only shipping code. `tools/rawmetal/` contains standalone
bring-up and diagnostic binaries, including the not-yet-live Gemma 4 work.
`tests/mlx/` is an opt-in MLX reference oracle used by smoke and parity tests;
the shipped driver never links it.

At boot, `pie-worker` first creates the Metal device context and reads device
facts, then `runtime/load-planner` compiles the checkpoint headers into a
versioned `LoadPlan`. `load_model` executes that plan into one resident
weights region; checkpoint payload bytes stay driver-local.

Metal intentionally executes PTIR programs on the CPU. CUDA uses a GPU tier-0
runner, but both drivers keep program, instance, and channel ownership under
`pipeline/`.

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

Useful options:

- `PIE_METAL_BUILD_TOOLS=ON` builds the native bring-up and diagnostic tools.
- `PIE_METAL_WITH_MLX=ON` enables only the MLX test oracle.
- `PIE_METAL_BUILD_TESTS=ON` builds the MLX smoke/KV targets.
- `PIE_METAL_BUILD_PARITY=ON` builds `tests/parity/parity_driver`.
- `PIE_METAL_MLX_PROVIDER=system` uses an installed MLX CMake package.

The development binary lands at `build/bin/pie_driver_metal`. Its default
configuration file is `dev.toml`; pass another path with `--config`.

System requirements: CMake, Ninja, the Xcode command-line tools (Metal /
Foundation / Accelerate frameworks), and an Apple Silicon GPU.

## Config

```toml
[model.driver]
type = "metal"
device = ["metal:0"]
activation_dtype = "bfloat16"
```

KV pages, page size, recurrent-state slots, and forward limits are derived at
startup and reported in `DriverCapabilities`. The live path currently retains
the 4096-token resident-ring limit.
