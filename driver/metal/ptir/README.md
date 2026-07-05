# driver/metal/ptir — PTIR sampling-IR on Metal (cross-backend parity track)

Metal duals of charlie's CUDA sampling-IR primitives. Every op is validated
**bit-exact** against echo's canonical Rust reference semantics
(`interface/sampling-ir/src/eval.rs`) so the chain

```
Metal kernel  ==  CPU reference  ==  echo golden vector  ==  CUDA kernel
```

holds byte-for-byte. This is the macOS/Metal cross-backend oracle track — it
runs on Apple Silicon and has **no MLX dependency** and no GPU contention with
the model-forward path.

## Toolchain reality (this box: M1 Max, CLT-only)

No offline `metal`/`metallib` compiler (Command Line Tools only, no full Xcode
Metal Toolchain). Kernels compile at **runtime** via
`[MTLDevice newLibraryWithSource:]` — same constraint as raw_metal Phase-0.
Verified: `clang++` links `Metal` + `Foundation`, runtime shader compile,
compute dispatch, and Shared/UMA readback all work on Apple M1 Max.

## Layout

| file | role |
|------|------|
| `kernels/sampling_ir.metal`   | the op kernels (runtime-compiled) |
| `reference.hpp`               | CPU reference — bit-exact port of `eval.rs` (the interim oracle) |
| `metal_harness.hpp/.mm`       | `MetalHarness`: device + runtime compile + Shared-buffer bind + 1-D dispatch + bit-exact compare |
| `ptir_metal_test.mm`          | per-op parity tests (`PTIR_METAL_TEST_OK` on success) |
| `CMakeLists.txt`              | standalone build (Metal + Foundation, no MLX) |

## Ops implemented (M2)

| op | Rust `Op` | semantics |
|----|-----------|-----------|
| `mask_apply_packed`        | `MaskApply` | `out[j] = bit_j(mask) ? logits[j] : -inf`, `bit_j=(mask[j>>5]>>(j&31))&1` (vector) |
| `mask_apply_packed_matrix` | `MaskApply` | per-row packed bitmap over `[rows, vocab]` (`matrix_mask_apply_packed`) |
| `dselect_f32`              | `Select`    | `out[i] = cond[i] ? a[i] : b[i]` with per-operand len-1 broadcast |
| `broadcast_matrix_f32`     | `Broadcast` | rank≤2 left-aligned row-major broadcast (scalar→`[k,vocab]`, `[m]`→`[m,n]`) |

## Build & run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/ptir_metal_test        # -> PTIR_METAL_TEST_OK
```

Direct (no CMake):

```sh
clang++ -std=c++17 -fobjc-arc -O2 -DPTIR_KERNELS_DIR="\"$PWD/kernels\"" \
  -x objective-c++ metal_harness.mm ptir_metal_test.mm \
  -framework Metal -framework Foundation -o ptir_metal_test && ./ptir_metal_test
```

## Golden wiring (pending echo)

The CPU reference is the **interim** oracle. Once echo's value/output golden
vector files land (e.g. `matrix_mask_apply_packed`, matrix `dselect`,
`broadcast_matrix(neg_inf,k,vocab)`), a golden reader will assert
`CPU-reference == golden bytes`, closing the loop so Metal is certified against
the exact same bytes CUDA consumes.
