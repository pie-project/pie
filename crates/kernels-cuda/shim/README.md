# `shim/` — the headers NVRTC does not ship

NVRTC before CUDA 13.3 carries no device headers. A `#include <cuda_fp16.h>`
under it finds nothing and the compile stops, so every header the device text
names has to be answered by a file of the same name. These are those files,
written here: none is copied from a CUDA toolkit or a CCCL checkout (see
`../NOTICE`).

## The name is the contract

NVRTC has no `-I` and no search list. It matches the literal string in the
`#include` directive against the names it is handed, and `src/source.rs`
hands every file in this directory under its path relative to `shim/`. So a
file here is never renamed: `cuda_fp16.h` answers for NVIDIA's `cuda_fp16.h`
and stops answering under any other name.

Each file declares only what the device text reaches through it. Guarding an
include out (`#ifndef __CUDACC_RTC__`) instead of answering it is not a
substitute: the host headers' names reach device code (`uint32_t`,
`CUDA_VERSION`, the fp4 vector types), and a guard deletes them, sometimes with
no diagnostic.

## The files

| file | what it answers |
|---|---|
| `cstdint` | the fixed-width integers and the `*_C` literal macros |
| `cstddef` | `std::size_t` |
| `cstring` | `std::memcpy`, as a byte loop (device code type-puns with it) |
| `type_traits` | the traits a dtype dispatch is written in (`is_same_v`, `conditional_t`, ...) |
| `cuda.h`, `cuda_runtime.h` | `CUDA_VERSION` and the runtime's device-side types |
| `cuda_fp16.h`, `cuda_bf16.h` | `__half`/`__nv_bfloat16` and their vector forms over Pie's `f16`/`bf16` (sm_80 up) |
| `cuda_fp8.h` | e4m3/e5m2: the sm_89 conversions, integer arithmetic below |
| `cuda_fp4.h` | the e2m1 vector types |
| `cuda_pipeline.h` | `__pipeline_memcpy_async` / commit / wait |
| `cooperative_groups.h` | the grid and thread-block groups `attn/fa2` uses |
| `math_constants.h` | `CUDART_INF_F` and its relatives |
| `cuda/cmath` | `cuda::fast_mod_div<unsigned>`, `attn/fa2/fastdiv.cuh`'s one use |
| `cuda/std/limits` | `cuda::std::numeric_limits<float>::infinity()` |

Every file here has at least one includer under `kernels/`; a file without
one is deleted, not kept for later (`array`, `cassert` and
`cuda/std/optional` went with the code that included them).

## Adding one

Add a file only when a carried unit includes a header NVRTC cannot find, and
give it only what that unit reaches. `every_unit_lowers` (`src/jit/nvrtc.rs`)
compiles every unit against this directory.
