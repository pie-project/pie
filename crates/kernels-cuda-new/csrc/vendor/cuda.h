// `cuda.h` for NVRTC -- the two things the FlashInfer closure actually takes from the
// driver header, and nothing else.
//
// Three files in the vendored closure include `<cuda.h>`: `vec_dtypes.cuh`,
// `profiler.cuh` and `attention/mla_params.cuh`. The obvious move is a guard, and it was
// measured and rejected -- twice over. Guarding it in `profiler.cuh` cost 26 errors:
// every `uint32_t` in that file's `%%globaltimer`/`%%smid` wrappers came from the driver
// header's transitive `<cstdint>`. Guarding it in `vec_dtypes.cuh` cost something worse
// than an error, because it compiled: with `CUDA_VERSION` unset, `#if CUDA_VERSION >=
// 12080` went false, `<cuda_fp4.h>` was never included, and the fp4 vector types
// vanished from the translation unit without a diagnostic. The JIT would then have been
// compiling a quietly different FlashInfer from the ahead-of-time build -- the failure
// mode this crate exists to eliminate.
//
// So the driver header is carried, under the rule that a host header whose names reach
// device code is carried rather than guarded. What device code names is: fixed-width
// integers, and `CUDA_VERSION`. The driver API proper -- `CUresult`, `CUtensorMap`,
// `cuLaunchKernel` -- is named nowhere in the closure; grep for `CU[a-z]` across the 28
// files returns only `CUDA_VERSION`. Declaring it would be dead text in every compile.
//
// `CUDA_VERSION` is derived rather than hardcoded, and the distinction matters: under a
// JIT the number that decides which types exist is the version of the NVRTC doing the
// compiling, not the version of the toolkit that built the binary. `__CUDACC_VER_MAJOR__`
// and `__CUDACC_VER_MINOR__` are predefined by NVRTC 13.0 (measured; `CUDA_VERSION`
// itself is not), and the real header's number is formed the same way -- 13.0 -> 13000.
//
// This file is PIE's, not FlashInfer's. It is not upstream and never was.

#ifndef PIE_VENDOR_CUDA_H_
#define PIE_VENDOR_CUDA_H_

#include <cstdint>

#define CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10)

#endif  // PIE_VENDOR_CUDA_H_
