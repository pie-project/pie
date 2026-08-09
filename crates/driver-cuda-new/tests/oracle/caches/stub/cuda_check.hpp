#pragma once
// Stub for csrc/src/cuda_check.hpp.
//
// The real macro wraps a call, compares against cudaSuccess and throws with
// the error string. Here every call succeeds except the memset the harness
// deliberately fails, and that one is checked by hand in the shipping code
// rather than through this macro -- `dsv4_compress_cache.cpp` writes
// `if (cudaMemset(...) != cudaSuccess)`, precisely because it does not want
// the throw.
#include <cuda_runtime.h>
#define CUDA_CHECK(x) do { (void)(x); } while (0)
