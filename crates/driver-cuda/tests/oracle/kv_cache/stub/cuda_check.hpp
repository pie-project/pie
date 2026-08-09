#pragma once
// Stub for csrc/src/cuda_check.hpp. `kv_cache.cpp` uses CUDA_CHECK once, to
// synchronise after seeding the envelopes; there is no device here and nothing
// to synchronise with.
#include <cstddef>
using cudaError_t = int;
inline cudaError_t cudaStreamSynchronize(void*) { return 0; }
#define CUDA_CHECK(x) do { (void)(x); } while (0)
