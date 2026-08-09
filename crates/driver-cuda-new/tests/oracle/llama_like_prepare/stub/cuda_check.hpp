#pragma once
// Stub csrc/src/cuda_check.hpp for the prepare oracle.
//
// Includes the runtime stub because `kv_cache.cpp` reaches
// `cudaStreamSynchronize` through this header alone. Still THROWS on
// failure so a recorder changed to fail is visible rather than swallowed.
#include <stdexcept>

#include "cuda_runtime.h"

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            throw std::runtime_error("CUDA_CHECK failed");                   \
        }                                                                    \
    } while (0)
