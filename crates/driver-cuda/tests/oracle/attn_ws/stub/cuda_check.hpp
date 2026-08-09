#pragma once
// Stub csrc/src/cuda_check.hpp.
//
// It must still THROW on failure: the failure paths under test -- the
// `allocate` catch block, the lazy pin failing mid-rotation -- only exist
// because the real macro throws, and a stub that swallowed the error would
// make them unreachable instead of exercised.
#include <stdexcept>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            throw std::runtime_error("CUDA_CHECK failed");                   \
        }                                                                    \
    } while (0)
