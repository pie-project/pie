#pragma once
#include <stdexcept>
#include <string>

// The planner's three CUDA calls cannot fail in this harness -- the stubs
// above always succeed -- but the macro must exist for the source to compile
// unmodified, and it must still throw so a stub that IS changed to fail is
// visible rather than silently ignored.
#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        const cudaError_t _e = (expr);                                       \
        if (_e != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("cuda: ") +                 \
                                     cudaGetErrorString(_e));                \
        }                                                                    \
    } while (0)
