#pragma once

// Tiny error-check macros for CUDA runtime calls. Turn cudaError_t into
// std::runtime_error so the direct FFI entry point can surface failures
// loop or shutdown path.

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pie_cuda_driver {

inline void cuda_check_impl(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(err)
            << " (" << err << ") at " << file << ":" << line
            << " — " << expr;
        throw std::runtime_error(oss.str());
    }
}

}  // namespace pie_cuda_driver

#define CUDA_CHECK(expr) ::pie_cuda_driver::cuda_check_impl((expr), #expr, __FILE__, __LINE__)
