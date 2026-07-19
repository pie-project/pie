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

class StreamCaptureGuard {
  public:
    explicit StreamCaptureGuard(
        cudaStream_t stream,
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed)
        : stream_(stream), active_(true) {
        cuda_check_impl(
            cudaStreamBeginCapture(stream_, mode),
            "cudaStreamBeginCapture", __FILE__, __LINE__);
    }

    ~StreamCaptureGuard() noexcept {
        if (!active_) return;
        cudaGraph_t graph = nullptr;
        const cudaError_t status = cudaStreamEndCapture(stream_, &graph);
        if (status == cudaSuccess && graph != nullptr) {
            cudaGraphDestroy(graph);
        } else {
            cudaGetLastError();
        }
    }

    StreamCaptureGuard(const StreamCaptureGuard&) = delete;
    StreamCaptureGuard& operator=(const StreamCaptureGuard&) = delete;

    cudaGraph_t end() {
        if (!active_) {
            throw std::logic_error("CUDA stream capture already ended");
        }
        cudaGraph_t graph = nullptr;
        const cudaError_t status = cudaStreamEndCapture(stream_, &graph);
        active_ = false;
        if (status != cudaSuccess && graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        cuda_check_impl(status, "cudaStreamEndCapture", __FILE__, __LINE__);
        return graph;
    }

  private:
    cudaStream_t stream_ = nullptr;
    bool active_ = false;
};

}  // namespace pie_cuda_driver

#define CUDA_CHECK(expr) ::pie_cuda_driver::cuda_check_impl((expr), #expr, __FILE__, __LINE__)
