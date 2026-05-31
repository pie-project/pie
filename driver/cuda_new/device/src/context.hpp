// PieDevCtx — per-device persistent state for the thin device library.
//
// Holds the CUDA stream and cuBLAS handle every kernel sequence runs on.
// The fat state (weights, KV cache, workspaces, graph cache) lives in
// its own handles; this is only the device + stream + handle triple that
// everything else threads through.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>

namespace pie_cuda_device {

// Thread-local last-error string backing `pie_cuda_last_error`.
void set_last_error(std::string msg);
const char* last_error();

}  // namespace pie_cuda_device

// Opaque to the ABI; concrete here so abi.cpp and the (future) ported
// forward/cache TUs can use it directly.
struct PieDevCtx {
    int device_ordinal = -1;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
};
