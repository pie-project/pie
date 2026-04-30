#include "attention_workspace.hpp"

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

AttentionWorkspace AttentionWorkspace::allocate(
    std::size_t float_workspace_bytes,
    std::size_t int_workspace_bytes)
{
    AttentionWorkspace ws;
    ws.float_buf_ = DeviceTensor::allocate(
        DType::UINT8, {static_cast<std::int64_t>(float_workspace_bytes)});
    ws.int_buf_ = DeviceTensor::allocate(
        DType::UINT8, {static_cast<std::int64_t>(int_workspace_bytes)});
    CUDA_CHECK(cudaMallocHost(&ws.page_locked_int_, int_workspace_bytes));
    return ws;
}

AttentionWorkspace::~AttentionWorkspace() {
    if (page_locked_int_) {
        cudaFreeHost(page_locked_int_);
        page_locked_int_ = nullptr;
    }
}

}  // namespace pie_cuda_driver
