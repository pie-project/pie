#include "ops/attention_workspace.hpp"

#include <cstdlib>
#include <utility>

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
    try {
        CUDA_CHECK(cudaEventCreateWithFlags(
            &ws.plan_upload_done_, cudaEventDisableTiming));
    } catch (...) {
        cudaFreeHost(ws.page_locked_int_);
        ws.page_locked_int_ = nullptr;
        throw;
    }
    return ws;
}

AttentionWorkspace::AttentionWorkspace(AttentionWorkspace&& other) noexcept
    : float_buf_(std::move(other.float_buf_)),
      int_buf_(std::move(other.int_buf_)),
      page_locked_int_(other.page_locked_int_),
      plan_upload_done_(other.plan_upload_done_),
      plan_upload_pending_(other.plan_upload_pending_)
{
    other.page_locked_int_ = nullptr;
    other.plan_upload_done_ = nullptr;
    other.plan_upload_pending_ = false;
}

AttentionWorkspace& AttentionWorkspace::operator=(AttentionWorkspace&& other) noexcept {
    if (this != &other) {
        if (plan_upload_pending_) {
            cudaEventSynchronize(plan_upload_done_);
        }
        if (plan_upload_done_) {
            cudaEventDestroy(plan_upload_done_);
        }
        if (page_locked_int_) {
            cudaFreeHost(page_locked_int_);
        }
        float_buf_ = std::move(other.float_buf_);
        int_buf_ = std::move(other.int_buf_);
        page_locked_int_ = other.page_locked_int_;
        plan_upload_done_ = other.plan_upload_done_;
        plan_upload_pending_ = other.plan_upload_pending_;
        other.page_locked_int_ = nullptr;
        other.plan_upload_done_ = nullptr;
        other.plan_upload_pending_ = false;
    }
    return *this;
}

AttentionWorkspace::~AttentionWorkspace() {
    if (plan_upload_pending_) {
        cudaEventSynchronize(plan_upload_done_);
        plan_upload_pending_ = false;
    }
    if (plan_upload_done_) {
        cudaEventDestroy(plan_upload_done_);
        plan_upload_done_ = nullptr;
    }
    if (page_locked_int_) {
        cudaFreeHost(page_locked_int_);
        page_locked_int_ = nullptr;
    }
}

void AttentionWorkspace::begin_plan_update() {
    if (!plan_upload_pending_) return;
    CUDA_CHECK(cudaEventSynchronize(plan_upload_done_));
    plan_upload_pending_ = false;
}

void AttentionWorkspace::end_plan_update(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(plan_upload_done_, stream));
    plan_upload_pending_ = true;
}

bool flashinfer_decode_supports_gqa(int gqa) {
    return gqa == 1 || gqa == 2 || gqa == 3 || gqa == 4 || gqa == 8;
}

bool xqa_decode_enabled_by_env() {
    const char* v = std::getenv("PIE_CUDA_XQA_DECODE");
    if (v == nullptr || v[0] == '\0') return true;
    return v[0] != '0';
}

}  // namespace pie_cuda_driver
