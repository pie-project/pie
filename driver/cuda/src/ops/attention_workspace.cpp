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
    try {
        for (auto& staging : ws.plan_staging_) {
            CUDA_CHECK(cudaMallocHost(
                &staging.host, int_workspace_bytes));
            CUDA_CHECK(cudaEventCreateWithFlags(
                &staging.upload_done, cudaEventDisableTiming));
        }
    } catch (...) {
        for (auto& staging : ws.plan_staging_) {
            if (staging.upload_done != nullptr) {
                cudaEventDestroy(staging.upload_done);
                staging.upload_done = nullptr;
            }
            if (staging.host != nullptr) {
                cudaFreeHost(staging.host);
                staging.host = nullptr;
            }
        }
        throw;
    }
    return ws;
}

AttentionWorkspace::AttentionWorkspace(AttentionWorkspace&& other) noexcept
    : float_buf_(std::move(other.float_buf_)),
      int_buf_(std::move(other.int_buf_)),
      plan_staging_(other.plan_staging_),
      active_plan_slot_(other.active_plan_slot_),
      next_plan_slot_(other.next_plan_slot_)
{
    other.plan_staging_ = {};
    other.active_plan_slot_ = 0;
    other.next_plan_slot_ = 0;
}

AttentionWorkspace& AttentionWorkspace::operator=(AttentionWorkspace&& other) noexcept {
    if (this != &other) {
        for (auto& staging : plan_staging_) {
            if (staging.upload_pending) {
                cudaEventSynchronize(staging.upload_done);
            }
            if (staging.upload_done != nullptr) {
                cudaEventDestroy(staging.upload_done);
            }
            if (staging.host != nullptr) {
                cudaFreeHost(staging.host);
            }
        }
        float_buf_ = std::move(other.float_buf_);
        int_buf_ = std::move(other.int_buf_);
        plan_staging_ = other.plan_staging_;
        active_plan_slot_ = other.active_plan_slot_;
        next_plan_slot_ = other.next_plan_slot_;
        other.plan_staging_ = {};
        other.active_plan_slot_ = 0;
        other.next_plan_slot_ = 0;
    }
    return *this;
}

AttentionWorkspace::~AttentionWorkspace() {
    for (auto& staging : plan_staging_) {
        if (staging.upload_pending) {
            cudaEventSynchronize(staging.upload_done);
            staging.upload_pending = false;
        }
        if (staging.upload_done != nullptr) {
            cudaEventDestroy(staging.upload_done);
            staging.upload_done = nullptr;
        }
        if (staging.host != nullptr) {
            cudaFreeHost(staging.host);
            staging.host = nullptr;
        }
    }
}

void AttentionWorkspace::begin_plan_update() {
    active_plan_slot_ = next_plan_slot_;
    next_plan_slot_ =
        (next_plan_slot_ + 1) % kPlanStagingSlots;
    auto& staging = plan_staging_[active_plan_slot_];
    if (!staging.upload_pending) return;
    CUDA_CHECK(cudaEventSynchronize(staging.upload_done));
    staging.upload_pending = false;
}

void AttentionWorkspace::end_plan_update(cudaStream_t stream) {
    auto& staging = plan_staging_[active_plan_slot_];
    CUDA_CHECK(cudaEventRecord(staging.upload_done, stream));
    staging.upload_pending = true;
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
