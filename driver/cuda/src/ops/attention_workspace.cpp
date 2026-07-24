#include "ops/attention_workspace.hpp"

#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

void AttentionWorkspace::ensure_plan_slot(PlanStaging& slot) {
    if (slot.host == nullptr && staging_bytes_ > 0) {
        CUDA_CHECK(cudaMallocHost(&slot.host, staging_bytes_));
    }
    if (slot.upload_done == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(
            &slot.upload_done, cudaEventDisableTiming));
    }
}

AttentionWorkspace AttentionWorkspace::allocate(
    std::size_t float_workspace_bytes,
    std::size_t int_workspace_bytes)
{
    AttentionWorkspace ws;
    ws.float_buf_ = DeviceTensor::allocate(
        DType::UINT8, {static_cast<std::int64_t>(float_workspace_bytes)});
    ws.int_buf_ = DeviceTensor::allocate(
        DType::UINT8, {static_cast<std::int64_t>(int_workspace_bytes)});
    ws.staging_bytes_ = int_workspace_bytes;
    try {
        ws.ensure_plan_slot(ws.plan_staging_[0]);
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
      staging_bytes_(other.staging_bytes_),
      plan_staging_(other.plan_staging_),
      active_plan_slot_(other.active_plan_slot_),
      next_plan_slot_(other.next_plan_slot_)
{
    other.plan_staging_ = {};
    other.staging_bytes_ = 0;
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
        staging_bytes_ = other.staging_bytes_;
        plan_staging_ = other.plan_staging_;
        active_plan_slot_ = other.active_plan_slot_;
        next_plan_slot_ = other.next_plan_slot_;
        other.plan_staging_ = {};
        other.staging_bytes_ = 0;
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
    ensure_plan_slot(staging);
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
