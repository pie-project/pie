#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::model {

enum class StageHookPoint : std::uint8_t {
    OnAttnProj = 1,
    OnAttn = 2,
};

struct StageHooks {
    void* context = nullptr;
    void (*execute)(
        void* context,
        StageHookPoint point,
        const void* query_data,
        std::uint32_t query_rows,
        std::uint32_t query_columns,
        std::uint32_t layer,
        cudaStream_t stream,
        bool query_is_f32) = nullptr;
};

inline thread_local const StageHooks* active_stage_hooks = nullptr;

class ScopedStageHooks {
  public:
    explicit ScopedStageHooks(const StageHooks* hooks)
        : previous_(active_stage_hooks) {
        active_stage_hooks = hooks;
    }
    ~ScopedStageHooks() { active_stage_hooks = previous_; }

    ScopedStageHooks(const ScopedStageHooks&) = delete;
    ScopedStageHooks& operator=(const ScopedStageHooks&) = delete;

  private:
    const StageHooks* previous_ = nullptr;
};

inline void invoke_stage_hook(
    StageHookPoint point,
    const void* query_data,
    std::uint32_t query_rows,
    std::uint32_t query_columns,
    std::uint32_t layer,
    cudaStream_t stream,
    bool query_is_f32 = false) {
    if (active_stage_hooks == nullptr ||
        active_stage_hooks->execute == nullptr) {
        return;
    }
    active_stage_hooks->execute(
        active_stage_hooks->context, point, query_data, query_rows,
        query_columns, layer, stream, query_is_f32);
}

}  // namespace pie_cuda_driver::model
