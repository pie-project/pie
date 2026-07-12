#pragma once

#include <cstdint>
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct NemotronHLayerWeights {
    enum class Kind { Mamba, Attention, MoE };
    Kind kind = Kind::Mamba;

    const DeviceTensor* norm = nullptr;

    // Mamba2 mixer.
    const DeviceTensor* mamba_in_proj = nullptr;
    const DeviceTensor* mamba_conv_w = nullptr;
    const DeviceTensor* mamba_conv_b = nullptr;
    const DeviceTensor* mamba_A_log = nullptr;
    const DeviceTensor* mamba_D = nullptr;
    const DeviceTensor* mamba_dt_bias = nullptr;
    const DeviceTensor* mamba_norm_w = nullptr;
    const DeviceTensor* mamba_out_proj = nullptr;
    DeviceTensor mamba_in_proj_tp;
    DeviceTensor mamba_conv_w_tp;
    DeviceTensor mamba_conv_b_tp;
    DeviceTensor mamba_A_log_tp;
    DeviceTensor mamba_D_tp;
    DeviceTensor mamba_dt_bias_tp;
    DeviceTensor mamba_norm_w_tp;
    DeviceTensor mamba_out_proj_tp;
    DeviceBuffer<float> mamba_A;
    DeviceBuffer<float> mamba_D_f32;
    DeviceBuffer<float> mamba_dt_bias_f32;
    bool mamba_tp_sharded = false;

    // Attention mixer.
    const DeviceTensor* q_proj = nullptr;
    const DeviceTensor* k_proj = nullptr;
    const DeviceTensor* v_proj = nullptr;
    const DeviceTensor* o_proj = nullptr;
    int kv_layer = -1;

    // Sparse MoE mixer. Published Nemotron-H stores experts as a
    // ModuleList, so each expert has its own up/down tensor.
    const DeviceTensor* router = nullptr;
    const DeviceTensor* router_correction_bias = nullptr;  // fp32 [E]
    // Packed expert backing tensors emitted by the Rust loader. The legacy
    // per-expert pointers below are views into these buffers when present.
    const DeviceTensor* expert_up_packed = nullptr;    // [E * I_local, H]
    const DeviceTensor* expert_down_packed = nullptr;  // [E * H, I_local]
    std::vector<const DeviceTensor*> expert_up;
    std::vector<const DeviceTensor*> expert_down;
    const DeviceTensor* shared_up = nullptr;
    const DeviceTensor* shared_down = nullptr;

    DeviceBuffer<const std::uint16_t*> expert_up_ptrs;
    DeviceBuffer<const std::uint16_t*> expert_down_ptrs;
};

struct NemotronHWeights {
    const DeviceTensor* embed = nullptr;
    const DeviceTensor* final_norm = nullptr;
    const DeviceTensor* lm_head = nullptr;
    std::vector<NemotronHLayerWeights> layers;
};

bool nemotron_h_tp_mamba_sharding_enabled(int tp_size);
NemotronHWeights bind_nemotron_h(const LoadedModel& engine);

}  // namespace pie_cuda_driver::model
