#pragma once

// Per-request state caches for Qwen3.5's linear-attention layers.
// Each layer of type `linear_attention` carries two persistent
// per-request tensors:
//
//   * conv_state     : [K, conv_dim] bf16
//                      The trailing K-window of the in-projection
//                      output, oldest-first. Read at every decode
//                      step, updated in-place by the causal-conv1d
//                      update kernel.
//
//   * recurrent_state: [V_h, K_d, V_d] fp32
//                      Linear-attention running state. Held in fp32
//                      to avoid drift across hundreds of decode
//                      steps; promoted at the bf16 boundary.
//
// Single-request layout for the parity rig and initial bring-up; a
// future patch will extend this with per-request indices when the
// runtime starts batching multiple Qwen3.5 prompts.

#include <cstdint>
#include <vector>

#include "device_buffer.hpp"

namespace pie_cuda_driver {

class Qwen3_5StateCache {
public:
    // Allocate state for `num_linear_layers` linear-attention layers.
    // Other layer indices (full-attention) get null entries.
    //
    //     conv_dim = 2 * (num_k_heads * head_k_dim) + (num_v_heads * head_v_dim)
    //     K        = conv_kernel_size
    //     V_h      = num_v_heads, K_d = head_k_dim, V_d = head_v_dim
    //
    // After repeat_interleave to V_h heads on the q/k path, both q
    // and k carry V_h heads of size head_k_dim, so the recurrent
    // state is [V_h, head_k_dim, head_v_dim].
    static Qwen3_5StateCache allocate(
        const std::vector<bool>& layer_is_linear,
        int conv_dim,
        int conv_kernel,
        int v_heads,
        int head_k_dim,
        int head_v_dim);

    // Zero the state of every linear-attention layer. Called at the
    // start of each fresh prefill — no cross-prefill state continuity
    // in this driver's batching model.
    void reset(cudaStream_t stream = 0);

    // Per-layer accessors (return nullptr on full-attention layers).
    void* conv_state(int layer)      { return conv_states_[layer].data(); }
    float* recurrent_state(int layer) { return recurrent_states_[layer].data(); }

    int num_layers() const noexcept { return static_cast<int>(layer_is_linear_.size()); }
    bool is_linear(int layer) const noexcept { return layer_is_linear_[layer]; }

    int conv_dim()    const noexcept { return conv_dim_; }
    int conv_kernel() const noexcept { return conv_kernel_; }
    int v_heads()     const noexcept { return v_heads_; }
    int head_k_dim()  const noexcept { return head_k_dim_; }
    int head_v_dim()  const noexcept { return head_v_dim_; }

    Qwen3_5StateCache() = default;

private:

    std::vector<bool> layer_is_linear_;
    std::vector<DeviceBuffer<std::uint16_t>> conv_states_;       // bf16 stored as u16
    std::vector<DeviceBuffer<float>>         recurrent_states_;
    int conv_dim_ = 0;
    int conv_kernel_ = 0;
    int v_heads_ = 0;
    int head_k_dim_ = 0;
    int head_v_dim_ = 0;
};

}  // namespace pie_cuda_driver
