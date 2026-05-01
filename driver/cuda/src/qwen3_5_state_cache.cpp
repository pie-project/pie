#include "qwen3_5_state_cache.hpp"

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

Qwen3_5StateCache Qwen3_5StateCache::allocate(
    const std::vector<bool>& layer_is_linear,
    int conv_dim,
    int conv_kernel,
    int v_heads,
    int head_k_dim,
    int head_v_dim)
{
    Qwen3_5StateCache c;
    c.layer_is_linear_ = layer_is_linear;
    c.conv_dim_    = conv_dim;
    c.conv_kernel_ = conv_kernel;
    c.v_heads_     = v_heads;
    c.head_k_dim_  = head_k_dim;
    c.head_v_dim_  = head_v_dim;

    const std::size_t conv_elems = static_cast<std::size_t>(conv_kernel) * conv_dim;
    const std::size_t rec_elems  = static_cast<std::size_t>(v_heads) *
                                    head_k_dim * head_v_dim;

    c.conv_states_.reserve(layer_is_linear.size());
    c.recurrent_states_.reserve(layer_is_linear.size());
    for (bool linear : layer_is_linear) {
        if (linear) {
            c.conv_states_.push_back(
                DeviceBuffer<std::uint16_t>::alloc(conv_elems));
            c.recurrent_states_.push_back(
                DeviceBuffer<float>::alloc(rec_elems));
        } else {
            // Empty placeholders for full-attention layers — keeps
            // indexing by `layer` direct.
            c.conv_states_.emplace_back();
            c.recurrent_states_.emplace_back();
        }
    }
    c.reset();
    return c;
}

void Qwen3_5StateCache::reset(cudaStream_t stream)
{
    for (std::size_t L = 0; L < layer_is_linear_.size(); ++L) {
        if (!layer_is_linear_[L]) continue;
        const std::size_t conv_bytes =
            static_cast<std::size_t>(conv_kernel_) * conv_dim_ *
            sizeof(std::uint16_t);
        const std::size_t rec_bytes =
            static_cast<std::size_t>(v_heads_) * head_k_dim_ * head_v_dim_ *
            sizeof(float);
        CUDA_CHECK(cudaMemsetAsync(conv_states_[L].data(), 0, conv_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(recurrent_states_[L].data(), 0, rec_bytes, stream));
    }
}

}  // namespace pie_cuda_driver
