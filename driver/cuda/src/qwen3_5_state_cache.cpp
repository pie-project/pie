#include "qwen3_5_state_cache.hpp"

#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

Qwen3_5StateCache Qwen3_5StateCache::allocate_impl(
    const std::vector<bool>& layer_is_linear,
    int conv_dim,
    int conv_kernel,
    int v_heads,
    int head_k_dim,
    int head_v_dim,
    int max_slots,
    bool recurrent_bf16)
{
    if (max_slots < 1) max_slots = 1;
    Qwen3_5StateCache c;
    c.layer_is_linear_ = layer_is_linear;
    c.recurrent_bf16_ = recurrent_bf16;
    c.max_slots_   = max_slots;
    c.conv_dim_    = conv_dim;
    c.conv_kernel_ = conv_kernel;
    c.v_heads_     = v_heads;
    c.head_k_dim_  = head_k_dim;
    c.head_v_dim_  = head_v_dim;

    const std::size_t conv_slot_elems =
        static_cast<std::size_t>(conv_kernel) * conv_dim;
    const std::size_t rec_slot_elems  =
        static_cast<std::size_t>(v_heads) * head_k_dim * head_v_dim;
    const std::size_t conv_total = conv_slot_elems * max_slots;
    const std::size_t rec_total  = rec_slot_elems  * max_slots;

    c.conv_states_.reserve(layer_is_linear.size());
    c.recurrent_states_.reserve(layer_is_linear.size());
    c.recurrent_bf16_states_.reserve(layer_is_linear.size());
    for (bool linear : layer_is_linear) {
        if (linear) {
            c.conv_states_.push_back(
                DeviceBuffer<std::uint16_t>::alloc(conv_total));
            if (recurrent_bf16) {
                c.recurrent_bf16_states_.push_back(
                    DeviceBuffer<std::uint16_t>::alloc(rec_total));
                c.recurrent_states_.emplace_back();
            } else {
                c.recurrent_states_.push_back(
                    DeviceBuffer<float>::alloc(rec_total));
                c.recurrent_bf16_states_.emplace_back();
            }
        } else {
            // Empty placeholders for full-attention layers — keeps
            // indexing by `layer` direct.
            c.conv_states_.emplace_back();
            c.recurrent_states_.emplace_back();
            c.recurrent_bf16_states_.emplace_back();
        }
    }
    c.reset();
    return c;
}

Qwen3_5StateCache Qwen3_5StateCache::allocate(
    const std::vector<bool>& layer_is_linear,
    int conv_dim,
    int conv_kernel,
    int v_heads,
    int head_k_dim,
    int head_v_dim,
    int max_slots)
{
    return Qwen3_5StateCache::allocate_impl(
        layer_is_linear, conv_dim, conv_kernel,
        v_heads, head_k_dim, head_v_dim, max_slots,
        /*recurrent_bf16=*/false);
}

Qwen3_5StateCache Qwen3_5StateCache::allocate_bf16_recurrent(
    const std::vector<bool>& layer_is_linear,
    int conv_dim,
    int conv_kernel,
    int v_heads,
    int head_k_dim,
    int head_v_dim,
    int max_slots)
{
    return Qwen3_5StateCache::allocate_impl(
        layer_is_linear, conv_dim, conv_kernel,
        v_heads, head_k_dim, head_v_dim, max_slots,
        /*recurrent_bf16=*/true);
}

void Qwen3_5StateCache::reset(cudaStream_t stream)
{
    const std::size_t conv_bytes = conv_slot_stride_bytes() * max_slots_;
    const std::size_t rec_bytes  = recurrent_slot_stride_bytes() * max_slots_;
    for (std::size_t L = 0; L < layer_is_linear_.size(); ++L) {
        if (!layer_is_linear_[L]) continue;
        CUDA_CHECK(cudaMemsetAsync(conv_states_[L].data(), 0, conv_bytes, stream));
        void* rec = recurrent_bf16_
            ? static_cast<void*>(recurrent_bf16_states_[L].data())
            : static_cast<void*>(recurrent_states_[L].data());
        CUDA_CHECK(cudaMemsetAsync(rec, 0, rec_bytes, stream));
    }
}

void Qwen3_5StateCache::reset_slot(int slot, cudaStream_t stream)
{
    if (slot < 0 || slot >= max_slots_) {
        throw std::out_of_range("Qwen3_5StateCache::reset_slot: slot out of range");
    }
    const std::size_t conv_bytes  = conv_slot_stride_bytes();
    const std::size_t rec_bytes   = recurrent_slot_stride_bytes();
    for (std::size_t L = 0; L < layer_is_linear_.size(); ++L) {
        if (!layer_is_linear_[L]) continue;
        auto* conv_base = reinterpret_cast<std::uint8_t*>(conv_states_[L].data());
        auto* rec_base  = recurrent_bf16_
            ? reinterpret_cast<std::uint8_t*>(recurrent_bf16_states_[L].data())
            : reinterpret_cast<std::uint8_t*>(recurrent_states_[L].data());
        CUDA_CHECK(cudaMemsetAsync(
            conv_base + slot * conv_bytes, 0, conv_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(
            rec_base + static_cast<std::size_t>(slot) * rec_bytes,
            0, rec_bytes, stream));
    }
}

void Qwen3_5StateCache::copy_slot_d2d(int src_slot, int dst_slot, cudaStream_t stream)
{
    if (src_slot < 0 || src_slot >= max_slots_ || dst_slot < 0 || dst_slot >= max_slots_) {
        throw std::out_of_range("Qwen3_5StateCache::copy_slot_d2d: slot out of range");
    }
    if (src_slot == dst_slot) {
        return;
    }
    const std::size_t conv_bytes = conv_slot_stride_bytes();
    const std::size_t rec_bytes = recurrent_slot_stride_bytes();
    for (std::size_t L = 0; L < layer_is_linear_.size(); ++L) {
        if (!layer_is_linear_[L]) continue;
        auto* conv_base = reinterpret_cast<std::uint8_t*>(conv_states_[L].data());
        auto* rec_base = recurrent_bf16_
            ? reinterpret_cast<std::uint8_t*>(recurrent_bf16_states_[L].data())
            : reinterpret_cast<std::uint8_t*>(recurrent_states_[L].data());
        CUDA_CHECK(cudaMemcpyAsync(
            conv_base + static_cast<std::size_t>(dst_slot) * conv_bytes,
            conv_base + static_cast<std::size_t>(src_slot) * conv_bytes,
            conv_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            rec_base + static_cast<std::size_t>(dst_slot) * rec_bytes,
            rec_base + static_cast<std::size_t>(src_slot) * rec_bytes,
            rec_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
    }
}

void* Qwen3_5StateCache::conv_state(int layer, int slot)
{
    if (slot < 0 || slot >= max_slots_) {
        throw std::out_of_range("Qwen3_5StateCache::conv_state: slot out of range");
    }
    auto* base = reinterpret_cast<std::uint8_t*>(conv_states_[layer].data());
    if (base == nullptr) return nullptr;  // full-attention layer
    return base + static_cast<std::size_t>(slot) * conv_slot_stride_bytes();
}

float* Qwen3_5StateCache::recurrent_state(int layer, int slot)
{
    if (recurrent_bf16_) {
        throw std::runtime_error(
            "Qwen3_5StateCache::recurrent_state: recurrent storage is bf16");
    }
    if (slot < 0 || slot >= max_slots_) {
        throw std::out_of_range("Qwen3_5StateCache::recurrent_state: slot out of range");
    }
    float* base = recurrent_states_[layer].data();
    if (base == nullptr) return nullptr;  // full-attention layer
    return base + static_cast<std::size_t>(slot) *
        recurrent_slot_stride_floats();
}

void* Qwen3_5StateCache::recurrent_state_bf16(int layer, int slot)
{
    if (!recurrent_bf16_) {
        throw std::runtime_error(
            "Qwen3_5StateCache::recurrent_state_bf16: recurrent storage is fp32");
    }
    if (slot < 0 || slot >= max_slots_) {
        throw std::out_of_range(
            "Qwen3_5StateCache::recurrent_state_bf16: slot out of range");
    }
    auto* base = reinterpret_cast<std::uint8_t*>(
        recurrent_bf16_states_[layer].data());
    if (base == nullptr) return nullptr;
    return base + static_cast<std::size_t>(slot) *
        recurrent_slot_stride_bytes();
}

}  // namespace pie_cuda_driver
