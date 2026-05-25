#include "qwen3_5_state_cache.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

bool qwen35_rs_state_bf16_enabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_RS_STATE_DTYPE");
        if (v == nullptr || v[0] == '\0') return true;
        if (std::strcmp(v, "fp32") == 0 || std::strcmp(v, "FP32") == 0 ||
            std::strcmp(v, "float32") == 0 || v[0] == '0') {
            return false;
        }
        return std::strcmp(v, "bf16") == 0 || std::strcmp(v, "BF16") == 0 ||
               std::strcmp(v, "bfloat16") == 0 || v[0] == '1';
    }();
    return enabled;
}

}  // namespace

Qwen3_5StateCache Qwen3_5StateCache::allocate(
    const std::vector<bool>& layer_is_linear,
    int conv_dim,
    int conv_kernel,
    int v_heads,
    int head_k_dim,
    int head_v_dim,
    int hidden_size,
    int max_slots)
{
    if (max_slots < 1) max_slots = 1;
    Qwen3_5StateCache c;
    c.layer_is_linear_ = layer_is_linear;
    c.max_slots_   = max_slots;
    c.conv_dim_    = conv_dim;
    c.conv_kernel_ = conv_kernel;
    c.v_heads_     = v_heads;
    c.head_k_dim_  = head_k_dim;
    c.head_v_dim_  = head_v_dim;
    c.hidden_size_  = std::max(0, hidden_size);
    c.recurrent_state_bf16_ = qwen35_rs_state_bf16_enabled();

    const std::size_t conv_slot_elems =
        static_cast<std::size_t>(conv_kernel) * conv_dim;
    const std::size_t rec_slot_elems  =
        static_cast<std::size_t>(v_heads) * head_k_dim * head_v_dim;
    const std::size_t conv_total = conv_slot_elems * max_slots;
    const std::size_t rec_total  = rec_slot_elems  * max_slots;

    c.linear_layer_index_.assign(layer_is_linear.size(), -1);
    for (std::size_t layer = 0; layer < layer_is_linear.size(); ++layer) {
        if (!layer_is_linear[layer]) continue;
        c.linear_layer_index_[layer] = c.num_linear_layers_++;
    }
    if (c.num_linear_layers_ > 0) {
        c.conv_states_ = DeviceBuffer<std::uint16_t>::alloc(
            conv_total * static_cast<std::size_t>(c.num_linear_layers_));
        if (c.recurrent_state_bf16_) {
            c.recurrent_states_bf16_ = DeviceBuffer<std::uint16_t>::alloc(
                rec_total * static_cast<std::size_t>(c.num_linear_layers_));
        } else {
            c.recurrent_states_ = DeviceBuffer<float>::alloc(
                rec_total * static_cast<std::size_t>(c.num_linear_layers_));
        }
    }
    if (c.hidden_size_ > 0) {
        c.mtp_pending_hidden_ = DeviceBuffer<std::uint16_t>::alloc(
            static_cast<std::size_t>(c.hidden_size_) * max_slots);
    }
    c.reset();
    return c;
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
    Qwen3_5StateCache c = Qwen3_5StateCache::allocate(
        layer_is_linear, conv_dim, conv_kernel,
        v_heads, head_k_dim, head_v_dim,
        /*hidden_size=*/0, max_slots);
    // The env-gated default in `allocate` may have selected fp32 storage; for
    // Nemotron-H's Mamba2 cache, force bf16 by re-allocating the recurrent
    // slab if needed.
    if (!c.recurrent_state_bf16_ && c.num_linear_layers_ > 0) {
        const std::size_t rec_slot_elems =
            static_cast<std::size_t>(v_heads) * head_k_dim * head_v_dim;
        const std::size_t rec_total =
            rec_slot_elems * static_cast<std::size_t>(std::max(1, max_slots));
        c.recurrent_states_ = DeviceBuffer<float>{};
        c.recurrent_states_bf16_ = DeviceBuffer<std::uint16_t>::alloc(
            rec_total * static_cast<std::size_t>(c.num_linear_layers_));
        c.recurrent_state_bf16_ = true;
        c.reset();
    }
    return c;
}

void Qwen3_5StateCache::reset(cudaStream_t stream)
{
    if (num_linear_layers_ > 0) {
        const std::size_t conv_bytes =
            conv_slot_stride_bytes() * static_cast<std::size_t>(max_slots_) *
            static_cast<std::size_t>(num_linear_layers_);
        const std::size_t rec_bytes =
            recurrent_slot_stride_bytes() * static_cast<std::size_t>(max_slots_) *
            static_cast<std::size_t>(num_linear_layers_);
        CUDA_CHECK(cudaMemsetAsync(conv_states_.data(), 0, conv_bytes, stream));
        void* rec_base = recurrent_state_bf16_
            ? static_cast<void*>(recurrent_states_bf16_.data())
            : static_cast<void*>(recurrent_states_.data());
        CUDA_CHECK(cudaMemsetAsync(rec_base, 0, rec_bytes, stream));
    }
    if (mtp_pending_hidden_.data() != nullptr && hidden_size_ > 0) {
        CUDA_CHECK(cudaMemsetAsync(
            mtp_pending_hidden_.data(), 0,
            static_cast<std::size_t>(hidden_size_) * max_slots_ *
                sizeof(std::uint16_t),
            stream));
    }
}

void Qwen3_5StateCache::reset_slot(int slot, cudaStream_t stream)
{
    if (slot < 0 || slot >= max_slots_) {
        throw std::out_of_range("Qwen3_5StateCache::reset_slot: slot out of range");
    }
    const std::size_t conv_bytes  = conv_slot_stride_bytes();
    const std::size_t rec_bytes   = recurrent_slot_stride_bytes();
    if (num_linear_layers_ > 0) {
        const std::size_t conv_pitch =
            conv_bytes * static_cast<std::size_t>(max_slots_);
        const std::size_t rec_pitch =
            rec_bytes * static_cast<std::size_t>(max_slots_);
        auto* conv_base = reinterpret_cast<std::uint8_t*>(conv_states_.data());
        auto* rec_base = recurrent_state_bf16_
            ? reinterpret_cast<std::uint8_t*>(recurrent_states_bf16_.data())
            : reinterpret_cast<std::uint8_t*>(recurrent_states_.data());
        CUDA_CHECK(cudaMemset2DAsync(
            conv_base + static_cast<std::size_t>(slot) * conv_bytes,
            conv_pitch, 0, conv_bytes,
            static_cast<std::size_t>(num_linear_layers_), stream));
        CUDA_CHECK(cudaMemset2DAsync(
            rec_base + static_cast<std::size_t>(slot) * rec_bytes,
            rec_pitch, 0, rec_bytes,
            static_cast<std::size_t>(num_linear_layers_), stream));
    }
    if (mtp_pending_hidden_.data() != nullptr && hidden_size_ > 0) {
        CUDA_CHECK(cudaMemsetAsync(
            mtp_pending_hidden_.data() +
                static_cast<std::size_t>(slot) * hidden_size_,
            0,
            static_cast<std::size_t>(hidden_size_) * sizeof(std::uint16_t),
            stream));
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
    if (num_linear_layers_ > 0) {
        const std::size_t conv_pitch =
            conv_bytes * static_cast<std::size_t>(max_slots_);
        const std::size_t rec_pitch =
            rec_bytes * static_cast<std::size_t>(max_slots_);
        auto* conv_base = reinterpret_cast<std::uint8_t*>(conv_states_.data());
        auto* rec_base = recurrent_state_bf16_
            ? reinterpret_cast<std::uint8_t*>(recurrent_states_bf16_.data())
            : reinterpret_cast<std::uint8_t*>(recurrent_states_.data());
        CUDA_CHECK(cudaMemcpy2DAsync(
            conv_base + static_cast<std::size_t>(dst_slot) * conv_bytes,
            conv_pitch,
            conv_base + static_cast<std::size_t>(src_slot) * conv_bytes,
            conv_pitch,
            conv_bytes, static_cast<std::size_t>(num_linear_layers_),
            cudaMemcpyDeviceToDevice,
            stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            rec_base + static_cast<std::size_t>(dst_slot) * rec_bytes,
            rec_pitch,
            rec_base + static_cast<std::size_t>(src_slot) * rec_bytes,
            rec_pitch,
            rec_bytes, static_cast<std::size_t>(num_linear_layers_),
            cudaMemcpyDeviceToDevice,
            stream));
    }
    if (mtp_pending_hidden_.data() != nullptr && hidden_size_ > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            mtp_pending_hidden_.data() +
                static_cast<std::size_t>(dst_slot) * hidden_size_,
            mtp_pending_hidden_.data() +
                static_cast<std::size_t>(src_slot) * hidden_size_,
            static_cast<std::size_t>(hidden_size_) * sizeof(std::uint16_t),
            cudaMemcpyDeviceToDevice,
            stream));
    }
}

void Qwen3_5StateCache::copy_linear_state_slot_d2d(
    int src_slot, int dst_slot, cudaStream_t stream)
{
    if (src_slot < 0 || src_slot >= max_slots_ ||
        dst_slot < 0 || dst_slot >= max_slots_) {
        throw std::out_of_range(
            "Qwen3_5StateCache::copy_linear_state_slot_d2d: slot out of range");
    }
    if (src_slot == dst_slot) {
        return;
    }
    const std::size_t conv_bytes = conv_slot_stride_bytes();
    const std::size_t rec_bytes = recurrent_slot_stride_bytes();
    if (num_linear_layers_ > 0) {
        const std::size_t conv_pitch =
            conv_bytes * static_cast<std::size_t>(max_slots_);
        const std::size_t rec_pitch =
            rec_bytes * static_cast<std::size_t>(max_slots_);
        auto* conv_base = reinterpret_cast<std::uint8_t*>(conv_states_.data());
        auto* rec_base = recurrent_state_bf16_
            ? reinterpret_cast<std::uint8_t*>(recurrent_states_bf16_.data())
            : reinterpret_cast<std::uint8_t*>(recurrent_states_.data());
        CUDA_CHECK(cudaMemcpy2DAsync(
            conv_base + static_cast<std::size_t>(dst_slot) * conv_bytes,
            conv_pitch,
            conv_base + static_cast<std::size_t>(src_slot) * conv_bytes,
            conv_pitch,
            conv_bytes, static_cast<std::size_t>(num_linear_layers_),
            cudaMemcpyDeviceToDevice,
            stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            rec_base + static_cast<std::size_t>(dst_slot) * rec_bytes,
            rec_pitch,
            rec_base + static_cast<std::size_t>(src_slot) * rec_bytes,
            rec_pitch,
            rec_bytes, static_cast<std::size_t>(num_linear_layers_),
            cudaMemcpyDeviceToDevice,
            stream));
    }
}

void* Qwen3_5StateCache::conv_state(int layer, int slot)
{
    if (slot < 0 || slot >= max_slots_) {
        throw std::out_of_range("Qwen3_5StateCache::conv_state: slot out of range");
    }
    if (layer < 0 || layer >= static_cast<int>(linear_layer_index_.size())) {
        throw std::out_of_range("Qwen3_5StateCache::conv_state: layer out of range");
    }
    const int linear_idx = linear_layer_index_[layer];
    if (linear_idx < 0) return nullptr;  // full-attention layer
    auto* base = reinterpret_cast<std::uint8_t*>(conv_states_.data());
    const std::size_t layer_stride =
        static_cast<std::size_t>(max_slots_) * conv_slot_stride_bytes();
    return base + static_cast<std::size_t>(linear_idx) * layer_stride +
           static_cast<std::size_t>(slot) * conv_slot_stride_bytes();
}

void* Qwen3_5StateCache::recurrent_state_raw(int layer, int slot)
{
    if (slot < 0 || slot >= max_slots_) {
        throw std::out_of_range("Qwen3_5StateCache::recurrent_state: slot out of range");
    }
    if (layer < 0 || layer >= static_cast<int>(linear_layer_index_.size())) {
        throw std::out_of_range(
            "Qwen3_5StateCache::recurrent_state: layer out of range");
    }
    const int linear_idx = linear_layer_index_[layer];
    if (linear_idx < 0) return nullptr;  // full-attention layer
    const std::size_t layer_stride =
        static_cast<std::size_t>(max_slots_) * recurrent_slot_stride_bytes();
    auto* base = recurrent_state_bf16_
        ? reinterpret_cast<std::uint8_t*>(recurrent_states_bf16_.data())
        : reinterpret_cast<std::uint8_t*>(recurrent_states_.data());
    return base + static_cast<std::size_t>(linear_idx) * layer_stride +
           static_cast<std::size_t>(slot) * recurrent_slot_stride_bytes();
}

float* Qwen3_5StateCache::recurrent_state(int layer, int slot)
{
    if (recurrent_state_bf16_) {
        throw std::runtime_error(
            "Qwen3_5StateCache::recurrent_state: recurrent state is bf16");
    }
    return static_cast<float*>(recurrent_state_raw(layer, slot));
}

void* Qwen3_5StateCache::mtp_pending_hidden(int slot)
{
    if (slot < 0 || slot >= max_slots_) {
        throw std::out_of_range("Qwen3_5StateCache::mtp_pending_hidden: slot out of range");
    }
    if (mtp_pending_hidden_.data() == nullptr || hidden_size_ <= 0) {
        return nullptr;
    }
    return mtp_pending_hidden_.data() +
        static_cast<std::size_t>(slot) * hidden_size_;
}

}  // namespace pie_cuda_driver
