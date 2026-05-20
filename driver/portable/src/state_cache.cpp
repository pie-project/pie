#include "state_cache.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_portable_driver {

StateCache::StateCache(ggml_backend_t backend,
                       std::int32_t n_slots,
                       const std::vector<std::int32_t>& linear_layer_indices,
                       std::int32_t n_heads,
                       std::int32_t head_k_dim,
                       std::int32_t head_v_dim,
                       std::int32_t conv_dim,
                       std::int32_t conv_kernel)
    : backend_(backend),
      n_slots_(n_slots),
      n_heads_(n_heads),
      head_k_dim_(head_k_dim),
      head_v_dim_(head_v_dim),
      conv_dim_(conv_dim),
      conv_kernel_(conv_kernel) {
    if (n_slots <= 0 || n_heads <= 0 || head_k_dim <= 0 ||
        head_v_dim <= 0 || conv_dim <= 0 || conv_kernel <= 1) {
        throw std::runtime_error("state_cache: invalid dimensions");
    }
    const std::int32_t max_il = linear_layer_indices.empty()
        ? -1 : *std::max_element(linear_layer_indices.begin(),
                                  linear_layer_indices.end());
    is_linear_.assign(max_il + 1, false);
    for (auto il : linear_layer_indices) {
        if (il < 0) throw std::runtime_error("state_cache: negative layer idx");
        is_linear_[il] = true;
    }
    ssm_states_.assign(max_il + 1, nullptr);
    conv_states_.assign(max_il + 1, nullptr);
    allocate_();
}

void StateCache::allocate_() {
    constexpr std::size_t MAX_TENSORS = 4096;
    ggml_init_params ip{
        /*.mem_size   =*/ ggml_tensor_overhead() * MAX_TENSORS,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_ = ggml_init(ip);
    if (!ctx_) throw std::runtime_error("state_cache: ggml_init failed");

    for (std::size_t il = 0; il < is_linear_.size(); ++il) {
        if (!is_linear_[il]) continue;
        // SSM state: [head_k_dim, head_v_dim, n_heads, n_slots].
        auto* ssm = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32,
                                        head_k_dim_, head_v_dim_,
                                        n_heads_, n_slots_);
        ggml_set_name(ssm, ("state.ssm." + std::to_string(il)).c_str());
        ssm_states_[il] = ssm;
        // Conv state: [conv_kernel-1, conv_dim, n_slots].
        auto* conv = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32,
                                        conv_kernel_ - 1, conv_dim_, n_slots_);
        ggml_set_name(conv, ("state.conv." + std::to_string(il)).c_str());
        conv_states_[il] = conv;
    }

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buf_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
        throw std::runtime_error(
            "state_cache: ggml_backend_alloc_ctx_tensors failed");
    }

    // Zero the entire backing buffer — the recurrent state must start
    // at zero for the first token of every context.
    std::vector<float> zeros_ssm(static_cast<std::size_t>(head_k_dim_) *
                                  head_v_dim_ * n_heads_ * n_slots_, 0.0f);
    std::vector<float> zeros_conv(static_cast<std::size_t>(conv_kernel_ - 1) *
                                   conv_dim_ * n_slots_, 0.0f);
    for (std::size_t il = 0; il < is_linear_.size(); ++il) {
        if (!is_linear_[il]) continue;
        ggml_backend_tensor_set(ssm_states_[il], zeros_ssm.data(), 0,
                                zeros_ssm.size() * sizeof(float));
        ggml_backend_tensor_set(conv_states_[il], zeros_conv.data(), 0,
                                zeros_conv.size() * sizeof(float));
    }
}

StateCache::~StateCache() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

ggml_tensor* StateCache::ssm(std::int32_t layer) const noexcept {
    if (layer < 0 || static_cast<std::size_t>(layer) >= ssm_states_.size()) {
        return nullptr;
    }
    return ssm_states_[layer];
}

ggml_tensor* StateCache::conv(std::int32_t layer) const noexcept {
    if (layer < 0 || static_cast<std::size_t>(layer) >= conv_states_.size()) {
        return nullptr;
    }
    return conv_states_[layer];
}

bool StateCache::is_linear_layer(std::int32_t il) const noexcept {
    if (il < 0 || static_cast<std::size_t>(il) >= is_linear_.size()) {
        return false;
    }
    return is_linear_[il];
}

void StateCache::zero_slot(std::int32_t slot) {
    if (slot < 0 || slot >= n_slots_) return;
    const std::size_t ssm_slot_bytes =
        static_cast<std::size_t>(head_k_dim_) * head_v_dim_ *
        n_heads_ * sizeof(float);
    const std::size_t conv_slot_bytes =
        static_cast<std::size_t>(conv_kernel_ - 1) * conv_dim_ * sizeof(float);
    std::vector<float> zeros_ssm(ssm_slot_bytes / sizeof(float), 0.0f);
    std::vector<float> zeros_conv(conv_slot_bytes / sizeof(float), 0.0f);
    for (std::size_t il = 0; il < is_linear_.size(); ++il) {
        if (!is_linear_[il]) continue;
        ggml_backend_tensor_set(
            ssm_states_[il], zeros_ssm.data(),
            static_cast<std::size_t>(slot) * ssm_slot_bytes, ssm_slot_bytes);
        ggml_backend_tensor_set(
            conv_states_[il], zeros_conv.data(),
            static_cast<std::size_t>(slot) * conv_slot_bytes, conv_slot_bytes);
    }
}

void StateCache::copy_slot(std::int32_t src_slot, std::int32_t dst_slot) {
    if (src_slot < 0 || src_slot >= n_slots_ ||
        dst_slot < 0 || dst_slot >= n_slots_ || src_slot == dst_slot) {
        return;
    }
    const std::size_t ssm_slot_bytes =
        static_cast<std::size_t>(head_k_dim_) * head_v_dim_ *
        n_heads_ * sizeof(float);
    const std::size_t conv_slot_bytes =
        static_cast<std::size_t>(conv_kernel_ - 1) * conv_dim_ * sizeof(float);
    std::vector<float> tmp_ssm(ssm_slot_bytes / sizeof(float));
    std::vector<float> tmp_conv(conv_slot_bytes / sizeof(float));
    for (std::size_t il = 0; il < is_linear_.size(); ++il) {
        if (!is_linear_[il]) continue;
        ggml_backend_tensor_get(
            ssm_states_[il], tmp_ssm.data(),
            static_cast<std::size_t>(src_slot) * ssm_slot_bytes, ssm_slot_bytes);
        ggml_backend_tensor_set(
            ssm_states_[il], tmp_ssm.data(),
            static_cast<std::size_t>(dst_slot) * ssm_slot_bytes, ssm_slot_bytes);
        ggml_backend_tensor_get(
            conv_states_[il], tmp_conv.data(),
            static_cast<std::size_t>(src_slot) * conv_slot_bytes, conv_slot_bytes);
        ggml_backend_tensor_set(
            conv_states_[il], tmp_conv.data(),
            static_cast<std::size_t>(dst_slot) * conv_slot_bytes, conv_slot_bytes);
    }
}

std::size_t StateCache::buffer_size() const noexcept {
    return buf_ ? ggml_backend_buffer_get_size(buf_) : 0;
}

}  // namespace pie_portable_driver
