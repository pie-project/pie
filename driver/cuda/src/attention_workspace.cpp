#include "attention_workspace.hpp"

#include <algorithm>
#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

#include "config.hpp"
#include "cuda_check.hpp"
#include "loader/hf_config.hpp"

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
    return ws;
}

AttentionWorkspace::AttentionWorkspace(AttentionWorkspace&& other) noexcept
    : float_buf_(std::move(other.float_buf_)),
      int_buf_(std::move(other.int_buf_)),
      page_locked_int_(other.page_locked_int_)
{
    other.page_locked_int_ = nullptr;
}

AttentionWorkspace& AttentionWorkspace::operator=(AttentionWorkspace&& other) noexcept {
    if (this != &other) {
        if (page_locked_int_) {
            cudaFreeHost(page_locked_int_);
        }
        float_buf_ = std::move(other.float_buf_);
        int_buf_ = std::move(other.int_buf_);
        page_locked_int_ = other.page_locked_int_;
        other.page_locked_int_ = nullptr;
    }
    return *this;
}

AttentionWorkspace::~AttentionWorkspace() {
    if (page_locked_int_) {
        cudaFreeHost(page_locked_int_);
        page_locked_int_ = nullptr;
    }
}

bool flashinfer_decode_supports_gqa(int gqa) {
    return gqa == 1 || gqa == 2 || gqa == 3 || gqa == 4 || gqa == 8;
}

bool xqa_decode_enabled_by_env() {
    const char* v = std::getenv("PIE_CUDA_XQA_DECODE");
    if (v == nullptr || v[0] == '\0') return true;
    return v[0] != '0';
}

bool has_non_full_attention_layers(const HfConfig& hf) {
    for (const auto& kind : hf.layer_types) {
        if (kind != "full_attention") return true;
    }
    return false;
}

std::size_t attention_float_workspace_bytes(const HfConfig& hf,
                                            const Config& cfg,
                                            const cudaDeviceProp&,
                                            int max_requests) {
    const bool qwen_hybrid =
        hf.model_type == "qwen3_5" ||
        hf.model_type == "qwen3_5_text" ||
        hf.model_type == "qwen3_5_moe" ||
        hf.model_type == "qwen3_5_moe_text";
    const std::size_t base =
        qwen_hybrid ? 128ull * 1024 * 1024 : 80ull * 1024 * 1024;
    const int tp_size = std::max(1, cfg.distributed.tp_size);
    if (tp_size != 1 || max_requests <= 0) {
        return base;
    }
    if (hf.num_key_value_heads <= 0 ||
        hf.num_attention_heads % hf.num_key_value_heads != 0) {
        return base;
    }
    const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
    const bool gqa_in_decode_set = flashinfer_decode_supports_gqa(gqa);
    const bool supported_head_dim =
        hf.head_dim_kernel == 64 || hf.head_dim_kernel == 128 ||
        hf.head_dim_kernel == 256 || hf.head_dim_kernel == 512;
    if (!gqa_in_decode_set || !supported_head_dim || hf.sliding_window >= 0 ||
        has_non_full_attention_layers(hf)) {
        return base;
    }
    auto align_up = [](std::size_t n, std::size_t a) {
        return (n + (a - 1)) & ~(a - 1);
    };
    const std::size_t q_heads =
        static_cast<std::size_t>(hf.num_attention_heads / tp_size);
    const std::size_t head_dim = static_cast<std::size_t>(hf.head_dim_kernel);
    const std::size_t cta_tile_q = 16;
    const std::size_t padded_batch =
        align_up(static_cast<std::size_t>(max_requests) * 2, 128);
    const std::size_t tmp_v =
        q_heads * padded_batch * cta_tile_q * head_dim * sizeof(float);
    const std::size_t tmp_s =
        q_heads * padded_batch * cta_tile_q * sizeof(float);
    const std::size_t planned = tmp_v + tmp_s + 16ull * 1024 * 1024;
    return std::max(base, align_up(planned, 16ull * 1024 * 1024));
}

}  // namespace pie_cuda_driver
