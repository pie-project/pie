#include "kv_cache_format.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace pie_cuda_driver {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

}  // namespace

std::int64_t KvCacheFormat::storage_head_dim(int head_dim) const noexcept {
    if (scheme == KvCacheScheme::Fp4Block) {
        return (static_cast<std::int64_t>(head_dim) + 1) / 2;
    }
    return head_dim;
}

std::size_t KvCacheFormat::kv_bytes_per_page(int page_size,
                                             int num_kv_heads,
                                             int head_dim) const noexcept {
    return static_cast<std::size_t>(page_size) *
           static_cast<std::size_t>(num_kv_heads) *
           static_cast<std::size_t>(storage_head_dim(head_dim)) *
           dtype_bytes(storage_dtype);
}

std::size_t KvCacheFormat::scale_bytes_per_page(int page_size,
                                                int num_kv_heads,
                                                int head_dim) const noexcept {
    if (scale_layout == KvCacheScaleLayout::None) return 0;
    std::size_t scales_per_head = 1;
    if (scale_layout == KvCacheScaleLayout::PerTokenHeadBlock) {
        const int bs = block_size > 0 ? block_size : 16;
        scales_per_head = static_cast<std::size_t>((head_dim + bs - 1) / bs);
    }
    return static_cast<std::size_t>(page_size) *
           static_cast<std::size_t>(num_kv_heads) *
           scales_per_head *
           dtype_bytes(DType::FP32);
}

KvCacheFormat kv_cache_format_from_string(const std::string& value,
                                          const std::string& activation_dtype) {
    const std::string v = lower(value.empty() ? "auto" : value);
    const std::string act = lower(activation_dtype);

    if (v == "auto") {
        if (act == "bf16" || act == "bfloat16" || act.empty()) {
            return KvCacheFormat{};
        }
        // CUDA native currently computes activations in bf16. Keep auto
        // behavior identical to the historical path.
        return KvCacheFormat{};
    }
    if (v == "bf16" || v == "bfloat16") {
        return KvCacheFormat{};
    }
    if (v == "fp8_e4m3") {
        return KvCacheFormat{
            .name = "fp8_e4m3",
            .scheme = KvCacheScheme::Fp8PerTensor,
            .scale_layout = KvCacheScaleLayout::None,
            .storage_dtype = DType::FP8_E4M3,
            .block_size = 0,
        };
    }
    if (v == "fp8_e5m2") {
        return KvCacheFormat{
            .name = "fp8_e5m2",
            .scheme = KvCacheScheme::Fp8PerTensor,
            .scale_layout = KvCacheScaleLayout::None,
            .storage_dtype = DType::FP8_E5M2,
            .block_size = 0,
        };
    }
    if (v == "int8_per_token_head") {
        return KvCacheFormat{
            .name = "int8_per_token_head",
            .scheme = KvCacheScheme::Int8PerTokenHead,
            .scale_layout = KvCacheScaleLayout::PerTokenHead,
            .storage_dtype = DType::INT8,
            .block_size = 0,
        };
    }
    if (v == "fp8_per_token_head") {
        return KvCacheFormat{
            .name = "fp8_per_token_head",
            .scheme = KvCacheScheme::Fp8PerTokenHead,
            .scale_layout = KvCacheScaleLayout::PerTokenHead,
            .storage_dtype = DType::FP8_E4M3,
            .block_size = 0,
        };
    }
    if (v == "fp4_e2m1" || v == "nvfp4") {
        return KvCacheFormat{
            .name = v,
            .scheme = KvCacheScheme::Fp4Block,
            .scale_layout = KvCacheScaleLayout::PerTokenHeadBlock,
            .storage_dtype = DType::UINT8,
            .block_size = 16,
        };
    }
    throw std::runtime_error(
        "invalid kv_cache_dtype '" + value + "'; expected one of: " +
        valid_kv_cache_dtype_values());
}

bool is_valid_kv_cache_dtype(const std::string& value) noexcept {
    try {
        (void)kv_cache_format_from_string(value);
        return true;
    } catch (...) {
        return false;
    }
}

std::string valid_kv_cache_dtype_values() {
    return "auto, bf16, bfloat16, fp8_e4m3, fp8_e5m2, "
           "int8_per_token_head, fp8_per_token_head, fp4_e2m1, nvfp4";
}

}  // namespace pie_cuda_driver
