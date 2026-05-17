#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include <ggml.h>

namespace pie_portable_driver {

enum class KvCacheQuantScheme {
    Native,
    Fp8PerTensor,
    Int8PerTokenHead,
    Fp8PerTokenHead,
    Fp4Block,
};

struct KvCacheQuantFormat {
    std::string name = "bf16";
    KvCacheQuantScheme scheme = KvCacheQuantScheme::Native;
    std::int32_t exponent_bits = 0;
    std::int32_t mantissa_bits = 0;
    float max_value = 0.0f;
    std::int32_t block_size = 0;

    bool is_native() const noexcept { return scheme == KvCacheQuantScheme::Native; }
};

std::string valid_kv_cache_dtype_values();

KvCacheQuantFormat kv_cache_quant_format_from_string(std::string_view dtype);

void qdq_kv_row(float* row,
                std::int32_t kv_heads,
                std::int32_t head_dim,
                const KvCacheQuantFormat& format);

ggml_tensor* qdq_tensor_for_append(ggml_context* ctx,
                                   ggml_tensor* tensor,
                                   const KvCacheQuantFormat& format,
                                   std::int32_t kv_heads,
                                   std::int32_t head_dim);

}  // namespace pie_portable_driver
