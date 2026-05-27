#include "kv_cache_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace pie_portable_driver {

namespace {

constexpr float kMaxFp8E4M3 = 448.0f;
constexpr float kMaxFp8E5M2 = 57344.0f;
constexpr std::int32_t kFp4BlockSize = 16;

std::string normalize_dtype(std::string_view dtype) {
    std::string out(dtype.empty() ? "auto" : dtype);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

float fake_fp_qdq_scalar(float x,
                         int mantissa_bits,
                         float max_value) {
    if (x == 0.0f || !std::isfinite(x)) return x;
    x = std::clamp(x, -max_value, max_value);
    const float ax = std::fabs(x);
    const float exp = std::floor(std::log2(ax));
    const float step = std::ldexp(1.0f, static_cast<int>(exp) - mantissa_bits);
    const float q = std::round(ax / step) * step;
    return std::copysign(std::min(q, max_value), x);
}

void qdq_fp8_per_tensor(float* row,
                        std::int32_t n,
                        int mantissa_bits,
                        float max_value) {
    for (std::int32_t i = 0; i < n; ++i) {
        row[i] = fake_fp_qdq_scalar(row[i], mantissa_bits, max_value);
    }
}

void qdq_int8_per_token_head(float* row,
                             std::int32_t kv_heads,
                             std::int32_t head_dim) {
    for (std::int32_t h = 0; h < kv_heads; ++h) {
        float* seg = row + static_cast<std::size_t>(h) * head_dim;
        float max_abs = 0.0f;
        for (std::int32_t d = 0; d < head_dim; ++d) {
            max_abs = std::max(max_abs, std::fabs(seg[d]));
        }
        const float scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        for (std::int32_t d = 0; d < head_dim; ++d) {
            const float q = std::clamp(std::round(seg[d] / scale), -128.0f, 127.0f);
            seg[d] = q * scale;
        }
    }
}

void qdq_fp8_per_token_head(float* row,
                            std::int32_t kv_heads,
                            std::int32_t head_dim) {
    for (std::int32_t h = 0; h < kv_heads; ++h) {
        float* seg = row + static_cast<std::size_t>(h) * head_dim;
        float max_abs = 0.0f;
        for (std::int32_t d = 0; d < head_dim; ++d) {
            max_abs = std::max(max_abs, std::fabs(seg[d]));
        }
        const float scale = max_abs > 0.0f ? (max_abs / kMaxFp8E4M3) : 1.0f;
        for (std::int32_t d = 0; d < head_dim; ++d) {
            seg[d] = fake_fp_qdq_scalar(seg[d] / scale, 3, kMaxFp8E4M3) * scale;
        }
    }
}

void qdq_fp4_block(float* row,
                   std::int32_t kv_heads,
                   std::int32_t head_dim,
                   std::int32_t block_size) {
    constexpr float kLevels[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (std::int32_t h = 0; h < kv_heads; ++h) {
        float* seg = row + static_cast<std::size_t>(h) * head_dim;
        for (std::int32_t b0 = 0; b0 < head_dim; b0 += block_size) {
            const std::int32_t b1 = std::min<std::int32_t>(head_dim, b0 + block_size);
            float max_abs = 0.0f;
            for (std::int32_t d = b0; d < b1; ++d) {
                max_abs = std::max(max_abs, std::fabs(seg[d]));
            }
            const float scale = max_abs > 0.0f ? (max_abs / 6.0f) : 1.0f;
            for (std::int32_t d = b0; d < b1; ++d) {
                const float y = std::fabs(seg[d] / scale);
                float best = kLevels[0];
                float best_dist = std::fabs(y - best);
                for (float level : kLevels) {
                    const float dist = std::fabs(y - level);
                    if (dist < best_dist) {
                        best = level;
                        best_dist = dist;
                    }
                }
                seg[d] = std::copysign(best * scale, seg[d]);
            }
        }
    }
}

struct QdqOpParams {
    KvCacheQuantFormat format;
    std::int32_t kv_heads = 0;
    std::int32_t head_dim = 0;
};

const QdqOpParams* intern_params(const KvCacheQuantFormat& format,
                                 std::int32_t kv_heads,
                                 std::int32_t head_dim) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<QdqOpParams>> params;

    std::lock_guard<std::mutex> lock(mu);
    for (const auto& p : params) {
        if (p->format.name == format.name &&
            p->kv_heads == kv_heads &&
            p->head_dim == head_dim) {
            return p.get();
        }
    }
    auto next = std::make_unique<QdqOpParams>();
    next->format = format;
    next->kv_heads = kv_heads;
    next->head_dim = head_dim;
    const QdqOpParams* out = next.get();
    params.push_back(std::move(next));
    return out;
}

float load_scalar(const char* base, ggml_type type, std::size_t offset) {
    switch (type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float*>(base + offset);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(base + offset));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(base + offset));
        default:
            return 0.0f;
    }
}

void store_scalar(char* base, ggml_type type, std::size_t offset, float value) {
    switch (type) {
        case GGML_TYPE_F32:
            *reinterpret_cast<float*>(base + offset) = value;
            break;
        case GGML_TYPE_F16:
            *reinterpret_cast<ggml_fp16_t*>(base + offset) = ggml_fp32_to_fp16(value);
            break;
        case GGML_TYPE_BF16:
            *reinterpret_cast<ggml_bf16_t*>(base + offset) = ggml_fp32_to_bf16(value);
            break;
        default:
            break;
    }
}

void qdq_append_op(ggml_tensor* dst,
                   const ggml_tensor* src,
                   int ith,
                   int nth,
                   void* userdata) {
    const auto* params = static_cast<const QdqOpParams*>(userdata);
    const std::int64_t row_elems =
        static_cast<std::int64_t>(params->kv_heads) * params->head_dim;
    const std::int64_t n_cols =
        src->ne[1] * src->ne[2] * src->ne[3];
    const char* src_base = static_cast<const char*>(src->data);
    char* dst_base = static_cast<char*>(dst->data);
    std::vector<float> row(static_cast<std::size_t>(row_elems));

    for (std::int64_t c = ith; c < n_cols; c += nth) {
        const std::int64_t i1 = c % src->ne[1];
        const std::int64_t i2 = (c / src->ne[1]) % src->ne[2];
        const std::int64_t i3 = c / (src->ne[1] * src->ne[2]);
        const std::size_t src_col_off =
            static_cast<std::size_t>(i1) * src->nb[1] +
            static_cast<std::size_t>(i2) * src->nb[2] +
            static_cast<std::size_t>(i3) * src->nb[3];
        const std::size_t dst_col_off =
            static_cast<std::size_t>(i1) * dst->nb[1] +
            static_cast<std::size_t>(i2) * dst->nb[2] +
            static_cast<std::size_t>(i3) * dst->nb[3];

        for (std::int64_t i = 0; i < row_elems; ++i) {
            row[static_cast<std::size_t>(i)] = load_scalar(
                src_base, src->type, src_col_off + static_cast<std::size_t>(i) * src->nb[0]);
        }
        qdq_kv_row(row.data(), params->kv_heads, params->head_dim, params->format);
        for (std::int64_t i = 0; i < row_elems; ++i) {
            store_scalar(dst_base, dst->type,
                         dst_col_off + static_cast<std::size_t>(i) * dst->nb[0],
                         row[static_cast<std::size_t>(i)]);
        }
    }
}

}  // namespace

std::string valid_kv_cache_dtype_values() {
    return "auto, bf16, bfloat16, fp8_e4m3, fp8_e5m2, int8_per_token_head, "
           "fp8_per_token_head, fp4_e2m1, nvfp4";
}

KvCacheQuantFormat kv_cache_quant_format_from_string(std::string_view dtype) {
    const std::string name = normalize_dtype(dtype);
    if (name == "auto" || name == "bf16" || name == "bfloat16") {
        return {};
    }
    if (name == "fp8_e4m3") {
        return {name, KvCacheQuantScheme::Fp8PerTensor, 4, 3, kMaxFp8E4M3, 0};
    }
    if (name == "fp8_e5m2") {
        return {name, KvCacheQuantScheme::Fp8PerTensor, 5, 2, kMaxFp8E5M2, 0};
    }
    if (name == "int8_per_token_head") {
        return {name, KvCacheQuantScheme::Int8PerTokenHead, 0, 0, 0.0f, 0};
    }
    if (name == "fp8_per_token_head") {
        return {name, KvCacheQuantScheme::Fp8PerTokenHead, 4, 3, kMaxFp8E4M3, 0};
    }
    if (name == "fp4_e2m1" || name == "nvfp4") {
        return {name, KvCacheQuantScheme::Fp4Block, 2, 1, 6.0f, kFp4BlockSize};
    }
    throw std::runtime_error("invalid kv_cache_dtype '" + std::string(dtype) +
                             "'; expected one of: " + valid_kv_cache_dtype_values());
}

void qdq_kv_row(float* row,
                std::int32_t kv_heads,
                std::int32_t head_dim,
                const KvCacheQuantFormat& format) {
    const std::int32_t n = kv_heads * head_dim;
    switch (format.scheme) {
        case KvCacheQuantScheme::Native:
            return;
        case KvCacheQuantScheme::Fp8PerTensor:
            qdq_fp8_per_tensor(row, n, format.mantissa_bits, format.max_value);
            return;
        case KvCacheQuantScheme::Int8PerTokenHead:
            qdq_int8_per_token_head(row, kv_heads, head_dim);
            return;
        case KvCacheQuantScheme::Fp8PerTokenHead:
            qdq_fp8_per_token_head(row, kv_heads, head_dim);
            return;
        case KvCacheQuantScheme::Fp4Block:
            qdq_fp4_block(row, kv_heads, head_dim,
                          format.block_size > 0 ? format.block_size : kFp4BlockSize);
            return;
    }
}

ggml_tensor* qdq_tensor_for_append(ggml_context* ctx,
                                   ggml_tensor* tensor,
                                   const KvCacheQuantFormat& format,
                                   std::int32_t kv_heads,
                                   std::int32_t head_dim) {
    if (format.is_native()) {
        return tensor;
    }
    const std::int64_t expected = static_cast<std::int64_t>(kv_heads) * head_dim;
    if (tensor->ne[0] != expected) {
        throw std::runtime_error("portable kv qdq: append tensor row width mismatch");
    }
    if (tensor->type != GGML_TYPE_F32 &&
        tensor->type != GGML_TYPE_F16 &&
        tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error("portable kv qdq: append tensor must be F32/F16/BF16");
    }
    auto* out = ggml_map_custom1(
        ctx, tensor, qdq_append_op, GGML_N_TASKS_MAX,
        const_cast<QdqOpParams*>(intern_params(format, kv_heads, head_dim)));
    ggml_set_name(out, "kv_append_qdq");
    return out;
}

}  // namespace pie_portable_driver
