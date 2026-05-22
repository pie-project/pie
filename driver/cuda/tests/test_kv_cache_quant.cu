#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/kv_paged.hpp"
#include "kv_cache.hpp"
#include "kv_cache_format.hpp"
#include "tensor.hpp"

using pie_cuda_driver::DType;
using pie_cuda_driver::DeviceTensor;
using pie_cuda_driver::KvCacheScheme;
using pie_cuda_driver::cuda_check_impl;
using pie_cuda_driver::kernels::launch_write_kv_to_pages;
using pie_cuda_driver::kv_cache_format_from_string;

namespace {

constexpr int kTotalTokens = 2;
constexpr int kNumPages = 2;
constexpr int kPageSize = 4;
constexpr int kHeads = 2;
constexpr int kHeadDim = 17;
constexpr int kActivePage = 1;

float source_value(int token, int head, int dim, bool value) {
    const float sign = ((token + head + dim + (value ? 1 : 0)) & 1) ? -1.f : 1.f;
    const float base = 0.03f * static_cast<float>(dim - 8) +
                       0.11f * static_cast<float>(head + 1) +
                       0.17f * static_cast<float>(token + 1);
    return sign * base;
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return ptr;
}

std::vector<__nv_bfloat16> make_current(bool value) {
    std::vector<__nv_bfloat16> out(kTotalTokens * kHeads * kHeadDim);
    for (int t = 0; t < kTotalTokens; ++t) {
        for (int h = 0; h < kHeads; ++h) {
            for (int d = 0; d < kHeadDim; ++d) {
                const int idx = (t * kHeads + h) * kHeadDim + d;
                out[idx] = __float2bfloat16(source_value(t, h, d, value));
            }
        }
    }
    return out;
}

float fp8_qdq(float x, __nv_fp8_interpretation_t kind) {
    const auto q = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, kind);
    const __half h = __nv_cvt_fp8_to_halfraw(q, kind);
    return __half2float(h);
}

float bf16_round(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

float fp4_value(std::uint8_t code) {
    static constexpr float levels[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = levels[code & 0x7];
    return (code & 0x8) ? -v : v;
}

std::uint8_t quant_fp4(float x) {
    static constexpr float levels[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const bool neg = x < 0.f;
    const float ax = std::fabs(x);
    int best = 0;
    float best_err = std::fabs(ax - levels[0]);
    for (int i = 1; i < 8; ++i) {
        const float err = std::fabs(ax - levels[i]);
        if (err < best_err) {
            best_err = err;
            best = i;
        }
    }
    return static_cast<std::uint8_t>((neg ? 0x8 : 0) | best);
}

float expected_qdq(const std::vector<__nv_bfloat16>& src,
                   const pie_cuda_driver::KvCacheFormat& fmt,
                   int token,
                   int head,
                   int dim) {
    const int idx = (token * kHeads + head) * kHeadDim + dim;
    const float x = __bfloat162float(src[idx]);

    if (fmt.scheme == KvCacheScheme::Fp8PerTensor) {
        const auto kind = fmt.storage_dtype == DType::FP8_E5M2 ? __NV_E5M2 : __NV_E4M3;
        return bf16_round(fp8_qdq(x, kind));
    }
    if (fmt.scheme == KvCacheScheme::Int8PerTokenHead) {
        float max_abs = 0.f;
        for (int d = 0; d < kHeadDim; ++d) {
            const int j = (token * kHeads + head) * kHeadDim + d;
            max_abs = std::max(max_abs, std::fabs(__bfloat162float(src[j])));
        }
        const float scale = max_abs > 0.f ? max_abs / 127.f : 1.f;
        int q = static_cast<int>(std::rint(x / scale));
        q = std::max(-128, std::min(127, q));
        return bf16_round(static_cast<float>(q) * scale);
    }
    if (fmt.scheme == KvCacheScheme::Fp8PerTokenHead) {
        float max_abs = 0.f;
        for (int d = 0; d < kHeadDim; ++d) {
            const int j = (token * kHeads + head) * kHeadDim + d;
            max_abs = std::max(max_abs, std::fabs(__bfloat162float(src[j])));
        }
        const float scale = max_abs > 0.f ? max_abs / 448.f : 1.f;
        return bf16_round(fp8_qdq(x / scale, __NV_E4M3) * scale);
    }
    if (fmt.scheme == KvCacheScheme::Fp4Block) {
        const int block = fmt.block_size > 0 ? fmt.block_size : 16;
        const int start = (dim / block) * block;
        const int end = std::min(start + block, kHeadDim);
        float max_abs = 0.f;
        for (int d = start; d < end; ++d) {
            const int j = (token * kHeads + head) * kHeadDim + d;
            max_abs = std::max(max_abs, std::fabs(__bfloat162float(src[j])));
        }
        const float scale = max_abs > 0.f ? max_abs / 6.f : 1.f;
        return bf16_round(fp4_value(quant_fp4(x / scale)) * scale);
    }
    return x;
}

float expected_scale(const std::vector<__nv_bfloat16>& src,
                     const pie_cuda_driver::KvCacheFormat& fmt,
                     int token,
                     int head,
                     int block_idx) {
    if (fmt.scheme == KvCacheScheme::Int8PerTokenHead ||
        fmt.scheme == KvCacheScheme::Fp8PerTokenHead) {
        const float qmax = fmt.scheme == KvCacheScheme::Fp8PerTokenHead ? 448.f : 127.f;
        float max_abs = 0.f;
        for (int d = 0; d < kHeadDim; ++d) {
            const int j = (token * kHeads + head) * kHeadDim + d;
            max_abs = std::max(max_abs, std::fabs(__bfloat162float(src[j])));
        }
        return max_abs > 0.f ? max_abs / qmax : 1.f;
    }
    if (fmt.scheme == KvCacheScheme::Fp4Block) {
        const int block = fmt.block_size > 0 ? fmt.block_size : 16;
        const int start = block_idx * block;
        const int end = std::min(start + block, kHeadDim);
        float max_abs = 0.f;
        for (int d = start; d < end; ++d) {
            const int j = (token * kHeads + head) * kHeadDim + d;
            max_abs = std::max(max_abs, std::fabs(__bfloat162float(src[j])));
        }
        return max_abs > 0.f ? max_abs / 6.f : 1.f;
    }
    return 1.f;
}

void assert_close(float actual, float expected, float tol, const std::string& label) {
    const float diff = std::fabs(actual - expected);
    if (diff > tol) {
        throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected) +
                                 " diff=" + std::to_string(diff));
    }
}

void run_format(const char* dtype) {
    auto fmt = kv_cache_format_from_string(dtype);
    auto cache = pie_cuda_driver::KvCache::allocate(
        1, kNumPages, kPageSize, kHeads, kHeadDim, fmt);

    const auto k_host = make_current(false);
    const auto v_host = make_current(true);
    DeviceTensor k_curr = DeviceTensor::allocate(
        DType::BF16, {kTotalTokens, kHeads, kHeadDim});
    DeviceTensor v_curr = DeviceTensor::allocate(
        DType::BF16, {kTotalTokens, kHeads, kHeadDim});
    CUDA_CHECK(cudaMemcpy(k_curr.data(), k_host.data(),
                          k_host.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_curr.data(), v_host.data(),
                          v_host.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    auto* qo_indptr = device_copy<std::uint32_t>({0, kTotalTokens});
    auto* kv_page_indices = device_copy<std::uint32_t>({kActivePage});
    auto* kv_page_indptr = device_copy<std::uint32_t>({0, 1});
    auto* kv_last_page_lens = device_copy<std::uint32_t>({kTotalTokens});

    auto layer = cache.layer_view(0);
    launch_write_kv_to_pages(
        layer, k_curr.data(), v_curr.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        kTotalTokens, 1, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int scale_blocks = fmt.scheme == KvCacheScheme::Fp4Block
        ? (kHeadDim + fmt.block_size - 1) / fmt.block_size
        : 1;
    std::vector<float> k_scales;
    std::vector<float> v_scales;
    if (fmt.has_side_scales()) {
        k_scales.resize(kNumPages * kPageSize * kHeads * scale_blocks);
        v_scales.resize(kNumPages * kPageSize * kHeads * scale_blocks);
        CUDA_CHECK(cudaMemcpy(k_scales.data(), cache.k_scale(0),
                              k_scales.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_scales.data(), cache.v_scale(0),
                              v_scales.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    const int packed_dim = fmt.scheme == KvCacheScheme::Fp4Block
        ? (kHeadDim + 1) / 2
        : kHeadDim;
    const std::size_t storage_elems =
        static_cast<std::size_t>(kNumPages) * kPageSize * kHeads * packed_dim;
    std::vector<std::uint8_t> k_u8;
    std::vector<std::uint8_t> v_u8;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    if (fmt.storage_dtype == DType::INT8) {
        k_i8.resize(storage_elems);
        v_i8.resize(storage_elems);
        CUDA_CHECK(cudaMemcpy(k_i8.data(), cache.k(0),
                              storage_elems * sizeof(std::int8_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_i8.data(), cache.v(0),
                              storage_elems * sizeof(std::int8_t),
                              cudaMemcpyDeviceToHost));
    } else {
        k_u8.resize(storage_elems);
        v_u8.resize(storage_elems);
        CUDA_CHECK(cudaMemcpy(k_u8.data(), cache.k(0),
                              storage_elems * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_u8.data(), cache.v(0),
                              storage_elems * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToHost));
    }

    auto cached_value = [&](bool value, int t, int h, int d) -> float {
        const long long token_head =
            (static_cast<long long>(kActivePage) * kPageSize + t) * kHeads + h;
        if (fmt.scheme == KvCacheScheme::Fp8PerTensor ||
            fmt.scheme == KvCacheScheme::Fp8PerTokenHead) {
            const auto kind =
                fmt.storage_dtype == DType::FP8_E5M2 ? __NV_E5M2 : __NV_E4M3;
            const auto& raw = value ? v_u8 : k_u8;
            const auto fp8 =
                static_cast<__nv_fp8_storage_t>(
                    raw[token_head * kHeadDim + d]);
            const __half hval = __nv_cvt_fp8_to_halfraw(fp8, kind);
            float out = __half2float(hval);
            if (fmt.scheme == KvCacheScheme::Fp8PerTokenHead) {
                const auto& scales = value ? v_scales : k_scales;
                out *= scales[token_head];
            }
            return out;
        }
        if (fmt.scheme == KvCacheScheme::Int8PerTokenHead) {
            const auto& raw = value ? v_i8 : k_i8;
            const auto& scales = value ? v_scales : k_scales;
            return static_cast<float>(raw[token_head * kHeadDim + d]) *
                   scales[token_head];
        }
        if (fmt.scheme == KvCacheScheme::Fp4Block) {
            const auto& raw = value ? v_u8 : k_u8;
            const auto& scales = value ? v_scales : k_scales;
            const long long packed_idx = token_head * packed_dim + d / 2;
            const int shift = (d & 1) ? 4 : 0;
            const std::uint8_t code = (raw[packed_idx] >> shift) & 0xf;
            return fp4_value(code) *
                   scales[token_head * scale_blocks +
                          d / (fmt.block_size > 0 ? fmt.block_size : 16)];
        }
        return 0.f;
    };

    for (int t = 0; t < kTotalTokens; ++t) {
        for (int h = 0; h < kHeads; ++h) {
            for (int d = 0; d < kHeadDim; ++d) {
                const int dst = ((kActivePage * kPageSize + t) * kHeads + h) *
                                kHeadDim + d;
                const std::string where =
                    std::string(dtype) + " token=" + std::to_string(t) +
                    " head=" + std::to_string(h) + " dim=" + std::to_string(d);
                (void)dst;
                assert_close(cached_value(false, t, h, d),
                             expected_qdq(k_host, fmt, t, h, d), 0.025f,
                             "K " + where);
                assert_close(cached_value(true, t, h, d),
                             expected_qdq(v_host, fmt, t, h, d), 0.025f,
                             "V " + where);
            }
        }
    }

    if (fmt.has_side_scales()) {
        for (int t = 0; t < kTotalTokens; ++t) {
            for (int h = 0; h < kHeads; ++h) {
                for (int b = 0; b < scale_blocks; ++b) {
                    const int idx =
                        ((kActivePage * kPageSize + t) * kHeads + h) *
                        scale_blocks + b;
                    assert_close(k_scales[idx], expected_scale(k_host, fmt, t, h, b),
                                 1e-5f, std::string(dtype) + " K scale");
                    assert_close(v_scales[idx], expected_scale(v_host, fmt, t, h, b),
                                 1e-5f, std::string(dtype) + " V scale");
                }
            }
        }
    }

    CUDA_CHECK(cudaFree(qo_indptr));
    CUDA_CHECK(cudaFree(kv_page_indices));
    CUDA_CHECK(cudaFree(kv_page_indptr));
    CUDA_CHECK(cudaFree(kv_last_page_lens));
}

}  // namespace

int main() {
    try {
        run_format("fp8_e4m3");
        run_format("fp8_e5m2");
        run_format("int8_per_token_head");
        run_format("fp8_per_token_head");
        run_format("fp4_e2m1");
        run_format("nvfp4");
        std::puts("kv_cache_quant ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_kv_cache_quant failed: %s\n", e.what());
        return 1;
    }
}
