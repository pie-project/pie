// Numerical parity for the composable transcode dispatch. Each registered
// (source, target) pair must produce BIT-IDENTICAL output to its reference:
//   * FP8E4m3PerGroup -> Mxfp4  vs  dequant_fp8_..._per_group + quantize_bf16_to_mxfp4
//   * Bf16            -> Mxfp4  vs  quantize_bf16_to_mxfp4 (decode is a no-op)
// Adding a pair here = add a Decode/Encode functor + one switch arm in
// transcode.cu, then a case below.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/dequant_fp8.hpp"
#include "kernels/quant_bf16_to_mxfp4.hpp"
#include "kernels/transcode.hpp"

namespace {

namespace K = pie_cuda_driver::kernels;

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                         #cond);                                            \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t _err = (expr);                                          \
        if (_err != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA FAIL %s:%d: %s (%s)\n", __FILE__,    \
                         __LINE__, #expr, cudaGetErrorString(_err));        \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

template <typename T>
T* device_from_host(const std::vector<T>& host) {
    T* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return device;
}

void compare(const char* label, int rows, int cols,
             std::uint8_t* d_packed_ref, std::uint8_t* d_scale_ref,
             std::uint8_t* d_packed_f, std::uint8_t* d_scale_f) {
    const std::size_t packed_n = static_cast<std::size_t>(rows) * cols / 2;
    const std::size_t scale_n = static_cast<std::size_t>(rows) * cols / 32;
    std::vector<std::uint8_t> packed_ref(packed_n), packed_f(packed_n);
    std::vector<std::uint8_t> scale_ref(scale_n), scale_f(scale_n);
    CUDA_CHECK(cudaMemcpy(packed_ref.data(), d_packed_ref, packed_n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(packed_f.data(), d_packed_f, packed_n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scale_ref.data(), d_scale_ref, scale_n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scale_f.data(), d_scale_f, scale_n, cudaMemcpyDeviceToHost));
    std::size_t pm = 0, sm = 0;
    for (std::size_t i = 0; i < packed_n; ++i) if (packed_ref[i] != packed_f[i]) ++pm;
    for (std::size_t i = 0; i < scale_n; ++i)  if (scale_ref[i] != scale_f[i]) ++sm;
    CHECK(pm == 0);
    CHECK(sm == 0);
    std::printf("[transcode] %-18s rows=%d cols=%d: packed_mismatch=%zu/%zu "
                "scale_mismatch=%zu/%zu\n",
                label, rows, cols, pm, packed_n, sm, scale_n);
}

// FP8E4m3PerGroup -> Mxfp4
void run_fp8(int rows, int cols, int gs) {
    const int scale_rows = (rows + gs - 1) / gs;
    const int scale_cols = (cols + gs - 1) / gs;
    std::mt19937 rng(0xC0FFEE ^ (rows * 131 + cols * 17 + gs));
    std::vector<std::uint8_t> fp8(static_cast<std::size_t>(rows) * cols);
    for (auto& b : fp8) {
        std::uint8_t v = static_cast<std::uint8_t>(rng() & 0xFF);
        if ((v & 0x7F) == 0x7F) v &= 0x7E;  // avoid E4M3 NaN
        b = v;
    }
    std::vector<float> scales(static_cast<std::size_t>(scale_rows) * scale_cols);
    std::uniform_real_distribution<float> sd(0.25f, 3.0f);
    for (auto& s : scales) s = sd(rng);

    std::uint8_t* d_fp8 = device_from_host(fp8);
    float* d_scale = device_from_host(scales);
    const std::size_t n = static_cast<std::size_t>(rows) * cols;

    void* d_bf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bf16, n * 2));
    K::launch_dequant_fp8_e4m3_to_bf16_per_group(d_fp8, d_bf16, d_scale, rows, cols, gs, 0);
    std::uint8_t* d_pr = nullptr; std::uint8_t* d_sr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pr, n / 2)); CUDA_CHECK(cudaMalloc(&d_sr, n / 32));
    K::quantize_bf16_to_mxfp4_e2m1_per_block(d_bf16, d_pr, d_sr, rows, cols, 0);

    std::uint8_t* d_pf = nullptr; std::uint8_t* d_sf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pf, n / 2)); CUDA_CHECK(cudaMalloc(&d_sf, n / 32));
    K::TranscodeParams p;
    p.src = d_fp8; p.src_scale = d_scale; p.src_group_size = gs;
    p.dst_packed = d_pf; p.dst_scale = d_sf; p.rows = rows; p.cols = cols;
    K::launch_transcode(K::TranscodeSource::Fp8E4m3PerGroup,
                        K::TranscodeTarget::Mxfp4E2m1E8m0, p, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    compare("fp8->mxfp4", rows, cols, d_pr, d_sr, d_pf, d_sf);
    for (void* q : {(void*)d_fp8, (void*)d_scale, (void*)d_bf16,
                    (void*)d_pr, (void*)d_sr, (void*)d_pf, (void*)d_sf})
        CUDA_CHECK(cudaFree(q));
}

// Bf16 -> Mxfp4 (decode is a no-op; validates a second registered source).
void run_bf16(int rows, int cols) {
    std::mt19937 rng(0xBF16 ^ (rows * 7 + cols));
    std::vector<std::uint16_t> bf16(static_cast<std::size_t>(rows) * cols);
    for (auto& h : bf16) {
        std::uint16_t v = static_cast<std::uint16_t>(rng() & 0xFFFF);
        if (((v >> 7) & 0xFF) == 0xFF) v &= 0xBFFF;  // avoid bf16 inf/NaN exponent
        h = v;
    }
    std::uint16_t* d_bf16 = device_from_host(bf16);
    const std::size_t n = static_cast<std::size_t>(rows) * cols;

    std::uint8_t* d_pr = nullptr; std::uint8_t* d_sr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pr, n / 2)); CUDA_CHECK(cudaMalloc(&d_sr, n / 32));
    K::quantize_bf16_to_mxfp4_e2m1_per_block(d_bf16, d_pr, d_sr, rows, cols, 0);

    std::uint8_t* d_pf = nullptr; std::uint8_t* d_sf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pf, n / 2)); CUDA_CHECK(cudaMalloc(&d_sf, n / 32));
    K::TranscodeParams p;
    p.src = d_bf16; p.dst_packed = d_pf; p.dst_scale = d_sf; p.rows = rows; p.cols = cols;
    K::launch_transcode(K::TranscodeSource::Bf16,
                        K::TranscodeTarget::Mxfp4E2m1E8m0, p, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    compare("bf16->mxfp4", rows, cols, d_pr, d_sr, d_pf, d_sf);
    for (void* q : {(void*)d_bf16, (void*)d_pr, (void*)d_sr, (void*)d_pf, (void*)d_sf})
        CUDA_CHECK(cudaFree(q));
}

}  // namespace

int main() {
    run_fp8(64, 256, 128);
    run_fp8(128, 512, 128);
    run_fp8(257, 256, 128);  // non-multiple rows
    run_bf16(64, 256);
    run_bf16(33, 128);
    if (g_failures == 0) {
        std::printf("test_transcode_fused: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "test_transcode_fused: %d FAILURE(S)\n", g_failures);
    return 1;
}
