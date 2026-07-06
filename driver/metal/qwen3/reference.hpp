#pragma once
//
// CPU f32 references for the Qwen3-0.6B decoder-layer Metal parity kernels —
// each a port of the corresponding CUDA kernel formula:
//   RMSNorm   driver/cuda/src/kernels/rmsnorm.cu   (y = x·rsqrt(mean(x²)+eps)·w)
//   RoPE      driver/cuda/src/kernels/rope.cu      (Qwen half-rotation)
//   SwiGLU    driver/cuda/src/kernels/swiglu.cu    (silu(g)·u)
//   GEMM      x @ Wᵀ (projections), f32 accumulation
// Numeric ops are validated within tolerance (f32 epsilon), matching the same
// accumulation order — exp/sin/cos are not bit-exact GPU-vs-host.
//
// Pure C++17, no Metal dependency.

#include <cmath>
#include <cstdint>
#include <vector>

namespace qwen3::ref {

// y[r,:] = x[r,:] · rsqrt(mean_i x[r,i]² + eps) · w   (per-row over `dim`).
inline std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& w,
                                  int rows, int dim, float eps) {
    std::vector<float> y(static_cast<std::size_t>(rows) * dim);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;  // fp32 sum in CUDA; use double-free f32 to match: keep f32
        float facc = 0.0f;
        for (int i = 0; i < dim; ++i) { float v = x[r * dim + i]; facc += v * v; }
        (void)acc;
        float inv_rms = 1.0f / std::sqrt(facc / static_cast<float>(dim) + eps);
        for (int i = 0; i < dim; ++i) y[r * dim + i] = x[r * dim + i] * inv_rms * w[i];
    }
    return y;
}

// Qwen half-rotation RoPE applied in place over [rows, n_heads, head_dim].
// pair i with i+half; freq = theta^(-2i/head_dim); ang = pos·freq.
inline void rope(std::vector<float>& h, const std::vector<std::int32_t>& positions,
                 int rows, int n_heads, int head_dim, float theta) {
    const int half = head_dim / 2;
    for (int r = 0; r < rows; ++r) {
        const int pos = positions[r];
        for (int hh = 0; hh < n_heads; ++hh) {
            float* row = &h[(static_cast<std::size_t>(r) * n_heads + hh) * head_dim];
            for (int i = 0; i < half; ++i) {
                float freq = std::pow(theta, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                float ang = static_cast<float>(pos) * freq;
                float c = std::cos(ang), s = std::sin(ang);
                float a = row[i], b = row[i + half];
                row[i] = a * c - b * s;
                row[i + half] = b * c + a * s;
            }
        }
    }
}

// SwiGLU: y = silu(gate) · up, silu(g) = g / (1 + e^-g).
inline std::vector<float> swiglu(const std::vector<float>& gate, const std::vector<float>& up) {
    std::vector<float> y(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        float g = gate[i];
        float silu = g / (1.0f + std::exp(-g));
        y[i] = silu * up[i];
    }
    return y;
}

// GEMM: y[m, n] = sum_k x[m, k] · W[n, k]   (W row-major [N, K], i.e. y = x·Wᵀ),
// f32 sequential-k accumulation (matches a straightforward CUDA dot).
inline std::vector<float> matmul_xwt(const std::vector<float>& x, const std::vector<float>& w,
                                     int M, int N, int K) {
    std::vector<float> y(static_cast<std::size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            const float* xr = &x[static_cast<std::size_t>(m) * K];
            const float* wr = &w[static_cast<std::size_t>(n) * K];
            for (int k = 0; k < K; ++k) acc += xr[k] * wr[k];
            y[static_cast<std::size_t>(m) * N + n] = acc;
        }
    return y;
}

}  // namespace qwen3::ref
