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

// Paged decode/prefill attention over an NHD KV cache (matches the Metal
// paged_attention kernel + kvattn). Online flash-softmax; row attends [0, q_pos].
struct AttnConfig {
    int N = 0, n_q_heads = 0, n_kv_heads = 0, d = 0, page_size = 0;
    float scale = 1.0f;
    std::vector<std::int32_t> position_ids;
    std::vector<std::int32_t> req_of_token;
    std::vector<std::uint32_t> kv_page_indices;
    std::vector<std::uint32_t> kv_page_indptr;
    int gqa() const { return n_q_heads / n_kv_heads; }
};

inline std::vector<float> paged_attention(const AttnConfig& c, const std::vector<float>& queries,
                                          const std::vector<float>& k_pages,
                                          const std::vector<float>& v_pages) {
    std::vector<float> out(static_cast<std::size_t>(c.N) * c.n_q_heads * c.d, 0.0f);
    for (int row = 0; row < c.N; ++row) {
        const int r = c.req_of_token[row];
        const int q_pos = c.position_ids[row];
        const int page_base = static_cast<int>(c.kv_page_indptr[r]);
        for (int qh = 0; qh < c.n_q_heads; ++qh) {
            const int kv_head = qh / c.gqa();
            const std::size_t qbase = (static_cast<std::size_t>(row) * c.n_q_heads + qh) * c.d;
            float m = -3.0e38f, l = 0.0f;
            std::vector<float> acc(c.d, 0.0f);
            for (int kp = 0; kp <= q_pos; ++kp) {
                const int page = static_cast<int>(c.kv_page_indices[page_base + kp / c.page_size]);
                const std::size_t slot =
                    static_cast<std::size_t>(page) * c.page_size + (kp % c.page_size);
                const std::size_t kb = (slot * c.n_kv_heads + kv_head) * c.d;
                float score = 0.0f;
                for (int i = 0; i < c.d; ++i) score += queries[qbase + i] * k_pages[kb + i];
                score *= c.scale;
                const float nm = m > score ? m : score;
                const float factor = std::exp(m - nm);
                const float e = std::exp(score - nm);
                l = l * factor + e;
                for (int i = 0; i < c.d; ++i) acc[i] = acc[i] * factor + e * v_pages[kb + i];
                m = nm;
            }
            for (int i = 0; i < c.d; ++i) out[qbase + i] = l == 0.0f ? 0.0f : acc[i] / l;
        }
    }
    return out;
}

// ── full Qwen3 decoder-layer forward (prefill: N tokens, single request) ─────
struct LayerWeights {
    std::vector<float> input_ln;   // [hidden]
    std::vector<float> wq, wk, wv; // [q_dim,hidden] / [kv_dim,hidden] / [kv_dim,hidden]
    std::vector<float> q_norm, k_norm;  // [head_dim]
    std::vector<float> wo;         // [hidden, q_dim]
    std::vector<float> post_ln;    // [hidden]
    std::vector<float> wgate, wup; // [intermediate, hidden]
    std::vector<float> wdown;      // [hidden, intermediate]
};

struct LayerDims {
    int N, hidden, n_q_heads, n_kv_heads, head_dim, intermediate;
    float rms_eps, rope_theta, attn_scale;
    int q_dim() const { return n_q_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
};

inline std::vector<float> decoder_layer(const std::vector<float>& x,
                                        const std::vector<std::int32_t>& positions,
                                        const LayerWeights& w, const LayerDims& D) {
    const int N = D.N, H = D.hidden, hd = D.head_dim;
    // 1. input RMSNorm
    auto nx = rmsnorm(x, w.input_ln, N, H, D.rms_eps);
    // 2. QKV projections
    auto q = matmul_xwt(nx, w.wq, N, D.q_dim(), H);   // [N, q_dim]
    auto k = matmul_xwt(nx, w.wk, N, D.kv_dim(), H);
    auto v = matmul_xwt(nx, w.wv, N, D.kv_dim(), H);
    // 3. per-head QK-norm
    q = rmsnorm(q, w.q_norm, N * D.n_q_heads, hd, D.rms_eps);
    k = rmsnorm(k, w.k_norm, N * D.n_kv_heads, hd, D.rms_eps);
    // 4. RoPE
    rope(q, positions, N, D.n_q_heads, hd, D.rope_theta);
    rope(k, positions, N, D.n_kv_heads, hd, D.rope_theta);
    // 5+6. attention (single request, page_size=N holds all tokens; k/v = pages)
    AttnConfig c;
    c.N = N; c.n_q_heads = D.n_q_heads; c.n_kv_heads = D.n_kv_heads; c.d = hd;
    c.page_size = N; c.scale = D.attn_scale;
    c.position_ids = positions;
    c.req_of_token.assign(N, 0);
    c.kv_page_indices = {0};
    c.kv_page_indptr = {0, 1};
    auto attn = paged_attention(c, q, k, v);   // [N, q_dim]
    // 7. O projection + residual
    auto o = matmul_xwt(attn, w.wo, N, H, D.q_dim());   // [N, hidden]
    std::vector<float> y(static_cast<std::size_t>(N) * H);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = x[i] + o[i];
    // 8. post-attention RMSNorm
    auto ny = rmsnorm(y, w.post_ln, N, H, D.rms_eps);
    // 9. gate / up
    auto gate = matmul_xwt(ny, w.wgate, N, D.intermediate, H);
    auto up = matmul_xwt(ny, w.wup, N, D.intermediate, H);
    // 10. SwiGLU
    auto g = swiglu(gate, up);   // [N, intermediate]
    // 11. down projection + residual
    auto down = matmul_xwt(g, w.wdown, N, H, D.intermediate);
    std::vector<float> out(y.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = y[i] + down[i];
    return out;
}

// Token embedding gather: out[n,:] = embed[token_ids[n], :]. embed is [vocab,hidden].
inline std::vector<float> embedding(const std::vector<float>& embed,
                                    const std::vector<std::int32_t>& token_ids, int hidden) {
    std::vector<float> out(token_ids.size() * hidden);
    for (std::size_t n = 0; n < token_ids.size(); ++n) {
        const std::size_t s = static_cast<std::size_t>(token_ids[n]) * hidden;
        for (int j = 0; j < hidden; ++j) out[n * hidden + j] = embed[s + j];
    }
    return out;
}

}  // namespace qwen3::ref
