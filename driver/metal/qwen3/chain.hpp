#pragma once
//
// Chain — host-chained Metal kernel wrappers that assemble the Qwen3-0.6B
// decoder-layer / full forward from the individual layer.metal kernels. Shared
// by the parity test (qwen3_test) and the real-weight forward driver
// (qwen3_forward). Each method uploads inputs, dispatches one kernel, and copies
// the output back — parity-first (not perf); an MPS GEMM swap-in is the perf
// follow-on.

#include <cstdint>
#include <string>
#include <vector>

#include "metal_harness.hpp"
#include "reference.hpp"

namespace qwen3 {

using ptir_metal::Arg;
using ptir_metal::MetalHarness;

struct Chain {
    MetalHarness& h;
    bool ok = true;
    bool use_mps = true;  // MPS GEMM by default (perf); false = sequential kernel

    std::vector<float> matmul(const std::vector<float>& x, const std::vector<float>& w,
                              int M, int N, int K) {
        std::vector<float> y((std::size_t)M * N, 0.0f);
        if (use_mps) {
            ok = ok && h.mps_gemm(x.data(), w.data(), y.data(), M, N, K);
            return y;
        }
        int m = M, n = N, k = K; std::uint32_t total = (std::uint32_t)(M * N);
        std::vector<Arg> a = {Arg::in(x.data(), x.size() * 4), Arg::in(w.data(), w.size() * 4),
                              Arg::out(y.data(), y.size() * 4), Arg::in(&m, 4), Arg::in(&n, 4),
                              Arg::in(&k, 4), Arg::in(&total, 4)};
        ok = ok && h.run("matmul_xwt", a, total);
        return y;
    }
    std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& w,
                               int rows, int dim, float eps) {
        std::vector<float> y((std::size_t)rows * dim, 0.0f);
        int r = rows, d = dim;
        std::vector<Arg> a = {Arg::in(x.data(), x.size() * 4), Arg::in(w.data(), w.size() * 4),
                              Arg::out(y.data(), y.size() * 4), Arg::in(&r, 4), Arg::in(&d, 4),
                              Arg::in(&eps, 4)};
        ok = ok && h.run("rmsnorm", a, rows);
        return y;
    }
    void rope_ip(std::vector<float>& io, const std::vector<std::int32_t>& pos, int rows,
                 int n_heads, int hd, float theta) {
        int nh = n_heads, d = hd; float th = theta;
        std::uint32_t total = (std::uint32_t)(rows * n_heads * (hd / 2));
        std::vector<Arg> a = {Arg::inout(io.data(), io.size() * 4), Arg::in(pos.data(), pos.size() * 4),
                              Arg::in(&nh, 4), Arg::in(&d, 4), Arg::in(&th, 4), Arg::in(&total, 4)};
        ok = ok && h.run("rope_qwen", a, total);
    }
    std::vector<float> attention(const ref::AttnConfig& c, const std::vector<float>& q,
                                 const std::vector<float>& k, const std::vector<float>& v) {
        std::vector<float> out((std::size_t)c.N * c.n_q_heads * c.d, 0.0f);
        int nqh = c.n_q_heads, nkv = c.n_kv_heads, d = c.d, ps = c.page_size, gqa = c.gqa();
        float scale = c.scale; std::uint32_t total = (std::uint32_t)(c.N * c.n_q_heads);
        std::vector<Arg> a = {Arg::in(q.data(), q.size() * 4), Arg::in(k.data(), k.size() * 4),
                              Arg::in(v.data(), v.size() * 4), Arg::out(out.data(), out.size() * 4),
                              Arg::in(c.position_ids.data(), c.position_ids.size() * 4),
                              Arg::in(c.req_of_token.data(), c.req_of_token.size() * 4),
                              Arg::in(c.kv_page_indices.data(), c.kv_page_indices.size() * 4),
                              Arg::in(c.kv_page_indptr.data(), c.kv_page_indptr.size() * 4),
                              Arg::in(&nqh, 4), Arg::in(&nkv, 4), Arg::in(&d, 4), Arg::in(&ps, 4),
                              Arg::in(&gqa, 4), Arg::in(&scale, 4), Arg::in(&total, 4)};
        ok = ok && h.run("paged_attention", a, total);
        return out;
    }
    std::vector<float> swiglu(const std::vector<float>& gate, const std::vector<float>& up) {
        std::vector<float> y(gate.size(), 0.0f);
        std::uint32_t n = (std::uint32_t)gate.size();
        std::vector<Arg> a = {Arg::in(gate.data(), gate.size() * 4), Arg::in(up.data(), up.size() * 4),
                              Arg::out(y.data(), y.size() * 4), Arg::in(&n, 4)};
        ok = ok && h.run("swiglu", a, n);
        return y;
    }
    void add_ip(std::vector<float>& a, const std::vector<float>& b) {
        std::uint32_t n = (std::uint32_t)a.size();
        std::vector<Arg> ar = {Arg::inout(a.data(), a.size() * 4), Arg::in(b.data(), b.size() * 4),
                               Arg::in(&n, 4)};
        ok = ok && h.run("add_inplace", ar, n);
    }
    std::vector<float> embedding(const std::vector<float>& embed,
                                 const std::vector<std::int32_t>& tokens, int N, int hidden) {
        std::vector<float> out((std::size_t)N * hidden, 0.0f);
        int hd = hidden; std::uint32_t total = (std::uint32_t)(N * hidden);
        std::vector<Arg> a = {Arg::in(embed.data(), embed.size() * 4),
                              Arg::in(tokens.data(), tokens.size() * 4),
                              Arg::out(out.data(), out.size() * 4), Arg::in(&hd, 4), Arg::in(&total, 4)};
        ok = ok && h.run("embedding", a, total);
        return out;
    }
    // Full Metal decoder-layer forward (same op sequence as ref::decoder_layer).
    std::vector<float> layer(const std::vector<float>& x, const std::vector<std::int32_t>& positions,
                             const ref::LayerWeights& w, const ref::LayerDims& D) {
        const int N = D.N, H = D.hidden, hd = D.head_dim, qd = D.q_dim(), kd = D.kv_dim(),
                  I = D.intermediate;
        auto nx = rmsnorm(x, w.input_ln, N, H, D.rms_eps);
        auto q = matmul(nx, w.wq, N, qd, H);
        auto k = matmul(nx, w.wk, N, kd, H);
        auto v = matmul(nx, w.wv, N, kd, H);
        q = rmsnorm(q, w.q_norm, N * D.n_q_heads, hd, D.rms_eps);
        k = rmsnorm(k, w.k_norm, N * D.n_kv_heads, hd, D.rms_eps);
        rope_ip(q, positions, N, D.n_q_heads, hd, D.rope_theta);
        rope_ip(k, positions, N, D.n_kv_heads, hd, D.rope_theta);
        ref::AttnConfig c;
        c.N = N; c.n_q_heads = D.n_q_heads; c.n_kv_heads = D.n_kv_heads; c.d = hd;
        c.page_size = N; c.scale = D.attn_scale; c.position_ids = positions;
        c.req_of_token.assign(N, 0); c.kv_page_indices = {0}; c.kv_page_indptr = {0, 1};
        auto attn = attention(c, q, k, v);
        auto o = matmul(attn, w.wo, N, H, qd);
        std::vector<float> y = x;
        add_ip(y, o);
        auto ny = rmsnorm(y, w.post_ln, N, H, D.rms_eps);
        auto gate = matmul(ny, w.wgate, N, I, H);
        auto up = matmul(ny, w.wup, N, I, H);
        auto g = swiglu(gate, up);
        auto down = matmul(g, w.wdown, N, H, I);
        std::vector<float> out = y;
        add_ip(out, down);
        return out;
    }
};

}  // namespace qwen3
