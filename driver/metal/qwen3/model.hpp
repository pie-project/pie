#pragma once
//
// Model — a reusable Qwen3-0.6B Metal forward with an externally-owned, growing
// KV cache, so higher-level algorithm loops (greedy / beam search / speculative
// decoding) are pure composition over the certified primitives. The heavy
// compute (projections, attention, MLP) runs on Metal via Chain; the KV cache is
// a plain host-owned NHD buffer per layer that each beam / draft owns
// independently.

#include <cmath>
#include <cstdint>
#include <vector>

#include "arch.hpp"
#include "chain.hpp"
#include "metal_harness.hpp"
#include "reference.hpp"

namespace qwen3 {

namespace A = arch;

// Per-layer growing KV cache (NHD: [pos, n_kv_heads, head_dim] per layer).
struct KVCache {
    std::vector<std::vector<float>> k, v;
    int len = 0;
    KVCache() : k(A::N_LAYERS), v(A::N_LAYERS) {}
};

struct Model {
    Chain& ch;
    ptir_metal::MetalHarness& h;
    const std::vector<float>& embed;
    const std::vector<float>& lm_head;
    const std::vector<float>& final_norm;
    const std::vector<ref::LayerWeights>& W;
    ref::LayerDims D;

    Model(Chain& c, ptir_metal::MetalHarness& hh, const std::vector<float>& e,
          const std::vector<float>& lh, const std::vector<float>& fn,
          const std::vector<ref::LayerWeights>& w)
        : ch(c), h(hh), embed(e), lm_head(lh), final_norm(fn), W(w) {
        D.hidden = A::HIDDEN; D.n_q_heads = A::N_Q_HEADS; D.n_kv_heads = A::N_KV_HEADS;
        D.head_dim = A::HEAD_DIM; D.intermediate = A::INTERMEDIATE;
        D.rms_eps = A::RMS_EPS; D.rope_theta = A::ROPE_THETA;
        D.attn_scale = 1.0f / std::sqrt((float)A::HEAD_DIM);
    }

    std::vector<float> embed_tokens(const std::vector<std::int32_t>& toks) {
        return ch.embedding(embed, toks, (int)toks.size(), D.hidden);
    }

    // Forward a block of tokens at absolute `positions` over `kv` (appends this
    // block's K/V, attends each query over [0, position]). Returns hidden[n_cur,H].
    std::vector<float> forward_block(std::vector<float> x,
                                     const std::vector<std::int32_t>& positions, KVCache& kv) {
        const int H = D.hidden, hd = D.head_dim, qd = D.q_dim(), kd = D.kv_dim(), I = D.intermediate;
        int n_cur = (int)positions.size();
        int total = kv.len + n_cur;
        for (int l = 0; l < A::N_LAYERS; ++l) {
            const auto& w = W[l];
            auto nx = ch.rmsnorm(x, w.input_ln, n_cur, H, D.rms_eps);
            auto q = ch.matmul(nx, w.wq, n_cur, qd, H);
            auto k = ch.matmul(nx, w.wk, n_cur, kd, H);
            auto v = ch.matmul(nx, w.wv, n_cur, kd, H);
            q = ch.rmsnorm(q, w.q_norm, n_cur * D.n_q_heads, hd, D.rms_eps);
            k = ch.rmsnorm(k, w.k_norm, n_cur * D.n_kv_heads, hd, D.rms_eps);
            ch.rope_ip(q, positions, n_cur, D.n_q_heads, hd, D.rope_theta);
            ch.rope_ip(k, positions, n_cur, D.n_kv_heads, hd, D.rope_theta);
            kv.k[l].insert(kv.k[l].end(), k.begin(), k.end());
            kv.v[l].insert(kv.v[l].end(), v.begin(), v.end());
            ref::AttnConfig c;
            c.N = n_cur; c.n_q_heads = D.n_q_heads; c.n_kv_heads = D.n_kv_heads; c.d = hd;
            c.page_size = total; c.scale = D.attn_scale; c.position_ids = positions;
            c.req_of_token.assign(n_cur, 0); c.kv_page_indices = {0}; c.kv_page_indptr = {0, 1};
            auto attn = ch.attention(c, q, kv.k[l], kv.v[l]);
            auto o = ch.matmul(attn, w.wo, n_cur, H, qd);
            std::vector<float> y = x;
            ch.add_ip(y, o);
            auto ny = ch.rmsnorm(y, w.post_ln, n_cur, H, D.rms_eps);
            auto gate = ch.matmul(ny, w.wgate, n_cur, I, H);
            auto up = ch.matmul(ny, w.wup, n_cur, I, H);
            auto g = ch.swiglu(gate, up);
            auto down = ch.matmul(g, w.wdown, n_cur, H, I);
            x = y;
            ch.add_ip(x, down);
        }
        kv.len = total;
        return x;
    }

    // Logits for row `row` of a hidden[n_cur,H] block.
    std::vector<float> logits_at(const std::vector<float>& hidden, int row) {
        const int H = D.hidden;
        std::vector<float> h_row(hidden.begin() + (std::size_t)row * H,
                                 hidden.begin() + (std::size_t)(row + 1) * H);
        auto ln = ch.rmsnorm(h_row, final_norm, 1, H, D.rms_eps);
        return ch.matmul(ln, lm_head, 1, A::VOCAB, H);  // [vocab]
    }
    // On-device greedy argmax over a logits row.
    std::int32_t argmax(const std::vector<float>& logits) {
        std::vector<std::int32_t> out(1, 0);
        int v = A::VOCAB; std::uint32_t r = 1;
        std::vector<ptir_metal::Arg> a = {ptir_metal::Arg::in(logits.data(), logits.size() * 4),
                                          ptir_metal::Arg::out(out.data(), 4),
                                          ptir_metal::Arg::in(&v, 4), ptir_metal::Arg::in(&r, 4)};
        h.run("argmax_row", a, 1);
        return out[0];
    }
};

// Numerically-stable log-softmax over a single logits row (the beam score op).
inline std::vector<float> log_softmax(const std::vector<float>& logits) {
    float m = -3.0e38f;
    for (float x : logits) m = x > m ? x : m;
    double sum = 0.0;
    for (float x : logits) sum += std::exp((double)x - m);
    float lse = m + (float)std::log(sum);
    std::vector<float> out(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) out[i] = logits[i] - lse;
    return out;
}

}  // namespace qwen3
