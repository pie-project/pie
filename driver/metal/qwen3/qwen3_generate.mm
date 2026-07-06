// qwen3_generate — Metal end-to-end greedy AUTOREGRESSIVE generation for
// Qwen3-0.6B. Loads the real safetensors, runs a prompt through the full 28-layer
// Metal forward, then decodes N tokens fully on-device (Metal forward → Metal
// argmax → append the new token's K/V to a growing per-layer KV cache → repeat).
// This exercises the multi-step KV-decode path (KV growth across steps) that a
// single forward doesn't, and exports the generated token-id sequence for the
// definitive CUDA cross-check (matching greedy sequences = tight parity).
//
// Usage: qwen3_generate <model.safetensors> [kernels_dir] [n_gen] [out.txt]

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>

#include "arch.hpp"
#include "chain.hpp"
#include "metal_harness.hpp"
#include "reference.hpp"
#include "weights.hpp"

#ifndef QWEN3_KERNELS_DIR
#define QWEN3_KERNELS_DIR "."
#endif

using namespace ptir_metal;
namespace A = qwen3::arch;

namespace {

// Metal on-device greedy argmax over logits[rows, vocab] -> token[rows].
std::vector<std::int32_t> metal_argmax(MetalHarness& h, const std::vector<float>& logits,
                                       int rows, int vocab) {
    std::vector<std::int32_t> out(rows, 0);
    int v = vocab; std::uint32_t r = (std::uint32_t)rows;
    std::vector<Arg> a = {Arg::in(logits.data(), logits.size() * 4), Arg::out(out.data(), rows * 4),
                          Arg::in(&v, 4), Arg::in(&r, 4)};
    h.run("argmax_row", a, rows);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.safetensors> [kernels_dir] [n_gen] [out.txt]\n", argv[0]); return 2; }
    std::string model = argv[1];
    std::string kernels_dir = argc > 2 ? argv[2] : QWEN3_KERNELS_DIR;
    int n_gen = argc > 3 ? std::atoi(argv[3]) : 20;
    std::string out_path = argc > 4 ? argv[4] : "qwen3_generate.txt";

    qwen3::SafeTensors st;
    if (!st.open(model)) { std::fprintf(stderr, "cannot open %s\n", model.c_str()); return 2; }
    std::printf("loaded safetensors: %s (%.2f GiB)\n", model.c_str(), st.file_size / (1024.0 * 1024 * 1024));

    qwen3::ref::LayerDims D;
    D.hidden = A::HIDDEN; D.n_q_heads = A::N_Q_HEADS; D.n_kv_heads = A::N_KV_HEADS;
    D.head_dim = A::HEAD_DIM; D.intermediate = A::INTERMEDIATE;
    D.rms_eps = A::RMS_EPS; D.rope_theta = A::ROPE_THETA;
    D.attn_scale = 1.0f / std::sqrt((float)A::HEAD_DIM);
    const int H = D.hidden, hd = D.head_dim, qd = D.q_dim(), kd = D.kv_dim(), I = D.intermediate;

    std::printf("loading weights...\n");
    std::vector<float> embed, lm_head, final_norm;
    if (!st.load_f32("model.embed_tokens.weight", embed) || !st.load_f32("model.norm.weight", final_norm)) return 2;
    if (!st.load_f32("lm_head.weight", lm_head)) lm_head = embed;
    std::vector<qwen3::ref::LayerWeights> W(A::N_LAYERS);
    for (int l = 0; l < A::N_LAYERS; ++l) if (!qwen3::load_layer(st, l, W[l])) return 2;

    MetalHarness h;
    if (!h.ok()) { std::fprintf(stderr, "no Metal device\n"); return 2; }
    if (!h.load_library(kernels_dir + "/layer.metal")) { std::fprintf(stderr, "%s\n", h.error().c_str()); return 2; }
    qwen3::Chain ch{h};

    // Growing per-layer KV cache (NHD: [pos, n_kv_heads, head_dim]).
    std::vector<std::vector<float>> kcache(A::N_LAYERS), vcache(A::N_LAYERS);
    int cur_len = 0;

    // Process a block of n_cur tokens at absolute `positions`; append K/V to the
    // cache and attend each query over [0, position]. Returns hidden [n_cur,H].
    auto process_block = [&](std::vector<float> x, const std::vector<std::int32_t>& positions) {
        int n_cur = (int)positions.size();
        int total = cur_len + n_cur;
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
            // Append this block's K/V (contiguous NHD rows = positions cur_len..).
            kcache[l].insert(kcache[l].end(), k.begin(), k.end());
            vcache[l].insert(vcache[l].end(), v.begin(), v.end());
            qwen3::ref::AttnConfig c;
            c.N = n_cur; c.n_q_heads = D.n_q_heads; c.n_kv_heads = D.n_kv_heads; c.d = hd;
            c.page_size = total; c.scale = D.attn_scale; c.position_ids = positions;
            c.req_of_token.assign(n_cur, 0); c.kv_page_indices = {0}; c.kv_page_indptr = {0, 1};
            auto attn = ch.attention(c, q, kcache[l], vcache[l]);  // [n_cur, q_dim]
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
        cur_len = total;
        return x;  // [n_cur, H]
    };

    // Next-token greedy: logits = lm_head(final_norm(last_hidden)); Metal argmax.
    auto next_token = [&](const std::vector<float>& hidden, int n_cur) {
        std::vector<float> last(hidden.begin() + (std::size_t)(n_cur - 1) * H, hidden.end());
        auto ln = ch.rmsnorm(last, final_norm, 1, H, D.rms_eps);
        auto logits = ch.matmul(ln, lm_head, 1, A::VOCAB, H);
        return metal_argmax(h, logits, 1, A::VOCAB)[0];
    };

    // Real prompt (raw token ids; the export records them so CUDA uses the same).
    std::vector<std::int32_t> prompt = {9707, 3838, 374, 279, 6722, 315, 9625, 30};
    std::printf("prompt (%zu tokens): ", prompt.size());
    for (int t : prompt) std::printf("%d ", t);
    std::printf("\ngenerating %d tokens (greedy, on-device)...\n", n_gen);

    // Prefill.
    std::vector<std::int32_t> pos0;
    for (std::size_t i = 0; i < prompt.size(); ++i) pos0.push_back((int)i);
    auto hx = ch.embedding(embed, prompt, (int)prompt.size(), H);
    auto hidden = process_block(hx, pos0);
    std::int32_t tok = next_token(hidden, (int)prompt.size());

    std::vector<std::int32_t> generated;
    generated.push_back(tok);
    // Decode.
    for (int s = 1; s < n_gen; ++s) {
        std::vector<std::int32_t> one = {tok};
        auto ex = ch.embedding(embed, one, 1, H);
        std::vector<std::int32_t> posd = {cur_len};
        auto hd1 = process_block(ex, posd);
        tok = next_token(hd1, 1);
        generated.push_back(tok);
    }
    if (!ch.ok) { std::fprintf(stderr, "metal generation failed: %s\n", h.error().c_str()); return 2; }

    std::printf("generated: ");
    for (int t : generated) std::printf("%d ", t);
    std::printf("\n");

    // Export the sequence for the CUDA cross-check.
    std::ofstream g(out_path);
    g << "# Qwen3-0.6B Metal greedy autoregressive generation — CUDA cross-check\n";
    g << "model: Qwen/Qwen3-0.6B model.safetensors (BF16, HF [out,in])\n";
    g << "backend: Metal (Apple " << h.device_name() << "), f32 compute + MPS GEMM, greedy argmax on-device\n";
    g << "prompt:";
    for (int t : prompt) g << " " << t;
    g << "\nn_gen: " << n_gen << "\n";
    g << "generated:";
    for (int t : generated) g << " " << t;
    g << "\nfull_sequence:";
    for (int t : prompt) g << " " << t;
    for (int t : generated) g << " " << t;
    g << "\nnote: greedy (argmax) — per-step divergence compounds, so matching the full "
         "generated sequence on CUDA (same prompt+weights) = tight cross-backend parity.\n";
    g.close();
    std::printf("sequence written: %s\n", out_path.c_str());
    std::printf("QWEN3_GENERATE_OK\n");
    return 0;
}
