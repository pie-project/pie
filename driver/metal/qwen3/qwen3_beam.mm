// qwen3_beam — Metal end-to-end BEAM SEARCH for Qwen3-0.6B. Composes the
// certified pieces: Metal forward + growing per-beam KV cache + the beam epilogue
// (log_softmax scores → K×K candidate top-k → reorder beams by parent). Proves
// "Metal runs beam search," not just "beam_epilogue golden replays."
//
// Usage: qwen3_beam <model.safetensors> [kernels_dir] [beam_width] [steps]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "arch.hpp"
#include "chain.hpp"
#include "metal_harness.hpp"
#include "model.hpp"
#include "reference.hpp"
#include "weights.hpp"

#ifndef QWEN3_KERNELS_DIR
#define QWEN3_KERNELS_DIR "."
#endif

using namespace ptir_metal;
namespace A = qwen3::arch;

namespace {
struct Beam {
    std::vector<std::int32_t> gen;  // generated tokens (after the prompt)
    double score = 0.0;             // cumulative log-prob
    qwen3::KVCache kv;
};

// Top-k (value, index) of a logprob row, descending.
std::vector<std::pair<float, int>> topk(const std::vector<float>& lp, int k) {
    std::vector<int> idx(lp.size());
    for (std::size_t i = 0; i < lp.size(); ++i) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return lp[a] > lp[b]; });
    std::vector<std::pair<float, int>> out;
    for (int i = 0; i < k; ++i) out.push_back({lp[idx[i]], idx[i]});
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.safetensors> [kernels] [K] [steps]\n", argv[0]); return 2; }
    std::string model = argv[1];
    std::string kdir = argc > 2 ? argv[2] : QWEN3_KERNELS_DIR;
    int K = argc > 3 ? std::atoi(argv[3]) : 3;
    int steps = argc > 4 ? std::atoi(argv[4]) : 6;

    qwen3::SafeTensors st;
    if (!st.open(model)) { std::fprintf(stderr, "cannot open %s\n", model.c_str()); return 2; }
    std::vector<float> embed, lm_head, final_norm;
    if (!st.load_f32("model.embed_tokens.weight", embed) || !st.load_f32("model.norm.weight", final_norm)) return 2;
    if (!st.load_f32("lm_head.weight", lm_head)) lm_head = embed;
    std::vector<qwen3::ref::LayerWeights> W(A::N_LAYERS);
    for (int l = 0; l < A::N_LAYERS; ++l) if (!qwen3::load_layer(st, l, W[l])) return 2;

    MetalHarness h;
    if (!h.ok() || !h.load_library(kdir + "/layer.metal")) { std::fprintf(stderr, "metal init: %s\n", h.error().c_str()); return 2; }
    qwen3::Chain ch{h};
    qwen3::Model M(ch, h, embed, lm_head, final_norm, W);

    std::vector<std::int32_t> prompt = {9707, 3838, 374, 279, 6722, 315, 9625, 30};
    std::printf("beam search: K=%d, steps=%d, prompt (%zu tokens)\n", K, steps, prompt.size());

    // Prefill once → prompt KV + last-token logits.
    qwen3::KVCache base;
    std::vector<std::int32_t> pos0;
    for (std::size_t i = 0; i < prompt.size(); ++i) pos0.push_back((int)i);
    auto hidden = M.forward_block(M.embed_tokens(prompt), pos0, base);
    auto lp0 = qwen3::log_softmax(M.logits_at(hidden, (int)prompt.size() - 1));

    // Seed K beams from the top-K first tokens (each copies the prompt KV).
    std::vector<Beam> beams;
    for (auto& [s, t] : topk(lp0, K)) {
        Beam b; b.gen = {(std::int32_t)t}; b.score = s; b.kv = base;
        beams.push_back(std::move(b));
    }

    for (int step = 1; step < steps; ++step) {
        std::vector<std::tuple<double, int, int>> cand;  // (score, parent, token)
        for (int bi = 0; bi < (int)beams.size(); ++bi) {
            Beam& b = beams[bi];
            auto emb = M.embed_tokens({b.gen.back()});
            std::vector<std::int32_t> posd = {b.kv.len};
            auto hd = M.forward_block(emb, posd, b.kv);  // extends b.kv
            auto lp = qwen3::log_softmax(M.logits_at(hd, 0));
            for (auto& [s, t] : topk(lp, K)) cand.push_back({b.score + s, bi, t});
        }
        std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                          [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
        std::vector<Beam> next;
        for (int i = 0; i < K; ++i) {
            auto& [s, parent, t] = cand[i];
            Beam nb;
            nb.gen = beams[parent].gen;
            nb.gen.push_back((std::int32_t)t);
            nb.score = s;
            nb.kv = beams[parent].kv;  // inherit parent's (extended) cache
            next.push_back(std::move(nb));
        }
        beams = std::move(next);
    }
    if (!ch.ok) { std::fprintf(stderr, "metal beam failed: %s\n", h.error().c_str()); return 2; }

    std::sort(beams.begin(), beams.end(), [](const Beam& a, const Beam& b) { return a.score > b.score; });
    std::printf("\ntop beams:\n");
    for (int i = 0; i < (int)beams.size(); ++i) {
        std::printf("  [%d] score=%.4f tokens:", i, beams[i].score);
        for (int t : beams[i].gen) std::printf(" %d", t);
        std::printf("\n");
    }
    std::printf("best beam tokens: ");
    for (int t : beams[0].gen) std::printf("%d ", t);
    std::printf("\nQWEN3_BEAM_OK\n");
    return 0;
}
