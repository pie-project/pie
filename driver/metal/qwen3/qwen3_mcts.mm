// qwen3_mcts — Metal MCTS (Monte-Carlo Tree Search) decoding for Qwen3-0.6B, the
// 3rd algorithm. A host-orchestrated search TREE (selection via PUCT / expansion /
// backprop — host control, like interp.rs's geometry/control model) with the
// tensor work on Metal: each node evaluation is a Metal forward + log_softmax
// (priors) + reduce (value); the final per-position choice is the most-visited
// child. Proves "Metal runs MCTS decoding too."
//
// Op surface: forward (Metal) + log_softmax (reduce/exp/log) + argmax (Metal) +
// host tree bookkeeping — no tensor op beyond the certified set.
//
// Usage: qwen3_mcts <model.safetensors> [kernels] [sims] [branch] [out_tokens]

#include <algorithm>
#include <cmath>
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
struct Node {
    std::int32_t token = -1;
    double prior = 1.0;       // P (prob) from the parent's log_softmax
    double W = 0.0;           // total backed-up value
    int N = 0;                // visit count
    bool expanded = false;
    std::vector<int> children;  // arena indices
    qwen3::KVCache kv;          // context INCLUDING this node's token (valid once expanded)
    double q() const { return N > 0 ? W / N : 0.0; }
};

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
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.safetensors> [kernels] [sims] [branch] [out]\n", argv[0]); return 2; }
    std::string model = argv[1];
    std::string kdir = argc > 2 ? argv[2] : QWEN3_KERNELS_DIR;
    int sims = argc > 3 ? std::atoi(argv[3]) : 12;
    int branch = argc > 4 ? std::atoi(argv[4]) : 4;
    int out_tokens = argc > 5 ? std::atoi(argv[5]) : 6;
    const double c_puct = 1.5;

    qwen3::SafeTensors st;
    if (!st.open(model)) { std::fprintf(stderr, "cannot open %s\n", model.c_str()); return 2; }
    std::vector<float> embed, lm_head, final_norm;
    if (!st.load_f32("model.embed_tokens.weight", embed) || !st.load_f32("model.norm.weight", final_norm)) return 2;
    if (!st.load_f32("lm_head.weight", lm_head)) lm_head = embed;
    std::vector<qwen3::ref::LayerWeights> W(A::N_LAYERS);
    for (int l = 0; l < A::N_LAYERS; ++l) if (!qwen3::load_layer(st, l, W[l])) return 2;

    MetalHarness h;
    if (!h.ok() || !h.load_library(kdir + "/layer.metal")) { std::fprintf(stderr, "metal: %s\n", h.error().c_str()); return 2; }
    qwen3::Chain ch{h};
    qwen3::Model M(ch, h, embed, lm_head, final_norm, W);

    std::vector<std::int32_t> prompt = {9707, 3838, 374, 279, 6722, 315, 9625, 30};
    std::printf("MCTS decode: sims=%d, branch=%d, out=%d tokens, c_puct=%.1f\n", sims, branch, out_tokens, c_puct);

    // Committed context KV (advances as we pick output tokens).
    qwen3::KVCache ctx;
    std::vector<std::int32_t> pos0;
    for (std::size_t i = 0; i < prompt.size(); ++i) pos0.push_back((int)i);
    auto hprompt = M.forward_block(M.embed_tokens(prompt), pos0, ctx);
    std::vector<float> root_lp = qwen3::log_softmax(M.logits_at(hprompt, (int)prompt.size() - 1));

    std::vector<std::int32_t> chosen;
    for (int outpos = 0; outpos < out_tokens; ++outpos) {
        std::vector<Node> tree;
        tree.reserve(sims * branch + 4);
        // Root: context already forwarded; children = top-`branch` of root_lp.
        tree.push_back(Node{});
        Node& root = tree[0];
        root.expanded = true;
        root.kv = ctx;
        for (auto& [lp, t] : topk(root_lp, branch)) {
            int ci = (int)tree.size();
            tree.push_back(Node{});
            tree[ci].token = (std::int32_t)t;
            tree[ci].prior = std::exp((double)lp);
            tree[0].children.push_back(ci);
        }

        for (int s = 0; s < sims; ++s) {
            // 1. SELECTION: descend by PUCT to an unexpanded leaf.
            std::vector<int> path = {0};
            int cur = 0;
            while (tree[cur].expanded && !tree[cur].children.empty()) {
                double best = -1e30; int bc = tree[cur].children[0];
                for (int ci : tree[cur].children) {
                    double u = tree[ci].q() + c_puct * tree[ci].prior *
                               std::sqrt((double)std::max(1, tree[cur].N)) / (1.0 + tree[ci].N);
                    if (u > best) { best = u; bc = ci; }
                }
                cur = bc; path.push_back(cur);
            }
            // 2. EXPANSION + EVALUATION on Metal: forward the leaf's token.
            int leaf = cur;
            int parent = path.size() >= 2 ? path[path.size() - 2] : 0;
            double value;
            {
                tree[leaf].kv = tree[parent].kv;  // context up to parent
                auto hd = M.forward_block(M.embed_tokens({tree[leaf].token}), {tree[leaf].kv.len}, tree[leaf].kv);
                auto lp = qwen3::log_softmax(M.logits_at(hd, 0));
                // value = model confidence at the leaf (max log-softmax prob).
                float mx = -3.0e38f;
                for (float x : lp) mx = x > mx ? x : mx;
                value = (double)mx;  // in (-inf, 0]; higher = more confident continuation
                // children = top-`branch` next tokens.
                for (auto& [clp, t] : topk(lp, branch)) {
                    int ci = (int)tree.size();
                    tree.push_back(Node{});
                    tree[ci].token = (std::int32_t)t;
                    tree[ci].prior = std::exp((double)clp);
                    tree[leaf].children.push_back(ci);
                }
                tree[leaf].expanded = true;
            }
            // 3. BACKPROP: update visit counts + values up the path (host).
            for (int idx : path) { tree[idx].N += 1; tree[idx].W += value; }
        }
        if (!ch.ok) { std::fprintf(stderr, "metal mcts failed: %s\n", h.error().c_str()); return 2; }

        // Pick the most-visited root child; commit it + advance the context.
        int best_child = tree[0].children[0];
        for (int ci : tree[0].children) if (tree[ci].N > tree[best_child].N) best_child = ci;
        std::int32_t tok = tree[best_child].token;
        chosen.push_back(tok);
        std::printf("  pos %d: chose %d (visits=%d, Q=%.3f) from %zu root children\n",
                    outpos, tok, tree[best_child].N, tree[best_child].q(), tree[0].children.size());
        // Advance committed context by the chosen token, refresh root_lp.
        auto hc = M.forward_block(M.embed_tokens({tok}), {ctx.len}, ctx);
        root_lp = qwen3::log_softmax(M.logits_at(hc, 0));
    }

    std::printf("\nMCTS-decoded tokens: ");
    for (int t : chosen) std::printf("%d ", t);
    std::printf("\nfull_sequence:");
    for (int t : prompt) std::printf(" %d", t);
    for (int t : chosen) std::printf(" %d", t);
    std::printf("\nQWEN3_MCTS_OK\n");
    return 0;
}
