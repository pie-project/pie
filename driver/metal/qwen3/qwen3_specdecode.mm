// qwen3_specdecode — Metal CONSTRAINED SPECULATIVE DECODING for Qwen3-0.6B.
// The mtpverify pattern composed from certified pieces: draft k tokens greedily,
// then spec-VERIFY them against the target under a GRAMMAR MASK — masked logits
// via the certified `dselect` (matrix_select_mask op) + `argmax` (both Metal) —
// and accept the longest matching prefix; a grammar-disallowed draft is rejected
// and the mask-corrected token substituted. Proves "Metal runs constrained
// speculative decoding," not just "the spec-verify ops replay."
//
// Usage: qwen3_specdecode <model.safetensors> [layer_kernels] [sampling_kernels] [k] [iters]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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
#ifndef PTIR_KERNELS_DIR
#define PTIR_KERNELS_DIR "."
#endif

using namespace ptir_metal;
namespace A = qwen3::arch;

namespace {
// Grammar-masked argmax via the certified ops: masked = dselect(mask, logits,
// -inf) then reduce_argmax — both Metal kernels (sampling_ir.metal), on `h2`.
std::int32_t masked_argmax(MetalHarness& h2, const std::vector<std::uint8_t>& mask,
                           const std::vector<float>& logits, const std::vector<float>& neg_inf) {
    std::uint32_t n = (std::uint32_t)logits.size(), lc = n, la = n, lb = n;
    std::vector<float> masked(n, 0.0f);
    std::vector<Arg> a = {Arg::in(mask.data(), n), Arg::in(logits.data(), n * 4),
                          Arg::in(neg_inf.data(), n * 4), Arg::out(masked.data(), n * 4),
                          Arg::in(&n, 4), Arg::in(&lc, 4), Arg::in(&la, 4), Arg::in(&lb, 4)};
    h2.run("dselect_f32", a, n);
    std::vector<std::int32_t> tok(1, 0);
    int rows = 1, len = (int)n;
    std::vector<Arg> b = {Arg::in(masked.data(), n * 4), Arg::out(tok.data(), 4),
                          Arg::in(&rows, 4), Arg::in(&len, 4)};
    h2.run("reduce_argmax_rows", b, 1);
    return tok[0];
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.safetensors> [layer_k] [samp_k] [k] [iters]\n", argv[0]); return 2; }
    std::string model = argv[1];
    std::string layer_k = argc > 2 ? argv[2] : QWEN3_KERNELS_DIR;
    std::string samp_k = argc > 3 ? argv[3] : PTIR_KERNELS_DIR;
    int k = argc > 4 ? std::atoi(argv[4]) : 4;
    int iters = argc > 5 ? std::atoi(argv[5]) : 3;

    qwen3::SafeTensors st;
    if (!st.open(model)) { std::fprintf(stderr, "cannot open %s\n", model.c_str()); return 2; }
    std::vector<float> embed, lm_head, final_norm;
    if (!st.load_f32("model.embed_tokens.weight", embed) || !st.load_f32("model.norm.weight", final_norm)) return 2;
    if (!st.load_f32("lm_head.weight", lm_head)) lm_head = embed;
    std::vector<qwen3::ref::LayerWeights> W(A::N_LAYERS);
    for (int l = 0; l < A::N_LAYERS; ++l) if (!qwen3::load_layer(st, l, W[l])) return 2;

    MetalHarness h;   // forward (layer.metal)
    MetalHarness h2;  // grammar ops (sampling_ir.metal)
    if (!h.ok() || !h.load_library(layer_k + "/layer.metal")) { std::fprintf(stderr, "metal: %s\n", h.error().c_str()); return 2; }
    if (!h2.ok() || !h2.load_library(samp_k + "/sampling_ir.metal")) { std::fprintf(stderr, "metal2: %s\n", h2.error().c_str()); return 2; }
    qwen3::Chain ch{h};
    qwen3::Model M(ch, h, embed, lm_head, final_norm, W);

    // Grammar: allow-all EXCEPT a disallowed set (demonstrates the constraint —
    // a greedy draft landing on a disallowed token is rejected + corrected).
    std::vector<std::uint8_t> grammar(A::VOCAB, 1);
    std::vector<int> disallow = {13};  // e.g. forbid "." to force longer spans
    for (int t : disallow) grammar[t] = 0;
    std::vector<float> neg_inf(A::VOCAB, -std::numeric_limits<float>::infinity());

    std::vector<std::int32_t> prompt = {9707, 3838, 374, 279, 6722, 315, 9625, 30};
    std::printf("constrained spec-decode: k=%d drafts/iter, %d iters, grammar forbids {13}\n", k, iters);

    // Prefill.
    qwen3::KVCache kv;
    std::vector<std::int32_t> pos0;
    for (std::size_t i = 0; i < prompt.size(); ++i) pos0.push_back((int)i);
    auto hidden = M.forward_block(M.embed_tokens(prompt), pos0, kv);
    // First accepted token = grammar-masked argmax of the prompt's last logits.
    std::int32_t cur = masked_argmax(h2, grammar, M.logits_at(hidden, (int)prompt.size() - 1), neg_inf);
    std::vector<std::int32_t> seq = {cur};

    int total_drafted = 0, total_accepted = 0;
    for (int it = 0; it < iters; ++it) {
        // DRAFT: greedily extend from `cur` for k tokens over a DRAFT copy of kv.
        qwen3::KVCache dkv = kv;
        std::vector<std::int32_t> draft;
        std::vector<std::vector<float>> draft_logits;  // logits used to pick each draft
        std::int32_t d = cur;
        for (int j = 0; j < k; ++j) {
            auto hd = M.forward_block(M.embed_tokens({d}), {dkv.len}, dkv);
            auto lg = M.logits_at(hd, 0);
            std::int32_t nxt = M.argmax(lg);   // UNCONSTRAINED greedy draft
            draft.push_back(nxt);
            draft_logits.push_back(lg);
            d = nxt;
        }
        total_drafted += k;

        // VERIFY under the grammar mask: masked-argmax of each draft's logits;
        // accept the prefix where the grammar-allowed token == the draft token.
        int accepted = 0;
        std::int32_t correction = -1;
        for (int j = 0; j < k; ++j) {
            std::int32_t want = masked_argmax(h2, grammar, draft_logits[j], neg_inf);
            if (want == draft[j]) { ++accepted; }
            else { correction = want; break; }
        }

        // COMMIT: accepted drafts advance the real kv; on a reject, substitute
        // the grammar-corrected token (still 1 real forward's worth of progress).
        // Re-forward the accepted tokens (+correction) over the real kv.
        std::vector<std::int32_t> commit(draft.begin(), draft.begin() + accepted);
        if (correction >= 0) commit.push_back(correction);
        std::int32_t last_committed = cur;
        for (std::int32_t t : commit) {
            M.forward_block(M.embed_tokens({last_committed}), {kv.len}, kv);
            seq.push_back(t);
            last_committed = t;
        }
        cur = last_committed;
        total_accepted += accepted;
        std::printf("  iter %d: drafted %d, accepted %d%s\n", it, k, accepted,
                    correction >= 0 ? " (+1 grammar-corrected)" : "");
    }
    if (!ch.ok) { std::fprintf(stderr, "metal spec-decode failed: %s\n", h.error().c_str()); return 2; }

    std::printf("\nsequence:");
    for (int t : prompt) std::printf(" %d", t);
    std::printf(" |");
    for (int t : seq) std::printf(" %d", t);
    std::printf("\nspeculative acceptance: %d/%d drafts accepted (grammar-constrained, {13} forbidden)\n",
                total_accepted, total_drafted);
    std::printf("QWEN3_SPECDECODE_OK\n");
    return 0;
}
