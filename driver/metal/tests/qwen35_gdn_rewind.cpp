// qwen35_gdn_rewind — validate the GDN recurrent-state rewind-on-reject on the
// REAL Qwen3.5-0.8B backbone (M2 sub-step 1 of the real-model MTP spec-decode).
//
// The GDN recurrent state is an irreversible fold, so a speculative window
// forward `[cur, draft]` that gets partially rejected leaves the state corrupted
// (the rejected draft's contribution is folded in). This proves the rewind fixes
// it on the real interleaved backbone (18 GDN + 6 attention layers):
//
//   S_ref   = GDN state after a greedy single-step forward of `cur`   (the truth)
//   S_bad   = GDN state after a 2-token window forward `[cur, draft]` (corrupted)
//   S_fixed = GDN state after rewind (restore snapshot + re-fold `cur` only)
//
// Assert: S_bad ≠ S_ref (rewind is NEEDED) and S_fixed == S_ref (rewind is
// CORRECT), across all 18 GDN layers. Snapshot = gather the slot's recurrent +
// conv state (MLX-immutable ⇒ holding the handle IS the snapshot); restore =
// write it back. See `ptir-metal-gdn-rewind-design`. The isolated frozen-
// checkpointed op is separately bit-exact-validated (gdn_checkpoint_test / M1).
//
// Usage: qwen35_gdn_rewind <hf_path> [prompt_len=8]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"
#include "linear_state_cache.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;

namespace {

model::ForwardBatch make_batch(const std::vector<int>& toks, int base, int ps,
                               bool all_rows) {
    const int n = (int)toks.size();
    std::vector<int> pos(n), rows, wr(n);
    for (int i = 0; i < n; ++i) { pos[i] = base + i; wr[i] = base + i; }
    if (all_rows) { rows.resize(n); for (int i = 0; i < n; ++i) rows[i] = i; }
    else rows = {n - 1};
    const int last = base + n - 1;
    const int n_pages = last / ps + 1;
    std::vector<int> pgi(n_pages);
    for (int i = 0; i < n_pages; ++i) pgi[i] = i;
    return model::ForwardBatch{
        mx::array(toks.data(), {n}, mx::int32),
        mx::array(pos.data(), {n}, mx::int32),
        mx::array(rows.data(), {(int)rows.size()}, mx::int32),
        mx::array(pgi.data(), {n_pages}, mx::int32),
        mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last % ps + 1}, {1}, mx::int32),
        mx::array({0, n}, {2}, mx::int32),
        mx::array(wr.data(), {n}, mx::int32),
        n, 1, (int)rows.size(), /*pure_decode=*/n == 1,
    };
}

// ── GDN state snapshot / restore over the LinearStateCache (slot 0) ──────────
// MLX arrays are immutable ⇒ gathering the slot's state IS a snapshot (holding
// the handle); restore scatters it back. The "free per-link copy" from the
// investigation, applied to the recurrent-state rewind.
struct GdnSnapshot {
    std::vector<mx::array> recur, conv;  // per linear layer, slot-0 state
};
GdnSnapshot snapshot_gdn(LinearStateCache& c) {
    GdnSnapshot s;
    const mx::array slot = mx::array({0}, {1}, mx::int32);
    for (int l = 0; l < c.n_linear_layers(); ++l) {
        s.recur.push_back(c.gather_recurrent_state(l, slot));
        s.conv.push_back(c.gather_conv_state(l, slot));
    }
    for (auto& t : s.recur) mx::eval(t);
    for (auto& t : s.conv) mx::eval(t);
    return s;
}
void restore_gdn(LinearStateCache& c, const GdnSnapshot& s) {
    const mx::array slot = mx::array({0}, {1}, mx::int32);
    for (int l = 0; l < c.n_linear_layers(); ++l) {
        c.write_recurrent_state(l, slot, s.recur[l]);
        c.write_conv_state(l, slot, s.conv[l]);
    }
    c.eval();
}
double max_state_diff(const GdnSnapshot& a, const GdnSnapshot& b) {
    double worst = 0.0;
    for (std::size_t l = 0; l < a.recur.size(); ++l) {
        mx::array d = mx::max(mx::abs(mx::subtract(a.recur[l], b.recur[l])));
        mx::eval(d);
        worst = std::max(worst, (double)d.item<float>());
    }
    return worst;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: qwen35_gdn_rewind <hf_path> [plen=8]\n"); return 2; }
    const std::string hf = argv[1];
    const int plen = argc > 2 ? std::atoi(argv[2]) : 8;

    BatchingConfig batching;
    loader::LoadedModel model = loader::load_model(hf, batching);
    const int ps = model.kv->page_size();
    std::printf("[gdn-rewind] arch=%s layers=%d lin_cache=%s\n",
                model.caps.arch_name.c_str(), model.caps.num_hidden_layers,
                model.lin_cache ? "yes(GDN)" : "no(std-attn)");
    if (model.lin_cache == nullptr) {
        std::fprintf(stderr, "[gdn-rewind] model has no GDN state — need a hybrid model (Qwen3.5)\n");
        return 2;
    }
    auto& cache = *model.lin_cache;
    auto attach = [&](model::ForwardBatch& b) {
        b.lin_cache = &cache;
        b.slot_ids = mx::array({0}, {1}, mx::int32);
        b.qo_indptr_host.assign({0, b.n_total});
    };

    // Prefill a prompt → GDN state folded over positions 0..plen-1.
    std::vector<int> prompt;
    for (int i = 0; i < plen; ++i) prompt.push_back(785 + i * 13);
    auto b0 = make_batch(prompt, 0, ps, /*all_rows=*/true);
    attach(b0);
    mx::array lg = model.graph->forward(b0, *model.kv);
    mx::eval(lg);
    cache.eval();
    int cur = mx::argmax(mx::slice(lg, {plen - 1, 0}, {plen, lg.shape(1)}), 1).item<int>();
    const int P = plen;  // cur will be forwarded at absolute position P

    // Snapshot the committed state (after the prompt) — the common starting point.
    GdnSnapshot S0 = snapshot_gdn(cache);

    // ── S_ref: greedy single-step forward of `cur` at P (persists after cur) ──
    {
        auto b = make_batch({cur}, P, ps, /*all_rows=*/false);
        attach(b);
        mx::array l = model.graph->forward(b, *model.kv);
        mx::eval(l);
        cache.eval();
    }
    GdnSnapshot S_ref = snapshot_gdn(cache);

    // Reset to the committed state for the speculative path.
    restore_gdn(cache, S0);

    // A deliberately-wrong draft (so the window's 2nd token is rejected).
    const int wrong_draft = (cur + 12345) % model.caps.vocab_size;

    // ── S_bad: 2-token window forward [cur, wrong_draft] at [P, P+1] ──
    {
        auto b = make_batch({cur, wrong_draft}, P, ps, /*all_rows=*/true);
        attach(b);
        mx::array l = model.graph->forward(b, *model.kv);
        mx::eval(l);
        cache.eval();
    }
    GdnSnapshot S_bad = snapshot_gdn(cache);

    // ── S_fixed: rewind — restore the pre-window snapshot, re-fold `cur` only ──
    restore_gdn(cache, S0);
    {
        auto b = make_batch({cur}, P, ps, /*all_rows=*/false);
        attach(b);
        mx::array l = model.graph->forward(b, *model.kv);
        mx::eval(l);
        cache.eval();
    }
    GdnSnapshot S_fixed = snapshot_gdn(cache);

    // ── verdict ──
    const double d_bad   = max_state_diff(S_bad, S_ref);    // should be LARGE (rewind needed)
    const double d_fixed = max_state_diff(S_fixed, S_ref);  // should be ~0   (rewind correct)
    const double TOL = 1e-5;
    std::printf("\n[gdn-rewind] real Qwen3.5-0.8B backbone, %d GDN layers, cur=%d wrong_draft=%d:\n",
                cache.n_linear_layers(), cur, wrong_draft);
    std::printf("  |S_bad   - S_ref| = %.4e   (speculative window folds the draft → CORRUPTED; want LARGE)\n", d_bad);
    std::printf("  |S_fixed - S_ref| = %.4e   (rewind restores committed-only state; want ~0)\n", d_fixed);
    const bool need = d_bad > 1e-3;      // the draft genuinely perturbs the state
    const bool ok   = d_fixed <= TOL;    // rewind recovers the reference exactly
    std::printf("  rewind NEEDED (S_bad ≠ S_ref): %s ; rewind CORRECT (S_fixed == S_ref): %s\n",
                need ? "YES" : "NO", ok ? "YES" : "NO");
    std::printf("%s\n", (need && ok) ? "GDN_REWIND_OK" : "GDN_REWIND_FAIL");
    return (need && ok) ? 0 : 1;
}
