// qwen35_mtp_specdecode — REAL Qwen3.5-0.8B MTP speculative decode with the GDN
// recurrent-state rewind-on-reject (M2 sub-step 2: the full autoregressive loop).
//
// Composes: the real trained MTP head (drafter) + a 2-token window verify forward
// on the GDN hybrid backbone + accept/reject + the GDN state rewind (snapshot/
// restore, per `ptir-metal-gdn-rewind-design`; the frozen-checkpointed op is the
// in-graph optimization, bit-exact-validated in gdn_checkpoint_test / M1).
//
// K=1 chained MTP: the head drafts token_{P+1} from (hidden_{P-1}, cur=token_P);
// the window [cur, draft] is forwarded in ONE fire → verify draft vs the backbone
// argmax. On ACCEPT commit 2 tokens (the speedup); on REJECT commit 1 + REWIND
// the GDN state (the window folded the rejected draft — restore + re-fold cur).
//
// CORRECTNESS GATE: the committed sequence == plain greedy decode, BIT-EXACT —
// the spec-decode invariant, and the proof the GDN rewind keeps the recurrent
// state correct (a wrong rewind would diverge the committed tokens from greedy).
//
// Usage: qwen35_mtp_specdecode <hf_path> [gen=48] [prompt_ids_csv]
//        env MTP_SD_JSON=<path> writes the committed sequence (cross-check).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"
#include "linear_state_cache.hpp"
#include "ops/norm.hpp"
#include "ops/activation.hpp"
#include "ops/rope.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;

namespace {

mx::array f32(const mx::array& a) { return mx::astype(a, mx::float32); }
mx::array linW(const mx::array& x, const mx::array& W) {
    return mx::matmul(x, mx::transpose(f32(W)));
}

// A forward batch over `toks` at absolute positions [base..base+n-1]. all_rows ⇒
// logits+hidden at every position (window verify); else only the last row.
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

// ── the real trained MTP head (verbatim recipe from qwen35_mtp.cpp) ──────────
struct Mtp {
    std::unordered_map<std::string, mx::array> t;
    int H = 0, n_q = 0, n_kv = 0, d = 0, rope_dims = 0;
    float eps = 1e-6f, theta = 1e7f;
    const mx::array& g(const std::string& k) const { return t.at(k); }
};
int mtp_draft(const Mtp& m, const std::vector<mx::array>& H_list,
              const std::vector<int>& T_list, const std::vector<int>& P_list,
              const mx::array& embed) {
    using namespace mx;
    const int n = (int)T_list.size();
    const int H = m.H, nq = m.n_q, nkv = m.n_kv, d = m.d;
    const float scale = 1.0f / std::sqrt((float)d);
    array hid = concatenate(std::vector<array>(H_list.begin(), H_list.end()), 0);
    array emb = take(f32(embed), array(T_list.data(), {n}, int32), 0);
    array nx = ops::rms_norm(emb, f32(m.g("mtp.pre_fc_norm_embedding.weight")), m.eps, true);
    array ny = ops::rms_norm(f32(hid), f32(m.g("mtp.pre_fc_norm_hidden.weight")), m.eps, true);
    array cat = concatenate({nx, ny}, 1);
    array y = linW(cat, m.g("mtp.fc.weight"));
    const std::string p = "mtp.layers.0.";
    array res = y;
    array h = ops::rms_norm(y, f32(m.g(p + "input_layernorm.weight")), m.eps, true);
    array qg = reshape(linW(h, m.g(p + "self_attn.q_proj.weight")), {n, nq, 2, d});
    array Q    = reshape(slice(qg, {0, 0, 0, 0}, {n, nq, 1, d}), {n, nq, d});
    array gate = reshape(slice(qg, {0, 0, 1, 0}, {n, nq, 2, d}), {n, nq, d});
    array K = reshape(linW(h, m.g(p + "self_attn.k_proj.weight")), {n, nkv, d});
    array V = reshape(linW(h, m.g(p + "self_attn.v_proj.weight")), {n, nkv, d});
    Q = ops::rms_norm(Q, f32(m.g(p + "self_attn.q_norm.weight")), m.eps, true);
    K = ops::rms_norm(K, f32(m.g(p + "self_attn.k_norm.weight")), m.eps, true);
    array positions = array(P_list.data(), {n}, int32);
    ops::RopeParams rp; rp.theta = m.theta;
    Q = ops::rope(Q, positions, m.rope_dims, rp);
    K = ops::rope(K, positions, m.rope_dims, rp);
    array Kr = repeat(K, nq / nkv, 1), Vr = repeat(V, nq / nkv, 1);
    array Qt = transpose(Q, {1, 0, 2}), Kt = transpose(Kr, {1, 0, 2}), Vt = transpose(Vr, {1, 0, 2});
    array scores = multiply(matmul(Qt, transpose(Kt, {0, 2, 1})), array(scale));
    array ri = reshape(arange(n), {n, 1}), ci = reshape(arange(n), {1, n});
    array mask = reshape(where(greater(ci, ri), array(-1e30f), array(0.0f)), {1, n, n});
    array w = softmax(add(scores, mask), std::vector<int>{-1}, true);
    array ao = transpose(matmul(w, Vt), {1, 0, 2});
    array attn = multiply(reshape(ao, {n, nq * d}), sigmoid(reshape(gate, {n, nq * d})));
    array o = linW(attn, m.g(p + "self_attn.o_proj.weight"));
    y = add(res, o);
    array res2 = y;
    array hn = ops::rms_norm(y, f32(m.g(p + "post_attention_layernorm.weight")), m.eps, true);
    array ffn = linW(ops::swiglu(linW(hn, m.g(p + "mlp.gate_proj.weight")),
                                 linW(hn, m.g(p + "mlp.up_proj.weight"))),
                     m.g(p + "mlp.down_proj.weight"));
    y = add(res2, ffn);
    array fn = ops::rms_norm(y, f32(m.g("mtp.norm.weight")), m.eps, true);
    array last = reshape(slice(fn, {n - 1, 0}, {n, H}), {1, H});
    array am = argmax(linW(last, embed), 1);
    eval(am);
    return am.item<int>();
}

// ── GDN state snapshot / restore (slot 0) — MLX-immutable ⇒ gather IS snapshot ─
struct GdnSnapshot { std::vector<mx::array> recur, conv; };
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

struct Loaded {
    loader::LoadedModel model;
    Mtp m;
    mx::array embed;
    int ps;
};

// Greedy reference decode of `gen` tokens (single-token, states persist normally).
std::vector<int> greedy_decode(Loaded& L, const std::vector<int>& prompt, int gen) {
    auto& model = L.model;
    L.model.lin_cache->reset();
    auto attach = [&](model::ForwardBatch& b) {
        b.lin_cache = model.lin_cache.get();
        b.slot_ids = mx::array({0}, {1}, mx::int32);
        b.qo_indptr_host.assign({0, b.n_total});
    };
    const int plen = (int)prompt.size();
    auto b0 = make_batch(prompt, 0, L.ps, /*all_rows=*/true);
    attach(b0);
    mx::array lg = model.graph->forward(b0, *model.kv);
    mx::eval(lg); model.kv->eval();
    int cur = mx::argmax(mx::slice(lg, {plen - 1, 0}, {plen, lg.shape(1)}), 1).item<int>();
    std::vector<int> seq;
    int P = plen;
    for (int i = 0; i < gen; ++i) {
        seq.push_back(cur);
        auto b = make_batch({cur}, P, L.ps, /*all_rows=*/false);
        attach(b);
        mx::array l = model.graph->forward(b, *model.kv);
        mx::eval(l); model.kv->eval();
        cur = mx::argmax(l, 1).item<int>();
        ++P;
    }
    return seq;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: qwen35_mtp_specdecode <hf_path> [gen=48] [prompt_csv]\n"); return 2; }
    const std::string hf = argv[1];
    const int gen = argc > 2 ? std::atoi(argv[2]) : 48;

    std::vector<int> prompt;
    if (argc > 3) { std::string s = argv[3], num;
        for (char c : s) { if (c==',' || c==' ') { if(!num.empty()){prompt.push_back(std::atoi(num.c_str()));num.clear();} } else num.push_back(c); }
        if (!num.empty()) prompt.push_back(std::atoi(num.c_str()));
    } else {
        prompt = {760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 6124, 369};
    }
    const int plen = (int)prompt.size();

    Loaded L{loader::load_model(hf, BatchingConfig{}), Mtp{}, mx::array(0.0f), 0};
    L.ps = L.model.kv->page_size();
    auto& model = L.model;
    std::printf("[mtp-sd] arch=%s layers=%d lin_cache=%s plen=%d gen=%d\n",
                model.caps.arch_name.c_str(), model.caps.num_hidden_layers,
                model.lin_cache ? "yes(GDN)" : "no", plen, gen);
    if (model.lin_cache == nullptr) { std::fprintf(stderr, "[mtp-sd] need a GDN model\n"); return 2; }

    // MTP + embed weights.
    auto st = mx::load_safetensors(hf + "/model.safetensors-00001-of-00001.safetensors");
    for (auto& kv : st.first) if (kv.first.rfind("mtp.", 0) == 0) L.m.t.emplace(kv.first, kv.second);
    L.embed = st.first.at("model.language_model.embed_tokens.weight");
    L.m.H = L.embed.shape(1); L.m.n_q = 8; L.m.n_kv = 2; L.m.eps = 1e-6f; L.m.theta = 1e7f;
    L.m.d = L.m.t.at("mtp.layers.0.self_attn.v_proj.weight").shape(0) / L.m.n_kv;
    L.m.rope_dims = std::max(2, 2 * (int)std::floor(0.5f * 0.25f * L.m.d));

    // ── reference greedy (the correctness oracle) ──
    std::vector<int> greedy = greedy_decode(L, prompt, gen);

    // ── real MTP spec-decode with GDN rewind ──
    model.lin_cache->reset();
    auto attach = [&](model::ForwardBatch& b) {
        b.lin_cache = model.lin_cache.get();
        b.slot_ids = mx::array({0}, {1}, mx::int32);
        b.qo_indptr_host.assign({0, b.n_total});
    };
    auto rowH = [&](const mx::array& a, int i) {
        return mx::reshape(mx::slice(a, {i, 0}, {i + 1, a.shape(1)}), {1, a.shape(1)});
    };
    auto b0 = make_batch(prompt, 0, L.ps, /*all_rows=*/true);
    attach(b0);
    mx::array lg = model.graph->forward(b0, *model.kv);
    mx::eval(lg); model.kv->eval();
    mx::array hidden_all = model.graph->last_hidden();  // [plen, H]
    mx::eval(hidden_all);

    // MTP history over the prompt: entry (hidden_{i}, token_{i+1}, i+1).
    std::vector<mx::array> H_list; std::vector<int> T_list, P_list;
    for (int i = 0; i + 1 < plen; ++i) {
        H_list.push_back(rowH(hidden_all, i)); T_list.push_back(prompt[i + 1]); P_list.push_back(i + 1);
    }
    int cur = mx::argmax(mx::slice(lg, {plen - 1, 0}, {plen, lg.shape(1)}), 1).item<int>();
    mx::array hidden_prev = rowH(hidden_all, plen - 1);  // hidden at P-1
    int P = plen;

    std::vector<int> committed;
    committed.push_back(cur);  // the first committed token (g[0], = argmax at prefill);
                               // each fire then commits its successors g[1], g[2], ...
    int fires = 0, accepts = 0;
    while ((int)committed.size() < gen) {
        ++fires;
        // Draft token_{P+1}: append (hidden_{P-1}, cur=token_P, P), MTP → t_{P+1}.
        H_list.push_back(hidden_prev); T_list.push_back(cur); P_list.push_back(P);
        int draft = mtp_draft(L.m, H_list, T_list, P_list, L.embed);

        GdnSnapshot snap = snapshot_gdn(*model.lin_cache);
        // Window verify: forward [cur, draft] at [P, P+1] in one fire.
        auto bw = make_batch({cur, draft}, P, L.ps, /*all_rows=*/true);
        attach(bw);
        mx::array lw = model.graph->forward(bw, *model.kv);
        mx::eval(lw); model.kv->eval();
        mx::array hw = model.graph->last_hidden();  // [2, H]
        mx::eval(hw);
        int t1 = mx::argmax(rowH(lw, 0), 1).item<int>();  // truth token_{P+1}
        mx::array hidden_P = rowH(hw, 0);

        if (draft == t1) {  // ACCEPT — commit 2, GDN state (folded [cur,t1]) correct
            ++accepts;
            int t2 = mx::argmax(rowH(lw, 1), 1).item<int>();
            committed.push_back(t1);
            if ((int)committed.size() < gen) committed.push_back(t2);
            H_list.push_back(hidden_P); T_list.push_back(t1); P_list.push_back(P + 1);
            cur = t2; hidden_prev = rowH(hw, 1); P += 2;
        } else {            // REJECT — commit 1, REWIND GDN (fold cur only)
            committed.push_back(t1);
            restore_gdn(*model.lin_cache, snap);
            auto br = make_batch({cur}, P, L.ps, /*all_rows=*/false);
            attach(br);
            mx::array lr = model.graph->forward(br, *model.kv);
            mx::eval(lr); model.kv->eval();
            cur = t1; hidden_prev = hidden_P; P += 1;
        }
    }
    committed.resize(gen);

    // ── verdict: committed == greedy (bit-exact) ──
    bool ok = std::equal(committed.begin(), committed.end(), greedy.begin());
    std::printf("\n[mtp-sd] REAL Qwen3.5-0.8B MTP spec-decode + GDN rewind (%d GDN layers):\n",
                model.lin_cache->n_linear_layers());
    std::printf("  fires=%d accepts=%d (%.1f%%) → committed %d tokens\n",
                fires, accepts, 100.0 * accepts / std::max(1, fires), (int)committed.size());
    std::printf("  committed == greedy (spec-decode invariant + GDN-rewind correctness): %s\n",
                ok ? "YES" : "NO");
    std::printf("  committed:");
    for (int i = 0; i < 16 && i < gen; ++i) std::printf(" %d", committed[i]);
    std::printf(" ...\n");
    if (!ok) for (int i = 0; i < gen; ++i) if (committed[i] != greedy[i]) {
        std::printf("  MISMATCH at %d: committed=%d greedy=%d\n", i, committed[i], greedy[i]); break;
    }

    if (const char* jp = std::getenv("MTP_SD_JSON")) {
        std::FILE* f = std::fopen(jp, "w");
        if (f) {
            auto arr = [&](const char* nm, const std::vector<int>& v, bool last) {
                std::fprintf(f, "  \"%s\": [", nm);
                for (std::size_t i = 0; i < v.size(); ++i) std::fprintf(f, "%s%d", i ? ", " : "", v[i]);
                std::fprintf(f, "]%s\n", last ? "" : ",");
            };
            std::fprintf(f, "{\n  \"schema\": \"mtp_specdecode_crosscheck_v1\",\n");
            std::fprintf(f, "  \"backend\": \"Metal (Apple M1 Max, MLX 0.31.2)\",\n");
            std::fprintf(f, "  \"model\": \"Qwen3.5-0.8B\",\n  \"draft_depth_K\": 1,\n  \"verify\": \"greedy\",\n");
            std::fprintf(f, "  \"steps\": %d,\n  \"fires\": %d,\n  \"accepted_total\": %d,\n", gen, fires, accepts);
            arr("prompt_ids", prompt, false);
            arr("generated_ids", committed, true);
            std::fprintf(f, "}\n");
            std::fclose(f);
            std::printf("  cross-check JSON written: %s\n", jp);
        }
    }
    std::printf("%s\n", ok ? "MTP_SPECDECODE_OK" : "MTP_SPECDECODE_FAIL");
    return ok ? 0 : 1;
}
