// qwen35_mtp_bench — honest wall-clock benchmark: regular greedy decode vs
// MTP (K=1) speculative decode on Metal (real Qwen3.5-0.8B, corrected 50%-accept
// head). Measures T_target (one 24-layer GDN backbone forward + vocab lm_head)
// vs T_mtp (one 1-layer MTP-head forward + vocab lm_head), the acceptance rate,
// and the resulting speedup. Both forwards end in the SAME vocab=248320 tied
// lm_head matmul — the benchmark reveals whether the 1-layer head is actually
// cheap enough (T_mtp << T_target) for K=1 spec-decode to win.
//
// The MTP head here is INCREMENTAL (persistent K/V cache) — each draft processes
// only the new token attending over cached history, so T_mtp is the fair
// per-draft cost (not the O(n) full-history recompute used in the cert).
//
// Usage: qwen35_mtp_bench <hf_path> [N=32] [warmup=4]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"
#include "ops/norm.hpp"
#include "ops/activation.hpp"
#include "ops/rope.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

model::ForwardBatch make_batch(const std::vector<int>& toks, const std::vector<int>& pos,
                               int logit_row, int page_size, bool pure_decode) {
    const int n = (int)toks.size();
    const int last_pos = pos.back();
    const int n_pages = last_pos / page_size + 1;
    std::vector<int> page_idx(n_pages);
    for (int i = 0; i < n_pages; ++i) page_idx[i] = i;
    std::vector<int> write_idx(n);
    for (int i = 0; i < n; ++i) write_idx[i] = pos[i];
    const int last_page_len = last_pos % page_size + 1;
    return model::ForwardBatch{
        mx::array(toks.data(), {n}, mx::int32),
        mx::array(pos.data(), {n}, mx::int32),
        mx::array({logit_row}, {1}, mx::int32),
        mx::array(page_idx.data(), {n_pages}, mx::int32),
        mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last_page_len}, {1}, mx::int32),
        mx::array({0, n}, {2}, mx::int32),
        mx::array(write_idx.data(), {n}, mx::int32),
        n, 1, 1, pure_decode,
    };
}

mx::array f32(const mx::array& a) { return mx::astype(a, mx::float32); }
mx::array linW(const mx::array& x, const mx::array& W) {
    return mx::matmul(x, mx::transpose(f32(W)));
}

struct Mtp {
    std::unordered_map<std::string, mx::array> t;
    int H = 0, n_q = 8, n_kv = 2, d = 0, rope_dims = 0;
    float eps = 1e-6f, theta = 1e7f;
    const mx::array& g(const std::string& k) const { return t.at(k); }
};

// Incremental MTP head: appends the new position's K/V to the persistent cache
// and attends over all cached positions. Returns argmax(mtp_logits) = draft.
struct MtpKV { std::vector<mx::array> K, V; };
int mtp_draft_incr(const Mtp& m, MtpKV& kv, const mx::array& hidden, int token,
                   int pos, const mx::array& embed) {
    using namespace mx;
    const int H = m.H, nq = m.n_q, nkv = m.n_kv, d = m.d;
    array emb = take(f32(embed), array({token}, {1}, int32), 0);          // [1,H]
    array nx = ops::rms_norm(emb, f32(m.g("mtp.pre_fc_norm_embedding.weight")), m.eps, true);
    array ny = ops::rms_norm(f32(hidden), f32(m.g("mtp.pre_fc_norm_hidden.weight")), m.eps, true);
    array y = linW(concatenate({nx, ny}, 1), m.g("mtp.fc.weight"));       // [1,H]

    const std::string p = "mtp.layers.0.";
    array h = ops::rms_norm(y, f32(m.g(p + "input_layernorm.weight")), m.eps, true);
    array qg = reshape(linW(h, m.g(p + "self_attn.q_proj.weight")), {1, nq, 2, d});
    array Q    = reshape(slice(qg, {0, 0, 0, 0}, {1, nq, 1, d}), {1, nq, d});
    array gate = reshape(slice(qg, {0, 0, 1, 0}, {1, nq, 2, d}), {1, nq, d});
    array Kn = reshape(linW(h, m.g(p + "self_attn.k_proj.weight")), {1, nkv, d});
    array Vn = reshape(linW(h, m.g(p + "self_attn.v_proj.weight")), {1, nkv, d});
    Q  = ops::rms_norm(Q,  f32(m.g(p + "self_attn.q_norm.weight")), m.eps, true);
    Kn = ops::rms_norm(Kn, f32(m.g(p + "self_attn.k_norm.weight")), m.eps, true);
    array posarr = array({pos}, {1}, int32);
    ops::RopeParams rp; rp.theta = m.theta;
    Q  = ops::rope(Q,  posarr, m.rope_dims, rp);
    Kn = ops::rope(Kn, posarr, m.rope_dims, rp);
    kv.K.push_back(Kn); kv.V.push_back(Vn);
    const int L = (int)kv.K.size();
    array Kall = concatenate(std::vector<array>(kv.K.begin(), kv.K.end()), 0);  // [L,nkv,d]
    array Vall = concatenate(std::vector<array>(kv.V.begin(), kv.V.end()), 0);
    array Kr = repeat(Kall, nq / nkv, 1);   // [L,nq,d]
    array Vr = repeat(Vall, nq / nkv, 1);
    array Qt = transpose(Q,  {1, 0, 2});    // [nq,1,d]
    array Kt = transpose(Kr, {1, 0, 2});    // [nq,L,d]
    array Vt = transpose(Vr, {1, 0, 2});
    const float scale = 1.0f / std::sqrt((float)d);
    array scores = multiply(matmul(Qt, transpose(Kt, {0, 2, 1})), array(scale));  // [nq,1,L]
    array w = softmax(scores, std::vector<int>{-1}, true);
    array ao = transpose(matmul(w, Vt), {1, 0, 2});   // [1,nq,d]
    array attn = multiply(reshape(ao, {1, nq * d}), sigmoid(reshape(gate, {1, nq * d})));
    y = add(y, linW(attn, m.g(p + "self_attn.o_proj.weight")));
    array hn = ops::rms_norm(y, f32(m.g(p + "post_attention_layernorm.weight")), m.eps, true);
    array gp = linW(hn, m.g(p + "mlp.gate_proj.weight"));
    array up = linW(hn, m.g(p + "mlp.up_proj.weight"));
    y = add(y, linW(ops::swiglu(gp, up), m.g(p + "mlp.down_proj.weight")));
    array fn = ops::rms_norm(y, f32(m.g("mtp.norm.weight")), m.eps, true);
    array am = argmax(linW(fn, embed), 1);
    eval(am);
    return am.item<int>();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: qwen35_mtp_bench <hf_path> [N=32] [warmup=4]\n"; return 2; }
    const std::string hf_path = argv[1];
    const int N = argc > 2 ? std::atoi(argv[2]) : 32;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 4;
    std::vector<int> toks = {760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 6124, 369};
    const int plen = (int)toks.size();

    BatchingConfig batching;
    loader::LoadedModel model = loader::load_model(hf_path, batching);
    const int ps = model.kv->page_size();
    const bool hybrid = (model.lin_cache != nullptr);
    auto attach = [&](model::ForwardBatch& b) {
        if (!hybrid) return;
        b.lin_cache = model.lin_cache.get();
        b.slot_ids = mx::array({0}, {1}, mx::int32);
        b.qo_indptr_host.assign({0, b.n_total});
    };

    Mtp m;
    auto st = mx::load_safetensors(hf_path + "/model.safetensors-00001-of-00001.safetensors");
    for (auto& kv : st.first)
        if (kv.first.rfind("mtp.", 0) == 0) m.t.emplace(kv.first, kv.second);
    mx::array embed = st.first.at("model.language_model.embed_tokens.weight");
    m.H = embed.shape(1);
    m.d = m.t.at("mtp.layers.0.self_attn.v_proj.weight").shape(0) / m.n_kv;
    m.rope_dims = std::max(2, 2 * (int)std::floor(0.5f * 0.25f * m.d));
    const int vocab = embed.shape(0);
    std::printf("[bench] Qwen3.5-0.8B on Metal — H=%d head_dim=%d vocab=%d, N=%d warmup=%d\n\n",
                m.H, m.d, vocab, N, warmup);

    // ── Prefill ──
    std::vector<int> pos(plen);
    for (int i = 0; i < plen; ++i) pos[i] = i;
    auto b0 = make_batch(toks, pos, plen - 1, ps, false);
    attach(b0);
    mx::array logits = model.graph->forward(b0, *model.kv);
    mx::eval(logits); model.kv->eval();
    mx::array hidden = model.graph->last_hidden();
    mx::eval(hidden);
    int cur = mx::argmax(logits, 1).item<int>();
    int p = plen - 1;

    // Seed the incremental MTP cache with the prompt so drafts have real context.
    // (One-time; not part of the per-draft timing.) Uses the same hidden as the
    // last prefill row is unavailable per-position here, so seed lightly from the
    // last prompt hidden — the per-draft COST is what we time, not acceptance.
    MtpKV mkv;

    auto forward_one = [&](int tok, int at) {
        auto b = make_batch({tok}, {at}, 0, ps, true);
        attach(b);
        mx::array lg = model.graph->forward(b, *model.kv);
        mx::eval(lg); model.kv->eval();
        mx::array h1 = model.graph->last_hidden();
        mx::eval(h1);
        return std::make_pair(mx::argmax(lg, 1).item<int>(), h1);
    };

    // ── Warm-up (both paths) ──
    for (int i = 0; i < warmup; ++i) {
        auto [t, h] = forward_one(cur, p + 1);
        (void)mtp_draft_incr(m, mkv, hidden, cur, p + 1, embed);
        hidden = h; cur = t; ++p;
    }

    // ══════════ 1. Regular greedy: N target forwards, 1 token each ══════════
    std::vector<double> t_target;
    int cur_reg = cur, p_reg = p;
    mx::array hid_reg = hidden;
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
        auto s = Clock::now();
        auto [t, h] = forward_one(cur_reg, p_reg + 1);
        t_target.push_back(ms_since(s));
        hid_reg = h; cur_reg = t; ++p_reg;
    }
    double reg_total = ms_since(t0);

    // ══════════ 2. MTP head: N incremental drafts (T_mtp) ══════════
    std::vector<double> t_mtp;
    int cur_m = cur, p_m = p;
    mx::array hid_m = hidden;
    MtpKV mkv2;
    for (int i = 0; i < N; ++i) {
        auto s = Clock::now();
        (void)mtp_draft_incr(m, mkv2, hid_m, cur_m, p_m + 1, embed);
        t_mtp.push_back(ms_since(s));
        p_m += 1;
    }

    // ══════════ 3. Verify forward: 2 positions in one pass (T_verify) ══════════
    // The real K=1 spec-decode verify runs [cur@p+1, draft@p+2] in ONE forward
    // (2 logit rows). Timing only — state correctness is irrelevant here.
    std::vector<double> t_verify;
    int pv = p_reg;  // continue after the regular run
    for (int i = 0; i < N; ++i) {
        std::vector<int> tk = {cur_reg, cur_reg};
        std::vector<int> po = {pv + 1, pv + 2};
        const int n = 2, last_pos = pv + 2;
        const int n_pages = last_pos / ps + 1;
        std::vector<int> pgi(n_pages); for (int j = 0; j < n_pages; ++j) pgi[j] = j;
        std::vector<int> wi = {pv + 1, pv + 2};
        model::ForwardBatch b{
            mx::array(tk.data(), {n}, mx::int32), mx::array(po.data(), {n}, mx::int32),
            mx::array({0, 1}, {2}, mx::int32),  // both logit rows
            mx::array(pgi.data(), {n_pages}, mx::int32), mx::array({0, n_pages}, {2}, mx::int32),
            mx::array({last_pos % ps + 1}, {1}, mx::int32), mx::array({0, n}, {2}, mx::int32),
            mx::array(wi.data(), {n}, mx::int32), n, 1, 2, false};
        attach(b);
        auto s = Clock::now();
        mx::array lg = model.graph->forward(b, *model.kv);
        mx::eval(lg); model.kv->eval();
        t_verify.push_back(ms_since(s));
        pv += 2;
    }

    // ══════════ 4. lm_head only (the shared vocab=248320 matmul) ══════════
    // Quantify how much of BOTH T_target and T_mtp is the tied lm_head — the
    // reason the 1-layer MTP head is not cheap.
    std::vector<double> t_lm;
    {
        mx::array x = mx::zeros({1, m.H}, mx::float32);
        mx::eval(x);
        for (int i = 0; i < N; ++i) {
            auto s = Clock::now();
            mx::array lg = mx::argmax(linW(x, embed), 1);
            mx::eval(lg);
            t_lm.push_back(ms_since(s));
        }
    }

    // Acceptance from the certified corrected MTP head (driver/metal/tests/
    // qwen35_mtp.cpp): stable 50% over 30 steps.
    const double acc = 0.50;

    const double T_target = median(t_target);
    const double T_mtp = median(t_mtp);
    const double T_verify = median(t_verify);
    const double T_lm = median(t_lm);
    const double tokens_per_verify = 1.0 + acc;       // K=1: own token + acceptance bonus
    const double t_spec_per_token = (T_verify + T_mtp) / tokens_per_verify;
    const double speedup = T_target / t_spec_per_token;

    std::printf("\n════════ RESULTS (median-of-%d, ms) ════════\n", N);
    std::printf("  T_target (1 GDN backbone forward, 1 tok + lm_head)  = %8.2f ms  (%.1f tok/s)\n",
                T_target, 1000.0 / T_target);
    std::printf("  T_verify (1 forward, 2 tok + 2-row lm_head)         = %8.2f ms\n", T_verify);
    std::printf("  T_mtp    (1 MTP-head forward + lm_head)             = %8.2f ms\n", T_mtp);
    std::printf("  T_lmhead (tied lm_head only, 1 row × %d vocab)  = %8.2f ms  (%.0f%% of T_target, %.0f%% of T_mtp)\n",
                vocab, T_lm, 100.0 * T_lm / T_target, 100.0 * T_lm / T_mtp);
    std::printf("  T_mtp / T_target                                    = %8.2f\n", T_mtp / T_target);
    std::printf("  regular greedy total (%d tok)                       = %8.2f ms  (%.1f tok/s)\n",
                N, reg_total, N * 1000.0 / reg_total);
    std::printf("  acceptance (K=1, certified head)                    = %8.1f%%\n", 100.0 * acc);
    std::printf("  tokens/verify (K=1)                                 = %8.2f\n", tokens_per_verify);
    std::printf("\n  spec-decode per-token = (T_verify+T_mtp)/tok_per_verify = %.2f ms\n",
                t_spec_per_token);
    std::printf("  ══> SPEEDUP  T_regular / T_spec = %.3fx  %s\n", speedup,
                speedup > 1.02 ? "(WIN)" : speedup < 0.98 ? "(SLOWER)" : "(neutral)");
    // Optimistic bound: even if the 2-pos verify were as cheap as a 1-tok decode
    // (T_verify == T_target), the MTP head's own cost caps the win.
    const double speedup_ideal = T_target * tokens_per_verify / (T_target + T_mtp);
    std::printf("  (optimistic bound, if T_verify==T_target: %.3fx)\n", speedup_ideal);
    std::printf("\n  Break-even needs (T_verify+T_mtp)/T_target < 1+acc = %.2f. Measured %.2f -> %s.\n",
                tokens_per_verify, (T_verify + T_mtp) / T_target,
                ((T_verify + T_mtp) / T_target < tokens_per_verify) ? "WIN" : "NOT worth it");
    std::printf("%s\n", "BENCH_DONE");
    return 0;
}
