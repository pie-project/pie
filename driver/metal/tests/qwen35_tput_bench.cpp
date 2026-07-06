// qwen35_tput_bench — high-concurrency THROUGHPUT benchmark: regular batched
// decode vs MTP (K=1) speculative decode on Metal (Qwen3.5-0.8B). The core
// honest question In Gim asks: under GPU saturation (many concurrent streams
// batched together), does spec-decode's extra per-token compute HELP or HURT
// aggregate tokens/sec?
//
// Concurrency is modeled by continuous batching (B concurrent streams = a forward
// over B tokens with B logit rows — the compute shape a server presents to the
// GPU). The backbone matmuls + lm_head dominate, so a B-token forward is a fair
// proxy for B concurrent 1-token decodes.
//
//   Regular(B):  1 forward of B  tokens (B  logit rows)          -> B    tokens
//   Spec(B):     1 forward of 2B tokens (2B logit rows, verify)  -> 1.5B tokens
//                + 1 MTP-head forward over B tokens
//   throughput = tokens / wall-clock.  As B grows the GPU saturates; if spec
//   never beats regular at saturation, spec HURTS throughput (it burns compute
//   the saturated GPU could spend serving more streams).
//
// Usage: qwen35_tput_bench <hf_path> [maxB=64] [reps=5]

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
mx::array f32(const mx::array& a) { return mx::astype(a, mx::float32); }
mx::array linW(const mx::array& x, const mx::array& W) { return mx::matmul(x, mx::transpose(f32(W))); }

struct Mtp {
    std::unordered_map<std::string, mx::array> t;
    int H = 0, n_q = 8, n_kv = 2, d = 0;
    float eps = 1e-6f;
    const mx::array& g(const std::string& k) const { return t.at(k); }
};

// Backbone forward over `n` tokens with `n` logit rows (the B-concurrent-decode
// compute proxy: n tokens through 24 GDN/attn layers + an n-row lm_head).
model::ForwardBatch batch_n(int n, int page_size) {
    std::vector<int> toks(n, 100), pos(n), rows(n), wi(n);
    for (int i = 0; i < n; ++i) { pos[i] = i; rows[i] = i; wi[i] = i; }
    const int last_pos = n - 1, n_pages = last_pos / page_size + 1;
    std::vector<int> pgi(n_pages);
    for (int i = 0; i < n_pages; ++i) pgi[i] = i;
    return model::ForwardBatch{
        mx::array(toks.data(), {n}, mx::int32), mx::array(pos.data(), {n}, mx::int32),
        mx::array(rows.data(), {n}, mx::int32),
        mx::array(pgi.data(), {n_pages}, mx::int32), mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last_pos % page_size + 1}, {1}, mx::int32), mx::array({0, n}, {2}, mx::int32),
        mx::array(wi.data(), {n}, mx::int32), n, 1, n, false};
}

// MTP head over `n` independent draft tokens (fc + 1 layer + n-row lm_head).
// Self-attn = V (per-stream history attention is negligible); timing proxy.
double time_mtp_batch(const Mtp& m, const mx::array& embed, int n, int reps) {
    using namespace mx;
    const int H = m.H, nq = m.n_q, nkv = m.n_kv, d = m.d;
    array hidden = zeros({n, H}, float32);
    std::vector<int> tk(n, 100);
    array tokarr = array(tk.data(), {n}, int32);
    const std::string p = "mtp.layers.0.";
    std::vector<double> ts;
    for (int r = 0; r < reps + 1; ++r) {
        auto s = Clock::now();
        array emb = take(f32(embed), tokarr, 0);
        array nx = ops::rms_norm(emb, f32(m.g("mtp.pre_fc_norm_embedding.weight")), m.eps, true);
        array ny = ops::rms_norm(hidden, f32(m.g("mtp.pre_fc_norm_hidden.weight")), m.eps, true);
        array y = linW(concatenate({nx, ny}, 1), m.g("mtp.fc.weight"));
        array h = ops::rms_norm(y, f32(m.g(p + "input_layernorm.weight")), m.eps, true);
        array qg = reshape(linW(h, m.g(p + "self_attn.q_proj.weight")), {n, nq, 2, d});
        array gate = reshape(slice(qg, {0, 0, 1, 0}, {n, nq, 2, d}), {n, nq * d});
        array V = reshape(linW(h, m.g(p + "self_attn.v_proj.weight")), {n, nkv, d});
        array Vr = reshape(repeat(V, nq / nkv, 1), {n, nq * d});
        array attn = multiply(Vr, sigmoid(gate));
        y = add(y, linW(attn, m.g(p + "self_attn.o_proj.weight")));
        array hn = ops::rms_norm(y, f32(m.g(p + "post_attention_layernorm.weight")), m.eps, true);
        array gp = linW(hn, m.g(p + "mlp.gate_proj.weight"));
        array up = linW(hn, m.g(p + "mlp.up_proj.weight"));
        y = add(y, linW(ops::swiglu(gp, up), m.g(p + "mlp.down_proj.weight")));
        array fn = ops::rms_norm(y, f32(m.g("mtp.norm.weight")), m.eps, true);
        array am = argmax(linW(fn, embed), 1);
        eval(am);
        if (r > 0) ts.push_back(ms_since(s));  // drop first (warm)
    }
    return median(ts);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: qwen35_tput_bench <hf_path> [maxB=64] [reps=5]\n"; return 2; }
    const std::string hf_path = argv[1];
    const int maxB = argc > 2 ? std::atoi(argv[2]) : 64;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

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
    const int vocab = embed.shape(0);
    const double acc = 0.50;  // certified corrected head

    std::printf("[tput] Qwen3.5-0.8B on Metal — vocab=%d, acc=%.0f%%, reps=%d\n", vocab, 100 * acc, reps);
    std::printf("[tput] concurrency = continuous batching (B streams = B-token forward)\n\n");

    auto time_fwd = [&](int n) {
        std::vector<double> ts;
        for (int r = 0; r < reps + 1; ++r) {
            auto b = batch_n(n, ps);
            attach(b);
            auto s = Clock::now();
            mx::array lg = model.graph->forward(b, *model.kv);
            mx::eval(lg); model.kv->eval();
            if (r > 0) ts.push_back(ms_since(s));
        }
        return median(ts);
    };

    std::printf("  %4s | %9s %10s | %9s %9s | %10s %10s | %8s\n",
                "B", "T_reg(B)", "reg tok/s", "T_ver2B", "T_mtp(B)", "spec ms", "spec tok/s", "spec/reg");
    std::printf("  %s\n", std::string(88, '-').c_str());

    std::vector<int> Bs;
    for (int b = 1; b <= maxB; b *= 2) Bs.push_back(b);
    double best_reg = 0, best_spec = 0;
    for (int B : Bs) {
        double T_reg = time_fwd(B);
        double T_ver = time_fwd(2 * B);
        double T_mtp = time_mtp_batch(m, embed, B, reps);
        double reg_tps = B * 1000.0 / T_reg;
        double spec_ms = T_ver + T_mtp;
        double spec_tps = (1.0 + acc) * B * 1000.0 / spec_ms;
        best_reg = std::max(best_reg, reg_tps);
        best_spec = std::max(best_spec, spec_tps);
        std::printf("  %4d | %9.2f %10.1f | %9.2f %9.2f | %10.2f %10.1f | %7.3fx\n",
                    B, T_reg, reg_tps, T_ver, T_mtp, spec_ms, spec_tps, spec_tps / reg_tps);
    }

    std::printf("\n  peak regular throughput = %.1f tok/s\n", best_reg);
    std::printf("  peak spec-decode throughput = %.1f tok/s\n", best_spec);
    std::printf("  ══> under saturation spec-decode is %.3fx regular throughput -> %s\n",
                best_spec / best_reg,
                best_spec > best_reg * 1.02 ? "HELPS" : best_spec < best_reg * 0.98 ? "HURTS" : "neutral");
    std::printf("BENCH_DONE\n");
    return 0;
}
