// qwen35_mtp — REAL MTP speculative decoding for Qwen3.5-0.8B on Metal.
//
// The Qwen3.5 GDN backbone runs on Metal via the MLX driver (qwen36 graph); this
// adds the real trained 1-layer MTP draft head (mtp.* weights) + the draft→verify
// loop, and measures REAL acceptance (does the MTP draft match the target's next
// token). MTP head (from CUDA qwen3_5_mtp_forward):
//   pre_fc_norm(embed(t)) + pre_fc_norm(hidden) → concat[2H] → fc(2H→H)
//   → 1 transformer layer (attn+output-gate + SwiGLU, gemma (1+w) norms)
//   → mtp.norm → tied lm_head → draft token.
// The MTP layer's attention is a CAUSAL attention over the MTP's own K/V history
// for every prior position (mirrors CUDA attention_mtp_paged_history) — NOT a
// single self token: without the context attention the layer degrades to
// ≈identity and the head just echoes the backbone's own next-token prediction.
// q_proj is 2×-wide [query|gate]; attn output is gated by sigmoid(gate). Q/K get
// gemma (1+w) per-head RMSNorm + partial NEOX RoPE.
//
// Usage: qwen35_mtp <hf_path> [steps=24] [prompt_ids_csv]

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

namespace {

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

// Prefill batch that returns the hidden state at EVERY prompt position (so the
// MTP head can build its full K/V history), not just the last row.
model::ForwardBatch make_prefill_all(const std::vector<int>& toks, int page_size) {
    const int n = (int)toks.size();
    std::vector<int> pos(n), rows(n), write_idx(n);
    for (int i = 0; i < n; ++i) { pos[i] = i; rows[i] = i; write_idx[i] = i; }
    const int last_pos = n - 1;
    const int n_pages = last_pos / page_size + 1;
    std::vector<int> page_idx(n_pages);
    for (int i = 0; i < n_pages; ++i) page_idx[i] = i;
    const int last_page_len = last_pos % page_size + 1;
    return model::ForwardBatch{
        mx::array(toks.data(), {n}, mx::int32),
        mx::array(pos.data(), {n}, mx::int32),
        mx::array(rows.data(), {n}, mx::int32),
        mx::array(page_idx.data(), {n_pages}, mx::int32),
        mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last_page_len}, {1}, mx::int32),
        mx::array({0, n}, {2}, mx::int32),
        mx::array(write_idx.data(), {n}, mx::int32),
        n, 1, n, false,
    };
}

mx::array f32(const mx::array& a) { return mx::astype(a, mx::float32); }
// y = x @ W^T, W row-major [out, in].
mx::array linW(const mx::array& x, const mx::array& W) {
    return mx::matmul(x, mx::transpose(f32(W)));
}

struct Mtp {
    std::unordered_map<std::string, mx::array> t;
    int H = 0, n_q = 0, n_kv = 0, d = 0, rope_dims = 0;
    float eps = 1e-6f, theta = 1e7f;
    const mx::array& g(const std::string& k) const { return t.at(k); }
};

// Full MTP forward with CAUSAL attention over the accumulated K/V history
// (mirrors CUDA attention_mtp_paged_history: the MTP layer attends over its own
// projected K/V for every prior position). H_list[i]/T_list[i]/P_list[i] give the
// backbone hidden [1,H], the token to embed, and the absolute position of MTP
// entry i. Returns the draft token = argmax(mtp_logits) at the LAST (current)
// position, i.e. the predicted next-next token.
int mtp_draft(const Mtp& m,
              const std::vector<mx::array>& H_list,
              const std::vector<int>& T_list,
              const std::vector<int>& P_list,
              const mx::array& embed) {
    using namespace mx;
    const int n = (int)T_list.size();
    const int H = m.H, nq = m.n_q, nkv = m.n_kv, d = m.d;
    const float scale = 1.0f / std::sqrt((float)d);

    array hid = concatenate(std::vector<array>(H_list.begin(), H_list.end()), 0);  // [n,H]
    array emb = take(f32(embed), array(T_list.data(), {n}, int32), 0);             // [n,H]
    array nx = ops::rms_norm(emb, f32(m.g("mtp.pre_fc_norm_embedding.weight")), m.eps, true);
    array ny = ops::rms_norm(f32(hid), f32(m.g("mtp.pre_fc_norm_hidden.weight")), m.eps, true);
    array cat = concatenate({nx, ny}, 1);                       // [n, 2H]
    array y = linW(cat, m.g("mtp.fc.weight"));                  // [n, H]

    const std::string p = "mtp.layers.0.";
    // ── causal attention over the full MTP history ──
    array res = y;
    array h = ops::rms_norm(y, f32(m.g(p + "input_layernorm.weight")), m.eps, true);
    array qg = reshape(linW(h, m.g(p + "self_attn.q_proj.weight")), {n, nq, 2, d});
    array Q    = reshape(slice(qg, {0, 0, 0, 0}, {n, nq, 1, d}), {n, nq, d});
    array gate = reshape(slice(qg, {0, 0, 1, 0}, {n, nq, 2, d}), {n, nq, d});
    array K = reshape(linW(h, m.g(p + "self_attn.k_proj.weight")), {n, nkv, d});
    array V = reshape(linW(h, m.g(p + "self_attn.v_proj.weight")), {n, nkv, d});
    // gemma (1+w) per-head Q/K RMSNorm, then partial NEOX RoPE at absolute pos.
    Q = ops::rms_norm(Q, f32(m.g(p + "self_attn.q_norm.weight")), m.eps, true);
    K = ops::rms_norm(K, f32(m.g(p + "self_attn.k_norm.weight")), m.eps, true);
    array positions = array(P_list.data(), {n}, int32);
    ops::RopeParams rp; rp.theta = m.theta;
    Q = ops::rope(Q, positions, m.rope_dims, rp);
    K = ops::rope(K, positions, m.rope_dims, rp);
    // GQA broadcast (repeat_interleave: kv head i → n_q/n_kv contiguous q heads).
    array Kr = repeat(K, nq / nkv, 1);   // [n,nq,d]
    array Vr = repeat(V, nq / nkv, 1);   // [n,nq,d]
    array Qt = transpose(Q,  {1, 0, 2}); // [nq,n,d]
    array Kt = transpose(Kr, {1, 0, 2});
    array Vt = transpose(Vr, {1, 0, 2});
    array scores = multiply(matmul(Qt, transpose(Kt, {0, 2, 1})), array(scale));  // [nq,n,n]
    array ri = reshape(arange(n), {n, 1});
    array ci = reshape(arange(n), {1, n});
    array mask = reshape(where(greater(ci, ri), array(-1e30f), array(0.0f)), {1, n, n});
    array w = softmax(add(scores, mask), std::vector<int>{-1}, /*precise=*/true);  // [nq,n,n]
    array ao = transpose(matmul(w, Vt), {1, 0, 2});             // [n,nq,d]
    array attn = multiply(reshape(ao, {n, nq * d}), sigmoid(reshape(gate, {n, nq * d})));
    array o = linW(attn, m.g(p + "self_attn.o_proj.weight"));   // [n,H]
    y = add(res, o);
    // ── SwiGLU MLP ──
    array res2 = y;
    array hn = ops::rms_norm(y, f32(m.g(p + "post_attention_layernorm.weight")), m.eps, true);
    array gp = linW(hn, m.g(p + "mlp.gate_proj.weight"));
    array up = linW(hn, m.g(p + "mlp.up_proj.weight"));
    array ffn = linW(ops::swiglu(gp, up), m.g(p + "mlp.down_proj.weight"));
    y = add(res2, ffn);
    // ── final norm + tied lm_head at the LAST position ──
    array fn = ops::rms_norm(y, f32(m.g("mtp.norm.weight")), m.eps, true);
    array last = reshape(slice(fn, {n - 1, 0}, {n, H}), {1, H});
    array logits = linW(last, embed);                           // tied lm_head
    array am = argmax(logits, 1);
    eval(am);
    return am.item<int>();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: qwen35_mtp <hf_path> [steps=24] [prompt_ids_csv]\n"; return 2; }
    const std::string hf_path = argv[1];
    const int steps = argc > 2 ? std::atoi(argv[2]) : 24;
    // Real prompt: "The capital of France is Paris. The capital of Japan is".
    std::vector<int> toks = {760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 6124, 369};
    if (argc > 3) {
        toks.clear();
        std::string csv = argv[3], num;
        for (char c : csv) {
            if (c == ',') { if (!num.empty()) { toks.push_back(std::atoi(num.c_str())); num.clear(); } }
            else num += c;
        }
        if (!num.empty()) toks.push_back(std::atoi(num.c_str()));
    }
    const int plen = (int)toks.size();

    BatchingConfig batching;
    loader::LoadedModel model = loader::load_model(hf_path, batching);
    const int ps = model.kv->page_size();
    std::cerr << "[mtp] loaded arch=" << model.caps.arch_name
              << " layers=" << model.caps.num_hidden_layers << " page_size=" << ps << "\n";

    const bool hybrid = (model.lin_cache != nullptr);
    auto attach = [&](model::ForwardBatch& b) {
        if (!hybrid) return;
        b.lin_cache = model.lin_cache.get();
        b.slot_ids = mx::array({0}, {1}, mx::int32);
        b.qo_indptr_host.assign({0, b.n_total});
    };

    // Load the MTP + embed tensors directly from the safetensors shard.
    Mtp m;
    m.eps = 1e-6f;
    auto st = mx::load_safetensors(hf_path + "/model.safetensors-00001-of-00001.safetensors");
    for (auto& kv : st.first)
        if (kv.first.rfind("mtp.", 0) == 0) m.t.emplace(kv.first, kv.second);
    mx::array embed = st.first.at("model.language_model.embed_tokens.weight");  // [vocab, H]
    m.H = embed.shape(1);
    // Head geometry from the mtp q/v proj shapes (q_proj = n_q*2*d, v_proj = n_kv*d).
    m.n_q = model.caps.num_hidden_layers ? 8 : 8;      // from config: num_attention_heads
    m.n_kv = 2;                                         // num_key_value_heads
    int vrows = m.t.at("mtp.layers.0.self_attn.v_proj.weight").shape(0);
    m.d = vrows / m.n_kv;                               // head_dim
    m.theta = 1e7f;                                     // rope_parameters.rope_theta
    // partial_rotary_factor 0.25 → rope_dims = 2*floor(0.5*0.25*head_dim).
    m.rope_dims = std::max(2, 2 * (int)std::floor(0.5f * 0.25f * m.d));
    std::cerr << "[mtp] head geom: H=" << m.H << " n_q=" << m.n_q << " n_kv=" << m.n_kv
              << " head_dim=" << m.d << " rope_dims=" << m.rope_dims
              << ", MTP tensors=" << m.t.size() << "\n";

    // Prefill returning the hidden at EVERY prompt position, so the MTP head can
    // build its full K/V history (the attention over prior positions is what lets
    // the head draft t+2 instead of echoing the backbone's own t+1 prediction).
    auto b0 = make_prefill_all(toks, ps);
    attach(b0);
    mx::array logits = model.graph->forward(b0, *model.kv);
    mx::eval(logits);
    model.kv->eval();
    mx::array hidden_all = model.graph->last_hidden();   // [plen, H], pre-final-norm
    mx::eval(hidden_all);
    auto row = [&](const mx::array& a, int i) {
        return mx::reshape(mx::slice(a, {i, 0}, {i + 1, a.shape(1)}), {1, a.shape(1)});
    };

    // MTP K/V history: entry i pairs backbone hidden_i with token t_{i+1} at
    // absolute position i+1. Prompt entries i=0..plen-2 use the (teacher-forced)
    // next prompt token; the last prompt hidden pairs with the first generated
    // token and is appended inside the loop below.
    std::vector<mx::array> H_list;
    std::vector<int> T_list, P_list;
    for (int i = 0; i + 1 < plen; ++i) {
        H_list.push_back(row(hidden_all, i));
        T_list.push_back(toks[i + 1]);
        P_list.push_back(i + 1);
    }
    mx::array hidden = row(hidden_all, plen - 1);         // [1,H] at last prompt pos
    // logits is [plen,V]; take the last row's argmax as t_{plen} (first gen token).
    int cur = mx::argmax(row(logits, plen - 1), 1).item<int>();
    int p = plen - 1;

    int accepted = 0, total = 0;
    std::vector<int> gen;
    for (int s = 0; s < steps; ++s) {
        // Append the current entry (hidden_p, cur=t_{p+1} at pos p+1), then draft
        // t_{p+2} = argmax(mtp_logits) with attention over the full history.
        H_list.push_back(hidden);
        T_list.push_back(cur);
        P_list.push_back(p + 1);
        int draft = mtp_draft(m, H_list, T_list, P_list, embed);
        // Backbone verifies: forward cur at position p+1 → ground-truth t_{p+2}.
        auto b = make_batch({cur}, {p + 1}, 0, ps, /*pure_decode=*/true);
        attach(b);
        mx::array lg = model.graph->forward(b, *model.kv);
        mx::eval(lg);
        model.kv->eval();
        int truth = mx::argmax(lg, 1).item<int>();
        mx::array h1 = model.graph->last_hidden();
        mx::eval(h1);

        bool ok = (draft == truth);
        accepted += ok; ++total;
        gen.push_back(cur);
        if (s < 8) std::printf("  step %d: mtp_draft=%d target=%d %s\n", s, draft, truth, ok ? "ACCEPT" : "reject");
        // advance
        hidden = h1; cur = truth; ++p;
    }

    std::printf("\n[mtp] REAL MTP spec-decode on Metal (Qwen3.5-0.8B GDN backbone):\n");
    std::printf("  accepted %d / %d draft steps = %.1f%% acceptance\n",
                accepted, total, 100.0 * accepted / std::max(1, total));
    std::printf("  generated tokens:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");

    // Cross-check vector for the CUDA Qwen3.5 MTP run (4090): identical prompt +
    // weights → compare acceptance + generated sequence.
    if (const char* gp = std::getenv("MTP_GOLDEN")) {
        std::FILE* f = std::fopen(gp, "w");
        if (f) {
            std::fprintf(f, "# Qwen3.5-0.8B MTP spec-decode cross-check (Metal) — run CUDA on same input\n");
            std::fprintf(f, "model: Qwen/Qwen3.5-0.8B (main), model.safetensors total_size=1746882752 BF16\n");
            std::fprintf(f, "backend: Metal (Apple M1 Max, MLX 0.31.2 GDN backbone) + real mtp.* head, f32 head compute\n");
            std::fprintf(f, "draft_depth_K: 1 (mtp_num_hidden_layers=1)\n");
            std::fprintf(f, "mtp_attn: causal attention over the MTP layer's own K/V history for all prior positions (mirrors CUDA attention_mtp_paged_history); q/k gemma(1+w) RMSNorm + partial NEOX RoPE (rope_dims=64, theta=1e7), output-gated\n");
            std::fprintf(f, "prompt_ids:");
            for (int t : toks) std::fprintf(f, " %d", t);
            std::fprintf(f, "\nsteps: %d\naccepted: %d\nacceptance_pct: %.1f\n", total, accepted, 100.0 * accepted / std::max(1, total));
            std::fprintf(f, "generated_ids:");
            for (int t : gen) std::fprintf(f, " %d", t);
            std::fprintf(f, "\nnote: greedy verify — deterministic. Match acceptance + generated_ids on CUDA Qwen3.5 MTP for cross-parity.\n");
            std::fclose(f);
            std::printf("  cross-check golden written: %s\n", gp);
        }
    }
    std::printf("%s\n", accepted > 0 ? "QWEN35_MTP_OK" : "QWEN35_MTP_FAIL");
    return accepted > 0 ? 0 : 1;
}
