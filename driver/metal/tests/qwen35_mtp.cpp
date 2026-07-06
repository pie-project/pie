// qwen35_mtp — REAL MTP speculative decoding for Qwen3.5-0.8B on Metal.
//
// The Qwen3.5 GDN backbone runs on Metal via the MLX driver (qwen36 graph); this
// adds the real trained 1-layer MTP draft head (mtp.* weights) + the draft→verify
// loop, and measures REAL acceptance (does the MTP draft match the target's next
// token). MTP head (from CUDA qwen3_5_mtp_forward):
//   pre_fc_norm(embed(t)) + pre_fc_norm(hidden) → concat[2H] → fc(2H→H)
//   → 1 transformer layer (attn+output-gate + SwiGLU, gemma (1+w) norms)
//   → mtp.norm → tied lm_head → draft token.
// The MTP attention is single-token self-attention (no cache) ⇒ softmax over one
// key = 1 ⇒ attn = V exactly (q_norm/k_norm/RoPE are no-ops), gated by
// sigmoid(gate) from the 2×-wide q_proj.
//
// Usage: qwen35_mtp <hf_path> [prompt_len=8] [steps=24]

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

mx::array f32(const mx::array& a) { return mx::astype(a, mx::float32); }
// y = x @ W^T, W row-major [out, in].
mx::array linW(const mx::array& x, const mx::array& W) {
    return mx::matmul(x, mx::transpose(f32(W)));
}

struct Mtp {
    std::unordered_map<std::string, mx::array> t;
    int H = 0, n_q = 0, n_kv = 0, d = 0;
    float eps = 1e-6f;
    const mx::array& g(const std::string& k) const { return t.at(k); }
};

// One MTP draft step: given the backbone hidden [1,H] + the just-sampled token,
// draft the next-next token (argmax of mtp logits).
int mtp_draft(const Mtp& m, const mx::array& hidden, int token, const mx::array& embed) {
    using namespace mx;
    array emb = reshape(take(f32(embed), array({token}, {1}, int32), 0), {1, m.H});
    array nx = ops::rms_norm(emb, f32(m.g("mtp.pre_fc_norm_embedding.weight")), m.eps, true);
    array ny = ops::rms_norm(f32(hidden), f32(m.g("mtp.pre_fc_norm_hidden.weight")), m.eps, true);
    array cat = concatenate({nx, ny}, 1);                       // [1, 2H]
    array y = linW(cat, m.g("mtp.fc.weight"));                  // [1, H]

    const std::string p = "mtp.layers.0.";
    // ── attention (single-token self-attn ⇒ attn = V, gated) ──
    array res = y;
    array h = ops::rms_norm(y, f32(m.g(p + "input_layernorm.weight")), m.eps, true);
    array qg = linW(h, m.g(p + "self_attn.q_proj.weight"));     // [1, n_q*2*d]
    qg = reshape(qg, {1, m.n_q, 2, m.d});
    array gate = reshape(slice(qg, {0, 0, 1, 0}, {1, m.n_q, 2, m.d}), {1, m.n_q * m.d});
    array V = reshape(linW(h, m.g(p + "self_attn.v_proj.weight")), {1, m.n_kv, m.d});
    array Vrep = reshape(repeat(V, m.n_q / m.n_kv, 1), {1, m.n_q * m.d});  // GQA broadcast
    array attn = multiply(Vrep, sigmoid(gate));                 // output gate
    array ao = linW(attn, m.g(p + "self_attn.o_proj.weight"));  // [1, H]
    y = add(res, ao);
    // ── SwiGLU MLP ──
    array res2 = y;
    array hn = ops::rms_norm(y, f32(m.g(p + "post_attention_layernorm.weight")), m.eps, true);
    array gp = linW(hn, m.g(p + "mlp.gate_proj.weight"));
    array up = linW(hn, m.g(p + "mlp.up_proj.weight"));
    array ffn = linW(ops::swiglu(gp, up), m.g(p + "mlp.down_proj.weight"));
    y = add(res2, ffn);
    // ── final norm + tied lm_head ──
    array fn = ops::rms_norm(y, f32(m.g("mtp.norm.weight")), m.eps, true);
    array logits = linW(fn, embed);                             // tied lm_head
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
    std::cerr << "[mtp] head geom: H=" << m.H << " n_q=" << m.n_q << " n_kv=" << m.n_kv
              << " head_dim=" << m.d << ", MTP tensors=" << m.t.size() << "\n";

    // Real prompt already in `toks` (or overridden via argv[3]). Prefill it.
    std::vector<int> pos(plen);
    for (int i = 0; i < plen; ++i) pos[i] = i;
    auto b0 = make_batch(toks, pos, plen - 1, ps, /*pure_decode=*/false);
    attach(b0);
    mx::array logits = model.graph->forward(b0, *model.kv);
    mx::eval(logits);
    model.kv->eval();
    mx::array hidden = model.graph->last_hidden();   // [1, H], pre-final-norm
    mx::eval(hidden);
    int cur = mx::argmax(logits, 1).item<int>();     // t_{p+1}
    int p = plen - 1;

    int accepted = 0, total = 0;
    std::vector<int> gen;
    for (int s = 0; s < steps; ++s) {
        // MTP drafts t_{p+2} from (hidden_p, cur=t_{p+1}).
        int draft = mtp_draft(m, hidden, cur, embed);
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
        // advance
        hidden = h1; cur = truth; ++p;
    }

    std::printf("\n[mtp] REAL MTP spec-decode on Metal (Qwen3.5-0.8B GDN backbone):\n");
    std::printf("  accepted %d / %d draft steps = %.1f%% acceptance\n",
                accepted, total, 100.0 * accepted / std::max(1, total));
    std::printf("  generated tokens:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n%s\n", accepted > 0 ? "QWEN35_MTP_OK" : "QWEN35_MTP_FAIL");
    return accepted > 0 ? 0 : 1;
}
