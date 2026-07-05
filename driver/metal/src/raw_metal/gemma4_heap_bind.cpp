// gemma4_heap_bind.cpp — gemma4 weight-name registry + Metal staging/bind + const params.
//
// (1) Pure registry: maps each Kernel dispatch -> its load-once HF tensors, and the unique
//     stage list (tied lm_head once, non-shared k/v/k_norm only).
// (2) Staging: heap_alloc + memcpy every weight; alloc the per-(non-shared)-layer KV cache
//     and the IO scalars / logits buffers.
// (3) Per-ordinal weight/KV/IO bind + the const geometry params (gemma4_consts).

#include "gemma4_heap_bind.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "safetensors_view.hpp"

namespace pie_metal_driver::raw_metal::gemma4 {

// ── tensor-name helpers ──────────────────────────────────────────────────────
std::string gemma4_layer_prefix(int layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}
std::vector<std::string> quant_triple(const std::string& base) {
    return {base + ".weight", base + ".scales", base + ".biases"};
}

namespace {

constexpr const char* kLmHead          = "lm_head";  // tied (embed + logits)
constexpr const char* kFinalNorm       = "model.language_model.norm.weight";
constexpr const char* kPleTokenTable   = "model.language_model.embed_tokens_per_layer";
constexpr const char* kPleModelProj    = "model.language_model.per_layer_model_projection";
constexpr const char* kPleProjNorm     = "model.language_model.per_layer_projection_norm.weight";

// push a 4-bit quant triple to W/Scales/Biases (0/1/2) — shared by Embed/Qmv/EmbedScaled.
void push_quant(std::vector<WeightBind>& out, const std::string& base) {
    out.push_back({0, base + ".weight"});
    out.push_back({1, base + ".scales"});
    out.push_back({2, base + ".biases"});
}

}  // namespace

std::vector<WeightBind> gemma4_weight_binds(Kernel kind, int layer, const Gemma4Geometry& g) {
    (void)g;
    std::vector<WeightBind> w;
    const std::string p = layer >= 0 ? gemma4_layer_prefix(layer) : std::string();

    switch (kind) {
        // ── layer-less PLE precompute + tail ──
        case Kernel::EmbedGather:  push_quant(w, kLmHead);        break;  // tied, * sqrt(hidden)
        case Kernel::PleTokenGather: push_quant(w, kPleTokenTable); break;
        case Kernel::PleProjGemv:  push_quant(w, kPleModelProj);  break;
        case Kernel::PleProjNorm:
            w.push_back({static_cast<uint8_t>(bind::Rms::W), kPleProjNorm});
            break;
        case Kernel::PleCombine:   break;  // pointwise
        case Kernel::FinalRms:
            w.push_back({static_cast<uint8_t>(bind::Rms::W), kFinalNorm});
            break;
        case Kernel::LmHead:       push_quant(w, kLmHead);        break;  // tied
        case Kernel::FinalSoftcap: case Kernel::Argmax: break;

        // ── attention ──
        case Kernel::AttnNorm:
            w.push_back({static_cast<uint8_t>(bind::Rms::W), p + "input_layernorm.weight"});
            break;
        case Kernel::QmvQ: push_quant(w, p + "self_attn.q_proj"); break;
        case Kernel::QmvK: push_quant(w, p + "self_attn.k_proj"); break;
        case Kernel::QmvV: push_quant(w, p + "self_attn.v_proj"); break;
        case Kernel::QmvO: push_quant(w, p + "self_attn.o_proj"); break;
        case Kernel::QNorm:
            w.push_back({static_cast<uint8_t>(bind::Rms::W), p + "self_attn.q_norm.weight"});
            break;
        case Kernel::KNorm:
            w.push_back({static_cast<uint8_t>(bind::Rms::W), p + "self_attn.k_norm.weight"});
            break;
        case Kernel::VNorm: break;  // weightless
        case Kernel::RopeQ: case Kernel::RopeK: case Kernel::KvAppend: case Kernel::Sdpa: break;
        case Kernel::PostAttnNorm:
            w.push_back({static_cast<uint8_t>(bind::Rms::W),
                         p + "post_attention_layernorm.weight"});
            break;
        case Kernel::AttnResidual: break;

        // ── FFN (GeGLU-tanh) ──
        case Kernel::FfnNorm:
            w.push_back({static_cast<uint8_t>(bind::Rms::W),
                         p + "pre_feedforward_layernorm.weight"});
            break;
        case Kernel::QmvGate: push_quant(w, p + "mlp.gate_proj"); break;
        case Kernel::QmvUp:   push_quant(w, p + "mlp.up_proj");   break;
        case Kernel::GegluTanh: break;
        case Kernel::QmvDown: push_quant(w, p + "mlp.down_proj"); break;
        case Kernel::PostFfnNorm:
            w.push_back({static_cast<uint8_t>(bind::Rms::W),
                         p + "post_feedforward_layernorm.weight"});
            break;
        case Kernel::FfnResidual: break;

        // ── PLE residual + layer scalar ──
        case Kernel::PleGateGemv:      push_quant(w, p + "per_layer_input_gate"); break;
        case Kernel::PleGeglu:         break;
        case Kernel::PleProjLayerGemv: push_quant(w, p + "per_layer_projection"); break;
        case Kernel::PleNorm:
            w.push_back({static_cast<uint8_t>(bind::Rms::W),
                         p + "post_per_layer_input_norm.weight"});
            break;
        case Kernel::PleResidual: break;
        case Kernel::LayerScalar:
            w.push_back({static_cast<uint8_t>(bind::LayerScalar::Scalar), p + "layer_scalar"});
            break;
    }
    return w;
}

std::vector<std::string> gemma4_weight_tensors(const Gemma4Geometry& g) {
    std::vector<std::string> t;
    // tied lm_head bundle (EmbedGather + LmHead) — staged once.
    for (auto& n : quant_triple(kLmHead)) t.push_back(n);
    // PLE precompute tables + final norm.
    for (auto& n : quant_triple(kPleTokenTable)) t.push_back(n);
    for (auto& n : quant_triple(kPleModelProj)) t.push_back(n);
    t.push_back(kPleProjNorm);
    t.push_back(kFinalNorm);

    for (int L = 0; L < g.n_layers; ++L) {
        const std::string p = gemma4_layer_prefix(L);
        t.push_back(p + "input_layernorm.weight");
        t.push_back(p + "post_attention_layernorm.weight");
        t.push_back(p + "pre_feedforward_layernorm.weight");
        t.push_back(p + "post_feedforward_layernorm.weight");
        t.push_back(p + "post_per_layer_input_norm.weight");
        t.push_back(p + "layer_scalar");

        for (const char* proj : {"q_proj", "o_proj"})
            for (auto& n : quant_triple(p + "self_attn." + proj)) t.push_back(n);
        t.push_back(p + "self_attn.q_norm.weight");
        // k/v_proj + k_norm exist for all layers but are computed (and thus staged) only on
        // the non-shared layers; the shared copies are vestigial.
        if (!g.is_kv_shared(L)) {
            for (const char* proj : {"k_proj", "v_proj"})
                for (auto& n : quant_triple(p + "self_attn." + proj)) t.push_back(n);
            t.push_back(p + "self_attn.k_norm.weight");
        }

        for (const char* proj : {"gate_proj", "up_proj", "down_proj"})
            for (auto& n : quant_triple(p + "mlp." + proj)) t.push_back(n);
        for (auto& n : quant_triple(p + "per_layer_input_gate")) t.push_back(n);
        for (auto& n : quant_triple(p + "per_layer_projection")) t.push_back(n);
    }
    return t;
}

// ── staging ──────────────────────────────────────────────────────────────────
namespace {

SlotHandle stage_tensor(RawMetalContext& ctx, const RawTensor& rt) {
    SlotHandle s = ctx.heap_alloc(rt.nbytes);
    if (!s.valid()) throw std::runtime_error("gemma4 heap_alloc failed (heap too small)");
    std::memcpy(s.contents(), rt.data, rt.nbytes);
    return s;
}
SlotHandle alloc_zeroed(RawMetalContext& ctx, size_t nbytes) {
    SlotHandle s = ctx.heap_alloc(nbytes);
    if (!s.valid()) throw std::runtime_error("gemma4 heap_alloc failed");
    std::memset(s.contents(), 0, nbytes);
    return s;
}

}  // namespace

BoundGemma4 stage_gemma4_weights(RawMetalContext& ctx, const SafetensorsView& view,
                                 const Gemma4Geometry& g, int max_ctx) {
    BoundGemma4 b;
    b.kv.resize(g.n_layers);

    for (const auto& name : gemma4_weight_tensors(g))
        b.weights.emplace(name, stage_tensor(ctx, view.get(name)));

    // KV cache: per non-shared layer (L0..14), k/v each [n_kv_heads, max_ctx, head_dim_at].
    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_kv_shared(L)) continue;
        const size_t one = size_t(g.n_kv_heads) * size_t(max_ctx) * g.head_dim_at(L) * 2;  // bf16
        b.kv[L].k_pages = alloc_zeroed(ctx, one);
        b.kv[L].v_pages = alloc_zeroed(ctx, one);
    }

    b.io_token    = alloc_zeroed(ctx, 4);
    b.io_position = alloc_zeroed(ctx, 4);
    b.io_seqlen   = alloc_zeroed(ctx, 4);
    b.logits        = alloc_zeroed(ctx, size_t(g.vocab) * 2);  // bf16
    b.logits_capped = alloc_zeroed(ctx, size_t(g.vocab) * 2);  // bf16
    return b;
}

// ── per-ordinal weight / KV / IO bind ────────────────────────────────────────
void bind_gemma4_weights(RawMetalContext& ctx, const BoundGemma4& b,
                         const std::vector<Gemma4Dispatch>& dag, const Gemma4Geometry& g) {
    for (const auto& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;

        for (const auto& wb : gemma4_weight_binds(d.kind, L, g)) {
            auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end())
                throw std::runtime_error("gemma4 bind: unstaged weight " + wb.tensor);
            ctx.arg_bind_ordinal(ord, wb.bind_index, it->second);
        }

        switch (d.kind) {
            case Kernel::EmbedGather:
            case Kernel::PleTokenGather:
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::EmbedScaled::TokenId, b.io_token);
                break;
            case Kernel::RopeQ:
            case Kernel::RopeK:
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::Rope::Position, b.io_position);
                break;
            case Kernel::KvAppend: {
                const auto& kv = b.kv[L];
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::KvAppend::KPages, kv.k_pages);
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::KvAppend::VPages, kv.v_pages);
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::KvAppend::PositionPtr, b.io_position);
                break;
            }
            case Kernel::Sdpa: {
                const int src = g.kv_source(L);  // self if non-shared
                const auto& kv = b.kv[src];
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::Sdpa::K, kv.k_pages);
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::Sdpa::V, kv.v_pages);
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::Sdpa::N, b.io_seqlen);
                break;
            }
            case Kernel::LmHead:
                ctx.arg_bind_ordinal(ord, (uint8_t)bind::Qmv::Out, b.logits);
                break;
            default:
                break;
        }
    }
}

}  // namespace pie_metal_driver::raw_metal::gemma4
