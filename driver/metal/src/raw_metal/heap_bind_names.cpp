// heap_bind_names.cpp — pure (no Metal) weight-name registry for the raw-Metal decode.
// Maps each Kernel dispatch -> its load-once weight tensors, and enumerates the complete
// unique tensor set to stage. Verified vs the real checkpoint by heap_bind_probe.

#include "heap_bind.hpp"

namespace pie_metal_driver::raw_metal {

std::string layer_prefix(int layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

std::string final_norm_name() { return "model.language_model.norm.weight"; }

std::vector<std::string> quant_triple(const std::string& base) {
    return {base + ".weight", base + ".scales", base + ".biases"};
}

namespace {

// The shared tied table (canonical stored side; embed aliases it). 4-bit g64.
constexpr const char* kLmHead = "lm_head";

// bind a 4-bit quant triple to a kernel that uses bind::Qmv W/Scales/Biases (0/1/2)
// or bind::Embed W/Scales/Biases (0/1/2) — both share the 0/1/2 packed-quant prefix.
void push_quant(std::vector<WeightBind>& out, const std::string& base) {
    out.push_back({0, base + ".weight"});
    out.push_back({1, base + ".scales"});
    out.push_back({2, base + ".biases"});
}

}  // namespace

std::vector<WeightBind> weight_binds(Kernel kind, int layer, const DecodeGeometry& g) {
    (void)g;
    std::vector<WeightBind> w;
    const std::string p = layer >= 0 ? layer_prefix(layer) : std::string();

    switch (kind) {
        // ── Singletons ──
        case Kernel::EmbedGather:  // tied lm_head bundle -> bind::Embed W/Scales/Biases
            push_quant(w, kLmHead);
            break;
        case Kernel::QmvLmHead:    // same shared bundle -> bind::Qmv W/Scales/Biases
            push_quant(w, kLmHead);
            break;
        case Kernel::FinalRms:     // bind::Rms W=1
            w.push_back({static_cast<uint8_t>(bind::Rms::W), final_norm_name()});
            break;
        case Kernel::Argmax:
            break;  // logits/next-token are IO

        // ── Norms (bind::Rms W=1) ──
        case Kernel::Rms:          // input_layernorm
            w.push_back({static_cast<uint8_t>(bind::Rms::W), p + "input_layernorm.weight"});
            break;
        case Kernel::FfnRms:       // post_attention_layernorm
            w.push_back({static_cast<uint8_t>(bind::Rms::W),
                         p + "post_attention_layernorm.weight"});
            break;
        case Kernel::QNorm:        // self_attn.q_norm
            w.push_back({static_cast<uint8_t>(bind::Rms::W), p + "self_attn.q_norm.weight"});
            break;
        case Kernel::KNorm:        // self_attn.k_norm
            w.push_back({static_cast<uint8_t>(bind::Rms::W), p + "self_attn.k_norm.weight"});
            break;

        // ── Full-attention 4-bit projections (bind::Qmv W/Scales/Biases) ──
        case Kernel::QmvQ: push_quant(w, p + "self_attn.q_proj"); break;
        case Kernel::QmvK: push_quant(w, p + "self_attn.k_proj"); break;
        case Kernel::QmvV: push_quant(w, p + "self_attn.v_proj"); break;
        case Kernel::QmvO: push_quant(w, p + "self_attn.o_proj"); break;

        // ── GDN in/out 4-bit projections ──
        case Kernel::QmvIn:  push_quant(w, p + "linear_attn.in_proj_qkv"); break;
        case Kernel::QmvInZ: push_quant(w, p + "linear_attn.in_proj_z");   break;
        case Kernel::QmvOut: push_quant(w, p + "linear_attn.out_proj");    break;

        // ── GDN dense bf16 a/b projections (bind::Dense W=0) ──
        case Kernel::GdnInA:
            w.push_back({static_cast<uint8_t>(bind::Dense::W), p + "linear_attn.in_proj_a.weight"});
            break;
        case Kernel::GdnInB:
            w.push_back({static_cast<uint8_t>(bind::Dense::W), p + "linear_attn.in_proj_b.weight"});
            break;

        // ── GDN fused core: conv1d.weight + A_log + dt_bias are load-once weights.
        //    ConvB has NO checkpoint tensor (conv1d is bias-less) -> zeroed slot (bound in
        //    bind_decode_dag). ConvState/RecurrentState/ConvStateOut are persistent STATE.
        case Kernel::GdnCore:
            w.push_back({static_cast<uint8_t>(bind::GdnCore::ConvW), p + "linear_attn.conv1d.weight"});
            w.push_back({static_cast<uint8_t>(bind::GdnCore::ALog),  p + "linear_attn.A_log"});
            w.push_back({static_cast<uint8_t>(bind::GdnCore::DtBias), p + "linear_attn.dt_bias"});
            break;

        // ── GDN gated-RMSNorm gate weight (bind::GatedRms W=2 = linear_attn.norm) ──
        case Kernel::GatedRms:
            w.push_back({static_cast<uint8_t>(bind::GatedRms::W), p + "linear_attn.norm.weight"});
            break;

        // ── Shared SwiGLU MLP 4-bit projections ──
        case Kernel::QmvGate: push_quant(w, p + "mlp.gate_proj"); break;
        case Kernel::QmvUp:   push_quant(w, p + "mlp.up_proj");   break;
        case Kernel::QmvDown: push_quant(w, p + "mlp.down_proj"); break;

        // ── Weight-less kinds (scratch / KV / IO / const only) ──
        case Kernel::QSplit:
        case Kernel::Rope:
        case Kernel::RopeK:
        case Kernel::KvAppend:
        case Kernel::Sdpa:
        case Kernel::AttnGate:
        case Kernel::SiluMul:
        case Kernel::Residual:
        case Kernel::LayerOut:
            break;
    }
    return w;
}

std::vector<std::string> decode_weight_tensors(const DecodeGeometry& g) {
    std::vector<std::string> t;
    // tied lm_head bundle (shared by EmbedGather + QmvLmHead) — staged ONCE.
    for (auto& n : quant_triple(kLmHead)) t.push_back(n);
    t.push_back(final_norm_name());

    for (int L = 0; L < g.n_layers; ++L) {
        const std::string p = layer_prefix(L);
        t.push_back(p + "input_layernorm.weight");
        t.push_back(p + "post_attention_layernorm.weight");

        if (DecodeGeometry::is_full_attn(L)) {
            for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"})
                for (auto& n : quant_triple(p + "self_attn." + proj)) t.push_back(n);
            t.push_back(p + "self_attn.q_norm.weight");
            t.push_back(p + "self_attn.k_norm.weight");
        } else {
            for (const char* proj : {"in_proj_qkv", "in_proj_z", "out_proj"})
                for (auto& n : quant_triple(p + "linear_attn." + proj)) t.push_back(n);
            t.push_back(p + "linear_attn.in_proj_a.weight");   // dense bf16
            t.push_back(p + "linear_attn.in_proj_b.weight");   // dense bf16
            t.push_back(p + "linear_attn.conv1d.weight");      // no bias
            t.push_back(p + "linear_attn.A_log");
            t.push_back(p + "linear_attn.dt_bias");
            t.push_back(p + "linear_attn.norm.weight");        // gate_norm
        }

        for (const char* proj : {"gate_proj", "up_proj", "down_proj"})
            for (auto& n : quant_triple(p + "mlp." + proj)) t.push_back(n);
    }
    return t;
}

}  // namespace pie_metal_driver::raw_metal
