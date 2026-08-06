#include "model/qwen3_5/qwen3_5_moe.hpp"
#include "kernels/swiglu.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/gated_delta_net.hpp"  // launch_bf16_to_fp32

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("qwen3_5_moe: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe(const LoadedModel& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}

// Bind this layer's routed experts, whichever form the contract published.
//
// A stacking contract publishes one fused `[E, ...]` slab per projection; a
// streaming contract publishes neither and declares a group instead. It is
// never both, so there is no second copy to pay for and no ambiguity here.
void bind_routed_experts(const LoadedModel& engine, const std::string& lp, int experts,
                         Qwen3_5MoeLayerWeights& Lw) {
    Lw.moe_gate_up_proj = maybe(engine, lp + "mlp.experts.gate_up_proj");
    Lw.moe_down_proj = maybe(engine, lp + "mlp.experts.down_proj");
    if (Lw.moe_gate_up_proj != nullptr && Lw.moe_down_proj != nullptr) {
        return;
    }
    GroupStreamCache* cache = engine.group_cache();
    const std::string group_name = lp + "mlp.experts";
    const std::size_t g =
        cache != nullptr ? cache->find_group(group_name) : GroupStreamCache::kNoGroup;
    if (g == GroupStreamCache::kNoGroup) {
        throw std::runtime_error(
            "qwen3_5_moe: layer has neither fused expert weights nor a '" +
            group_name + "' group");
    }
    if (cache->arity(g) != static_cast<std::uint32_t>(experts)) {
        throw std::runtime_error(
            "qwen3_5_moe: group '" + group_name + "' holds " +
            std::to_string(cache->arity(g)) + " experts but the config says " +
            std::to_string(experts));
    }
    Lw.expert_cache = cache;
    Lw.expert_group = g;
}

// The shared expert's scalar gate is a [1, H] projection. Packed onto the
// gate/up weight it is one extra row of a GEMM that already runs; left on
// its own it is a whole kernel (`sigmoid_dot_scalar_gate_add`, ~7 us per
// layer, 283 us per Qwen3.6 decode step) for a single dot product. The
// packed weight is the same size either way, so this is on by default.
// Fusing the shared expert's scalar gate row onto its gate/up projection
// saves one GEMM, but it makes the fused weight `2*I_shared + 1` rows --
// an ODD N, so both N and ldc break 16-byte alignment and cuBLAS drops off
// its aligned tensor-core kernels onto a much slower fallback. The cost is
// superlinear in M, which hides at small batches and is brutal at large
// ones: on Qwen3.6-35B-A3B tp2 the shared gate_up GEMM measured 1.41 ms at
// N=128 but 6.18 ms at N=256 (4.4x for 2x the rows), against 0.83 ms
// unfused, where N is 512. End to end that is +2.3% at 128 requests and
// +9.5% at 256; tp=1 is unchanged (1496 against 1502 tok/s).
//
// So the split path wins despite the extra launch, and is the default.
// Padding the fused weight up to a multiple of 8 would recover the saved
// launch as well -- the swiglu and scalar-gate kernels already take the
// row stride as an argument -- but is worth only ~0.3 ms more.
} // namespace

bool qwen35_fused_shared_scalar_gate_enabled() { return false; }

namespace {

// Qwen 3.5 / 3.6 ship as multimodal containers, so their text-tower
// weights live under `model.language_model.…`. Qwen3-MoE (Qwen3-30B-A3B)
// is a pure text model and uses `model.…` directly. Pick the prefix from
// what the engine actually loaded so a single bind covers both.
const char* select_prefix(const LoadedModel& e) {
    if (e.has("model.language_model.embed_tokens.weight")) {
        return "model.language_model.";
    }
    return "model.";
}

}  // namespace


bool qwen35_mtp_int8_lm_head_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_INT8_LM_HEAD");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool qwen35_moe_gate_up_swapped() { return true; }

Qwen3_5MoeWeights bind_qwen3_5_moe(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    const int L = cfg.num_hidden_layers;
    // Qwen3-MoE (Qwen3-30B-A3B) is full-attention only — its config has
    // no `layer_types` field and no shared expert. Qwen3.5 / 3.6-MoE
    // ship a hybrid linear-attn / full-attn layer schedule and an
    // always-on shared expert. We bind the same struct in both cases;
    // the per-arch flags below decide which sections to require.
    const bool is_qwen3_moe = (cfg.model_type == "qwen3_moe");
    const bool has_shared_expert = cfg.shared_expert_intermediate_size > 0;

    std::vector<std::string> synth_layer_types;
    const std::vector<std::string>* layer_types = &cfg.layer_types;
    if (is_qwen3_moe && cfg.layer_types.empty()) {
        synth_layer_types.assign(static_cast<std::size_t>(L), "full_attention");
        layer_types = &synth_layer_types;
    }
    if (layer_types->empty() ||
        static_cast<int>(layer_types->size()) != L) {
        throw std::runtime_error(
            "qwen3_5_moe: HfConfig.layer_types must match num_hidden_layers");
    }
    const bool has_linear_attn =
        std::any_of(layer_types->begin(), layer_types->end(),
                    [](const std::string& t) { return t == "linear_attention"; });
    if (has_linear_attn &&
        (cfg.linear_num_value_heads <= 0 || cfg.linear_num_key_heads <= 0
            || cfg.linear_key_head_dim <= 0 || cfg.linear_value_head_dim <= 0
            || cfg.linear_conv_kernel_dim <= 0)) {
        throw std::runtime_error("qwen3_5_moe: linear-attn dimensions are unset");
    }
    if (cfg.num_experts <= 0 || cfg.num_experts_per_tok <= 0) {
        throw std::runtime_error(
            "qwen3_5_moe: num_experts and num_experts_per_tok must be > 0");
    }
    if (cfg.moe_intermediate_size <= 0) {
        throw std::runtime_error(
            "qwen3_5_moe: moe_intermediate_size must be > 0");
    }

    Qwen3_5MoeWeights w;
    w.layers.resize(static_cast<std::size_t>(L));

    const std::string p = select_prefix(engine);

    w.embed      = &must(engine, p + "embed_tokens.weight");
    w.final_norm = &must(engine, p + "norm.weight");
    w.lm_head    = cfg.tie_word_embeddings
                       ? w.embed
                       : &must(engine, "lm_head.weight");

    // Stable storage for sliced linear-attn + routed-expert tensors.
    // Per layer we may push: 3 linear-attn slices (qkv, conv_w, conv_b),
    // 2 fused linear-attn projection tensors, and a fused shared-expert
    // gate/up tensor.

    int kv_slot = 0;
    for (int li = 0; li < L; ++li) {
        const std::string lp = p + "layers." + std::to_string(li) + ".";
        auto& Lw = w.layers[li];
        const auto& kind = (*layer_types)[li];

        Lw.attn_norm_pre = &must(engine, lp + "input_layernorm.weight");
        Lw.mlp_norm_pre  = &must(engine, lp + "post_attention_layernorm.weight");

        // Token-mixer weights: linear-attn or full-attn.
        if (kind == "linear_attention") {
            Lw.kind = Qwen3_5MoeLayerWeights::Kind::LinearAttn;
            const std::string la = lp + "linear_attn.";
            // Whichever layout the contract published. The fused switch makes
            // the join a load-time `Concat`, so the separate tensors are simply
            // absent -- there is never a moment when both are resident. That
            // is what retired the old objection to fusing qkv/z on
            // Qwen3.6-35B-A3B: the bind-time concatenation needed 1.4 GB of
            // duplicate weights, because the arena-backed sources reclaimed
            // nothing when erased. Joining b/a is still measured as a wash.
            Lw.la_in_proj_qkv = maybe(engine, la + "in_proj_qkv.weight");
            Lw.la_in_proj_z = maybe(engine, la + "in_proj_z.weight");
            Lw.la_in_proj_b = maybe(engine, la + "in_proj_b.weight");
            Lw.la_in_proj_a = maybe(engine, la + "in_proj_a.weight");
            // This rank's `[K/T | K/T | V/T]`: the contract states the
            // per-block shard, so the whole tensor is never resident here.
            Lw.la_conv1d_w = &must(engine, la + "conv1d.weight");
            Lw.la_conv1d_b = maybe(engine, la + "conv1d.bias");
            Lw.la_dt_bias     = &must(engine, la + "dt_bias");
            // fp32 by contract (`gdn_fp32_parameters`), so these are the
            // loaded bytes rather than a bind-time copy of them.
            Lw.la_A_log_fp32 =
                static_cast<const float*>(must(engine, la + "A_log").data());
            Lw.la_norm_w_fp32 =
                static_cast<const float*>(must(engine, la + "norm.weight").data());
            Lw.la_out_proj    = &must(engine, la + "out_proj.weight");
            Lw.kv_layer = -1;
        } else if (kind == "full_attention") {
            Lw.kind = Qwen3_5MoeLayerWeights::Kind::FullAttn;
            const std::string fa = lp + "self_attn.";
            Lw.fa_q_proj = &must(engine, fa + "q_proj.weight");
            Lw.fa_k_proj = &must(engine, fa + "k_proj.weight");
            Lw.fa_v_proj = &must(engine, fa + "v_proj.weight");
            Lw.fa_o_proj = &must(engine, fa + "o_proj.weight");
            Lw.fa_q_norm = &must(engine, fa + "q_norm.weight");
            Lw.fa_k_norm = &must(engine, fa + "k_norm.weight");
            Lw.fa_q_proj_quant = engine.quant_meta(fa + "q_proj.weight");
            Lw.fa_k_proj_quant = engine.quant_meta(fa + "k_proj.weight");
            Lw.fa_v_proj_quant = engine.quant_meta(fa + "v_proj.weight");
            Lw.fa_o_proj_quant = engine.quant_meta(fa + "o_proj.weight");
            Lw.kv_layer = kv_slot++;
        } else {
            throw std::runtime_error(
                "qwen3_5_moe: unknown layer_type '" + kind + "' at layer " +
                std::to_string(li));
        }

        // ── Sparse-MoE block (every layer) ────────────────────────
        // Routed and shared experts both shard along the intermediate
        // axis (column-parallel gate/up + row-parallel down). The engine
        // load loop streams per-rank slices of `experts.gate_up_proj` /
        // `experts.down_proj` straight from the safetensors mmap, so we
        // never materialise the full fused tensors on a rank — the
        // bind-time slice helpers above are now unused on this path.
        // The moe_forward block emits a single all-reduce on the
        // combined routed+shared partial sum.
        Lw.moe_router       = &must(engine, lp + "mlp.gate.weight");
        bind_routed_experts(engine, lp, cfg.num_experts, Lw);
        if (has_shared_expert) {
            Lw.shared_gate_proj = &must(engine, lp + "mlp.shared_expert.gate_proj.weight");
            Lw.shared_up_proj   = &must(engine, lp + "mlp.shared_expert.up_proj.weight");
            Lw.shared_down_proj = &must(engine, lp + "mlp.shared_expert.down_proj.weight");
            Lw.shared_gate      = &must(engine, lp + "mlp.shared_expert_gate.weight");
            Lw.shared_gate_proj_quant =
                engine.quant_meta(lp + "mlp.shared_expert.gate_proj.weight");
            Lw.shared_up_proj_quant =
                engine.quant_meta(lp + "mlp.shared_expert.up_proj.weight");
            Lw.shared_down_proj_quant =
                engine.quant_meta(lp + "mlp.shared_expert.down_proj.weight");
            Lw.shared_gate_quant =
                engine.quant_meta(lp + "mlp.shared_expert_gate.weight");
            // Published by `shared_expert_gate_up_joins`, which decides
            // between the two shapes; absent when the sources are quantized.
            Lw.shared_gate_up_gate_proj =
                maybe(engine, lp + "mlp.shared_expert.gate_up_gate_proj.weight");
            Lw.shared_gate_up_proj =
                maybe(engine, lp + "mlp.shared_expert.gate_up_proj.weight");
        }
    }

    if (cfg.mtp_num_hidden_layers > 0 && engine.has("mtp.fc.weight")) {
        Qwen3_5MoeWeights::MtpWeights mtp;
        mtp.pre_fc_norm_embedding = &must(engine, "mtp.pre_fc_norm_embedding.weight");
        mtp.pre_fc_norm_hidden = &must(engine, "mtp.pre_fc_norm_hidden.weight");
        mtp.fc = &must(engine, "mtp.fc.weight");
        mtp.norm = &must(engine, "mtp.norm.weight");
        mtp.embed = cfg.mtp_use_dedicated_embeddings
            ? &must(engine, "mtp.embed_tokens.weight")
            : w.embed;

        const std::string lp = "mtp.layers.0.";
        auto& Lw = mtp.layer;
        Lw.kind = Qwen3_5MoeLayerWeights::Kind::FullAttn;
        Lw.attn_norm_pre = &must(engine, lp + "input_layernorm.weight");
        Lw.mlp_norm_pre = &must(engine, lp + "post_attention_layernorm.weight");
        const std::string fa = lp + "self_attn.";
        Lw.fa_q_proj = &must(engine, fa + "q_proj.weight");
        Lw.fa_k_proj = &must(engine, fa + "k_proj.weight");
        Lw.fa_v_proj = &must(engine, fa + "v_proj.weight");
        Lw.fa_o_proj = &must(engine, fa + "o_proj.weight");
        Lw.fa_q_norm = &must(engine, fa + "q_norm.weight");
        Lw.fa_k_norm = &must(engine, fa + "k_norm.weight");
        Lw.fa_q_proj_quant = engine.quant_meta(fa + "q_proj.weight");
        Lw.fa_k_proj_quant = engine.quant_meta(fa + "k_proj.weight");
        Lw.fa_v_proj_quant = engine.quant_meta(fa + "v_proj.weight");
        Lw.fa_o_proj_quant = engine.quant_meta(fa + "o_proj.weight");
        Lw.moe_router = &must(engine, lp + "mlp.gate.weight");
        bind_routed_experts(engine, lp, cfg.num_experts, Lw);
        if (has_shared_expert) {
            Lw.shared_gate_proj = &must(engine, lp + "mlp.shared_expert.gate_proj.weight");
            Lw.shared_up_proj = &must(engine, lp + "mlp.shared_expert.up_proj.weight");
            Lw.shared_down_proj = &must(engine, lp + "mlp.shared_expert.down_proj.weight");
            Lw.shared_gate = &must(engine, lp + "mlp.shared_expert_gate.weight");
            Lw.shared_gate_proj_quant =
                engine.quant_meta(lp + "mlp.shared_expert.gate_proj.weight");
            Lw.shared_up_proj_quant =
                engine.quant_meta(lp + "mlp.shared_expert.up_proj.weight");
            Lw.shared_down_proj_quant =
                engine.quant_meta(lp + "mlp.shared_expert.down_proj.weight");
            Lw.shared_gate_quant =
                engine.quant_meta(lp + "mlp.shared_expert_gate.weight");
            // Published by `shared_expert_gate_up_joins`, which decides
            // between the two shapes; absent when the sources are quantized.
            Lw.shared_gate_up_gate_proj =
                maybe(engine, lp + "mlp.shared_expert.gate_up_gate_proj.weight");
            Lw.shared_gate_up_proj =
                maybe(engine, lp + "mlp.shared_expert.gate_up_proj.weight");
        }
        Lw.kv_layer = kv_slot++;
        w.mtp = mtp;
    }

    return w;
}

}  // namespace pie_cuda_driver::model
