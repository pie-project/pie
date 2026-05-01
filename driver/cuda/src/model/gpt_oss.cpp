#include "model/gpt_oss.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/dequant_fp4.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const Engine& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("gpt_oss: missing weight '" + name + "'");
    }
    return e.get(name);
}

// Compute total slots reserved up front so the `owned_expert_buffers`
// vector never reallocates — pointers we hand to MixtralWeights stay
// stable for the full lifetime of MixtralWeights. We push 3 owning
// (w_gate, w_up, w_down) plus 3 view (b_gate, b_up, b_down) per expert.
constexpr int kTensorsPerExpert = 6;

}  // namespace

MixtralWeights bind_gpt_oss(Engine& engine) {
    const auto& cfg = engine.hf_config();
    const int E = cfg.num_experts;
    if (E <= 0) {
        throw std::runtime_error(
            "gpt_oss: hf_config.num_experts must be > 0; check the loader");
    }
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int L = cfg.num_hidden_layers;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;

    // MXFP4 packs 32 elements per E8M0 scale; dequant kernel requires
    // in_dim divisible by 32. GPT-OSS uses H = I = 2880 → 90, fine.
    if (H % 32 != 0 || I % 32 != 0) {
        throw std::runtime_error(
            "gpt_oss: hidden_size and intermediate_size must be multiples of 32 "
            "for MXFP4 dequant; got hidden=" + std::to_string(H) +
            ", intermediate=" + std::to_string(I));
    }

    MixtralWeights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "gpt_oss: lm_head missing and tie_word_embeddings=false");
    }

    // Reserve enough slots so push_back never reallocates and pointers
    // we cache into MixtralExpertWeights stay valid.
    w.owned_expert_buffers.reserve(static_cast<std::size_t>(L) * E * kTensorsPerExpert);
    w.layers.resize(static_cast<std::size_t>(L));

    cudaStream_t stream = nullptr;

    for (int li = 0; li < L; ++li) {
        const std::string p = "model.layers." + std::to_string(li) + ".";
        auto& Lw = w.layers[li];

        // ── Attention ──
        Lw.attn_norm = &must(engine, p + "input_layernorm.weight");
        Lw.mlp_norm  = &must(engine, p + "post_attention_layernorm.weight");
        Lw.q_proj    = &must(engine, p + "self_attn.q_proj.weight");
        Lw.k_proj    = &must(engine, p + "self_attn.k_proj.weight");
        Lw.v_proj    = &must(engine, p + "self_attn.v_proj.weight");
        Lw.o_proj    = &must(engine, p + "self_attn.o_proj.weight");
        Lw.q_bias    = &must(engine, p + "self_attn.q_proj.bias");
        Lw.k_bias    = &must(engine, p + "self_attn.k_proj.bias");
        Lw.v_bias    = &must(engine, p + "self_attn.v_proj.bias");
        Lw.o_bias    = &must(engine, p + "self_attn.o_proj.bias");
        Lw.attn_sinks = &must(engine, p + "self_attn.sinks");

        // Sanity-check the bias / sink shapes before wiring them in —
        // a mismatched checkpoint here would corrupt activations
        // silently in the bias-add kernel.
        if (Lw.q_bias->numel() != static_cast<std::size_t>(Hq) ||
            Lw.k_bias->numel() != static_cast<std::size_t>(Hk) ||
            Lw.v_bias->numel() != static_cast<std::size_t>(Hk) ||
            Lw.o_bias->numel() != static_cast<std::size_t>(H)) {
            throw std::runtime_error(
                "gpt_oss: attention bias shape mismatch at layer " +
                std::to_string(li));
        }
        if (Lw.attn_sinks->numel() != static_cast<std::size_t>(cfg.num_attention_heads)) {
            throw std::runtime_error(
                "gpt_oss: attn_sinks shape mismatch at layer " +
                std::to_string(li) + ": expected " +
                std::to_string(cfg.num_attention_heads) + ", got " +
                std::to_string(Lw.attn_sinks->numel()));
        }

        // ── Router ──
        Lw.router      = &must(engine, p + "mlp.router.weight");
        Lw.router_bias = &must(engine, p + "mlp.router.bias");

        // ── MXFP4 expert weights → bf16 ──
        // gate_up_proj_blocks: [E, 2*I, H/2] uint8 (2 nibbles per byte)
        // gate_up_proj_scales: [E, 2*I, H/32] uint8 (E8M0)
        // gate_up_proj_bias  : [E, 2*I]      bf16
        // down_proj_blocks   : [E, H, I/2]   uint8
        // down_proj_scales   : [E, H, I/32]  uint8
        // down_proj_bias     : [E, H]        bf16
        const auto& gu_blocks = must(engine, p + "mlp.experts.gate_up_proj_blocks");
        const auto& gu_scales = must(engine, p + "mlp.experts.gate_up_proj_scales");
        const auto& gu_bias   = must(engine, p + "mlp.experts.gate_up_proj_bias");
        const auto& dn_blocks = must(engine, p + "mlp.experts.down_proj_blocks");
        const auto& dn_scales = must(engine, p + "mlp.experts.down_proj_scales");
        const auto& dn_bias   = must(engine, p + "mlp.experts.down_proj_bias");

        // Validate dtypes — the safetensors loader maps F4_E2M1 / F8_E4M3
        // / U8 all to UINT8 (raw byte storage); bias tensors must already
        // be bf16.
        auto is_u8  = [](const DeviceTensor& t) { return t.dtype() == DType::UINT8; };
        auto is_bf16 = [](const DeviceTensor& t) { return t.dtype() == DType::BF16; };
        if (!is_u8(gu_blocks) || !is_u8(gu_scales) ||
            !is_u8(dn_blocks) || !is_u8(dn_scales)) {
            throw std::runtime_error(
                "gpt_oss layer " + std::to_string(li) +
                ": expected MXFP4 packed weights as uint8 byte storage");
        }
        if (!is_bf16(gu_bias) || !is_bf16(dn_bias)) {
            throw std::runtime_error(
                "gpt_oss layer " + std::to_string(li) +
                ": expected expert biases as bf16");
        }

        // Per-expert byte strides into the fused tensors.
        const std::size_t gu_blocks_per_expert =
            static_cast<std::size_t>(2) * I * (H / 2);            // bytes
        const std::size_t gu_scales_per_expert =
            static_cast<std::size_t>(2) * I * (H / 32);           // bytes
        const std::size_t dn_blocks_per_expert =
            static_cast<std::size_t>(H) * (I / 2);                // bytes
        const std::size_t dn_scales_per_expert =
            static_cast<std::size_t>(H) * (I / 32);               // bytes
        const std::size_t bias_bf16 = 2;                           // bytes
        const std::size_t gu_bias_per_expert =
            static_cast<std::size_t>(2) * I;                       // elements
        const std::size_t dn_bias_per_expert =
            static_cast<std::size_t>(H);                           // elements

        Lw.experts.resize(static_cast<std::size_t>(E));
        std::uint8_t* gu_blocks_base = const_cast<std::uint8_t*>(
            static_cast<const std::uint8_t*>(gu_blocks.data()));
        std::uint8_t* gu_scales_base = const_cast<std::uint8_t*>(
            static_cast<const std::uint8_t*>(gu_scales.data()));
        std::uint8_t* dn_blocks_base = const_cast<std::uint8_t*>(
            static_cast<const std::uint8_t*>(dn_blocks.data()));
        std::uint8_t* dn_scales_base = const_cast<std::uint8_t*>(
            static_cast<const std::uint8_t*>(dn_scales.data()));
        std::uint8_t* gu_bias_base = const_cast<std::uint8_t*>(
            static_cast<const std::uint8_t*>(gu_bias.data()));
        std::uint8_t* dn_bias_base = const_cast<std::uint8_t*>(
            static_cast<const std::uint8_t*>(dn_bias.data()));

        for (int e = 0; e < E; ++e) {
            auto& Ew = Lw.experts[e];

            // gate = rows [0, I) of the fused gate_up; up = rows [I, 2I).
            // Each "row" within the fused tensor has H elements stored as
            // H/2 bytes (2 nibbles per byte) and H/32 scale bytes.
            const std::size_t e_gu_blocks = static_cast<std::size_t>(e) * gu_blocks_per_expert;
            const std::size_t e_gu_scales = static_cast<std::size_t>(e) * gu_scales_per_expert;
            const std::size_t up_off_blocks = static_cast<std::size_t>(I) * (H / 2);
            const std::size_t up_off_scales = static_cast<std::size_t>(I) * (H / 32);

            DeviceTensor w_gate = DeviceTensor::allocate(DType::BF16, {I, H});
            kernels::launch_dequant_mxfp4_to_bf16(
                gu_blocks_base + e_gu_blocks,
                gu_scales_base + e_gu_scales,
                w_gate.data(), I, H, stream);
            w.owned_expert_buffers.push_back(std::move(w_gate));
            Ew.w_gate = &w.owned_expert_buffers.back();

            DeviceTensor w_up = DeviceTensor::allocate(DType::BF16, {I, H});
            kernels::launch_dequant_mxfp4_to_bf16(
                gu_blocks_base + e_gu_blocks + up_off_blocks,
                gu_scales_base + e_gu_scales + up_off_scales,
                w_up.data(), I, H, stream);
            w.owned_expert_buffers.push_back(std::move(w_up));
            Ew.w_up = &w.owned_expert_buffers.back();

            DeviceTensor w_down = DeviceTensor::allocate(DType::BF16, {H, I});
            kernels::launch_dequant_mxfp4_to_bf16(
                dn_blocks_base + static_cast<std::size_t>(e) * dn_blocks_per_expert,
                dn_scales_base + static_cast<std::size_t>(e) * dn_scales_per_expert,
                w_down.data(), H, I, stream);
            w.owned_expert_buffers.push_back(std::move(w_down));
            Ew.w_down = &w.owned_expert_buffers.back();

            // Biases live as views into the fused bias tensors. The fused
            // layouts are [E, 2*I] for gate_up_bias (gate = first I,
            // up = last I) and [E, H] for down_bias.
            std::uint8_t* gu_bias_e = gu_bias_base +
                static_cast<std::size_t>(e) * gu_bias_per_expert * bias_bf16;
            std::uint8_t* dn_bias_e = dn_bias_base +
                static_cast<std::size_t>(e) * dn_bias_per_expert * bias_bf16;

            DeviceTensor b_gate = DeviceTensor::view(
                gu_bias_e, DType::BF16, {I});
            w.owned_expert_buffers.push_back(std::move(b_gate));
            Ew.b_gate = &w.owned_expert_buffers.back();

            DeviceTensor b_up = DeviceTensor::view(
                gu_bias_e + static_cast<std::size_t>(I) * bias_bf16,
                DType::BF16, {I});
            w.owned_expert_buffers.push_back(std::move(b_up));
            Ew.b_up = &w.owned_expert_buffers.back();

            DeviceTensor b_down = DeviceTensor::view(
                dn_bias_e, DType::BF16, {H});
            w.owned_expert_buffers.push_back(std::move(b_down));
            Ew.b_down = &w.owned_expert_buffers.back();
        }
    }
    // The dequant kernels were enqueued on stream=0; sync once at the end
    // so the host-side bind returns with all weights resident.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return w;
}

}  // namespace pie_cuda_driver::model
