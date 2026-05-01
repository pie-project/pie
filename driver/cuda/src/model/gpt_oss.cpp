#include "model/gpt_oss.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/dequant_fp4.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const Engine& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("gpt_oss: missing weight '" + name + "'");
    }
    return e.get(name);
}

// Owning + view tensors per expert: w_gate, w_up, w_down (owning bf16
// from MXFP4 dequant + deinterleave) and b_gate, b_up (owning bf16 from
// bias-deinterleave). b_down is a non-owning view into the
// `down_proj_bias` tensor that the safetensors loader already owns.
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

    w.owned_expert_buffers.reserve(static_cast<std::size_t>(L) * E * kTensorsPerExpert);
    w.layers.resize(static_cast<std::size_t>(L));

    cudaStream_t stream = nullptr;

    // Scratch: dequantized fused gate_up bf16 of shape [2*I, H]. Reused
    // for every expert across every layer; sized once for the worst case.
    auto fused_gu = DeviceTensor::allocate(DType::BF16, {2 * I, H});

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
        // Real on-disk shapes (gpt-oss-20b safetensors):
        //   gate_up_proj_blocks : [E, 2*I, H/32, 16] uint8  (16 bytes per
        //                          32-element block, 2 nibbles per byte)
        //   gate_up_proj_scales : [E, 2*I, H/32]   uint8  (E8M0)
        //   gate_up_proj_bias   : [E, 2*I]         bf16
        //   down_proj_blocks    : [E, H, I/32, 16] uint8
        //   down_proj_scales    : [E, H, I/32]     uint8
        //   down_proj_bias      : [E, H]           bf16
        //
        // The output axis (2*I or H) is the *second*; flattened, each
        // expert's blocks slab is `[out_dim, in_dim/2]` bytes. The dequant
        // kernel takes that flat layout and writes bf16 `[out_dim, in_dim]`.
        //
        // For gate_up the LAST axis is interleaved per HF's
        // `gate = gate_up[..., ::2], up = gate_up[..., 1::2]` — but the
        // MXFP4 storage is the transpose, so post-dequant the *rows*
        // (output axis) are interleaved instead: row 2k is gate channel
        // k, row 2k+1 is up channel k. We dequant the full 2*I rows once
        // and split into per-expert gate/up bf16 tensors with a separate
        // deinterleave kernel.
        const auto& gu_blocks = must(engine, p + "mlp.experts.gate_up_proj_blocks");
        const auto& gu_scales = must(engine, p + "mlp.experts.gate_up_proj_scales");
        const auto& gu_bias   = must(engine, p + "mlp.experts.gate_up_proj_bias");
        const auto& dn_blocks = must(engine, p + "mlp.experts.down_proj_blocks");
        const auto& dn_scales = must(engine, p + "mlp.experts.down_proj_scales");
        const auto& dn_bias   = must(engine, p + "mlp.experts.down_proj_bias");

        auto is_u8   = [](const DeviceTensor& t) { return t.dtype() == DType::UINT8; };
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

        // Per-expert byte strides into the fused storage.
        const std::size_t gu_blocks_per_expert =
            static_cast<std::size_t>(2) * I * (H / 2);     // 16 × H/32 = H/2 bytes
        const std::size_t gu_scales_per_expert =
            static_cast<std::size_t>(2) * I * (H / 32);
        const std::size_t dn_blocks_per_expert =
            static_cast<std::size_t>(H) * (I / 2);
        const std::size_t dn_scales_per_expert =
            static_cast<std::size_t>(H) * (I / 32);
        const std::size_t bias_bf16_bytes = 2;
        const std::size_t gu_bias_per_expert = static_cast<std::size_t>(2) * I;
        const std::size_t dn_bias_per_expert = static_cast<std::size_t>(H);

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

            // 1. Dequant fused gate_up MXFP4 → bf16 [2*I, H].
            kernels::launch_dequant_mxfp4_to_bf16(
                gu_blocks_base + static_cast<std::size_t>(e) * gu_blocks_per_expert,
                gu_scales_base + static_cast<std::size_t>(e) * gu_scales_per_expert,
                fused_gu.data(), 2 * I, H, stream);

            // 2. Split fused [2*I, H] into per-channel gate / up bf16
            //    [I, H] tensors via even/odd row stride.
            DeviceTensor w_gate = DeviceTensor::allocate(DType::BF16, {I, H});
            DeviceTensor w_up   = DeviceTensor::allocate(DType::BF16, {I, H});
            kernels::launch_deinterleave_rows_bf16(
                fused_gu.data(), w_gate.data(), w_up.data(), I, H, stream);
            w.owned_expert_buffers.push_back(std::move(w_gate));
            Ew.w_gate = &w.owned_expert_buffers.back();
            w.owned_expert_buffers.push_back(std::move(w_up));
            Ew.w_up = &w.owned_expert_buffers.back();

            // 3. Dequant down [H, I] → bf16 directly (no deinterleave).
            DeviceTensor w_down = DeviceTensor::allocate(DType::BF16, {H, I});
            kernels::launch_dequant_mxfp4_to_bf16(
                dn_blocks_base + static_cast<std::size_t>(e) * dn_blocks_per_expert,
                dn_scales_base + static_cast<std::size_t>(e) * dn_scales_per_expert,
                w_down.data(), H, I, stream);
            w.owned_expert_buffers.push_back(std::move(w_down));
            Ew.w_down = &w.owned_expert_buffers.back();

            // 4. Deinterleave gate_up_bias [2*I] → b_gate [I], b_up [I].
            std::uint8_t* gu_bias_e = gu_bias_base +
                static_cast<std::size_t>(e) * gu_bias_per_expert * bias_bf16_bytes;
            DeviceTensor b_gate = DeviceTensor::allocate(DType::BF16, {I});
            DeviceTensor b_up   = DeviceTensor::allocate(DType::BF16, {I});
            kernels::launch_deinterleave_vec_bf16(
                gu_bias_e, b_gate.data(), b_up.data(), I, stream);
            w.owned_expert_buffers.push_back(std::move(b_gate));
            Ew.b_gate = &w.owned_expert_buffers.back();
            w.owned_expert_buffers.push_back(std::move(b_up));
            Ew.b_up = &w.owned_expert_buffers.back();

            // 5. b_down is a non-owning view into the loader-owned bias
            //    tensor; no deinterleave or dequant needed.
            std::uint8_t* dn_bias_e = dn_bias_base +
                static_cast<std::size_t>(e) * dn_bias_per_expert * bias_bf16_bytes;
            DeviceTensor b_down = DeviceTensor::view(
                dn_bias_e, DType::BF16, {H});
            w.owned_expert_buffers.push_back(std::move(b_down));
            Ew.b_down = &w.owned_expert_buffers.back();
        }
    }
    // Sync stream once at the end so weights are resident before bind returns.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return w;
}

}  // namespace pie_cuda_driver::model
