#include "model/gpt_oss.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("gpt_oss: missing weight '" + name + "'");
    }
    return e.get(name);
}

DeviceTensor tensor_view(
    const DeviceTensor& backing,
    std::size_t byte_offset,
    std::vector<std::int64_t> shape)
{
    auto* base = const_cast<std::uint8_t*>(
        static_cast<const std::uint8_t*>(backing.data()));
    return DeviceTensor::view(
        base + byte_offset,
        backing.dtype(),
        std::move(shape));
}

constexpr int kViewsPerExpert = 8;

}  // namespace

MixtralWeights bind_gpt_oss(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    const int E = cfg.num_experts;
    if (E <= 0) {
        throw std::runtime_error(
            "gpt_oss: hf_config.num_experts must be > 0; check the loader");
    }

    const int T = std::max(1, engine.distributed().tp_size);
    const int H = cfg.hidden_size;
    const int I_full = cfg.intermediate_size;
    if (I_full % T != 0) {
        throw std::runtime_error(
            "gpt_oss: intermediate_size must be divisible by tp_size");
    }
    const int I = I_full / T;
    const int L = cfg.num_hidden_layers;
    const int Hq = (cfg.num_attention_heads * cfg.head_dim) / T;
    const int Hk = (cfg.num_key_value_heads * cfg.head_dim) / T;
    const int sinks_local = cfg.num_attention_heads / T;

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

    w.owned_expert_buffers.reserve(
        static_cast<std::size_t>(L) * E * kViewsPerExpert);
    w.layers.resize(static_cast<std::size_t>(L));

    const bool use_mxfp4_packed_experts =
        engine.has("model.layers.0.mlp.experts.gate_up_proj.weight");
    if (use_mxfp4_packed_experts) {
        if (H % 32 != 0 || I % 32 != 0) {
            throw std::runtime_error(
                "gpt_oss: packed MXFP4 expert backend requires hidden and "
                "tp-local intermediate dimensions divisible by 32");
        }
        w.mxfp4_gate_up_bf16_scratch =
            DeviceTensor::allocate(DType::BF16, {2 * I, H});
        w.mxfp4_down_bf16_scratch =
            DeviceTensor::allocate(DType::BF16, {H, I});
    }

    const std::size_t bf16 = 2;
    const std::size_t gate_up_expert_bytes =
        static_cast<std::size_t>(I) * H * bf16;
    const std::size_t down_expert_bytes =
        static_cast<std::size_t>(H) * I * bf16;
    const std::size_t gate_up_bias_expert_bytes =
        static_cast<std::size_t>(I) * bf16;
    const std::size_t down_bias_expert_bytes =
        static_cast<std::size_t>(H) * bf16;
    const std::size_t mxfp4_gate_up_expert_bytes =
        static_cast<std::size_t>(2 * I) *
        static_cast<std::size_t>(H / 32) * 16;
    const std::size_t mxfp4_gate_up_scale_expert_bytes =
        static_cast<std::size_t>(2 * I) * static_cast<std::size_t>(H / 32);
    const std::size_t mxfp4_down_expert_bytes =
        static_cast<std::size_t>(H) *
        static_cast<std::size_t>(I / 32) * 16;
    const std::size_t mxfp4_down_scale_expert_bytes =
        static_cast<std::size_t>(H) * static_cast<std::size_t>(I / 32);
    const std::size_t mxfp4_gate_up_bias_expert_bytes =
        static_cast<std::size_t>(2 * I) * bf16;

    for (int li = 0; li < L; ++li) {
        const std::string p = "model.layers." + std::to_string(li) + ".";
        auto& Lw = w.layers[li];

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
        if (Lw.attn_sinks->numel() != static_cast<std::size_t>(sinks_local)) {
            throw std::runtime_error(
                "gpt_oss: attn_sinks shape mismatch at layer " +
                std::to_string(li));
        }

        Lw.router      = &must(engine, p + "mlp.router.weight");
        Lw.router_bias = &must(engine, p + "mlp.router.bias");

        Lw.experts.resize(static_cast<std::size_t>(E));
        if (use_mxfp4_packed_experts) {
            const std::string gate_up_name =
                p + "mlp.experts.gate_up_proj.weight";
            const std::string down_name =
                p + "mlp.experts.down_proj.weight";
            const std::string gate_up_bias_name =
                p + "mlp.experts.gate_up_proj.bias";
            const std::string down_bias_name =
                p + "mlp.experts.down_proj.bias";
            const auto& gate_up_all = must(engine, gate_up_name);
            const auto& down_all = must(engine, down_name);
            const auto gate_up_meta = engine.quant_meta(gate_up_name);
            const auto down_meta = engine.quant_meta(down_name);
            if (!gate_up_meta.has_value() || !gate_up_meta->scale ||
                !down_meta.has_value() || !down_meta->scale) {
                throw std::runtime_error(
                    "gpt_oss: packed MXFP4 expert tensors are missing "
                    "quant metadata at layer " + std::to_string(li));
            }
            const auto& gate_up_scale_all = *gate_up_meta->scale;
            const auto& down_scale_all = *down_meta->scale;
            const auto& b_gate_up_all = must(engine, gate_up_bias_name);
            const auto& b_down_all = must(engine, down_bias_name);

            const std::vector<std::int64_t> gate_up_shape =
                {E, 2 * I, H / 32, 16};
            const std::vector<std::int64_t> gate_up_scale_shape =
                {E, 2 * I, H / 32};
            const std::vector<std::int64_t> down_shape =
                {E, H, I / 32, 16};
            const std::vector<std::int64_t> down_scale_shape =
                {E, H, I / 32};
            const std::vector<std::int64_t> gate_up_bias_shape = {E, 2 * I};
            const std::vector<std::int64_t> down_bias_shape = {E, H};
            if (gate_up_all.dtype() != DType::UINT8 ||
                down_all.dtype() != DType::UINT8 ||
                gate_up_scale_all.dtype() != DType::UINT8 ||
                down_scale_all.dtype() != DType::UINT8 ||
                b_gate_up_all.dtype() != DType::BF16 ||
                b_down_all.dtype() != DType::BF16 ||
                gate_up_all.shape() != gate_up_shape ||
                gate_up_scale_all.shape() != gate_up_scale_shape ||
                down_all.shape() != down_shape ||
                down_scale_all.shape() != down_scale_shape ||
                b_gate_up_all.shape() != gate_up_bias_shape ||
                b_down_all.shape() != down_bias_shape) {
                throw std::runtime_error(
                    "gpt_oss: packed MXFP4 expert tensor shape mismatch at "
                    "layer " + std::to_string(li));
            }

            for (int e = 0; e < E; ++e) {
                auto& Ew = Lw.experts[e];
                Ew.format = MixtralExpertWeightFormat::Mxfp4RoutedDequant;

                w.owned_expert_buffers.push_back(tensor_view(
                    gate_up_all,
                    static_cast<std::size_t>(e) *
                        mxfp4_gate_up_expert_bytes,
                    {2 * I, H / 32, 16}));
                Ew.w_gate_up = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    gate_up_scale_all,
                    static_cast<std::size_t>(e) *
                        mxfp4_gate_up_scale_expert_bytes,
                    {2 * I, H / 32}));
                Ew.w_gate_up_scale = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    down_all,
                    static_cast<std::size_t>(e) *
                        mxfp4_down_expert_bytes,
                    {H, I / 32, 16}));
                Ew.w_down_packed = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    down_scale_all,
                    static_cast<std::size_t>(e) *
                        mxfp4_down_scale_expert_bytes,
                    {H, I / 32}));
                Ew.w_down_scale = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    b_gate_up_all,
                    static_cast<std::size_t>(e) *
                        mxfp4_gate_up_bias_expert_bytes,
                    {I}));
                Ew.b_gate = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    b_gate_up_all,
                    static_cast<std::size_t>(e) *
                            mxfp4_gate_up_bias_expert_bytes +
                        static_cast<std::size_t>(I) * bf16,
                    {I}));
                Ew.b_up = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    b_down_all,
                    static_cast<std::size_t>(e) * down_bias_expert_bytes,
                    {H}));
                Ew.b_down = &w.owned_expert_buffers.back();
            }
        } else {
            const std::string gate_name =
                p + "mlp.experts.gate_proj.weight";
            const std::string up_name =
                p + "mlp.experts.up_proj.weight";
            const std::string down_weight_name =
                p + "mlp.experts.down_proj.weight";
            const std::string gate_bias_name =
                p + "mlp.experts.gate_proj.bias";
            const std::string up_bias_name =
                p + "mlp.experts.up_proj.bias";
            const std::string down_bias_name =
                p + "mlp.experts.down_proj.bias";
            const auto& gate_all = must(engine, gate_name);
            const auto& up_all = must(engine, up_name);
            const auto& down_all = must(engine, down_weight_name);
            const auto& b_gate_all = must(engine, gate_bias_name);
            const auto& b_up_all = must(engine, up_bias_name);
            const auto& b_down_all = must(engine, down_bias_name);

            const std::vector<std::int64_t> gate_shape = {E, I, H};
            const std::vector<std::int64_t> down_shape = {E, H, I};
            const std::vector<std::int64_t> gate_bias_shape = {E, I};
            const std::vector<std::int64_t> down_bias_shape = {E, H};
            if (gate_all.dtype() != DType::BF16 ||
                up_all.dtype() != DType::BF16 ||
                down_all.dtype() != DType::BF16 ||
                b_gate_all.dtype() != DType::BF16 ||
                b_up_all.dtype() != DType::BF16 ||
                b_down_all.dtype() != DType::BF16 ||
                gate_all.shape() != gate_shape ||
                up_all.shape() != gate_shape ||
                down_all.shape() != down_shape ||
                b_gate_all.shape() != gate_bias_shape ||
                b_up_all.shape() != gate_bias_shape ||
                b_down_all.shape() != down_bias_shape) {
                throw std::runtime_error(
                    "gpt_oss: materialized expert tensor shape mismatch at "
                    "layer " + std::to_string(li));
            }

            for (int e = 0; e < E; ++e) {
                auto& Ew = Lw.experts[e];

                w.owned_expert_buffers.push_back(tensor_view(
                    gate_all,
                    static_cast<std::size_t>(e) * gate_up_expert_bytes,
                    {I, H}));
                Ew.w_gate = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    up_all,
                    static_cast<std::size_t>(e) * gate_up_expert_bytes,
                    {I, H}));
                Ew.w_up = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    down_all,
                    static_cast<std::size_t>(e) * down_expert_bytes,
                    {H, I}));
                Ew.w_down = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    b_gate_all,
                    static_cast<std::size_t>(e) * gate_up_bias_expert_bytes,
                    {I}));
                Ew.b_gate = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    b_up_all,
                    static_cast<std::size_t>(e) * gate_up_bias_expert_bytes,
                    {I}));
                Ew.b_up = &w.owned_expert_buffers.back();

                w.owned_expert_buffers.push_back(tensor_view(
                    b_down_all,
                    static_cast<std::size_t>(e) * down_bias_expert_bytes,
                    {H}));
                Ew.b_down = &w.owned_expert_buffers.back();
            }
        }
    }

    return w;
}

}  // namespace pie_cuda_driver::model
