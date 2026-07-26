#include "model/mixtral/gpt_oss.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/deinterleave.hpp"

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
constexpr int kNativeMxfp4HandlesPerExpert = 9;

int align_up_int(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

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
    const bool native_mxfp4 = engine.mxfp4_moe_policy() ==
        Mxfp4MoePolicy::NativeGemm;
    const int I_native = native_mxfp4 ? align_up_int(I, 128) : I;
    const int L = cfg.num_hidden_layers;
    const int Hq = (cfg.num_attention_heads * cfg.head_dim) / T;
    const int Hk = (cfg.num_key_value_heads * cfg.head_dim) / T;
    const int sinks_local = cfg.num_attention_heads / T;

    MixtralWeights w;
    w.mxfp4_intermediate_padded = native_mxfp4 ? I_native : 0;
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
        static_cast<std::size_t>(L) * E *
        static_cast<std::size_t>(
            native_mxfp4 ? kNativeMxfp4HandlesPerExpert : kViewsPerExpert));
    w.layers.resize(static_cast<std::size_t>(L));

    const bool has_routed_mxfp4_experts =
        engine.has("model.layers.0.mlp.experts.gate_up_proj.weight");
    const bool has_native_mxfp4_experts =
        engine.has("model.layers.0.mlp.experts.gate_proj.weight") &&
        engine.has("model.layers.0.mlp.experts.up_proj.weight") &&
        engine.has("model.layers.0.mlp.experts.down_proj.weight");
    if (native_mxfp4 && !has_native_mxfp4_experts) {
        throw std::runtime_error(
            "gpt_oss: native MXFP4 selected, but loader did not finalize "
            "native gate/up/down expert tensors");
    }
    if (!native_mxfp4 && has_native_mxfp4_experts) {
        throw std::runtime_error(
            "gpt_oss: native MXFP4 tensors were loaded, but runtime lowering "
            "did not select native MXFP4");
    }
    // Streamed experts publish no resident weight at all, so the packed
    // backend is announced by the group's presence instead. The group is the
    // packed declaration; it just has the expert axis fixed at one.
    const bool has_streamed_mxfp4_experts =
        engine.group_cache() != nullptr &&
        engine.group_cache()->find_group("model.layers.0.mlp.experts") !=
            GroupStreamCache::kNoGroup;
    const bool use_mxfp4_packed_experts =
        has_routed_mxfp4_experts || has_native_mxfp4_experts ||
        has_streamed_mxfp4_experts;
    if (use_mxfp4_packed_experts) {
        if (H % 32 != 0 || I % 32 != 0) {
            throw std::runtime_error(
                "gpt_oss: packed MXFP4 expert backend requires hidden and "
                "tp-local intermediate dimensions divisible by 32");
        }
        if (has_native_mxfp4_experts) {
            if (H % 64 != 0 || I_native % 64 != 0) {
                throw std::runtime_error(
                    "gpt_oss: native MXFP4/Marlin expert backend requires "
                    "hidden and padded tp-local intermediate dimensions "
                    "divisible by 64");
            }
        } else {
            w.mxfp4_gate_up_bf16_scratch =
                DeviceTensor::allocate(DType::BF16, {2 * I, H});
            w.mxfp4_gate_bf16_scratch =
                DeviceTensor::allocate(DType::BF16, {I, H});
            w.mxfp4_up_bf16_scratch =
                DeviceTensor::allocate(DType::BF16, {I, H});
            w.mxfp4_down_bf16_scratch =
                DeviceTensor::allocate(DType::BF16, {H, I});
        }
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
    const std::size_t native_gate_expert_bytes =
        (static_cast<std::size_t>(I_native) * static_cast<std::size_t>(H)) / 2;
    const std::size_t native_gate_scale_expert_bytes =
        static_cast<std::size_t>(I_native) * static_cast<std::size_t>(H / 32);
    const std::size_t native_down_expert_bytes =
        (static_cast<std::size_t>(H) * static_cast<std::size_t>(I_native)) / 2;
    const std::size_t native_down_scale_expert_bytes =
        static_cast<std::size_t>(H) * static_cast<std::size_t>(I_native / 32);

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
            if (has_native_mxfp4_experts) {
                const std::string gate_name =
                    p + "mlp.experts.gate_proj.weight";
                const std::string up_name =
                    p + "mlp.experts.up_proj.weight";
                const std::string down_name =
                    p + "mlp.experts.down_proj.weight";
                const std::string gate_scale_name =
                    p + "mlp.experts.gate_proj.weight_scale";
                const std::string up_scale_name =
                    p + "mlp.experts.up_proj.weight_scale";
                const std::string down_scale_name =
                    p + "mlp.experts.down_proj.weight_scale";
                const std::string gate_bias_name =
                    p + "mlp.experts.gate_proj.bias";
                const std::string up_bias_name =
                    p + "mlp.experts.up_proj.bias";
                const std::string down_bias_name =
                    p + "mlp.experts.down_proj.bias";
                const auto& gate_all = must(engine, gate_name);
                const auto& up_all = must(engine, up_name);
                const auto& down_all = must(engine, down_name);
                const auto& gate_scale_all = must(engine, gate_scale_name);
                const auto& up_scale_all = must(engine, up_scale_name);
                const auto& down_scale_all = must(engine, down_scale_name);
                const auto& b_gate_all = must(engine, gate_bias_name);
                const auto& b_up_all = must(engine, up_bias_name);
                const auto& b_down_all = must(engine, down_bias_name);

                const std::vector<std::int64_t> gate_scale_shape =
                    {E, I_native, H / 32};
                const std::vector<std::int64_t> down_scale_shape =
                    {E, H, I_native / 32};
                const std::vector<std::int64_t> gate_bias_shape = {E, I};
                const std::vector<std::int64_t> down_bias_shape = {E, H};
                if (gate_all.dtype() != DType::UINT8 ||
                    up_all.dtype() != DType::UINT8 ||
                    down_all.dtype() != DType::UINT8 ||
                    gate_scale_all.dtype() != DType::UINT8 ||
                    up_scale_all.dtype() != DType::UINT8 ||
                    down_scale_all.dtype() != DType::UINT8 ||
                    b_gate_all.dtype() != DType::BF16 ||
                    b_up_all.dtype() != DType::BF16 ||
                    b_down_all.dtype() != DType::BF16 ||
                    gate_all.nbytes() != static_cast<std::size_t>(E) * native_gate_expert_bytes ||
                    up_all.nbytes() != static_cast<std::size_t>(E) * native_gate_expert_bytes ||
                    down_all.nbytes() != static_cast<std::size_t>(E) * native_down_expert_bytes ||
                    gate_scale_all.shape() != gate_scale_shape ||
                    up_scale_all.shape() != gate_scale_shape ||
                    down_scale_all.shape() != down_scale_shape ||
                    b_gate_all.shape() != gate_bias_shape ||
                    b_up_all.shape() != gate_bias_shape ||
                    b_down_all.shape() != down_bias_shape) {
                    throw std::runtime_error(
                        "gpt_oss: native MXFP4 expert tensor shape mismatch at "
                        "layer " + std::to_string(li));
                }
                for (int e = 0; e < E; ++e) {
                    auto& Ew = Lw.experts[e];
                    const auto expert_idx = static_cast<std::size_t>(e);
                    Ew.format = MixtralExpertWeightFormat::Mxfp4NativeGemm;

                    w.owned_expert_buffers.push_back(tensor_view(
                        gate_all,
                        expert_idx * native_gate_expert_bytes,
                        {I_native, H / 2}));
                    Ew.w_gate_mxfp4 = &w.owned_expert_buffers.back();
                    w.owned_expert_buffers.push_back(tensor_view(
                        gate_scale_all,
                        expert_idx * native_gate_scale_expert_bytes,
                        {I_native, H / 32}));
                    Ew.w_gate_mxfp4_scale = &w.owned_expert_buffers.back();

                    w.owned_expert_buffers.push_back(tensor_view(
                        up_all,
                        expert_idx * native_gate_expert_bytes,
                        {I_native, H / 2}));
                    Ew.w_up_mxfp4 = &w.owned_expert_buffers.back();
                    w.owned_expert_buffers.push_back(tensor_view(
                        up_scale_all,
                        expert_idx * native_gate_scale_expert_bytes,
                        {I_native, H / 32}));
                    Ew.w_up_mxfp4_scale = &w.owned_expert_buffers.back();

                    w.owned_expert_buffers.push_back(tensor_view(
                        down_all,
                        expert_idx * native_down_expert_bytes,
                        {H, I_native / 2}));
                    Ew.w_down_mxfp4 = &w.owned_expert_buffers.back();
                    w.owned_expert_buffers.push_back(tensor_view(
                        down_scale_all,
                        expert_idx * native_down_scale_expert_bytes,
                        {H, I_native / 32}));
                    Ew.w_down_mxfp4_scale = &w.owned_expert_buffers.back();

                    w.owned_expert_buffers.push_back(tensor_view(
                        b_gate_all,
                        expert_idx * gate_up_bias_expert_bytes,
                        {I}));
                    Ew.b_gate = &w.owned_expert_buffers.back();
                    w.owned_expert_buffers.push_back(tensor_view(
                        b_up_all,
                        expert_idx * gate_up_bias_expert_bytes,
                        {I}));
                    Ew.b_up = &w.owned_expert_buffers.back();
                    w.owned_expert_buffers.push_back(tensor_view(
                        b_down_all,
                        expert_idx * down_bias_expert_bytes,
                        {H}));
                    Ew.b_down = &w.owned_expert_buffers.back();
                }
            } else if (GroupStreamCache* cache = engine.group_cache();
                       cache != nullptr &&
                       cache->find_group(p + "mlp.experts") !=
                           GroupStreamCache::kNoGroup) {
                // Streamed. The weights and their factors live in the group,
                // so there is nothing here to point at yet and the forward
                // fills those four fields from a slot. The biases were left
                // resident by the contract for the reason they always are:
                // the gate/up pair arrives interleaved and splitting it is a
                // kernel, which is host work a group plan cannot carry.
                const std::size_t g = cache->find_group(p + "mlp.experts");
                if (cache->arity(g) != static_cast<std::uint32_t>(E)) {
                    throw std::runtime_error(
                        "gpt_oss: expert group at layer " + std::to_string(li) +
                        " holds " + std::to_string(cache->arity(g)) +
                        " experts but the config says " + std::to_string(E));
                }
                Lw.expert_cache = cache;
                Lw.expert_group = g;

                const auto& b_gate_up_all =
                    must(engine, p + "mlp.experts.gate_up_proj.bias");
                const auto& b_down_all =
                    must(engine, p + "mlp.experts.down_proj.bias");
                for (int e = 0; e < E; ++e) {
                    auto& Ew = Lw.experts[e];
                    const auto expert_idx = static_cast<std::size_t>(e);
                    Ew.format = MixtralExpertWeightFormat::Mxfp4RoutedDequant;

                    auto gate_bias = DeviceTensor::allocate(DType::BF16, {I});
                    auto up_bias = DeviceTensor::allocate(DType::BF16, {I});
                    kernels::launch_deinterleave_vec_bf16(
                        static_cast<const std::uint8_t*>(b_gate_up_all.data()) +
                            expert_idx * mxfp4_gate_up_bias_expert_bytes,
                        gate_bias.data(), up_bias.data(), I, /*stream=*/0);
                    w.owned_expert_buffers.push_back(std::move(gate_bias));
                    Ew.b_gate = &w.owned_expert_buffers.back();
                    w.owned_expert_buffers.push_back(std::move(up_bias));
                    Ew.b_up = &w.owned_expert_buffers.back();

                    w.owned_expert_buffers.push_back(tensor_view(
                        b_down_all, expert_idx * down_bias_expert_bytes, {H}));
                    Ew.b_down = &w.owned_expert_buffers.back();
                }
            } else {
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
                    const auto expert_idx = static_cast<std::size_t>(e);
                    Ew.format = MixtralExpertWeightFormat::Mxfp4RoutedDequant;

                    w.owned_expert_buffers.push_back(tensor_view(
                        gate_up_all,
                        expert_idx * mxfp4_gate_up_expert_bytes,
                        {2 * I, H / 32, 16}));
                    Ew.w_gate_up = &w.owned_expert_buffers.back();

                    w.owned_expert_buffers.push_back(tensor_view(
                        gate_up_scale_all,
                        expert_idx * mxfp4_gate_up_scale_expert_bytes,
                        {2 * I, H / 32}));
                    Ew.w_gate_up_scale = &w.owned_expert_buffers.back();

                    w.owned_expert_buffers.push_back(tensor_view(
                        down_all,
                        expert_idx * mxfp4_down_expert_bytes,
                        {H, I / 32, 16}));
                    Ew.w_down_packed = &w.owned_expert_buffers.back();

                    w.owned_expert_buffers.push_back(tensor_view(
                        down_scale_all,
                        expert_idx * mxfp4_down_scale_expert_bytes,
                        {H, I / 32}));
                    Ew.w_down_scale = &w.owned_expert_buffers.back();

                    auto gate_bias = DeviceTensor::allocate(DType::BF16, {I});
                    auto up_bias = DeviceTensor::allocate(DType::BF16, {I});
                    kernels::launch_deinterleave_vec_bf16(
                        static_cast<const std::uint8_t*>(b_gate_up_all.data()) +
                            expert_idx * mxfp4_gate_up_bias_expert_bytes,
                        gate_bias.data(), up_bias.data(), I, /*stream=*/0);
                    w.owned_expert_buffers.push_back(std::move(gate_bias));
                    Ew.b_gate = &w.owned_expert_buffers.back();
                    w.owned_expert_buffers.push_back(std::move(up_bias));
                    Ew.b_up = &w.owned_expert_buffers.back();

                    w.owned_expert_buffers.push_back(tensor_view(
                        b_down_all,
                        expert_idx * down_bias_expert_bytes,
                        {H}));
                    Ew.b_down = &w.owned_expert_buffers.back();
                }
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

    if (use_mxfp4_packed_experts && !has_native_mxfp4_experts) {
        // Device-resident routing tables for the fused decode GEMV. Built
        // once here rather than per-step: the arrays are tiny, but staging
        // them from the host every layer would reintroduce exactly the
        // synchronisation the fused path exists to remove.
        for (auto& L : w.layers) {
            const std::size_t E = L.experts.size();
            std::vector<const std::uint8_t*> gu_packed(E), gu_scale(E);
            std::vector<const std::uint8_t*> dn_packed(E), dn_scale(E);
            std::vector<const void*> gb(E), ub(E), db(E);
            bool ok = true;
            // A streamed layer has no weight to point at yet -- the four
            // weight arrays are rewritten each step from whichever slots the
            // layer's routed set landed in. The biases are resident and stable,
            // so those three are final here, which is the whole reason the
            // per-step upload is 4 arrays and not 7.
            const bool streamed = L.expert_cache != nullptr;
            for (std::size_t e = 0; e < E; ++e) {
                const auto& Ew = L.experts[e];
                if (Ew.format != MixtralExpertWeightFormat::Mxfp4RoutedDequant ||
                    !Ew.b_gate || !Ew.b_up || !Ew.b_down ||
                    (!streamed &&
                     (!Ew.w_gate_up || !Ew.w_gate_up_scale ||
                      !Ew.w_down_packed || !Ew.w_down_scale))) {
                    ok = false;
                    break;
                }
                if (!streamed) {
                    gu_packed[e] = static_cast<const std::uint8_t*>(
                        Ew.w_gate_up->data());
                    gu_scale[e] = static_cast<const std::uint8_t*>(
                        Ew.w_gate_up_scale->data());
                    dn_packed[e] = static_cast<const std::uint8_t*>(
                        Ew.w_down_packed->data());
                    dn_scale[e] = static_cast<const std::uint8_t*>(
                        Ew.w_down_scale->data());
                }
                gb[e] = Ew.b_gate->data();
                ub[e] = Ew.b_up->data();
                db[e] = Ew.b_down->data();
            }
            if (!ok) continue;
            L.expert_gate_up_packed_ptrs =
                DeviceBuffer<const std::uint8_t*>::from_host(gu_packed);
            L.expert_gate_up_scale_ptrs =
                DeviceBuffer<const std::uint8_t*>::from_host(gu_scale);
            L.expert_down_packed_ptrs =
                DeviceBuffer<const std::uint8_t*>::from_host(dn_packed);
            L.expert_down_scale_ptrs =
                DeviceBuffer<const std::uint8_t*>::from_host(dn_scale);
            L.expert_gate_bias_ptrs = DeviceBuffer<const void*>::from_host(gb);
            L.expert_up_bias_ptrs = DeviceBuffer<const void*>::from_host(ub);
            L.expert_down_bias_ptrs = DeviceBuffer<const void*>::from_host(db);
        }
    }

    if (use_mxfp4_packed_experts) {
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return w;
}

}  // namespace pie_cuda_driver::model
