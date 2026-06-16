#include "model/qwen3_5_moe.hpp"

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

bool fused_gdn_projection_weights_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_FUSED_GDN_PROJ");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool fused_shared_scalar_gate_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_FUSED_SHARED_SCALAR_GATE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

// Materialise an owned fp32 copy of `t`. If `t` is already fp32 we still
// copy (kernels read through `const float*` regardless of source layout).
DeviceBuffer<float> to_fp32(const DeviceTensor& t) {
    std::size_t n = 1;
    for (auto d : t.shape()) n *= static_cast<std::size_t>(d);
    auto buf = DeviceBuffer<float>::alloc(n);
    if (t.dtype() == DType::FP32) {
        CUDA_CHECK(cudaMemcpy(buf.data(), t.data(),
                              n * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    } else if (t.dtype() == DType::BF16) {
        kernels::launch_bf16_to_fp32(t.data(), buf.data(), n, /*stream=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        throw std::runtime_error(
            "qwen3_5_moe: unsupported dtype for fp32 conversion");
    }
    return buf;
}

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

// Per-rank slice of fused linear-attn QKV / conv1d tensors. Shares the
// same [K1 | K2 | V] block layout as Qwen3_5; copy the helper here
// rather than expose it across translation units.
DeviceTensor slice_la_kkv_blocked(
    const DeviceTensor& full, int K_dim, int V_dim,
    int rank, int world_size)
{
    if (world_size <= 1) {
        throw std::runtime_error(
            "slice_la_kkv_blocked: world_size must be > 1");
    }
    if (K_dim % world_size != 0 || V_dim % world_size != 0) {
        throw std::runtime_error(
            "slice_la_kkv_blocked: K_dim/V_dim must divide world_size");
    }
    const int K_local = K_dim / world_size;
    const int V_local = V_dim / world_size;
    const int conv_dim       = 2 * K_dim   + V_dim;
    const int conv_dim_local = 2 * K_local + V_local;
    if (full.shape().empty() ||
        static_cast<int>(full.shape()[0]) != conv_dim) {
        throw std::runtime_error("slice_la_kkv_blocked: leading-dim mismatch");
    }
    std::size_t trailing = 1;
    for (std::size_t i = 1; i < full.shape().size(); ++i) {
        trailing *= static_cast<std::size_t>(full.shape()[i]);
    }
    const std::size_t row_bytes = trailing * dtype_bytes(full.dtype());
    std::vector<std::int64_t> out_shape = full.shape();
    out_shape[0] = conv_dim_local;
    auto sliced = DeviceTensor::allocate(full.dtype(), out_shape);
    const auto* src = static_cast<const std::uint8_t*>(full.data());
    auto* dst = static_cast<std::uint8_t*>(sliced.data());
    auto copy_block = [&](int src_offset_rows, int dst_offset_rows,
                          int rows_per_rank) {
        const std::size_t src_off =
            static_cast<std::size_t>(src_offset_rows + rank * rows_per_rank) *
            row_bytes;
        const std::size_t dst_off =
            static_cast<std::size_t>(dst_offset_rows) * row_bytes;
        const std::size_t bytes =
            static_cast<std::size_t>(rows_per_rank) * row_bytes;
        CUDA_CHECK(cudaMemcpy(dst + dst_off, src + src_off, bytes,
                              cudaMemcpyDeviceToDevice));
    };
    copy_block(0,             0,           K_local);
    copy_block(K_dim,         K_local,     K_local);
    copy_block(2 * K_dim,     2 * K_local, V_local);
    return sliced;
}

DeviceTensor concat_axis0_bf16(
    const DeviceTensor& first,
    const DeviceTensor& second,
    const char* what)
{
    if (first.dtype() != DType::BF16 || second.dtype() != DType::BF16) {
        throw std::runtime_error(std::string(what) + ": expected bf16 tensors");
    }
    if (first.shape().empty() || first.shape().size() != second.shape().size()) {
        throw std::runtime_error(std::string(what) + ": rank mismatch");
    }
    for (std::size_t i = 1; i < first.shape().size(); ++i) {
        if (first.shape()[i] != second.shape()[i]) {
            throw std::runtime_error(std::string(what) + ": trailing shape mismatch");
        }
    }

    std::vector<std::int64_t> shape = first.shape();
    shape[0] += second.shape()[0];
    auto fused = DeviceTensor::allocate(DType::BF16, shape);
    auto* dst = static_cast<std::uint8_t*>(fused.data());
    CUDA_CHECK(cudaMemcpy(
        dst, first.data(), first.nbytes(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        dst + first.nbytes(), second.data(), second.nbytes(),
        cudaMemcpyDeviceToDevice));
    return fused;
}

DeviceTensor concat_axis0_bf16(
    const DeviceTensor& first,
    const DeviceTensor& second,
    const DeviceTensor& third,
    const char* what)
{
    if (first.dtype() != DType::BF16 || second.dtype() != DType::BF16 ||
        third.dtype() != DType::BF16) {
        throw std::runtime_error(std::string(what) + ": expected bf16 tensors");
    }
    if (first.shape().empty() || first.shape().size() != second.shape().size() ||
        first.shape().size() != third.shape().size()) {
        throw std::runtime_error(std::string(what) + ": rank mismatch");
    }
    for (std::size_t i = 1; i < first.shape().size(); ++i) {
        if (first.shape()[i] != second.shape()[i] ||
            first.shape()[i] != third.shape()[i]) {
            throw std::runtime_error(std::string(what) + ": trailing shape mismatch");
        }
    }

    std::vector<std::int64_t> shape = first.shape();
    shape[0] += second.shape()[0] + third.shape()[0];
    auto fused = DeviceTensor::allocate(DType::BF16, shape);
    auto* dst = static_cast<std::uint8_t*>(fused.data());
    CUDA_CHECK(cudaMemcpy(
        dst, first.data(), first.nbytes(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        dst + first.nbytes(), second.data(), second.nbytes(),
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        dst + first.nbytes() + second.nbytes(), third.data(), third.nbytes(),
        cudaMemcpyDeviceToDevice));
    return fused;
}

}  // namespace

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
    w.owned_bf16_buffers.reserve(static_cast<std::size_t>(L) * 8);

    const int T = std::max(1, engine.distributed().tp_size);
    const int rank = engine.distributed().tp_rank;
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
            const auto* full_qkv = &must(engine, la + "in_proj_qkv.weight");
            const auto* full_conv_w = &must(engine, la + "conv1d.weight");
            const auto* full_conv_b = maybe(engine, la + "conv1d.bias");
            // Linear-attn fused QKV / conv1d use the [K1 | K2 | V] block
            // layout — same custom slicing as Qwen3_5 dense.
            const int K_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            const int V_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
            if (T > 1) {
                w.owned_bf16_buffers.push_back(
                    slice_la_kkv_blocked(*full_qkv, K_dim, V_dim, rank, T));
                Lw.la_in_proj_qkv = &w.owned_bf16_buffers.back();
                w.owned_bf16_buffers.push_back(
                    slice_la_kkv_blocked(*full_conv_w, K_dim, V_dim, rank, T));
                Lw.la_conv1d_w = &w.owned_bf16_buffers.back();
                if (full_conv_b) {
                    w.owned_bf16_buffers.push_back(
                        slice_la_kkv_blocked(*full_conv_b, K_dim, V_dim, rank, T));
                    Lw.la_conv1d_b = &w.owned_bf16_buffers.back();
                } else {
                    Lw.la_conv1d_b = nullptr;
                }
            } else {
                Lw.la_in_proj_qkv = full_qkv;
                Lw.la_conv1d_w    = full_conv_w;
                Lw.la_conv1d_b    = full_conv_b;
            }
            Lw.la_in_proj_z   = &must(engine, la + "in_proj_z.weight");
            Lw.la_in_proj_b   = &must(engine, la + "in_proj_b.weight");
            Lw.la_in_proj_a   = &must(engine, la + "in_proj_a.weight");
            if (fused_gdn_projection_weights_enabled()) {
                w.owned_bf16_buffers.push_back(concat_axis0_bf16(
                    *Lw.la_in_proj_qkv, *Lw.la_in_proj_z,
                    "qwen3_5_moe: fuse linear_attn.in_proj_qkvz"));
                Lw.la_in_proj_qkvz = &w.owned_bf16_buffers.back();
                w.owned_bf16_buffers.push_back(concat_axis0_bf16(
                    *Lw.la_in_proj_b, *Lw.la_in_proj_a,
                    "qwen3_5_moe: fuse linear_attn.in_proj_ba"));
                Lw.la_in_proj_ba = &w.owned_bf16_buffers.back();
            }
            Lw.la_dt_bias     = &must(engine, la + "dt_bias");
            // Materialise fp32 copies of A_log + RMSNormGated weight so
            // the kernel signature stays uniform across Qwen3.5 (fp32 on
            // disk) and Qwen3.6-MoE (bf16 on disk).
            w.owned_fp32_buffers.push_back(to_fp32(must(engine, la + "A_log")));
            Lw.la_A_log_fp32 = w.owned_fp32_buffers.back().data();
            w.owned_fp32_buffers.push_back(to_fp32(must(engine, la + "norm.weight")));
            Lw.la_norm_w_fp32 = w.owned_fp32_buffers.back().data();
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
        Lw.moe_gate_up_proj = &must(engine, lp + "mlp.experts.gate_up_proj");
        Lw.moe_down_proj    = &must(engine, lp + "mlp.experts.down_proj");
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
            if (!Lw.shared_gate_proj_quant.has_value() &&
                !Lw.shared_up_proj_quant.has_value() &&
                Lw.shared_gate_proj->dtype() == DType::BF16 &&
                Lw.shared_up_proj->dtype() == DType::BF16) {
                if (fused_shared_scalar_gate_enabled() &&
                    !Lw.shared_gate_quant.has_value() &&
                    Lw.shared_gate != nullptr &&
                    Lw.shared_gate->dtype() == DType::BF16) {
                    w.owned_bf16_buffers.push_back(concat_axis0_bf16(
                        *Lw.shared_gate_proj, *Lw.shared_up_proj, *Lw.shared_gate,
                        "qwen3_5_moe: fuse mlp.shared_expert.gate_up_gate_proj"));
                    Lw.shared_gate_up_gate_proj = &w.owned_bf16_buffers.back();
                } else {
                    w.owned_bf16_buffers.push_back(concat_axis0_bf16(
                        *Lw.shared_gate_proj, *Lw.shared_up_proj,
                        "qwen3_5_moe: fuse mlp.shared_expert.gate_up_proj"));
                    Lw.shared_gate_up_proj = &w.owned_bf16_buffers.back();
                }
            }
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
        Lw.moe_gate_up_proj = &must(engine, lp + "mlp.experts.gate_up_proj");
        Lw.moe_down_proj = &must(engine, lp + "mlp.experts.down_proj");
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
            if (!Lw.shared_gate_proj_quant.has_value() &&
                !Lw.shared_up_proj_quant.has_value() &&
                Lw.shared_gate_proj->dtype() == DType::BF16 &&
                Lw.shared_up_proj->dtype() == DType::BF16) {
                if (fused_shared_scalar_gate_enabled() &&
                    !Lw.shared_gate_quant.has_value() &&
                    Lw.shared_gate != nullptr &&
                    Lw.shared_gate->dtype() == DType::BF16) {
                    w.owned_bf16_buffers.push_back(concat_axis0_bf16(
                        *Lw.shared_gate_proj, *Lw.shared_up_proj, *Lw.shared_gate,
                        "qwen3_5_moe: fuse mtp.shared_expert.gate_up_gate_proj"));
                    Lw.shared_gate_up_gate_proj = &w.owned_bf16_buffers.back();
                } else {
                    w.owned_bf16_buffers.push_back(concat_axis0_bf16(
                        *Lw.shared_gate_proj, *Lw.shared_up_proj,
                        "qwen3_5_moe: fuse mtp.shared_expert.gate_up_proj"));
                    Lw.shared_gate_up_proj = &w.owned_bf16_buffers.back();
                }
            }
        }
        Lw.kv_layer = kv_slot++;
        w.mtp = mtp;
    }

    return w;
}

}  // namespace pie_cuda_driver::model
