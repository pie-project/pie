#include "model/qwen3_5.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/gated_delta_net.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("qwen3_5: missing weight '" + name + "'");
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

bool mtp_int8_lm_head_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_INT8_LM_HEAD");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool fused_full_attn_qgkv_weights_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_FUSED_FULL_ATTN_QGKV");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}


// Materialise an owned fp32 copy of `t`, accepting either fp32 or bf16
// on disk. Copies even when the source is already fp32 so the bound
// pointer is owned by `Qwen3_5Weights::owned_fp32_buffers` (uniform
// lifetime).
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
        throw std::runtime_error("qwen3_5: unsupported dtype for fp32 conversion");
    }
    return buf;
}

// Qwen3.5 (multimodal config) nests the text tower under
// `model.language_model.`; the vision tower lives under `model.visual.`
// and is unused on the text-only path.
constexpr const char* kPrefix = "model.language_model.";

// Slice a fused [K1|K2|V] tensor along its leading `conv_dim` dimension
// into a per-rank tensor of shape [conv_dim_local, ...trailing] where
// conv_dim_local = 2*K_dim/T + V_dim/T. The trailing element count is
// the product of every shape[1:].
//
// Used for `linear_attn.in_proj_qkv.weight`, `linear_attn.conv1d.weight`,
// and `linear_attn.conv1d.bias` — their first axis stacks K-K-V blocks
// that don't shard cleanly under uniform axis-0 partitioning, so we copy
// the per-rank slice of each block by hand.
DeviceTensor slice_la_kkv_blocked(
    const DeviceTensor& full, int K_dim, int V_dim,
    int rank, int world_size)
{
    if (world_size <= 1) {
        // Caller shouldn't invoke this on a single rank — keep the path
        // explicit for readability.
        throw std::runtime_error("slice_la_kkv_blocked: world_size must be > 1");
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
        throw std::runtime_error(
            "slice_la_kkv_blocked: full.shape()[0]=" +
            std::to_string(full.shape().empty() ? 0
                                                : static_cast<int>(full.shape()[0])) +
            " does not match conv_dim=" + std::to_string(conv_dim));
    }
    // Trailing element count per row of the leading axis.
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
    // K1 block: full[0..K_dim) → sliced[0..K_local)
    copy_block(/*src=*/0,                  /*dst=*/0,           K_local);
    // K2 block: full[K_dim..2*K_dim)      → sliced[K_local..2*K_local)
    copy_block(/*src=*/K_dim,              /*dst=*/K_local,     K_local);
    // V  block: full[2*K_dim..conv_dim)   → sliced[2*K_local..conv_dim_local)
    copy_block(/*src=*/2 * K_dim,          /*dst=*/2 * K_local, V_local);
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
    if (first.dtype() != DType::BF16 ||
        second.dtype() != DType::BF16 ||
        third.dtype() != DType::BF16) {
        throw std::runtime_error(std::string(what) + ": expected bf16 tensors");
    }
    if (first.shape().empty() ||
        first.shape().size() != second.shape().size() ||
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

void maybe_fuse_full_attn_qgkv_projection(
    Qwen3_5Weights& w,
    Qwen3_5LayerWeights& Lw,
    const char* what)
{
    if (Lw.fa_q_proj == nullptr || Lw.fa_k_proj == nullptr ||
        Lw.fa_v_proj == nullptr ||
        Lw.fa_q_proj_quant.has_value() ||
        Lw.fa_k_proj_quant.has_value() ||
        Lw.fa_v_proj_quant.has_value() ||
        Lw.fa_q_proj->dtype() != DType::BF16 ||
        Lw.fa_k_proj->dtype() != DType::BF16 ||
        Lw.fa_v_proj->dtype() != DType::BF16) {
        return;
    }
    w.owned_bf16_buffers.push_back(
        concat_axis0_bf16(*Lw.fa_q_proj, *Lw.fa_k_proj, *Lw.fa_v_proj, what));
    Lw.fa_qgkv_proj_fused = &w.owned_bf16_buffers.back();
}

}  // namespace

Qwen3_5Weights bind_qwen3_5(LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    const int L = cfg.num_hidden_layers;

    if (cfg.layer_types.empty() ||
        static_cast<int>(cfg.layer_types.size()) != L) {
        throw std::runtime_error(
            "qwen3_5: HfConfig.layer_types must have num_hidden_layers entries");
    }
    if (cfg.linear_num_value_heads <= 0 || cfg.linear_num_key_heads <= 0
            || cfg.linear_key_head_dim <= 0 || cfg.linear_value_head_dim <= 0
            || cfg.linear_conv_kernel_dim <= 0) {
        throw std::runtime_error(
            "qwen3_5: linear-attn dimensions are unset; check the loader's "
            "HfConfig parsing for linear_num_*_heads / linear_*_head_dim / "
            "linear_conv_kernel_dim.");
    }

    Qwen3_5Weights w;
    w.layers.resize(static_cast<std::size_t>(L));

    const std::string p = kPrefix;

    w.embed      = &must(engine, p + "embed_tokens.weight");
    w.final_norm = &must(engine, p + "norm.weight");
    // Tied lm_head: HF omits the tensor and aliases to embed_tokens.
    w.lm_head    = cfg.tie_word_embeddings
                       ? w.embed
                       : &must(engine, "lm_head.weight");

    // KV cache slot is assigned only to full-attention layers, in
    // ascending order. Linear layers don't occupy KV-cache slots —
    // their state lives in the recurrent/conv caches built by the
    // forward.
    // owned_bf16_buffers must not reallocate after we hand out pointers
    // — `Lw.la_in_proj_qkv` etc. are observers into this vector. Reserve
    // up front for the worst case (3 sliced tensors + 2 fused GDN
    // tensors + fused full-attn QGKV, plus MTP) so push_back never
    // moves the storage.
    w.owned_bf16_buffers.reserve(static_cast<std::size_t>(L) * 6 + 4);

    int kv_slot = 0;
    for (int li = 0; li < L; ++li) {
        const std::string lp = p + "layers." + std::to_string(li) + ".";
        auto& Lw = w.layers[li];
        const auto& kind = cfg.layer_types[li];

        Lw.attn_norm_pre = &must(engine, lp + "input_layernorm.weight");
        Lw.mlp_norm_pre  = &must(engine, lp + "post_attention_layernorm.weight");

        // MLP weights are present on every layer (linear or full).
        Lw.gate_proj = &must(engine, lp + "mlp.gate_proj.weight");
        Lw.up_proj   = &must(engine, lp + "mlp.up_proj.weight");
        Lw.down_proj = &must(engine, lp + "mlp.down_proj.weight");
        Lw.gate_proj_quant = engine.quant_meta(lp + "mlp.gate_proj.weight");
        Lw.up_proj_quant   = engine.quant_meta(lp + "mlp.up_proj.weight");
        Lw.down_proj_quant = engine.quant_meta(lp + "mlp.down_proj.weight");
        if (Lw.gate_proj != nullptr && Lw.up_proj != nullptr &&
            !Lw.gate_proj_quant.has_value() && !Lw.up_proj_quant.has_value() &&
            Lw.gate_proj->dtype() == DType::BF16 &&
            Lw.up_proj->dtype() == DType::BF16) {
            w.owned_bf16_buffers.push_back(
                concat_axis0_bf16(*Lw.gate_proj, *Lw.up_proj,
                    "qwen3_5: fuse mlp.gate_up_proj"));
            Lw.gate_up_proj_fused = &w.owned_bf16_buffers.back();
            engine.erase_runtime_weight(lp + "mlp.gate_proj.weight");
            engine.erase_runtime_weight(lp + "mlp.up_proj.weight");
            Lw.gate_proj = nullptr;
            Lw.up_proj = nullptr;
        }

        if (kind == "linear_attention") {
            Lw.kind = Qwen3_5LayerWeights::Kind::LinearAttn;
            const std::string la = lp + "linear_attn.";
            const auto* full_qkv = &must(engine, la + "in_proj_qkv.weight");
            const auto* full_conv_w = &must(engine, la + "conv1d.weight");
            const auto* full_conv_b = maybe(engine, la + "conv1d.bias");
            // Slice the [K1|K2|V] block layout per-rank when TP > 1; the
            // engine load left these tensors replicated because uniform
            // axis-0 partitioning crosses block boundaries.
            const int T = std::max(1, engine.distributed().tp_size);
            const int rank = engine.distributed().tp_rank;
            const int K_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            const int V_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
            const auto qkv_quant = engine.quant_meta(la + "in_proj_qkv.weight");
            if (T > 1 && qkv_quant.has_value()) {
                throw std::runtime_error(
                    "qwen3_5: TP-sliced FP8 linear_attn.in_proj_qkv "
                    "requires matching scale slicing");
            }
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
                Lw.la_conv1d_w = full_conv_w;
                Lw.la_conv1d_b = full_conv_b;
            }
            Lw.la_in_proj_z   = &must(engine, la + "in_proj_z.weight");
            Lw.la_in_proj_b   = &must(engine, la + "in_proj_b.weight");
            Lw.la_in_proj_a   = &must(engine, la + "in_proj_a.weight");
            Lw.la_in_proj_qkv_quant = qkv_quant;
            Lw.la_in_proj_z_quant   = engine.quant_meta(la + "in_proj_z.weight");
            Lw.la_in_proj_b_quant   = engine.quant_meta(la + "in_proj_b.weight");
            Lw.la_in_proj_a_quant   = engine.quant_meta(la + "in_proj_a.weight");
            const bool can_fuse_gdn_projection_weights =
                fused_gdn_projection_weights_enabled() &&
                !Lw.la_in_proj_qkv_quant.has_value() &&
                !Lw.la_in_proj_z_quant.has_value() &&
                !Lw.la_in_proj_b_quant.has_value() &&
                !Lw.la_in_proj_a_quant.has_value();
            if (can_fuse_gdn_projection_weights) {
                w.owned_bf16_buffers.push_back(concat_axis0_bf16(
                    *Lw.la_in_proj_qkv, *Lw.la_in_proj_z,
                    "qwen3_5: fuse linear_attn.in_proj_qkvz"));
                Lw.la_in_proj_qkvz = &w.owned_bf16_buffers.back();
                engine.erase_runtime_weight(la + "in_proj_qkv.weight");
                engine.erase_runtime_weight(la + "in_proj_z.weight");
                w.owned_bf16_buffers.push_back(concat_axis0_bf16(
                    *Lw.la_in_proj_b, *Lw.la_in_proj_a,
                    "qwen3_5: fuse linear_attn.in_proj_ba"));
                Lw.la_in_proj_ba = &w.owned_bf16_buffers.back();
                engine.erase_runtime_weight(la + "in_proj_b.weight");
                engine.erase_runtime_weight(la + "in_proj_a.weight");
            }
            Lw.la_dt_bias  = &must(engine, la + "dt_bias");
            // Materialise fp32 copies of A_log + RMSNormGated.weight.
            // HF ships these as fp32 on Qwen3.5-4B and bf16 on
            // Qwen3.6-35B-A3B; the kernels read fp32 either way.
            w.owned_fp32_buffers.push_back(to_fp32(must(engine, la + "A_log")));
            Lw.la_A_log_fp32  = w.owned_fp32_buffers.back().data();
            w.owned_fp32_buffers.push_back(to_fp32(must(engine, la + "norm.weight")));
            Lw.la_norm_w_fp32 = w.owned_fp32_buffers.back().data();
            Lw.la_out_proj = &must(engine, la + "out_proj.weight");
            Lw.la_out_proj_quant = engine.quant_meta(la + "out_proj.weight");
            Lw.kv_layer = -1;
        } else if (kind == "full_attention") {
            Lw.kind = Qwen3_5LayerWeights::Kind::FullAttn;
            const std::string fa = lp + "self_attn.";
            // q_proj is 2× wide (query + gate fused).
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
            if (fused_full_attn_qgkv_weights_enabled()) {
                maybe_fuse_full_attn_qgkv_projection(
                    w, Lw, "qwen3_5: fuse self_attn.qgkv_proj");
            }
            Lw.kv_layer = kv_slot++;
        } else {
            throw std::runtime_error(
                "qwen3_5: unknown layer_type '" + kind + "' at layer " +
                std::to_string(li));
        }
    }

    if (cfg.mtp_num_hidden_layers > 0 && engine.has("mtp.fc.weight")) {
        Qwen3_5Weights::MtpWeights mtp;
        mtp.pre_fc_norm_embedding = &must(engine, "mtp.pre_fc_norm_embedding.weight");
        mtp.pre_fc_norm_hidden = &must(engine, "mtp.pre_fc_norm_hidden.weight");
        mtp.fc = &must(engine, "mtp.fc.weight");
        mtp.norm = &must(engine, "mtp.norm.weight");
        mtp.embed = cfg.mtp_use_dedicated_embeddings
            ? &must(engine, "mtp.embed_tokens.weight")
            : w.embed;
        mtp.lm_head = w.lm_head;

        const std::string lp = "mtp.layers.0.";
        auto& Lw = mtp.layer;
        Lw.kind = Qwen3_5LayerWeights::Kind::FullAttn;
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
        if (fused_full_attn_qgkv_weights_enabled()) {
            maybe_fuse_full_attn_qgkv_projection(
                w, Lw, "qwen3_5: fuse mtp.self_attn.qgkv_proj");
        }
        Lw.gate_proj = &must(engine, lp + "mlp.gate_proj.weight");
        Lw.up_proj = &must(engine, lp + "mlp.up_proj.weight");
        Lw.down_proj = &must(engine, lp + "mlp.down_proj.weight");
        Lw.gate_proj_quant = engine.quant_meta(lp + "mlp.gate_proj.weight");
        Lw.up_proj_quant = engine.quant_meta(lp + "mlp.up_proj.weight");
        Lw.down_proj_quant = engine.quant_meta(lp + "mlp.down_proj.weight");
        if (Lw.gate_proj != nullptr && Lw.up_proj != nullptr &&
            !Lw.gate_proj_quant.has_value() && !Lw.up_proj_quant.has_value() &&
            Lw.gate_proj->dtype() == DType::BF16 &&
            Lw.up_proj->dtype() == DType::BF16) {
            w.owned_bf16_buffers.push_back(
                concat_axis0_bf16(*Lw.gate_proj, *Lw.up_proj,
                    "qwen3_5: fuse mtp.mlp.gate_up_proj"));
            Lw.gate_up_proj_fused = &w.owned_bf16_buffers.back();
        }
        Lw.kv_layer = kv_slot++;
        if (mtp_int8_lm_head_enabled() &&
            mtp.lm_head != nullptr &&
            mtp.lm_head->dtype() == DType::BF16) {
            const auto& shape = mtp.lm_head->shape();
            const int rows = static_cast<int>(shape[0]); // vocab
            const int cols = static_cast<int>(shape[1]); // hidden
            w.owned_int8_buffers.push_back(
                DeviceTensor::allocate(DType::INT8, shape));
            w.owned_scale_buffers.push_back(
                DeviceBuffer<float>::alloc(static_cast<std::size_t>(rows)));
            kernels::quantize_bf16_to_int8_per_channel(
                mtp.lm_head->data(),
                static_cast<std::int8_t*>(w.owned_int8_buffers.back().data()),
                w.owned_scale_buffers.back().data(),
                rows, cols, /*stream=*/0);
            CUDA_CHECK(cudaStreamSynchronize(0));
            mtp.lm_head = &w.owned_int8_buffers.back();
            mtp.lm_head_scale_inv = &w.owned_scale_buffers.back();
        }
        w.mtp = mtp;
    }

    return w;
}

}  // namespace pie_cuda_driver::model
