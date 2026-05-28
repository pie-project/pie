#include "model/glm5.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/dequant_fp8.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("glm5: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe(const LoadedModel& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}

void require_rank2(const DeviceTensor& t, const std::string& name) {
    if (t.shape().size() != 2) {
        throw std::runtime_error("glm5: expected rank-2 tensor for '" + name + "'");
    }
}

// Materialise a BF16 copy of a possibly-FP8 weight tensor. The kimi_mla
// kernels (`launch_kimi_q_nope_to_latent_bf16`, `launch_kimi_latent_to_v_bf16`)
// require kv_b_proj in BF16. GLM-5.1 ships kv_b_proj as FP8_E4M3 with a
// per-channel `weight_scale_inv`; we dequantise once at bind time.
DeviceTensor materialise_bf16(
    const DeviceTensor& w,
    const std::optional<QuantMeta>& meta,
    const std::string& name)
{
    if (w.dtype() == DType::BF16) {
        // Already BF16: copy into a new owned tensor so the caller has a
        // uniform DeviceTensor handle.
        auto out = DeviceTensor::allocate(DType::BF16, w.shape());
        CUDA_CHECK(cudaMemcpy(out.data(), w.data(), w.nbytes(),
                              cudaMemcpyDeviceToDevice));
        return out;
    }
    if (w.dtype() != DType::FP8_E4M3) {
        throw std::runtime_error(
            "glm5: kv_b_proj has unsupported dtype " +
            std::string(dtype_name(w.dtype())) + " for '" + name + "'");
    }
    if (!meta.has_value()) {
        throw std::runtime_error(
            "glm5: FP8 weight '" + name + "' has no QuantMeta companion");
    }
    if (meta->scale == nullptr) {
        throw std::runtime_error(
            "glm5: FP8 weight '" + name + "' has null scale tensor");
    }
    if (w.shape().size() != 2) {
        throw std::runtime_error(
            "glm5: FP8 weight '" + name + "' must be rank-2");
    }
    const int rows = static_cast<int>(w.shape()[0]);
    const int cols = static_cast<int>(w.shape()[1]);
    auto out = DeviceTensor::allocate(DType::BF16, w.shape());
    if (meta->kind == QuantMeta::Kind::PerGroup && meta->group_size > 0) {
        kernels::launch_dequant_fp8_e4m3_to_bf16_per_group(
            static_cast<const std::uint8_t*>(w.data()),
            out.data(),
            static_cast<const float*>(meta->scale->data()),
            rows, cols, meta->group_size, /*stream=*/0);
    } else if (meta->kind == QuantMeta::Kind::PerChannel) {
        kernels::launch_dequant_fp8_e4m3_to_bf16_per_channel(
            static_cast<const std::uint8_t*>(w.data()),
            out.data(),
            static_cast<const float*>(meta->scale->data()),
            rows, cols, /*stream=*/0);
    } else {
        throw std::runtime_error(
            "glm5: FP8 weight '" + name +
            "' has unsupported quant kind");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

}  // namespace

Glm5Weights bind_glm5(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    if (cfg.q_lora_rank <= 0 || cfg.kv_lora_rank <= 0 ||
        cfg.qk_nope_head_dim <= 0 || cfg.qk_rope_head_dim <= 0 ||
        cfg.v_head_dim <= 0) {
        throw std::runtime_error("glm5: MLA dimensions are missing from HfConfig");
    }
    if (cfg.num_experts <= 0 || cfg.num_experts_per_tok <= 0 ||
        cfg.moe_intermediate_size <= 0) {
        throw std::runtime_error("glm5: MoE dimensions are missing from HfConfig");
    }

    Glm5Weights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error("glm5: lm_head missing and tie_word_embeddings=false");
    }

    require_rank2(*w.embed, "model.embed_tokens.weight");
    require_rank2(*w.lm_head, "lm_head.weight");
    if (w.embed->shape()[0] == cfg.vocab_size) {
        w.embed_tp_vocab_offset = 0;
        w.embed_tp_sharded = false;
    } else if (engine.distributed().tp_size > 1 &&
               w.embed->shape()[0] * engine.distributed().tp_size == cfg.vocab_size) {
        w.embed_tp_vocab_offset =
            static_cast<int>(w.embed->shape()[0] * engine.distributed().tp_rank);
        w.embed_tp_sharded = true;
    } else {
        throw std::runtime_error("glm5: embed row count does not match vocab or TP shard");
    }
    if (w.lm_head->shape()[0] == cfg.vocab_size) {
        w.lm_head_tp_vocab_offset = 0;
        w.lm_head_tp_sharded = false;
    } else if (engine.distributed().tp_size > 1 &&
               w.lm_head->shape()[0] * engine.distributed().tp_size == cfg.vocab_size) {
        w.lm_head_tp_vocab_offset =
            static_cast<int>(w.lm_head->shape()[0] * engine.distributed().tp_rank);
        w.lm_head_tp_sharded = true;
    } else {
        throw std::runtime_error("glm5: lm_head row count does not match vocab or TP shard");
    }

    const int T = std::max(1, engine.distributed().tp_size);
    const int local_heads = cfg.num_attention_heads / T;
    const int q_b_rows = local_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
    const int kv_b_rows = local_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);

    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int li = 0; li < cfg.num_hidden_layers; ++li) {
        const std::string lp = "model.layers." + std::to_string(li) + ".";
        auto& L = w.layers[static_cast<std::size_t>(li)];

        L.attn_norm = &must(engine, lp + "input_layernorm.weight");
        L.mlp_norm  = &must(engine, lp + "post_attention_layernorm.weight");

        const std::string ap = lp + "self_attn.";
        L.q_a_proj           = &must(engine, ap + "q_a_proj.weight");
        L.q_a_norm           = &must(engine, ap + "q_a_layernorm.weight");
        L.q_b_proj           = &must(engine, ap + "q_b_proj.weight");
        L.kv_a_proj_with_mqa = &must(engine, ap + "kv_a_proj_with_mqa.weight");
        L.kv_a_norm          = &must(engine, ap + "kv_a_layernorm.weight");
        L.kv_b_proj          = &must(engine, ap + "kv_b_proj.weight");
        L.o_proj             = &must(engine, ap + "o_proj.weight");

        L.q_a_proj_quant            = engine.quant_meta(ap + "q_a_proj.weight");
        L.q_b_proj_quant            = engine.quant_meta(ap + "q_b_proj.weight");
        L.kv_a_proj_with_mqa_quant  = engine.quant_meta(ap + "kv_a_proj_with_mqa.weight");
        L.kv_b_proj_quant           = engine.quant_meta(ap + "kv_b_proj.weight");
        L.o_proj_quant              = engine.quant_meta(ap + "o_proj.weight");

        require_rank2(*L.q_a_proj, ap + "q_a_proj.weight");
        require_rank2(*L.q_b_proj, ap + "q_b_proj.weight");
        require_rank2(*L.kv_a_proj_with_mqa, ap + "kv_a_proj_with_mqa.weight");
        require_rank2(*L.kv_b_proj, ap + "kv_b_proj.weight");
        require_rank2(*L.o_proj, ap + "o_proj.weight");

        if (L.q_b_proj->shape()[0] != q_b_rows) {
            throw std::runtime_error("glm5: q_b_proj TP row count mismatch at layer " +
                                     std::to_string(li));
        }
        if (L.kv_b_proj->shape()[0] != kv_b_rows) {
            throw std::runtime_error("glm5: kv_b_proj TP row count mismatch at layer " +
                                     std::to_string(li));
        }

        // The kimi_mla kernels read kv_b_proj as a BF16 tensor; materialise
        // a BF16 copy when the on-disk weight is FP8. For BF16 checkpoints
        // we still copy, so the kernel always sees an owned BF16 view.
        L.kv_b_proj_bf16 = std::make_unique<DeviceTensor>(
            materialise_bf16(*L.kv_b_proj, L.kv_b_proj_quant,
                             ap + "kv_b_proj.weight"));

        // DSA lightning-indexer weights (glm_moe_dsa). Present on every layer.
        if (engine.has(ap + "indexer.wq_b.weight")) {
            L.idx_wq_b = &must(engine, ap + "indexer.wq_b.weight");
            L.idx_wk = &must(engine, ap + "indexer.wk.weight");
            L.idx_weights_proj = &must(engine, ap + "indexer.weights_proj.weight");
            L.idx_k_norm_weight = &must(engine, ap + "indexer.k_norm.weight");
            L.idx_k_norm_bias = &must(engine, ap + "indexer.k_norm.bias");
            L.idx_wq_b_quant = engine.quant_meta(ap + "indexer.wq_b.weight");
            L.idx_wk_quant = engine.quant_meta(ap + "indexer.wk.weight");
        }

        L.is_moe = li >= cfg.first_k_dense_replace;
        const std::string mp = lp + "mlp.";
        if (!L.is_moe) {
            L.dense_gate_proj = &must(engine, mp + "gate_proj.weight");
            L.dense_up_proj   = &must(engine, mp + "up_proj.weight");
            L.dense_down_proj = &must(engine, mp + "down_proj.weight");
            L.dense_gate_quant = engine.quant_meta(mp + "gate_proj.weight");
            L.dense_up_quant   = engine.quant_meta(mp + "up_proj.weight");
            L.dense_down_quant = engine.quant_meta(mp + "down_proj.weight");
            require_rank2(*L.dense_gate_proj, mp + "gate_proj.weight");
            require_rank2(*L.dense_up_proj, mp + "up_proj.weight");
            require_rank2(*L.dense_down_proj, mp + "down_proj.weight");
            continue;
        }

        L.router = &must(engine, mp + "gate.weight");
        require_rank2(*L.router, mp + "gate.weight");
        L.e_score_correction_bias =
            maybe(engine, mp + "gate.e_score_correction_bias");

        L.experts.resize(static_cast<std::size_t>(cfg.num_experts));
        for (int e = 0; e < cfg.num_experts; ++e) {
            const std::string ep =
                mp + "experts." + std::to_string(e) + ".";
            auto& Ew = L.experts[static_cast<std::size_t>(e)];
            Ew.gate_proj = &must(engine, ep + "gate_proj.weight");
            Ew.up_proj   = &must(engine, ep + "up_proj.weight");
            Ew.down_proj = &must(engine, ep + "down_proj.weight");
            Ew.gate_quant = engine.quant_meta(ep + "gate_proj.weight");
            Ew.up_quant   = engine.quant_meta(ep + "up_proj.weight");
            Ew.down_quant = engine.quant_meta(ep + "down_proj.weight");
            // Expert weights may be packed nibbles (MXFP4) stored as
            // a 1-D byte buffer — skip the rank-2 check in that case.
            // The forward path reads logical rows/cols from the call
            // site (Ne, routed_I, H) and uses the QuantMeta scale.
            auto is_packed = [](const DeviceTensor* t) {
                return t->dtype() == DType::MXFP4_PACKED ||
                       t->dtype() == DType::INT4_PACKED ||
                       t->dtype() == DType::UINT8 ||
                       t->shape().size() == 1;
            };
            if (!is_packed(Ew.gate_proj))
                require_rank2(*Ew.gate_proj, ep + "gate_proj.weight");
            if (!is_packed(Ew.up_proj))
                require_rank2(*Ew.up_proj, ep + "up_proj.weight");
            if (!is_packed(Ew.down_proj))
                require_rank2(*Ew.down_proj, ep + "down_proj.weight");
        }

        if (cfg.n_shared_experts > 0) {
            const std::string sp = mp + "shared_experts.";
            L.shared_gate_proj = &must(engine, sp + "gate_proj.weight");
            L.shared_up_proj   = &must(engine, sp + "up_proj.weight");
            L.shared_down_proj = &must(engine, sp + "down_proj.weight");
            L.shared_gate_quant = engine.quant_meta(sp + "gate_proj.weight");
            L.shared_up_quant   = engine.quant_meta(sp + "up_proj.weight");
            L.shared_down_quant = engine.quant_meta(sp + "down_proj.weight");
            // shared_experts go through runtime fp4 quant too: skip
            // rank-2 check for packed/byte-buffer forms.
            auto is_packed = [](const DeviceTensor* t) {
                return t->dtype() == DType::MXFP4_PACKED ||
                       t->dtype() == DType::INT4_PACKED ||
                       t->dtype() == DType::UINT8 ||
                       t->shape().size() == 1;
            };
            if (!is_packed(L.shared_gate_proj))
                require_rank2(*L.shared_gate_proj, sp + "gate_proj.weight");
            if (!is_packed(L.shared_up_proj))
                require_rank2(*L.shared_up_proj, sp + "up_proj.weight");
            if (!is_packed(L.shared_down_proj))
                require_rank2(*L.shared_down_proj, sp + "down_proj.weight");
        }
    }

    return w;
}

}  // namespace pie_cuda_driver::model
