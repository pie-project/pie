#include "model/kimi/kimi.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("kimi: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe(const LoadedModel& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}

void require_rank2(const DeviceTensor& t, const std::string& name) {
    if (t.shape().size() != 2) {
        throw std::runtime_error("kimi: expected rank-2 tensor for '" + name + "'");
    }
}

void require_dtype(const DeviceTensor& t, DType dtype, const std::string& name) {
    if (t.dtype() != dtype) {
        throw std::runtime_error(
            "kimi: expected " + std::string(dtype_name(dtype)) + " tensor for '" +
            name + "', got " + dtype_name(t.dtype()));
    }
}

void bind_expert(
    const LoadedModel& engine,
    const std::string& ep,
    KimiExpertWeights& ew)
{
    ew.gate_packed = &must(engine, ep + "gate_proj.weight_packed");
    ew.gate_scale  = &must(engine, ep + "gate_proj.weight_scale");
    ew.gate_shape  = &must(engine, ep + "gate_proj.weight_shape");
    ew.up_packed   = &must(engine, ep + "up_proj.weight_packed");
    ew.up_scale    = &must(engine, ep + "up_proj.weight_scale");
    ew.up_shape    = &must(engine, ep + "up_proj.weight_shape");
    ew.down_packed = &must(engine, ep + "down_proj.weight_packed");
    ew.down_scale  = &must(engine, ep + "down_proj.weight_scale");
    ew.down_shape  = &must(engine, ep + "down_proj.weight_shape");

    require_dtype(*ew.gate_packed, DType::INT32, ep + "gate_proj.weight_packed");
    require_dtype(*ew.up_packed, DType::INT32, ep + "up_proj.weight_packed");
    require_dtype(*ew.down_packed, DType::INT32, ep + "down_proj.weight_packed");
    require_dtype(*ew.gate_scale, DType::BF16, ep + "gate_proj.weight_scale");
    require_dtype(*ew.up_scale, DType::BF16, ep + "up_proj.weight_scale");
    require_dtype(*ew.down_scale, DType::BF16, ep + "down_proj.weight_scale");
}

}  // namespace

KimiWeights bind_kimi(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    if (cfg.q_lora_rank <= 0 || cfg.kv_lora_rank <= 0 ||
        cfg.qk_nope_head_dim <= 0 || cfg.qk_rope_head_dim <= 0 ||
        cfg.v_head_dim <= 0) {
        throw std::runtime_error("kimi: MLA dimensions are missing from HfConfig");
    }
    if (cfg.num_experts <= 0 || cfg.num_experts_per_tok <= 0 ||
        cfg.moe_intermediate_size <= 0) {
        throw std::runtime_error("kimi: MoE dimensions are missing from HfConfig");
    }

    KimiWeights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error("kimi: lm_head missing and tie_word_embeddings=false");
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
        throw std::runtime_error("kimi: embed row count does not match vocab or TP shard");
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
        throw std::runtime_error("kimi: lm_head row count does not match vocab or TP shard");
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
        L.q_kv_a_fused = maybe(engine, ap + "q_kv_a_proj.fused.weight");
        if (L.q_kv_a_fused != nullptr) {
            L.q_a_proj = nullptr;
            L.kv_a_proj_with_mqa = nullptr;
        } else {
            L.q_a_proj = &must(engine, ap + "q_a_proj.weight");
            L.kv_a_proj_with_mqa = &must(engine, ap + "kv_a_proj_with_mqa.weight");
        }
        L.q_a_norm = &must(engine, ap + "q_a_layernorm.weight");
        L.q_b_proj = &must(engine, ap + "q_b_proj.weight");
        L.kv_a_norm = &must(engine, ap + "kv_a_layernorm.weight");
        L.kv_b_proj = &must(engine, ap + "kv_b_proj.weight");
        L.o_proj    = &must(engine, ap + "o_proj.weight");

        require_rank2(*L.q_a_proj, ap + "q_a_proj.weight");
        require_rank2(*L.q_b_proj, ap + "q_b_proj.weight");
        require_rank2(*L.kv_a_proj_with_mqa, ap + "kv_a_proj_with_mqa.weight");
        require_rank2(*L.kv_b_proj, ap + "kv_b_proj.weight");
        require_rank2(*L.o_proj, ap + "o_proj.weight");

        if (L.q_b_proj->shape()[0] != q_b_rows) {
            throw std::runtime_error("kimi: q_b_proj TP row count mismatch at layer " +
                                     std::to_string(li));
        }
        if (L.kv_b_proj->shape()[0] != kv_b_rows) {
            throw std::runtime_error("kimi: kv_b_proj TP row count mismatch at layer " +
                                     std::to_string(li));
        }

        L.is_moe = li >= cfg.first_k_dense_replace;
        const std::string mp = lp + "mlp.";
        if (!L.is_moe) {
            L.dense_gate_proj = &must(engine, mp + "gate_proj.weight");
            L.dense_up_proj   = &must(engine, mp + "up_proj.weight");
            L.dense_down_proj = &must(engine, mp + "down_proj.weight");
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
        std::vector<const std::int32_t*> gate_packed_ptrs;
        std::vector<const void*> gate_scale_ptrs;
        std::vector<const std::int32_t*> up_packed_ptrs;
        std::vector<const void*> up_scale_ptrs;
        std::vector<const std::int32_t*> down_packed_ptrs;
        std::vector<const void*> down_scale_ptrs;
        gate_packed_ptrs.reserve(static_cast<std::size_t>(cfg.num_experts));
        gate_scale_ptrs.reserve(static_cast<std::size_t>(cfg.num_experts));
        up_packed_ptrs.reserve(static_cast<std::size_t>(cfg.num_experts));
        up_scale_ptrs.reserve(static_cast<std::size_t>(cfg.num_experts));
        down_packed_ptrs.reserve(static_cast<std::size_t>(cfg.num_experts));
        down_scale_ptrs.reserve(static_cast<std::size_t>(cfg.num_experts));
        for (int e = 0; e < cfg.num_experts; ++e) {
            const std::string ep =
                mp + "experts." + std::to_string(e) + ".";
            auto& Ew = L.experts[static_cast<std::size_t>(e)];
            bind_expert(engine, ep, Ew);
            gate_packed_ptrs.push_back(
                static_cast<const std::int32_t*>(Ew.gate_packed->data()));
            gate_scale_ptrs.push_back(Ew.gate_scale->data());
            up_packed_ptrs.push_back(
                static_cast<const std::int32_t*>(Ew.up_packed->data()));
            up_scale_ptrs.push_back(Ew.up_scale->data());
            down_packed_ptrs.push_back(
                static_cast<const std::int32_t*>(Ew.down_packed->data()));
            down_scale_ptrs.push_back(Ew.down_scale->data());
        }
        L.expert_gate_packed_ptrs =
            DeviceBuffer<const std::int32_t*>::from_host(gate_packed_ptrs);
        L.expert_gate_scale_ptrs =
            DeviceBuffer<const void*>::from_host(gate_scale_ptrs);
        L.expert_up_packed_ptrs =
            DeviceBuffer<const std::int32_t*>::from_host(up_packed_ptrs);
        L.expert_up_scale_ptrs =
            DeviceBuffer<const void*>::from_host(up_scale_ptrs);
        L.expert_down_packed_ptrs =
            DeviceBuffer<const std::int32_t*>::from_host(down_packed_ptrs);
        L.expert_down_scale_ptrs =
            DeviceBuffer<const void*>::from_host(down_scale_ptrs);

        L.shared_gate_up_fused = maybe(engine,
            mp + "shared_experts.gate_up_proj.fused.weight");
        if (L.shared_gate_up_fused != nullptr) {
            L.shared_gate_proj = nullptr;
            L.shared_up_proj = nullptr;
        } else {
            L.shared_gate_proj = maybe(engine, mp + "shared_experts.gate_proj.weight");
            L.shared_up_proj   = maybe(engine, mp + "shared_experts.up_proj.weight");
        }
        L.shared_down_proj = maybe(engine, mp + "shared_experts.down_proj.weight");
        if (cfg.n_shared_experts > 0) {
            if (L.shared_gate_up_fused == nullptr &&
                (L.shared_gate_proj == nullptr || L.shared_up_proj == nullptr)) {
                throw std::runtime_error(
                    "kimi: shared experts configured but weights are missing at layer " +
                    std::to_string(li));
            }
            if (L.shared_down_proj == nullptr) {
                throw std::runtime_error(
                    "kimi: shared_down_proj missing at layer " + std::to_string(li));
            }
            if (L.shared_gate_up_fused != nullptr) {
                require_rank2(*L.shared_gate_up_fused,
                    mp + "shared_experts.gate_up_proj.fused.weight");
            } else {
                require_rank2(*L.shared_gate_proj, mp + "shared_experts.gate_proj.weight");
                require_rank2(*L.shared_up_proj, mp + "shared_experts.up_proj.weight");
            }
            require_rank2(*L.shared_down_proj, mp + "shared_experts.down_proj.weight");
        }
    }

    return w;
}

}  // namespace pie_cuda_driver::model
