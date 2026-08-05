#include "model/kimi_k3/kimi_k3.hpp"

#include <cstdlib>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "cuda_check.hpp"
#include "kernels/gated_delta_net.hpp"

namespace pie_cuda_driver::model {

// Stack the routed experts' fc1 as `[up | gate]` instead of `[gate | up]`,
// which is the order flashinfer's fused CUTLASS grouped GEMM reads it in.
// Off by default for the same reason Kimi-K2's is: the fused runner is a real
// GPU-time win but resonates with the scheduler at some concurrencies. Enable
// with `PIE_KIMI_K3_MOE_FLASHINFER=1`.
bool kimi_k3_moe_gate_up_swapped() { return false; }

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("kimi_k3: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe(const LoadedModel& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}

void require_rank(const DeviceTensor& t, std::size_t rank, const std::string& name) {
    if (t.shape().size() != rank) {
        throw std::runtime_error("kimi_k3: expected rank-" + std::to_string(rank) +
                                 " tensor for '" + name + "'");
    }
}

void require_rows(const DeviceTensor& t, std::int64_t rows, const std::string& name) {
    if (t.shape().empty() || t.shape()[0] != rows) {
        throw std::runtime_error(
            "kimi_k3: '" + name + "' has " +
            std::to_string(t.shape().empty() ? -1 : t.shape()[0]) +
            " rows, expected " + std::to_string(rows) +
            " -- the TP shard axis for this name is wrong");
    }
}

/// Materialise an owned bf16 copy of a weight the kernels read as bf16.
///
/// K3 ships the three short-convolution kernels as fp32 while Qwen3.5 -- whose
/// `causal_conv1d` kernels this path reuses -- ships them bf16. Handing the
/// fp32 bytes straight to a bf16 kernel is not a type error anywhere: it reads
/// each fp32's high half as a bf16 and produces ~3.4e38 garbage that only
/// shows up as NaN two kernels later.
DeviceTensor to_bf16(const DeviceTensor& t) {
    std::size_t n = 1;
    for (auto d : t.shape()) n *= static_cast<std::size_t>(d);
    DeviceTensor out = DeviceTensor::allocate(DType::BF16, t.shape());
    if (t.dtype() == DType::BF16) {
        CUDA_CHECK(cudaMemcpy(out.data(), t.data(), n * 2, cudaMemcpyDeviceToDevice));
    } else if (t.dtype() == DType::FP32) {
        kernels::launch_fp32_to_bf16(static_cast<const float*>(t.data()), out.data(), n,
                                     /*stream=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        throw std::runtime_error("kimi_k3: unsupported dtype for bf16 conversion");
    }
    return out;
}

/// Materialise an owned fp32 copy. The KDA recurrence and its gated RMSNorm
/// read `const float*`; K3 ships these three as fp32 already, but the copy is
/// what gives the binder a buffer it owns and can hand out a raw pointer to.
DeviceBuffer<float> to_fp32(const DeviceTensor& t) {
    std::size_t n = 1;
    for (auto d : t.shape()) n *= static_cast<std::size_t>(d);
    auto buf = DeviceBuffer<float>::alloc(n);
    if (t.dtype() == DType::FP32) {
        CUDA_CHECK(cudaMemcpy(buf.data(), t.data(), n * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    } else if (t.dtype() == DType::BF16) {
        kernels::launch_bf16_to_fp32(t.data(), buf.data(), n, /*stream=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        throw std::runtime_error("kimi_k3: unsupported dtype for fp32 conversion");
    }
    return buf;
}

}  // namespace

KimiK3Weights bind_kimi_k3(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    if (cfg.q_lora_rank <= 0 || cfg.kv_lora_rank <= 0 ||
        cfg.qk_nope_head_dim <= 0 || cfg.qk_rope_head_dim <= 0 ||
        cfg.v_head_dim <= 0) {
        throw std::runtime_error("kimi_k3: MLA dimensions are missing from HfConfig");
    }
    if (cfg.num_experts <= 0 || cfg.num_experts_per_tok <= 0 ||
        cfg.moe_intermediate_size <= 0) {
        throw std::runtime_error("kimi_k3: MoE dimensions are missing from HfConfig");
    }
    if (cfg.linear_num_value_heads <= 0 || cfg.linear_value_head_dim <= 0 ||
        cfg.linear_conv_kernel_dim <= 0) {
        throw std::runtime_error("kimi_k3: KDA dimensions are missing from HfConfig");
    }
    if (static_cast<int>(cfg.layer_types.size()) != cfg.num_hidden_layers) {
        throw std::runtime_error(
            "kimi_k3: layer_types has " + std::to_string(cfg.layer_types.size()) +
            " entries for " + std::to_string(cfg.num_hidden_layers) + " layers");
    }
    if (!cfg.mla_use_nope) {
        // Every K3 checkpoint sets this, and the forward has no rotation to
        // apply if one did not -- fail rather than silently drop positions.
        throw std::runtime_error(
            "kimi_k3: mla_use_nope is false; rotated MLA is not implemented");
    }

    const int T = std::max(1, engine.distributed().tp_size);
    if (cfg.num_attention_heads % T != 0 || cfg.linear_num_value_heads % T != 0) {
        throw std::runtime_error(
            "kimi_k3: tp_size " + std::to_string(T) +
            " does not divide the MLA or KDA head count");
    }
    const int local_mla_heads = cfg.num_attention_heads / T;
    const int local_kda_heads = cfg.linear_num_value_heads / T;
    const int kda_dim = cfg.linear_value_head_dim;
    const int local_kda_width = local_kda_heads * kda_dim;
    const int q_b_rows = local_mla_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
    const int kv_b_rows = local_mla_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);
    const int latent = cfg.routed_expert_hidden_size > 0
                           ? cfg.routed_expert_hidden_size
                           : cfg.hidden_size;

    KimiK3Weights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "kimi_k3: lm_head missing and tie_word_embeddings=false");
    }
    require_rank(*w.embed, 2, "model.embed_tokens.weight");
    require_rank(*w.lm_head, 2, "lm_head.weight");

    if (w.embed->shape()[0] == cfg.vocab_size) {
        w.embed_tp_sharded = false;
    } else if (T > 1 && w.embed->shape()[0] * T == cfg.vocab_size) {
        w.embed_tp_vocab_offset =
            static_cast<int>(w.embed->shape()[0] * engine.distributed().tp_rank);
        w.embed_tp_sharded = true;
    } else {
        throw std::runtime_error(
            "kimi_k3: embed row count does not match vocab or TP shard");
    }
    if (w.lm_head->shape()[0] == cfg.vocab_size) {
        w.lm_head_tp_sharded = false;
    } else if (T > 1 && w.lm_head->shape()[0] * T == cfg.vocab_size) {
        w.lm_head_tp_sharded = true;
    } else {
        throw std::runtime_error(
            "kimi_k3: lm_head row count does not match vocab or TP shard");
    }

    if (cfg.attn_res_block_size > 0) {
        w.output_res_norm = &must(engine, "model.output_attn_res_norm.weight");
        w.output_res_proj = &must(engine, "model.output_attn_res_proj.weight");
    }

    // A_log and dt_bias are the only per-head tensors the binder still has to
    // reserve room for up front: `owned_fp32_buffers` must not reallocate
    // after a layer has taken a pointer into it.
    w.owned_fp32_buffers.reserve(static_cast<std::size_t>(cfg.num_hidden_layers) * 3);
    w.owned_bf16_buffers.reserve(static_cast<std::size_t>(cfg.num_hidden_layers) * 3);
    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));

    int kv_slot = 0;
    int kda_slot = 0;
    for (int li = 0; li < cfg.num_hidden_layers; ++li) {
        const std::string lp = "model.layers." + std::to_string(li) + ".";
        const std::string ap = lp + "self_attn.";
        auto& L = w.layers[static_cast<std::size_t>(li)];

        L.attn_norm = &must(engine, lp + "input_layernorm.weight");
        L.mlp_norm  = &must(engine, lp + "post_attention_layernorm.weight");

        if (cfg.attn_res_block_size > 0) {
            L.attn_res_norm = &must(engine, lp + "self_attention_res_norm.weight");
            L.attn_res_proj = &must(engine, lp + "self_attention_res_proj.weight");
            L.mlp_res_norm  = &must(engine, lp + "mlp_res_norm.weight");
            L.mlp_res_proj  = &must(engine, lp + "mlp_res_proj.weight");
        }

        const std::string& kind = cfg.layer_types[static_cast<std::size_t>(li)];
        if (kind == "linear_attention") {
            L.kind = KimiK3LayerWeights::Kind::Kda;
            L.kda_layer = kda_slot++;
            L.kv_layer = -1;

            // The contract does not join these -- each goes through its own
            // short convolution -- so a fused tensor here means the contract
            // and the forward disagree, and the forward would read a null.
            if (engine.has(ap + "qkv_proj.fused.weight")) {
                throw std::runtime_error(
                    "kimi_k3: layer " + std::to_string(li) +
                    " has a fused QKV projection, which the KDA path cannot use");
            }
            L.kda_q_proj = &must(engine, ap + "q_proj.weight");
            L.kda_k_proj = &must(engine, ap + "k_proj.weight");
            L.kda_v_proj = &must(engine, ap + "v_proj.weight");
            require_rows(*L.kda_q_proj, local_kda_width, ap + "q_proj.weight");
            require_rows(*L.kda_k_proj, local_kda_width, ap + "k_proj.weight");
            require_rows(*L.kda_v_proj, local_kda_width, ap + "v_proj.weight");
            require_rows(must(engine, ap + "q_conv1d.weight"), local_kda_width,
                         ap + "q_conv1d.weight");
            w.owned_bf16_buffers.push_back(to_bf16(must(engine, ap + "q_conv1d.weight")));
            L.kda_q_conv1d = &w.owned_bf16_buffers.back();
            w.owned_bf16_buffers.push_back(to_bf16(must(engine, ap + "k_conv1d.weight")));
            L.kda_k_conv1d = &w.owned_bf16_buffers.back();
            w.owned_bf16_buffers.push_back(to_bf16(must(engine, ap + "v_conv1d.weight")));
            L.kda_v_conv1d = &w.owned_bf16_buffers.back();

            // `f_a_proj` is the rank-`head_dim` bottleneck every head reads,
            // so it stays whole on each rank while `f_b_proj` follows the
            // heads. A shard on `f_a_proj` would give each rank a partial
            // bottleneck and still produce plausible numbers.
            L.kda_f_a_proj = &must(engine, ap + "f_a_proj.weight");
            require_rows(*L.kda_f_a_proj, kda_dim, ap + "f_a_proj.weight");
            L.kda_f_b_proj = &must(engine, ap + "f_b_proj.weight");
            require_rows(*L.kda_f_b_proj, local_kda_width, ap + "f_b_proj.weight");
            L.kda_b_proj = &must(engine, ap + "b_proj.weight");
            require_rows(*L.kda_b_proj, local_kda_heads, ap + "b_proj.weight");
            if (!cfg.kda_full_rank_gate) {
                throw std::runtime_error(
                    "kimi_k3: low-rank KDA output gate (g_a_proj/g_b_proj) is "
                    "not implemented; this checkpoint sets use_full_rank_gate=false");
            }
            L.kda_g_proj = &must(engine, ap + "g_proj.weight");
            require_rows(*L.kda_g_proj, local_kda_width, ap + "g_proj.weight");
            L.kda_o_proj = &must(engine, ap + "o_proj.weight");

            const DeviceTensor& a_log = must(engine, ap + "A_log");
            // The contract bands A_log to the head count before sharding, so
            // by the time it reaches here it is exactly this rank's heads --
            // the F32[head_dim] storage padding is already gone.
            require_rows(a_log, local_kda_heads, ap + "A_log");
            w.owned_fp32_buffers.push_back(to_fp32(a_log));
            L.kda_A_log_fp32 = w.owned_fp32_buffers.back().data();

            const DeviceTensor& dt_bias = must(engine, ap + "dt_bias");
            require_rows(dt_bias, local_kda_width, ap + "dt_bias");
            w.owned_fp32_buffers.push_back(to_fp32(dt_bias));
            L.kda_dt_bias_fp32 = w.owned_fp32_buffers.back().data();

            const DeviceTensor& o_norm = must(engine, ap + "o_norm.weight");
            require_rows(o_norm, kda_dim, ap + "o_norm.weight");
            w.owned_fp32_buffers.push_back(to_fp32(o_norm));
            L.kda_o_norm_fp32 = w.owned_fp32_buffers.back().data();
        } else if (kind == "full_attention") {
            L.kind = KimiK3LayerWeights::Kind::Mla;
            L.kv_layer = kv_slot++;
            L.kda_layer = -1;

            L.q_a_proj = &must(engine, ap + "q_a_proj.weight");
            L.kv_a_proj_with_mqa = &must(engine, ap + "kv_a_proj_with_mqa.weight");
            require_rank(*L.q_a_proj, 2, ap + "q_a_proj.weight");
            require_rank(*L.kv_a_proj_with_mqa, 2, ap + "kv_a_proj_with_mqa.weight");
            L.q_a_norm  = &must(engine, ap + "q_a_layernorm.weight");
            L.q_b_proj  = &must(engine, ap + "q_b_proj.weight");
            L.kv_a_norm = &must(engine, ap + "kv_a_layernorm.weight");
            L.kv_b_proj = &must(engine, ap + "kv_b_proj.weight");
            L.o_proj    = &must(engine, ap + "o_proj.weight");
            require_rows(*L.q_b_proj, q_b_rows, ap + "q_b_proj.weight");
            require_rows(*L.kv_b_proj, kv_b_rows, ap + "kv_b_proj.weight");

            if (cfg.mla_output_gate) {
                L.mla_g_proj = &must(engine, ap + "g_proj.weight");
                require_rows(*L.mla_g_proj,
                             static_cast<std::int64_t>(local_mla_heads) * cfg.v_head_dim,
                             ap + "g_proj.weight");
            }
        } else {
            throw std::runtime_error("kimi_k3: unknown layer_type '" + kind +
                                     "' at layer " + std::to_string(li));
        }

        // ── MLP ────────────────────────────────────────────────────
        L.is_moe = li >= cfg.first_k_dense_replace;
        if (!L.is_moe) {
            const std::string mp = lp + "mlp.";
            L.dense_gate_proj = &must(engine, mp + "gate_proj.weight");
            L.dense_up_proj   = &must(engine, mp + "up_proj.weight");
            L.dense_down_proj = &must(engine, mp + "down_proj.weight");
            continue;
        }

        const std::string mp = lp + "block_sparse_moe.";
        L.router = &must(engine, mp + "gate.weight");
        require_rows(*L.router, cfg.num_experts, mp + "gate.weight");
        L.e_score_correction_bias = maybe(engine, mp + "gate.e_score_correction_bias");

        // Latent MoE: the routed experts run at `routed_expert_hidden_size`,
        // reached through these three replicated projections. They are
        // replicated because the experts all-reduce their partial outputs, so
        // the latent that enters and leaves has to be full width everywhere.
        L.routed_down_proj = maybe(engine, mp + "routed_expert_down_proj.weight");
        L.routed_up_proj   = maybe(engine, mp + "routed_expert_up_proj.weight");
        L.routed_norm      = maybe(engine, mp + "routed_expert_norm.weight");
        if (cfg.routed_expert_hidden_size > 0) {
            if (L.routed_down_proj == nullptr || L.routed_up_proj == nullptr) {
                throw std::runtime_error(
                    "kimi_k3: routed_expert_hidden_size is set but the latent "
                    "projections are missing at layer " + std::to_string(li));
            }
            require_rows(*L.routed_down_proj, latent,
                         mp + "routed_expert_down_proj.weight");
            require_rows(*L.routed_up_proj, cfg.hidden_size,
                         mp + "routed_expert_up_proj.weight");
            if (cfg.latent_moe_use_norm && L.routed_norm == nullptr) {
                throw std::runtime_error(
                    "kimi_k3: latent_moe_use_norm is set but routed_expert_norm "
                    "is missing at layer " + std::to_string(li));
            }
        }

        L.moe_gate_up_bf16 = maybe(engine, mp + "experts.gate_up_proj");
        L.moe_down_bf16    = maybe(engine, mp + "experts.down_proj");
        if (L.moe_gate_up_bf16 == nullptr || L.moe_down_bf16 == nullptr) {
            throw std::runtime_error(
                "kimi_k3: the contract did not publish stacked routed experts at "
                "layer " + std::to_string(li) +
                "; per-expert MXFP4 GEMV is not implemented");
        }
        require_rank(*L.moe_gate_up_bf16, 3, mp + "experts.gate_up_proj");
        require_rank(*L.moe_down_bf16, 3, mp + "experts.down_proj");
        if (L.moe_gate_up_bf16->shape()[2] != latent ||
            L.moe_down_bf16->shape()[1] != latent) {
            throw std::runtime_error(
                "kimi_k3: routed experts are not at the latent width at layer " +
                std::to_string(li));
        }

        // The four-bit originals, per expert, for the decode GEMV. A decode
        // step reads the entire routed bank to produce one token, so at bf16
        // the MoE alone is four times the traffic it needs to be; the stacks
        // above stay for prefill, where the dequantized GEMM amortises.
        {
            std::vector<const std::uint8_t*> gu_w, gu_s, dn_w, dn_s;
            const int experts = cfg.num_experts;
            for (int e = 0; e < experts; ++e) {
                const std::string ep = mp + "experts." + std::to_string(e) + ".";
                const DeviceTensor* gw = maybe(engine, ep + "gate_up.weight_packed");
                const DeviceTensor* gs = maybe(engine, ep + "gate_up.weight_scale");
                const DeviceTensor* dw = maybe(engine, ep + "down.weight_packed");
                const DeviceTensor* ds = maybe(engine, ep + "down.weight_scale");
                if (gw == nullptr || gs == nullptr || dw == nullptr || ds == nullptr) {
                    gu_w.clear();
                    break;
                }
                gu_w.push_back(static_cast<const std::uint8_t*>(gw->data()));
                gu_s.push_back(static_cast<const std::uint8_t*>(gs->data()));
                dn_w.push_back(static_cast<const std::uint8_t*>(dw->data()));
                dn_s.push_back(static_cast<const std::uint8_t*>(ds->data()));
            }
            if (!gu_w.empty()) {
                L.expert_gate_up_packed_ptrs =
                    DeviceBuffer<const std::uint8_t*>::from_host(gu_w);
                L.expert_gate_up_scale_ptrs =
                    DeviceBuffer<const std::uint8_t*>::from_host(gu_s);
                L.expert_down_packed_ptrs =
                    DeviceBuffer<const std::uint8_t*>::from_host(dn_w);
                L.expert_down_scale_ptrs =
                    DeviceBuffer<const std::uint8_t*>::from_host(dn_s);
            }
        }

        L.shared_gate_proj = maybe(engine, mp + "shared_experts.gate_proj.weight");
        L.shared_up_proj   = maybe(engine, mp + "shared_experts.up_proj.weight");
        L.shared_down_proj = maybe(engine, mp + "shared_experts.down_proj.weight");
        if (cfg.n_shared_experts > 0) {
            const bool have_gate_up =
                L.shared_gate_proj != nullptr && L.shared_up_proj != nullptr;
            if (!have_gate_up || L.shared_down_proj == nullptr) {
                throw std::runtime_error(
                    "kimi_k3: shared experts configured but weights are missing at "
                    "layer " + std::to_string(li));
            }
        }
    }

    return w;
}

}  // namespace pie_cuda_driver::model
