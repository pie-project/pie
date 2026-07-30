#include "model/deepseek_v4/deepseek_v4.hpp"

#include "loader/group_stream_cache.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "cuda_check.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("deepseek_v4: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe(const LoadedModel& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}

void bind_compressor(
    const LoadedModel& engine,
    const std::string& prefix,
    DsV4CompressorWeights& c)
{
    c.ape   = maybe(engine, prefix + "ape");
    c.norm  = maybe(engine, prefix + "norm.weight");
    c.wkv   = maybe(engine, prefix + "wkv.weight");
    c.wgate = maybe(engine, prefix + "wgate.weight");
}

void bind_indexer(
    const LoadedModel& engine,
    const std::string& prefix,
    DsV4IndexerWeights& idx)
{
    idx.wq_b         = maybe(engine, prefix + "wq_b.weight");
    idx.wq_b_scale   = maybe(engine, prefix + "wq_b.scale");
    idx.weights_proj  = maybe(engine, prefix + "weights_proj.weight");
    bind_compressor(engine, prefix + "compressor.", idx.compressor);
}

void bind_expert(
    const LoadedModel& engine,
    const std::string& prefix,
    DsV4ExpertWeights& ew)
{
    ew.w1       = maybe(engine, prefix + "w1.weight");
    ew.w1_scale = maybe(engine, prefix + "w1.scale");
    ew.w2       = maybe(engine, prefix + "w2.weight");
    ew.w2_scale = maybe(engine, prefix + "w2.scale");
    ew.w3       = maybe(engine, prefix + "w3.weight");
    ew.w3_scale = maybe(engine, prefix + "w3.scale");
}

// Build a QuantMeta for a block-scaled FP8 weight.
//
// The scale arrives already decoded: `dsv4_block_scales_to_fp32` states in the
// contract that the checkpoint's E8M0 bytes are read as fp32, so the loader
// casts them on the device during the load. This used to download every scale,
// run `ldexpf` over it on the host and upload the result -- into a global
// vector whose `push_back` invalidated the pointers it had already handed out.
std::optional<QuantMeta> make_block_fp8_quant(
    const DeviceTensor* weight,
    const DeviceTensor* scale,
    int group_size = 128)
{
    if (!weight || !scale) return std::nullopt;
    if (weight->dtype() != DType::FP8_E4M3) return std::nullopt;
    if (scale->dtype() != DType::FP32) {
        throw std::runtime_error(
            std::string("deepseek_v4: block scale arrived as ") +
            dtype_name(scale->dtype()) +
            ", expected fp32 -- the contract should have declared the E8M0 "
            "bytes as fp32");
    }

    QuantMeta meta;
    meta.kind = QuantMeta::Kind::PerGroup;
    meta.scale = scale;
    meta.group_size = group_size;
    meta.channel_axis = 0;
    return meta;
}

}  // namespace

DsV4Weights bind_deepseek_v4(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();

    DsV4Weights w;
    w.embed      = &must(engine, "embed.weight");
    w.final_norm = &must(engine, "norm.weight");

    if (engine.has("head.weight")) {
        w.lm_head = &engine.get("head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error("deepseek_v4: head.weight missing and tie_word_embeddings=false");
    }

    if (w.embed->shape()[0] == cfg.vocab_size) {
        w.embed_tp_vocab_offset = 0;
        w.embed_tp_sharded = false;
    } else if (engine.distributed().tp_size > 1 &&
               w.embed->shape()[0] * engine.distributed().tp_size == cfg.vocab_size) {
        w.embed_tp_vocab_offset =
            static_cast<int>(w.embed->shape()[0] * engine.distributed().tp_rank);
        w.embed_tp_sharded = true;
    } else {
        throw std::runtime_error("deepseek_v4: embed row count does not match vocab or TP shard");
    }
    if (w.lm_head->shape()[0] == cfg.vocab_size) {
        w.lm_head_tp_sharded = false;
    } else if (engine.distributed().tp_size > 1 &&
               w.lm_head->shape()[0] * engine.distributed().tp_size == cfg.vocab_size) {
        w.lm_head_tp_sharded = true;
    } else {
        throw std::runtime_error("deepseek_v4: lm_head row count does not match vocab or TP shard");
    }

    // HC head
    w.hc_head_fn    = maybe(engine, "hc_head_fn");
    w.hc_head_scale = maybe(engine, "hc_head_scale");
    w.hc_head_base  = maybe(engine, "hc_head_base");

    const int E = cfg.num_experts;
    const int num_layers = cfg.num_hidden_layers;

    w.layers.resize(static_cast<std::size_t>(num_layers));
    for (int li = 0; li < num_layers; ++li) {
        const std::string lp = "layers." + std::to_string(li) + ".";
        auto& L = w.layers[static_cast<std::size_t>(li)];

        // Norms
        L.attn_norm = &must(engine, lp + "attn_norm.weight");
        L.ffn_norm  = &must(engine, lp + "ffn_norm.weight");

        // Attention projections
        const std::string ap = lp + "attn.";
        L.wq_a       = &must(engine, ap + "wq_a.weight");
        L.wq_a_scale = maybe(engine, ap + "wq_a.scale");
        L.wq_a_quant = engine.quant_meta(ap + "wq_a.weight");
        if (!L.wq_a_quant) L.wq_a_quant = make_block_fp8_quant(L.wq_a, L.wq_a_scale);
        L.wq_b       = &must(engine, ap + "wq_b.weight");
        L.wq_b_scale = maybe(engine, ap + "wq_b.scale");
        L.wq_b_quant = engine.quant_meta(ap + "wq_b.weight");
        if (!L.wq_b_quant) L.wq_b_quant = make_block_fp8_quant(L.wq_b, L.wq_b_scale);
        L.q_norm     = &must(engine, ap + "q_norm.weight");
        L.wkv        = &must(engine, ap + "wkv.weight");
        L.wkv_scale  = maybe(engine, ap + "wkv.scale");
        L.wkv_quant  = engine.quant_meta(ap + "wkv.weight");
        if (!L.wkv_quant) L.wkv_quant = make_block_fp8_quant(L.wkv, L.wkv_scale);
        L.kv_norm    = &must(engine, ap + "kv_norm.weight");
        L.wo_a       = &must(engine, ap + "wo_a.weight");
        L.wo_a_scale = maybe(engine, ap + "wo_a.scale");
        L.wo_a_quant = engine.quant_meta(ap + "wo_a.weight");
        if (!L.wo_a_quant) L.wo_a_quant = make_block_fp8_quant(L.wo_a, L.wo_a_scale);
        L.wo_b       = &must(engine, ap + "wo_b.weight");
        L.wo_b_scale = maybe(engine, ap + "wo_b.scale");
        L.wo_b_quant = engine.quant_meta(ap + "wo_b.weight");
        if (!L.wo_b_quant) L.wo_b_quant = make_block_fp8_quant(L.wo_b, L.wo_b_scale);
        L.attn_sink  = maybe(engine, ap + "attn_sink");

        // HC mixing
        L.hc_attn_fn    = maybe(engine, lp + "hc_attn_fn");
        L.hc_attn_scale = maybe(engine, lp + "hc_attn_scale");
        L.hc_attn_base  = maybe(engine, lp + "hc_attn_base");
        L.hc_ffn_fn     = maybe(engine, lp + "hc_ffn_fn");
        L.hc_ffn_scale  = maybe(engine, lp + "hc_ffn_scale");
        L.hc_ffn_base   = maybe(engine, lp + "hc_ffn_base");

        // Per-layer compression ratio
        if (li < static_cast<int>(cfg.dsv4_compress_ratios.size())) {
            L.compress_ratio = cfg.dsv4_compress_ratios[static_cast<std::size_t>(li)];
        }

        // Compressor (C4/C128 layers)
        if (L.compress_ratio > 0) {
            bind_compressor(engine, ap + "compressor.", L.compressor);
        }

        // Indexer (C4 layers)
        if (L.compress_ratio == 4) {
            bind_indexer(engine, ap + "indexer.", L.indexer);
        }

        // MoE
        const std::string fp = lp + "ffn.";
        L.is_hash_layer = (li < cfg.dsv4_num_hash_layers);

        L.router      = &must(engine, fp + "gate.weight");
        L.router_bias = maybe(engine, fp + "gate.bias");
        L.tid2eid     = maybe(engine, fp + "gate.tid2eid");

        // Routed experts. Which of the two forms is present is the contract's
        // answer, so both are optional here and the forward pass branches on
        // whichever it got. `author_deepseek_v4_contract` publishes the stacks
        // and consumes the packed originals, or publishes the originals and no
        // stacks -- never both, so there is no second copy to pay for.
        L.moe_gate_up_bf16 = maybe(engine, fp + "experts.gate_up.weight");
        L.moe_down_bf16 = maybe(engine, fp + "experts.down.weight");
        if (GroupStreamCache* cache = engine.group_cache();
            cache != nullptr && L.moe_gate_up_bf16 == nullptr) {
            const std::string group_name = fp + "experts";
            const std::size_t g = cache->find_group(group_name);
            if (g != GroupStreamCache::kNoGroup) {
                if (cache->arity(g) != static_cast<std::uint32_t>(E)) {
                    throw std::runtime_error(
                        "deepseek_v4: group '" + group_name + "' holds " +
                        std::to_string(cache->arity(g)) + " experts but the "
                        "config says " + std::to_string(E));
                }
                L.expert_cache = cache;
                L.expert_group = g;
            }
        }
        if (L.moe_gate_up_bf16 == nullptr && L.expert_cache == nullptr) {
            L.experts.resize(static_cast<std::size_t>(E));
            for (int e = 0; e < E; ++e) {
                const std::string ep =
                    fp + "experts." + std::to_string(e) + ".";
                bind_expert(engine, ep, L.experts[static_cast<std::size_t>(e)]);
            }
        }

        // Shared experts
        L.shared_w1       = maybe(engine, fp + "shared_experts.w1.weight");
        L.shared_w1_scale = maybe(engine, fp + "shared_experts.w1.scale");
        L.shared_w1_quant = engine.quant_meta(fp + "shared_experts.w1.weight");
        if (!L.shared_w1_quant) L.shared_w1_quant = make_block_fp8_quant(L.shared_w1, L.shared_w1_scale);
        L.shared_w2       = maybe(engine, fp + "shared_experts.w2.weight");
        L.shared_w2_scale = maybe(engine, fp + "shared_experts.w2.scale");
        L.shared_w2_quant = engine.quant_meta(fp + "shared_experts.w2.weight");
        if (!L.shared_w2_quant) L.shared_w2_quant = make_block_fp8_quant(L.shared_w2, L.shared_w2_scale);
        L.shared_w3       = maybe(engine, fp + "shared_experts.w3.weight");
        L.shared_w3_scale = maybe(engine, fp + "shared_experts.w3.scale");
        L.shared_w3_quant = engine.quant_meta(fp + "shared_experts.w3.weight");
        if (!L.shared_w3_quant) L.shared_w3_quant = make_block_fp8_quant(L.shared_w3, L.shared_w3_scale);
    }

    return w;
}

}  // namespace pie_cuda_driver::model
