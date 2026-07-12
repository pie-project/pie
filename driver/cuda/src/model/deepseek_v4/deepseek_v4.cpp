#include "model/deepseek_v4/deepseek_v4.hpp"

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

// Persistent storage for converted FP32 scale tensors.
// The DeviceTensors here outlive the bind call and are referenced via QuantMeta.
std::vector<DeviceTensor> g_dsv4_f32_scales;

// Convert an E8M0 block-scale tensor to FP32 and build a QuantMeta.
// E8M0: each byte encodes scale as 2^(byte - 127).
// The FP8 GEMM expects FP32 *inverse* scales (1/scale) for per-channel,
// or direct scales for PerGroup depending on the implementation.
// We store direct scales and let the dispatcher handle it.
std::optional<QuantMeta> make_block_fp8_quant(
    const DeviceTensor* weight,
    const DeviceTensor* scale_e8m0,
    int group_size = 128)
{
    if (!weight || !scale_e8m0) return std::nullopt;
    if (weight->dtype() != DType::FP8_E4M3) return std::nullopt;

    // Download E8M0 bytes, convert to FP32 on host, upload as new DeviceTensor
    const std::size_t n = scale_e8m0->numel();
    std::vector<std::uint8_t> e8m0_h(n);
    CUDA_CHECK(cudaMemcpy(e8m0_h.data(), scale_e8m0->data(),
                          n, cudaMemcpyDeviceToHost));

    std::vector<float> f32_h(n);
    for (std::size_t i = 0; i < n; ++i) {
        f32_h[i] = ldexpf(1.0f, static_cast<int>(e8m0_h[i]) - 127);
    }

    auto t = DeviceTensor::allocate(DType::FP32, {static_cast<int>(n)});
    CUDA_CHECK(cudaMemcpy(t.data(), f32_h.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    g_dsv4_f32_scales.push_back(std::move(t));
    const DeviceTensor* scale_ptr = &g_dsv4_f32_scales.back();

    QuantMeta meta;
    meta.kind = QuantMeta::Kind::PerGroup;
    meta.scale = scale_ptr;
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
        w.lm_head_tp_vocab_offset = 0;
        w.lm_head_tp_sharded = false;
    } else if (engine.distributed().tp_size > 1 &&
               w.lm_head->shape()[0] * engine.distributed().tp_size == cfg.vocab_size) {
        w.lm_head_tp_vocab_offset =
            static_cast<int>(w.lm_head->shape()[0] * engine.distributed().tp_rank);
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

        // Routed experts
        L.experts.resize(static_cast<std::size_t>(E));
        for (int e = 0; e < E; ++e) {
            const std::string ep =
                fp + "experts." + std::to_string(e) + ".";
            bind_expert(engine, ep, L.experts[static_cast<std::size_t>(e)]);
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
