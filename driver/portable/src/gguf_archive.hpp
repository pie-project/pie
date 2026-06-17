#pragma once

// GGUF archive — read-only mmap-backed wrapper around llama.cpp's
// `gguf.h` API. Exposes the same `WeightArchive` interface as
// `SafetensorsArchive`, with two key conveniences:
//
//  * Native ggml type passthrough (Q4_K / Q5_K / Q8_0 / IQ*_*): the
//    StTensor's `ggml_type_override` is set so downstream loaders use
//    the GGUF type directly, no dequant.
//  * Tensor-name translation: GGUF's `blk.{N}.attn_q.weight` is
//    rewritten to HF's `model.layers.{N}.self_attn.q_proj.weight` at
//    construction so the per-arch loaders don't need to branch.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "safetensors.hpp"          // StTensor / WeightArchive

struct gguf_context;
struct ggml_context;

namespace pie_portable_driver {

// Parsed subset of GGUF KV metadata that the per-arch hparams parser
// needs. Stored on the archive so `Hparams` can be built without
// re-opening the file.
struct GgufMeta {
    std::string  general_architecture;   // "qwen3", "llama", "qwen3_5_moe", ...
    std::string  general_name;
    bool         has_output_weight = false; // false → tied embeddings
    // Length of the `tokenizer.ggml.tokens` array, captured during archive
    // construction. Many GGUFs omit `<arch>.vocab_size` and rely on the
    // tokens-array length as the canonical vocab count, so we surface it
    // here for hparams fallback.
    std::size_t  tokens_count = 0;
    // All KV pairs as (key, type-erased) — used by the hparams parser.
    // Type is one of gguf_type's underlying values.
    struct KV { std::string key; std::int32_t type; std::string str_value;
                double num_value = 0.0; bool bool_value = false;
                // Integer/bool ARRAY payload (e.g. gemma4 per-layer
                // head_count_kv / sliding_window_pattern). Empty for
                // scalar or string/float-array KVs.
                std::vector<std::int64_t> arr_int; };
    std::unordered_map<std::string, KV> kv;
};

// GGUF tensor name → HF-canonical name (e.g. `blk.3.attn_q.weight` →
// `model.layers.3.self_attn.q_proj.weight`). Returns an empty string when a
// tensor has no canonical mapping. Pure function. Defined `inline` here (not
// in gguf_archive.cpp) so the qwen35 mapper-coverage testhook — a separate TU
// in the same static lib — carries its own copy: the dummy/portable
// `--bin pie` test never instantiates GGUFArchive, so a cross-TU reference to
// a definition in gguf_archive.o would be dropped by gc-sections and fail to
// link. The archive constructor also uses it to key the tensor table.
//
// Currently covers the Llama-family layout used by qwen3 / qwen2 / llama3 /
// mistral3 / olmo3 and Qwen3-MoE / Mixtral / gpt-oss MoE variants, plus the
// Qwen 3.5 / 3.6 hybrid (linear-attn + shared expert) and Gemma 4.
//
// `gemma_norm_layout` selects the Gemma `ffn_norm` interpretation (see below);
// the caller derives it from `general.architecture` (gemma* → true).
inline std::string gguf_to_hf_name(const std::string& g, bool gemma_norm_layout) {
    // Top-level tensors.
    if (g == "token_embd.weight")  return "model.embed_tokens.weight";
    if (g == "output_norm.weight") return "model.norm.weight";
    if (g == "output.weight")      return "lm_head.weight";

    // Per-block: `blk.{N}.<part>` → `model.layers.{N}.<...>`.
    const std::string blk = "blk.";
    if (g.compare(0, blk.size(), blk) != 0) return {};
    auto dot = g.find('.', blk.size());
    if (dot == std::string::npos) return {};
    const std::string n = g.substr(blk.size(), dot - blk.size());
    const std::string suffix = g.substr(dot + 1);
    const std::string layer = "model.layers." + n + ".";

    // Norms.
    if (suffix == "attn_norm.weight")        return layer + "input_layernorm.weight";
    // `ffn_norm` is the pre-MLP norm. In the Llama family it is the only
    // post-attention norm, so it maps to `post_attention_layernorm`. The
    // Gemma family instead carries BOTH a distinct `post_attention_norm`
    // (the post-attention norm) AND `ffn_norm` (the PRE-feedforward norm),
    // so for gemma `ffn_norm` is the pre-feedforward layernorm.
    if (suffix == "ffn_norm.weight")
        return layer + (gemma_norm_layout ? "pre_feedforward_layernorm.weight"
                                          : "post_attention_layernorm.weight");
    if (suffix == "post_attention_norm.weight") return layer + "post_attention_layernorm.weight";
    if (suffix == "post_ffw_norm.weight")    return layer + "post_feedforward_layernorm.weight";
    // Gemma 4 per-layer residual scalar (HF `layer_scalar`).
    if (suffix == "layer_output_scale.weight") return layer + "layer_scalar";
    // Attention projections.
    if (suffix == "attn_q.weight")           return layer + "self_attn.q_proj.weight";
    if (suffix == "attn_k.weight")           return layer + "self_attn.k_proj.weight";
    if (suffix == "attn_v.weight")           return layer + "self_attn.v_proj.weight";
    if (suffix == "attn_output.weight")      return layer + "self_attn.o_proj.weight";
    if (suffix == "attn_q.bias")             return layer + "self_attn.q_proj.bias";
    if (suffix == "attn_k.bias")             return layer + "self_attn.k_proj.bias";
    if (suffix == "attn_v.bias")             return layer + "self_attn.v_proj.bias";
    if (suffix == "attn_output.bias")        return layer + "self_attn.o_proj.bias";
    // Per-head q/k norms (Qwen 3, Olmo 3, Gemma 3).
    if (suffix == "attn_q_norm.weight")      return layer + "self_attn.q_norm.weight";
    if (suffix == "attn_k_norm.weight")      return layer + "self_attn.k_norm.weight";
    // Dense FFN.
    if (suffix == "ffn_gate.weight")         return layer + "mlp.gate_proj.weight";
    if (suffix == "ffn_up.weight")           return layer + "mlp.up_proj.weight";
    if (suffix == "ffn_down.weight")         return layer + "mlp.down_proj.weight";
    // MoE: stacked-expert tensors land directly on the same canonical
    // names our existing model.cpp uses for `moe_gate_exps` etc. We
    // include both the qwen3_moe / mixtral / gpt-oss style stacked
    // tensors and the legacy per-expert form (some converters still
    // emit `blk.N.ffn_gate.E.weight`).
    if (suffix == "ffn_gate_inp.weight")     return layer + "mlp.gate.weight";
    if (suffix == "ffn_gate_exps.weight")    return layer + "mlp.experts.gate_proj.weight";
    if (suffix == "ffn_up_exps.weight")      return layer + "mlp.experts.up_proj.weight";
    if (suffix == "ffn_down_exps.weight")    return layer + "mlp.experts.down_proj.weight";
    // Qwen 3.5 / 3.6 shared expert (qwen35moe). `ffn_gate_inp_shexp` is the
    // 1-D sigmoid gate over the shared expert (HF `mlp.shared_expert_gate`),
    // distinct from the per-token router `ffn_gate_inp`.
    if (suffix == "ffn_gate_shexp.weight")     return layer + "mlp.shared_expert.gate_proj.weight";
    if (suffix == "ffn_up_shexp.weight")       return layer + "mlp.shared_expert.up_proj.weight";
    if (suffix == "ffn_down_shexp.weight")     return layer + "mlp.shared_expert.down_proj.weight";
    if (suffix == "ffn_gate_inp_shexp.weight") return layer + "mlp.shared_expert_gate.weight";
    // Qwen 3.5 / 3.6 hybrid gated-delta-rule linear attention. The GGUF
    // (llama.cpp qwen35 convention) packs the linear-attn input projection
    // as `attn_qkv` (the fused q/k/v of the delta net) + `attn_gate` (the
    // output Z gate), and the gated-delta params as `ssm_*`. These names
    // only ever appear on the *linear* layers; full-attention layers carry
    // the ordinary `attn_q/k/v` handled above. The convert-time value
    // transforms (A_log = -exp(raw); folded norm.weight +1) are inverted at
    // load in `model.cpp::build_qwen3_5_`, not here.
    if (suffix == "attn_qkv.weight")         return layer + "linear_attn.in_proj_qkv.weight";
    if (suffix == "attn_gate.weight")        return layer + "linear_attn.in_proj_z.weight";
    if (suffix == "ssm_alpha.weight")        return layer + "linear_attn.in_proj_a.weight";
    if (suffix == "ssm_beta.weight")         return layer + "linear_attn.in_proj_b.weight";
    if (suffix == "ssm_a")                   return layer + "linear_attn.A_log";
    if (suffix == "ssm_dt.bias")             return layer + "linear_attn.dt_bias";
    if (suffix == "ssm_conv1d.weight")       return layer + "linear_attn.conv1d.weight";
    if (suffix == "ssm_norm.weight")         return layer + "linear_attn.norm.weight";
    if (suffix == "ssm_out.weight")          return layer + "linear_attn.out_proj.weight";
    return {};
}

class GGUFArchive : public WeightArchive {
public:
    explicit GGUFArchive(const std::filesystem::path& gguf_file);

    // Open a split/sharded GGUF set given the first shard path.
    // Discovers sibling shards via the *-NNNNN-of-MMMMM.gguf pattern,
    // merges all tensors into one logical archive.
    static std::unique_ptr<GGUFArchive>
    open_sharded(const std::filesystem::path& first_shard);
    ~GGUFArchive() override;

    GGUFArchive(const GGUFArchive&) = delete;
    GGUFArchive& operator=(const GGUFArchive&) = delete;

    const StTensor* find(const std::string& name) const noexcept override;
    const StTensor& at(const std::string& name) const override;
    std::size_t num_tensors() const noexcept override { return tensors_.size(); }
    bool is_gguf() const noexcept override { return true; }

    const GgufMeta& meta() const noexcept { return meta_; }

private:
    // mmap state.
#ifdef _WIN32
    std::vector<std::uint8_t> owned_data_;
#else
    int          fd_ = -1;
#endif
    std::size_t  mmap_size_ = 0;
    const std::uint8_t* mmap_base_ = nullptr;

    // gguf state — owned by us; no_alloc=true so it only carries metadata.
    gguf_context* gctx_ = nullptr;

    // Tensors keyed by HF-canonical name (after translation from GGUF
    // names). Data pointers point into the mmap region.
    std::unordered_map<std::string, StTensor> tensors_;
    GgufMeta meta_;

    // Original GGUF tensor names, kept for diagnostics and untranslated
    // lookups (e.g. raw `output.weight` to detect tie_word_embeddings).
    std::vector<std::string> raw_names_;

    // Heap buffers for tensors we had to dequant at load time (e.g.
    // token_embd when its GGUF type isn't supported by ggml_get_rows on
    // the CUDA backend — Q6_K, Q4_K, IQ*). The corresponding StTensor
    // points into one of these buffers instead of the mmap region.
    std::vector<std::vector<std::uint8_t>> dequant_buffers_;

    // Keeps extra shard GGUFArchive objects alive so their mmap data
    // pointers remain valid after merging into the primary archive.
    std::vector<std::unique_ptr<GGUFArchive>> extra_mmaps_;
    void close_mmap_() noexcept;
};

}  // namespace pie_portable_driver
