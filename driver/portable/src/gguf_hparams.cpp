#include "gguf_hparams.hpp"
#include "gguf_archive.hpp"

#include <cmath>
#include <stdexcept>

namespace pie_portable_driver {

namespace {

// Map llama.cpp's `general.architecture` strings onto our PieArch enum.
// llama.cpp uses an arch-internal name (e.g. "qwen3", "llama"). For the
// hybrid Qwen 3.5 / 3.6 family the name is "qwen35" (per llama.cpp PR
// #19468) — we accept both that and a few seen-in-the-wild variants.
PieArch arch_from_gguf(const std::string& s) {
    if (s == "qwen3")              return PieArch::Qwen3;
    if (s == "qwen2")              return PieArch::Qwen2;
    if (s == "qwen3moe")           return PieArch::Qwen3Moe;
    if (s == "qwen35"
     || s == "qwen3_5"
     || s == "qwen3_5_text"
     || s == "qwen35moe"
     || s == "qwen3_5_moe"
     || s == "qwen3_5_moe_text")   return PieArch::Qwen3_5;
    if (s == "llama")              return PieArch::Llama3;
    if (s == "mistral")            return PieArch::Mistral3;
    if (s == "mixtral")            return PieArch::Mixtral;
    if (s == "phi3")               return PieArch::Phi3;
    if (s == "gemma2")             return PieArch::Gemma2;
    if (s == "gemma3")             return PieArch::Gemma3;
    if (s == "gemma4")             return PieArch::Gemma4;
    if (s == "olmo3" || s == "olmo2") return PieArch::Olmo3;
    if (s == "gpt-oss" || s == "gptoss") return PieArch::GptOss;
    throw std::runtime_error(
        "gguf: unsupported general.architecture '" + s + "'");
}

// Try `<arch>.<suffix>` first, then `<suffix>` (some converters strip
// the prefix on a few keys). Returns 0 / nullptr when missing.
const GgufMeta::KV* find_kv(const GgufMeta& m,
                             const std::string& arch,
                             const char* suffix) {
    const std::string a = arch + "." + suffix;
    if (auto it = m.kv.find(a); it != m.kv.end()) return &it->second;
    if (auto it = m.kv.find(suffix); it != m.kv.end()) return &it->second;
    return nullptr;
}

double get_num(const GgufMeta& m, const std::string& arch, const char* key,
               double dflt) {
    const auto* kv = find_kv(m, arch, key);
    return kv ? kv->num_value : dflt;
}

bool get_bool(const GgufMeta& m, const std::string& arch, const char* key,
              bool dflt) {
    const auto* kv = find_kv(m, arch, key);
    return kv ? kv->bool_value : dflt;
}

}  // namespace

Hparams parse_gguf_hparams(const GgufMeta& meta) {
    Hparams h{};

    // ---- Identity / arch dispatch --------------------------------------
    h.hf_model_type = meta.general_architecture;
    h.arch          = arch_from_gguf(meta.general_architecture);
    const std::string a = meta.general_architecture;

    // ---- Common transformer hparams ------------------------------------
    h.num_hidden_layers   = static_cast<std::int32_t>(get_num(meta, a, "block_count",        0));
    h.num_attention_heads = static_cast<std::int32_t>(get_num(meta, a, "attention.head_count", 0));
    h.num_key_value_heads = static_cast<std::int32_t>(get_num(meta, a, "attention.head_count_kv",
                                                              h.num_attention_heads));
    h.hidden_size         = static_cast<std::int32_t>(get_num(meta, a, "embedding_length",   0));
    h.intermediate_size   = static_cast<std::int32_t>(get_num(meta, a, "feed_forward_length", 0));
    h.max_position_embeddings = static_cast<std::int32_t>(
        get_num(meta, a, "context_length", 4096));

    // GGUF carries head_dim split as key/value lengths; HF combines into
    // a single head_dim. Default to hidden / num_heads when missing.
    const int kl = static_cast<int>(get_num(meta, a, "attention.key_length", 0));
    if (kl > 0) {
        h.head_dim = kl;
    } else if (h.num_attention_heads > 0) {
        h.head_dim = h.hidden_size / h.num_attention_heads;
    }

    h.rms_norm_eps = static_cast<float>(
        get_num(meta, a, "attention.layer_norm_rms_epsilon", 1e-6));
    h.rope_theta = static_cast<float>(get_num(meta, a, "rope.freq_base", 1e6f));

    // tie_word_embeddings: GGUF doesn't store this flag explicitly
    // (llama.cpp probes for the `output.weight` tensor). We use the
    // archive's pre-computed `has_output_weight`.
    h.tie_word_embeddings = !meta.has_output_weight;

    // Vocab size: GGUF doesn't always store it explicitly either. The
    // tokenizer.ggml.tokens array length is the canonical source, but
    // hparams's vocab_size is also implied by tok_embd's outer dim.
    // Most converters DO write `<arch>.vocab_size`; fall back to 0 (the
    // model loader will sanity-check against the embed tensor shape).
    h.vocab_size = static_cast<std::int32_t>(get_num(meta, a, "vocab_size", 0));

    // ---- MoE -----------------------------------------------------------
    h.num_experts         = static_cast<std::int32_t>(get_num(meta, a, "expert_count",        0));
    h.num_experts_per_tok = static_cast<std::int32_t>(get_num(meta, a, "expert_used_count",   0));
    h.moe_intermediate_size = static_cast<std::int32_t>(
        get_num(meta, a, "expert_feed_forward_length", h.intermediate_size));
    // norm_topk_prob is on by default for Qwen-MoE / Mixtral.
    h.norm_topk_prob = get_bool(meta, a, "expert_norm_topk_prob", true);

    // ---- Sliding window (gemma2 / gemma3 / older mistrals) -------------
    if (auto v = static_cast<int>(get_num(meta, a, "attention.sliding_window", 0)); v > 0) {
        h.sliding_window = v;
    }

    return h;
}

}  // namespace pie_portable_driver
