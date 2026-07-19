// Host-oriented structural test for `model/registry.{hpp,cpp}` (THE arch
// table, cpp-refact.md Phase 5). Proves every supported `model_type`
// string resolves to its intended family + binder key, that kind-sharing
// aliases route through their own binder (never a shared fallback), and
// that an unrecognized arch is rejected. Does not load weights or touch a
// GPU: it only inspects the static table returned by `arch_table()` /
// `find_arch_entry()`, never invoking a row's `bind`/`create_model`
// closures (those need a real `LoadedModel` + device). This is a
// structural supplement to (not a substitute for) real load-parity
// coverage in `tests/load_parity/`.

#include <cstddef>
#include <cstdio>
#include <string>

#include "model/registry.hpp"

namespace {

using pie_cuda_driver::model::Family;
using pie_cuda_driver::model::find_arch_entry;
using pie_cuda_driver::model::arch_table;

int g_failures = 0;

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++g_failures;
    }
    return condition;
}

// Looks up `model_type`, then asserts it resolves to `family`/`binder_key`
// and carries a usable (non-empty) bind + create_model factory pair.
void expect_entry(const char* model_type, Family family, const char* binder_key) {
    const auto* entry = find_arch_entry(model_type);
    if (!expect(entry != nullptr, std::string(model_type) + ": not found in arch table")) {
        return;
    }
    expect(entry->model_type == model_type,
           std::string(model_type) + ": model_type field mismatch");
    expect(entry->family == family,
           std::string(model_type) + ": resolved to unexpected family");
    expect(entry->binder_key == binder_key,
           std::string(model_type) + ": expected binder_key '" + binder_key +
               "', got '" + entry->binder_key + "'");
    expect(static_cast<bool>(entry->bind),
           std::string(model_type) + ": missing bind (plan) factory");
    expect(static_cast<bool>(entry->create_model),
           std::string(model_type) + ": missing create_model factory");
    expect(static_cast<bool>(entry->validate_config),
           std::string(model_type) + ": missing validate_config hook");
}

}  // namespace

int main() {
    // ── LlamaLike: dense/GQA decoder. "qwen3"/"qwen2"/"llama"/"llama3"/
    //    "mistral" (bare) share one binder; mistral3/ministral3, phi3, and
    //    olmo2/olmo3 each get their own distinct binder within the family.
    expect_entry("qwen3", Family::LlamaLike, "llama_like");
#ifndef PIE_CUDA_QWEN_ONLY
    expect_entry("qwen2", Family::LlamaLike, "llama_like");
    expect_entry("llama", Family::LlamaLike, "llama_like");
    expect_entry("llama3", Family::LlamaLike, "llama_like");
    expect_entry("mistral", Family::LlamaLike, "llama_like");
    expect_entry("mistral3", Family::LlamaLike, "mistral3");
    expect_entry("ministral3", Family::LlamaLike, "mistral3");
    expect_entry("phi3", Family::LlamaLike, "phi3");
    expect_entry("olmo2", Family::LlamaLike, "olmo3");
    expect_entry("olmo3", Family::LlamaLike, "olmo3");

    // ── Mixtral: sparse top-k MoE. gpt_oss is the same family/model class
    //    through its own binder (MXFP4 experts + attention sinks).
    expect_entry("mixtral", Family::Mixtral, "mixtral");
    expect_entry("gpt_oss", Family::Mixtral, "gpt_oss");

    // ── Gemma generations.
    expect_entry("gemma2", Family::Gemma, "gemma2");
    expect_entry("gemma3", Family::Gemma, "gemma3");
    expect_entry("gemma3_text", Family::Gemma, "gemma3");
    expect_entry("gemma4", Family::Gemma4, "gemma4");
    expect_entry("gemma4_text", Family::Gemma4, "gemma4");
    expect_entry("gemma3n", Family::Gemma3n, "gemma3n");
    expect_entry("gemma3n_text", Family::Gemma3n, "gemma3n");

    // ── Hybrid recurrent / MoE families.
    expect_entry("nemotron_h", Family::NemotronH, "nemotron_h");
    expect_entry("deepseek_v4", Family::DeepSeekV4, "deepseek_v4");
    expect_entry("deepseek_v2", Family::Kimi, "kimi");
    expect_entry("deepseek_v3", Family::Kimi, "kimi");
    expect_entry("kimi_k2", Family::Kimi, "kimi");
    expect_entry("glm_moe_dsa", Family::Glm5, "glm5");
#endif

    // ── Qwen3.5 hybrid dense/MoE (aliases share one binder per variant).
    expect_entry("qwen3_5", Family::Qwen3_5, "qwen3_5");
    expect_entry("qwen3_5_text", Family::Qwen3_5, "qwen3_5");
    expect_entry("qwen3_5_moe", Family::Qwen3_5Moe, "qwen3_5_moe");
    expect_entry("qwen3_5_moe_text", Family::Qwen3_5Moe, "qwen3_5_moe");
    expect_entry("qwen3_moe", Family::Qwen3_5Moe, "qwen3_5_moe");

#ifndef PIE_CUDA_QWEN_ONLY
    // ── Multimodal / audio-output families.
    expect_entry("qwen3_vl", Family::Qwen3VL, "qwen3_vl_text");
    expect_entry("qwen3_vl_text", Family::Qwen3VL, "qwen3_vl_text");
    expect_entry("csm", Family::Csm, "csm");
#endif

    // ── Table shape: exactly the allowlisted set, one row per string, no
    //    duplicates.
    const auto& table = arch_table();
    const std::size_t expected_rows =
#ifdef PIE_CUDA_QWEN_ONLY
        6;
#else
        33;
#endif
    expect(table.size() == expected_rows,
           "arch table has the wrong number of supported model_type strings, got " +
               std::to_string(table.size()));
    for (std::size_t i = 0; i < table.size(); ++i) {
        for (std::size_t j = i + 1; j < table.size(); ++j) {
            expect(table[i].model_type != table[j].model_type,
                   "duplicate model_type row: " + table[i].model_type);
        }
    }

    // ── Unknown arch must be rejected — never a silent fallback to any
    //    family (this is the whole point of killing the allowlist chain).
    expect(find_arch_entry("totally_unknown_arch") == nullptr,
           "unknown arch must be rejected, not silently resolved");
    expect(find_arch_entry("") == nullptr,
           "empty model_type must be rejected");
    expect(find_arch_entry("Qwen3") == nullptr,
           "lookup must be exact-match (case-sensitive), not fuzzy");

    if (g_failures != 0) {
        std::fprintf(stderr, "model_registry_test: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("model_registry_test: OK (%zu arch rows)\n", table.size());
    return 0;
}
