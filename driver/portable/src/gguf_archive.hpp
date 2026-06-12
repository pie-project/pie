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
                double num_value = 0.0; bool bool_value = false; };
    std::unordered_map<std::string, KV> kv;
};

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
