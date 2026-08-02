#pragma once

/// The facts vocabulary the boot sends across the request boundary — all
/// that remains of `model/contract.hpp` after the harvest.
///
/// The four family authors and the shared authoring toolkit that used to
/// live beside these types are gone: authoring happens on the far side of
/// `pie_loader_compile_model` (`plan/model-in-rust.md` §2), in the same
/// `pie_model::contract` registry the CUDA driver already boots through.
/// What the driver still owns is what only it can know: the facts it
/// parsed from `config.json`, and which decode DAG it is about to run.

#include <cstdint>
#include <string_view>

namespace pie::metal::model {

/// Everything a schema needs that is a property of the checkpoint's CONFIG
/// rather than of its tensors. A contract only sees tensors, so anything the
/// author has to know about the shape of the model arrives here.
struct ContractFacts {
    /// Gemma4's KV sharing, in the config's own vocabulary: a layer
    /// `>= num_hidden_layers - num_kv_shared_layers` attends KV another layer
    /// wrote. Zero `num_kv_shared_layers` means every layer owns its own,
    /// which is every family except gemma4's "E" variants. (This replaced the
    /// precomputed `first_kv_shared_layer` when authoring moved: the request
    /// carries config facts, and the subtraction is the author's.)
    std::uint32_t num_hidden_layers = 0;
    std::uint32_t num_kv_shared_layers = 0;

    /// Whether `lm_head` IS `embed_tokens` (`tie_word_embeddings`, defaulting
    /// to true when the config omits it). The llama and Qwen3.5 schemas read
    /// it, and only as a hint: a checkpoint that actually ships an `lm_head`
    /// overrides it, because mapping a real tensor onto `shared_embedding`
    /// declares that name twice and fails the load.
    bool tied_embeddings = true;

    /// `config.json`'s `quantization` width and group. A contract sees only
    /// tensors, and the tensors of an 8-bit g64 weight are indistinguishable
    /// from those of a 4-bit g128 one: same packed u32 count, same number of
    /// scales. Solving for the group while assuming the width is how a
    /// published 8-bit llama checkpoint was refused as "g128/b4".
    ///
    /// Zero means the config declared no quantization. gpt-oss deliberately
    /// does NOT read these: its `quantization` block sets a global mxfp4 g32
    /// and then overrides nearly every module back to affine g64, so the global
    /// pair is wrong for most of its tensors and that schema solves per tensor.
    std::uint32_t quant_bits = 0;
    std::uint32_t quant_group_size = 0;
};

/// Which storage schema and decode DAG a checkpoint asks for.
///
/// One place answers this. The geometry and the executor need the same answer,
/// and two independent readings of `model_type` would be two chances to
/// disagree. The lists mirror the `Naming::Mlx` rows of the Rust author
/// registry (`model/src/contract.rs`) — the loader dispatches by the same
/// strings on its side of the call, and refuses one this table would too.
enum class ModelFamily { Unknown, Qwen35, Gemma4, GptOss, Llama };

inline ModelFamily model_family_of(std::string_view model_type) {
    for (std::string_view name : {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
                                  "qwen3_next", "qwen3_next_text", "qwen3_6"}) {
        if (model_type == name) return ModelFamily::Qwen35;
    }
    if (model_type == "gemma4" || model_type == "gemma4_text") return ModelFamily::Gemma4;
    if (model_type == "gpt_oss") return ModelFamily::GptOss;
    for (std::string_view name : {"llama", "llama3", "mistral", "qwen2", "qwen3",
                                  "qwen3_moe", "qwen2_moe"}) {
        if (model_type == name) return ModelFamily::Llama;
    }
    return ModelFamily::Unknown;
}

inline bool is_supported_model_type(std::string_view model_type) {
    return model_family_of(model_type) != ModelFamily::Unknown;
}

}  // namespace pie::metal::model
