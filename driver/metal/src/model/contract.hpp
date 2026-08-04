#pragma once

/// Which storage schema to author against, chosen by `model_type`.
///
/// One table, on this side of the loader call. The loader is never told which
/// model this is — it type-checks whatever contract it is handed against
/// whatever the files contain (`loader/architecture.md` §12 row 12) — so this
/// dispatch exists purely so the driver can pick the family whose tensor names
/// it is about to bind.

#include <string>
#include <string_view>

#include "gemma4/gemma4_contract.hpp"
#include "gptoss/gptoss_contract.hpp"
#include "llama/llama_contract.hpp"
#include "qwen3_5/qwen3_5_contract.hpp"

namespace pie::metal::model {

/// Everything a schema needs that is a property of the checkpoint's CONFIG
/// rather than of its tensors. A contract only sees tensors, so anything it has
/// to know about the shape of the model arrives here.
struct ContractFacts {
    /// First layer whose KV is another layer's (gemma4:
    /// `num_hidden_layers - num_kv_shared_layers`). -1 when every layer owns its
    /// own, which is every family except gemma4's "E" variants.
    int first_kv_shared_layer = -1;

    /// Whether `lm_head` IS `embed_tokens` (`tie_word_embeddings`, defaulting
    /// to true when the config omits it). The llama and Qwen3.5 schemas read
    /// it, and only as a hint: a checkpoint that actually ships an `lm_head` overrides
    /// it, because mapping a real tensor onto `shared_embedding` declares that
    /// name twice and fails the load.
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
    int quant_bits = 0;
    int quant_group_size = 0;
};

/// Which storage schema and decode DAG a checkpoint asks for.
///
/// One place answers this. The contract, the geometry and the executor all need
/// the same answer, and three independent readings of `model_type` would be
/// three chances to disagree.
enum class ModelFamily { Unknown, Qwen35, Gemma4, GptOss, Llama };

inline ModelFamily model_family_of(std::string_view model_type) {
    if (qwen3_5::is_supported_model_type(model_type)) return ModelFamily::Qwen35;
    if (gemma4::is_supported_model_type(model_type)) return ModelFamily::Gemma4;
    if (gptoss::is_supported_model_type(model_type)) return ModelFamily::GptOss;
    if (llama::is_supported_model_type(model_type)) return ModelFamily::Llama;
    return ModelFamily::Unknown;
}

inline bool is_supported_model_type(std::string_view model_type) {
    return qwen3_5::is_supported_model_type(model_type) ||
           gemma4::is_supported_model_type(model_type) ||
           gptoss::is_supported_model_type(model_type) ||
           llama::is_supported_model_type(model_type);
}

inline void author_model_contract(const Checkpoint& checkpoint, std::string_view model_type,
                                  const pie_loader::DeviceTarget& target, ModelContract& out,
                                  const ContractFacts& facts = {}) {
    if (gemma4::is_supported_model_type(model_type)) {
        gemma4::author_model_contract(checkpoint, model_type, target, out,
                                      facts.first_kv_shared_layer, facts.quant_bits,
                                      facts.quant_group_size);
        return;
    }
    if (gptoss::is_supported_model_type(model_type)) {
        gptoss::author_model_contract(checkpoint, model_type, target, out);
        return;
    }
    if (llama::is_supported_model_type(model_type)) {
        llama::author_model_contract(checkpoint, model_type, target, out,
                                     facts.tied_embeddings, facts.quant_bits,
                                     facts.quant_group_size);
        return;
    }
    // Refusal for an unknown family belongs to the schema that would have
    // handled it, so the message names the schema that was actually consulted.
    qwen3_5::author_model_contract(checkpoint, model_type, target, out, facts.tied_embeddings,
                                   facts.quant_bits, facts.quant_group_size);
}

}  // namespace pie::metal::model
