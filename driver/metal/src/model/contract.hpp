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
};

inline bool is_supported_model_type(std::string_view model_type) {
    return qwen3_5::is_supported_model_type(model_type) ||
           gemma4::is_supported_model_type(model_type);
}

inline void author_model_contract(const Checkpoint& checkpoint, std::string_view model_type,
                                  const pie_loader::DeviceTarget& target, ModelContract& out,
                                  const ContractFacts& facts = {}) {
    if (gemma4::is_supported_model_type(model_type)) {
        gemma4::author_model_contract(checkpoint, model_type, target, out,
                                      facts.first_kv_shared_layer);
        return;
    }
    // Refusal for an unknown family belongs to the schema that would have
    // handled it, so the message names the schema that was actually consulted.
    qwen3_5::author_model_contract(checkpoint, model_type, target, out);
}

}  // namespace pie::metal::model
