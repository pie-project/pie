#pragma once

/// What Gemma-4 binds (`model/registry.cpp` rows).
///
/// The only family the encode pipeline can scope a load to. An encode-scoped
/// rank declares the vision and audio towers and nothing else, so it never
/// allocates the language model — and because that is a *declaration* rather
/// than a filter applied to a finished contract, the plan it compiles has no
/// trace of the tensors it skipped.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {

/// gemma4, gemma4_text.
inline void author_gemma4_contract(ContractBuilder& b) {
    b.allow_encode_scope();
    // The decoder is nested; the vision and audio towers are not, and they
    // have `self_attn.q_proj.weight` of their own.
    b.decoder_layer_prefix("model.language_model.layers.");
    author_dense_contract(b);
}

}  // namespace pie_cuda_driver::model
