#pragma once

// Build an `Hparams` from GGUF KV metadata (parallel to hf_config.cpp's
// `parse_hf_config` for safetensors checkpoints). GGUF stores most
// hparams under arch-prefixed keys like `qwen3.attention.head_count`;
// we map those into the same Hparams struct so the rest of the driver
// doesn't branch on the source format.

#include <vector>

#include "hf_config.hpp"

namespace pie_portable_driver {

struct GgufMeta;

Hparams parse_gguf_hparams(const GgufMeta& meta);

// Fail loudly when a GGUF gemma4 checkpoint is missing the full-layer
// proportional-RoPE frequency factors. On the GGUF path those factors live
// ONLY in the root `rope_freqs.weight` tensor — no scalar partial-rotary KV
// exists to reconstruct them. If the tensor is absent while the model has any
// full-attention ('g') layer, every full layer would silently fall back to a
// full rotation (`ggml_rope_ext` with freq_factors=nullptr), corrupting its
// position encoding with no error. Throws std::runtime_error in that case.
// Exposed (rather than inlined at the load site) so the pie-bin regression
// test can exercise the invariant without a model file.
void validate_gemma4_rope_freqs(const std::vector<char>& layer_types,
                                bool rope_freqs_present);

}  // namespace pie_portable_driver
