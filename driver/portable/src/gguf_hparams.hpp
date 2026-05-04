#pragma once

// Build an `Hparams` from GGUF KV metadata (parallel to hf_config.cpp's
// `parse_hf_config` for safetensors checkpoints). GGUF stores most
// hparams under arch-prefixed keys like `qwen3.attention.head_count`;
// we map those into the same Hparams struct so the rest of the driver
// doesn't branch on the source format.

#include "hf_config.hpp"

namespace pie_portable_driver {

struct GgufMeta;

Hparams parse_gguf_hparams(const GgufMeta& meta);

}  // namespace pie_portable_driver
