#pragma once

// Reading `pie.model/1`, the compiled model descriptor pie writes into every
// `.zt` artifact.
//
// The replacement for `parse_hf_config`. HuggingFace's `config.json` is
// normalized once, at `pie model convert`, and what reaches a driver is the
// result: a flat document with every defaulting rule already resolved, no
// `text_config` nesting to step through, no alternate spellings to probe, and
// no per-`model_type` conditionals. Two implementations of that normalization
// in two languages is the skew the artifact exists to remove.

#include <string>

#include "model/config.hpp"

namespace pie_cuda_driver {

/// Parses a `pie.model/1` descriptor into the same struct `parse_hf_config`
/// produces. Throws on a version this build does not read, or on a field the
/// descriptor should carry and does not.
HfConfig parse_pie_model_descriptor(const std::string& json);

}  // namespace pie_cuda_driver
