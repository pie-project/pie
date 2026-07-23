#pragma once

// Sizing policy for the attention plan/dispatch scratch. The scratch buffer
// type itself (`AttentionWorkspace`) is defined in ops/attention_workspace.hpp
// (every attention kernel wrapper takes one by reference); this header pulls
// it in and adds the batch-level sizing functions below, which need the
// model's HfConfig and are therefore not ops-owned themselves.
//
// Allocated once at boot (see Context::create); reused across all forward
// passes.

#include <cstddef>

#include <cuda_runtime.h>

#include "ops/attention_workspace.hpp"

namespace pie_cuda_driver {

class HfConfig;
class Config;

// True if any layer in the HF config uses a non-full attention shape
// (e.g. sliding window). Cheap test the planner uses to gate the
// FlashInfer fast-path attention budget.
bool has_non_full_attention_layers(const HfConfig& hf);

// Byte budget for FlashInfer's per-fire float scratch on the active arch
// and TP layout. Returns a conservative base when the fast-path budget
// doesn't apply. With `cuda_graphs`, additionally covers the graph-mode
// prefill plan's forced split-KV carve for shapes up to
// (max_forward_tokens, max_requests) — graph-mode planning always carves
// (its work split must be content-independent), so capturable prefill
// coverage is bounded by this budget (the plan demotes to a non-graph
// plan when its carve would not fit).
std::size_t attention_float_workspace_bytes(const HfConfig& hf,
                                            const Config& cfg,
                                            const cudaDeviceProp& prop,
                                            int max_requests,
                                            int max_forward_tokens,
                                            bool cuda_graphs);

}  // namespace pie_cuda_driver
