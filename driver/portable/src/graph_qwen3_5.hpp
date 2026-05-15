#pragma once

// Qwen 3.5 / 3.6 graph builder (hybrid: gated-delta-rule "linear
// attention" + standard GQA with mrope + output gate). Per-layer-type
// dispatch happens inside the builder. Linear-attention layers maintain
// per-request recurrent state (held in StateCache); full-attention
// layers use the existing paged KV cache.
//
// V0 limitations:
//   - Per-request slow path only (no M11 packed-decode).
//   - GPU-greedy / uniform-top-K fast paths disabled (logits-only output).
//   - MTP head (`mtp.*` weights) ignored.
//   - Vision tower ignored — text-only.

#include <ggml.h>

#include "executor/executor.hpp"
#include "graph_common.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "state_cache.hpp"

namespace pie_portable_driver {

GraphResult build_qwen3_5_graph(ggml_context* ctx,
                                const Model& model,
                                KvCachePaged& kv,
                                StateCache& state,
                                const Executor::BatchPlan& plan);

}  // namespace pie_portable_driver
