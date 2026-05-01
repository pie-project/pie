#pragma once

// Gemma 4 / 3n graph builder. Distinct from `build_qwen3_graph` because
// gemma4 has per-layer head_dim (sliding=head_dim, full=head_dim_global)
// and KV-cache sharing across layer types — the shared builder's
// uniform-head-dim assumption doesn't hold.

#include <ggml.h>

#include "forward.hpp"
#include "graph_common.hpp"
#include "kv_cache.hpp"
#include "model.hpp"

namespace pie_portable_driver {

GraphResult build_gemma4_graph(ggml_context* ctx,
                               const Model& model,
                               KvCachePaged& kv,
                               const ForwardEngine::BatchPlan& plan);

}  // namespace pie_portable_driver
