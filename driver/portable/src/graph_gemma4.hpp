#pragma once

// Gemma 4 / 3n graph builder. Distinct from `build_qwen3_graph` because
// gemma4 has per-layer head_dim (sliding=head_dim, full=head_dim_global)
// and KV-cache sharing across layer types — the shared builder's
// uniform-head-dim assumption doesn't hold.

#include <cstdint>

#include <ggml.h>

#include "executor/executor.hpp"
#include "graph_common.hpp"
#include "kv_cache.hpp"
#include "model.hpp"

namespace pie_portable_driver {

GraphResult build_gemma4_graph(ggml_context* ctx,
                               const Model& model,
                               KvCachePaged& kv,
                               const Executor::BatchPlan& plan);

// View a pure-decode packed mask as the single-query mask used by manual SDPA.
ggml_tensor* gemma4_manual_decode_mask_view(ggml_context* ctx,
                                            ggml_tensor* packed_mask,
                                            std::int32_t max_n_kv,
                                            std::int32_t n_req);

}  // namespace pie_portable_driver
