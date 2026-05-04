#pragma once

// Phi-3.5-MoE graph builder. See graph_phi3_5moe.cpp for arch notes.

#include <ggml.h>

#include "forward.hpp"
#include "graph_common.hpp"
#include "kv_cache.hpp"
#include "model.hpp"

namespace pie_portable_driver {

GraphResult build_phi3_5moe_graph(ggml_context* ctx,
                               const Model& model,
                               KvCachePaged& kv,
                               const ForwardEngine::BatchPlan& plan);

}  // namespace pie_portable_driver
