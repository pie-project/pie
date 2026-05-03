#pragma once

// Phi-3-small graph builder. See graph_phi3small.cpp for arch notes.

#include <ggml.h>

#include "forward.hpp"
#include "graph_common.hpp"
#include "kv_cache.hpp"
#include "model.hpp"

namespace pie_portable_driver {

GraphResult build_phi3small_graph(ggml_context* ctx,
                                  const Model& model,
                                  KvCachePaged& kv,
                                  const ForwardEngine::BatchPlan& plan);

}  // namespace pie_portable_driver
