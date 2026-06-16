#pragma once

// Shared graph builder for the L4MA family + dense Gemma + MoE archs.
// Driven by `ArchSpec` flags; covers qwen2/qwen3, llama3, mistral3, phi3,
// gemma2/gemma3, olmo3, gpt-oss, mixtral, qwen3-moe.
//
// Gemma 4 uses a separate builder (`build_gemma4_graph`) because per-layer
// head_dim + KV-share don't fit the shared layout.

#include <ggml.h>

#include "executor/executor.hpp"
#include "graph_common.hpp"
#include "kv_cache.hpp"
#include "model.hpp"

namespace pie_portable_driver {

GraphResult build_qwen3_graph(ggml_context* ctx,
                              const Model& model,
                              KvCachePaged& kv,
                              const Executor::BatchPlan& plan);

}  // namespace pie_portable_driver
