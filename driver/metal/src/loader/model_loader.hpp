#pragma once

// model_loader — the loader's top-level entry point (delta owns this seam).
// Given an HF snapshot directory it parses config.json, loads the safetensors
// weights straight into MLX arrays, binds them onto charlie's ModelWeights via
// the per-arch bind_* schema, constructs the runtime ModelGraph, and allocates
// the paged-KV cache sized from the real model geometry. The result is a
// self-contained bundle the in-process service (alpha) drives for a forward.

#include <memory>
#include <string>

#include "../config.hpp"            // BatchingConfig
#include "../kv_cache.hpp"          // PagedKvCache
#include "../model/config.hpp"      // model::ModelConfig
#include "../model/model_graph.hpp" // model::ModelGraph
#include "../ops/tensor.hpp"        // DType

namespace pie_metal_driver::loader {

// Capability geometry derived from the parsed config — the real numbers the
// driver advertises in its READY caps (supersedes entry.cpp's best-effort
// config.json peek). Field names mirror the caps JSON alpha emits.
struct ModelCapabilities {
    std::string  arch_name        = "unknown";
    std::int32_t vocab_size       = 0;
    std::int32_t max_model_len    = 0;
    std::int32_t num_hidden_layers   = 0;
    std::int32_t num_attention_heads = 0;
    std::int32_t num_key_value_heads = 0;
    std::int32_t head_dim         = 0;
    std::int32_t hidden_size      = 0;
    std::string  activation_dtype = "bf16";  // bf16 / fp16 / fp32
};

ModelCapabilities derive_capabilities(const model::ModelConfig& cfg);

// Lower-level handoff for callers that own the runtime assembly (alpha's
// entry.cpp): parse config.json + load/bind the safetensors weights, but stop
// short of building the graph/KV. The caller drives make_model_graph + the
// PagedKvCache + Executor itself. Throws on missing config / unsupported arch /
// absent weights.
struct LoadedWeights {
    model::ModelConfig  config;
    model::ModelWeights weights;
};

LoadedWeights load_weights(const std::string& hf_path);

// A fully-loaded, ready-to-run model: config + graph + KV cache + caps.
struct LoadedModel {
    model::ModelConfig                config;
    ModelCapabilities                 caps;
    std::unique_ptr<model::ModelGraph> graph;
    std::unique_ptr<PagedKvCache>      kv;
};

// Load the model snapshot at `hf_path` and allocate its paged-KV cache using
// the batching geometry (page size / total pages / kv dtype). Throws with a
// clear diagnostic on a missing config, unsupported architecture, or absent
// weights.
LoadedModel load_model(const std::string& hf_path, const BatchingConfig& batching);

}  // namespace pie_metal_driver::loader
