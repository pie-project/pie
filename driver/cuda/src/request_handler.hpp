#pragma once

// Single-request handler for the BPIQ `fire_batch` shmem method.
// Lifted from `main.cpp` so the entry point stays focused on init and
// the dispatch loop. Body is unchanged from its lambda incarnation;
// follow-up refactors (spec expansion, sampling dispatch, msgpack
// sub-passes) split this further.

#include <cstddef>
#include <cstdint>
#include <span>

#include "forward_graph.hpp"
#include "persistent_inputs.hpp"

namespace pie_cuda_driver {

class Engine;
class KvCache;
class AttentionWorkspace;
struct SlotRequest;

namespace model {
struct Qwen3Weights;
struct Qwen3Workspace;
}  // namespace model

namespace ops {
class CublasHandle;
}  // namespace ops

// Stable references the request handler needs across calls. Constructed
// once after engine/workspace allocation in `main()` and held alongside
// the shmem server.
struct ForwardContext {
    Engine& engine;
    const model::Qwen3Weights& weights;
    model::Qwen3Workspace& ws;
    KvCache& kv_cache;
    AttentionWorkspace& attn_ws;
    ops::CublasHandle& cublas;
    int max_workspace_tokens;
    // Pre-allocated input buffers — refreshed per fire via memcpy
    // rather than re-allocated. See `persistent_inputs.hpp`.
    PersistentInputs& inputs;
    // Optional CUDA-graph cache. When non-null, decode-only fires
    // attempt graph capture/replay; otherwise the forward runs directly.
    ForwardGraphCache* graph_cache = nullptr;
};

// Decode a `fire_batch` BPIQ payload, run the forward pass + sampling
// pipeline, and write a BPIS response into `response`. Returns the
// number of bytes written (0 on error). `handled` is the cumulative
// fire_batch counter — used as the PRNG offset and to gate logging
// cadence.
std::size_t handle_fire_batch(
    const SlotRequest& req,
    std::span<std::uint8_t> response,
    ForwardContext& ctx,
    std::uint64_t handled);

}  // namespace pie_cuda_driver
