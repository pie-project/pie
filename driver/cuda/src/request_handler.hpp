#pragma once

// Single-request handler for the BPIQ `fire_batch` shmem method.
// Lifted from `main.cpp` so the entry point stays focused on init and
// the dispatch loop. Body is unchanged from its lambda incarnation;
// follow-up refactors (spec expansion, sampling dispatch, msgpack
// sub-passes) split this further.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

#include "forward_graph.hpp"
#include "model/llama_like.hpp"
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

// Type-erased forward call. The closure captures the per-arch weights
// + cfg (Qwen3Weights+LlamaLikeForwardCfg, MixtralWeights, Gemma2Weights+
// Gemma2ForwardCfg, …) and exposes a single signature so the request
// handler doesn't have to branch on model_type. main.cpp builds the
// closure once and stows it on `ForwardContext::forward_fn`.
using ForwardFn = std::function<void(
    model::Qwen3Workspace&,
    KvCache&,
    AttentionWorkspace&,
    ops::CublasHandle&,
    const std::int32_t*  /* token_ids        device */,
    const std::int32_t*  /* positions        device */,
    const std::uint32_t* /* qo_indptr        device */,
    const std::uint32_t* /* kv_page_indices  device */,
    const std::uint32_t* /* kv_page_indptr   device */,
    const std::uint32_t* /* kv_last_page_lens device */,
    const std::uint32_t* /* qo_indptr_h        host */,
    const std::uint32_t* /* kv_page_indptr_h   host */,
    int                  /* total_tokens N */,
    int                  /* num_requests R */,
    bool                 /* is_pure_decode */,
    const std::uint8_t*  /* custom_mask_d  (nullable) */,
    const std::int32_t*  /* custom_mask_indptr_d (nullable) */
)>;

// Stable references the request handler needs across calls. Constructed
// once after engine/workspace allocation in `main()` and held alongside
// the shmem server.
struct ForwardContext {
    Engine& engine;
    model::Qwen3Workspace& ws;
    KvCache& kv_cache;
    AttentionWorkspace& attn_ws;
    ops::CublasHandle& cublas;
    int max_workspace_tokens;
    // Pre-allocated input buffers — refreshed per fire via memcpy
    // rather than re-allocated. See `persistent_inputs.hpp`.
    PersistentInputs& inputs;
    // Type-erased forward call. The captured weights / cfg / model
    // function are model-specific; the call site is uniform.
    ForwardFn forward_fn;
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
