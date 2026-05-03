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
#include <type_traits>

#include <atomic>

#include "distributed.hpp"
#include "forward_graph.hpp"
#include "model/llama_like.hpp"
#include "persistent_inputs.hpp"
#include "slot_allocator.hpp"

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

// Type-erased forward call. Two-phase API to support CUDA-graph capture
// without staleness bugs:
//
//   * `prepare`  — optional host-side hook. Called BEFORE every fire
//                  (direct, graph-capture, or graph-replay). Computes
//                  per-step planning state (e.g. flashinfer's decode
//                  plan) and uploads it to pinned/device buffers that
//                  the captured body reads via cudaMemcpyAsync. When
//                  empty, the body is treated as self-contained — but
//                  the request handler then disables graph capture for
//                  that arch (host-side work inside a captured region
//                  freezes the plan at first-fire KV size, producing
//                  garbage after the KV grows past one page).
//
//   * `body`     — required device-side kernel sequence. Same shape
//                  every fire of a given (R, num_pages, …) bucket; the
//                  captured graph re-reads buffer contents that
//                  `prepare` refreshes. No host loops, no allocs, no
//                  std::vector inside.
//
// main.cpp builds both closures from the per-arch weights + cfg.
struct ForwardFn {
    // Whether the request handler may capture this forward into a CUDA
    // graph for replay. False by default — flashinfer's decode plan
    // bakes per-fire metadata (`padded_batch_size`, `split_kv`, etc.)
    // into kernel args at capture time, which goes stale once the KV
    // length grows past the capture-time bucket. Setting this true
    // requires the body to upload kernel args via mechanisms that
    // graph capture re-reads each replay (e.g. fixed_split_size +
    // cudaGraphExecKernelNodeSetParams). None of the current archs
    // satisfy that yet — left here as the explicit flip we'll set
    // alongside the graph-safe attention dispatch refactor.
    bool graph_safe = false;

    using BodyFn = std::function<void(
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
        const std::int32_t*  /* custom_mask_indptr_d (nullable) */,
        const std::int32_t*  /* slot_ids_h     host, len R, nullable */,
        const std::uint8_t*  /* is_fresh_h     host, len R, nullable */,
        const std::int32_t*  /* slot_ids_d     device, len R, nullable */
    )>;

    using PrepareFn = std::function<void(
        AttentionWorkspace&,
        const std::uint32_t* /* kv_page_indptr_h */,
        int                  /* num_requests R */,
        bool                 /* is_pure_decode */
    )>;

    // Empty by default → request handler falls back to "direct call only;
    // no graph capture" mode for this arch.
    PrepareFn prepare;
    BodyFn    body;

    // Convenience: `forward_fn = [...]` assigns the lambda as the body.
    // entry.cpp uses this terser pattern; the older `forward_fn.body =
    // [...]` form continues to work because we leave `body` public.
    template <class F>
        requires(!std::is_same_v<std::decay_t<F>, ForwardFn>)
    ForwardFn& operator=(F&& f) {
        body = std::forward<F>(f);
        return *this;
    }
    ForwardFn() = default;
    ForwardFn(const ForwardFn&) = default;
    ForwardFn(ForwardFn&&) noexcept = default;
    ForwardFn& operator=(const ForwardFn&) = default;
    ForwardFn& operator=(ForwardFn&&) noexcept = default;
};

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

    // Tensor-parallel comm. Non-null on rank 0 when tp_size > 1 — the
    // request handler broadcasts the per-fire inputs to TP followers
    // before invoking the forward kernels. On TP followers a parallel
    // service loop (`tp_follower_serve`) consumes those broadcasts.
    NcclComm* tp_comm = nullptr;

    // Per-request linear-attention state-cache slot mapping. Rank 0
    // owns the LRU; followers receive the pre-resolved slot_ids and
    // is_fresh flags via NCCL broadcast (see tp_broadcast_inputs).
    // Inert on archs that don't use a linear-attention state cache.
    SlotAllocator slot_alloc;
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

// TP-follower service loop. Called only on TP ranks > 0. Mirrors
// `handle_fire_batch` minus shmem decode, sampling, and response: the
// loop blocks on `ncclBroadcast(root=0)` for each fire's header + inputs,
// runs the same `forward_fn.body` so the all-reduces inside complete
// against rank 0, then loops. `stop` is checked between fires; rank 0
// also sends an explicit shutdown header before tearing down so a
// follower waiting on a broadcast unblocks cleanly.
void tp_follower_serve(ForwardContext& ctx, std::atomic<bool>& stop);

// Send the shutdown sentinel header from rank 0 so followers exit their
// `tp_follower_serve` loop on the next broadcast.
void tp_send_shutdown(NcclComm& comm);

}  // namespace pie_cuda_driver
