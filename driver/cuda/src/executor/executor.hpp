#pragma once

// Forward executor for the `fire_batch` inproc method. Lifted from
// entry.cpp so the entry point stays focused on startup and service
// wiring. Body is unchanged from its lambda incarnation; follow-up
// refactors (spec expansion, sampling dispatch, sub-passes) split this
// further.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <type_traits>

#include <atomic>

#include "distributed.hpp"
#include "executor/forward_graph.hpp"
#include <pie_bridge/view.hpp>
#include "model/llama_like.hpp"
#include "executor/persistent_inputs.hpp"
#include <pie_bridge/response_builder.hpp>

namespace pie_cuda_driver {

class LoadedModel;
class KvCache;
class AttentionWorkspace;
class Qwen3_5StateCache;

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
//                  the executor then disables graph capture for
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
// entry.cpp builds both closures from the per-arch weights + cfg.
struct ForwardFn {
    // Whether the executor may capture this forward into a CUDA
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
    bool supports_tp_greedy_argmax = false;
    bool supports_compact_logits = false;
    bool supports_small_prefill_graph = false;

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
        const std::uint32_t* /* kv_page_indices_h  host */,
        const std::uint32_t* /* kv_page_indptr_h   host */,
        const std::uint32_t* /* kv_last_page_lens_h host */,
        int                  /* total_tokens N */,
        int                  /* num_requests R */,
        bool                 /* is_pure_decode */,
        const std::uint8_t*  /* custom_mask_d  (nullable) */,
        const std::int32_t*  /* custom_mask_indptr_d (nullable) */,
        const std::int32_t*  /* slot_ids_h     host, len R, nullable */,
        const std::uint8_t*  /* is_fresh_h     host, len R, nullable */,
        const std::int32_t*  /* slot_ids_d     device, len R, nullable */,
        const std::int32_t*  /* logit_row_indices_d device, nullable */,
        int                  /* num_logit_rows */,
        bool                 /* tp_greedy_argmax */
    )>;

    using MtpFn = std::function<void(
        model::Qwen3Workspace&,
        KvCache&,
        ops::CublasHandle&,
        const std::int32_t*  /* token_ids device */,
        const std::int32_t*  /* position_ids device */,
        const std::int32_t*  /* base_hidden_row_indices device */,
        const std::int32_t*  /* request_ids device */,
        const std::uint32_t* /* kv_page_indices device */,
        const std::uint32_t* /* kv_page_indptr device */,
        const std::uint32_t* /* kv_last_page_lens device */,
        int                  /* num_tokens */,
        int                  /* draft_step */,
        int                  /* max_global_tokens */
    )>;

    using MtpProcessFn = std::function<void(
        model::Qwen3Workspace&,
        KvCache&,
        ops::CublasHandle&,
        const std::int32_t*  /* token_ids device */,
        const std::int32_t*  /* positions device */,
        const std::uint32_t* /* qo_indptr device */,
        const std::uint32_t* /* kv_page_indices device */,
        const std::uint32_t* /* kv_page_indptr device */,
        const std::uint32_t* /* kv_last_page_lens device */,
        const std::int32_t*  /* slot_ids device, nullable */,
        const std::int32_t*  /* source_row_indices device, nullable */,
        int                  /* total_tokens */,
        int                  /* num_requests */
    )>;

    using MtpPrepareFn = std::function<void(
        const std::uint32_t* /* kv_page_indptr_h */,
        const std::uint32_t* /* kv_last_page_lens_h */,
        int                  /* num_rows */,
        int                  /* page_size */,
        cudaStream_t         /* stream */
    )>;

    struct PrepareInputs {
        const std::uint32_t* qo_indptr_h = nullptr;
        const std::uint32_t* kv_page_indices_h = nullptr;
        const std::uint32_t* kv_page_indices_d = nullptr;
        const std::uint32_t* kv_page_indptr_h = nullptr;
        const std::uint32_t* kv_page_indptr_d = nullptr;
        const std::uint32_t* kv_last_page_lens_h = nullptr;
        const std::uint32_t* kv_last_page_lens_d = nullptr;
        int total_tokens = 0;
        int num_requests = 0;
        bool is_pure_decode = false;
    };

    using PrepareFn = std::function<void(
        AttentionWorkspace&,
        const PrepareInputs&
    )>;

    using GraphLayoutFn = std::function<std::uint32_t()>;
    using LogitsModeFn = std::function<void(bool)>;
    using SetFusedArgmaxOutputFn = std::function<void(std::int32_t*)>;
    using FusedArgmaxDoneFn = std::function<bool()>;

    // Empty by default → executor falls back to "direct call only;
    // no graph capture" mode for this arch.
    PrepareFn prepare;
    GraphLayoutFn graph_layout;
    LogitsModeFn set_logits_argmax_only;
    SetFusedArgmaxOutputFn set_fused_argmax_output;
    FusedArgmaxDoneFn fused_argmax_done;
    bool supports_fused_lmhead_argmax = false;
    BodyFn    body;
    MtpFn     mtp;
    MtpPrepareFn mtp_prepare;
    MtpProcessFn mtp_process;
    int mtp_num_drafts = 1;

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

struct SystemSpecDraftRequest {
    int request_index = -1;
    int source_row = -1;
    std::uint32_t accepted_token = 0;
    std::uint32_t source_position = 0;
    std::uint32_t first_draft_position = 0;
    int last_match = -1;
    int last_num_drafts = 0;
};

struct SystemSpecDraftInputs {
    model::Qwen3Workspace& target_ws;
    KvCache& kv_cache;
    AttentionWorkspace& attn_ws;
    ops::CublasHandle& cublas;
    std::span<const SystemSpecDraftRequest> requests;
    std::span<const std::uint32_t> kv_page_indices;
    std::span<const std::uint32_t> kv_page_indptr;
    int page_size = 0;
    int max_drafts = 0;
};

using NativeSystemDraftNextFn = std::function<void(
    const SystemSpecDraftInputs&,
    std::span<pie_driver::PerRequestOutput>)>;

struct NativeSystemCommitInputs {
    model::Qwen3Workspace& target_ws;
    KvCache& kv_cache;
    ops::CublasHandle& cublas;
    const std::int32_t* token_ids = nullptr;
    const std::int32_t* positions = nullptr;
    const std::uint32_t* qo_indptr = nullptr;
    const std::uint32_t* kv_page_indices = nullptr;
    const std::uint32_t* kv_page_indptr = nullptr;
    const std::uint32_t* kv_last_page_lens = nullptr;
    const std::int32_t* slot_ids = nullptr;
    const std::int32_t* source_row_indices = nullptr;
    int total_tokens = 0;
    int num_requests = 0;
};

struct NativeSystemDrafter {
    using CommitVerifiedPrefixFn = std::function<void(
        const NativeSystemCommitInputs&)>;

    using DraftStepFn = std::function<void(
        model::Qwen3Workspace&,
        KvCache&,
        ops::CublasHandle&,
        const std::int32_t*  /* token_ids device */,
        const std::int32_t*  /* position_ids device */,
        const std::int32_t*  /* base_hidden_row_indices device */,
        const std::int32_t*  /* request_ids device */,
        const std::uint32_t* /* kv_page_indices device */,
        const std::uint32_t* /* kv_page_indptr device */,
        const std::uint32_t* /* kv_last_page_lens device */,
        std::int32_t*        /* sampled_token_ids device, optional */,
        int                  /* num_tokens */,
        int                  /* draft_step */,
        int                  /* max_global_tokens */
    )>;

    int max_drafts = 0;
    // Position passed to the first low-level draft step is
    // source_position + draft_position_offset; later steps advance by one.
    int draft_position_offset = 1;
    // Some native drafters keep the just-generated draft chain in local
    // history instead of writing it into their paged cache. For those,
    // global-cache attention length is the fixed prefix position, not the
    // current draft position.
    bool draft_global_cache_uses_prefix_position = false;
    // When true, draft_step writes sampled token ids directly to the supplied
    // output buffer and the executor skips the generic logits argmax.
    bool draft_step_writes_sampled_tokens = false;
    // Optional phase used by drafters that maintain native cache/recurrent
    // state for the prefix that target verification accepted.
    CommitVerifiedPrefixFn commit_verified_prefix;
    // Generic model-owned drafter. Gemma4 MTP implements the whole draft loop
    // behind this callback.
    NativeSystemDraftNextFn draft_next;
    // Lower-level single-step drafter. The executor can chain this on GPU while
    // keeping shared response plumbing model-neutral.
    DraftStepFn draft_step;

    explicit operator bool() const noexcept {
        return max_drafts > 0 &&
               (static_cast<bool>(draft_next) ||
                static_cast<bool>(draft_step));
    }
};

// Stable references the executor needs across calls. Constructed
// once after loaded-model/workspace allocation in entry.cpp and held by
// the service.
struct Executor {
    LoadedModel& loaded_model;
    model::Qwen3Workspace& ws;
    KvCache& kv_cache;
    AttentionWorkspace& attn_ws;
    ops::CublasHandle& cublas;
    int max_workspace_tokens;
    int max_forward_requests;
    // Private physical KV page used only for CUDA-graph padding rows.
    // This page is not reported in DriverCapabilities.total_pages, so the
    // runtime never assigns it to a context.
    int graph_pad_page = -1;
    // Private recurrent-state slot used only for CUDA-graph padding rows.
    // Like graph_pad_page, it is allocated in CUDA storage but hidden from
    // runtime capabilities.
    int graph_pad_slot = -1;
    // Pre-allocated input buffers — refreshed per fire via memcpy
    // rather than re-allocated. See `persistent_inputs.hpp`.
    PersistentInputs& inputs;
    bool verbose = false;
    // Type-erased forward call. The captured weights / cfg / model
    // function are model-specific; the call site is uniform.
    ForwardFn forward_fn;
    // Optional driver-native drafter for `.system_speculation()`.
    NativeSystemDrafter system_drafter;
    // Optional CUDA-graph cache. When non-null, decode-only fires
    // attempt graph capture/replay; otherwise the forward runs directly.
    ForwardGraphCache* graph_cache = nullptr;

    // Tensor-parallel comm. Non-null on rank 0 when tp_size > 1 — the
    // executor broadcasts the per-fire inputs to TP followers
    // before invoking the forward kernels. On TP followers a parallel
    // service loop (`tp_follower_serve`) consumes those broadcasts.
    NcclComm* tp_comm = nullptr;
    // Optional in-process CPU gate keyed by the embedded launch barrier path.
    // Rank 0 notifies before posting NCCL broadcasts; followers wait here
    // before entering their NCCL receive. This prevents idle followers from
    // burning GPU cycles while rank 0 is between requests.
    std::string tp_cpu_gate_key;

    // Runtime-managed rs_cache storage. Null on models without
    // recurrent-state slots.
    Qwen3_5StateCache* rs_cache = nullptr;
    // Private rs_cache slot reserved for speculative rollback. This slot is
    // not advertised to the runtime.
    int rs_cache_scratch_slot = -1;

    // Response-view builder. Reused fire-to-fire — the builder owns the
    // concat scratch the `PieForwardResponseView` slices point into. The
    // view stays valid until the next `build()` call, which is long
    // enough for the `send_response` that immediately follows.
    pie_driver::ResponseBuilder response_builder;

};

// Run the forward pass + sampling pipeline on one forward-pass request
// and fill out `out_resp` via `executor.response_builder`. `req_id` is the
// per-fire identifier used in error logging; `handled` is the
// cumulative fire counter used as PRNG offset and logging-cadence gate.
// The caller's inproc transport hands `out_resp` to `send_response`
// immediately after this returns.
void handle_fire_batch(
    std::uint32_t req_id,
    const pie_driver::PieForwardRequestView& view,
    pie_driver::PieForwardResponseView& out_resp,
    Executor& executor,
    std::uint64_t handled);

// Pre-capture the pure-decode CUDA graph lattice for graph-safe forwards.
// Returns the number of graph execs inserted into `executor.graph_cache`.
std::size_t capture_forward_graph_lattice(Executor& executor);

// TP-follower service loop. Called only on TP ranks > 0. Mirrors
// `handle_fire_batch` minus shmem decode, sampling, and response: the
// loop blocks on `ncclBroadcast(root=0)` for each fire's header + inputs,
// runs the same `forward_fn.body` so the all-reduces inside complete
// against rank 0, then loops. `stop` is checked between fires; rank 0
// also sends an explicit shutdown header before tearing down so a
// follower waiting on a broadcast unblocks cleanly.
void tp_follower_serve(Executor& executor, std::atomic<bool>& stop);

// Send the shutdown sentinel header from rank 0 so followers exit their
// `tp_follower_serve` loop on the next broadcast.
void tp_send_shutdown(NcclComm& comm, const std::string& cpu_gate_key = {});

}  // namespace pie_cuda_driver
