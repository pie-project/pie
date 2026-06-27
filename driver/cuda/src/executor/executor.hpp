#pragma once

// Forward executor for the `fire_batch` inproc method. Owns the per-
// device persistent state (workspaces, KV cache, attention scratch,
// graph cache) and dispatches each fire through ForwardFn onto the
// active IModel implementation.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <type_traits>

#include <atomic>

#include "distributed.hpp"
#include "executor/forward_graph.hpp"
#include <pie_driver_abi/view.hpp>
#include "model/llama_like.hpp"
#include "executor/persistent_inputs.hpp"
#include <pie_driver_abi/response_builder.hpp>

namespace pie_cuda_driver {

class LoadedModel;
class KvCache;
class AttentionWorkspace;
class RecurrentStateCache;

namespace model {
struct Qwen3Weights;
struct Qwen3Workspace;
class IModel;
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

    // All metadata needed to execute one forward body call. Bundled as a
    // struct so adding a new field is a one-site addition rather than a
    // signature change touching every arch's body.
    struct ForwardInputs {
        // Per-token device buffers
        const std::int32_t*  token_ids            = nullptr;
        const std::int32_t*  positions            = nullptr;

        // CSR-style indptrs (device + host where required)
        const std::uint32_t* qo_indptr_d          = nullptr;
        const std::uint32_t* kv_page_indices_d    = nullptr;
        const std::uint32_t* kv_page_indptr_d     = nullptr;
        const std::uint32_t* kv_last_page_lens_d  = nullptr;
        const std::uint32_t* qo_indptr_h          = nullptr;
        const std::uint32_t* kv_page_indices_h    = nullptr;
        const std::uint32_t* kv_page_indptr_h     = nullptr;
        const std::uint32_t* kv_last_page_lens_h  = nullptr;

        // Shape
        int total_tokens   = 0;     // N
        int num_requests   = 0;     // R
        bool is_pure_decode = false;

        // Optional: custom attention mask (BRLE-packed)
        const std::uint8_t*  custom_mask_d        = nullptr;
        const std::int32_t*  custom_mask_indptr_d = nullptr;

        // Optional: per-request rs-cache slot info
        const std::int32_t*  slot_ids_h           = nullptr;
        const std::uint8_t*  is_fresh_h           = nullptr;
        const std::int32_t*  slot_ids_d           = nullptr;

        // Optional: logit row gather indices (for compact-logit modes)
        const std::int32_t*  logit_row_indices_d  = nullptr;
        int                  num_logit_rows       = 0;

        // False when this fire samples nothing (e.g. a multimodal image-token
        // KV-fill pass): the forward should skip the lm_head entirely rather
        // than materialize dense logits over all N rows. Defaults true so
        // ordinary text fires (which always sample ≥ 1 row) are unaffected.
        bool emit_logits = true;

        // Sampling hint: if the executor only needs argmax, body may skip
        // dense logits and write straight into the fused-argmax output.
        bool tp_greedy_argmax = false;

        // Recurrent-only commit-advance: when non-null, the forward runs ONLY
        // the linear-attn block of each linear layer (conv + recurrence,
        // write_state=true) over `total_tokens` accepted tokens, gathering each
        // layer's input from the verify-stashed hidden via these row indices
        // (into the verify token layout). Attention, MLP, non-linear layers,
        // and lm_head are skipped. Used to advance rs_cache state after a
        // frozen verify without re-running the whole backbone.
        const std::int32_t*  commit_advance_gather_d = nullptr;

        // Ph7 RS working-set fold-from-buffer. Per-request CSR of buffered-slab
        // block ids into the recurrent_state_cache buffered-activation pool
        // (host — the gather/scatter is a host-driven loop of per-slab d2d
        // memcpys, page-major: slab s holds tokens [s*page, (s+1)*page)).
        //   rs_buffer_write : after in_proj, scatter [mixed_qkv|a|b] for each
        //     request's tokens INTO its pool slabs (rs-output W10; write_state
        //     forced false — buffered, not folded).
        //   rs_buffer_fold (with commit_advance_gather_d) : the replay loads each
        //     linear layer's activations FROM the pool slabs (vs the verify
        //     stash) and folds commit_len[r]=rs_fold_lens[r] tokens into
        //     recurrent_state[slot_ids[r]].
        const std::uint32_t* rs_buffer_slot_ids_h    = nullptr;  // flattened CSR
        const std::uint32_t* rs_buffer_slot_indptr_h = nullptr;  // R+1, leading 0
        bool                 rs_buffer_write          = false;
        bool                 rs_buffer_fold           = false;

        // Multimodal (gemma4 vision, option-B pixel path) — host pointers into
        // the request view. The model encodes each image and scatters the
        // projected soft tokens into the embed output. Empty for text-only.
        const float*         image_pixels_h           = nullptr;  // f32 pixel_values
        const std::uint32_t* image_pixel_byte_indptr_h = nullptr; // n_img+1 byte offsets
        const std::uint32_t* image_patch_positions_h  = nullptr;  // 2 per patch
        const std::uint32_t* image_anchor_rows_h      = nullptr;  // n_img row offsets
        int                  num_images               = 0;

        // Qwen3-VL extras: per-image patch grids (t,h,w) and the M-RoPE 3-axis
        // position ids for the whole batch. `mrope_positions_h` is the host
        // `[total_tokens, 3]` (t,h,w) per-token positions (built by the runtime
        // for mrope models; image rows carry the grid positions, text rows
        // carry t==h==w). The Qwen3-VL forward uploads these for the M-RoPE
        // kernel. Null on gemma4 / non-mrope models.
        const std::uint32_t* image_grids_h            = nullptr;  // 3 per image (t,h,w)
        const std::uint32_t* mrope_positions_h        = nullptr;  // [total_tokens, 3]
        int                  num_mrope_positions       = 0;        // == total_tokens when set

        // Multimodal audio (gemma4_audio) — host pointers into the request
        // view. The model encodes each clip's log-mel features and scatters
        // the projected soft tokens at its anchor row. Empty for non-audio.
        const float*         audio_features_h          = nullptr;  // f32 log-mel
        const std::uint32_t* audio_feature_byte_indptr_h = nullptr; // n_clip+1 byte offsets
        const std::uint32_t* audio_anchor_rows_h      = nullptr;  // n_clip row offsets
        int                  num_clips                 = 0;
    };

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

    // The arch implementation. entry.cpp sets this once at construction;
    // the executor dispatches every per-fire call through these methods.
    model::IModel* model = nullptr;
    bool supports_fused_lmhead_argmax = false;

    // Wire `m` as the active arch and copy its capability bits onto the
    // ForwardFn caps that the executor consults each fire. Subsumes the
    // ~5-line "graph_safe/supports_*/model" boilerplate every arch was
    // repeating in entry.cpp.
    void attach_model(model::IModel* m);

    // Dispatch helpers — null-safe so an executor with no model attached
    // is harmless rather than a segfault.
    void invoke_prepare(AttentionWorkspace& aws, const PrepareInputs& in);
    void invoke_body(model::Qwen3Workspace& ws,
                     KvCache& kv,
                     AttentionWorkspace& aws,
                     ops::CublasHandle& cublas,
                     const ForwardInputs& in);
    std::uint32_t invoke_graph_layout();
    void invoke_set_logits_argmax_only(bool enabled);
    void invoke_set_fused_argmax_output(std::int32_t* ptr);
    bool invoke_fused_argmax_done();
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
    RecurrentStateCache* rs_cache = nullptr;

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
