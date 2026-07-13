#pragma once

// Batch engine for direct PTIR launches. Owns the per-device persistent
// state and dispatches each fire through ForwardFn onto the active IModel.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

#include <atomic>
#include <chrono>
#include <iostream>
#include <map>

#include "cuda_check.hpp"
#include "distributed.hpp"
#include "batch/forward_graph.hpp"
#include "pie_native/launch_view.hpp"
#include "model/llama_like/llama_like.hpp"
#include "batch/persistent_inputs.hpp"
#include <memory>
#include <pie_driver_abi.h>

namespace pie_cuda_driver {

struct PrecomputedEmbeddingInputs {
    const std::uint8_t* rows_h = nullptr;
    const std::uint32_t* byte_indptr_h = nullptr;
    const std::uint32_t* shapes_h = nullptr;
    const std::uint8_t* dtypes_h = nullptr;
    const std::uint32_t* anchor_rows_h = nullptr;
    int num_blocks = 0;
};

class LoadedModel;
class KvCache;
class AttentionWorkspace;
class RecurrentStateCache;

namespace pipeline {
class Dispatch;
}  // namespace pipeline

namespace model {
struct Qwen3Weights;
struct Workspace;
class IModel;
struct StageHooks;
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
//                  the batch engine then disables graph capture for
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
// context.cpp builds both closures from the per-arch weights + cfg.
struct ForwardFn {
    // Whether the batch engine may capture this forward into a CUDA
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
    bool supports_runtime_window = false;

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
        // -2: no runtime override; -1: full causal; >=0: sliding window.
        int                  runtime_window_left = -2;

        // Optional: explicit KV-write descriptor (device-geometry WSlot/WOff,
        // B2). When `has_write_desc`, the per-layer KV append lands each lane's
        // new-token K/V at the EXPLICIT (physical page id `w_page_d[lane]`,
        // offset `w_off_d[lane]`) target via `launch_write_kv_explicit_bf16`,
        // instead of re-deriving the position from the page-table +
        // last_page_len. Required for beam fork/freeze correctness (a frozen
        // fork's cell must not be overwritten by the standard derivation).
        const std::uint32_t* w_page_d             = nullptr;
        const std::uint32_t* w_off_d              = nullptr;
        bool                 has_write_desc       = false;

        // Optional: per-request rs-cache slot info
        const std::int32_t*  slot_ids_h           = nullptr;
        const std::uint8_t*  is_fresh_h            = nullptr;
        const std::int32_t*  slot_ids_d           = nullptr;
        const std::uint8_t*  rs_slot_flags_h      = nullptr;

        // Optional: logit row gather indices (for compact-logit modes)
        const std::int32_t*  logit_row_indices_d  = nullptr;
        int                  num_logit_rows       = 0;

        // False when this fire samples nothing (e.g. a multimodal image-token
        // KV-fill pass): the forward should skip the lm_head entirely rather
        // than materialize dense logits over all N rows. Defaults true so
        // ordinary text fires (which always sample ≥ 1 row) are unaffected.
        bool emit_logits = true;

        // Internal TP/graph hint; direct PTIR launches leave this disabled so
        // the selected logits remain available to the stage program.
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
        const std::uint32_t* rs_fold_lens_h           = nullptr;  // R
        const std::int32_t*  rs_fold_lens_d           = nullptr;  // R
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
        PrecomputedEmbeddingInputs precomputed_embeddings;

        // Launch-scoped PTIR anatomical hooks. Null for ordinary forwards and
        // TP followers; direct staged launches install this only on rank 0.
        const model::StageHooks* stage_hooks = nullptr;
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
        int runtime_window_left = -2;
    };

    // The arch implementation. context.cpp sets this once at construction;
    // the batch engine dispatches every per-fire call through these methods.
    model::IModel* model = nullptr;
    bool supports_fused_lmhead_argmax = false;

    // Wire `m` as the active arch and copy its capability bits onto the
    // ForwardFn caps that the batch engine consults each fire. Subsumes the
    // ~5-line "graph_safe/supports_*/model" boilerplate every arch was
    // repeating in context.cpp.
    void attach_model(model::IModel* m);

    // Dispatch helpers — null-safe so a batch engine with no model attached
    // is harmless rather than a segfault.
    void invoke_prepare(AttentionWorkspace& aws, const PrepareInputs& in);
    void invoke_body(model::Workspace& ws,
                     KvCache& kv,
                     AttentionWorkspace& aws,
                     ops::CublasHandle& cublas,
                     const ForwardInputs& in);
    std::uint32_t invoke_graph_layout();
    void invoke_set_logits_argmax_only(bool enabled);
    void invoke_set_fused_argmax_output(std::int32_t* ptr);
    bool invoke_fused_argmax_done();
};

struct NativeSystemCommitInputs {
    model::Workspace& target_ws;
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
        model::Workspace&,
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
    // output buffer and the batch engine skips the generic logits argmax.
    bool draft_step_writes_sampled_tokens = false;
    // Optional phase used by drafters that maintain native cache/recurrent
    // state for the prefix that target verification accepted.
    CommitVerifiedPrefixFn commit_verified_prefix;
    // Lower-level single-step drafter. The batch engine can chain this on GPU
    // while keeping the implementation model-neutral.
    DraftStepFn draft_step;

    explicit operator bool() const noexcept {
        return max_drafts > 0 && static_cast<bool>(draft_step);
    }
};

// Stable references the batch engine needs across calls. Constructed
// once after loaded-model/workspace allocation in context.cpp and held by
// the service.
struct BatchEngine {
    BatchEngine(LoadedModel& loaded_model,
                model::Workspace& ws,
                KvCache& kv_cache,
                AttentionWorkspace& attn_ws,
                ops::CublasHandle& cublas,
                int max_workspace_tokens,
                int max_forward_requests,
                int graph_pad_page,
                int graph_pad_slot,
                PersistentInputs& inputs,
                bool verbose,
                ForwardFn forward_fn,
                NativeSystemDrafter system_drafter,
                ForwardGraphCache* graph_cache = nullptr,
                NcclComm* tp_comm = nullptr,
                std::string tp_cpu_gate_key = {},
                RecurrentStateCache* rs_cache = nullptr)
        : loaded_model(loaded_model),
          ws(ws),
          kv_cache(kv_cache),
          attn_ws(attn_ws),
          cublas(cublas),
          max_workspace_tokens(max_workspace_tokens),
          max_forward_requests(max_forward_requests),
          graph_pad_page(graph_pad_page),
          graph_pad_slot(graph_pad_slot),
          inputs(inputs),
          verbose(verbose),
          forward_fn(std::move(forward_fn)),
          system_drafter(std::move(system_drafter)),
          graph_cache(graph_cache),
          tp_comm(tp_comm),
          tp_cpu_gate_key(std::move(tp_cpu_gate_key)),
          rs_cache(rs_cache) {}

    LoadedModel& loaded_model;
    model::Workspace& ws;
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

    // Tensor-parallel comm. Non-null on all TP ranks when tp_size > 1. Rank 0
    // broadcasts the per-fire inputs to TP followers before invoking the
    // forward kernels; TP followers consume those broadcasts from a background
    // `tp_follower_serve` loop.
    NcclComm* tp_comm = nullptr;
    // Optional in-process CPU gate keyed by the embedded launch barrier path.
    // Rank 0 notifies before posting NCCL broadcasts; followers wait here
    // before entering their NCCL receive. This prevents idle followers from
    // burning GPU cycles while rank 0 is between requests.
    std::string tp_cpu_gate_key;

    // Runtime-managed rs_cache storage. Null on models without
    // recurrent-state slots.
    RecurrentStateCache* rs_cache = nullptr;

    // Pipeline stage-program dispatch. Non-owning: constructed exactly once
    // by `pipeline::Registry` (the single owner of program/instance/channel
    // state) and handed to the batch engine after construction. Opaque,
    // CUDA-free handle (the impl + the tier-0 device kernels live behind
    // `dispatch.cu`, so this header — included by host `.cpp` TUs — never
    // pulls `__global__` code). Owns the C3 hash-decode cache + the
    // persistent per-instance execution contexts keyed by the bound
    // instance id. Never null once the driver has finished composing.
    pipeline::Dispatch* dispatch = nullptr;
};

// Pre-capture the pure-decode CUDA graph lattice for graph-safe forwards.
// Returns the number of graph execs inserted into `engine.graph_cache`. Only
// called when `tp_size > 1` (rank-0-only replay is unreachable — see
// `batch/forward.cpp`); TP followers replay this lattice in
// `tp_follower_serve`.
std::size_t capture_forward_graph_lattice(BatchEngine& engine);

// Capture one forward-body invocation into a CUDA graph exec (decode-only
// shape). Shared by `capture_forward_graph_lattice` (batch/forward.cpp) and
// `tp_follower_serve` (batch/tp.cpp) so TP followers replay the identical
// kernel sequence rank 0's own graph engine would.
cudaGraphExec_t capture_forward_graph_exec(
    BatchEngine& engine,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int N,
    int R,
    bool is_pure_decode,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    bool single_gpu_greedy_argmax,
    bool tp_greedy_argmax,
    const std::uint32_t* w_page_d = nullptr,
    const std::uint32_t* w_off_d = nullptr,
    bool has_write_desc = false,
    int runtime_window_left = -2);

// Env-gated (`PIE_STEP_PROFILE`) forward-body wall-clock timer. Declared here
// (not private to batch/forward.cpp) because `handle_fire_batch` in
// batch/compose.cpp times the `run_forward_dispatch` call it makes. Backed
// by `step_profile_take()` (batch/forward.cpp) so the sampling budget
// (`PIE_STEP_PROFILE_LIMIT`) is shared across every construction site.
bool step_profile_take();

struct StepProfileTimer {
    bool enabled = false;
    const char* label = "";
    int tokens = 0;
    int requests = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    StepProfileTimer(
        const char* label_, cudaStream_t stream, int tokens_, int requests_)
        : enabled(step_profile_take()),
          label(label_),
          tokens(tokens_),
          requests(requests_)
    {
        if (!enabled) return;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    ~StepProfileTimer() {
        if (start) CUDA_CHECK(cudaEventDestroy(start));
        if (stop) CUDA_CHECK(cudaEventDestroy(stop));
    }

    void finish(cudaStream_t stream) {
        if (!enabled) return;
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        std::cerr << "[pie-step-profile] label=" << label
                  << " tokens=" << tokens
                  << " requests=" << requests
                  << " ms=" << ms << "\n";
        enabled = false;
    }
};

// Wire-shaped forward input spans for one fire, resolved in
// `batch/compose.cpp` (either straight from the wire view or from the
// composed device-geometry batch) and handed to `run_forward_dispatch`.
struct ForwardInputViews {
    std::span<const std::uint32_t> tokens;
    std::span<const std::uint32_t> positions;
    std::span<const std::uint32_t> qo_indptr;
    std::span<const std::uint32_t> kv_page_indices;
    std::span<const std::uint32_t> kv_page_indptr;
    std::span<const std::uint32_t> kv_last_page_lens;
    int total_tokens = 0;
    int num_requests = 0;
};

// Wraps the fire's already-resolved spans into `ForwardInputViews`. Direct
// PTIR launches always dispatch at the exact wire/composed geometry — there
// is no separate "graph-padded" request count at this callsite — so this is
// a thin, allocation-free adapter rather than a padding/copy path.
ForwardInputViews make_forward_input_views(
    std::span<const std::uint32_t> tokens,
    std::span<const std::uint32_t> positions,
    std::span<const std::uint32_t> qo_indptr,
    std::span<const std::uint32_t> kv_page_indices,
    std::span<const std::uint32_t> kv_page_indptr,
    std::span<const std::uint32_t> kv_last_page_lens,
    int num_requests);

// Inputs to the forward-dispatch phase. Built once at the call site
// and passed by-ref so the dispatcher can pick between graph replay
// and a direct `forward_fn.body` call without a 20-arg signature.
//
// Direct PTIR launches keep the forward geometry exact. A launch with
// anatomical stages takes the direct body path so its per-layer hooks execute
// on every invocation; boundary-only programs may still use a compatible graph
// variant keyed by their compiled program set.
struct ForwardDispatchInputs {
    int forward_R = 0;
    int forward_N = 0;
    int num_sampling = 0;
    bool is_pure_decode = false;
    bool have_custom_mask = false;
    int structured_window_left = -2;
    std::uint64_t program_set_hash = 0;
    // Explicit KV-write descriptor present (device-geometry WSlot/WOff, B2).
    // When set, the forward routes the per-layer KV append through the explicit
    // (physical page, offset) kernel from pi.w_page/pi.w_off.
    bool has_write_desc = false;
    bool use_slots = false;
    const std::uint32_t* h_qo_forward = nullptr;
    const std::uint32_t* h_kvpi_forward = nullptr;
    const std::uint32_t* h_kvpp_forward = nullptr;
    const std::uint32_t* h_kvlpl_forward = nullptr;
    const std::int32_t*  slot_ids_h_data = nullptr;
    const std::uint8_t*  is_fresh_h_data = nullptr;
    // Ph7 RS rs-output (W10): when rs_buffer_write, the linear layers scatter
    // their in-proj [mixed_qkv|a|b] page-major into the buffered-activation pool
    // at these per-request CSR slabs (write_state forced false). FOLD passes use
    // the separate fold-replay dispatch instead (not this path).
    const std::uint32_t* rs_buffer_slot_ids_h = nullptr;
    const std::uint32_t* rs_buffer_slot_indptr_h = nullptr;
    const std::uint32_t* rs_fold_lens_h = nullptr;
    const std::int32_t*  rs_fold_lens_d = nullptr;
    bool                 rs_buffer_write = false;
    bool                 rs_buffer_fold = false;
    // Multimodal (gemma4 vision): image side-channel, set from the view.
    const float*         image_pixels_h = nullptr;
    const std::uint32_t* image_pixel_byte_indptr_h = nullptr;
    const std::uint32_t* image_patch_positions_h = nullptr;
    const std::uint32_t* image_anchor_rows_h = nullptr;
    int                  num_images = 0;
    // Qwen3-VL: per-image (t,h,w) grids and the assembled per-token M-RoPE
    // 3-axis positions for the whole batch.
    const std::uint32_t* image_grids_h = nullptr;
    const std::uint32_t* mrope_positions_h = nullptr;
    int                  num_mrope_positions = 0;
    // Multimodal (gemma4 audio): log-mel side-channel, set from the view.
    const float*         audio_features_h = nullptr;
    const std::uint32_t* audio_feature_byte_indptr_h = nullptr;
    const std::uint32_t* audio_anchor_rows_h = nullptr;
    int                  num_clips = 0;
    PrecomputedEmbeddingInputs precomputed_embeddings;
    const model::StageHooks* stage_hooks = nullptr;
};

// Run the per-fire forward body directly against `forward_fn.body`. See
// `ForwardDispatchInputs` for why this is not a graph-replay dispatcher.
void run_forward_dispatch(BatchEngine& engine, const ForwardDispatchInputs& in);

}  // namespace pie_cuda_driver
