#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::model {

struct AttentionObservation;
struct AttentionScores;
struct AttentionMaskSink;
class HookSidebandArena;

enum class StageHookPoint : std::uint8_t {
    OnAttnProj = 1,
    OnAttn = 2,
};

// What crosses the model -> dispatch boundary alongside the hook itself.
//
// These used to be three thread-locals published around the model body
// (`active_attention_observation`, `active_attention_scores`,
// `active_attention_mask_sink`). They are the same three facts, now carried
// BY the call: the fire's KV geometry, the scores the layer just captured,
// and the destination a page-mask sink may write. A hook that is handed its
// sideband cannot read another fire's, which is the property the
// thread-locals only approximated — and per-lane hooks (a lane keeping the
// fused path while its neighbour taps scores) are only expressible when the
// binding is an argument rather than ambient state.
struct StageHookSideband {
    // The fire's KV geometry, for the duration of the body. Filled in by
    // `invoke_stage_hook` from the hooks' own observation when the call site
    // leaves it null, so most sites never mention it.
    const AttentionObservation* observation = nullptr;

    // The layer's captured scores (`OnAttn` on a capture-capable branch);
    // null when nothing was captured, and the PTIR side then fails loudly at
    // the `attn_score` intrinsic instead of reading a stale row.
    const AttentionScores* scores = nullptr;

    // Where a program's `attn_page_mask` sink writes (`OnAttnProj` when the
    // fire wants one). Owned by the model body; the dispatch only fills it.
    AttentionMaskSink* mask_sink = nullptr;
};

struct StageHooks {
    void* context = nullptr;

    // The fire's PTIR programs read `AttnScore` at `OnAttn`. Carried on the
    // hooks rather than as separate state because it is a property of the
    // programs the dispatch is about to run, and the dispatch is what
    // constructs the hooks -- so the two can never disagree about which fire
    // they describe. A model family that does not check it simply captures
    // nothing, and the PTIR side then fails LOUDLY at the hook rather than
    // reading a buffer nobody wrote.
    bool wants_attn_score = false;

    // How many query rows at the TAIL of a prefill chunk are observed, when
    // the fire is a prefill and `wants_attn_score` is set. Decode ignores it:
    // a decode step has exactly one query row, so the window is 1 by
    // construction.
    //
    // This is an observation parameter, not a policy one -- the same way
    // "averaged over heads" is. It lives on the hooks because that is where
    // "what this fire's programs need" is expressed; a future PTIR-declared
    // window would be a one-line change at the construction site rather than a
    // new plumbing path. `attn_score_window()` in attn_score.cu supplies the
    // default (SnapKV's 32) and the `PIE_ATTN_SCORE_WINDOW` override.
    std::uint32_t attn_score_window = 0;

    // The fire's PTIR programs write `attn_page_mask` at `OnAttnProj`, and the
    // model is expected to honour it by compacting its page table before the
    // attention call.
    bool wants_page_mask = false;

    // How many LEADING request rows belong to no attention-stage program.
    // This is what turns "any hook, anywhere, disables the fast path" into a
    // row count: a model body may run its hook-free fast path over rows
    // [0, n) and the hook-visible path over the rest, in the same fire. 0
    // means no row is provably hook-free (today's fire-wide behaviour). The
    // count is in wire request rows — a body whose request set is not the
    // wire rows in wire order must clamp it to what it can prove, and the
    // slow path is always correct. See
    // `Dispatch::launch_hook_free_prefix_rows`.
    std::uint32_t hook_free_prefix_rows = 0;

    // The grow-only device arena the fire's sideband captures draw their
    // buffers from (`hook_sideband_arena.hpp`). Engine-lifetime, owned beside
    // `model::Workspace` in context.cpp and carried on the hooks because the
    // sidebands only exist when hooks fire — the constructors that used to
    // cudaMallocAsync per layer (`ScoreBuffers`, `FirePageMask`) acquire from
    // it instead. Null means the launch path forgot to wire it; the captures
    // then refuse loudly rather than silently reintroducing the churn.
    HookSidebandArena* sideband_arena = nullptr;

    // The fire's KV geometry. Set by `ForwardFn::invoke_body` -- the single
    // choke point every family passes through -- on its own copy of the
    // frame's hooks, so it is valid exactly while the body runs and no model
    // file has to construct it.
    const AttentionObservation* observation = nullptr;

    void (*execute)(
        void* context,
        StageHookPoint point,
        const void* query_data,
        std::uint32_t query_rows,
        std::uint32_t query_columns,
        std::uint32_t layer,
        cudaStream_t stream,
        bool query_is_f32,
        const StageHookSideband& sideband) = nullptr;

    // Stage 6 increment 4 + eager unification — the fire-level prepare
    // pass. The batch engine calls this for EVERY pure-decode hook fire,
    // eager and graph alike, BEFORE the body runs (and before capture, on a
    // capturing fire): the dispatch hoists every attention-phase PREPARE —
    // host metadata build, channel-cursor reads, stable-buffer uploads,
    // score-sideband sizing — to fire level, leaving the in-body `execute`
    // calls a pure launch replay against prepared state; graph mode merely
    // captures what the eager body would have launched anyway. Returns a
    // nonzero fingerprint of every address and grid a captured body would
    // bake (the batch engine recaptures on change), or 0 when this fire
    // cannot run prepared and must take the legacy interleaved eager body.
    // A 0 return has NO side effects on the launch.
    // Null when the frame did not wire the seam (non-staged paths).
    std::uint64_t (*prepare_replay)(
        void* context,
        cudaStream_t stream) = nullptr;

    // Companion to `prepare_replay`: called by the batch engine right after
    // CAPTURING a hook fire's body and right after every prepared-EAGER
    // body, to assert the body consumed every prepared attention invocation
    // (i.e. the model really invoked its hooks at every layer — the prepare
    // pass pre-credits the coverage counter, so this is the only place the
    // omission is visible). Throws on violation.
    void (*verify_replay_capture)(void* context) = nullptr;
};

// `hooks` is the fire's hook set, threaded down from
// `ForwardInputs::stage_hooks` as a parameter -- there is deliberately no
// ambient fallback. A null `hooks` (or a hook set with no `execute`) is the
// "no program attached" case and costs one branch.
inline void invoke_stage_hook(
    const StageHooks* hooks,
    StageHookPoint point,
    const void* query_data,
    std::uint32_t query_rows,
    std::uint32_t query_columns,
    std::uint32_t layer,
    cudaStream_t stream,
    bool query_is_f32 = false,
    StageHookSideband sideband = {}) {
    if (hooks == nullptr || hooks->execute == nullptr) {
        return;
    }
    if (sideband.observation == nullptr) {
        sideband.observation = hooks->observation;
    }
    hooks->execute(
        hooks->context, point, query_data, query_rows, query_columns, layer,
        stream, query_is_f32, sideband);
}


// ── Upstream ambient-hook compatibility (merge 2026-08-04) ────────────
// github/dev's model bodies (qwen3_5 family) invoke hooks through an
// ambient pointer instead of tart's explicit threading + sideband. Both
// styles coexist: tart bodies keep the explicit `const StageHooks*`
// overload above (scores/page-mask sidebands need it); upstream bodies
// compile against this point-first overload. Until a caller installs
// ScopedStageHooks, the ambient is null and upstream-style invocations
// are no-ops — qwen3_5 hooks are DORMANT this era (re-port pending).
inline thread_local const StageHooks* active_stage_hooks = nullptr;

class ScopedStageHooks {
  public:
    explicit ScopedStageHooks(const StageHooks* hooks)
        : previous_(active_stage_hooks) {
        active_stage_hooks = hooks;
    }
    ~ScopedStageHooks() { active_stage_hooks = previous_; }

    ScopedStageHooks(const ScopedStageHooks&) = delete;
    ScopedStageHooks& operator=(const ScopedStageHooks&) = delete;

  private:
    const StageHooks* previous_ = nullptr;
};

inline void invoke_stage_hook(
    StageHookPoint point,
    const void* query_data,
    std::uint32_t query_rows,
    std::uint32_t query_columns,
    std::uint32_t layer,
    cudaStream_t stream,
    bool query_is_f32 = false) {
    if (active_stage_hooks == nullptr ||
        active_stage_hooks->execute == nullptr) {
        return;
    }
    active_stage_hooks->execute(
        active_stage_hooks->context, point, query_data, query_rows,
        query_columns, layer, stream, query_is_f32,
        StageHookSideband{active_stage_hooks->observation});
}

}  // namespace pie_cuda_driver::model
