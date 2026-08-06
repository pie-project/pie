#pragma once

// Declared-forward executor for the llama_like family — DUMB since rung 2
// (north-star-dsl.md): it walks the CLASS trace for the fire's shape
// (decode/prefill), in which the declaration already STATED every kernel
// choice — the fused decode-QKV epilogue, the fused qk-norm+rope, the
// attention kernel — as `Launch` ops carrying launcher symbols. The
// executor resolves symbols in its name→launcher registry and BINDS
// (buffers, plan caches, pad/strip staging); it never chooses between two
// kernels for semantic reasons. The peepholes and `use_*_path` booleans
// that used to re-derive those choices per fire are deleted; their
// predicates live in `forward/src/family.rs`'s class arms, evaluated once
// at model load against `PieForwardLlamaLikeCudaFacts` derived in `build`.
// Bit-parity requires the same launches, not just the same math — the
// fused kernels round differently from their unfused sequences.
//
// What remains driver-side, documented as remaining and why:
//   * pad/strip staging around KV-write/attention for padded head dims —
//     buffer routing, the mechanical binding the dumbness criterion
//     leaves with the driver;
//   * per-layer window resolution and plan-cache binding — parameters OF
//     the stated kernel, not choices between kernels;
//   * the fused decode-QKV epilogue's `has_write_desc ? w_page_d :
//     nullptr` ternary — ARG binding of one stated kernel, not a kernel
//     choice. (The KV-write mechanism itself — explicit vs page-derived,
//     two different kernels — IS expressed now: the declaration's
//     `HasWriteDesc` Guard states both arms, the first live Guard;
//     rung 4a.)
// Everything the trace cannot express yet (mixed-hook fires, TP, vision,
// quantized projections, non-standard rope, qkv bias) falls back to
// `llama_like_forward_paged` — the caller gates, `build` refuses.
// Custom masks ARE expressed — since A1 (the class-collapse amendment)
// as the `HasCustomMask` guard arm INSIDE the decode/prefill traces: the
// mask arm carries the general QKV sequence (with the nested
// HasWriteDesc guard — the walk keeps a skip STACK) and states the
// custom-mask prefill dispatch; the mask data rides as runtime args. Post-norm placement (olmo2) and the global qk-norm convention
// ARE in scope: the trace states both as facts (the matmul(beta=0) →
// rmsnorm → residual_add triplet; plain row Rmsnorm on q/k), and this
// executor launches the hand-written post-norm / `rmsnorm_qk`-global
// kernels for them. Padded head_dim (Phi-3-mini's 96 → 128) is IN scope: the
// pad/strip staging around KV-write/attention is emitter knowledge, not
// trace vocabulary — the trace speaks the logical head_dim throughout.
//
// `dyn` ops are OUT of scope and refused loudly. The forward crate's MoE
// vocabulary (`TopK`, selector-carrying `Matmul`s over `expert.{e}.*`
// weight templates, `WeightedSum`, `SigmoidGateAdd` — the
// `pie_forward_trace_qwen3_5_moe_mlp` fragment) is toolchain-side only for
// now: this executor's op-kind switch ends in a default arm that throws
// "op kind N has no emission rule", so a dyn trace can never half-emit.
// (Tested on the Metal side, whose DAG builder shares the discipline:
// driver/metal/tests/llama_like_declared_dag_test.cpp traces the MoE
// fragment through the ABI and pins the refusal.) Emitting the expert
// axis — grouped GEMM over gathered tokens, the kernels qwen3_5_moe
// already owns — is a later, much larger lift, not a switch arm to add
// casually here.
//
// The GDN vocabulary (`SplitGdn`, `CausalConv1d`, `GdnPrep`, `GatedDelta`,
// `RmsnormGated` — the `pie_forward_trace_qwen3_5_gdn` fragment) is refused
// by the same default arm, and for a stronger reason than the MoE kinds:
// `CausalConv1d`/`GatedDelta` address PER-REQUEST conv/recurrent state
// (`RecurrentStateCache` slots, slot_ids indirection — the reason RS fires
// are forced solo today), so emitting them means wiring the state cache and
// its slot plumbing through this walk, not adding launch arms. The Metal
// test pins this refusal too (same file as the MoE one).
//
// The full-attention/hybrid vocabulary (`SplitQGate`, `SigmoidGateMul` —
// the `pie_forward_trace_qwen3_5_full_attn` fragment and the
// `pie_forward_trace_qwen3_5_hybrid` model) is refused by the same default
// arm. Two of qwen3.5's four gated-attention pieces ride as APPENDED PARAMS
// on existing kinds rather than new kinds — `Rope.param1` (partial rotary
// width) and `RmsnormPerHead.param1` (Gemma fold) — and this executor
// neither reads nor emits them; that is safe because no trace can reach
// either op past its refusals: every qwen3_5 trace opens each layer with a
// GEMMA `Rmsnorm` (refused above by the variant check) and, were the fold
// Plain, would still hit an unknown qwen3_5-only weight name or the
// `SplitQGate` default arm before its Rope. Emitting the gated attention
// (split-q-gate, partial rope, sigmoid output gate — kernels qwen3_5
// already owns) is a smaller lift than MoE/GDN but still a rung of its
// own, not a switch arm to add casually here. The Metal test pins the
// full-attention refusal too (same file).
//
// Explicit KV-write descriptors ARE handled (the hand-written
// `has_write_desc` branch, verbatim): every pure-decode fire that replays a
// forward graph carries them — `forward_graph_replay_eligible` REQUIRES
// `has_write_desc` (batch/forward.cpp) because pure-decode captures record
// the w_page/w_off write path — so excluding them would exclude decode
// entirely and reduce Stage 3's parity claim to the prefill step.

#include <cstdint>
#include <string>

#include "model/llama_like/llama_like.hpp"
#include "pie_forward/plan.hpp"

namespace pie_cuda_driver::batch {
class SupergraphBuilder;
}  // namespace pie_cuda_driver::batch

namespace pie_cuda_driver::model {

// The traced forms plus what the executor needs to know about how they
// were traced. Built once at model construction (the facts are load-time
// facts; re-tracing per fire would contradict the trace's whole premise).
//
// Rung 2 (north-star-dsl.md): `decode`/`prefill` are the CLASS traces —
// the declaration run with this deployment's derived CUDA facts and a
// fire class, so every kernel choice is STATED as a `Launch` op. The
// executor walks whichever class the fire is and resolves launcher
// symbols in its name→launcher registry; the peepholes and `use_*_path`
// booleans it used to derive are deleted. `plan` is the SEMANTIC trace,
// kept as the parity reference and for consumers that predate lowering
// (site summaries, the Metal emitter).
struct LlamaLikeDeclaredPlan {
    pie_forward::ForwardPlan plan;
    pie_forward::ForwardPlan decode;
    pie_forward::ForwardPlan prefill;
    // (The masked and hooked traces are gone with their classes — A1/A2,
    // the class-collapse amendment: a masked fire takes decode/prefill's
    // HasCustomMask guard arm; an all-hooked fire takes their
    // HasStageHooks arm — the general body, the two per-layer hook
    // sites and the WantsAttnScore-guarded attention, all region ops.
    // Mixed fires (0 < fast_rows < R) stay hand-written until the Peel
    // op.)
    // What the class traces were taken from, in the format the generated
    // .inc embeds (`emit_cuda::facts_digest`); rung 3's dispatch runs the
    // static form only on exact match.
    std::string facts_digest;
    // The binding fact the traces were taken against; the per-fire gate
    // re-checks it against the workspace (`ws.qkv_fused` may be empty even
    // when the weight is bound) and falls back on mismatch.
    bool fused_qkv = false;

    explicit operator bool() const noexcept {
        return static_cast<bool>(plan) && static_cast<bool>(decode) &&
               static_cast<bool>(prefill);
    }
};

// Trace the family against this deployment's facts — the semantic form
// and both class traces (the CUDA facts derive here: env gates, kernel
// support, cache format, binding; the same terms the executor booleans
// used to compute per fire, computed once and stated). Every Launch op's
// kernel symbol is validated against the executor's registry at build, so
// a trace/executor drift fails at model load, not mid-fire. Returns an
// empty plan (operator bool false) when the configuration is outside the
// trace's vocabulary — the caller then keeps the hand-written path,
// silently: an unrepresentable config is a fallback, not an error.
LlamaLikeDeclaredPlan build_llama_like_declared_plan(
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const Qwen3Weights& w,
    const KvCache& cache);

// Do this fire's rows take the STEP's depth bands?
//
// One function, called by the executor (where it decides whether the
// band arrays are read at all) and by the model's eligibility gate
// (where it decides whether a missing band plan is a reason to fall
// back). It exists because those two were DUPLICATES, and the duplicate
// went stale: the gate's copy carried the argument "a prefill trace does
// not state the depth axis, so the bands are never read", which the very
// next commit invalidated by teaching the Prefill class to state it.
// Qwen2.5-1.5B then threw 5,080 times. A mirror is not a proof.
//
// The rule itself is the hand-written path's, term for term
// (`llama_like.cpp`'s `bands_runnable`): bands describe a PURE-DECODE
// fire's rows. `derive_depth_bands` refuses a region table carrying any
// multi-token region, so a step's bands never describe a prefill fire's
// rows even when the step stamped them for its decode fires — and the
// hand path ignores them there too, which is what makes ignoring them
// the parity-preserving answer rather than a demotion.
bool llama_like_bands_apply(const LlamaLikeDeclaredPlan& declared,
                            const LlamaLikePlanState& plan_state,
                            bool is_pure_decode);

// Execute the traced form. Same argument surface as
// `llama_like_forward_paged` minus the inputs the eligibility gate already
// excluded (hooks, custom mask, write descriptor, vision). Reads the SAME
// `plan_state` the prepare hook filled — prepare() is unchanged and runs
// for both paths.
void llama_like_forward_declared(
    const LlamaLikeDeclaredPlan& declared,
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    int runtime_window_left,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    // The fire's attached stage programs; null on unhooked fires. Sites
    // and the score guard live in the shape traces (A3); the
    // page-mask/score sidebands are this executor's bracket mechanics.
    const StageHooks* stage_hooks,
    // The fire's resolved lora configuration (null = none). Usable lanes
    // take the HasLora guard arms — the general sequence plus the
    // `pie_lora_qkv_correction` pseudo-symbol (LoraFireState::apply
    // behind the registry, staged once per fire).
    const LoraTable* lora,
    // The Peel device window word ({tail_start, tail_len} in device
    // memory), non-null ONLY on hook-graph captures: the walk then emits
    // BOTH Peel regions unconditionally through the devwin kernel forms,
    // so the captured exec replays across row splits. Null (eager, and
    // every non-hook path) keeps the host windows — no wasted threads
    // where no capture needs stability.
    const std::uint32_t* peel_window_d = nullptr,
    // NS-2 (the spatial mask fire): when != UINT32_MAX the attention
    // splits at this wire row — decode dispatch over [0, split), the
    // custom-mask kernel over the REBASED suffix CSRs below. Must agree
    // with plan_state.spatial_mask_split (drift throws).
    std::uint32_t unmasked_prefix_rows = 0xffffffffu,
    const std::uint32_t* mask_suffix_qo_indptr_d = nullptr,
    const std::uint32_t* mask_suffix_kv_page_indptr_d = nullptr,
    // STRUCTURAL S-3/S-4: the depth axis — the suffix's uniform k and
    // the union's request split (UINT32_MAX = unset). Honoured only when
    // the fire's plan STATES the axis (ForwardPlan::depth_window); the
    // router keeps unsupported shapes on the hand-written body.
    std::uint32_t declared_max_layers = 0xffffffffu,
    std::uint32_t declared_full_depth_rows = 0xffffffffu);

// The unionized supergraph (S3): whether this deployment's digest has an
// emitted `..._supergraph_build`, and the digest-dispatched build call
// itself (false = no emitted build; the caller must not have promised a
// supergraph capture). Same argument surface as the declared walk minus
// the attachments the union excludes (hooks, lora), plus the builder.
bool llama_like_supergraph_supported(const LlamaLikeDeclaredPlan& declared);

bool llama_like_forward_supergraph_build(
    const LlamaLikeDeclaredPlan& declared,
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    batch::SupergraphBuilder& sg);

}  // namespace pie_cuda_driver::model
