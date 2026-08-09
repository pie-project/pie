//! `qwen3_5`'s per-backend binding facts.
//!
//! The SHAPE moved to `../spec.rs` (ungated: a row is written in it, and
//! a row must exist under every aspect). What a deployment BOUND -- an
//! env-gated join, a CUTLASS workspace's row ceiling, the recurrent
//! store's dtype -- is known only when that backend's aspect is compiled,
//! so it stays here.

use model_compiler::dsl::WeightRepr;
use serde::{Deserialize, Serialize};

/// The shape, re-exported so a declaration reaches its facts and the
/// words they are stated in from one place.
pub use super::super::spec::{
    Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts,
};

/// CUDA backend facts for a LOWERED qwen3_5 hybrid trace
/// (`family::qwen3_5_hybrid_cuda`; north-star-dsl.md rung 4c).
///
/// Everything here is load-time, [`LlamaLikeCudaFacts`]-style: env
/// defaults and kernel-eligibility predicates the hand-written
/// `linear_attn_layer_body` / `declared_forward.cpp` derive per fire
/// today, hoisted to where a fact belongs. The N-thresholds are VALUES
/// carried into [`model_compiler::trace::GuardPred`]s — the one branch kind a
/// lowered trace keeps — because only N varies per fire; the predicates
/// AROUND them (env gates, head geometry) resolve here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35CudaFacts {
    /// The recurrent-state store dtype is bf16 (vs fp32) — the
    /// `state_bf16` parameter every GDN recurrence launcher family
    /// suffixes (`kernels::ssm::recurrent_gated_delta_step_batched[_state_bf16]`
    /// and the chunked prefill families).
    pub state_bf16: bool,
    /// The warp-tiled prefill arm EXISTS at all: `K_d <= 256` && the
    /// state-persist env gate
    /// (`qwen35_gdn_warp_tiled_state_persist_enabled`,
    /// `PIE_QWEN35_GDN_WARP_TILED_STATE_PERSIST`). The hand-written
    /// predicate reads `!write_state || state_persist_enabled()` — but
    /// for the normal-fire classes (Decode/Prefill) `write_state` is
    /// always true, so the env term is the WHOLE eligibility here; the
    /// verify-frozen service classes where `write_state` is false are
    /// rung 4c-iv's, not this struct's. (`commit_lens == nullptr`, the
    /// other hand-written term, is likewise a CLASS — CommitAdvance —
    /// not a fact.)
    pub warp_tiled: bool,
    /// `qwen35_gdn_warp_tiled_max_tokens()`
    /// (`PIE_QWEN35_GDN_WARP_TILED_MAX_TOKENS`, default 64) — an
    /// env-tunable driver constant, resolved into the trace as the
    /// warp-tiled arm's `TokensLE` payload the way every fact resolves.
    pub warp_tiled_max: u32,
    /// `qwen35_gdn_cached_prefill_max_tokens()`
    /// (`PIE_QWEN35_GDN_CACHED_PREFILL_MAX_TOKENS`, default 0 = the
    /// cached family off) — the cached arm's `TokensLE` payload.
    pub cached_max: u32,
    /// The deployment configures the verify hidden stash
    /// (`RecurrentStateCache::configure_verify_hidden_stash`): the
    /// engine owns that configuration, so the fact is stated here and the
    /// driver cross-checks its own derivation per fire rather than
    /// choosing. With the stash live, the CommitAdvance class
    /// replays each linear layer's in-proj outputs from the stash
    /// (`cuda::verify_stash_load`) and skips the GEMMs; without it, the
    /// commit pass re-runs the in-projections against whatever the
    /// workspace holds. Serde-defaulted so pre-4c-iv facts JSON reads
    /// back unchanged (the append-only discipline).
    ///
    /// NOTHING READS THIS TODAY, and `tests/facts_are_read.rs` says so
    /// out loud. It used to be stated "like [`Self::state_bf16`]", but
    /// that comparison broke: `state_bf16` is read by the trace and
    /// travels to the driver as a lowering argument, while this fact's
    /// only reader was the static-C++ emission that d91c85bf8 retired.
    /// Its opposite number, the driver's `configure_verify_hidden_stash`,
    /// has no production caller either — both ends of the cross-check are
    /// waiting on the same in-flight Rust port.
    #[serde(default)]
    pub verify_stash: bool,
    /// `Qwen3_5MoeMlpWorkspace::cutlass_max_rows` — `min(max_tokens, 512)`
    /// when `kernels::moe::flashinfer_cutlass_moe_enabled()` sized a workspace,
    /// else 0. Zero means the fused leg does not exist on this
    /// deployment; non-zero is the ROW BOUND of the MoE text, and fires
    /// above it decline rather than the declaration guessing which of
    /// the remaining three legs the pass would have taken.
    #[serde(default)]
    pub moe_cutlass_max_rows: u32,
    /// `PIE_QWEN35_PREFILL_DECODE` (default ON) AND the cache terms the
    /// hand-written prepare reads beside it (`is_native_bf16()` and no
    /// HND layout). With it set, a SINGLE-REQUEST pure-decode fire is
    /// planned and dispatched through the PREFILL flashinfer path, not
    /// the decode one -- measured at ~7x on this model's attention shape
    /// (`prepare_qwen3_5_decode_plan`'s table).
    ///
    /// It is a fact rather than a class because the per-fire term is
    /// `num_requests == 1`, and in a pure-decode fire that IS
    /// `TokensLE(1)` -- already in the guard vocabulary. Before this
    /// existed, the decode class stated the decode dispatch
    /// unconditionally and the prepare built no decode plan, so every
    /// single-request decode threw a drift and FAILED THE MODEL LOAD.
    #[serde(default)]
    pub prefill_decode: bool,
    /// `add_to_residual` — tp==1, so the MoE block's output lands on the
    /// residual stream inside this pass. At tp>1 the block writes to
    /// scratch and an allreduce follows, which is a different (and
    /// unstated) shape.
    #[serde(default)]
    pub moe_residual_fold: bool,
    /// The shared expert's gate weight is bound and unquantized, so its
    /// landing is the fused dot form. False sends it to the
    /// `[Tokens, 1]` GEMM plus a separate scalar-gate add, which this
    /// text does not state.
    #[serde(default)]
    pub moe_shared_gate_dot: bool,
    /// `Lw.expert_cache != nullptr` — the experts are paged one at a
    /// time, so every device-side leg that strides a fused slab is off
    /// the table and the pass takes the host-routed path.
    #[serde(default)]
    pub moe_streamed_experts: bool,
    /// `qwen35_moe_force_general_path()` — the env that pins the pass to
    /// the host-routed path regardless of shape.
    #[serde(default)]
    pub moe_force_general: bool,
    /// The DENSE MLP's gate_up BINDING — `Lw.gate_up_proj_fused !=
    /// nullptr`, so the packed GEMM lands in one buffer and the
    /// activation is the CHUNKED swiglu over it; without it the
    /// projection writes two and the activation is the pair form.
    ///
    /// [`LlamaLikeCudaFacts::gate_up_fused`]'s reasoning applies
    /// verbatim, including why the workspace term the executor also
    /// tested is dead. Only the MoE arm's shared expert is unaffected:
    /// it always binds a packed bank, so its text states the chunked
    /// form outright.
    #[serde(default)]
    pub gate_up_fused: bool,
    /// How this deployment STORES its linear projections — the weight
    /// representation axis ([`model_compiler::dsl::WeightRepr`]).
    ///
    /// [`LlamaLikeCudaFacts::proj_repr`]'s reasoning applies verbatim,
    /// and this family had EIGHT of the eighteen `make_weight_view`
    /// sites the axis removes. Serde-defaulted to dense (append-only
    /// discipline).
    #[serde(default)]
    pub proj_repr: WeightRepr,
    /// The SLIDING WINDOW each layer attends over, `-1` for none —
    /// read through [`model_compiler::facts::window_left_at`], which is
    /// where the shape of this list is documented.
    ///
    /// The dispatch statements carry it, so no executor reaches into
    /// `fwd_cfg.per_layer_window_left` for it. Serde-defaulted, and
    /// empty reads as "no window", which is what every fixture written
    /// before this field meant.
    #[serde(default)]
    pub window_left: Vec<i32>,
}

impl Qwen35CudaFacts {
    /// SYNTHETIC fixture — NOT a measurement. These values (bf16 state,
    /// warp-tiled arm live, thresholds 64 / 4096 — plausible defaults)
    /// pin the GOLDEN FORM of the lowered qwen3_5 traces only: the live
    /// derivation and its digest validation against the driver's own
    /// booleans are rung 4c-iii. The precedent for refusing to call this
    /// "measured" is [`LlamaLikeCudaFacts::qwen3_0_6b_l40s`]: its first
    /// version guessed `xqa_decode: true` and called it measured, and the
    /// rung-3 digest caught the lie on its first live run. This
    /// constructor makes no such claim — every consumer of these goldens
    /// must treat the arm structure as the artifact under review, not the
    /// deployment's truth.
    ///
    /// And the digest DID catch this one too (its fourth catch): the
    /// LIVE L40S default-env derivation is
    /// `warp_tiled: false, cached_max: 0` (both env-gated off), which is
    /// what the emission fact set in `bin/emit-cuda.rs` uses. This
    /// fixture keeps the synthetic values deliberately: the goldens pin
    /// the guard-chain STRUCTURE (a warp arm that exists, a cached arm
    /// with a real threshold), which the live-default set would erase.
    pub fn qwen3_5_0_8b_synthetic() -> Self {
        Self {
            // 0.8B attends the whole context.
            window_left: Vec::new(),
            state_bf16: true,
            warp_tiled: true,
            warp_tiled_max: 64,
            cached_max: 4096,
            verify_stash: true,
            // The MoE fields describe the fused leg as the driver has it
            // today: the CUTLASS workspace is always sized
            // (`flashinfer_cutlass_moe_enabled()` returns true
            // unconditionally), 512 is `kFusedMoeMaxRows`, tp=1 folds the
            // residual, and neither the streamed-expert cache nor the
            // force-general env is on by default. Synthetic like the rest
            // of this fixture — the 0.8B checkpoint is dense and reaches
            // none of them; they pin the MoE block's golden form.
            moe_cutlass_max_rows: 512,
            prefill_decode: false,
            moe_residual_fold: true,
            moe_shared_gate_dot: true,
            moe_streamed_experts: false,
            moe_force_general: false,
            // 0.8B binds the packed bank (qwen3_5.cpp's loader takes the
            // same `dense_fused_projection_joins` contract llama_like's
            // does), so the chunked form is the golden's shape.
            gate_up_fused: true,
            // Dense, and for the same contract reason: a group the join
            // packs is a BF16 one.
            proj_repr: WeightRepr::Bf16,
        }
    }
}
