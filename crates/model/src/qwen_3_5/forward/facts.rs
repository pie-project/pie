//! `qwen3_5`'s per-backend binding facts.
//!
//! The SHAPE lives in `../spec.rs` (ungated). What a deployment BOUND is known
//! only when that backend's aspect is compiled, so it is stated here.

use model_dsl::WeightRepr;
use serde::{Deserialize, Serialize};

/// The shape, re-exported so a declaration reaches its facts from one place.
pub use super::super::spec::{
    Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts,
};

/// CUDA backend facts for a LOWERED qwen3_5 hybrid trace
/// (`family::qwen3_5_hybrid_cuda`).
///
/// Everything here is load-time. The N-thresholds are VALUES carried into
/// [`model_ir::trace::GuardPred`]s because only N varies per fire; the
/// predicates around them resolve here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35CudaFacts {
    /// Recurrent-state store dtype is bf16 — the `state_bf16` launcher suffix.
    pub state_bf16: bool,
    /// The warp-tiled prefill arm EXISTS: `K_d <= 256` && the state-persist env
    /// gate. `write_state` is always true for the normal-fire classes, so the
    /// env term is the whole eligibility; `commit_lens == nullptr` is a CLASS.
    pub warp_tiled: bool,
    /// `PIE_QWEN35_GDN_WARP_TILED_MAX_TOKENS` (64) — that arm's `TokensLE`.
    pub warp_tiled_max: u32,
    /// `PIE_QWEN35_GDN_CACHED_PREFILL_MAX_TOKENS` (default 0 = off).
    pub cached_max: u32,
    /// The deployment configures the verify hidden stash: the engine owns that
    /// configuration, so the driver cross-checks its own derivation per fire
    /// rather than choosing. Serde-defaulted so older facts JSON still reads.
    ///
    /// NOTHING READS THIS TODAY, and `tests/facts_are_read.rs` says so; the
    /// driver's `configure_verify_hidden_stash` has no production caller either.
    #[serde(default)]
    pub verify_stash: bool,
    /// `Qwen3_5MoeMlpWorkspace::cutlass_max_rows` — 0 means the fused leg does
    /// not exist here; non-zero is the ROW BOUND of the MoE text, and fires
    /// above it decline rather than the declaration guessing which leg ran.
    #[serde(default)]
    pub moe_cutlass_max_rows: u32,
    /// `PIE_QWEN35_PREFILL_DECODE` (default ON) AND the cache terms beside it.
    /// A single-request pure-decode fire then dispatches through the PREFILL
    /// flashinfer path — measured ~7x on this attention shape. A fact, not a
    /// class, because `num_requests == 1` in a pure-decode fire IS `TokensLE(1)`.
    #[serde(default)]
    pub prefill_decode: bool,
    /// `add_to_residual` — tp==1, so the MoE output lands on the residual stream
    /// inside this pass. tp>1 is a different, unstated shape.
    #[serde(default)]
    pub moe_residual_fold: bool,
    /// Shared expert's gate weight is bound and unquantized, so its landing is
    /// the fused dot form; false is a shape this text does not state.
    #[serde(default)]
    pub moe_shared_gate_dot: bool,
    /// `Lw.expert_cache != nullptr` — experts page one at a time, so every leg
    /// that strides a fused slab is off the table.
    #[serde(default)]
    pub moe_streamed_experts: bool,
    /// `qwen35_moe_force_general_path()` — pins the pass to the host-routed leg.
    #[serde(default)]
    pub moe_force_general: bool,
    /// The DENSE MLP's gate_up BINDING — one packed buffer, so the activation is
    /// the CHUNKED swiglu. The MoE shared expert always binds a packed bank.
    #[serde(default)]
    pub gate_up_fused: bool,
    /// Projection storage ([`model_dsl::WeightRepr`]); serde-defaults to dense.
    #[serde(default)]
    pub proj_repr: WeightRepr,
    /// Per-layer sliding window, `-1` for none, read through
    /// [`model_ir::facts::window_left_at`]. Empty reads as "no window".
    #[serde(default)]
    pub window_left: Vec<i32>,
}

impl Qwen35CudaFacts {
    /// SYNTHETIC fixture — NOT a measurement. It pins the GOLDEN FORM of the
    /// lowered traces only. The live L40S default-env derivation is
    /// `warp_tiled: false, cached_max: 0`; the synthetic values stay because the
    /// goldens pin the guard-chain STRUCTURE, which live defaults would erase.
    pub fn qwen3_5_0_8b_synthetic() -> Self {
        Self {
            // 0.8B attends the whole context.
            window_left: Vec::new(),
            state_bf16: true,
            warp_tiled: true,
            warp_tiled_max: 64,
            cached_max: 4096,
            verify_stash: true,
            // The fused leg as the driver has it today: 512 is
            // `kFusedMoeMaxRows`, tp=1 folds the residual. Synthetic — the 0.8B
            // checkpoint is dense and reaches none of them.
            moe_cutlass_max_rows: 512,
            prefill_decode: false,
            moe_residual_fold: true,
            moe_shared_gate_dot: true,
            moe_streamed_experts: false,
            moe_force_general: false,
            // 0.8B binds the packed bank, so the chunked form is the golden's.
            gate_up_fused: true,
            // Dense, same contract reason: a group the join packs is a BF16 one.
            proj_repr: WeightRepr::Bf16,
        }
    }
}
