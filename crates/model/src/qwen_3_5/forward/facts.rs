//! `qwen3_5`'s load-time facts: one struct per fragment, plus the hybrid's
//! layer schedule and the CUDA deployment's bindings.

use serde::{Deserialize, Serialize};
use model_compiler::dsl::WeightRepr;
use model_compiler::trace::NormVariant;

/// Facts for one qwen3_5_moe-family MoE MLP block — a traced FRAGMENT, not
/// a model.
///
/// Deliberately narrower than a `Qwen35MoeFacts` would be: the qwen3.5
/// family alternates GDN linear-attention layers with full-attention layers
/// (the HYBRID part, `crates/driver-cuda/csrc/src/model/qwen3_5/qwen3_5_forward.cpp`),
/// and declaring the MoE MLP inside the llama_like skeleton would trace a
/// model that does not exist. So these facts describe exactly the unit the
/// qwen3_5 hybrid composes per layer — `y += moe_mlp(rmsnorm(y))`, the
/// [`Qwen35MlpKind::Moe`] arm of [`Qwen35HybridFacts`] — and
/// `family::qwen3_5_moe_mlp_block` traces that unit standalone. The GDN
/// attention half is [`Qwen35GdnFacts`] / `family::qwen3_5_gdn_block` —
/// its own fragment, same reasoning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35MoeMlpFacts {
    pub hidden: u32,
    /// Routed expert count (HF `num_experts` / `num_local_experts`).
    pub num_experts: u32,
    /// Experts per token (HF `num_experts_per_tok`) — the router's k.
    pub top_k: u32,
    /// Per-expert intermediate width (HF `moe_intermediate_size`).
    pub moe_intermediate: u32,
    /// Shared-expert intermediate width (HF
    /// `shared_expert_intermediate_size`); 0 means no shared expert, which
    /// is the qwen3_moe shape — the hand-written pass skips the whole
    /// shared block when the bind wired no `shared_*` pointers.
    pub shared_expert_intermediate: u32,
    /// qwen3.5/3.6 use the Gemma `(1 + w)` fold
    /// (`qwen3_5_moe_forward.cpp::uses_gemma_rmsnorm`: everything but plain
    /// `qwen3_moe`).
    pub norm_variant: NormVariant,
}

impl Qwen35MoeMlpFacts {
    /// Qwen3.5-35B-A3B, the small qwen3_5_moe checkpoint.
    ///
    /// No config.json is committed in this tree, so these dims are pinned
    /// from the driver's own measured notes on this checkpoint
    /// (`qwen3_5_moe_forward.cpp`): 256 routed experts ("with 256 experts
    /// holding only a few routes each"); gate_up bytes per expert 4.2 MB at
    /// tp=1 / 2.1 MB at tp=2 = `2 * moe_intermediate * hidden * 2B` with
    /// `moe_intermediate = 512, hidden = 2048`; top-k 8 (the profiled
    /// N=128 decode step's "352 blocks for ~252 active experts" matches
    /// the aligned-decode block formula only at `routes = N * 8`); and a
    /// shared expert with `Is == Im` (the precondition of the shared-fold
    /// experiment, which rode it along "as one more expert").
    pub fn qwen3_5_35b_a3b() -> Self {
        Self {
            hidden: 2048,
            num_experts: 256,
            top_k: 8,
            moe_intermediate: 512,
            shared_expert_intermediate: 512,
            norm_variant: NormVariant::Gemma,
        }
    }
}

/// Facts for one qwen3_5 GDN (gated-deltanet) linear-attention block — the
/// second traced FRAGMENT, and the other layer kind of the qwen3.5 hybrid.
///
/// Describes exactly the unit the qwen3_5 hybrid composes on a `Linear`
/// layer — `y += gdn(l, rmsnorm(y, attn_norm))` (plan.md Part 1's
/// `match layers[l] { ..., Linear => gdn(l, x, h) }`) — traced standalone by
/// `family::qwen3_5_gdn_block`, mirroring
/// `qwen3_5_forward.cpp::linear_attn_layer_body` launch for launch. The
/// full-attention layer kind is [`Qwen35FullAttnFacts`] /
/// `family::qwen3_5_full_attn_block` — its own fragment, same reasoning —
/// and [`Qwen35HybridFacts`] composes both into the full model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35GdnFacts {
    pub hidden: u32,
    /// GDN key heads (HF `linear_num_key_heads`).
    pub key_heads: u32,
    /// GDN value heads (HF `linear_num_value_heads`); a multiple of
    /// `key_heads` (GQA share) or equal to it.
    pub value_heads: u32,
    /// Per-head key width (HF `linear_key_head_dim`).
    pub key_head_dim: u32,
    /// Per-head value width (HF `linear_value_head_dim`).
    pub value_head_dim: u32,
    /// Depthwise conv window (HF `linear_conv_kernel_dim`).
    pub conv_kernel: u32,
    /// The deployment bound the fused `in_proj_qkvz` + `in_proj_ba` banks.
    /// A *binding* fact, llama_like's `fused_qkv` precedent: the checkpoint
    /// ships four raw projections (`in_proj_{qkv,z,b,a}`) and the CUDA
    /// contract's `gdn_fused_in_proj_joins` re-joins them — but only behind
    /// `PIE_QWEN35_FUSED_GDN_PROJ` (default OFF,
    /// `qwen35_fused_gdn_projection_enabled()`), so the default deployment
    /// binds four projections and the trace writes four matmuls; with the
    /// join enabled it writes two matmuls + two `SplitGdn`s
    /// (`qwen3_5_forward.cpp` branches on `la_in_proj_qkvz`/`la_in_proj_ba`
    /// the same way).
    pub fused_in_proj: bool,
    /// qwen3.5/3.6 use the Gemma `(1 + w)` fold for the block norms
    /// (`kernels::norm::rmsnorm_gemma_bf16` on the pre-attention norm). The GATED
    /// norm inside the block is not governed by this: its weight fold is
    /// plain by kernel contract (`rmsnorm.hpp`).
    pub norm_variant: NormVariant,
}

impl Qwen35GdnFacts {
    /// `key_heads * key_head_dim` — one leg of the packed conv input.
    pub fn key_width(&self) -> u32 {
        self.key_heads * self.key_head_dim
    }

    /// `value_heads * value_head_dim` — the v leg, the z gate width, and
    /// the o_proj input width.
    pub fn value_width(&self) -> u32 {
        self.value_heads * self.value_head_dim
    }

    /// The packed `[q | k | v]` conv width: `2 * key_width + value_width`.
    pub fn conv_dim(&self) -> u32 {
        2 * self.key_width() + self.value_width()
    }

    /// Qwen3.5-0.8B, the workspace's linear-attention parity checkpoint
    /// (`crates/driver-cuda/csrc/tests/parity_qwen3_5_multireq.py` defaults to
    /// `Qwen/Qwen3.5-0.8B-Base`).
    ///
    /// No config.json is committed in this tree, so every dimension is
    /// pinned from the drivers' own statements of this checkpoint:
    ///
    /// * `crates/driver-metal/csrc/src/model/qwen3_5/geometry.hpp` (`DecodeGeometry`
    ///   defaults, the Metal driver's 0.8B target): `hidden = 1024`,
    ///   `gdn_k_heads = 16`, `gdn_v_heads = 16`, `gdn_k_dim = 128`,
    ///   `gdn_v_dim = 128`, `gdn_conv_k = 4`, `gdn_conv_dim = 6144`,
    ///   `gdn_v_total = 2048` — and `conv_dim()`/`value_width()` here
    ///   reproduce those last two (2·2048 + 2048 = 6144, 16·128 = 2048).
    /// * `crates/driver-metal/csrc/src/model/qwen3_5/decode_consts.cpp` corroborates
    ///   the widths as launch geometry: in-proj 1024 → 6144, z 1024 →
    ///   2048, out-proj 2048 → 1024, "in_proj_a / in_proj_b — DENSE bf16
    ///   GEMV [16, 1024]" (= value_heads × hidden).
    /// * `crates/driver-cuda/csrc/src/model/config.hpp:357` pins the conv window: 4.
    /// * `fused_in_proj: false` is the live default binding
    ///   (`PIE_QWEN35_FUSED_GDN_PROJ` unset — see the field doc).
    /// * `norm_variant: Gemma`: `qwen3_5_forward.cpp` launches
    ///   `kernels::norm::rmsnorm_gemma_bf16` for every block norm, and the Metal
    ///   port states "All RMSNorm gains use the Gemma (1+w) convention"
    ///   (`crates/driver-metal/csrc/tests/mlx/model/qwen3_5.hpp`).
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            hidden: 1024,
            key_heads: 16,
            value_heads: 16,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel: 4,
            fused_in_proj: false,
            norm_variant: NormVariant::Gemma,
        }
    }
}

/// Facts for one qwen3_5 FULL-attention block — the third traced FRAGMENT,
/// and the last layer kind the qwen3.5 hybrid needed.
///
/// This is NOT llama_like's attention, which is why it gets its own facts
/// instead of a `LlamaLikeFacts` configuration: the q projection is 2× wide
/// with an interleaved per-head `[query | gate]` split
/// (`kernels::layout::split_q_gate_bf16`), the attention output is multiplied by
/// `sigmoid(gate)` (`kernels::mlp::sigmoid_gate_inplace_bf16` — no residual, not
/// the shared-expert `SigmoidGateAdd`), rope is PARTIAL
/// (`partial_rotary_factor`, `kernels::rope::rope_partial_bf16`), and the per-head
/// q/k norms fold Gemma-style (`kernels::norm::rmsnorm_gemma_bf16` over `N * heads`
/// rows of `head_dim`). The qk-norm is not a tri-state here: the
/// hand-written `full_attn_layer_body` launches the per-head pair
/// unconditionally, so the declaration does too, and only the fold is a
/// fact ([`Qwen35FullAttnFacts::norm_variant`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35FullAttnFacts {
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Partial-rotary width: the leading channels of each head that rotate
    /// (`OpKind::Rope`'s `partial`). Stated as the resolved channel count,
    /// not HF's factor — the driver's `max(2, 2 * int(0.5 *
    /// partial_rotary_factor * head_dim))` derivation
    /// (`qwen3_5_forward.cpp`) is config parsing, and the fixture pins its
    /// result with provenance.
    pub rotary_dim: u32,
    /// The deployment bound one packed `[2q | k | v]` projection
    /// (`fa_qgkv_proj_fused`). A *binding* fact, llama_like's `fused_qkv`
    /// precedent: the join is env-gated default-OFF
    /// (`PIE_QWEN35_FUSED_FULL_ATTN_QGKV`,
    /// `qwen3_5.cpp::fused_full_attn_qgkv_weights_enabled`), so the default
    /// deployment binds three projections and the trace writes three
    /// matmuls; with the join enabled it writes Matmul(qgkv) + SplitQkv
    /// whose "q" leg is the 2×-wide `[query | gate]` bank
    /// (`full_attn_layer_body`'s `use_fused_qgkv` branch:
    /// `kernels::attn::split_qkv_bf16(packed, qg, k, v, N, 2*Hq, Hk)`).
    pub fused_qkv: bool,
    /// qwen3.5 folds `(1 + w)` on every norm of this block — the
    /// pre-attention norm AND the per-head q/k norms
    /// (`kernels::norm::rmsnorm_gemma_bf16` throughout `full_attn_layer_body`).
    pub norm_variant: NormVariant,
}

impl Qwen35FullAttnFacts {
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    /// Qwen3.5-0.8B's full-attention geometry, pinned from the drivers'
    /// own statements of this checkpoint (no config.json is committed;
    /// same provenance discipline as [`Qwen35GdnFacts::qwen3_5_0_8b`]):
    ///
    /// * `crates/driver-metal/csrc/src/model/qwen3_5/geometry.hpp` (`DecodeGeometry`
    ///   defaults, the Metal driver's 0.8B target): `hidden = 1024`,
    ///   `n_q_heads = 8`, `n_kv_heads = 2`, `head_dim = 256`,
    ///   `rotary_dims = 64` ("derived from partial_rotary_factor *
    ///   head_dim").
    /// * `crates/driver-metal/csrc/src/model/qwen3_5/decode_consts.cpp` corroborates
    ///   the widths as launch geometry: "2×-wide gated q_proj (4096)" =
    ///   2 · 8 · 256, k/v 1024 → 512 = 2 · 256, o_proj 2048 → 1024.
    /// * `rotary_dim = 64`: `partial_rotary_factor = 0.25` (the family
    ///   default `crates/driver-metal/csrc/src/batch/forward.hpp` states and
    ///   `crates/driver-cuda/csrc/src/model/qwen3_5/qwen3_5.hpp` documents —
    ///   "`partial_rotary_factor=0.25` — only the first 25% of head_dim is
    ///   rotated"); the CUDA derivation `max(2, 2·int(0.5·0.25·256))`
    ///   (`qwen3_5_forward.cpp`) and Metal's `rotary_dims` both land on 64.
    /// * `fused_qkv: false` is the live default binding
    ///   (`PIE_QWEN35_FUSED_FULL_ATTN_QGKV` unset — see the field doc).
    /// * `norm_variant: Gemma`: `full_attn_layer_body` launches
    ///   `kernels::norm::rmsnorm_gemma_bf16` for the block norm and both per-head
    ///   q/k norms, and the Metal port states "All RMSNorm gains use the
    ///   Gemma (1+w) convention" (`crates/driver-metal/csrc/tests/mlx/model/qwen3_5.hpp`).
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            hidden: 1024,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            fused_qkv: false,
            norm_variant: NormVariant::Gemma,
        }
    }
}

/// Which MLP the qwen3_5 hybrid runs on every layer: the dense SwiGLU block
/// (qwen3.5 dense checkpoints — `qwen35_dense_mlp_block`) or the MoE block
/// (qwen3.5/3.6-MoE — `run_moe_mlp`, the [`Qwen35MoeMlpFacts`] fragment).
/// One enum for the whole model because the family applies the same MLP
/// kind to every layer (`qwen3_5_forward.cpp` has no per-layer MLP switch;
/// the per-layer axis of this family is the ATTENTION kind).
///
/// WHICH ARM A CHECKPOINT TAKES IS A READING OF ITS CONFIG, not of its
/// `model_type`. Qwen3.6-27B is `model_type: qwen3_5` and takes `Dense`
/// (no `num_experts`, `intermediate_size` 17408); the MoE arm is the
/// 35B-A3B-shaped checkpoints'. Worth stating because the opposite was
/// once assumed here, and it aimed a stretch of work at a branch the
/// checkpoint in question never reaches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Qwen35MlpKind {
    Dense { intermediate: u32 },
    Moe(Qwen35MoeMlpFacts),
}

/// Facts for the full qwen3_5 HYBRID model — the declaration that composes
/// the three fragments: plan.md Part 1's
/// `match layers[l] { Full => full_attn(l, x), Linear => gdn(l, x) }`,
/// a static match resolved at trace time.
///
/// # How the layer kinds are known
///
/// The checkpoint states them explicitly: `config.json` ships a
/// `layer_types` array of `"linear_attention"` / `"full_attention"` with one
/// entry per layer, which is the CUDA driver's sole source
/// (`HfConfig::layer_types`, parsed in `crates/driver-cuda/csrc/src/model/config.cpp`;
/// `qwen3_5.cpp` refuses a length mismatch). The qwen3.5 checkpoints ship a
/// REGULAR pattern — one full-attention layer every
/// `full_attention_interval`, the rest linear
/// (`crates/driver-metal/csrc/tests/mlx/model/qwen3_5.hpp`) — and the Metal driver
/// reduces the array to exactly that interval, refusing irregular arrays
/// (`crates/driver-metal/csrc/src/batch/forward.hpp`: "-1: `layer_types` is irregular,
/// refuse"; `crates/driver-metal/csrc/src/model/qwen3_5/geometry.hpp::is_full_attn`).
/// These facts state the interval, mirroring that reduction: a hypothetical
/// irregular checkpoint is outside this declaration's vocabulary, exactly
/// as it is outside the Metal driver's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35HybridFacts {
    pub layers: u32,
    /// One full-attention layer every `full_attn_interval`-th, at the END
    /// of each block: `is_full_attn(l) = l % interval == interval - 1`
    /// (the Metal geometry's formula, verbatim). `1` makes every layer
    /// full attention.
    pub full_attn_interval: u32,
    pub vocab: u32,
    /// The lm_head weight is the embedding table (weight tying).
    pub tied_embeddings: bool,
    /// The fold of the FINAL norm (the per-block norms carry their own
    /// variant inside the sub-facts; qwen3.5 folds Gemma everywhere).
    pub norm_variant: NormVariant,
    /// The full-attention layer kind.
    pub attn: Qwen35FullAttnFacts,
    /// The GDN linear-attention layer kind.
    pub gdn: Qwen35GdnFacts,
    /// The (uniform) per-layer MLP.
    pub mlp: Qwen35MlpKind,
}

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
    /// engine owns that configuration, so — like [`Self::state_bf16`],
    /// whose dtype the engine likewise decides — the fact is stated here
    /// and the driver cross-checks its own derivation per fire rather
    /// than choosing. With the stash live, the CommitAdvance class
    /// replays each linear layer's in-proj outputs from the stash
    /// (`cuda::verify_stash_load`) and skips the GEMMs; without it, the
    /// commit pass re-runs the in-projections against whatever the
    /// workspace holds. Serde-defaulted so pre-4c-iv facts JSON reads
    /// back unchanged (the append-only discipline).
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

impl Qwen35HybridFacts {
    /// Whether layer `l` runs full attention —
    /// `crates/driver-metal/csrc/src/model/qwen3_5/geometry.hpp::is_full_attn`.
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_compiler::facts::full_attn_at(self.full_attn_interval, l)
    }

    /// The model's hidden size (the sub-facts each carry it for standalone
    /// tracing; [`qwen3_5_hybrid`] asserts they agree).
    pub fn hidden(&self) -> u32 {
        self.attn.hidden
    }

    /// Qwen3.5-0.8B, the workspace's hybrid parity checkpoint
    /// (`crates/driver-cuda/csrc/tests/parity_qwen3_5_multireq.py` defaults to
    /// `Qwen/Qwen3.5-0.8B-Base`). Sub-facts are the provenance-pinned 0.8B
    /// fixtures ([`Qwen35FullAttnFacts::qwen3_5_0_8b`],
    /// [`Qwen35GdnFacts::qwen3_5_0_8b`]); the model-level dims are pinned
    /// from `crates/driver-metal/csrc/src/model/qwen3_5/geometry.hpp` (`DecodeGeometry`
    /// defaults, the Metal driver's 0.8B target): `n_layers = 24`,
    /// `full_attn_interval = 4` (layers 3, 7, 11, 15, 19, 23 full — the
    /// family's 3:1 linear:full pattern), `vocab = 248320`
    /// (`decode_consts.cpp` corroborates: lm_head 1024 → 248320),
    /// `tied_embeddings = true`. The MLP is DENSE with `intermediate =
    /// 3584` (geometry.hpp; `decode_consts.cpp`: gate/up 1024 → 3584,
    /// down 3584 → 1024) — "Dense only on the 0.8B target (MoE deferred)"
    /// (`crates/driver-metal/csrc/tests/mlx/model/qwen3_5.hpp`), and the CUDA dense
    /// family (`model_type: qwen3_5`, `qwen3_5_forward.cpp`) runs
    /// `qwen35_dense_mlp_block` on every layer.
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            layers: 24,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: true,
            norm_variant: NormVariant::Gemma,
            attn: Qwen35FullAttnFacts::qwen3_5_0_8b(),
            gdn: Qwen35GdnFacts::qwen3_5_0_8b(),
            mlp: Qwen35MlpKind::Dense { intermediate: 3584 },
        }
    }

    /// Qwen3.6-27B — the DENSE hybrid, read from the checkpoint's own
    /// `config.json` (`text_config`), not inferred from the family name.
    ///
    /// Every value here is a field of that file or the driver's stated
    /// derivation from one: 64 layers, `full_attention_interval` 4,
    /// `vocab_size` 248320, `tie_word_embeddings` false,
    /// `intermediate_size` 17408 (no `num_experts` — this checkpoint
    /// takes the `Dense` arm, see [`Qwen35MlpKind`]), hidden 5120,
    /// 24 q heads over 4 kv heads at `head_dim` 256, and
    /// `partial_rotary_factor` 0.25 → `rotary_dim` 64 by the driver's
    /// `max(2, 2 * int(0.5 * f * head_dim))`. The GDN half is the
    /// `linear_*` block: 16 key heads, 48 value heads (a GQA ratio of 3,
    /// which `family.rs`'s gdn body already branches on), 128/128 head
    /// dims, `linear_conv_kernel_dim` 4.
    ///
    /// `fused_in_proj` / `fused_qkv` are false because both joins are
    /// env-gated default-off, the same as 0.8B's.
    ///
    /// NOT reachable on an L40S at bf16 — 27B is ~55 GB against 46. An
    /// FP8 checkpoint of the same geometry is what would boot here; the
    /// traced form is identical either way, which is why the fixture is
    /// worth having before the hardware is.
    pub fn qwen3_6_27b() -> Self {
        Self {
            layers: 64,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Gemma,
            attn: Qwen35FullAttnFacts {
                hidden: 5120,
                q_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                rotary_dim: 64,
                fused_qkv: false,
                norm_variant: NormVariant::Gemma,
            },
            gdn: Qwen35GdnFacts {
                hidden: 5120,
                key_heads: 16,
                value_heads: 48,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel: 4,
                fused_in_proj: false,
                norm_variant: NormVariant::Gemma,
            },
            mlp: Qwen35MlpKind::Dense {
                intermediate: 17_408,
            },
        }
    }
}

