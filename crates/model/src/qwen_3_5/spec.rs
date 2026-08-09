//! `qwen3_5`'s SHAPE: the numbers a checkpoint of the GDN hybrid has.
//!
//! Ungated, because a row is written in these words and a row is the
//! crate's identity under every aspect -- `chat` asks which template
//! speaks for it, `contract` asks how to author it, `forward` asks what
//! to trace. `llama_like`'s `spec.rs` made the argument first; this
//! family follows it, and has one more reason to: the hybrid's layer
//! SCHEDULE (which layers are gated-delta-net and which are full
//! attention) is what its manifest and its `Deployment` are both
//! projections of, and neither of those belongs to the tracer.
//!
//! What stayed behind in `forward/facts.rs` is [`Qwen35CudaFacts`], the
//! per-backend BINDING facts -- env gates, workspace ceilings, which
//! joins the load materialized. Those name kernels, so they belong to
//! the aspect that has them.
//!
//! [`Qwen35CudaFacts`]: super::forward::facts::Qwen35CudaFacts

use serde::{Deserialize, Serialize};
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The three GDN widths are arithmetic over the head counts, and the
    /// conv bank is the packed `[K | K | V]` one — which is why the
    /// contract shards it by BLOCK and not by row: a uniform row split
    /// hands a rank part of K where it needs V.
    #[test]
    fn the_gdn_widths_are_the_head_counts_multiplied_out() {
        let g = Qwen35GdnFacts::qwen3_5_0_8b();
        assert_eq!(g.key_width(), g.key_heads * g.key_head_dim);
        assert_eq!(g.value_width(), g.value_heads * g.value_head_dim);
        assert_eq!(g.conv_dim(), 2 * g.key_width() + g.value_width());
    }

    /// The full-attention widths, likewise — and `q_width` is the
    /// UNGATED one: the checkpoint's q bank is twice this, because it
    /// carries the output gate beside the query.
    #[test]
    fn the_attention_widths_are_the_head_counts_multiplied_out() {
        let a = Qwen35FullAttnFacts::qwen3_5_0_8b();
        assert_eq!(a.q_width(), 8 * 256);
        assert_eq!(a.kv_width(), 2 * 256);
    }

    /// The schedule: one full-attention layer at the END of each block
    /// of `full_attn_interval`, which is the Metal geometry's formula
    /// verbatim. It is the only statement of which layers attend, and
    /// both the deployment's `linear_layers` and the trace read it.
    #[test]
    fn the_schedule_puts_full_attention_at_the_end_of_each_block() {
        let f = Qwen35HybridFacts::qwen3_5_0_8b();
        assert_eq!(f.full_attn_interval, 4);
        let full: Vec<u32> = (0..f.layers).filter(|&l| f.is_full_attn(l)).collect();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23]);
        assert_eq!(f.hidden(), f.attn.hidden);
        assert_eq!(f.hidden(), f.gdn.hidden, "the sub-facts agree");
    }

    /// Qwen3.6-27B is the same shape at another size, and its GDN share
    /// is the widest in the lineage: 48 value heads over 16 key heads.
    #[test]
    fn the_27b_fixture_is_the_same_shape_at_another_size() {
        let f = Qwen35HybridFacts::qwen3_6_27b();
        assert_eq!(f.layers, 64);
        assert_eq!(f.full_attn_interval, 4);
        assert_eq!((f.gdn.key_heads, f.gdn.value_heads), (16, 48));
        assert!(matches!(f.mlp, Qwen35MlpKind::Dense { intermediate: 17_408 }));
        assert!(!f.tied_embeddings, "27B ships its own head");
    }

    /// The MoE block is a FRAGMENT of a hybrid layer and states its own
    /// widths, including the shared expert a `qwen3_moe` checkpoint does
    /// not have — 0 there, 512 here, and the difference is a bound
    /// tensor rather than a branch.
    #[test]
    fn the_mixture_block_states_its_shared_expert() {
        let m = Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
        assert_eq!(m.num_experts, 256);
        assert_eq!(m.top_k, 8);
        assert_eq!(m.moe_intermediate, 512);
        assert_eq!(m.shared_expert_intermediate, 512);
        assert_eq!(m.norm_variant, NormVariant::Gemma);
    }
}
