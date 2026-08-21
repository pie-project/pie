//! `qwen3_5`'s SHAPE: the numbers a checkpoint of the GDN hybrid has.
//!
//! Ungated, because a row is the crate's identity under every aspect — `chat`
//! asks which template speaks for it, `contract` how to author it, `forward`
//! what to trace. This family has one more reason than `llama_like`: the
//! hybrid's layer SCHEDULE is what its manifest and its `Deployment` are both
//! projections of, and neither belongs to the tracer. Per-backend BINDING facts
//! stay in `forward/facts.rs` as [`Qwen35CudaFacts`] — env gates, workspace
//! ceilings, which joins the load materialized — because those name kernels.
//!
//! [`Qwen35CudaFacts`]: super::forward::facts::Qwen35CudaFacts

use model_ir::trace::NormVariant;
use serde::{Deserialize, Serialize};

/// Facts for one qwen3_5_moe-family MoE MLP block — a traced FRAGMENT, not
/// a model.
///
/// Deliberately narrower than a `Qwen35MoeFacts`: the family alternates GDN
/// linear-attention with full-attention layers, so declaring the MoE MLP inside
/// the llama_like skeleton would trace a model that does not exist. These facts
/// describe exactly the unit the hybrid composes per layer —
/// `y += moe_mlp(rmsnorm(y))`, the [`Qwen35MlpKind::Moe`] arm of
/// [`Qwen35HybridFacts`] — which `family::qwen3_5_moe_mlp_block` traces alone.
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
    /// qwen3.5/3.6 fold PLAINLY — `xhat * w`, the `nn.RMSNorm` reading.
    ///
    /// This said Gemma, on the authority of
    /// `qwen3_5_moe_forward.cpp::uses_gemma_rmsnorm`. See this generation's
    /// module doc for the three measurements that overturned it and for what
    /// the other fold generated.
    pub norm_variant: NormVariant,
}

impl Qwen35MoeMlpFacts {
    /// Qwen3.5-35B-A3B, the small qwen3_5_moe checkpoint.
    ///
    /// No config.json is committed here, so these dims are pinned from
    /// `qwen3_5_moe_forward.cpp`'s measured notes: 256 routed experts; gate_up
    /// 4.2 MB/expert at tp=1 = `2 * moe_intermediate * hidden * 2B` with
    /// `moe_intermediate = 512, hidden = 2048`; top-k 8 (the profiled N=128 step's
    /// "352 blocks for ~252 active experts" matches the aligned-decode formula
    /// only at `routes = N * 8`); a shared expert with `Is == Im`.
    pub fn qwen3_5_35b_a3b() -> Self {
        Self {
            hidden: 2048,
            num_experts: 256,
            top_k: 8,
            moe_intermediate: 512,
            shared_expert_intermediate: 512,
            norm_variant: NormVariant::Plain,
        }
    }
}

/// Facts for one qwen3_5 GDN (gated-deltanet) linear-attention block — the
/// second traced FRAGMENT, and the other layer kind of the qwen3.5 hybrid.
///
/// Exactly the unit the hybrid composes on a `Linear` layer —
/// `y += gdn(l, rmsnorm(y, attn_norm))` — traced standalone by
/// `family::qwen3_5_gdn_block`, mirroring
/// `qwen3_5_forward.cpp::linear_attn_layer_body` launch for launch.
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
    /// A *binding* fact: the checkpoint ships four raw projections
    /// (`in_proj_{qkv,z,b,a}`) and `gdn_fused_in_proj_joins` re-joins them only
    /// behind `PIE_QWEN35_FUSED_GDN_PROJ` (default OFF), so the default trace
    /// writes four matmuls and the joined one two matmuls + two `SplitGdn`s.
    pub fused_in_proj: bool,
    /// qwen3.5/3.6 fold PLAINLY on the pre-attention norm — see this
    /// generation's module doc. The GATED norm inside the block never read
    /// this field either way: its weight fold is plain by kernel contract
    /// (`rmsnorm.hpp`), which is the one norm here that was RIGHT while the
    /// three around it were not, and `linear_attn.norm.weight` shipping at
    /// 0.88 next to `input_layernorm.weight` at 0.98 is the tell that both
    /// are the same fold.
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
    /// No config.json is committed here, so every dimension is pinned from the
    /// drivers' own statements of this checkpoint:
    ///
    /// * `driver-metal/csrc/src/model/qwen3_5/geometry.hpp` (`DecodeGeometry`
    ///   defaults): `hidden = 1024`, `gdn_k_heads = 16`, `gdn_v_heads = 16`,
    ///   `gdn_k_dim = 128`, `gdn_v_dim = 128`, `gdn_conv_k = 4`,
    ///   `gdn_conv_dim = 6144`, `gdn_v_total = 2048` — `conv_dim()`/`value_width()`
    ///   reproduce the last two (2·2048 + 2048 = 6144, 16·128 = 2048).
    /// * `decode_consts.cpp` corroborates them as launch geometry: in-proj
    ///   1024 → 6144, z 1024 → 2048, out-proj 2048 → 1024, in_proj_a/b a dense
    ///   bf16 GEMV `[16, 1024]` (= value_heads × hidden).
    /// * `driver-cuda/csrc/src/model/config.hpp:357` pins the conv window: 4.
    /// * `fused_in_proj: false` is the live default binding (see the field doc).
    /// * `norm_variant: Plain`. This said `Gemma` on the authority of
    ///   `qwen3_5_forward.cpp` launching `kernels::norm::rmsnorm_gemma`,
    ///   and no 0.8B is staged on this machine to check it against. What IS
    ///   measurable — the two Qwen3.6 checkpoints, and `mlx_lm`'s
    ///   `nn.RMSNorm` for the whole family — says plain, so the fixture
    ///   states plain rather than disagreeing with the row of the same
    ///   checkpoint next door. See this generation's module doc.
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            hidden: 1024,
            key_heads: 16,
            value_heads: 16,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel: 4,
            fused_in_proj: false,
            norm_variant: NormVariant::Plain,
        }
    }
}

/// Facts for one qwen3_5 FULL-attention block — the third traced FRAGMENT,
/// and the last layer kind the qwen3.5 hybrid needed.
///
/// NOT llama_like's attention, which is why it gets its own facts: the q
/// projection is 2× wide with an interleaved per-head `[query | gate]` split
/// (`kernels::layout::split_q_gate_bf16`), the output is multiplied by
/// `sigmoid(gate)` (`sigmoid_gate_inplace_bf16` — no residual, not the
/// shared-expert `SigmoidGateAdd`), rope is PARTIAL, and the per-head q/k norms
/// fold plainly over `N * heads` rows of `head_dim`. The qk-norm is not a
/// tri-state: `full_attn_layer_body` launches the pair unconditionally, so only
/// the fold is a fact.
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
    /// (`fa_qgkv_proj_fused`). A *binding* fact: the join is env-gated
    /// default-OFF (`PIE_QWEN35_FUSED_FULL_ATTN_QGKV`), so the default trace
    /// writes three matmuls; with the join it writes Matmul(qgkv) + SplitQkv
    /// whose "q" leg is the 2×-wide `[query | gate]` bank
    /// (`split_qkv_bf16(packed, qg, k, v, N, 2*Hq, Hk)`).
    pub fused_qkv: bool,
    /// qwen3.5 folds PLAINLY on every norm of this block — the
    /// pre-attention norm AND the per-head q/k norms. Qwen3.6-27B-4bit ships
    /// `layers.3.self_attn.q_norm.weight` at mean 1.22 and `k_norm` at 1.21;
    /// a gemma fold's gain is trained from zero. See this generation's
    /// module doc.
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
    /// own statements of this checkpoint (no config.json is committed; same
    /// provenance discipline as [`Qwen35GdnFacts::qwen3_5_0_8b`]):
    ///
    /// * `driver-metal/csrc/src/model/qwen3_5/geometry.hpp` (`DecodeGeometry`
    ///   defaults): `hidden = 1024`, `n_q_heads = 8`, `n_kv_heads = 2`,
    ///   `head_dim = 256`, `rotary_dims = 64`.
    /// * `decode_consts.cpp` corroborates the widths as launch geometry: the
    ///   2×-wide gated q_proj 4096 = 2·8·256, k/v 1024 → 512 = 2·256,
    ///   o_proj 2048 → 1024.
    /// * `rotary_dim = 64` from `partial_rotary_factor = 0.25`: the CUDA
    ///   derivation `max(2, 2·int(0.5·0.25·256))` and Metal's `rotary_dims`
    ///   both land on 64.
    /// * `fused_qkv: false` is the live default binding (see the field doc).
    /// * `norm_variant: Plain`, for the reason
    ///   [`Qwen35GdnFacts::qwen3_5_0_8b`] gives: the C++ this was pinned
    ///   from is the one source the checkpoints contradict.
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            hidden: 1024,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            fused_qkv: false,
            norm_variant: NormVariant::Plain,
        }
    }
}

/// Which MLP the qwen3_5 hybrid runs on every layer: the dense SwiGLU block
/// (qwen3.5 dense checkpoints — `qwen35_dense_mlp_block`) or the MoE block
/// (qwen3.5/3.6-MoE — `run_moe_mlp`, the [`Qwen35MoeMlpFacts`] fragment).
/// One enum for the whole model: `qwen3_5_forward.cpp` has no per-layer MLP
/// switch, and this family's per-layer axis is the ATTENTION kind. WHICH ARM A
/// CHECKPOINT TAKES IS A READING OF ITS CONFIG, not of its `model_type`:
/// Qwen3.6-27B is `model_type: qwen3_5` and takes `Dense` (no `num_experts`,
/// `intermediate_size` 17408).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Qwen35MlpKind {
    Dense { intermediate: u32 },
    Moe(Qwen35MoeMlpFacts),
}

/// Facts for the full qwen3_5 HYBRID model, composing the three fragments as
/// `match layers[l] { Full => full_attn(l, x), Linear => gdn(l, x) }`, a static
/// match resolved at trace time.
///
/// # How the layer kinds are known
///
/// `config.json` ships a `layer_types` array of `"linear_attention"` /
/// `"full_attention"`, one per layer, which is the CUDA driver's sole source
/// (`HfConfig::layer_types`; `qwen3_5.cpp` refuses a length mismatch). Shipped
/// checkpoints are REGULAR — one full-attention layer every
/// `full_attention_interval` — and the Metal driver reduces the array to that
/// interval, refusing irregular ones. These facts state the interval: an
/// irregular checkpoint is outside this vocabulary as it is outside Metal's.
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
    /// variant inside the sub-facts; qwen3.5 folds PLAINLY everywhere).
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
        model_ir::facts::full_attn_at(self.full_attn_interval, l)
    }

    /// The model's hidden size (the sub-facts each carry it for standalone
    /// tracing; [`qwen3_5_hybrid`] asserts they agree).
    pub fn hidden(&self) -> u32 {
        self.attn.hidden
    }

    /// Qwen3.5-0.8B, the workspace's hybrid parity checkpoint
    /// (`crates/driver-cuda/csrc/tests/parity_qwen3_5_multireq.py` defaults to
    /// `Qwen/Qwen3.5-0.8B-Base`). Sub-facts are the provenance-pinned 0.8B
    /// fixtures; the model-level dims come from `geometry.hpp`'s
    /// `DecodeGeometry` defaults: `n_layers = 24`, `full_attn_interval = 4`
    /// (layers 3, 7, 11, 15, 19, 23 full — the family's 3:1 linear:full
    /// pattern), `vocab = 248320` (lm_head 1024 → 248320),
    /// `tied_embeddings = true`. The MLP is DENSE at `intermediate = 3584`
    /// (gate/up 1024 → 3584, down 3584 → 1024).
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            layers: 24,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: true,
            norm_variant: NormVariant::Plain,
            attn: Qwen35FullAttnFacts::qwen3_5_0_8b(),
            gdn: Qwen35GdnFacts::qwen3_5_0_8b(),
            mlp: Qwen35MlpKind::Dense { intermediate: 3584 },
        }
    }

    /// Qwen3.6-27B — the DENSE hybrid, read from the checkpoint's own
    /// `config.json` (`text_config`), not inferred from the family name.
    ///
    /// Every value is a field of that file or the driver's stated derivation
    /// from one: 64 layers, `full_attention_interval` 4, `vocab_size` 248320,
    /// `tie_word_embeddings` false, `intermediate_size` 17408 (no `num_experts`,
    /// so the `Dense` arm), hidden 5120, 24 q heads over 4 kv heads at
    /// `head_dim` 256, `partial_rotary_factor` 0.25 → `rotary_dim` 64 by
    /// `max(2, 2 * int(0.5 * f * head_dim))`. The GDN half: 16 key heads, 48
    /// value heads (GQA ratio 3, which `family.rs`'s gdn body branches on),
    /// 128/128 head dims, `linear_conv_kernel_dim` 4. Both fusion joins are
    /// env-gated default-off, as at 0.8B.
    ///
    /// NOT reachable on an L40S at bf16 — 27B is ~55 GB against 46. An FP8
    /// checkpoint of the same geometry would boot; the traced form is identical
    /// either way, which is why the fixture predates the hardware.
    pub fn qwen3_6_27b() -> Self {
        Self {
            layers: 64,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Plain,
            attn: Qwen35FullAttnFacts {
                hidden: 5120,
                q_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                rotary_dim: 64,
                fused_qkv: false,
                norm_variant: NormVariant::Plain,
            },
            gdn: Qwen35GdnFacts {
                hidden: 5120,
                key_heads: 16,
                value_heads: 48,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel: 4,
                fused_in_proj: false,
                norm_variant: NormVariant::Plain,
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
        assert!(matches!(
            f.mlp,
            Qwen35MlpKind::Dense {
                intermediate: 17_408
            }
        ));
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
        assert_eq!(m.norm_variant, NormVariant::Plain);
    }
}
