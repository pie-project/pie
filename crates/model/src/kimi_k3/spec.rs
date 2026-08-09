//! Kimi K3's SHAPE, ungated.
//!
//! A HYBRID, like qwen3_5: some layers are MLA full attention, the rest
//! are KDA linear attention. Two things beside that are this
//! generation's own — an attention-residual BLOCK blend that spans
//! layers, and SITU where every other generation has swiglu.
//!
//! # Why this is not under `forward`
//!
//! It was, and that is the defect the split fixes. A catalog row is a
//! `const` written in these words, and a row has to exist under EVERY
//! aspect: a `--features chat` build answers "which template speaks for
//! kimi-k3" and a `--features contract` build authors its tensors, and
//! neither compiles the forward text. With the shape gated behind
//! `forward`, the row that names it could not be stated at all in those
//! builds — so the shape is here, ungated, and `forward/facts.rs` keeps
//! the BACKEND facts, which are the per-layer `Vec`s a tracer builds and
//! nothing else can hold.

use serde::{Deserialize, Serialize};

/// This generation's MLA geometry IS the shared one — see
/// [`model_compiler::facts::MlaFacts`]. Three generations carried
/// field-identical copies of it; the alias keeps every existing spelling
/// working while there is only one definition to disagree with.
pub type KimiK3MlaFacts = model_compiler::facts::MlaFacts;

/// The KDA half — Kimi Delta Attention: a per-KEY-CHANNEL decay, which is
/// what separates it from qwen3_5's GDN (a per-head scalar).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiK3KdaFacts {
    pub value_heads: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
    /// `cfg.kda_gate_lower_bound`, the decay's floor.
    pub gate_lower_bound_milli: u32,
}

impl KimiK3KdaFacts {
    /// The width every one of q, k, v and the gate is projected to.
    ///
    /// `const` because a row is: a `VARIANTS` entry is evaluated at
    /// compile time and cannot call a method that is not.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.value_heads * self.value_head_dim
    }
}

/// This generation's mixture IS the shared one — see
/// [`model_compiler::facts::MoeFacts`]. Three generations carried
/// field-identical copies; the alias keeps every spelling working while
/// there is one definition.
pub type KimiK3MoeFacts = model_compiler::facts::MoeFacts;

/// The whole generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiK3Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub dense_intermediate: u32,
    pub dense_layers: u32,
    /// Every `full_attn_interval`-th layer is MLA; the rest are KDA.
    pub full_attn_interval: u32,
    /// `cfg.attn_res_block_size`: the attention-residual block spans this
    /// many layers. Zero disables the blend entirely.
    pub attn_res_block: u32,
    pub attn: KimiK3MlaFacts,
    pub kda: KimiK3KdaFacts,
    pub moe: KimiK3MoeFacts,
}

impl KimiK3Facts {
    #[must_use]
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_compiler::facts::after_dense_prefix(self.dense_layers, l)
    }

    /// MLA or KDA. The hybrid's schedule, said once.
    #[must_use]
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_compiler::facts::full_attn_at(self.full_attn_interval, l)
    }

    /// Whether layer `l` opens with the attention-residual blend.
    ///
    /// A RULE rather than a per-layer list, and it has to be: a catalog
    /// row is a `const`, a `Vec<u32>` of block openers cannot be one, and
    /// the schedule is periodic anyway. Layer 0 opens the first block, so
    /// there is nothing accumulated to blend back in yet — which is the
    /// `l > 0` here and the same guard `forward::kimi_k3_cuda` states.
    #[must_use]
    pub const fn blends_attn_residual(&self, l: u32) -> bool {
        self.attn_res_block > 0 && l > 0 && l.is_multiple_of(self.attn_res_block)
    }

    pub fn kimi_k3_synthetic() -> Self {
        KimiK3Facts {
            layers: 8,
            vocab: 163840,
            hidden: 2048,
            dense_intermediate: 5632,
            dense_layers: 1,
            full_attn_interval: 4,
            attn_res_block: 4,
            attn: KimiK3MlaFacts {
                hidden: 2048,
                heads: 16,
                q_lora_rank: 768,
                kv_lora_rank: 256,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                // See `forward::kimi_k3_cuda`: the gate is refused, not
                // approximated, so the fixture states the shape the text
                // can actually declare.
                output_gate: false,
            },
            kda: KimiK3KdaFacts {
                value_heads: 16,
                value_head_dim: 128,
                conv_kernel: 4,
                gate_lower_bound_milli: 0,
            },
            moe: KimiK3MoeFacts {
                num_experts: 64,
                top_k: 6,
                // INHERITED from K2, which publishes `false`; no K3 config is
                // out to measure. Stated rather than defaulted so the day
                // one is, this is the line that gets corrected.
                norm_topk_prob: false,
                // Inherited from K2, as above.
                routed_scaling: 2.0,
                moe_intermediate: 1024,
                shared_intermediate: 1024,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k3() -> KimiK3Facts {
        KimiK3Facts::kimi_k3_synthetic()
    }

    #[test]
    fn the_hybrid_schedule_is_every_interval_th_layer() {
        let f = k3();
        assert!(!f.is_full_attn(0));
        assert!(!f.is_full_attn(2));
        assert!(f.is_full_attn(3));
        assert!(f.is_full_attn(7));
        // Which makes the KDA layers the majority, as the generation
        // intends.
        let full = (0..f.layers).filter(|l| f.is_full_attn(*l)).count();
        assert_eq!(full, 2);
    }

    /// An interval of zero means NO full-attention layer, which is the
    /// disagreement [`model_compiler::facts::full_attn_at`] was extracted
    /// to settle: this generation read a zero as "none" and two others
    /// read it as "every layer". A stack that answered "every" here would
    /// trace MLA in a layer whose checkpoint ships KDA weights.
    #[test]
    fn a_stack_with_no_interval_schedules_no_full_attention() {
        let mut f = k3();
        f.full_attn_interval = 0;
        assert!((0..f.layers).all(|l| !f.is_full_attn(l)));
    }

    /// The dense PREFIX and the mixture are the same statement read from
    /// its two ends, so the boundary is exactly `dense_layers`.
    #[test]
    fn the_dense_prefix_ends_where_the_mixture_begins() {
        let f = k3();
        assert!(!f.is_moe_layer(0), "the leading layer is the dense one");
        assert!(f.is_moe_layer(1));
        assert!((f.dense_layers..f.layers).all(|l| f.is_moe_layer(l)));
    }

    /// The blend opens a block and never the first one. A stack that
    /// blended at layer 0 would fold an unaccumulated residual — the
    /// embedding — back into itself.
    #[test]
    fn the_attention_residual_blend_opens_every_block_but_the_first() {
        let f = k3();
        assert!(!f.blends_attn_residual(0), "nothing has accumulated yet");
        assert!(!f.blends_attn_residual(3));
        assert!(f.blends_attn_residual(4));
        assert!(!f.blends_attn_residual(5));
        let openers = (0..f.layers).filter(|l| f.blends_attn_residual(*l)).count();
        assert_eq!(openers, 1, "eight layers in blocks of four opens one seam");
    }

    /// A zero block size disables the blend, which is what a config that
    /// states no `attn_res_block_size` says. It must not divide by it.
    #[test]
    fn a_stack_with_no_block_size_blends_nothing() {
        let mut f = k3();
        f.attn_res_block = 0;
        assert!((0..f.layers).all(|l| !f.blends_attn_residual(l)));
    }

    /// Every fixture states a stack that could exist: positive dims, and
    /// heads that divide the widths they are cut from.
    #[test]
    fn the_fixture_states_a_stack_that_could_exist() {
        let f = k3();
        for (what, n) in [
            ("layers", f.layers),
            ("vocab", f.vocab),
            ("hidden", f.hidden),
            ("dense_intermediate", f.dense_intermediate),
            ("heads", f.attn.heads),
            ("kv_lora_rank", f.attn.kv_lora_rank),
            ("value_heads", f.kda.value_heads),
            ("conv_kernel", f.kda.conv_kernel),
        ] {
            assert!(n > 0, "{what} is zero, which is a stack nothing can fire");
        }
        assert!(
            f.dense_layers < f.layers,
            "an all-dense stack is not a mixture"
        );
        assert_eq!(
            f.kda.width() % f.kda.value_heads,
            0,
            "the KDA projections must split evenly into heads",
        );
        assert_eq!(f.kda.width(), 2048, "16 heads of 128 channels");
    }

    /// A mixture states all of its numbers or none of them. Half a
    /// mixture — experts with no k, or a k with no width — is a router
    /// that dispatches into nothing.
    #[test]
    fn a_mixture_states_all_three_of_its_numbers_or_none() {
        let m = &k3().moe;
        let stated = [m.num_experts, m.top_k, m.moe_intermediate];
        assert!(
            stated.iter().all(|n| *n > 0) || stated.iter().all(|n| *n == 0),
            "a mixture missing one of its three numbers cannot be dispatched",
        );
        assert!(
            m.top_k <= m.num_experts,
            "a row cannot route to more experts than exist"
        );
        assert!(
            m.has_shared_expert(),
            "this generation rides a shared expert beside the routed"
        );
    }

    /// MLA's two widths are independent, and the fixture keeps them so.
    /// The CACHE row is `kv_lora_rank + qk_rope_head_dim`; the DOT is
    /// `qk_nope_head_dim + qk_rope_head_dim`. A shape where those
    /// coincide would hide every place the two are confused.
    #[test]
    fn the_stored_row_and_the_multiplied_row_are_different_numbers() {
        let a = &k3().attn;
        assert_eq!(
            a.kv_a_width(),
            320,
            "the latent plus the one shared rope half"
        );
        assert_eq!(a.qk_head_dim(), 192, "the nope half plus the rope half");
        assert_ne!(a.kv_a_width(), a.qk_head_dim());
        assert_eq!(a.q_b_width(), 16 * 192);
        assert_eq!(a.v_width(), 16 * 128);
    }

    /// The gate this generation's checkpoint ships and its text cannot
    /// yet state. The fixture says `false` because a golden is a trace of
    /// the text as written — see `forward::kimi_k3_cuda`, which asserts
    /// on it — and the ROW is where the model's own answer lives.
    #[test]
    fn the_fixture_states_the_ungated_mla_its_text_can_declare() {
        assert!(
            !k3().attn.output_gate,
            "the traced fixture must state the shape the text declares, or every \
             golden for this generation is a trace of a panic",
        );
    }
}
