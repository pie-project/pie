//! Kimi K2's SHAPE: the numbers a checkpoint of this generation has.
//!
//! Ungated, for the reason `shared/llama_like/spec.rs` states: a
//! catalog row is written in these words, and a row must exist under
//! every aspect — `chat` asks which template speaks for it, `contract`
//! asks how to author it, `forward` asks what to trace. One struct,
//! three readers. That cannot hold if the struct only exists when the
//! tracer is compiled in.
//!
//! What stayed behind in `forward/facts.rs` is [`KimiCudaFacts`], which
//! names what a LOAD bound rather than what the model is.
//!
//! [`KimiCudaFacts`]: super::forward::facts::KimiCudaFacts

use serde::{Deserialize, Serialize};

/// This family's MLA geometry IS the shared one — see
/// [`model_ir::facts::MlaFacts`]. Three families carried
/// field-identical copies of it; the alias keeps every existing spelling
/// working while there is only one definition to disagree with.
pub type KimiMlaFacts = model_ir::facts::MlaFacts;

/// This family's mixture IS the shared one — see
/// [`model_ir::facts::MoeFacts`]. Three families carried
/// field-identical copies; the alias keeps every spelling working while
/// there is one definition.
pub type KimiMoeFacts = model_ir::facts::MoeFacts;

/// The whole family.
///
/// No `Vec` in it, which is what makes a row `const`-constructible: the
/// one thing that reads like a per-layer list here — which layers take
/// the dense MLP — is a RULE (`dense_layers`, a prefix length) and was
/// never a table. A stack that wrote the prefix out longhand would be
/// stating 61 booleans where `first_k_dense_replace` states one number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiFacts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub dense_intermediate: u32,
    /// The leading layers that take the dense MLP instead of the MoE.
    pub dense_layers: u32,
    pub attn: KimiMlaFacts,
    pub moe: KimiMoeFacts,
}

impl KimiFacts {
    /// Whether layer `l` takes the routed MLP.
    ///
    /// The shared predicate, named locally. Four families spell this
    /// call and the LOGIC is already shared -- `after_dense_prefix`
    /// lives in `model_ir::facts` for exactly that reason, and its
    /// own doc explains why it is named for the PREFIX rather than for
    /// the mixture. What is left here is a one-line projection onto this
    /// family's own field, which is cheaper than the trait that would
    /// remove it: a required accessor plus a default body is more
    /// machinery than the line it replaces.
    #[must_use]
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_ir::facts::after_dense_prefix(self.dense_layers, l)
    }

    /// `moonshotai/Kimi-K2-Instruct`, read off its config the way the
    /// driver reads it.
    pub fn kimi_k2() -> Self {
        KimiFacts {
            layers: 61,
            vocab: 163840,
            hidden: 7168,
            dense_intermediate: 18432,
            dense_layers: 1,
            attn: KimiMlaFacts {
                hidden: 7168,
                heads: 64,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                // This family does not gate the MLA output; kimi-k3 does.
                output_gate: false,
            },
            moe: KimiMoeFacts {
                num_experts: 384,
                top_k: 8,
                // `moonshotai/Kimi-K2-Instruct` publishes `"norm_topk_prob":
                // false` -- this lineage routes on weights that sum to
                // less than one, and its `routed_scaling_factor` of 2.0
                // is the compensation.
                norm_topk_prob: false,
                // K2 publishes 2.0.
                routed_scaling: 2.0,
                moe_intermediate: 2048,
                shared_intermediate: 2048,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fused_latent_width_is_the_three_halves() {
        let f = KimiFacts::kimi_k2();
        assert_eq!(f.attn.q_kv_a_width(), 1536 + 512 + 64);
        // And the SPLIT binding's two widths sum to the same thing, which
        // is the only reason one GEMM can stand in for two.
        assert_eq!(
            f.attn.q_lora_rank + f.attn.kv_a_width(),
            f.attn.q_kv_a_width()
        );
    }

    /// The dense prefix is a RULE, and this is the whole of it: the
    /// leading `dense_layers` layers take the dense MLP and every layer
    /// after them takes the mixture. Stated as a length rather than as a
    /// list, which is why the shape holds no `Vec` and a row can be
    /// `const`.
    #[test]
    fn the_dense_prefix_is_a_length_and_not_a_table() {
        let f = KimiFacts::kimi_k2();
        assert_eq!(f.dense_layers, 1);
        assert!(!f.is_moe_layer(0), "the one dense leader");
        assert!(f.is_moe_layer(1));
        assert!(f.is_moe_layer(f.layers - 1));
        assert_eq!(
            (0..f.layers).filter(|&l| f.is_moe_layer(l)).count() as u32,
            f.layers - f.dense_layers,
        );
    }

    /// The shape is `const`-constructible: every field is a scalar or a
    /// struct of scalars, so a row can be a `const` in `.rodata` rather
    /// than something built at run time.
    #[test]
    fn the_shape_is_the_measurement_of_a_real_checkpoint() {
        let f = KimiFacts::kimi_k2();
        assert_eq!(f.layers, 61);
        assert_eq!(f.vocab, 163_840);
        assert_eq!(f.hidden, 7168);
        assert_eq!(f.dense_intermediate, 18_432);
        assert_eq!(f.attn.heads, 64);
        assert_eq!(f.moe.num_experts, 384);
        assert_eq!(f.moe.top_k, 8);
        assert!(f.moe.has_shared_expert(), "K2 rides a shared expert");
        assert!(!f.attn.output_gate, "K3 gates the MLA output; K2 does not");
    }
}
