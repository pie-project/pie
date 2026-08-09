//! GLM-5's SHAPE: the numbers a checkpoint of this generation has.
//!
//! Ungated, for the reason `kimi_k2/spec.rs` states: a catalog row is
//! written in these words, and a row must exist under every aspect —
//! `chat` asks which template speaks for it, `contract` asks how to
//! author it, `forward` asks what to trace. One struct, three readers.
//! That cannot hold if the struct only exists when the tracer is
//! compiled in.
//!
//! Read off `driver-cuda/csrc/src/model/glm5/` — `glm5_forward.cpp`'s
//! `cfg.` reads for the dims, `glm5.hpp`'s weight struct for which
//! tensors a layer has, and `Lw.is_moe` for the layer schedule.
//!
//! Two things here are this generation's own and neither llama_like nor
//! qwen3_5 has either: MLA (the query and the KV both go through a
//! LATENT of their own rank, so `hidden` never appears in the attention
//! core's widths) and DSA (a lightning indexer that scores pages and
//! hands attention a top-k mask, which is a SECOND, smaller attention
//! beside the real one).
//!
//! Nothing stayed behind in `forward/facts.rs`: this generation's only
//! binding fact was the shape itself, so that file is now the
//! re-export that keeps `forward`'s spellings working.

use serde::{Deserialize, Serialize};

/// This family's MLA geometry IS the shared one — see
/// [`model_compiler::facts::MlaFacts`]. Three families carried
/// field-identical copies of it; the alias keeps every existing spelling
/// working while there is only one definition to disagree with.
pub type Glm5MlaFacts = model_compiler::facts::MlaFacts;

/// The DSA lightning indexer. A separate small attention whose only
/// output is a top-k page mask for the real one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Glm5DsaFacts {
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
}

/// The MoE block. `first_k_dense` is not a config field in the driver —
/// it reads `Lw.is_moe` per layer — but the schedule it encodes is a
/// prefix of dense layers, so the declaration states the prefix length
/// and a layer asks whether it is past it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Glm5MoeFacts {
    pub hidden: u32,
    pub num_experts: u32,
    pub top_k: u32,
    /// Whether the router renormalizes over the SELECTED experts.
    ///
    /// `zai-org/GLM-4.5` publishes `"norm_topk_prob": true`, which is the
    /// half of the catalog the `DeepseekV3Config` class default gets
    /// wrong -- DeepSeek-V3 and Kimi-K2 publish `false` from the same
    /// sigmoid-plus-bias router. Stated per family for that reason.
    pub norm_topk_prob: bool,
    /// `routed_scaling_factor`, 2.5 on `zai-org/GLM-4.5`.
    pub routed_scaling: f32,
    pub moe_intermediate: u32,
    /// `n_shared_experts * moe_intermediate`; zero means no shared expert.
    pub shared_intermediate: u32,
    /// `kernels::moe::moe_aligned_block(maxR, num_experts)`, resolved once at
    /// workspace setup. A deployment fact rather than a per-fire number,
    /// which is why it can shape a `Dim`.
    pub aligned_block: u32,
}

impl Glm5MoeFacts {
    /// Whether a shared expert rides beside the routed ones. A predicate
    /// rather than a second field, for the reason every derived width in
    /// [`model_compiler::facts`] is a method: a stored answer is a
    /// second thing to keep in step.
    #[must_use]
    pub const fn has_shared_expert(&self) -> bool {
        self.shared_intermediate > 0
    }
}

/// The whole family.
///
/// No `Vec` in it, which is what makes a row `const`-constructible: the
/// one thing that reads like a per-layer list — which layers take the
/// dense MLP — is a RULE (`dense_layers`, a prefix length) and was never
/// a table. A stack that wrote the prefix out longhand would be stating
/// 46 booleans where `first_k_dense_replace` states one number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Glm5Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    /// The DENSE MLP width, used by the first `dense_layers` layers.
    pub dense_intermediate: u32,
    /// How many leading layers take the dense MLP instead of the MoE.
    pub dense_layers: u32,
    pub attn: Glm5MlaFacts,
    pub dsa: Glm5DsaFacts,
    pub moe: Glm5MoeFacts,
}

impl Glm5Facts {
    /// Whether layer `l` takes the routed MLP.
    ///
    /// The shared predicate, named locally. Four families spell this
    /// call and the LOGIC is already shared -- `after_dense_prefix`
    /// lives in `model_compiler::facts` for exactly that reason, and its
    /// own doc explains why it is named for the PREFIX rather than for
    /// the mixture. What is left here is a one-line projection onto this
    /// family's own field, which is cheaper than the trait that would
    /// remove it: a required accessor plus a default body is more
    /// machinery than the line it replaces.
    #[must_use]
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_compiler::facts::after_dense_prefix(self.dense_layers, l)
    }
}

impl Glm5Facts {
    /// `zai-org/GLM-5-106B-A12B`, read off its `config.json` the way the
    /// driver reads it. A fixture rather than a guess: every dim here is
    /// a `cfg.` field `glm5_forward.cpp` actually consumes.
    pub fn glm5_106b_a12b() -> Self {
        Glm5Facts {
            layers: 46,
            vocab: 151552,
            hidden: 4096,
            dense_intermediate: 10944,
            // The first three layers are dense; the rest route.
            dense_layers: 3,
            attn: Glm5MlaFacts {
                hidden: 4096,
                heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                // This family does not gate the MLA output; kimi-k3 does.
                output_gate: false,
            },
            dsa: Glm5DsaFacts {
                index_n_heads: 64,
                index_head_dim: 128,
                index_topk: 2048,
            },
            moe: Glm5MoeFacts {
                hidden: 4096,
                num_experts: 128,
                top_k: 8,
                norm_topk_prob: true,
                routed_scaling: 2.5,
                moe_intermediate: 1408,
                shared_intermediate: 1408,
                aligned_block: 16,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived widths, against the weight shapes `glm5.hpp` documents
    /// in its own comments — the only cross-check available without the
    /// checkpoint, and enough to catch a transposed pair.
    #[test]
    fn the_derived_widths_match_the_weight_comments() {
        let f = Glm5Facts::glm5_106b_a12b();
        // `q_b_proj` is `[local_heads*(nope+rope), q_lora_rank]`.
        assert_eq!(f.attn.qk_head_dim(), 192);
        assert_eq!(f.attn.q_b_width(), 96 * 192);
        // `kv_a_proj_with_mqa` is `[kv_lora_rank+rope, H]` — ONE rope
        // half for every head, which is the MQA in its name.
        assert_eq!(f.attn.kv_a_width(), 512 + 64);
        // `o_proj` is `[H, local_heads*v_dim]`.
        assert_eq!(f.attn.v_width(), 96 * 128);
    }

    #[test]
    fn the_dense_prefix_is_a_prefix() {
        let f = Glm5Facts::glm5_106b_a12b();
        assert!(!f.is_moe_layer(0));
        assert!(!f.is_moe_layer(2));
        assert!(f.is_moe_layer(3));
        assert!(f.is_moe_layer(f.layers - 1));
    }

    /// The dense prefix is a LENGTH and not a table, which is the whole
    /// reason a row of this generation can be `const`: the schedule is
    /// three layers' worth of one number, not 46 booleans.
    #[test]
    fn the_dense_prefix_is_a_length_and_not_a_table() {
        let f = Glm5Facts::glm5_106b_a12b();
        assert_eq!(
            (0..f.layers).filter(|&l| f.is_moe_layer(l)).count() as u32,
            f.layers - f.dense_layers,
            "every layer past the prefix routes, and none before it does",
        );
    }

    /// A stack that could exist: positive extents everywhere, and the
    /// heads divide the widths they are cut from. A row whose
    /// `q_b_width` is not a multiple of `heads` states an attention no
    /// tensor-parallel split can cut without halving a head.
    #[test]
    fn the_fixture_states_a_stack_that_could_exist() {
        let f = Glm5Facts::glm5_106b_a12b();
        assert!(f.layers > 0 && f.vocab > 0 && f.hidden > 0);
        assert!(f.dense_intermediate > 0);
        assert!(f.dense_layers < f.layers, "a prefix, not the whole stack");
        assert!(f.attn.heads > 0);
        assert_eq!(f.attn.q_b_width() % f.attn.heads, 0);
        assert_eq!(f.attn.v_width() % f.attn.heads, 0);
        assert_eq!(
            f.attn.hidden, f.hidden,
            "the attention block reads the stack's own residual width",
        );
        assert_eq!(
            f.moe.hidden, f.hidden,
            "and so does the mixture; two spellings of one number that \
             disagree is a row nobody can shard",
        );
    }

    /// A mixture states all three of its numbers or none. Experts with
    /// no width, or a width with no experts, is a mixture that would
    /// size a workspace to zero and route into it.
    #[test]
    fn a_mixture_states_all_three_of_its_numbers_or_none() {
        let m = Glm5Facts::glm5_106b_a12b().moe;
        assert!(m.num_experts > 0 && m.top_k > 0 && m.moe_intermediate > 0);
        assert!(
            m.top_k <= m.num_experts,
            "routing to more experts than exist is a router that cannot \
             normalise",
        );
        assert!(m.has_shared_expert(), "GLM-5 rides one shared expert");
        assert_eq!(
            m.shared_intermediate, m.moe_intermediate,
            "one shared expert at the routed width, which is what \
             `n_shared_experts * moe_intermediate` reads as at n=1",
        );
        assert!(m.aligned_block > 0);
    }

    /// The indexer is a SECOND attention, and its own dims are what say
    /// so: a top-k of zero would mask every page away and a head width
    /// of zero would score nothing.
    #[test]
    fn the_dsa_indexer_states_a_second_attention() {
        let d = Glm5Facts::glm5_106b_a12b().dsa;
        assert_eq!(d.index_n_heads, 64);
        assert_eq!(d.index_head_dim, 128);
        assert_eq!(d.index_topk, 2048);
        assert!(d.index_n_heads > 0 && d.index_head_dim > 0 && d.index_topk > 0);
    }

    /// The measurement itself, field for field. This is the row's
    /// source: a number that changes here changes a manifest extent and
    /// a deployment, so it is written down twice on purpose.
    #[test]
    fn the_shape_is_the_measurement_of_a_real_checkpoint() {
        let f = Glm5Facts::glm5_106b_a12b();
        assert_eq!(f.layers, 46);
        assert_eq!(f.vocab, 151_552);
        assert_eq!(f.hidden, 4096);
        assert_eq!(f.dense_intermediate, 10_944);
        assert_eq!(f.dense_layers, 3);
        assert_eq!(f.attn.heads, 96);
        assert_eq!(f.attn.q_lora_rank, 1536);
        assert_eq!(f.attn.kv_lora_rank, 512);
        assert_eq!(f.moe.num_experts, 128);
        assert_eq!(f.moe.top_k, 8);
        assert_eq!(f.moe.moe_intermediate, 1408);
        assert!(
            !f.attn.output_gate,
            "kimi-k3 gates the MLA output; this generation does not, and a \
             gate the text does not state is a projection nobody binds",
        );
    }
}
