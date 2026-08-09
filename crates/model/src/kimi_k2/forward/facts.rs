//! kimi's shape.
//!
//! MLA like [`crate::glm5`], and the second family to carry it — but
//! without glm5's DSA indexer and with WNA16 experts instead of bf16
//! ones, which changes the decode MoE leg's kernels rather than its
//! shape.

use serde::{Deserialize, Serialize};

/// This family's MLA geometry IS the shared one — see
/// [`model_compiler::facts::MlaFacts`]. Three families carried
/// field-identical copies of it; the alias keeps every existing spelling
/// working while there is only one definition to disagree with.
pub type KimiMlaFacts = model_compiler::facts::MlaFacts;



/// This family's mixture IS the shared one — see
/// [`model_compiler::facts::MoeFacts`]. Three families carried
/// field-identical copies; the alias keeps every spelling working while
/// there is one definition.
pub type KimiMoeFacts = model_compiler::facts::MoeFacts;


/// The CUDA reading's deployment facts — the choices the hand-written
/// pass makes from the BINDING and the config, resolved once at load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiCudaFacts {
    /// `Lw.q_kv_a_fused != nullptr`: one GEMM for the query's and the
    /// KV's latents instead of two, and a STRIDED norm over the query
    /// half — neither latent is a contiguous block of the result.
    pub q_kv_a_fused: bool,
    /// The config asks for YaRN (`rope_scaling_kind`), so the rope is
    /// `kernels::rope::rope_yarn_original_bf16` rather than the plain one.
    pub rope_yarn_original: bool,
}

/// The whole family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_compiler::facts::after_dense_prefix(self.dense_layers, l)
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
                moe_intermediate: 2048,
                shared_intermediate: 2048,
            },
        }
    }
}

impl KimiCudaFacts {
    pub fn kimi_k2_synthetic() -> Self {
        KimiCudaFacts {
            q_kv_a_fused: true,
            rope_yarn_original: true,
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
}
