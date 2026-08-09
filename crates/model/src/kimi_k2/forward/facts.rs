//! kimi's shape.
//!
//! MLA like [`crate::glm5`], and the second family to carry it — but
//! without glm5's DSA indexer and with WNA16 experts instead of bf16
//! ones, which changes the decode MoE leg's kernels rather than its
//! shape.

use serde::{Deserialize, Serialize};

/// kimi's MLA dims. Same vocabulary glm5's uses; see
/// [`crate::glm5::forward::facts::Glm5MlaFacts`] for why `kv_a_width` is
/// its own number.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiMlaFacts {
    pub hidden: u32,
    pub heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
}

impl KimiMlaFacts {
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    pub fn q_b_width(&self) -> u32 {
        self.heads * self.qk_head_dim()
    }
    pub fn kv_a_width(&self) -> u32 {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
    pub fn v_width(&self) -> u32 {
        self.heads * self.v_head_dim
    }
    /// The FUSED `q_kv_a` projection's width: `[q_lora | kv_lora | rope]`
    /// in one row-major buffer, which is why both consumers take a pitch.
    pub fn q_kv_a_width(&self) -> u32 {
        self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
    }
}

/// The MoE block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiMoeFacts {
    pub num_experts: u32,
    pub top_k: u32,
    pub moe_intermediate: u32,
    /// `shared_expert_intermediate_size`; zero means no shared expert.
    pub shared_intermediate: u32,
}

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
