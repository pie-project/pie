//! kimi_k3's shape.
//!
//! A HYBRID, like qwen3_5: some layers are MLA full attention, the rest
//! are KDA linear attention. Two things beside that are this family's
//! own — an attention-residual BLOCK blend that spans layers, and SITU
//! where every other family has swiglu.

use serde::{Deserialize, Serialize};

/// The MLA half. Same vocabulary [`crate::kimi`] and [`crate::glm5`] use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiK3MlaFacts {
    pub hidden: u32,
    pub heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    /// `cfg.mla_output_gate`: a sigmoid gate multiplied onto the
    /// attention output before `o_proj`, from its own projection.
    pub output_gate: bool,
}

impl KimiK3MlaFacts {
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
}

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
    pub fn width(&self) -> u32 {
        self.value_heads * self.value_head_dim
    }
}

/// The MoE block. MXFP4 experts, so the decode leg is the MXFP4 GEMV.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiK3MoeFacts {
    pub num_experts: u32,
    pub top_k: u32,
    pub moe_intermediate: u32,
    pub shared_intermediate: u32,
}

/// The whole family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_compiler::facts::after_dense_prefix(self.dense_layers, l)
    }
    /// MLA or KDA. The hybrid's schedule, said once.
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_compiler::facts::full_attn_at(self.full_attn_interval, l)
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
                moe_intermediate: 1024,
                shared_intermediate: 1024,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hybrid_schedule_is_every_interval_th_layer() {
        let f = KimiK3Facts::kimi_k3_synthetic();
        assert!(!f.is_full_attn(0));
        assert!(!f.is_full_attn(2));
        assert!(f.is_full_attn(3));
        assert!(f.is_full_attn(7));
        // Which makes the KDA layers the majority, as the family intends.
        let full = (0..f.layers).filter(|l| f.is_full_attn(*l)).count();
        assert_eq!(full, 2);
    }
}
