use serde::{Deserialize, Serialize};

pub type KimiK3MlaFacts = model_ir::facts::MlaFacts;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiK3KdaFacts {
    pub value_heads: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,

    pub gate_lower_bound_milli: u32,

    pub norm_eps_micro: u32,
}

impl KimiK3KdaFacts {

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.value_heads * self.value_head_dim
    }

    #[must_use]
    pub fn norm_eps(&self) -> f32 {
        self.norm_eps_micro as f32 / 1.0e6
    }
}

pub type KimiK3MoeFacts = model_ir::facts::MoeFacts;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiK3Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub dense_intermediate: u32,
    pub dense_layers: u32,

    pub full_attn_interval: u32,

    pub attn_res_block: u32,
    pub attn: KimiK3MlaFacts,
    pub kda: KimiK3KdaFacts,
    pub moe: KimiK3MoeFacts,
}

impl KimiK3Facts {

    #[must_use]
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_ir::facts::after_dense_prefix(self.dense_layers, l)
    }

    #[must_use]
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_ir::facts::full_attn_at(self.full_attn_interval, l)
    }

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

                output_gate: false,
            },
            kda: KimiK3KdaFacts {
                value_heads: 16,
                value_head_dim: 128,
                conv_kernel: 4,
                gate_lower_bound_milli: 0,
                norm_eps_micro: 10,
            },
            moe: KimiK3MoeFacts {
                num_experts: 64,
                top_k: 6,

                norm_topk_prob: false,

                routed_scaling: 2.0,
                moe_intermediate: 1024,
                shared_intermediate: 1024,
            },
        }
    }
}
