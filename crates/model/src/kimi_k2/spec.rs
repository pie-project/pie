use serde::{Deserialize, Serialize};

pub type KimiMlaFacts = model_ir::facts::MlaFacts;

pub type KimiMoeFacts = model_ir::facts::MoeFacts;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiFacts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub dense_intermediate: u32,

    pub dense_layers: u32,
    pub attn: KimiMlaFacts,
    pub moe: KimiMoeFacts,
}

impl KimiFacts {

    #[must_use]
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_ir::facts::after_dense_prefix(self.dense_layers, l)
    }

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

                output_gate: false,
            },
            moe: KimiMoeFacts {
                num_experts: 384,
                top_k: 8,

                norm_topk_prob: false,

                routed_scaling: 2.0,
                moe_intermediate: 2048,
                shared_intermediate: 2048,
            },
        }
    }
}
