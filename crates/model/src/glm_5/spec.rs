use serde::{Deserialize, Serialize};

pub type Glm5MlaFacts = model_ir::facts::MlaFacts;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Glm5DsaFacts {
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Glm5MoeFacts {
    pub hidden: u32,
    pub num_experts: u32,
    pub top_k: u32,

    pub norm_topk_prob: bool,

    pub routed_scaling: f32,
    pub moe_intermediate: u32,

    pub shared_intermediate: u32,

    pub aligned_block: u32,
}

impl Glm5MoeFacts {

    #[must_use]
    pub const fn has_shared_expert(&self) -> bool {
        self.shared_intermediate > 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Glm5Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,

    pub dense_intermediate: u32,

    pub dense_layers: u32,
    pub attn: Glm5MlaFacts,
    pub dsa: Glm5DsaFacts,
    pub moe: Glm5MoeFacts,
}

impl Glm5Facts {

    #[must_use]
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_ir::facts::after_dense_prefix(self.dense_layers, l)
    }
}

impl Glm5Facts {

    pub fn glm5_106b_a12b() -> Self {
        Glm5Facts {
            layers: 46,
            vocab: 151552,
            hidden: 4096,
            dense_intermediate: 10944,

            dense_layers: 3,
            attn: Glm5MlaFacts {
                hidden: 4096,
                heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,

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
