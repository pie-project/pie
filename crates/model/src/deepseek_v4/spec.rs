use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dsv4AttnFacts {
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,

    pub q_lora_rank: u32,
    pub qk_rope_head_dim: u32,

    pub sliding_window: u32,

    pub o_lora_rank: u32,
    pub o_groups: u32,
}

impl Dsv4AttnFacts {

    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.heads * self.head_dim
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dsv4HcFacts {

    pub mult: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dsv4MoeFacts {
    pub num_experts: u32,
    pub top_k: u32,

    pub norm_topk_prob: bool,

    pub routed_scaling: f32,
    pub moe_intermediate: u32,

    pub swiglu_limit_milli: u32,

    pub hash_routed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Dsv4Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub dense_intermediate: u32,
    pub dense_layers: u32,

    pub ratios: &'static [i32],
    pub attn: Dsv4AttnFacts,
    pub hc: Dsv4HcFacts,
    pub moe: Dsv4MoeFacts,
}

impl Dsv4Facts {

    #[must_use]
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_ir::facts::after_dense_prefix(self.dense_layers, l)
    }

    #[must_use]
    pub fn compress_ratio_at(&self, l: u32) -> i32 {
        self.ratios.get(l as usize).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn compresses(&self, l: u32) -> bool {
        self.compress_ratio_at(l) > 0
    }

    pub fn dsv4_synthetic() -> Self {
        Dsv4Facts {
            layers: 6,
            vocab: 129280,
            hidden: 2048,
            dense_intermediate: 5632,
            dense_layers: 1,

            ratios: &[1, 2, 4],
            attn: Dsv4AttnFacts {
                hidden: 2048,
                heads: 16,
                head_dim: 128,
                q_lora_rank: 768,
                qk_rope_head_dim: 64,
                sliding_window: 2048,
                o_lora_rank: 512,
                o_groups: 4,
            },
            hc: Dsv4HcFacts { mult: 4 },
            moe: Dsv4MoeFacts {
                num_experts: 64,
                top_k: 6,
                norm_topk_prob: false,
                routed_scaling: 2.5,
                moe_intermediate: 1024,
                swiglu_limit_milli: 7000,
                hash_routed: false,
            },
        }
    }
}
