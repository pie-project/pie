use model_ir::trace::NormVariant;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35MoeMlpFacts {
    pub hidden: u32,

    pub num_experts: u32,

    pub top_k: u32,

    pub moe_intermediate: u32,

    pub shared_expert_intermediate: u32,

    pub norm_variant: NormVariant,
}

impl Qwen35MoeMlpFacts {
    pub fn qwen3_5_35b_a3b() -> Self {
        Self {
            hidden: 2048,
            num_experts: 256,
            top_k: 8,
            moe_intermediate: 512,
            shared_expert_intermediate: 512,
            norm_variant: NormVariant::Gemma,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35GdnFacts {
    pub hidden: u32,

    pub key_heads: u32,

    pub value_heads: u32,

    pub key_head_dim: u32,

    pub value_head_dim: u32,

    pub conv_kernel: u32,

    pub fused_in_proj: bool,

    pub norm_variant: NormVariant,
}

impl Qwen35GdnFacts {
    pub fn key_width(&self) -> u32 {
        self.key_heads * self.key_head_dim
    }

    pub fn value_width(&self) -> u32 {
        self.value_heads * self.value_head_dim
    }

    pub fn conv_dim(&self) -> u32 {
        2 * self.key_width() + self.value_width()
    }

    pub fn qwen3_5_0_8b() -> Self {
        Self {
            hidden: 1024,
            key_heads: 16,
            value_heads: 16,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel: 4,
            fused_in_proj: false,
            norm_variant: NormVariant::Gemma,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35FullAttnFacts {
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    pub rotary_dim: u32,

    pub fused_qkv: bool,

    pub norm_variant: NormVariant,
}

impl Qwen35FullAttnFacts {
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    pub fn qwen3_5_0_8b() -> Self {
        Self {
            hidden: 1024,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            fused_qkv: false,
            norm_variant: NormVariant::Gemma,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Qwen35MlpKind {
    Dense { intermediate: u32 },
    Moe(Qwen35MoeMlpFacts),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35HybridFacts {
    pub layers: u32,

    pub full_attn_interval: u32,
    pub vocab: u32,

    pub tied_embeddings: bool,

    pub norm_variant: NormVariant,

    pub attn: Qwen35FullAttnFacts,

    pub gdn: Qwen35GdnFacts,

    pub mlp: Qwen35MlpKind,
}

impl Qwen35HybridFacts {
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_ir::facts::full_attn_at(self.full_attn_interval, l)
    }

    pub fn hidden(&self) -> u32 {
        self.attn.hidden
    }

    pub fn qwen3_5_0_8b() -> Self {
        Self {
            layers: 24,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: true,

            norm_variant: NormVariant::Gemma,
            attn: Qwen35FullAttnFacts::qwen3_5_0_8b(),
            gdn: Qwen35GdnFacts::qwen3_5_0_8b(),
            mlp: Qwen35MlpKind::Dense { intermediate: 3584 },
        }
    }

    pub fn qwen3_6_27b() -> Self {
        Self {
            layers: 64,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Gemma,
            attn: Qwen35FullAttnFacts {
                hidden: 5120,
                q_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                rotary_dim: 64,
                fused_qkv: false,
                norm_variant: NormVariant::Gemma,
            },
            gdn: Qwen35GdnFacts {
                hidden: 5120,
                key_heads: 16,
                value_heads: 48,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel: 4,
                fused_in_proj: false,
                norm_variant: NormVariant::Gemma,
            },
            mlp: Qwen35MlpKind::Dense {
                intermediate: 17_408,
            },
        }
    }
}
