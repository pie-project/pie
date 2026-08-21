use serde::{Deserialize, Serialize};

pub use model_ir::facts::{NormPlacement, QkNorm};
use model_ir::trace::{NormVariant, RopeKind};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    #[serde(default)]
    pub n_experts: u32,

    #[serde(default)]
    pub experts_per_token: u32,

    #[serde(default)]
    pub moe_intermediate: u32,

    #[serde(default)]
    pub shared_intermediate: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub rope: RopeKind,
    pub norm_variant: NormVariant,

    #[serde(default)]
    pub norm_placement: NormPlacement,

    pub qk_norm: QkNorm,

    pub fused_qkv: bool,

    pub tied_embeddings: bool,

    #[serde(default)]
    pub qkv_bias: bool,

    #[serde(default)]
    pub o_bias: bool,

    #[serde(default)]
    pub router_bias: bool,
}

impl LlamaLikeFacts {

    pub fn shape(&self, norm_eps: f32) -> model_dsl::ModelShape {
        model_dsl::ModelShape {
            hidden: self.hidden,
            intermediate: self.intermediate,
            n_experts: self.n_experts,
            moe_intermediate: self.moe_intermediate,
            shared_intermediate: self.shared_intermediate,
            vocab: self.vocab,
            head_dim: self.head_dim,
            q_width: self.q_width(),
            kv_width: self.kv_width(),
            qk_norm: self.qk_norm,
            norm_variant: self.norm_variant,

            norm_eps_micro: (norm_eps * 1.0e6).round() as u32,
            tied_embeddings: self.tied_embeddings,

            proj_repr: model_dsl::WeightRepr::Bf16,
        }
    }

    pub fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    pub fn qwen2_5_1_5b() -> Self {
        Self {

            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            hidden: 1536,
            layers: 28,
            q_heads: 12,
            kv_heads: 2,
            head_dim: 128,
            intermediate: 8960,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: true,
            o_bias: false,
            router_bias: false,
        }
    }

    pub fn llama_3_2_1b() -> Self {
        Self {

            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            hidden: 2048,
            layers: 16,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 8192,
            vocab: 128_256,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: false,
            tied_embeddings: true,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        }
    }

    pub fn qwen3_0_6b() -> Self {
        Self {

            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            hidden: 1024,
            layers: 28,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 3072,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        }
    }

    pub fn qwen3_30b_a3b() -> Self {
        Self {
            hidden: 2048,
            layers: 48,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,

            intermediate: 0,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,

            shared_intermediate: 0,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,

            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        }
    }

    pub fn gpt_oss_20b() -> Self {
        Self {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,

            intermediate: 0,
            n_experts: 32,
            experts_per_token: 4,
            moe_intermediate: 2880,

            shared_intermediate: 0,
            vocab: 201_088,

            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,

            qk_norm: QkNorm::Off,
            fused_qkv: false,
            tied_embeddings: false,

            qkv_bias: true,
            o_bias: true,
            router_bias: true,
        }
    }

    pub fn phi3_mini() -> Self {
        Self {

            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            hidden: 3072,
            layers: 32,
            q_heads: 32,
            kv_heads: 32,
            head_dim: 96,
            intermediate: 8192,
            vocab: 32_064,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        }
    }

    pub fn mistral_7b_v03() -> Self {
        Self {

            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 14_336,
            vocab: 32_768,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        }
    }

    pub fn olmo2_1b() -> Self {
        Self {

            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            hidden: 2048,
            layers: 16,
            q_heads: 16,
            kv_heads: 16,
            head_dim: 128,
            intermediate: 8192,
            vocab: 100_352,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Post,
            qk_norm: QkNorm::Global,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        }
    }
}
