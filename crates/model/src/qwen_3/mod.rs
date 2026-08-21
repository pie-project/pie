#[cfg(feature = "contract")]
pub mod import;

#[cfg(feature = "contract")]
pub mod import_moe;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;
use crate::shared::llama_like::project;
use crate::shared::llama_like::spec::LlamaLikeFacts;

use model_ir::facts::{NormPlacement, QkNorm};
use model_ir::trace::{NormVariant, RopeKind};

pub struct Qwen3 {

    pub id: &'static str,

    pub shape: LlamaLikeFacts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub window: i32,
}

const ARCH: &str = "qwen3";

const MAX_MODEL_LEN: u32 = 40_960;

pub const VARIANTS: &[Qwen3] = &[

    Qwen3 {
        id: "qwen3-0.6b",
        shape: LlamaLikeFacts {
            hidden: 1024,
            layers: 28,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
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
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Qwen3 {
        id: "qwen3-1.7b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 28,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 6144,
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
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Qwen3 {
        id: "qwen3-4b",
        shape: LlamaLikeFacts {
            hidden: 2560,
            layers: 36,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 9728,
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
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Qwen3 {
        id: "qwen3-8b",
        shape: LlamaLikeFacts {
            hidden: 4096,
            layers: 36,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 12_288,
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
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Qwen3 {
        id: "qwen3-14b",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 40,
            q_heads: 40,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 17_408,
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
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Qwen3 {
        id: "qwen3-32b",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 64,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 25_600,
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
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Qwen3 {
        id: "qwen3-30b-a3b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 48,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,
            shared_intermediate: 0,

            intermediate: 0,
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
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Qwen3 {
        id: "qwen3-235b-a22b",
        shape: LlamaLikeFacts {
            hidden: 4096,
            layers: 94,
            q_heads: 64,
            kv_heads: 4,
            head_dim: 128,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 1536,
            shared_intermediate: 0,
            intermediate: 12_288,
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
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
];

crate::rows_of!(Qwen3);

impl Qwen3 {

    fn row(&self) -> project::RowScalars {
        project::RowScalars {
            rope_theta: self.rope_theta,
            norm_eps: self.norm_eps,
            window: self.window,
            rope_rescaled: false,

            norm_topk_prob: true,
        }
    }
}

impl Variant for Qwen3 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {
        if self.shape.n_experts == 0 {
            LoadShape::dense(
                self.shape.layers,
                self.shape.head_dim,
                self.shape.tied_embeddings,
            )
        } else {
            LoadShape::mixture(
                self.shape.layers,
                self.shape.head_dim,
                self.shape.n_experts,
                self.shape.tied_embeddings,
            )
        }
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment = project::deployment(&self.shape, self.row());
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,

            media_encode: false,
        };
        Ok(deployment)
    }

    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => {
                crate::shared::llama_like::contract::author_llama_like(builder)
            }

            crate::shared::policy::Naming::Mlx => {
                crate::shared::llama_like::contract::author_llama_mlx(builder)
            }
        }
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, self.row(), class, load)
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::chatml::QwenInstruct::new(
            tokenizer,
            crate::shared::chatml::QWEN_CHATML,
        ))
    }
}
