#[cfg(feature = "chat")]
pub mod chat;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;
use crate::shared::llama_like::project;
use crate::shared::llama_like::spec::LlamaLikeFacts;

use model_ir::facts::{NormPlacement, QkNorm};
use model_ir::trace::{NormVariant, RopeKind};

pub struct Llama3 {

    pub id: &'static str,

    pub shape: LlamaLikeFacts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub window: i32,

    pub rope_factor: f32,
}

const ARCH: &str = "llama";

const ROPE_LOW_FREQ_FACTOR: f32 = 1.0;

const ROPE_HIGH_FREQ_FACTOR: f32 = 4.0;

const ROPE_ORIGINAL_MAX: u32 = 8_192;

const MAX_MODEL_LEN: u32 = 131_072;

pub const VARIANTS: &[Llama3] = &[

    Llama3 {
        id: "llama-3.2-1b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 16,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 64,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 8192,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,

        rope_factor: 32.0,
    },

    Llama3 {
        id: "llama-3.2-3b",
        shape: LlamaLikeFacts {
            hidden: 3072,
            layers: 28,
            q_heads: 24,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 8192,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,

        rope_factor: 32.0,
    },

    Llama3 {
        id: "llama-3.1-8b",
        shape: LlamaLikeFacts {
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 14_336,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,

        rope_factor: 8.0,
    },

    Llama3 {
        id: "llama-3.1-70b",
        shape: LlamaLikeFacts {
            hidden: 8192,
            layers: 80,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 28_672,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,

        rope_factor: 8.0,
    },

    Llama3 {
        id: "llama-3.3-70b",
        shape: LlamaLikeFacts {
            hidden: 8192,
            layers: 80,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 28_672,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,

        rope_factor: 8.0,
    },
];

crate::rows_of!(Llama3);

impl Llama3 {

    fn row(&self) -> project::RowScalars {
        project::RowScalars {
            rope_theta: self.rope_theta,
            norm_eps: self.norm_eps,
            window: self.window,
            rope_rescaled: true,

            norm_topk_prob: true,
        }
    }
}

impl Variant for Llama3 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape::dense(
            self.shape.layers,
            self.shape.head_dim,
            self.shape.tied_embeddings,
        )
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment = project::deployment(&self.shape, self.row());

        deployment.rope_scaling = Some(crate::deployment::RopeScaling::Piecewise {
            factor: self.rope_factor,
            low_freq_factor: ROPE_LOW_FREQ_FACTOR,
            high_freq_factor: ROPE_HIGH_FREQ_FACTOR,
            original_max_position: ROPE_ORIGINAL_MAX,
        });
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
        Arc::new(chat::LlamaInstruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
