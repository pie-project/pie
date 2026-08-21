#[cfg(feature = "chat")]
pub mod chat;

pub mod project;

#[cfg(feature = "contract")]
pub mod import;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;
use crate::shared::llama_like::spec::LlamaLikeFacts;

use self::project::Schedule;

use model_ir::facts::{NormPlacement, QkNorm};
use model_ir::trace::{NormVariant, RopeKind};

const NORM_EPS: f32 = 1e-6;

const ARCH: &str = "gemma3";

pub struct Gemma3 {

    pub id: &'static str,

    pub shape: LlamaLikeFacts,

    pub schedule: Schedule,

    pub max_model_len: u32,
}

const fn gemma_3_schedule(sliding_window: i32, query_pre_attn_scalar: u32) -> Schedule {
    Schedule {
        sliding_window,
        full_attn_interval: 6,
        rope_theta_local: 10_000.0,
        rope_theta_global: 1_000_000.0,
        query_pre_attn_scalar,
    }
}

const fn gemma_3_shape(
    hidden: u32,
    layers: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    vocab: u32,
) -> LlamaLikeFacts {
    LlamaLikeFacts {
        hidden,
        layers,
        q_heads,
        kv_heads,
        head_dim,

        n_experts: 0,
        experts_per_token: 0,
        moe_intermediate: 0,
        shared_intermediate: 0,
        intermediate,
        vocab,
        rope: RopeKind::Standard,
        norm_variant: NormVariant::Gemma,
        norm_placement: NormPlacement::Sandwich,
        qk_norm: QkNorm::PerHead,
        fused_qkv: true,
        tied_embeddings: true,
        qkv_bias: false,
        o_bias: false,
        router_bias: false,
    }
}

pub const VARIANTS: &[Gemma3] = &[

    Gemma3 {
        id: "gemma-3-1b",
        shape: gemma_3_shape(1152, 26, 4, 1, 256, 6912, 262_144),
        schedule: gemma_3_schedule(512, 256),

        max_model_len: 32_768,
    },

    Gemma3 {
        id: "gemma-3-4b",
        shape: gemma_3_shape(2560, 34, 8, 4, 256, 10_240, 262_208),
        schedule: gemma_3_schedule(1024, 256),

        max_model_len: 131_072,
    },

    Gemma3 {
        id: "gemma-3-12b",
        shape: gemma_3_shape(3840, 48, 16, 8, 256, 15_360, 262_208),
        schedule: gemma_3_schedule(1024, 256),
        max_model_len: 131_072,
    },

    Gemma3 {
        id: "gemma-3-27b",
        shape: gemma_3_shape(5376, 62, 32, 16, 128, 21_504, 262_208),
        schedule: gemma_3_schedule(1024, 168),
        max_model_len: 131_072,
    },

    Gemma3 {
        id: "embeddinggemma-300m",
        shape: gemma_3_shape(768, 24, 3, 1, 256, 1152, 262_144),
        schedule: gemma_3_schedule(512, 256),

        max_model_len: 2_048,
    },
];

crate::rows_of!(Gemma3);

impl Variant for Gemma3 {
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
        let mut deployment = project::deployment(&self.shape, &self.schedule, NORM_EPS);
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,

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
                crate::shared::llama_like::contract::author_dense(builder)
            }

            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "gemma-3: no MLX authoring pass exists for this generation, \
                 so there is no name layout to author against",
            ),
        }
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, &self.schedule, NORM_EPS, class, load)
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(self::chat::Gemma3Instruct::for_variant(
            tokenizer,
            self::chat::Gemma3Variant::Gemma3Text,
        ))
    }
}
