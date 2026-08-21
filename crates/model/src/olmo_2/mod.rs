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

pub struct Olmo2 {

    pub id: &'static str,

    pub shape: LlamaLikeFacts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub window: i32,
}

const ARCH: &str = "olmo2";

const MAX_MODEL_LEN: u32 = 4_096;

pub const VARIANTS: &[Olmo2] = &[

    Olmo2 {
        id: "olmo-2-1b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 16,
            q_heads: 16,
            kv_heads: 16,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
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
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Olmo2 {
        id: "olmo-2-7b",
        shape: LlamaLikeFacts {
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 32,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 11_008,
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
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-6,
        window: -1,
    },

    Olmo2 {
        id: "olmo-2-13b",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 40,
            q_heads: 40,
            kv_heads: 40,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 13_824,
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
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
];

crate::rows_of!(Olmo2);

impl Olmo2 {

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

impl Variant for Olmo2 {
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
                crate::shared::llama_like::contract::author_dense(builder)
            }

            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "olmo-2: no MLX authoring pass exists for this family, so \
                 there is no name layout to author against",
            ),
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
        Arc::new(chat::Olmo2Instruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
