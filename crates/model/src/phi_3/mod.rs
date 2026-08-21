#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;
use crate::shared::llama_like::project;
use crate::shared::llama_like::spec::LlamaLikeFacts;

use model_ir::facts::{NormPlacement, QkNorm};
use model_ir::trace::{NormVariant, RopeKind};

pub struct Phi3 {

    pub id: &'static str,

    pub shape: LlamaLikeFacts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub window: i32,
}

const ARCH: &str = "phi3";

const MAX_MODEL_LEN: u32 = 4_096;

pub const VARIANTS: &[Phi3] = &[

    Phi3 {
        id: "phi-3-mini-4k",
        shape: LlamaLikeFacts {
            hidden: 3072,
            layers: 32,
            q_heads: 32,
            kv_heads: 32,
            head_dim: 96,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
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
        },
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        window: 2047,
    },

    Phi3 {
        id: "phi-3-medium-4k",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 40,
            q_heads: 40,
            kv_heads: 10,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 17_920,
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
        },
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        window: 2047,
    },
];

crate::rows_of!(Phi3);

impl Phi3 {

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

impl Variant for Phi3 {
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
        contract::author_phi3(builder)
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
        Arc::new(chat::Phi3Instruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
