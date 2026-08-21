#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

pub mod forward;

pub mod spec;

pub mod project;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::Advertised;
use crate::manifest::Manifest;
use spec::KimiFacts;

const ARCH: &str = "kimi_k2";

pub struct KimiK2 {

    pub id: &'static str,

    pub shape: KimiFacts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub rope_yarn: bool,

    pub tied_embeddings: bool,

    pub max_model_len: u32,
}

pub const VARIANTS: &[KimiK2] = &[

    KimiK2 {
        id: "kimi-k2",
        shape: KimiFacts {
            layers: 61,
            vocab: 163_840,
            hidden: 7168,
            dense_intermediate: 18_432,
            dense_layers: 1,
            attn: spec::KimiMlaFacts {
                hidden: 7168,
                heads: 64,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                output_gate: false,
            },
            moe: spec::KimiMoeFacts {
                num_experts: 384,
                top_k: 8,

                norm_topk_prob: false,
                routed_scaling: 2.0,
                moe_intermediate: 2048,
                shared_intermediate: 2048,
            },
        },
        rope_theta: 50_000.0,
        norm_eps: 1e-6,
        rope_yarn: true,
        tied_embeddings: false,

        max_model_len: 131_072,
    },
];

const ROPE_SCALING: crate::deployment::RopeScaling = crate::deployment::RopeScaling::Yarn {
    factor: 32.0,
    beta_fast: 1.0,
    beta_slow: 1.0,
    attention_factor: 1.0,
    original_max_position: 4_096,

    truncate: true,
};

crate::rows_of!(KimiK2);

impl Variant for KimiK2 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape, self.tied_embeddings)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape::mixture(
            self.shape.layers,
            self.shape.attn.kv_a_width(),
            self.shape.moe.num_experts,
            self.tied_embeddings,
        )
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        project::deployment(
            &self.shape,
            self.rope_theta,
            self.norm_eps,
            self.rope_yarn,
            Advertised {
                arch: ARCH,
                max_model_len: self.max_model_len,

                media_encode: false,
            },
        )
    }

    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        contract::author_kimi(builder)
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {

        if let crate::catalog::Backend::Metal(_) = load.backend {
            return Err(crate::deployment::Refusal::Unsupported(project::NO_METAL));
        }

        self.deployment(load)
            .map(|_| project::trace(&self.shape, self.rope_yarn, class, self.norm_eps))
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::kimi::KimiInstruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
