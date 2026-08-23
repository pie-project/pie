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
use spec::Glm5Facts;

const ARCH: &str = "glm_moe_dsa";

pub struct Glm5 {
    pub id: &'static str,

    pub shape: Glm5Facts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub max_model_len: u32,

    pub tied_embeddings: bool,
}

pub const VARIANTS: &[Glm5] = &[Glm5 {
    id: "glm-5-106b-a12b",
    shape: Glm5Facts {
        layers: 46,
        vocab: 151_552,
        hidden: 4096,
        dense_intermediate: 10_944,
        dense_layers: 3,
        attn: spec::Glm5MlaFacts {
            hidden: 4096,
            heads: 96,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            output_gate: false,
        },
        dsa: spec::Glm5DsaFacts {
            index_n_heads: 64,
            index_head_dim: 128,
            index_topk: 2048,
        },
        moe: spec::Glm5MoeFacts {
            hidden: 4096,
            num_experts: 128,
            top_k: 8,
            norm_topk_prob: true,
            routed_scaling: 2.5,
            moe_intermediate: 1408,
            shared_intermediate: 1408,
            aligned_block: 16,
        },
    },

    rope_theta: 10_000.0,
    norm_eps: 1e-5,

    max_model_len: 0,
    tied_embeddings: false,
}];

crate::rows_of!(Glm5);

impl Glm5 {
    #[must_use]
    pub fn advertised(&self) -> Advertised {
        Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,

            media_encode: false,
        }
    }
}

impl Variant for Glm5 {
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
            self.advertised(),
        )
    }

    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => contract::author_glm5(builder),

            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "glm-5: no MLX authoring pass exists for this generation, so \
                 there is no name layout to author against",
            ),
        }
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
            .map(|_| project::trace(&self.shape, class, self.norm_eps, self.rope_theta))
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::chatml::QwenInstruct::new(
            tokenizer,
            crate::shared::chatml::GLM_CHATML,
        ))
    }
}

#[cfg(feature = "contract")]
pub mod import;
