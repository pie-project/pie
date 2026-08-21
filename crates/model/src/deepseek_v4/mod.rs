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
use spec::Dsv4Facts;

const ARCH: &str = "deepseek_v4";

pub struct Dsv4 {

    pub id: &'static str,

    pub shape: Dsv4Facts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub max_model_len: u32,

    pub tied_embeddings: bool,
}

pub const VARIANTS: &[Dsv4] = &[Dsv4 {
    id: "deepseek-v4",
    shape: Dsv4Facts {
        layers: 6,
        vocab: 129_280,
        hidden: 2048,
        dense_intermediate: 5632,
        dense_layers: 1,

        ratios: &[1, 2, 4],
        attn: spec::Dsv4AttnFacts {
            hidden: 2048,
            heads: 16,
            head_dim: 128,
            q_lora_rank: 768,
            qk_rope_head_dim: 64,
            sliding_window: 2048,
            o_lora_rank: 512,
            o_groups: 4,
        },
        hc: spec::Dsv4HcFacts { mult: 4 },
        moe: spec::Dsv4MoeFacts {
            num_experts: 64,
            top_k: 6,
            norm_topk_prob: false,
            routed_scaling: 2.5,
            moe_intermediate: 1024,
            swiglu_limit_milli: 7000,
            hash_routed: false,
        },
    },

    rope_theta: 10_000.0,
    norm_eps: 1e-5,

    max_model_len: 0,

    tied_embeddings: true,
}];

crate::rows_of!(Dsv4);

impl Dsv4 {

    #[must_use]
    pub fn advertised(&self) -> Advertised {
        Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,

            media_encode: false,
        }
    }
}

impl Variant for Dsv4 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape, self.tied_embeddings)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape::mixture(
            self.shape.layers,
            self.shape.attn.head_dim,
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
            crate::shared::policy::Naming::Hf => contract::author_deepseek_v4(builder),

            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "deepseek-v4: no MLX authoring pass exists for this generation, \
                 so there is no name layout to author against",
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
        Arc::new(crate::shared::deepseek::R1Instruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
