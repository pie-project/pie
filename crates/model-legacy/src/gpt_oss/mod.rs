use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;

use spec::GptOssFacts;

#[cfg(feature = "contract")]
pub mod contract;

pub mod forward;

pub mod spec;

pub mod project;

pub struct GptOss {
    pub id: &'static str,

    pub shape: GptOssFacts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub window: i32,
}

impl GptOss {
    #[must_use]
    pub const fn experts(&self) -> u32 {
        self.shape.experts
    }
}

const ARCH: &str = "gptoss";

const MAX_MODEL_LEN: u32 = 131_072;

const ROPE_SCALING: crate::deployment::RopeScaling = crate::deployment::RopeScaling::Yarn {
    factor: 32.0,
    beta_fast: 32.0,
    beta_slow: 1.0,
    attention_factor: 1.346_573_6,
    original_max_position: 4_096,

    truncate: false,
};

pub const VARIANTS: &[GptOss] = &[
    GptOss {
        id: "gpt-oss-20b",
        shape: GptOssFacts {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            experts: 32,
            top_k: 4,
            vocab: 201_088,
            tied_embeddings: false,
            swiglu_limit: 7.0,
        },
        rope_theta: 150_000.0,
        norm_eps: 1e-5,
        window: 128,
    },
    GptOss {
        id: "gpt-oss-120b",
        shape: GptOssFacts {
            hidden: 2880,
            layers: 36,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            experts: 128,
            top_k: 4,
            vocab: 201_088,
            tied_embeddings: false,
            swiglu_limit: 7.0,
        },
        rope_theta: 150_000.0,
        norm_eps: 1e-5,
        window: 128,
    },
];

crate::rows_of!(GptOss);

impl Variant for GptOss {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape::mixture(
            self.shape.layers,
            self.shape.head_dim,
            self.experts(),
            self.shape.tied_embeddings,
        )
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment =
            project::deployment(&self.shape, self.rope_theta, self.norm_eps, self.window);
        deployment.rope_scaling = Some(ROPE_SCALING);
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
            crate::shared::policy::Naming::Hf => contract::author_gpt_oss(builder),
            crate::shared::policy::Naming::Mlx => contract::author_gpt_oss_mlx(builder),
        }
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
        if let crate::catalog::Backend::Metal(bind) = load.backend {
            let shape = project::metal_shape(&self.shape);
            let facts = project::metal_facts(&self.shape, bind);

            crate::shared::llama_like::project::metal_kernel_refusal(&shape, &facts, load, bind)?;
            return Ok(crate::shared::llama_like::forward::llama_like_metal(
                &shape, &facts, class,
            ));
        }
        Ok(project::trace(
            &self.shape,
            class,
            load,
            self.norm_eps,
            self.rope_theta,
            self.window,
        ))
    }
}

#[cfg(feature = "contract")]
pub mod import;
