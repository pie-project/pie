#[cfg(feature = "chat")]
pub mod chat;

pub mod forward;

pub mod spec;

pub mod project;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;

use self::spec::{Gemma2AttnFacts, Gemma2Facts};

const NORM_EPS: f32 = 1e-6;

const ARCH: &str = "gemma2";

const MAX_MODEL_LEN: u32 = 8_192;

pub struct Gemma2 {

    pub id: &'static str,

    pub shape: Gemma2Facts,

    pub rope_theta: f32,
}

pub const VARIANTS: &[Gemma2] = &[

    Gemma2 {
        id: "gemma-2-2b",
        shape: Gemma2Facts {
            layers: 26,
            vocab: 256_000,
            hidden: 2304,
            intermediate: 9216,
            tied_embeddings: true,
            final_logit_softcap: true,
            sliding_window: 4096,
            full_attn_interval: 2,
            attn: Gemma2AttnFacts {
                heads: 8,
                kv_heads: 4,
                head_dim: 256,
                attn_logit_softcap: true,
            },
        },
        rope_theta: 10_000.0,
    },

    Gemma2 {
        id: "gemma-2-9b",
        shape: Gemma2Facts {
            layers: 42,
            vocab: 256_000,
            hidden: 3584,
            intermediate: 14336,
            tied_embeddings: true,
            final_logit_softcap: true,
            sliding_window: 4096,
            full_attn_interval: 2,
            attn: Gemma2AttnFacts {
                heads: 16,
                kv_heads: 8,
                head_dim: 256,
                attn_logit_softcap: true,
            },
        },
        rope_theta: 10_000.0,
    },

    Gemma2 {
        id: "gemma-2-27b",
        shape: Gemma2Facts {
            layers: 46,
            vocab: 256_000,
            hidden: 4608,
            intermediate: 36864,
            tied_embeddings: true,
            final_logit_softcap: true,
            sliding_window: 4096,
            full_attn_interval: 2,
            attn: Gemma2AttnFacts {
                heads: 32,
                kv_heads: 16,
                head_dim: 128,
                attn_logit_softcap: true,
            },
        },
        rope_theta: 10_000.0,
    },
];

crate::rows_of!(Gemma2);

impl Variant for Gemma2 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape::dense(
            self.shape.layers,
            self.shape.attn.head_dim,
            self.shape.tied_embeddings,
        )
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {

        let _ = load;
        let mut deployment = project::deployment(&self.shape, self.rope_theta, NORM_EPS);
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
                "gemma-2: no MLX authoring pass exists for this generation, \
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
        Ok(project::trace(
            &self.shape,
            class,
            load,
            NORM_EPS,
            self.rope_theta,
        ))
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::gemma_chat::Gemma3Instruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
