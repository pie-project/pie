pub mod forward;

pub mod project;

pub mod spec;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;

use self::spec::{Gemma3nAltUpFacts, Gemma3nAttnFacts, Gemma3nFacts, window_schedule};

const NORM_EPS: f32 = 1e-6;

const ARCH: &str = "gemma3n";

const MAX_MODEL_LEN: u32 = 32_768;

pub struct Gemma3n {

    pub id: &'static str,

    pub shape: Gemma3nFacts,

    pub rope_theta_global: f32,

    pub rope_theta_local: f32,
}

const E2B_WINDOWS: [i32; 30] = window_schedule(5, 512);

const E4B_WINDOWS: [i32; 35] = window_schedule(5, 512);

pub const VARIANTS: &[Gemma3n] = &[

    Gemma3n {
        id: "gemma-3n-e2b",
        shape: Gemma3nFacts {
            vocab: 262_400,
            hidden: 2048,

            per_layer_intermediate: &[8192; 30],
            laurel_rank: 64,
            ple_width: 256,
            ple_vocab: 262_144,

            sparsity_layers: 10,
            altup: Gemma3nAltUpFacts {
                num_streams: 4,
                active: 0,
            },
            attn: Gemma3nAttnFacts {
                heads: 8,
                kv_heads: 2,
                head_dim: 256,
            },
            window_left: &E2B_WINDOWS,
        },
        rope_theta_global: 1_000_000.0,
        rope_theta_local: 10_000.0,
    },

    Gemma3n {
        id: "gemma-3n-e4b",
        shape: Gemma3nFacts {
            vocab: 262_400,
            hidden: 2048,
            per_layer_intermediate: &[16384; 35],
            laurel_rank: 64,
            ple_width: 256,
            ple_vocab: 262_144,

            sparsity_layers: 10,
            altup: Gemma3nAltUpFacts {
                num_streams: 4,
                active: 0,
            },
            attn: Gemma3nAttnFacts {
                heads: 8,
                kv_heads: 2,
                head_dim: 256,
            },
            window_left: &E4B_WINDOWS,
        },
        rope_theta_global: 1_000_000.0,
        rope_theta_local: 10_000.0,
    },
];

crate::rows_of!(Gemma3n);

impl Variant for Gemma3n {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {

        LoadShape::dense(self.shape.layers(), self.shape.attn.head_dim, true)
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {

        let _ = load;
        let mut deployment = project::deployment(
            &self.shape,
            self.rope_theta_global,
            self.rope_theta_local,
            NORM_EPS,
        );
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
                "gemma-3n: no MLX authoring pass exists for this generation, \
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
            self.rope_theta_global,
            self.rope_theta_local,
        ))
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::gemma_chat::Gemma3Instruct::for_variant(
            tokenizer,
            crate::shared::gemma_chat::Gemma3Variant::Gemma3nText,
        ))
    }
}

#[cfg(feature = "contract")]
pub mod import;
