#[cfg(feature = "contract")]
pub mod contract;

pub mod forward;

pub mod spec;

pub mod project;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::{Advertised, Deployment, Refusal};
use crate::manifest::Manifest;

use self::spec::NemotronHFacts;

const ARCH: &str = "nemotron_h";

const NORM_EPS: f32 = 1e-5;

const ROPE_THETA: f32 = 10_000.0;

pub struct NemotronH {

    pub id: &'static str,

    pub shape: NemotronHFacts,

    pub max_model_len: u32,
}

pub const VARIANTS: &[NemotronH] = &[

    NemotronH {
        id: "nemotron-h-4b",
        shape: NemotronHFacts::nemotron_h_4b(),
        max_model_len: 8192,
    },

    NemotronH {
        id: "nemotron-h-8b",
        shape: NemotronHFacts::nemotron_h_8b(),
        max_model_len: 8192,
    },

    NemotronH {
        id: "nemotron-h-47b",
        shape: NemotronHFacts::nemotron_h_47b(),
        max_model_len: 8192,
    },
];

crate::rows_of!(NemotronH);

impl Variant for NemotronH {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape {
            layers: self.shape.layers(),
            head_dim: self.shape.attn.head_dim,
            n_experts: self.shape.moe.num_experts,
            mamba_groups: self.shape.mamba.n_groups,

            kv_shared_layers: 0,
            tied_embeddings: self.shape.tied_embeddings,
        }
    }

    fn deployment(&self, load: Deployed<'_>) -> Result<Deployment, Refusal> {
        let _ = load;
        let mut deployment = project::deployment(
            &self.shape,
            ROPE_THETA,
            NORM_EPS,

            crate::deployment::round_up_attn_head_dim(self.shape.attn.head_dim),
        );
        deployment.advertised = Advertised {
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
        self::contract::author_nemotron_h(builder)
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, Refusal> {

        if let crate::catalog::Backend::Metal(_) = load.backend {
            return Err(Refusal::Unsupported(project::NO_METAL));
        }
        Ok(project::trace(&self.shape, class, NORM_EPS, ROPE_THETA))
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::chatml::QwenInstruct::new(
            tokenizer,
            crate::shared::chatml::NEMOTRON_CHATML,
        ))
    }
}

#[cfg(feature = "contract")]
pub mod import;
