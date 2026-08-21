#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

pub mod spec;

pub mod project;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::{Deployment, Refusal};
use crate::manifest::Manifest;

use self::spec::CsmFacts;

pub struct Csm {

    pub id: &'static str,

    pub shape: CsmFacts,
}

pub const VARIANTS: &[Csm] = &[Csm {
    id: "csm-1b",
    shape: CsmFacts::csm_1b(),
}];

crate::rows_of!(Csm);

impl Variant for Csm {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape {
            layers: self.shape.backbone.layers,
            head_dim: self.shape.backbone.head_dim,

            n_experts: 0,

            mamba_groups: 0,

            kv_shared_layers: 0,

            tied_embeddings: self.shape.tied_embeddings,
        }
    }

    fn deployment(&self, load: Deployed<'_>) -> Result<Deployment, Refusal> {
        let _ = load;
        project::deployment(&self.shape)
    }

    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        self::contract::author_csm(builder)
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, Refusal> {
        let _ = (class, load);
        project::trace(&self.shape)
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(self::chat::CsmInstruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
