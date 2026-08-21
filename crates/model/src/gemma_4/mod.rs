#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

pub mod project;

pub mod spec;

pub mod forward;

#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::{Advertised, AudioTower, Deployment, Refusal, Towers, VisionTower};
use crate::manifest::Manifest;

use self::spec::{Gemma4Facts, Gemma4Mixture};

const NORM_EPS: f32 = 1e-6;

const ARCH: &str = "gemma4";

const E_SERIES_AUDIO: AudioTower = AudioTower {
    layers: 12,
    hidden: 1024,
    heads: 8,
    conv_kernel: 5,
    feature_size: 128,

    subsample_channels_0: 128,
    subsample_channels_1: 32,
    output_dims: 1536,
    chunk_size: 12,
    context_left: 13,
    context_right: 0,
    logit_cap: 50.0,
    residual_weight: 0.5,
    norm_eps: NORM_EPS,
};

const E_SERIES_VISION: VisionTower = VisionTower {
    layers: 16,
    hidden: 768,
    heads: 12,
    intermediate: 3072,
    pooling_kernel: 3,
    norm_eps: NORM_EPS,

    rope_theta: 100.0,
};

const A4B_VISION: VisionTower = VisionTower {
    layers: 27,
    hidden: 1152,
    heads: 16,
    intermediate: 4304,
    pooling_kernel: 3,
    norm_eps: NORM_EPS,
    rope_theta: 100.0,
};

const E_SERIES_TOWERS: Towers = Towers {
    audio: Some(E_SERIES_AUDIO),
    vision: Some(E_SERIES_VISION),
};

pub struct Gemma4 {

    pub id: &'static str,

    pub shape: Gemma4Facts,

    pub mixture: Option<Gemma4Mixture>,

    pub sliding_window: i32,

    pub k_eq_v: bool,

    pub max_model_len: u32,

    pub towers: Towers,
}

pub const VARIANTS: &[Gemma4] = &[

    Gemma4 {
        id: "gemma-4-e2b",
        shape: Gemma4Facts::gemma_4_e2b(),
        mixture: None,
        sliding_window: 512,
        k_eq_v: false,
        max_model_len: 131_072,
        towers: E_SERIES_TOWERS,
    },

    Gemma4 {
        id: "gemma-4-e4b",
        shape: Gemma4Facts::gemma_4_e4b(),
        mixture: None,
        sliding_window: 512,
        k_eq_v: false,
        max_model_len: 131_072,
        towers: E_SERIES_TOWERS,
    },

    Gemma4 {
        id: "gemma-4-31b",
        shape: Gemma4Facts::gemma_4_31b(),
        mixture: None,
        sliding_window: 1024,
        k_eq_v: true,
        max_model_len: 262_144,

        towers: Towers {
            audio: None,
            vision: Some(A4B_VISION),
        },
    },

    Gemma4 {
        id: "gemma-4-26b-a4b",
        shape: Gemma4Facts::gemma_4_26b_a4b(),
        mixture: Some(Gemma4Mixture::gemma_4_26b_a4b()),
        sliding_window: 1024,
        k_eq_v: true,
        max_model_len: 262_144,

        towers: Towers {
            audio: None,
            vision: Some(A4B_VISION),
        },
    },
];

crate::rows_of!(Gemma4);

impl Gemma4 {

    fn row(&self) -> project::RowScalars {
        project::RowScalars {
            mixture: self.mixture,
            sliding_window: self.sliding_window,
            norm_eps: NORM_EPS,
            k_eq_v: self.k_eq_v,
        }
    }

    fn untraced(&self) -> Option<Refusal> {
        None
    }
}

impl Variant for Gemma4 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape, self.mixture)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape {
            layers: self.shape.layers,

            head_dim: self.shape.head_dim,
            n_experts: self.mixture.map_or(0, |m| m.num_experts),

            mamba_groups: 0,
            kv_shared_layers: self.shape.kv_shared_layers,
            tied_embeddings: self.shape.tied_embeddings,
        }
    }

    fn deployment(&self, load: Deployed<'_>) -> Result<Deployment, Refusal> {
        if let Some(refusal) = self.untraced() {
            return Err(refusal);
        }
        let mut deployment = project::deployment(&self.shape, self.row(), load);
        deployment.towers = self.towers.clone();
        deployment.advertised = Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,

            media_encode: deployment.towers.audio.is_some() || deployment.towers.vision.is_some(),
        };
        Ok(deployment)
    }

    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => self::contract::author_gemma4(builder),
            crate::shared::policy::Naming::Mlx => self::contract::author_gemma4_mlx(builder),
        }
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, Refusal> {
        if let Some(refusal) = self.untraced() {
            return Err(refusal);
        }

        if let crate::catalog::Backend::Metal(bind) = load.backend {
            let shape = project::metal_shape(&self.shape, self.mixture);
            let facts = project::metal_facts(&self.shape, self.row(), bind);

            crate::shared::llama_like::project::metal_kernel_refusal(&shape, &facts, load, bind)?;
            return Ok(crate::shared::llama_like::forward::llama_like_metal(
                &shape, &facts, class,
            ));
        }

        if self.k_eq_v {
            return Err(Refusal::Unsupported(
                "gemma-4 31B/26B-A4B on CUDA: these rows read V out of the K projection (`attention_k_eq_v`) and ship no `v_proj`; the hand-written text projects one. The Metal text reads it (`LlamaLikeMetalFacts::v_from_k`) and serves these rows",
            ));
        }
        Ok(project::trace(
            &self.shape,
            self.sliding_window,
            class,
            load.layer_scalars,
            NORM_EPS,
        ))
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(self::chat::Gemma4Instruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
