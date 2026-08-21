#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;

use spec::{
    Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts,
};

use model_ir::trace::NormVariant;

#[cfg(feature = "contract")]
pub mod contract;

pub mod forward;

pub mod spec;

pub mod project;

pub struct Qwen35 {

    pub id: &'static str,

    pub shape: Qwen35HybridFacts,

    pub rope_theta: f32,

    pub norm_eps: f32,
}

impl Qwen35 {

    #[must_use]
    pub const fn is_mixture(&self) -> bool {
        matches!(self.shape.mlp, Qwen35MlpKind::Moe(_))
    }

    #[must_use]
    pub const fn experts(&self) -> u32 {
        match &self.shape.mlp {
            Qwen35MlpKind::Moe(moe) => moe.num_experts,
            Qwen35MlpKind::Dense { .. } => 0,
        }
    }
}

const fn attn(
    hidden: u32,
    q_heads: u32,
    kv_heads: u32,
    norm_variant: NormVariant,
) -> Qwen35FullAttnFacts {
    Qwen35FullAttnFacts {
        hidden,
        q_heads,
        kv_heads,
        head_dim: 256,
        rotary_dim: 64,

        fused_qkv: false,
        norm_variant,
    }
}

const fn gdn(
    hidden: u32,
    key_heads: u32,
    value_heads: u32,
    norm_variant: NormVariant,
) -> Qwen35GdnFacts {
    Qwen35GdnFacts {
        hidden,
        key_heads,
        value_heads,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel: 4,

        fused_in_proj: false,
        norm_variant,
    }
}

const ARCH: &str = "qwen3_5";

const MAX_MODEL_LEN: u32 = 262_144;

pub const VARIANTS: &[Qwen35] = &[

    Qwen35 {
        id: "qwen3.5-0.8b-base",
        shape: Qwen35HybridFacts {
            layers: 24,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: true,

            norm_variant: NormVariant::Gemma,
            attn: attn(1024, 8, 2, NormVariant::Gemma),
            gdn: gdn(1024, 16, 16, NormVariant::Gemma),
            mlp: Qwen35MlpKind::Dense { intermediate: 3584 },
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },

    Qwen35 {
        id: "qwen3.5-4b",
        shape: Qwen35HybridFacts {
            layers: 32,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: true,
            norm_variant: NormVariant::Plain,
            attn: attn(2560, 16, 4, NormVariant::Plain),
            gdn: gdn(2560, 16, 32, NormVariant::Plain),
            mlp: Qwen35MlpKind::Dense { intermediate: 9216 },
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },

    Qwen35 {
        id: "qwen3.5-9b",
        shape: Qwen35HybridFacts {
            layers: 32,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Plain,
            attn: attn(4096, 16, 4, NormVariant::Plain),
            gdn: gdn(4096, 16, 32, NormVariant::Plain),
            mlp: Qwen35MlpKind::Dense {
                intermediate: 12_288,
            },
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },

    Qwen35 {
        id: "qwen3.5-35b-a3b",
        shape: Qwen35HybridFacts {
            layers: 40,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Plain,
            attn: attn(2048, 16, 2, NormVariant::Plain),
            gdn: gdn(2048, 16, 32, NormVariant::Plain),
            mlp: Qwen35MlpKind::Moe(Qwen35MoeMlpFacts {
                hidden: 2048,
                num_experts: 256,
                top_k: 8,
                moe_intermediate: 512,
                shared_expert_intermediate: 512,
                norm_variant: NormVariant::Plain,
            }),
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },

    Qwen35 {
        id: "qwen3.6-27b",
        shape: Qwen35HybridFacts {
            layers: 64,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Plain,
            attn: attn(5120, 24, 4, NormVariant::Plain),
            gdn: gdn(5120, 16, 48, NormVariant::Plain),
            mlp: Qwen35MlpKind::Dense {
                intermediate: 17_408,
            },
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },
];

crate::rows_of!(Qwen35);

impl Variant for Qwen35 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {
        if self.is_mixture() {
            LoadShape::mixture(
                self.shape.layers,
                self.shape.attn.head_dim,
                self.experts(),
                self.shape.tied_embeddings,
            )
        } else {
            LoadShape::dense(
                self.shape.layers,
                self.shape.attn.head_dim,
                self.shape.tied_embeddings,
            )
        }
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment = project::deployment(&self.shape, self.rope_theta, self.norm_eps);
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
            crate::shared::policy::Naming::Hf if self.is_mixture() => {
                contract::author_qwen3_5_moe(builder)
            }
            crate::shared::policy::Naming::Hf => contract::author_qwen3_5(builder),

            crate::shared::policy::Naming::Mlx => contract::author_qwen3_5_mlx(builder),
        }
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {

        if let crate::catalog::Backend::Metal(bind) = load.backend {
            project::metal_kernel_refusal(&self.shape, load, bind)?;
            return Ok(project::trace_metal(
                &self.shape,
                class,
                self.rope_theta,
                self.norm_eps,
                bind,
            ));
        }
        Ok(project::trace(
            &self.shape,
            class,
            load,
            self.norm_eps,
            self.rope_theta,
        ))
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::chatml::QwenInstruct::new(
            tokenizer,
            crate::shared::chatml::QWEN_CHATML,
        ))
    }
}

#[cfg(feature = "contract")]
pub mod import;
