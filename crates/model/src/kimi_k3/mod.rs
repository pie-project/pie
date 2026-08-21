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
use spec::KimiK3Facts;

const ARCH: &str = "kimi_k3";

pub struct KimiK3 {

    pub id: &'static str,

    pub shape: KimiK3Facts,

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub max_model_len: u32,

    pub tied_embeddings: bool,
}

pub const VARIANTS: &[KimiK3] = &[KimiK3 {
    id: "kimi-k3",
    shape: KimiK3Facts {
        layers: 8,
        vocab: 163_840,
        hidden: 2048,
        dense_intermediate: 5632,
        dense_layers: 1,
        full_attn_interval: 4,
        attn_res_block: 4,
        attn: spec::KimiK3MlaFacts {
            hidden: 2048,
            heads: 16,
            q_lora_rank: 768,
            kv_lora_rank: 256,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,

            output_gate: true,
        },
        kda: spec::KimiK3KdaFacts {
            value_heads: 16,
            value_head_dim: 128,
            conv_kernel: 4,
            gate_lower_bound_milli: 0,
            norm_eps_micro: 10,
        },
        moe: spec::KimiK3MoeFacts {
            num_experts: 64,
            top_k: 6,

            norm_topk_prob: false,
            routed_scaling: 2.0,
            moe_intermediate: 1024,
            shared_intermediate: 1024,
        },
    },

    rope_theta: 10_000.0,
    norm_eps: 1e-5,

    max_model_len: 0,
    tied_embeddings: false,
}];

crate::rows_of!(KimiK3);

impl KimiK3 {

    #[must_use]
    pub fn advertised(&self) -> Advertised {
        Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,

            media_encode: false,
        }
    }
}

impl Variant for KimiK3 {
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
            crate::shared::policy::Naming::Hf => contract::author_kimi_k3(builder),

            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "kimi-k3: no MLX authoring pass exists for this generation, so \
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
            .and_then(|_| project::trace(&self.shape, class, self.norm_eps))
    }

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::kimi::KimiInstruct::new(tokenizer))
    }
}

#[cfg(feature = "contract")]
pub mod import;
