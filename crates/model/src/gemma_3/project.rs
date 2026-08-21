use crate::catalog::{Backend, Deployed, MetalBinding};
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle,
};
use crate::manifest::{Manifest, TensorSpec};
use crate::shared::llama_like::project as family;
use crate::shared::llama_like::spec::LlamaLikeFacts;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {

    pub sliding_window: i32,

    pub full_attn_interval: u32,

    pub rope_theta_local: f32,

    pub rope_theta_global: f32,

    pub query_pre_attn_scalar: u32,
}

impl Schedule {

    #[must_use]
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_ir::facts::full_attn_at(self.full_attn_interval, l)
    }

    #[must_use]
    pub fn window_at(&self, l: u32) -> i32 {
        if self.is_full_attn(l) {
            -1
        } else {
            self.sliding_window
        }
    }

    #[must_use]
    pub fn rope_theta_at(&self, l: u32) -> f32 {
        if self.is_full_attn(l) {
            self.rope_theta_global
        } else {
            self.rope_theta_local
        }
    }
}

#[must_use]
pub fn manifest(f: &LlamaLikeFacts) -> Manifest {
    let hidden = u64::from(f.hidden);
    let head_dim = u64::from(f.head_dim);
    let base = family::manifest(f);

    let mut out = Manifest::new(base.layers);
    for spec in base.tensors {

        if spec.name == "layer.{}.input_layernorm"
            || spec.name == "layer.{}.post_attention_layernorm"
            || spec.name == "layer.{}.post_feedforward_layernorm"
        {
            continue;
        }
        out = out.with(spec);
    }
    out.with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.pre_feedforward_layernorm",
            [hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.post_feedforward_layernorm",
            [hidden],
        ))

        .with(TensorSpec::required(
            "layer.{}.self_attn.k_norm",
            [head_dim],
        ))
}

#[must_use]
pub fn deployment(f: &LlamaLikeFacts, s: &Schedule, norm_eps: f32) -> Deployment {
    let head_dim = crate::deployment::round_up_attn_head_dim(f.head_dim);
    let sm_scale = 1.0 / (s.query_pre_attn_scalar as f32).sqrt();
    let attention = (0..f.layers)
        .map(|l| LayerAttention {

            kv_heads: f.kv_heads,
            head_dim,
            window: s.window_at(l),

            kv_source: l,
            sm_scale,
            rope_theta: s.rope_theta_at(l),

            rotary_dim: 0,
            q_gate: false,
        })
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.q_heads,
            kv_heads: f.kv_heads,
            head_dim: f.head_dim,
            head_dim_kernel: crate::deployment::round_up_attn_head_dim(f.head_dim),
            intermediate: f.intermediate,

            moe_intermediate: 0,
            experts_per_token: 0,
            shared_intermediate: 0,
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Paged,
        recurrent: None,
        prefill: PrefillStyle::Planned,
        attn_output: AttnOutput::DriverPinned,

        logit_softcap: 0.0,

        attn_logit_softcap: 0.0,

        ple_dim: 0,

        norm: NormPlacement::Pre,

        norm_unit_offset: true,

        v_norm: false,

        norm_topk_prob: true,

        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::GeluTanh,
        scales: std::collections::BTreeMap::new(),

        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

#[must_use]
pub fn metal_facts(
    f: &LlamaLikeFacts,
    s: &Schedule,
    norm_eps: f32,
    load: Deployed<'_>,
    bind: &MetalBinding,
) -> crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts {

    let base = family::metal_facts(
        family::RowScalars {
            rope_theta: s.rope_theta_global,
            norm_eps,
            window: -1,
            rope_rescaled: false,

            norm_topk_prob: true,
        },
        load,
        bind,
    );

    #[allow(clippy::cast_possible_truncation)]
    let embed_scale = f64::from(f.hidden).sqrt() as f32;
    crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts {

        window_left: (0..f.layers).map(|l| s.window_at(l)).collect(),

        rope_theta_sliding: s.rope_theta_local,

        activation: crate::shared::llama_like::forward::facts::Activation::Geglu,

        embed_scale,

        attn_scale: 1.0 / (s.query_pre_attn_scalar as f32).sqrt(),
        ..base
    }
}

pub fn trace(
    f: &LlamaLikeFacts,
    s: &Schedule,
    norm_eps: f32,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {

    let row = family::RowScalars {
        rope_theta: s.rope_theta_global,
        norm_eps,
        window: -1,
        rope_rescaled: false,

        norm_topk_prob: true,
    };
    match load.backend {
        Backend::Cuda => family::trace(f, row, class, load),
        Backend::Metal(bind) => {

            let m = metal_facts(f, s, norm_eps, load, bind);
            family::metal_kernel_refusal(f, &m, load, bind)?;
            Ok(crate::shared::llama_like::forward::llama_like_metal(
                f, &m, class,
            ))
        }
    }
}
