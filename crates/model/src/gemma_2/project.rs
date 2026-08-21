use crate::catalog::Deployed;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::Gemma2Facts;

pub const FINAL_LOGIT_SOFTCAP: f32 = 30.0;

pub const ATTN_LOGIT_SOFTCAP: f32 = 50.0;

#[must_use]
pub fn manifest(f: &Gemma2Facts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let (q, kv) = (u64::from(f.attn.q_width()), u64::from(f.attn.kv_width()));
    let inter = u64::from(f.intermediate);

    Manifest::new(f.layers)

        .holds_projections_as(<super::forward::ShippedW1 as model_dsl::axes::DtypeAxis>::REPR)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))

        .tie(f.tied_embeddings, "lm_head", [vocab, hidden])
        .with(TensorSpec::required(
            "layer.{}.self_attn.q_proj",
            [q, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.k_proj",
            [kv, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.v_proj",
            [kv, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.o_proj",
            [hidden, q],
        ))

        .with(TensorSpec::absent("layer.{}.self_attn.q_norm"))
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
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
            "layer.{}.mlp.gate_proj",
            [inter, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.mlp.up_proj",
            [inter, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.mlp.down_proj",
            [hidden, inter],
        ))
}

#[must_use]
pub fn deployment(f: &Gemma2Facts, rope_theta: f32, norm_eps: f32) -> Deployment {

    let head_dim = crate::deployment::round_up_attn_head_dim(f.attn.head_dim);
    let attention = (0..f.layers)
        .map(|l| LayerAttention {

            kv_heads: f.attn.kv_heads,
            head_dim,

            window: f.window_left_at(l),

            kv_source: l,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            rope_theta,

            rotary_dim: 0,
            q_gate: false,
        })
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.attn.heads,
            kv_heads: f.attn.kv_heads,
            head_dim: f.attn.head_dim,
            head_dim_kernel: head_dim,
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
        logit_softcap: if f.final_logit_softcap {
            FINAL_LOGIT_SOFTCAP
        } else {
            0.0
        },

        attn_logit_softcap: if f.attn.attn_logit_softcap {
            ATTN_LOGIT_SOFTCAP
        } else {
            0.0
        },

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

pub const NO_METAL: &str = "gemma-2 has no Metal text in this build: its forward is `gemma2_cuda`, whose \
     attention logit cap has no counterpart in the one Metal text here \
     (`llama_like_metal`), and whose shape is `Gemma2Facts` rather than the \
     `LlamaLikeFacts` that text takes; the CUDA backend serves this row";

#[must_use]
pub fn trace(
    f: &Gemma2Facts,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
    norm_eps: f32,
    rope_theta: f32,
) -> model_ir::trace::ForwardPlan {
    let _ = load;

    super::forward::gemma2_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, class, norm_eps, rope_theta)
}
