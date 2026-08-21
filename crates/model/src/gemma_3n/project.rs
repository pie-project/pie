use crate::catalog::Deployed;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::Gemma3nFacts;

pub const FINAL_LOGIT_SOFTCAP: f32 = 30.0;

#[must_use]
pub fn manifest(f: &Gemma3nFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let (q, kv) = (u64::from(f.attn.q_width()), u64::from(f.attn.kv_width()));
    let head_dim = u64::from(f.attn.head_dim);
    let inter = u64::from(f.intermediate(0));
    let ple = u64::from(f.ple_width);
    let layers = u64::from(f.layers());

    Manifest::new(f.layers())

        .holds_projections_as(<super::forward::ShippedW1 as model_dsl::axes::DtypeAxis>::REPR)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))

        .with(TensorSpec::absent("lm_head"))

        .with(TensorSpec::required(
            "embed_tokens_per_layer",
            [vocab, layers * ple],
        ))
        .with(TensorSpec::required(
            "per_layer_model_projection",
            [layers * ple, hidden],
        ))
        .with(TensorSpec::required("per_layer_projection_norm", [ple]))
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

        .with(TensorSpec::required(
            "layer.{}.self_attn.q_norm",
            [head_dim],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.k_norm",
            [head_dim],
        ))

        .with(TensorSpec::required(
            "layer.{}.self_attn.v_norm",
            [head_dim],
        ))
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

        .with(TensorSpec::required(
            "layer.{}.laurel.linear_left",
            [u64::from(f.laurel_rank), hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.laurel.linear_right",
            [hidden, u64::from(f.laurel_rank)],
        ))
        .with(TensorSpec::required(
            "layer.{}.laurel.post_laurel_norm",
            [hidden],
        ))

        .with(TensorSpec::required(
            "layer.{}.altup.modality_router",
            [u64::from(f.altup.num_streams), hidden],
        ))
        .with(TensorSpec::required("layer.{}.altup.router_norm", [hidden]))

        .with(TensorSpec::required(
            "layer.{}.per_layer_input_gate",
            [ple, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.per_layer_projection",
            [hidden, ple],
        ))
        .with(TensorSpec::required(
            "layer.{}.post_per_layer_input_norm",
            [hidden],
        ))
}

#[must_use]
pub fn deployment(
    f: &Gemma3nFacts,
    rope_theta_global: f32,
    rope_theta_local: f32,
    norm_eps: f32,
) -> Deployment {
    let head_dim = crate::deployment::round_up_attn_head_dim(f.attn.head_dim);
    let attention = (0..f.layers())
        .map(|l| {
            let window = model_ir::facts::window_left_at(f.window_left, l);
            LayerAttention {

                kv_heads: f.attn.kv_heads,
                head_dim,
                window,

                kv_source: l,
                sm_scale: 1.0 / (head_dim as f32).sqrt(),

                rope_theta: if window < 0 {
                    rope_theta_global
                } else {
                    rope_theta_local
                },

                rotary_dim: 0,
                q_gate: false,
            }
        })
        .collect();
    Deployment {
        layers: f.layers(),
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.attn.heads,
            kv_heads: f.attn.kv_heads,
            head_dim: f.attn.head_dim,
            head_dim_kernel: head_dim,

            intermediate: f.intermediate(0),

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
        logit_softcap: FINAL_LOGIT_SOFTCAP,

        attn_logit_softcap: 0.0,

        ple_dim: f.ple_width,
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

pub const NO_METAL: &str = "gemma-3n has no Metal text in this build: its forward is `gemma3n_cuda` — \
     AltUp's four-way hidden bundle, the Laurel residual, per-layer embeddings \
     and the shared-KV tail — and the one Metal text here (`llama_like_metal`) \
     states two of those four and takes a different shape; a text that is \
     recognisably gemma-3n and is not this one is the failure to avoid";

#[must_use]
pub fn trace(
    f: &Gemma3nFacts,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
    norm_eps: f32,
    rope_theta_global: f32,
    rope_theta_local: f32,
) -> model_ir::trace::ForwardPlan {
    let _ = load;

    super::forward::gemma3n_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, class, norm_eps, rope_theta_global, rope_theta_local)
}
