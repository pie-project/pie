use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, RecurrentShape, Towers,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::NemotronHFacts;

#[must_use]
pub fn manifest(f: &NemotronHFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let m = &f.mamba;
    let a = &f.attn;
    let (intermediate, conv_dim) = (u64::from(m.intermediate()), u64::from(m.conv_dim()));
    let heads = u64::from(m.num_heads);
    let mlp = u64::from(f.moe.moe_intermediate);

    Manifest::new(f.layers())

        .holds_projections_as(<super::forward::ShippedW1 as model_dsl::axes::DtypeAxis>::REPR)
        .holds_experts_as(<super::forward::ShippedW2 as model_dsl::axes::DtypeAxis>::REPR)
        .with(TensorSpec::required("backbone.embeddings", [vocab, hidden]))
        .with(TensorSpec::required("backbone.norm_f", [hidden]))

        .tie(f.tied_embeddings, "lm_head", [vocab, hidden])

        .with(TensorSpec::required("backbone.layer.{}.norm", [hidden]))

        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.in_proj",
            [u64::from(m.in_proj_width()), hidden],
        ))

        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.conv1d",
            [conv_dim, 1, u64::from(m.conv_kernel)],
        ))

        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.conv1d.bias",
            [conv_dim],
        ))

        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.A_log",
            [heads],
        ))
        .with(TensorSpec::required("backbone.layer.{}.mixer.D", [heads]))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.dt_bias",
            [heads],
        ))

        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.norm",
            [intermediate],
        ))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.out_proj",
            [hidden, intermediate],
        ))

        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.q_proj",
            [u64::from(a.q_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.k_proj",
            [u64::from(a.kv_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.v_proj",
            [u64::from(a.kv_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone.layer.{}.mixer.o_proj",
            [hidden, u64::from(a.q_width())],
        ))

        .with(TensorSpec::absent("backbone.layer.{}.mixer.q_norm"))
        .with(TensorSpec::absent("backbone.layer.{}.mixer.k_norm"))

        .with(TensorSpec::absent("backbone.layer.{}.mixer.gate_proj"))
        .either(
            !f.is_mixture(),
            "backbone.layer.{}.mixer.up_proj",
            [mlp, hidden],
        )
        .either(
            !f.is_mixture(),
            "backbone.layer.{}.mixer.down_proj",
            [hidden, mlp],
        )

        .with(if f.is_mixture() {
            TensorSpec::present("backbone.layer.{}.mixer.experts.0.up_proj")
        } else {
            TensorSpec::absent("backbone.layer.{}.mixer.experts.0.up_proj")
        })
        .with(if f.is_mixture() {
            TensorSpec::present("backbone.layer.{}.mixer.experts.0.down_proj")
        } else {
            TensorSpec::absent("backbone.layer.{}.mixer.experts.0.down_proj")
        })
}

#[must_use]
pub fn deployment(
    f: &NemotronHFacts,
    rope_theta: f32,
    norm_eps: f32,
    head_dim_kernel: u32,
) -> Deployment {
    let a = &f.attn;
    let head_dim = head_dim_kernel.max(a.head_dim);
    let attention = (0..f.layers())
        .map(|l| LayerAttention {

            kv_heads: a.kv_heads,
            head_dim,

            window: -1,
            kv_source: l,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            rope_theta,

            rotary_dim: 0,
            q_gate: false,
        })
        .collect();
    Deployment {
        layers: f.layers(),
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: a.heads,
            kv_heads: a.kv_heads,
            head_dim: a.head_dim,
            head_dim_kernel,

            intermediate: if f.is_mixture() {
                0
            } else {
                f.moe.moe_intermediate
            },
            moe_intermediate: if f.is_mixture() {
                f.moe.moe_intermediate
            } else {
                0
            },
            experts_per_token: if f.is_mixture() { f.moe.top_k } else { 0 },
            shared_intermediate: if f.is_mixture() {
                f.moe.shared_intermediate
            } else {
                0
            },
            vocab: f.vocab,
        },
        attention,

        kv: KvStyle::Paged,
        recurrent: Some(mamba_shape(f)),
        prefill: PrefillStyle::Planned,
        attn_output: AttnOutput::DriverPinned,
        logit_softcap: 0.0,

        attn_logit_softcap: 0.0,
        ple_dim: 0,
        norm: NormPlacement::Pre,

        norm_unit_offset: false,
        v_norm: false,

        norm_topk_prob: f.moe.norm_topk_prob,
        routed_scaling: f.moe.routed_scaling,
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),

        advertised: Advertised::default(),

        rope_scaling: None,
        towers: Towers::default(),
    }
}

#[must_use]
fn mamba_shape(f: &NemotronHFacts) -> RecurrentShape {
    let m = &f.mamba;
    RecurrentShape {
        linear_layers: f.mamba_layers(),
        conv_stride: (m.conv_kernel * m.conv_dim()) as usize,
        state_stride: (m.num_heads * m.head_dim * m.state_size) as usize,

        state_elem: 2,

        k_h: 0,
        v_h: m.num_heads as i32,
        k_d: m.state_size as i32,
        v_d: m.head_dim as i32,
        conv_dim: m.conv_dim() as i32,
        conv_k: m.conv_kernel as i32,
        n_groups: m.n_groups as i32,
    }
}

pub const NO_METAL: &str = "nemotron-h has no Metal text in this build: its forward is `nemotron_h_cuda`, \
     a hybrid of Mamba-2 state-space layers and attention layers, and the one \
     Metal text here (`llama_like_metal`) states attention only — it has no \
     recurrent layer kind and takes a different shape; the CUDA backend serves \
     this row";

#[must_use]
pub fn trace(
    f: &NemotronHFacts,
    class: model_ir::trace::FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> model_ir::trace::ForwardPlan {

    super::forward::nemotron_h_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedW2,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, class, norm_eps, rope_theta)
}
