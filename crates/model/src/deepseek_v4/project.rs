use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, Refusal,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::Dsv4Facts;

#[must_use]
pub fn manifest(f: &Dsv4Facts, tied_embeddings: bool) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let a = &f.attn;
    let q_width = u64::from(a.q_width());
    let q_lora = u64::from(a.q_lora_rank);
    let o_lora = u64::from(a.o_lora_rank);
    let dense_inter = u64::from(f.dense_intermediate);

    let latent_q = a.q_lora_rank > 0;
    let grouped_o = a.o_lora_rank > 0;
    let has_dense_prefix = f.dense_layers > 0;
    let all_dense = f.dense_layers >= f.layers;

    Manifest::new(f.layers)

        .holds_projections_as(<super::forward::ShippedW1 as model_dsl::axes::DtypeAxis>::REPR)
        .holds_experts_as(<super::forward::ShippedW2 as model_dsl::axes::DtypeAxis>::REPR)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))

        .tie(tied_embeddings, "lm_head", [vocab, hidden])

        .with(TensorSpec::required("layer.{}.attn_norm", [hidden]))
        .with(TensorSpec::required("layer.{}.mlp_norm", [hidden]))

        .either(latent_q, "layer.{}.attn.wq_a", [q_lora, hidden])

        .either(latent_q, "layer.{}.attn.q_norm", [q_lora])
        .either(latent_q, "layer.{}.attn.wq_b", [q_width, q_lora])
        .either(!latent_q, "layer.{}.attn.wq", [q_width, hidden])

        .with(TensorSpec::required("layer.{}.attn.wkv", [q_width, hidden]))
        .with(TensorSpec::required("layer.{}.attn.kv_norm", [q_width]))

        .either(grouped_o, "layer.{}.attn.wo_a", [o_lora, q_width])
        .either(grouped_o, "layer.{}.attn.wo_b", [hidden, o_lora])
        .either(!grouped_o, "layer.{}.attn.wo", [hidden, q_width])

        .either(
            has_dense_prefix,
            "layer.{}.mlp.gate_proj",
            [dense_inter, hidden],
        )
        .either(
            has_dense_prefix,
            "layer.{}.mlp.up_proj",
            [dense_inter, hidden],
        )
        .either(
            has_dense_prefix,
            "layer.{}.mlp.down_proj",
            [hidden, dense_inter],
        )
        .either(
            !all_dense,
            "layer.{}.ffn.gate",
            [u64::from(f.moe.num_experts), hidden],
        )
}

pub fn deployment(
    f: &Dsv4Facts,
    rope_theta: f32,
    norm_eps: f32,
    advertised: Advertised,
) -> Result<Deployment, Refusal> {
    let planned = plan(f, rope_theta, norm_eps, advertised);
    planned.provisioned()
}

#[must_use]
fn plan(f: &Dsv4Facts, rope_theta: f32, norm_eps: f32, advertised: Advertised) -> Deployment {
    let a = &f.attn;

    let window = i32::try_from(a.sliding_window).unwrap_or(i32::MAX);
    let window = if window > 0 { window } else { -1 };

    let sm_scale = 1.0 / (a.head_dim as f32).sqrt();
    let attention = (0..f.layers)
        .map(|l| LayerAttention {

            kv_heads: a.heads,
            head_dim: a.head_dim,
            window,

            kv_source: l,
            sm_scale,
            rope_theta,

            rotary_dim: a.qk_rope_head_dim,
            q_gate: false,
        })
        .collect();

    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: a.heads,

            kv_heads: a.heads,
            head_dim: a.head_dim,

            head_dim_kernel: a.head_dim,
            intermediate: f.dense_intermediate,

            moe_intermediate: f.moe.moe_intermediate,
            experts_per_token: f.moe.top_k,
            shared_intermediate: 0,

            vocab: f.vocab,
        },
        attention,

        kv: KvStyle::CompressedPlane {
            ratios: f.ratios.to_vec(),
        },

        recurrent: None,
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

        advertised,

        rope_scaling: None,
        towers: Default::default(),
    }
}

pub const NO_METAL: &str = "deepseek-v4 has no Metal text in this build: its forward is `dsv4_cuda` — \
     multi-head latent attention over a compressed KV, a per-token compression \
     boundary and a 256-expert router — and the one Metal text here \
     (`llama_like_metal`) states none of those and takes a different shape \
     entirely; the CUDA backend serves this row";

#[must_use]
pub fn trace(
    f: &Dsv4Facts,
    class: model_ir::trace::FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> model_ir::trace::ForwardPlan {

    super::forward::dsv4_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedW2,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, class, norm_eps, rope_theta)
}
