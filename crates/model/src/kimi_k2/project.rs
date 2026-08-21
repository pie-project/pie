use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, Refusal,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::KimiFacts;

#[must_use]
pub fn manifest(f: &KimiFacts, tied_embeddings: bool) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let a = &f.attn;

    let latent_q = a.q_lora_rank > 0;
    let q_lora = u64::from(a.q_lora_rank);
    let kv_lora = u64::from(a.kv_lora_rank);
    let q_b_width = u64::from(a.q_b_width());
    let kv_a_width = u64::from(a.kv_a_width());

    let kv_b_width = u64::from(a.heads * (a.qk_nope_head_dim + a.v_head_dim));
    let v_width = u64::from(a.v_width());
    let dense_inter = u64::from(f.dense_intermediate);
    let has_dense_prefix = f.dense_layers > 0;
    let all_dense = f.dense_layers >= f.layers;

    Manifest::new(f.layers)

        .holds_projections_as(<super::forward::ShippedW1 as model_dsl::axes::DtypeAxis>::REPR)
        .holds_experts_as(<super::forward::ShippedW2 as model_dsl::axes::DtypeAxis>::REPR)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))

        .tie(tied_embeddings, "lm_head", [vocab, hidden])
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ))

        .either(latent_q, "layer.{}.self_attn.q_a_proj", [q_lora, hidden])
        .either(latent_q, "layer.{}.self_attn.q_b_proj", [q_b_width, q_lora])

        .either(!latent_q, "layer.{}.self_attn.q_proj", [q_b_width, hidden])
        .with_if(
            latent_q,
            TensorSpec::required("layer.{}.self_attn.q_a_layernorm", [q_lora]),
        )
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_a_proj_with_mqa",
            [kv_a_width, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_a_layernorm",
            [kv_lora],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_b_proj",
            [kv_b_width, kv_lora],
        ))

        .with(TensorSpec::required(
            "layer.{}.self_attn.o_proj",
            [hidden, v_width],
        ))

        .either(
            has_dense_prefix,
            "layer.{}.mlp.gate_proj",
            [dense_inter, hidden],
        )
        .either(
            has_dense_prefix,
            "layer.{}.mlp.down_proj",
            [hidden, dense_inter],
        )

        .either(
            !all_dense,
            "layer.{}.mlp.gate",
            [u64::from(f.moe.num_experts), hidden],
        )
        .with_if(
            f.moe.has_shared_expert(),
            TensorSpec::required(
                "layer.{}.mlp.shared_experts.gate_proj",
                [u64::from(f.moe.shared_intermediate), hidden],
            ),
        )
}

pub fn deployment(
    f: &KimiFacts,
    rope_theta: f32,
    norm_eps: f32,
    rope_yarn: bool,
    advertised: Advertised,
) -> Result<Deployment, Refusal> {
    let planned = plan(f, rope_theta, norm_eps, rope_yarn, advertised);
    planned.provisioned()
}

#[must_use]
fn plan(
    f: &KimiFacts,
    rope_theta: f32,
    norm_eps: f32,
    rope_yarn: bool,
    advertised: Advertised,
) -> Deployment {
    let a = &f.attn;

    let page_row = a.kv_a_width();

    let sm_scale = 1.0 / (a.qk_head_dim() as f32).sqrt();
    let attention = (0..f.layers)
        .map(|l| LayerAttention {

            kv_heads: 1,
            head_dim: page_row,

            window: -1,

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

            kv_heads: 1,
            head_dim: page_row,

            head_dim_kernel: page_row,
            intermediate: f.dense_intermediate,

            moe_intermediate: f.moe.moe_intermediate,
            experts_per_token: f.moe.top_k,
            shared_intermediate: f.moe.shared_intermediate,
            vocab: f.vocab,
        },
        attention,

        kv: KvStyle::Mla {
            kv_lora_rank: a.kv_lora_rank,
            qk_rope_head_dim: a.qk_rope_head_dim,
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

        rope_scaling: rope_yarn.then_some(super::ROPE_SCALING),
        towers: Default::default(),
    }
}

#[must_use]
pub fn cuda_facts(rope_yarn_original: bool) -> super::forward::facts::KimiCudaFacts {
    super::forward::facts::KimiCudaFacts {
        q_kv_a_fused: true,
        rope_yarn_original,
    }
}

pub const NO_METAL: &str = "kimi-k2 has no Metal text in this build: its forward is `kimi_cuda` — latent \
     attention over a compressed KV with YaRN on the decoupled rotary half — and \
     the one Metal text here (`llama_like_metal`) states dense paged attention \
     over full K and V, a different cache reached through a different shape; the \
     CUDA backend serves this row";

#[must_use]
pub fn trace(
    f: &KimiFacts,
    rope_yarn_original: bool,
    class: model_ir::trace::FireClass,
    norm_eps: f32,
) -> model_ir::trace::ForwardPlan {

    super::forward::kimi_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedW2,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, &cuda_facts(rope_yarn_original), class, norm_eps)
}
