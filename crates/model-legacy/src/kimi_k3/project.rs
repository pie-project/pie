use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, RecurrentShape, Refusal,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::KimiK3Facts;

#[must_use]
fn shared_width(mla: Option<u64>, kda: Option<u64>) -> Option<u64> {
    match (mla, kda) {
        (Some(a), Some(b)) if a == b => Some(a),
        (Some(w), None) | (None, Some(w)) => Some(w),
        _ => None,
    }
}

#[must_use]
pub fn manifest(f: &KimiK3Facts, tied_embeddings: bool) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let a = &f.attn;
    let k = &f.kda;
    let latent_q = a.q_lora_rank > 0;
    let q_lora = u64::from(a.q_lora_rank);
    let kv_lora = u64::from(a.kv_lora_rank);
    let q_b_width = u64::from(a.q_b_width());
    let kv_a_width = u64::from(a.kv_a_width());

    let kv_b_width = u64::from(a.heads * (a.qk_nope_head_dim + a.v_head_dim));
    let v_width = u64::from(a.v_width());
    let kda_width = u64::from(k.width());
    let dense_inter = u64::from(f.dense_intermediate);
    let has_dense_prefix = f.dense_layers > 0;
    let all_dense = f.dense_layers >= f.layers;

    let has_mla = (0..f.layers).any(|l| f.is_full_attn(l));
    let has_kda = (0..f.layers).any(|l| !f.is_full_attn(l));

    let q_width = shared_width(
        (has_mla && !latent_q).then_some(q_b_width),
        has_kda.then_some(kda_width),
    );
    let o_width = shared_width(has_mla.then_some(v_width), has_kda.then_some(kda_width));
    let g_width = shared_width(
        (has_mla && a.output_gate).then_some(v_width),
        has_kda.then_some(kda_width),
    );

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
        .either(
            has_mla && latent_q,
            "layer.{}.self_attn.q_a_proj",
            [q_lora, hidden],
        )
        .either(
            has_mla && latent_q,
            "layer.{}.self_attn.q_b_proj",
            [q_b_width, q_lora],
        )
        .with_if(
            has_mla && latent_q && !has_kda,
            TensorSpec::absent("layer.{}.self_attn.q_proj"),
        )
        .with_if(
            q_width.is_some(),
            TensorSpec::required(
                "layer.{}.self_attn.q_proj",
                [q_width.unwrap_or_default(), hidden],
            ),
        )
        .with_if(
            has_mla && latent_q,
            TensorSpec::required("layer.{}.self_attn.q_a_layernorm", [q_lora]),
        )
        .either(
            has_mla,
            "layer.{}.self_attn.kv_a_proj_with_mqa",
            [kv_a_width, hidden],
        )
        .either(has_mla, "layer.{}.self_attn.kv_a_layernorm", [kv_lora])
        .either(
            has_mla,
            "layer.{}.self_attn.kv_b_proj",
            [kv_b_width, kv_lora],
        )
        .with_if(
            o_width.is_some(),
            TensorSpec::required(
                "layer.{}.self_attn.o_proj",
                [hidden, o_width.unwrap_or_default()],
            ),
        )
        .with_if(
            has_mla && !a.output_gate && !has_kda,
            TensorSpec::absent("layer.{}.self_attn.g_proj"),
        )
        .with_if(
            g_width.is_some(),
            TensorSpec::required(
                "layer.{}.self_attn.g_proj",
                [g_width.unwrap_or_default(), hidden],
            ),
        )
        .either(has_kda, "layer.{}.self_attn.k_proj", [kda_width, hidden])
        .either(has_kda, "layer.{}.self_attn.v_proj", [kda_width, hidden])
        .either(
            has_kda,
            "layer.{}.self_attn.f_a_proj",
            [u64::from(k.value_head_dim), hidden],
        )
        .either(
            has_kda,
            "layer.{}.self_attn.f_b_proj",
            [kda_width, u64::from(k.value_head_dim)],
        )
        .either(
            has_kda,
            "layer.{}.self_attn.b_proj",
            [u64::from(k.value_heads), hidden],
        )
        .either(
            has_kda,
            "layer.{}.self_attn.o_norm",
            [u64::from(k.value_head_dim)],
        )
        .with_if(
            has_kda,
            TensorSpec::optional("layer.{}.self_attn.A_log", [u64::from(k.value_heads)]),
        )
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
            "layer.{}.block_sparse_moe.gate",
            [u64::from(f.moe.num_experts), hidden],
        )
        .with_if(
            f.moe.has_shared_expert(),
            TensorSpec::required(
                "layer.{}.block_sparse_moe.shared_expert.gate_proj",
                [u64::from(f.moe.shared_intermediate), hidden],
            ),
        )
        .either(
            f.attn_res_block > 0,
            "layer.{}.self_attention_res_proj",
            [1, hidden],
        )
        .either(
            f.attn_res_block > 0,
            "layer.{}.self_attention_res_norm",
            [hidden],
        )
}

pub fn deployment(
    f: &KimiK3Facts,
    rope_theta: f32,
    norm_eps: f32,
    advertised: Advertised,
) -> Result<Deployment, Refusal> {
    let planned = plan(f, rope_theta, norm_eps, advertised);
    planned.provisioned()
}

#[must_use]
fn plan(f: &KimiK3Facts, rope_theta: f32, norm_eps: f32, advertised: Advertised) -> Deployment {
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

            rope_theta: if f.is_full_attn(l) { 0.0 } else { rope_theta },
            rotary_dim: 0,
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

        recurrent: Some(kda_shape(f)),
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

#[must_use]
fn kda_shape(f: &KimiK3Facts) -> RecurrentShape {
    let k = &f.kda;
    let head_dim = k.value_head_dim;
    RecurrentShape {
        linear_layers: (0..f.layers).filter(|l| !f.is_full_attn(*l)).collect(),
        conv_stride: (k.conv_kernel * k.width()) as usize,
        state_stride: (k.value_heads * head_dim * head_dim) as usize,

        state_elem: 2,
        k_h: k.value_heads as i32,
        v_h: k.value_heads as i32,
        k_d: head_dim as i32,
        v_d: head_dim as i32,

        conv_dim: (3 * k.width()) as i32,
        conv_k: k.conv_kernel as i32,

        n_groups: 0,
    }
}

pub const NO_METAL: &str = "kimi-k3 has no Metal text in this build: its forward is `kimi_k3_cuda` — \
     latent attention beside the KDA recurrence, which carries state across \
     tokens — and the one Metal text here (`llama_like_metal`) has no recurrent \
     layer kind and takes a different shape; the CUDA backend serves this row";

pub fn trace(
    f: &KimiK3Facts,
    class: model_ir::trace::FireClass,
    norm_eps: f32,
) -> Result<model_ir::trace::ForwardPlan, Refusal> {
    if f.attn.output_gate {
        return Err(Refusal::Unsupported(
            "kimi_k3: the MLA output gate is not stated by this build's text — \
             the semantic SigmoidGateMul wants equal Shapes and MLA's absorb is \
             rank-3",
        ));
    }

    Ok(super::forward::kimi_k3_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedW2,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, class, norm_eps))
}
