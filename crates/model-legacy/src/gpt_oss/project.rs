use crate::catalog::Deployed;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::GptOssFacts;

#[must_use]
pub fn manifest(f: &GptOssFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let q = u64::from(f.q_heads * f.head_dim);
    let kv = u64::from(f.kv_heads * f.head_dim);
    let experts = u64::from(f.experts);

    Manifest::new(f.layers)
        .holds_projections_as(<super::forward::ShippedW1 as model_dsl::axes::DtypeAxis>::REPR)
        .holds_experts_as(<super::forward::ShippedW2 as model_dsl::axes::DtypeAxis>::REPR)
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
        .with(TensorSpec::required("layer.{}.self_attn.q_proj.bias", [q]))
        .with(TensorSpec::required("layer.{}.self_attn.k_proj.bias", [kv]))
        .with(TensorSpec::required("layer.{}.self_attn.v_proj.bias", [kv]))
        .with(TensorSpec::required(
            "layer.{}.self_attn.o_proj.bias",
            [hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.sinks",
            [u64::from(f.q_heads)],
        ))
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.mlp.router",
            [experts, hidden],
        ))
        .with(TensorSpec::required("layer.{}.mlp.router.bias", [experts]))
        .with(
            TensorSpec::required(
                "layer.{}.mlp.experts.gate_up_proj_bias",
                [experts, u64::from(2 * f.intermediate)],
            )
            .or_published_as([
                (
                    "layer.{}.mlp.experts.gate_proj.bias",
                    [experts, u64::from(f.intermediate)],
                ),
                (
                    "layer.{}.mlp.experts.up_proj.bias",
                    [experts, u64::from(f.intermediate)],
                ),
            ]),
        )
        .with(
            TensorSpec::required("layer.{}.mlp.experts.down_proj_bias", [experts, hidden])
                .or_published_as([("layer.{}.mlp.experts.down_proj.bias", [experts, hidden])]),
        )
}

pub(crate) const GATE_ALPHA: f32 = 1.702;

#[must_use]
pub fn deployment(
    f: &GptOssFacts,
    rope_theta: f32,
    norm_eps: f32,
    sliding_window: i32,
) -> Deployment {
    let head_dim = f.head_dim;
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            kv_heads: f.kv_heads,
            head_dim,
            window: if f.is_sliding(l) { sliding_window } else { -1 },
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
            q_heads: f.q_heads,
            kv_heads: f.kv_heads,
            head_dim,

            head_dim_kernel: head_dim,

            intermediate: 0,
            moe_intermediate: f.intermediate,
            experts_per_token: f.top_k,
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

        norm_unit_offset: false,
        v_norm: false,

        norm_topk_prob: true,

        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::SiluClamped {
            limit: f.swiglu_limit,
            alpha: GATE_ALPHA,
        },
        scales: std::collections::BTreeMap::new(),

        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

#[must_use]
pub fn cuda_facts(f: &GptOssFacts, load: Deployed<'_>) -> super::forward::facts::GptOssCudaFacts {
    let _ = load;
    super::forward::facts::GptOssCudaFacts {
        mxfp4_decode_gemv: true,
        mxfp4_decode_max_routes: 32 * f.experts,
        streamed_experts: false,
    }
}

#[must_use]
pub fn metal_shape(f: &GptOssFacts) -> crate::shared::llama_like::spec::LlamaLikeFacts {
    use crate::shared::llama_like::spec::LlamaLikeFacts;
    use model_ir::facts::{NormPlacement, QkNorm};
    use model_ir::trace::{NormVariant, RopeKind};

    LlamaLikeFacts {
        hidden: f.hidden,
        layers: f.layers,
        q_heads: f.q_heads,
        kv_heads: f.kv_heads,
        head_dim: f.head_dim,
        n_experts: f.experts,
        experts_per_token: f.top_k,

        moe_intermediate: f.intermediate,

        shared_intermediate: 0,

        intermediate: 0,
        vocab: f.vocab,

        rope: RopeKind::Yarn,
        norm_variant: NormVariant::Plain,
        norm_placement: NormPlacement::Pre,
        qk_norm: QkNorm::Off,

        fused_qkv: false,
        tied_embeddings: f.tied_embeddings,

        qkv_bias: true,
        o_bias: true,
        router_bias: true,
    }
}

#[must_use]
pub fn metal_facts(
    f: &GptOssFacts,
    bind: &crate::catalog::MetalBinding,
) -> crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts {
    use crate::shared::llama_like::forward::facts::{Activation, LlamaLikeMetalFacts};
    use model_dsl::{ScaleLayout, WeightRepr};

    LlamaLikeMetalFacts {
        attn_sinks: true,

        activation: Activation::SwiGlu {
            limit: f.swiglu_limit,
            alpha: 1.702,
        },

        rope_theta: 150_000.0,
        rope_freq_table: true,
        rms_eps: 1e-5,

        window_left: (0..f.layers)
            .map(|l| if f.is_sliding(l) { 128 } else { -1 })
            .collect(),

        proj_repr: WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: bind.quant_group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: bind.quant_bits,

        router_repr: (bind.router_quant_group != 0).then_some(WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: bind.router_quant_group,
            axis: 0,
            zero_point: true,
        }),
        router_bits: bind.router_quant_bits,
        moe_repr: bind.moe_mxfp4.then_some(WeightRepr::Mxfp4Marlin),

        moe_bits: 4,
        fuse_residual_gemv: bind.fuse_residual_gemv,
        paged_multi_batch: bind.paged_multi_batch,
        qmm_multi_batch: bind.qmm_multi_batch,
        add_bias: bind.add_bias,

        norm_topk_prob: true,

        attn_scale: yarn_softmax_scale(f.head_dim),
        ..LlamaLikeMetalFacts::synthetic()
    }
}

fn yarn_softmax_scale(head_dim: u32) -> f32 {
    const ATTENTION_FACTOR: f32 = match super::ROPE_SCALING {
        crate::deployment::RopeScaling::Yarn {
            attention_factor, ..
        } => attention_factor,
        _ => panic!("gpt-oss rescales by YaRN; this scale is that factor squared"),
    };
    ATTENTION_FACTOR * ATTENTION_FACTOR / (head_dim as f32).sqrt()
}

#[must_use]
pub fn trace(
    f: &GptOssFacts,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
    norm_eps: f32,
    rope_theta: f32,
    sliding_window: i32,
) -> model_ir::trace::ForwardPlan {
    super::forward::gpt_oss_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedW2,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(
        f,
        &cuda_facts(f, load),
        class,
        norm_eps,
        rope_theta,
        sliding_window,
    )
}
