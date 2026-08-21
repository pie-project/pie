use model_dsl::axes::DtypeAxis;
use std::collections::BTreeMap;

use crate::catalog::Deployed;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, Towers,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::{Gemma4Facts, Gemma4Mixture};

pub const ROPE_THETA_LOCAL: f32 = 10_000.0;

pub const ROPE_THETA_GLOBAL: f32 = 1_000_000.0;

#[must_use]
pub fn manifest(f: &Gemma4Facts, mixture: Option<Gemma4Mixture>) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));

    let head_dim = u64::from(f.head_dim_of(0));
    let q = u64::from(f.q_heads) * head_dim;
    let kv = u64::from(f.kv_heads) * head_dim;
    let inter = u64::from(f.intermediate_of(0));
    let ple = u64::from(f.ple_dim);
    let layers = u64::from(f.layers);
    let has_ple = f.ple_dim > 0;

    Manifest::new(f.layers)

        .holds_projections_as(<super::forward::ShippedW1 as DtypeAxis>::REPR)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))

        .tie(f.tied_embeddings, "lm_head", [vocab, hidden])

        .with_if(
            has_ple,
            TensorSpec::required("embed_tokens_per_layer", [vocab, layers * ple]),
        )
        .with_if(
            has_ple,
            TensorSpec::required("per_layer_model_projection", [layers * ple, hidden]),
        )
        .with_if(
            has_ple,
            TensorSpec::required("per_layer_projection_norm", [ple]),
        )
        .with_if(!has_ple, TensorSpec::absent("embed_tokens_per_layer"))
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

        .with(TensorSpec::absent("layer.{}.self_attn.v_norm"))
        .with(TensorSpec::absent("layer.{}.altup.modality_router"))
        .with(TensorSpec::absent("layer.{}.laurel.linear_left"))

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

        .with_if(
            has_ple,
            TensorSpec::required("layer.{}.per_layer_input_gate", [ple, hidden]),
        )
        .with_if(
            has_ple,
            TensorSpec::required("layer.{}.per_layer_projection", [hidden, ple]),
        )
        .with_if(
            has_ple,
            TensorSpec::required("layer.{}.post_per_layer_input_norm", [hidden]),
        )

        .either(mixture.is_some(), "layer.{}.router.scale", [hidden])
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowScalars {

    pub mixture: Option<Gemma4Mixture>,

    pub sliding_window: i32,

    pub norm_eps: f32,

    pub k_eq_v: bool,
}

#[must_use]
pub fn deployment(f: &Gemma4Facts, row: RowScalars, load: Deployed<'_>) -> Deployment {
    let RowScalars {
        mixture,
        sliding_window,
        norm_eps,

        k_eq_v: _,
    } = row;
    let attention = (0..f.layers)
        .map(|l| layer_attention(f, sliding_window, l))
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.q_heads,
            kv_heads: f.kv_heads,

            head_dim: f.head_dim,

            head_dim_kernel: f.head_dim,
            intermediate: f.intermediate,

            moe_intermediate: mixture.map_or(0, |m| m.moe_intermediate),
            experts_per_token: mixture.map_or(0, |m| m.experts_per_token),
            shared_intermediate: 0,
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Paged,

        recurrent: None,
        prefill: PrefillStyle::Planless,
        attn_output: AttnOutput::StatedArgs,
        logit_softcap: f.logit_softcap,

        attn_logit_softcap: 0.0,

        ple_dim: f.ple_dim,
        norm: NormPlacement::Pre,

        norm_unit_offset: false,

        v_norm: true,

        norm_topk_prob: true,

        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::GeluTanh,
        scales: scales(f, load),

        advertised: Advertised::default(),

        rope_scaling: None,
        towers: Towers::default(),
    }
}

fn layer_attention(f: &Gemma4Facts, sliding_window: i32, l: u32) -> LayerAttention {
    let full = f.is_full_attn(l);
    LayerAttention {
        head_dim: f.head_dim_of(l),

        kv_heads: f.kv_heads_of(l),

        window: if full { -1 } else { sliding_window.max(0) },

        kv_source: f.kv_source(l).unwrap_or(l),

        sm_scale: 1.0,
        rope_theta: if full {
            ROPE_THETA_GLOBAL
        } else {
            ROPE_THETA_LOCAL
        },

        rotary_dim: if full {
            f.global_rotary_dim
        } else {
            f.head_dim
        },
        q_gate: false,
    }
}

fn scales(f: &Gemma4Facts, load: Deployed<'_>) -> BTreeMap<String, f32> {
    let mut scales = BTreeMap::new();
    let hidden = f.hidden as f32;
    scales.insert("sqrt_hidden".to_string(), hidden.sqrt());
    scales.insert("sqrt_ple_dim".to_string(), (f.ple_dim as f32).sqrt());
    scales.insert("rsqrt_hidden".to_string(), 1.0 / hidden.sqrt());
    scales.insert("rsqrt_2".to_string(), 1.0 / 2f32.sqrt());
    for (n, scalar) in load.layer_scalars.iter().enumerate() {
        scales.insert(format!("layer.{n}.ple_norm"), *scalar);
    }
    scales
}

#[must_use]
pub fn metal_shape(
    f: &Gemma4Facts,
    mixture: Option<Gemma4Mixture>,
) -> crate::shared::llama_like::spec::LlamaLikeFacts {
    use crate::shared::llama_like::spec::LlamaLikeFacts;
    use model_ir::facts::{NormPlacement as SpecNorm, QkNorm};
    use model_ir::trace::{NormVariant, RopeKind};

    LlamaLikeFacts {
        hidden: f.hidden,
        layers: f.layers,
        q_heads: f.q_heads,
        kv_heads: f.kv_heads,
        head_dim: f.head_dim,
        n_experts: mixture.map_or(0, |m| m.num_experts),
        experts_per_token: mixture.map_or(0, |m| m.experts_per_token),
        moe_intermediate: mixture.map_or(0, |m| m.moe_intermediate),

        shared_intermediate: 0,
        intermediate: f.intermediate,
        vocab: f.vocab,

        rope: RopeKind::Standard,

        norm_variant: NormVariant::Plain,
        norm_placement: SpecNorm::Sandwich,
        qk_norm: QkNorm::PerHead,

        fused_qkv: false,
        tied_embeddings: f.tied_embeddings,
        qkv_bias: false,
        o_bias: false,
        router_bias: false,
    }
}

#[must_use]
pub fn metal_facts(
    f: &Gemma4Facts,
    row: RowScalars,
    bind: &crate::catalog::MetalBinding,
) -> crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts {
    let RowScalars {
        mixture,
        sliding_window,
        norm_eps,
        k_eq_v,
    } = row;
    use crate::shared::llama_like::forward::facts::{Activation, LlamaLikeMetalFacts};
    use model_dsl::{ScaleLayout, WeightRepr};

    LlamaLikeMetalFacts {
        qmm_partial_rows: false,

        norm_topk_prob: true,

        fuse_residual_gemv: bind.fuse_residual_gemv,
        paged_multi_batch: bind.paged_multi_batch,
        qmm_multi_batch: bind.qmm_multi_batch,
        add_bias: bind.add_bias,
        fused_qk_rope: bind.fused_qk_rope,
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
        qmm_tile: crate::shared::llama_like::project::QMM_TILE,
        qmm_fp16_precast: bind.qmm_fp16_precast
            && crate::shared::llama_like::project::qmm_fp16_precast(
                bind.quant_group,
                bind.quant_bits,
            ),
        routed_qmm_fp16: crate::shared::llama_like::project::qmm_fp16_precast(
            bind.quant_group,
            bind.quant_bits,
        ),
        moe_tile: Some(crate::shared::llama_like::project::ROUTED_QMM_TILE),

        gate_up_fused: false,
        rms_eps: norm_eps,

        rope_theta: ROPE_THETA_GLOBAL,
        rope_theta_sliding: ROPE_THETA_LOCAL,

        global_head_dim: f.global_head_dim,
        global_kv_heads: f.global_kv_heads,

        full_partial_rotary: f64::from(f.global_rotary_dim) as f32 / f.global_head_dim as f32,

        v_from_k: k_eq_v,

        v_norm: true,

        dense_beside_moe: mixture.is_some(),

        router_input_norm: mixture.is_some(),
        router_expert_scale: mixture.is_some(),

        per_layer_scalar: f.ple_dim == 0,

        embed_scale: (f64::from(f.hidden) as f32).sqrt(),

        attn_scale: 1.0,
        per_layer_emb_dim: f.ple_dim,
        kv_shared_layers: f.kv_shared_layers,
        logit_softcap: f.logit_softcap,

        attn_sinks: false,

        activation: Activation::Geglu,

        rope_freq_table: false,

        rope_proportional: true,

        window_left: (0..f.layers)
            .map(|l| {
                if f.is_full_attn(l) {
                    -1
                } else {
                    sliding_window.max(0)
                }
            })
            .collect(),
    }
}

#[must_use]
pub fn trace(
    f: &Gemma4Facts,
    sliding_window: i32,
    class: model_ir::trace::FireClass,
    layer_scalars: &[f32],
    norm_eps: f32,
) -> model_ir::trace::ForwardPlan {
    let cuda = super::forward::facts::Gemma4CudaFacts {
        fused_qkv: true,
        gate_up_fused: true,
        kv_native_bf16: true,

        layer_scalars: layer_scalars.to_vec(),
        window_left: (0..f.layers)
            .map(|l| {
                if f.is_full_attn(l) {
                    -1
                } else {
                    sliding_window.max(0)
                }
            })
            .collect(),
    };

    super::forward::gemma4_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, &cuda, class, norm_eps)
}
