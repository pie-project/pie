use crate::catalog::Deployed;
use crate::deployment::round_up_attn_head_dim;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, RecurrentShape,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::{Qwen35HybridFacts, Qwen35MlpKind};

#[must_use]
pub fn manifest(f: &Qwen35HybridFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden()), u64::from(f.vocab));
    let a = &f.attn;
    let g = &f.gdn;

    let q2 = u64::from(2 * a.q_width());
    let kv = u64::from(a.kv_width());
    let head_dim = u64::from(a.head_dim);
    let (conv_dim, v_width) = (u64::from(g.conv_dim()), u64::from(g.value_width()));
    let v_heads = u64::from(g.value_heads);

    let m = Manifest::new(f.layers)

        .holds_projections_as(<super::forward::ShippedW1 as model_dsl::axes::DtypeAxis>::REPR)
        .holds_experts_as(<super::forward::ShippedW2 as model_dsl::axes::DtypeAxis>::REPR)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        .tie(f.tied_embeddings, "lm_head", [vocab, hidden])

        .with(TensorSpec::required(
            "layer.{}.self_attn.q_proj",
            [q2, hidden],
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
            [hidden, u64::from(a.q_width())],
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
            "layer.{}.linear_attn.in_proj_qkv",
            [conv_dim, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.in_proj_z",
            [v_width, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.in_proj_b",
            [v_heads, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.in_proj_a",
            [v_heads, hidden],
        ))

        .with(TensorSpec::required(
            "layer.{}.linear_attn.conv1d",
            [conv_dim, 1, u64::from(g.conv_kernel)],
        ))

        .with(TensorSpec::required(
            "layer.{}.linear_attn.A_log",
            [v_heads],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.dt_bias",
            [v_heads],
        ))

        .with(TensorSpec::required(
            "layer.{}.linear_attn.norm",
            [u64::from(g.value_head_dim)],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.out_proj",
            [hidden, v_width],
        ))
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ));

    match &f.mlp {
        Qwen35MlpKind::Dense { intermediate } => {
            let i = u64::from(*intermediate);
            m.with(TensorSpec::required("layer.{}.mlp.gate_proj", [i, hidden]))
                .with(TensorSpec::required("layer.{}.mlp.up_proj", [i, hidden]))
                .with(TensorSpec::required("layer.{}.mlp.down_proj", [hidden, i]))

                .with(TensorSpec::absent("layer.{}.mlp.gate"))
        }
        Qwen35MlpKind::Moe(moe) => {
            let shared = u64::from(moe.shared_expert_intermediate);
            m.with(TensorSpec::required(
                "layer.{}.mlp.gate",
                [u64::from(moe.num_experts), hidden],
            ))

            .with(
                TensorSpec::present("layer.{}.mlp.experts.0.gate_proj")
                    .or_published_as([("layer.{}.mlp.switch_mlp.gate_proj", [0u64; 0])]),
            )
            .with(
                TensorSpec::present("layer.{}.mlp.experts.0.down_proj")
                    .or_published_as([("layer.{}.mlp.switch_mlp.down_proj", [0u64; 0])]),
            )
            .either(
                shared != 0,
                "layer.{}.mlp.shared_expert.gate_proj",
                [shared, hidden],
            )
            .with(TensorSpec::absent("layer.{}.mlp.gate_proj"))
        }
    }
}

#[must_use]
pub fn deployment(f: &Qwen35HybridFacts, rope_theta: f32, norm_eps: f32) -> Deployment {
    let a = &f.attn;
    let head_dim = round_up_attn_head_dim(a.head_dim);
    let attention = (0..f.layers)
        .map(|l| LayerAttention {

            kv_heads: a.kv_heads,
            head_dim,
            window: -1,
            kv_source: l,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            rope_theta,

            rotary_dim: a.rotary_dim,

            q_gate: true,
        })
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden(),
            q_heads: a.q_heads,
            kv_heads: a.kv_heads,
            head_dim: a.head_dim,
            head_dim_kernel: head_dim,

            intermediate: match &f.mlp {
                Qwen35MlpKind::Dense { intermediate } => *intermediate,
                Qwen35MlpKind::Moe(_) => 0,
            },
            moe_intermediate: match &f.mlp {
                Qwen35MlpKind::Dense { .. } => 0,
                Qwen35MlpKind::Moe(moe) => moe.moe_intermediate,
            },
            experts_per_token: match &f.mlp {
                Qwen35MlpKind::Dense { .. } => 0,
                Qwen35MlpKind::Moe(moe) => moe.top_k,
            },
            shared_intermediate: match &f.mlp {
                Qwen35MlpKind::Dense { .. } => 0,
                Qwen35MlpKind::Moe(moe) => moe.shared_expert_intermediate,
            },
            vocab: f.vocab,
        },
        attention,

        kv: KvStyle::Paged,
        recurrent: Some(gdn_shape(f)),
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
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),

        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

#[must_use]
fn gdn_shape(f: &Qwen35HybridFacts) -> RecurrentShape {
    let g = &f.gdn;
    RecurrentShape {
        linear_layers: (0..f.layers).filter(|&l| !f.is_full_attn(l)).collect(),
        conv_stride: (g.conv_kernel * g.conv_dim()) as usize,
        state_stride: (g.value_heads * g.key_head_dim * g.value_head_dim) as usize,

        state_elem: 2,
        k_h: g.key_heads as i32,
        v_h: g.value_heads as i32,
        k_d: g.key_head_dim as i32,
        v_d: g.value_head_dim as i32,
        conv_dim: g.conv_dim() as i32,
        conv_k: g.conv_kernel as i32,

        n_groups: 0,
    }
}

#[must_use]
pub fn cuda_facts(
    f: &Qwen35HybridFacts,
    load: Deployed<'_>,
) -> super::forward::facts::Qwen35CudaFacts {
    let moe = matches!(f.mlp, Qwen35MlpKind::Moe(_));
    let shared_gate = match &f.mlp {
        Qwen35MlpKind::Moe(m) => m.shared_expert_intermediate != 0,
        Qwen35MlpKind::Dense { .. } => false,
    };
    super::forward::facts::Qwen35CudaFacts {
        state_bf16: true,

        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,

        moe_cutlass_max_rows: 0,
        prefill_decode: true,
        moe_residual_fold: moe && load.tp_size.max(1) == 1,
        moe_shared_gate_dot: shared_gate,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
        proj_repr: model_dsl::WeightRepr::Bf16,

        window_left: Vec::new(),
    }
}

#[must_use]
pub fn metal_facts(
    f: &Qwen35HybridFacts,
    rope_theta: f32,
    norm_eps: f32,
    bind: &crate::catalog::MetalBinding,
) -> super::forward::metal::Qwen35MetalFacts {
    use model_dsl::{ScaleLayout, WeightRepr};
    super::forward::metal::Qwen35MetalFacts {
        proj_repr: WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: bind.quant_group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: bind.quant_bits,
        moe_repr: bind.moe_mxfp4.then_some(WeightRepr::Mxfp4Marlin),
        moe_bits: 4,

        moe_tile: Some(crate::shared::llama_like::project::ROUTED_QMM_TILE),

        router_repr: (bind.router_quant_group != 0).then_some(WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: bind.router_quant_group,
            axis: 0,
            zero_point: true,
        }),
        router_bits: bind.router_quant_bits,
        qmm_tile: crate::shared::llama_like::project::QMM_TILE,
        qmm_fp16_precast: crate::shared::llama_like::project::qmm_fp16_precast(
            bind.quant_group,
            bind.quant_bits,
        ),

        routed_qmm_fp16: !bind.moe_mxfp4
            && crate::shared::llama_like::project::qmm_fp16_precast(
                bind.quant_group,
                bind.quant_bits,
            ),
        qmm_multi_batch: bind.qmm_multi_batch,
        fuse_residual_gemv: bind.fuse_residual_gemv,
        rms_eps: norm_eps,
        rope_theta,
        attn_scale: 1.0 / (f.attn.head_dim as f32).sqrt(),
        norm_topk_prob: true,
    }
}

pub fn metal_kernel_refusal(
    f: &Qwen35HybridFacts,
    load: Deployed<'_>,
    bind: &crate::catalog::MetalBinding,
) -> Result<(), crate::deployment::Refusal> {
    use crate::deployment::Refusal;
    use crate::shared::llama_like::project as ll;

    if load.tp_size > 1 {
        return Err(Refusal::Unsupported(ll::NO_METAL_SHARD));
    }

    if !ll::METAL_SDPA_HEAD_DIMS.contains(&f.attn.head_dim) {
        return Err(Refusal::Unsupported(ll::NO_METAL_HEAD_DIM));
    }

    if matches!(f.mlp, Qwen35MlpKind::Moe(_))
        && !bind.moe_mxfp4
        && (bind.quant_group, bind.quant_bits) != ll::METAL_ROUTED_AFFINE
    {
        return Err(Refusal::Unsupported(ll::NO_METAL_ROUTED_ENCODING));
    }
    Ok(())
}

#[must_use]
pub fn trace_metal(
    f: &Qwen35HybridFacts,
    class: model_ir::trace::FireClass,
    rope_theta: f32,
    norm_eps: f32,
    bind: &crate::catalog::MetalBinding,
) -> model_ir::trace::ForwardPlan {
    super::forward::metal::qwen3_5_hybrid_metal(
        f,
        &metal_facts(f, rope_theta, norm_eps, bind),
        class,
    )
}

#[must_use]
pub fn trace(
    f: &Qwen35HybridFacts,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
    norm_eps: f32,
    rope_theta: f32,
) -> model_ir::trace::ForwardPlan {

    super::forward::qwen3_5_hybrid_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedW2,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, &cuda_facts(f, load), class, norm_eps, rope_theta)
}
