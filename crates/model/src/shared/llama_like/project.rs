use crate::catalog::{Backend, Deployed, MetalBinding};
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, round_up_attn_head_dim,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::LlamaLikeFacts;

use model_ir::facts::{NormPlacement as SpecNorm, QkNorm};

#[must_use]
pub fn manifest(f: &LlamaLikeFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let (q, kv) = (u64::from(f.q_width()), u64::from(f.kv_width()));
    let head_dim = u64::from(f.head_dim);
    let dense = f.n_experts == 0;

    Manifest::new(f.layers)
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

        .with(match f.qk_norm {
            QkNorm::Off => TensorSpec::absent("layer.{}.self_attn.q_norm"),
            QkNorm::PerHead => TensorSpec::required("layer.{}.self_attn.q_norm", [head_dim]),
            QkNorm::Global => TensorSpec::required("layer.{}.self_attn.q_norm", [q]),
        })

        .either(
            f.norm_placement == SpecNorm::Pre,
            "layer.{}.input_layernorm",
            [hidden],
        )
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ))
        .either(
            f.norm_placement == SpecNorm::Post,
            "layer.{}.post_feedforward_layernorm",
            [hidden],
        )
        .with_if(
            f.qkv_bias,
            TensorSpec::required("layer.{}.self_attn.q_proj.bias", [q]),
        )
        .with_if(
            dense,
            TensorSpec::required(
                "layer.{}.mlp.gate_proj",
                [u64::from(f.intermediate), hidden],
            ),
        )
        .with_if(
            dense,
            TensorSpec::required(
                "layer.{}.mlp.down_proj",
                [hidden, u64::from(f.intermediate)],
            ),
        )
        .with_if(
            !dense,
            TensorSpec::required("layer.{}.mlp.gate", [u64::from(f.n_experts), hidden]),
        )
        .with_if(
            !dense,
            TensorSpec::present("layer.{}.mlp.experts.0.gate_proj"),
        )
}

#[must_use]
pub fn deployment(f: &LlamaLikeFacts, row: RowScalars) -> Deployment {
    let RowScalars {
        rope_theta,
        norm_eps,
        window: sliding_window,
        norm_topk_prob,
        ..
    } = row;
    let head_dim = round_up_attn_head_dim(f.head_dim);
    let attention = (0..f.layers)
        .map(|l| LayerAttention {

            kv_heads: f.kv_heads,
            head_dim,
            window: sliding_window,

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
            head_dim: f.head_dim,
            head_dim_kernel: round_up_attn_head_dim(f.head_dim),
            intermediate: f.intermediate,
            moe_intermediate: f.moe_intermediate,
            experts_per_token: f.experts_per_token,
            shared_intermediate: f.shared_intermediate,
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
        norm: match f.norm_placement {
            SpecNorm::Post => NormPlacement::Post,
            SpecNorm::Pre | SpecNorm::Sandwich => NormPlacement::Pre,
        },

        norm_unit_offset: false,
        v_norm: false,
        norm_topk_prob,

        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),

        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

#[must_use]
pub fn cuda_facts(
    f: &LlamaLikeFacts,
    load: Deployed<'_>,
) -> super::forward::facts::LlamaLikeCudaFacts {
    let kernel = round_up_attn_head_dim(f.head_dim);
    super::forward::facts::LlamaLikeCudaFacts {
        xqa_decode: false,
        decode_fused_post: false,
        rope_table: true,
        force_prefill_path: false,
        head_dim_padded: kernel != f.head_dim,
        head_dim_kernel: if kernel == f.head_dim { 0 } else { kernel },
        gate_up_fused: true,
        proj_repr: model_dsl::WeightRepr::Bf16,
        tp_size: load.tp_size.max(1),
        window_left: Vec::new(),
        all_reduce_p2p_max_rows: 0,
    }
}

pub const QMM_TILE: (u32, u32) = (32, 32);

#[must_use]
pub const fn qmm_fp16_precast(group: u32, bits: u32) -> bool {
    bits == 4 && group == 64
}

pub const ROUTED_QMM_TILE: (u32, u32) = (32, 64);

pub const NO_METAL_SHARD: &str = "this Metal load states a tensor-parallel width above one and \
     `LlamaLikeMetalFacts` has no shard vocabulary: the CUDA facts carry \
     a `tp_size` that narrows every projection width in the text, and the \
     Metal ones carry nothing, so the text would state the WHOLE model's \
     widths against one rank's slice of the weights and read past the end \
     of every projection. Refused rather than traced, because a shard \
     read at full width is arithmetic that runs";

pub const METAL_SDPA_HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

pub const NO_METAL_HEAD_DIM: &str = "this row's heads are a width `sdpa_paged.metal` does not \
     instantiate: the shader compiles the paged decode at 64, 128, 256 \
     and 512, and the text names `sdpa_paged_decode_bfloat16_d_<width>` \
     from the row. CUDA pads such a row to `head_dim_kernel` and strips \
     the pad back off; Metal has no pad kernel in the text, so the \
     choices here are a symbol no shader exports or an attention that \
     reads 32 columns of whatever the loader staged next. Refused";

pub const METAL_ROUTED_AFFINE: (u32, u32) = (64, 4);

pub const NO_METAL_ROUTED_ENCODING: &str = "this row's expert bank reached the device at an affine \
     point `quant/qmv.metal` does not instantiate the routed matvec at: \
     the shader compiles `affine_qmv_routed` only at group 64 / 4 bits, \
     because `AffineQ::group_size` is a template constant. A bank at \
     another group dequantised by that kernel reads every scale from the \
     wrong offset and answers bf16 garbage, which is NaN more often than \
     not. Refused";

pub const NO_METAL_NORMED_LANDING_BIAS: &str = "this row publishes a bias on its attention landing AND norms that \
     landing's output, and the Metal text adds the landing bias only on \
     the arm that fuses the residual into the projection. The normed \
     arms land through a shared statement that has no bias in it, so \
     this row would load the tensor, stage it, and never sum it. \
     Refused rather than dropped";

pub fn metal_kernel_refusal(
    f: &LlamaLikeFacts,
    m: &super::forward::facts::LlamaLikeMetalFacts,
    load: Deployed<'_>,
    bind: &MetalBinding,
) -> Result<(), crate::deployment::Refusal> {
    if load.tp_size > 1 {
        return Err(crate::deployment::Refusal::Unsupported(NO_METAL_SHARD));
    }
    if !METAL_SDPA_HEAD_DIMS.contains(&f.head_dim) {
        return Err(crate::deployment::Refusal::Unsupported(NO_METAL_HEAD_DIM));
    }
    if m.global_head_dim > 0 && !METAL_SDPA_HEAD_DIMS.contains(&m.global_head_dim) {
        return Err(crate::deployment::Refusal::Unsupported(NO_METAL_HEAD_DIM));
    }

    if f.o_bias && f.norm_placement != SpecNorm::Pre {
        return Err(crate::deployment::Refusal::Unsupported(
            NO_METAL_NORMED_LANDING_BIAS,
        ));
    }
    if f.n_experts > 0
        && !bind.moe_mxfp4
        && (bind.quant_group, bind.quant_bits) != METAL_ROUTED_AFFINE
    {
        return Err(crate::deployment::Refusal::Unsupported(
            NO_METAL_ROUTED_ENCODING,
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowScalars {

    pub rope_theta: f32,

    pub norm_eps: f32,

    pub window: i32,

    pub rope_rescaled: bool,

    pub norm_topk_prob: bool,
}

#[must_use]
pub fn metal_facts(
    row: RowScalars,
    load: Deployed<'_>,
    bind: &MetalBinding,
) -> super::forward::facts::LlamaLikeMetalFacts {

    let _ = load;
    super::forward::facts::LlamaLikeMetalFacts {

        fuse_residual_gemv: bind.fuse_residual_gemv,
        paged_multi_batch: bind.paged_multi_batch,
        qmm_multi_batch: bind.qmm_multi_batch,
        add_bias: bind.add_bias,
        fused_qk_rope: bind.fused_qk_rope,

        proj_repr: model_dsl::WeightRepr::Scaled {
            layout: model_dsl::ScaleLayout::PerGroup,
            group: bind.quant_group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: bind.quant_bits,

        router_repr: (bind.router_quant_group != 0).then_some(model_dsl::WeightRepr::Scaled {
            layout: model_dsl::ScaleLayout::PerGroup,
            group: bind.router_quant_group,
            axis: 0,
            zero_point: true,
        }),
        router_bits: bind.router_quant_bits,
        moe_repr: bind.moe_mxfp4.then_some(model_dsl::WeightRepr::Mxfp4Marlin),

        moe_bits: 4,

        qmm_tile: bind.qmm_tile.unwrap_or(QMM_TILE),
        qmm_partial_rows: bind.qmm_partial_rows,

        qmm_fp16_precast: bind.qmm_fp16_precast
            && qmm_fp16_precast(bind.quant_group, bind.quant_bits),

        routed_qmm_fp16: false,
        moe_tile: Some(ROUTED_QMM_TILE),

        gate_up_fused: false,

        rms_eps: row.norm_eps,
        rope_theta: row.rope_theta,

        rope_theta_sliding: 0.0,

        global_head_dim: 0,
        global_kv_heads: 0,
        full_partial_rotary: 0.0,

        v_from_k: false,

        v_norm: false,

        dense_beside_moe: false,

        router_input_norm: false,
        router_expert_scale: false,

        norm_topk_prob: row.norm_topk_prob,

        per_layer_scalar: false,

        embed_scale: 0.0,

        attn_scale: 0.0,

        per_layer_emb_dim: 0,

        kv_shared_layers: 0,

        logit_softcap: 0.0,

        attn_sinks: false,

        activation: super::forward::facts::Activation::SiluMul,

        rope_freq_table: row.rope_rescaled,

        rope_proportional: false,

        window_left: if row.window < 0 {
            Vec::new()
        } else {
            vec![row.window]
        },
    }
}

pub fn trace(
    f: &LlamaLikeFacts,
    row: RowScalars,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
    match load.backend {

        Backend::Cuda => Ok(super::forward::llama_like_cuda::<
            super::forward::ShippedA,
            super::forward::ShippedKv,
        >(
            f,
            &cuda_facts(f, load),
            class,
            row.norm_eps,
            row.rope_theta,
        )),
        Backend::Metal(bind) => {
            let m = metal_facts(row, load, bind);
            metal_kernel_refusal(f, &m, load, bind)?;
            Ok(super::forward::llama_like_metal(f, &m, class))
        }
    }
}
