#![cfg_attr(rustfmt, rustfmt::skip)]
#![allow(unused_imports)]

use kernels::bound::{Axis, BoundOp, Site};
use kernels::points::{Form, Mxfp4};
use kernels::points::{Attention, Gate, Gemm, Layout, Mlp, Moe, Norm, Rope, Ssm};
use kernels::plane::Refusal;

use crate::points::bfloat;
use crate::plane::Ctx;

pub const CLAIMED: &[(&str, Option<Site>, &[Axis])] = &[
    ("norm.rmsnorm", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_per_head", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_plus_one", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_per_head_plus_one", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_no_scale", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_gated", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_gated_by", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.residual_add", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.add_bias", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.mul_scalar", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.scale", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("mlp.swiglu", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("mlp.swiglu_clamp", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("mlp.swiglu_clamp_alpha", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("mlp.geglu_tanh", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("mlp.geglu_tanh_packed", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("mlp.situ", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("gemm.matmul", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("gemm.lm_head", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("gemm.attention_landing", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("rope.full", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("rope.partial", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("rope.partial_q", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("rope.partial_last", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("rope.yarn", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("moe.topk_softmax", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("moe.topk_sigmoid", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("moe.topk_sqrt_softplus", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("moe.matmul_select", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("moe.matmul_select_bias", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("moe.weighted_sum", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("moe.sigmoid_gate_add", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("gate.sigmoid_mul", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("layout.embed", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("layout.split_qkv", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("layout.split_q_gate", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("layout.split_rows", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("layout.select", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("ssm.causal_conv1d", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("ssm.causal_conv1d_chunked", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("ssm.gdn_prep", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("ssm.gated_delta", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("ssm.gated_delta_chunked", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("ssm.kda_step", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("ssm.kda_chunked", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.decode", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.prefill", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.masked", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.logit_softcap", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.kv_append", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.kv_append_shared", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
];

pub const TIER2: &[(&str, Option<Site>, &[Axis])] = &[
];

pub fn dispatch<'p, B>(ctx: &Ctx<'p>, op: &B) -> Result<(), Refusal>
where
    B: BoundOp<Plane = Ctx<'p>>,
{
    match op.point() {
        "norm.rmsnorm" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.f32(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.rmsnorm::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.f32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_per_head" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_per_head::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.rmsnorm_per_head::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_per_head`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_plus_one" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_plus_one::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.f32(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.rmsnorm_plus_one::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.f32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_plus_one`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_per_head_plus_one" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_per_head_plus_one::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.rmsnorm_per_head_plus_one::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_per_head_plus_one`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_no_scale" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_no_scale::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.rmsnorm_no_scale::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_no_scale`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_gated" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_gated::<bfloat>(op.tin::<f32>(0)?, op.tin::<bfloat>(1)?, op.tconst::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.rmsnorm_gated::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tconst::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_gated`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_gated_by" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_gated_by::<bfloat>(op.tin::<f32>(0)?, op.tin::<bfloat>(1)?, op.tconst::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.rmsnorm_gated_by::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tconst::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_gated_by`, at an element or repr this plane does not instantiate" }),
        },
        "norm.residual_add" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.residual_add::<bfloat>(op.tin::<bfloat>(0)?, op.tinout::<bfloat>(1, 0)?),
            Axis::F32 => ctx.residual_add::<f32>(op.tin::<f32>(0)?, op.tinout::<f32>(1, 0)?),
            _ => Err(Refusal::Absent { what: "`norm.residual_add`, at an element or repr this plane does not instantiate" }),
        },
        "norm.add_bias" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.add_bias::<bfloat>(op.tconst::<bfloat>(0)?, op.tinout::<bfloat>(0, 0)?),
            Axis::F32 => ctx.add_bias::<f32>(op.tconst::<f32>(0)?, op.tinout::<f32>(0, 0)?),
            _ => Err(Refusal::Absent { what: "`norm.add_bias`, at an element or repr this plane does not instantiate" }),
        },
        "norm.mul_scalar" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.mul_scalar::<bfloat>(op.f32(0)?, op.tinout::<bfloat>(0, 0)?),
            Axis::F32 => ctx.mul_scalar::<f32>(op.f32(0)?, op.tinout::<f32>(0, 0)?),
            _ => Err(Refusal::Absent { what: "`norm.mul_scalar`, at an element or repr this plane does not instantiate" }),
        },
        "norm.scale" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.scale::<bfloat>(op.tconst::<bfloat>(0)?, op.tinout::<bfloat>(0, 0)?),
            Axis::F32 => ctx.scale::<f32>(op.tconst::<f32>(0)?, op.tinout::<f32>(0, 0)?),
            _ => Err(Refusal::Absent { what: "`norm.scale`, at an element or repr this plane does not instantiate" }),
        },
        "mlp.swiglu" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.swiglu::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.swiglu::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`mlp.swiglu`, at an element or repr this plane does not instantiate" }),
        },
        "mlp.swiglu_clamp" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.swiglu_clamp::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.swiglu_clamp::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`mlp.swiglu_clamp`, at an element or repr this plane does not instantiate" }),
        },
        "mlp.swiglu_clamp_alpha" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.swiglu_clamp_alpha::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.f32(1)?, op.f32(2)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.swiglu_clamp_alpha::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`mlp.swiglu_clamp_alpha`, at an element or repr this plane does not instantiate" }),
        },
        "mlp.geglu_tanh" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.geglu_tanh::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<bfloat>(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.geglu_tanh::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`mlp.geglu_tanh`, at an element or repr this plane does not instantiate" }),
        },
        "mlp.geglu_tanh_packed" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.geglu_tanh_packed::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.geglu_tanh_packed::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`mlp.geglu_tanh_packed`, at an element or repr this plane does not instantiate" }),
        },
        "mlp.situ" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.situ::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.f32(1)?, op.f32(2)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.situ::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`mlp.situ`, at an element or repr this plane does not instantiate" }),
        },
        "gemm.matmul" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.matmul::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.matmul::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`gemm.matmul`, at an element or repr this plane does not instantiate" }),
        },
        "gemm.lm_head" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.lm_head::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.lm_head::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`gemm.lm_head`, at an element or repr this plane does not instantiate" }),
        },
        "gemm.attention_landing" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.attention_landing::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.layer()?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.attention_landing::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.layer()?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`gemm.attention_landing`, at an element or repr this plane does not instantiate" }),
        },
        "rope.full" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.full::<bfloat>(op.tinout::<bfloat>(0, 0)?, op.tinout::<bfloat>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.f32(1)?, op.bool(2)?),
            Axis::F32 => ctx.full::<f32>(op.tinout::<f32>(0, 0)?, op.tinout::<f32>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.f32(1)?, op.bool(2)?),
            _ => Err(Refusal::Absent { what: "`rope.full`, at an element or repr this plane does not instantiate" }),
        },
        "rope.partial" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.partial::<bfloat>(op.tinout::<bfloat>(0, 0)?, op.tinout::<bfloat>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.u32(1)?, op.f32(2)?),
            Axis::F32 => ctx.partial::<f32>(op.tinout::<f32>(0, 0)?, op.tinout::<f32>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.u32(1)?, op.f32(2)?),
            _ => Err(Refusal::Absent { what: "`rope.partial`, at an element or repr this plane does not instantiate" }),
        },
        "rope.partial_q" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.partial_q::<bfloat>(op.tinout::<bfloat>(0, 0)?, op.tin::<i32>(1)?, op.u32(0)?, op.u32(1)?, op.f32(2)?),
            Axis::F32 => ctx.partial_q::<f32>(op.tinout::<f32>(0, 0)?, op.tin::<i32>(1)?, op.u32(0)?, op.u32(1)?, op.f32(2)?),
            _ => Err(Refusal::Absent { what: "`rope.partial_q`, at an element or repr this plane does not instantiate" }),
        },
        "rope.partial_last" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.partial_last::<bfloat>(op.tinout::<bfloat>(0, 0)?, op.tin::<i32>(1)?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.bool(3)?),
            Axis::F32 => ctx.partial_last::<f32>(op.tinout::<f32>(0, 0)?, op.tin::<i32>(1)?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.bool(3)?),
            _ => Err(Refusal::Absent { what: "`rope.partial_last`, at an element or repr this plane does not instantiate" }),
        },
        "rope.yarn" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.yarn::<bfloat>(op.tinout::<bfloat>(0, 0)?, op.tinout::<bfloat>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.f32(1)?, op.f32(2)?, op.f32(3)?, op.f32(4)?, op.f32(5)?, op.u32(6)?, op.bool(7)?),
            Axis::F32 => ctx.yarn::<f32>(op.tinout::<f32>(0, 0)?, op.tinout::<f32>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.f32(1)?, op.f32(2)?, op.f32(3)?, op.f32(4)?, op.f32(5)?, op.u32(6)?, op.bool(7)?),
            _ => Err(Refusal::Absent { what: "`rope.yarn`, at an element or repr this plane does not instantiate" }),
        },
        "moe.topk_softmax" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.topk_softmax::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.u32(1)?, op.tout::<i32>(0)?, op.tout::<f32>(1)?),
            Axis::F32 => ctx.topk_softmax::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.u32(1)?, op.tout::<i32>(0)?, op.tout::<f32>(1)?),
            _ => Err(Refusal::Absent { what: "`moe.topk_softmax`, at an element or repr this plane does not instantiate" }),
        },
        "moe.topk_sigmoid" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.topk_sigmoid::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.u32(1)?, op.bool(2)?, op.f32(3)?, op.tout::<i32>(0)?, op.tout::<f32>(1)?),
            Axis::F32 => ctx.topk_sigmoid::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.u32(1)?, op.bool(2)?, op.f32(3)?, op.tout::<i32>(0)?, op.tout::<f32>(1)?),
            _ => Err(Refusal::Absent { what: "`moe.topk_sigmoid`, at an element or repr this plane does not instantiate" }),
        },
        "moe.topk_sqrt_softplus" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.topk_sqrt_softplus::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<f32>(0)?, op.u32(0)?, op.u32(1)?, op.bool(2)?, op.f32(3)?, op.tout::<i32>(0)?, op.tout::<f32>(1)?),
            Axis::F32 => ctx.topk_sqrt_softplus::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.u32(0)?, op.u32(1)?, op.bool(2)?, op.f32(3)?, op.tout::<i32>(0)?, op.tout::<f32>(1)?),
            _ => Err(Refusal::Absent { what: "`moe.topk_sqrt_softplus`, at an element or repr this plane does not instantiate" }),
        },
        "moe.matmul_select" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.matmul_select::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.tin::<i32>(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.matmul_select::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.tin::<i32>(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`moe.matmul_select`, at an element or repr this plane does not instantiate" }),
        },
        "moe.matmul_select_bias" => match (op.dtype(Site::Out(0))?, op.form(0)?) {
            (Axis::Bf16, Form::Mxfp4) => ctx.matmul_select_bias::<bfloat, Mxfp4>(op.tin::<bfloat>(0)?, op.bank::<Mxfp4>(0)?, op.tconst::<bfloat>(2)?, op.tin::<i32>(1)?, op.tout::<bfloat>(0)?),
            (Axis::F32, Form::Mxfp4) => ctx.matmul_select_bias::<f32, Mxfp4>(op.tin::<f32>(0)?, op.bank::<Mxfp4>(0)?, op.tconst::<f32>(2)?, op.tin::<i32>(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`moe.matmul_select_bias`, at an element or repr this plane does not instantiate" }),
        },
        "moe.weighted_sum" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.weighted_sum::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<f32>(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.weighted_sum::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`moe.weighted_sum`, at an element or repr this plane does not instantiate" }),
        },
        "moe.sigmoid_gate_add" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.sigmoid_gate_add::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<bfloat>(1)?, op.tin::<bfloat>(2)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.sigmoid_gate_add::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tin::<f32>(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`moe.sigmoid_gate_add`, at an element or repr this plane does not instantiate" }),
        },
        "gate.sigmoid_mul" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.sigmoid_mul::<bfloat>(op.tinout::<bfloat>(0, 0)?, op.tin::<bfloat>(1)?),
            Axis::F32 => ctx.sigmoid_mul::<f32>(op.tinout::<f32>(0, 0)?, op.tin::<f32>(1)?),
            _ => Err(Refusal::Absent { what: "`gate.sigmoid_mul`, at an element or repr this plane does not instantiate" }),
        },
        "layout.embed" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.embed::<bfloat>(op.tin::<i32>(0)?, op.tconst::<bfloat>(0)?, op.u32(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.embed::<f32>(op.tin::<i32>(0)?, op.tconst::<f32>(0)?, op.u32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`layout.embed`, at an element or repr this plane does not instantiate" }),
        },
        "layout.split_qkv" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.split_qkv::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.u32(1)?, op.tout::<bfloat>(0)?, op.tout::<bfloat>(1)?, op.tout::<bfloat>(2)?),
            Axis::F32 => ctx.split_qkv::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.u32(1)?, op.tout::<f32>(0)?, op.tout::<f32>(1)?, op.tout::<f32>(2)?),
            _ => Err(Refusal::Absent { what: "`layout.split_qkv`, at an element or repr this plane does not instantiate" }),
        },
        "layout.split_q_gate" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.split_q_gate::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.tout::<bfloat>(0)?, op.tout::<bfloat>(1)?),
            Axis::F32 => ctx.split_q_gate::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.tout::<f32>(0)?, op.tout::<f32>(1)?),
            _ => Err(Refusal::Absent { what: "`layout.split_q_gate`, at an element or repr this plane does not instantiate" }),
        },
        "layout.split_rows" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.split_rows::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.tout::<bfloat>(0)?, op.tout::<bfloat>(1)?),
            Axis::F32 => ctx.split_rows::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.tout::<f32>(0)?, op.tout::<f32>(1)?),
            _ => Err(Refusal::Absent { what: "`layout.split_rows`, at an element or repr this plane does not instantiate" }),
        },
        "layout.select" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.select::<bfloat>(op.tin::<bfloat>(0)?, op.u32(0)?, op.u32(1)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.select::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.u32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`layout.select`, at an element or repr this plane does not instantiate" }),
        },
        "ssm.causal_conv1d" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.causal_conv1d::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.recurrent()?, op.u32(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.causal_conv1d::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.recurrent()?, op.u32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`ssm.causal_conv1d`, at an element or repr this plane does not instantiate" }),
        },
        "ssm.causal_conv1d_chunked" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.causal_conv1d_chunked::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<i32>(1)?, op.tconst::<bfloat>(0)?, op.recurrent()?, op.u32(0)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.causal_conv1d_chunked::<f32>(op.tin::<f32>(0)?, op.tin::<i32>(1)?, op.tconst::<f32>(0)?, op.recurrent()?, op.u32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`ssm.causal_conv1d_chunked`, at an element or repr this plane does not instantiate" }),
        },
        "ssm.gdn_prep" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.gdn_prep::<bfloat>(op.tin::<bfloat>(0)?, op.tconst::<bfloat>(0)?, op.tconst::<f32>(1)?, op.tout::<f32>(0)?),
            Axis::F32 => ctx.gdn_prep::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.tconst::<f32>(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`ssm.gdn_prep`, at an element or repr this plane does not instantiate" }),
        },
        "ssm.gated_delta" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.gated_delta::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<bfloat>(1)?, op.tin::<f32>(2)?, op.recurrent()?, op.u32(0)?, op.u32(1)?, op.u32(2)?, op.u32(3)?, op.tout::<f32>(0)?),
            Axis::F32 => ctx.gated_delta::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tin::<f32>(2)?, op.recurrent()?, op.u32(0)?, op.u32(1)?, op.u32(2)?, op.u32(3)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`ssm.gated_delta`, at an element or repr this plane does not instantiate" }),
        },
        "ssm.gated_delta_chunked" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.gated_delta_chunked::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<i32>(1)?, op.tin::<bfloat>(2)?, op.tin::<f32>(3)?, op.recurrent()?, op.u32(0)?, op.u32(1)?, op.u32(2)?, op.u32(3)?, op.tout::<f32>(0)?),
            Axis::F32 => ctx.gated_delta_chunked::<f32>(op.tin::<f32>(0)?, op.tin::<i32>(1)?, op.tin::<f32>(2)?, op.tin::<f32>(3)?, op.recurrent()?, op.u32(0)?, op.u32(1)?, op.u32(2)?, op.u32(3)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`ssm.gated_delta_chunked`, at an element or repr this plane does not instantiate" }),
        },
        "ssm.kda_step" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.kda_step::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<bfloat>(1)?, op.tin::<bfloat>(2)?, op.tconst::<f32>(0)?, op.tconst::<f32>(1)?, op.recurrent()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            Axis::F32 => ctx.kda_step::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tin::<f32>(2)?, op.tconst::<f32>(0)?, op.tconst::<f32>(1)?, op.recurrent()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`ssm.kda_step`, at an element or repr this plane does not instantiate" }),
        },
        "ssm.kda_chunked" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.kda_chunked::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<i32>(1)?, op.tin::<bfloat>(2)?, op.tin::<bfloat>(3)?, op.tconst::<f32>(0)?, op.tconst::<f32>(1)?, op.recurrent()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            Axis::F32 => ctx.kda_chunked::<f32>(op.tin::<f32>(0)?, op.tin::<i32>(1)?, op.tin::<f32>(2)?, op.tin::<f32>(3)?, op.tconst::<f32>(0)?, op.tconst::<f32>(1)?, op.recurrent()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`ssm.kda_chunked`, at an element or repr this plane does not instantiate" }),
        },
        "attention.decode" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.decode::<bfloat>(op.tin::<bfloat>(0)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.decode::<f32>(op.tin::<f32>(0)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`attention.decode`, at an element or repr this plane does not instantiate" }),
        },
        "attention.prefill" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.prefill::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<i32>(1)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.u32(2)?, op.f32(3)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.prefill::<f32>(op.tin::<f32>(0)?, op.tin::<i32>(1)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.u32(2)?, op.f32(3)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`attention.prefill`, at an element or repr this plane does not instantiate" }),
        },
        "attention.masked" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.masked::<bfloat>(op.tin::<bfloat>(0)?, op.tin::<i32>(1)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<bfloat>(0)?),
            Axis::F32 => ctx.masked::<f32>(op.tin::<f32>(0)?, op.tin::<i32>(1)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`attention.masked`, at an element or repr this plane does not instantiate" }),
        },
        "attention.logit_softcap" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.logit_softcap::<bfloat>(op.tinout::<bfloat>(0, 0)?, op.f32(0)?),
            Axis::F32 => ctx.logit_softcap::<f32>(op.tinout::<f32>(0, 0)?, op.f32(0)?),
            _ => Err(Refusal::Absent { what: "`attention.logit_softcap`, at an element or repr this plane does not instantiate" }),
        },
        "attention.kv_append" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => Attention::kv_append::<bfloat>(ctx, op.tin::<bfloat>(0)?, op.tin::<bfloat>(1)?, op.pages()?),
            Axis::F32 => Attention::kv_append::<f32>(ctx, op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.pages()?),
            _ => Err(Refusal::Absent { what: "`attention.kv_append`, at an element or repr this plane does not instantiate" }),
        },
        "attention.kv_append_shared" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => ctx.kv_append_shared::<bfloat>(op.tin::<bfloat>(0)?, op.pages()?),
            Axis::F32 => ctx.kv_append_shared::<f32>(op.tin::<f32>(0)?, op.pages()?),
            _ => Err(Refusal::Absent { what: "`attention.kv_append_shared`, at an element or repr this plane does not instantiate" }),
        },
        _ => Err(Refusal::Absent {
            what: "a point this plane does not claim; see the family's `*_CLAIMS`, \
                   or `TIER2_POINTS` for an inherent one",
        }),
    }
}
