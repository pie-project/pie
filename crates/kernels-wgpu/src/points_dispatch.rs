#![cfg_attr(rustfmt, rustfmt::skip)]
#![allow(unused_imports)]

use kernels::bound::{Axis, BoundOp, Site};
use kernels::points::{Form, Mxfp4};
use kernels::points::{Attention, Gate, Layout, Mlp, Moe, Norm, Rope};
use kernels::plane::Refusal;

use crate::points::bf16;
use crate::plane::Ctx;

pub const CLAIMED: &[(&str, Option<Site>, &[Axis])] = &[
    ("norm.rmsnorm", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_per_head", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_plus_one", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_per_head_plus_one", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.rmsnorm_no_scale", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.residual_add", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.add_bias", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("norm.scale", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("mlp.geglu_tanh", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("rope.full", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("rope.partial", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("rope.partial_q", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("moe.sigmoid_gate_add", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("gate.sigmoid_mul", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("layout.split_qkv", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("layout.split_q_gate", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.decode", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.prefill", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.masked", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.logit_softcap", Some(Site::Out(0)), &[Axis::Bf16, Axis::F32]),
    ("attention.kv_append", Some(Site::In(0)), &[Axis::Bf16, Axis::F32]),
];

pub const TIER2: &[(&str, Option<Site>, &[Axis])] = &[
];

pub fn dispatch<'p, B>(ctx: &Ctx<'p>, op: &B) -> Result<(), Refusal>
where
    B: BoundOp<Plane = Ctx<'p>>,
{
    match op.point() {
        "norm.rmsnorm" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm::<bf16>(op.tin::<bf16>(0)?, op.tconst::<bf16>(0)?, op.f32(0)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.rmsnorm::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.f32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_per_head" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_per_head::<bf16>(op.tin::<bf16>(0)?, op.tconst::<bf16>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.rmsnorm_per_head::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_per_head`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_plus_one" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_plus_one::<bf16>(op.tin::<bf16>(0)?, op.tconst::<bf16>(0)?, op.f32(0)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.rmsnorm_plus_one::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.f32(0)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_plus_one`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_per_head_plus_one" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_per_head_plus_one::<bf16>(op.tin::<bf16>(0)?, op.tconst::<bf16>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.rmsnorm_per_head_plus_one::<f32>(op.tin::<f32>(0)?, op.tconst::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_per_head_plus_one`, at an element or repr this plane does not instantiate" }),
        },
        "norm.rmsnorm_no_scale" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.rmsnorm_no_scale::<bf16>(op.tin::<bf16>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.rmsnorm_no_scale::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.f32(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`norm.rmsnorm_no_scale`, at an element or repr this plane does not instantiate" }),
        },
        "norm.residual_add" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.residual_add::<bf16>(op.tin::<bf16>(0)?, op.tinout::<bf16>(1, 0)?),
            Axis::F32 => ctx.residual_add::<f32>(op.tin::<f32>(0)?, op.tinout::<f32>(1, 0)?),
            _ => Err(Refusal::Absent { what: "`norm.residual_add`, at an element or repr this plane does not instantiate" }),
        },
        "norm.add_bias" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.add_bias::<bf16>(op.tconst::<bf16>(0)?, op.tinout::<bf16>(0, 0)?),
            Axis::F32 => ctx.add_bias::<f32>(op.tconst::<f32>(0)?, op.tinout::<f32>(0, 0)?),
            _ => Err(Refusal::Absent { what: "`norm.add_bias`, at an element or repr this plane does not instantiate" }),
        },
        "norm.scale" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.scale::<bf16>(op.tconst::<bf16>(0)?, op.tinout::<bf16>(0, 0)?),
            Axis::F32 => ctx.scale::<f32>(op.tconst::<f32>(0)?, op.tinout::<f32>(0, 0)?),
            _ => Err(Refusal::Absent { what: "`norm.scale`, at an element or repr this plane does not instantiate" }),
        },
        "mlp.geglu_tanh" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.geglu_tanh::<bf16>(op.tin::<bf16>(0)?, op.tin::<bf16>(1)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.geglu_tanh::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`mlp.geglu_tanh`, at an element or repr this plane does not instantiate" }),
        },
        "rope.full" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.full::<bf16>(op.tinout::<bf16>(0, 0)?, op.tinout::<bf16>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.f32(1)?, op.bool(2)?),
            Axis::F32 => ctx.full::<f32>(op.tinout::<f32>(0, 0)?, op.tinout::<f32>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.f32(1)?, op.bool(2)?),
            _ => Err(Refusal::Absent { what: "`rope.full`, at an element or repr this plane does not instantiate" }),
        },
        "rope.partial" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.partial::<bf16>(op.tinout::<bf16>(0, 0)?, op.tinout::<bf16>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.u32(1)?, op.f32(2)?),
            Axis::F32 => ctx.partial::<f32>(op.tinout::<f32>(0, 0)?, op.tinout::<f32>(1, 1)?, op.tin::<i32>(2)?, op.u32(0)?, op.u32(1)?, op.f32(2)?),
            _ => Err(Refusal::Absent { what: "`rope.partial`, at an element or repr this plane does not instantiate" }),
        },
        "rope.partial_q" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.partial_q::<bf16>(op.tinout::<bf16>(0, 0)?, op.tin::<i32>(1)?, op.u32(0)?, op.u32(1)?, op.f32(2)?),
            Axis::F32 => ctx.partial_q::<f32>(op.tinout::<f32>(0, 0)?, op.tin::<i32>(1)?, op.u32(0)?, op.u32(1)?, op.f32(2)?),
            _ => Err(Refusal::Absent { what: "`rope.partial_q`, at an element or repr this plane does not instantiate" }),
        },
        "moe.sigmoid_gate_add" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.sigmoid_gate_add::<bf16>(op.tin::<bf16>(0)?, op.tin::<bf16>(1)?, op.tin::<bf16>(2)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.sigmoid_gate_add::<f32>(op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.tin::<f32>(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`moe.sigmoid_gate_add`, at an element or repr this plane does not instantiate" }),
        },
        "gate.sigmoid_mul" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.sigmoid_mul::<bf16>(op.tinout::<bf16>(0, 0)?, op.tin::<bf16>(1)?),
            Axis::F32 => ctx.sigmoid_mul::<f32>(op.tinout::<f32>(0, 0)?, op.tin::<f32>(1)?),
            _ => Err(Refusal::Absent { what: "`gate.sigmoid_mul`, at an element or repr this plane does not instantiate" }),
        },
        "layout.split_qkv" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.split_qkv::<bf16>(op.tin::<bf16>(0)?, op.u32(0)?, op.u32(1)?, op.tout::<bf16>(0)?, op.tout::<bf16>(1)?, op.tout::<bf16>(2)?),
            Axis::F32 => ctx.split_qkv::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.u32(1)?, op.tout::<f32>(0)?, op.tout::<f32>(1)?, op.tout::<f32>(2)?),
            _ => Err(Refusal::Absent { what: "`layout.split_qkv`, at an element or repr this plane does not instantiate" }),
        },
        "layout.split_q_gate" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.split_q_gate::<bf16>(op.tin::<bf16>(0)?, op.u32(0)?, op.tout::<bf16>(0)?, op.tout::<bf16>(1)?),
            Axis::F32 => ctx.split_q_gate::<f32>(op.tin::<f32>(0)?, op.u32(0)?, op.tout::<f32>(0)?, op.tout::<f32>(1)?),
            _ => Err(Refusal::Absent { what: "`layout.split_q_gate`, at an element or repr this plane does not instantiate" }),
        },
        "attention.decode" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.decode::<bf16>(op.tin::<bf16>(0)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.decode::<f32>(op.tin::<f32>(0)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`attention.decode`, at an element or repr this plane does not instantiate" }),
        },
        "attention.prefill" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.prefill::<bf16>(op.tin::<bf16>(0)?, op.tin::<i32>(1)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.u32(2)?, op.f32(3)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.prefill::<f32>(op.tin::<f32>(0)?, op.tin::<i32>(1)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.u32(2)?, op.f32(3)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`attention.prefill`, at an element or repr this plane does not instantiate" }),
        },
        "attention.masked" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.masked::<bf16>(op.tin::<bf16>(0)?, op.tin::<i32>(1)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<bf16>(0)?),
            Axis::F32 => ctx.masked::<f32>(op.tin::<f32>(0)?, op.tin::<i32>(1)?, op.pages()?, op.u32(0)?, op.u32(1)?, op.f32(2)?, op.tout::<f32>(0)?),
            _ => Err(Refusal::Absent { what: "`attention.masked`, at an element or repr this plane does not instantiate" }),
        },
        "attention.logit_softcap" => match op.dtype(Site::Out(0))? {
            Axis::Bf16 => ctx.logit_softcap::<bf16>(op.tinout::<bf16>(0, 0)?, op.f32(0)?),
            Axis::F32 => ctx.logit_softcap::<f32>(op.tinout::<f32>(0, 0)?, op.f32(0)?),
            _ => Err(Refusal::Absent { what: "`attention.logit_softcap`, at an element or repr this plane does not instantiate" }),
        },
        "attention.kv_append" => match op.dtype(Site::In(0))? {
            Axis::Bf16 => Attention::kv_append::<bf16>(ctx, op.tin::<bf16>(0)?, op.tin::<bf16>(1)?, op.pages()?),
            Axis::F32 => Attention::kv_append::<f32>(ctx, op.tin::<f32>(0)?, op.tin::<f32>(1)?, op.pages()?),
            _ => Err(Refusal::Absent { what: "`attention.kv_append`, at an element or repr this plane does not instantiate" }),
        },
        _ => Err(Refusal::Absent {
            what: "a point this plane does not claim; see the family's `*_CLAIMS`, \
                   or `TIER2_POINTS` for an inherent one",
        }),
    }
}
