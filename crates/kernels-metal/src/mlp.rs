use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise};

#[routine]
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("mlp/gated.metal", "geglu_tanh_bfloat16").apply(Grid::of(elementwise(width, rows)?, [256, 1, 1])),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

#[routine]
pub fn geglu_tanh_strided(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    stated_width: Const<u32>,
    stated_rows: Const<u32>,
    gate_pitch: Const<u32>,
    up_pitch: Const<u32>,
    out_pitch: Const<u32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("mlp/gated.metal", "geglu_tanh_strided_bfloat16").apply(Grid::of(elementwise(width, rows)?, [256, 1, 1])),
        &[
            gate.arg(),
            up.arg(),
            out.arg(),
            stated_width.arg(),
            stated_rows.arg(),
            gate_pitch.arg(),
            up_pitch.arg(),
            out_pitch.arg(),
        ],
    )
}

#[routine]
pub fn gptoss_swiglu(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    _stated_elements: Const<u32>,
    limit: Const<f32>,
    alpha: Const<f32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("mlp/gated.metal", "gptoss_swiglu_bfloat16").apply(Grid::of(elementwise(width, rows)?, [256, 1, 1])),
        &[gate.arg(), up.arg(), out.arg(), limit.arg(), alpha.arg()],
    )
}

#[routine]
pub fn silu_mul(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("mlp/gated.metal", "silu_mul_bfloat16").apply(Grid::of(elementwise(width, rows)?, [256, 1, 1])),
        &[gate.arg(), up.arg(), out.arg()],
    )
}
