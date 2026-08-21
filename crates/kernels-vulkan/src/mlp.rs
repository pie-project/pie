use crate::routine::{
    Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows,
};
use kernels::routine::Refusal;
use kernels_macros::routine;

#[routine(out(out = like(gate)))]
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("geglu_tanh_bfloat16", ctx.best()),
            "geglu_tanh_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

#[routine(out(out = rows(gate) x const(stated_width)))]
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
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("geglu_tanh_strided_bfloat16", ctx.best()),
            "geglu_tanh_strided_bfloat16",
        )
        .apply(elementwise(width, rows)?),
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

#[routine(out(out = like(gate)))]
pub fn gptoss_swiglu(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    _stated_elements: Const<u32>,
    limit: Const<f32>,
    alpha: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("gptoss_swiglu_bfloat16", ctx.best()),
            "gptoss_swiglu_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg(), limit.arg(), alpha.arg()],
    )
}

#[routine(out(out = like(gate)))]
pub fn silu_mul(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("silu_mul_bfloat16", ctx.best()),
            "silu_mul_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

#[routine]
pub fn silu_mul_strided(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let row_pitch = ctx.param(1)?;
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("silu_mul_strided_bfloat16", ctx.best()),
            "silu_mul_strided_bfloat16",
        )
        .apply(elementwise_rows(width, rows)?),
        &[gate.arg(), up.arg(), out.arg(), row_pitch.arg()],
    )
}
