use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Bind, Const, Ctx, Fire, In, InOut, Tensor, bf16};

fn rope_grid(rotary: i32, width: i32, head_dim: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if rotary <= 0 {
        return Err(Refusal::Empty { what: "rotary" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if rotary % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "rotary is not a whole number of pairs",
            at: i64::from(rotary),
        });
    }
    if width <= 0 || width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "width is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok([
        rotary.unsigned_abs() / 2,
        width.unsigned_abs() / head_dim.unsigned_abs(),
        rows.unsigned_abs(),
    ])
}

const fn rope_group(lanes: [u32; 3]) -> [u32; 3] {
    [lanes[0], 1, 1]
}

#[routine]
pub fn neox_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_decode_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
    )
}

#[routine(canon = rope)]
pub fn neox_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_mb_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
    )
}

#[routine]
pub fn neox_freqs_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>,
    rope_freqs: In<Tensor<f32>>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {
    let inv_freq = rope_freqs.ptr;

    let position = positions.ptr;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_freqs_decode_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
        ],
    )
}

#[routine]
pub fn neox_freqs_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>,
    rope_freqs: In<Tensor<f32>>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let inv_freq = rope_freqs.ptr;

    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_freqs_mb_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
        ],
    )
}

#[routine]
pub fn neox_prop_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_prop_decode_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
    )
}

#[routine]
pub fn neox_prop_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_prop_mb_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
    )
}

#[routine]
pub fn neox_strided(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    row_pitch: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    if *row_pitch < width {
        return Err(Refusal::Narrow {
            what: "row_pitch is narrower than the row it strides over",
            at: i64::from(*row_pitch),
        });
    }
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_strided_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            row_pitch.arg(),
        ],
    )
}
