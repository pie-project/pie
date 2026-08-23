use kernels_macros::routine;

use crate::routine::{Bind, Const, Ctx, Fire, In, InOut, Tensor, bf16};
use kernels::routine::Refusal;

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

    let pairs = rotary.unsigned_abs() / 2;
    Ok([
        pairs.div_ceil(2).div_ceil(32),
        width.unsigned_abs() / head_dim.unsigned_abs(),
        rows.unsigned_abs(),
    ])
}

#[routine(out(x = like(x)))]
pub fn neox_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_decode_bfloat16")
            .apply(rope_grid(*rotary, width, *head_dim, 1)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            rotary.arg(),
        ],
    )
}

#[routine(canon = "rope.full", out(x = like(x)))]
pub fn neox_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_mb_bfloat16")
            .apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            rotary.arg(),
        ],
    )
}

#[routine(out(x = like(x)))]
pub fn neox_freqs_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>,
    rope_freqs: In<Tensor<f32>>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let inv_freq = rope_freqs.ptr;

    let position = positions.ptr;
    let width = x.width;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_freqs_decode_bfloat16")
            .apply(rope_grid(*rotary, width, *head_dim, 1)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
            rotary.arg(),
        ],
    )
}

#[routine(out(x = like(x)))]
pub fn neox_freqs_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>,
    rope_freqs: In<Tensor<f32>>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let inv_freq = rope_freqs.ptr;

    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_freqs_mb_bfloat16")
            .apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
            rotary.arg(),
        ],
    )
}

#[routine(out(x = like(x)))]
pub fn neox_prop_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_prop_decode_bfloat16")
            .apply(rope_grid(*rotary, width, *head_dim, 1)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            rotary.arg(),
        ],
    )
}

#[routine(out(x = like(x)))]
pub fn neox_prop_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_prop_mb_bfloat16")
            .apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            rotary.arg(),
        ],
    )
}

#[routine(out(x = like(x)))]
pub fn neox_strided(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    row_pitch: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_strided_bfloat16")
            .apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            row_pitch.arg(),
            rotary.arg(),
        ],
    )
}
