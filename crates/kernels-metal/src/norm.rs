use kernels::Grid;
use kernels::routine::Refusal;
use kernels_macros::routine;

use crate::routine::{
    Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows,
};

fn rms_threads(axis: i32) -> Result<u32, Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
    }
    Ok(axis.unsigned_abs().div_ceil(4).min(1024))
}

fn rms_grid(width: i32, axis: i32, rows: i32) -> Result<([u32; 3], [u32; 3]), Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if axis > width {
        return Err(Refusal::Wide {
            what: "axis",
            at: i64::from(axis),
            max: i64::from(width),
        });
    }
    let t = rms_threads(axis)?;
    let norms = width.unsigned_abs() / axis.unsigned_abs();
    let lanes = t
        .checked_mul(norms)
        .and_then(|n| n.checked_mul(rows.unsigned_abs()))
        .ok_or(Refusal::Grid {
            what: "axis threads * norms per row * rows",
            at: i64::from(t) * i64::from(norms) * i64::from(rows),
        })?;
    Ok(([lanes, 1, 1], [t, 1, 1]))
}

fn head_row_grid(threads: u32, heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([threads, heads.unsigned_abs(), rows.unsigned_abs()])
}

#[routine(canon = "norm.rmsnorm", out(out = like(x)))]
pub fn rms_single_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    eps: Const<f32>,
    axis: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = x.width;
    let rows = *rows;
    let (lanes, group) = rms_grid(width, *axis, rows)?;
    ctx.fire(
        Fire::at("norm/rms.metal", "rms_single_row_bfloat16").apply(Grid::of(lanes, group)),
        &[
            x.arg(),
            w.arg(),
            out.arg(),
            eps.arg(),
            axis.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
        ],
    )
}

#[routine]
pub fn rms_strided_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    eps: Const<f32>,
    axis: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let row_pitch = x.width;
    let rows = *rows;
    let t = rms_threads(*axis)?;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let lanes = t.checked_mul(rows.unsigned_abs()).ok_or(Refusal::Grid {
        what: "axis threads rows",
        at: i64::from(t) * i64::from(rows),
    })?;
    ctx.fire(
        Fire::at("norm/rms.metal", "rms_strided_row_bfloat16")
            .apply(Grid::of([lanes, 1, 1], [t, 1, 1])),
        &[
            x.arg(),
            w.arg(),
            out.arg(),
            eps.arg(),
            axis.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
            row_pitch.arg(),
        ],
    )
}

#[routine]
pub fn rms_strided_head_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    eps: Const<f32>,
    axis: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let row_pitch = x.width;

    let heads = if *axis > 0 { row_pitch / *axis } else { 0 };
    let rows = *rows;
    let t = rms_threads(*axis)?;
    ctx.fire(
        Fire::at("norm/rms.metal", "rms_strided_head_row_bfloat16")
            .apply(Grid::of(head_row_grid(t, heads, rows)?, [t, 1, 1])),
        &[
            x.arg(),
            w.arg(),
            out.arg(),
            eps.arg(),
            axis.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
            row_pitch.arg(),
        ],
    )
}

#[routine(out(out = like(x)))]
pub fn rms_residual(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    r: In<Tensor<bf16>>,
    eps: Const<f32>,
    axis_size: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = x.width;

    let axis = x.width;
    let rows = *rows;
    let (lanes, group) = rms_grid(width, axis, rows)?;
    ctx.fire(
        Fire::at("norm/rms.metal", "rms_residual_bfloat16").apply(Grid::of(lanes, group)),
        &[
            x.arg(),
            w.arg(),
            out.arg(),
            eps.arg(),
            axis_size.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
            r.arg(),
        ],
    )
}

#[routine(out(out = like(x)))]
pub fn rms_residual_scaled(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    r: In<Tensor<bf16>>,
    s: In<Tensor<bf16>>,
    eps: Const<f32>,
    axis_size: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = x.width;

    let axis = x.width;
    let rows = *rows;
    let (lanes, group) = rms_grid(width, axis, rows)?;
    ctx.fire(
        Fire::at("norm/rms.metal", "rms_residual_scaled_bfloat16").apply(Grid::of(lanes, group)),
        &[
            x.arg(),
            w.arg(),
            out.arg(),
            eps.arg(),
            axis_size.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
            r.arg(),
            s.arg(),
        ],
    )
}

#[routine(out(out = like(x)))]
pub fn vnorm_single_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    eps: Const<f32>,
    axis: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = x.width;
    let rows = *rows;
    let (lanes, group) = rms_grid(width, *axis, rows)?;
    ctx.fire(
        Fire::at("norm/vector.metal", "vnorm_single_row_bfloat16").apply(Grid::of(lanes, group)),
        &[x.arg(), out.arg(), eps.arg(), axis.arg()],
    )
}

#[routine(canon = "norm.rmsnorm_gated")]
pub fn gated_rms(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    z: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    eps: Const<f32>,
    vd: Const<i32>,
    heads: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let heads = *heads;
    let rows = *rows;
    let t = head_width(*vd)?;
    ctx.fire(
        Fire::at("norm/gated_rms.metal", "gated_rms_bfloat16")
            .apply(Grid::of(head_row_grid(t, heads, rows)?, [t, 1, 1])),
        &[x.arg(), z.arg(), w.arg(), out.arg(), eps.arg(), vd.arg()],
    )
}

#[routine]
pub fn gated_rms_strided(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    z: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    eps: Const<f32>,
    vd: Const<i32>,
    heads: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let heads = *heads;
    let row_pitch = x.width;
    let rows = *rows;
    let t = head_width(*vd)?;
    ctx.fire(
        Fire::at("norm/gated_rms.metal", "gated_rms_strided_bfloat16")
            .apply(Grid::of(head_row_grid(t, heads, rows)?, [t, 1, 1])),
        &[
            x.arg(),
            z.arg(),
            w.arg(),
            out.arg(),
            eps.arg(),
            vd.arg(),
            row_pitch.arg(),
        ],
    )
}

fn head_width(vd: i32) -> Result<u32, Refusal> {
    if vd <= 0 {
        return Err(Refusal::Empty { what: "vd" });
    }
    if vd > 1024 {
        return Err(Refusal::Wide {
            what: "vd",
            at: i64::from(vd),
            max: 1024,
        });
    }
    Ok(vd.unsigned_abs())
}

#[routine(out(out = like(x)))]
pub fn layer_scalar_mul(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    scalar: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = x.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("norm/layer_scalar.metal", "layer_scalar_mul_bfloat16")
            .apply(Grid::of(elementwise(width, rows)?, [256, 1, 1])),
        &[x.arg(), scalar.arg(), out.arg()],
    )
}

#[routine(canon = "norm.residual_add", out(out = like(x)))]
pub fn residual_add(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = x.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("norm/residual_add.metal", "residual_add_bfloat16")
            .apply(Grid::of(elementwise(width, rows)?, [256, 1, 1])),
        &[x.arg(), residual.arg(), out.arg()],
    )
}

#[routine]
pub fn residual_add_strided(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    row_pitch: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = x.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("norm/residual_add.metal", "residual_add_strided_bfloat16")
            .apply(Grid::of(elementwise_rows(width, rows)?, [256, 1, 1])),
        &[x.arg(), residual.arg(), out.arg(), row_pitch.arg()],
    )
}

#[routine(canon = "norm.add_bias", out(out = like(out)))]
pub fn add_bias(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = out.width;
    let rows = *rows;
    let lanes = elementwise_rows(width, rows)?;
    ctx.fire(
        Fire::at("norm/add_bias.metal", "add_bias_bfloat16")
            .apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
        &[out.arg(), bias.arg(), width.arg()],
    )
}
