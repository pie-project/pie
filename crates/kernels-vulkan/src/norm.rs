use crate::routine::{
    Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows,
};
use kernels::routine::Refusal;
use kernels_macros::routine;

fn per_row(rows: i32) -> Result<[u32; 3], Refusal> {
    per_axis(1, 1, rows)
}

fn per_axis(width: i32, axis: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
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
    let axes = (width.unsigned_abs() / axis.unsigned_abs()) * rows.unsigned_abs();
    let lanes = axes.checked_mul(256).ok_or(Refusal::Grid {
        what: "axes * the workgroup width",
        at: i64::from(axes) * 256,
    })?;
    Ok([lanes, 1, 1])
}

fn per_head_row(heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([256, heads.unsigned_abs(), rows.unsigned_abs()])
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("rms_single_row_bfloat16", ctx.best()),
            "rms_single_row_bfloat16",
        )
        .apply(per_axis(width, *axis, rows)?),
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("rms_strided_row_bfloat16", ctx.best()),
            "rms_strided_row_bfloat16",
        )
        .apply(per_row(rows)?),
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("rms_strided_head_row_bfloat16", ctx.best()),
            "rms_strided_head_row_bfloat16",
        )
        .apply(per_head_row(heads, rows)?),
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

fn per_head_row_rotating(
    heads: i32,
    rows: i32,
    rotary: i32,
    axis: i32,
) -> Result<[u32; 3], Refusal> {
    if rotary <= 0 {
        return Err(Refusal::Empty { what: "rotary" });
    }
    if rotary % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "rotary",
            at: i64::from(rotary),
        });
    }
    if axis > 0 && rotary > axis {
        return Err(Refusal::Wide {
            what: "rotary",
            at: i64::from(rotary),
            max: i64::from(axis),
        });
    }
    per_head_row(heads, rows)
}

#[routine]
pub fn rms_rope(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    eps: Const<f32>,
    axis: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>,
    row_pitch: Const<i32>,
    rotary: Const<i32>,
    scale: Const<f32>,
    base_or_mscale: Const<f32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let axis_of = *axis;
    let pitch_of = *row_pitch;
    let rotary_of = *rotary;
    let rows = *rows;

    let heads = if axis_of > 0 { pitch_of / axis_of } else { 0 };
    ctx.fire(
        Fire::at(
            crate::routine::module_path("rms_rope_bfloat16", ctx.best()),
            "rms_rope_bfloat16",
        )
        .apply(per_head_row_rotating(heads, rows, rotary_of, axis_of)?),
        &[
            x.arg(),
            w.arg(),
            position.arg(),
            eps.arg(),
            axis.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
            row_pitch.arg(),
            rotary.arg(),
            scale.arg(),
            base_or_mscale.arg(),
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("rms_residual_bfloat16", ctx.best()),
            "rms_residual_bfloat16",
        )
        .apply(per_axis(width, axis, rows)?),
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("rms_residual_scaled_bfloat16", ctx.best()),
            "rms_residual_scaled_bfloat16",
        )
        .apply(per_axis(width, axis, rows)?),
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
    axis_size: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = x.width;

    let axis = x.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("vnorm_single_row_bfloat16", ctx.best()),
            "vnorm_single_row_bfloat16",
        )
        .apply(per_axis(width, axis, rows)?),
        &[x.arg(), out.arg(), eps.arg(), axis_size.arg()],
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("gated_rms_bfloat16", ctx.best()),
            "gated_rms_bfloat16",
        )
        .apply(per_head_row(heads, rows)?),
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("gated_rms_strided_bfloat16", ctx.best()),
            "gated_rms_strided_bfloat16",
        )
        .apply(per_head_row(heads, rows)?),
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
        Fire::at(
            crate::routine::module_path("layer_scalar_mul_bfloat16", ctx.best()),
            "layer_scalar_mul_bfloat16",
        )
        .apply(elementwise(width, rows)?),
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
        Fire::at(
            crate::routine::module_path("residual_add_bfloat16", ctx.best()),
            "residual_add_bfloat16",
        )
        .apply(elementwise(width, rows)?),
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
        Fire::at(
            crate::routine::module_path("residual_add_strided_bfloat16", ctx.best()),
            "residual_add_strided_bfloat16",
        )
        .apply(elementwise_rows(width, rows)?),
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("add_bias_bfloat16", ctx.best()),
            "add_bias_bfloat16",
        )
        .apply(elementwise_rows(width, rows)?),
        &[out.arg(), bias.arg(), width.arg()],
    )
}
