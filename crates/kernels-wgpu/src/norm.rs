use kernels_macros::routine;

use crate::routine::{Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise_rows};
use kernels::routine::Refusal;

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

fn per_row(rows: i32) -> Result<[u32; 3], Refusal> {
    per_axis(1, 1, rows)
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

#[routine(canon = rmsnorm, out(out = like(x)))]
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
        Fire::at("norm/rms.wgsl", "rms_single_row_bfloat16").apply(per_axis(width, *axis, rows)?),
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
        Fire::at("norm/rms.wgsl", "rms_strided_row_bfloat16").apply(per_row(rows)?),
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
        Fire::at("norm/rms.wgsl", "rms_strided_head_row_bfloat16")
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
        Fire::at("norm/rms.wgsl", "rms_residual_bfloat16").apply(per_axis(width, axis, rows)?),
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
        Fire::at("norm/rms.wgsl", "rms_residual_scaled_bfloat16")
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
        Fire::at("norm/vector.wgsl", "vnorm_single_row_bfloat16")
            .apply(per_axis(width, axis, rows)?),
        &[x.arg(), out.arg(), eps.arg(), axis_size.arg()],
    )
}

#[routine(canon = rmsnorm_gated)]
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
        Fire::at("norm/gated_rms.wgsl", "gated_rms_bfloat16").apply(per_head_row(heads, rows)?),
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
        Fire::at("norm/gated_rms.wgsl", "gated_rms_strided_bfloat16")
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
        Fire::at("norm/layer_scalar.wgsl", "layer_scalar_mul_bfloat16")
            .apply(elementwise_rows(width, rows)?),
        &[x.arg(), scalar.arg(), out.arg()],
    )
}

#[routine(canon = residual_add, out(out = like(x)))]
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
        Fire::at("norm/residual_add.wgsl", "residual_add_bfloat16")
            .apply(elementwise_rows(width, rows)?),
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
        Fire::at("norm/residual_add.wgsl", "residual_add_strided_bfloat16")
            .apply(elementwise_rows(width, rows)?),
        &[x.arg(), residual.arg(), out.arg(), row_pitch.arg()],
    )
}

#[routine(canon = add_bias, out(out = like(out)))]
pub fn add_bias(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = out.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("norm/add_bias.wgsl", "add_bias_bfloat16").apply(elementwise_rows(width, rows)?),
        &[out.arg(), bias.arg(), width.arg()],
    )
}

/// The grid `norm/rms_rope.wgsl` launches: one workgroup per head per row, on
/// the y and z axes, with x left at one.
///
/// The refusals are the kernel's preconditions written where a plan can still
/// be refused rather than answered wrongly.
fn per_head_row_rotating(
    heads: i32,
    rows: i32,
    rotary: i32,
    axis: i32,
    row_pitch: i32,
) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if rotary <= 0 {
        return Err(Refusal::Empty { what: "rotary" });
    }
    if rotary > axis {
        return Err(Refusal::Wide {
            what: "rotary",
            at: i64::from(rotary),
            max: i64::from(axis),
        });
    }
    // EVERY EDGE ON A WORD BOUNDARY, and this is the one refusal that is this
    // backend's and not the family's. Two bf16 share a four-byte word, so an
    // invocation writes whole words or it races the neighbour holding the other
    // half. The kernel's rotation walks `[base, base + rotary)` in words and its
    // tail walks `[base + rotary, base + axis)` in words, so all three of the
    // terms that build those bounds have to be even. They are, in every
    // checkpoint this tree loads -- head widths, rotary widths and projection
    // widths are multiples of four -- and a deployment where they are not gets
    // the unfused pair rather than a torn head.
    for (what, at) in [
        ("rotary", rotary),
        ("axis_size", axis),
        ("row_pitch", row_pitch),
    ] {
        if at % 2 != 0 {
            return Err(Refusal::Narrow {
                what,
                at: i64::from(at),
            });
        }
    }
    Ok([1, heads.unsigned_abs(), rows.unsigned_abs()])
}

/// The per-head RMS norm and the NEOX rotation as one dispatch.
///
/// `kernels-metal` declares this family in `ELSEWHERE` and has no text for it,
/// so the signature here is that declaration's -- `x` in place and the norm
/// weight -- and everything else rides the launch's params, in the order
/// `model-dsl::metal::rms_rope` writes them. The shader's `Params` block
/// repeats that order and nothing but the answers checks that the two agree.
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
    // THE VULKAN TWIN'S SIGNATURE, verbatim — the three tables hold equal,
    // and this fn arrived through a merge still spelled in the ask form the
    // sweep retired.
    let position = positions.ptr;
    let axis_of = *axis;
    let pitch_of = *row_pitch;
    let rotary_of = *rotary;
    let rows = *rows;

    if axis_of <= 0 {
        return Err(Refusal::Empty { what: "axis_size" });
    }
    let heads = pitch_of / axis_of;
    ctx.fire(
        Fire::at("norm/rms_rope.wgsl", "rms_rope_bfloat16").apply(per_head_row_rotating(
            heads, rows, rotary_of, axis_of, pitch_of,
        )?),
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
