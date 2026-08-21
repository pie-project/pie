use kernels::routine::Refusal;
use kernels_macros::routine;

use crate::routine::{
    Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows, f16,
};

#[must_use]
pub fn composable() -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut keep = |r: Result<&'static str, Refusal>| {
        out.push(r.expect("an axis point, by construction"));
    };
    for form in ["", "_bias", "_residual"] {
        for &gs in &[32, 64, 128] {
            for &b in &[4, 8] {
                for &bm in &[16, 32, 64] {
                    for &bn in &[16, 32, 64] {
                        keep(qmm_name(form, gs, b, bm, bn));
                    }
                }
            }
        }
    }

    for form in ["_splitk", "_splitk_f32", "_strided", "_strided_residual"] {
        for &gs in &[32, 64, 128] {
            for &b in &[4, 8] {
                for &bm in &[16, 32, 64] {
                    keep(qmm_name(form, gs, b, bm, 32));
                }
            }
        }
    }

    for (before, after) in [("", ""), ("_bias", ""), ("_residual", "")] {
        for &bm in &[16, 32, 64] {
            for &bn in &[16, 32, 64] {
                keep(qmm_precast_name(before, after, bm, bn));
            }
        }
    }
    for (before, after) in [
        ("_splitk", ""),
        ("_splitk", "_f32"),
        ("_strided", ""),
        ("_strided", "_residual"),
    ] {
        for &bm in &[16, 32, 64] {
            keep(qmm_precast_name(before, after, bm, 32));
        }
    }
    for form in ["fast", "fast_residual"] {
        for &gs in &[32, 64, 128] {
            for &b in &[4, 8] {
                keep(qmv_name(form, gs, b));
            }
        }
    }

    for form in ["tail", "tail_bias"] {
        for &b in &[4, 8] {
            keep(qmv_name(form, 64, b));
        }
    }
    for &b in &[4, 8] {
        keep(qmv_wide_strided_name(b));
    }
    out
}

fn qmm_name(form: &str, group: i32, bits: i32, bm: i32, bn: i32) -> Result<&'static str, Refusal> {
    check(&[32, 64, 128], group, "the group size")?;
    check(&[4, 8], bits, "the bit width")?;
    check(&[16, 32, 64], bm, "the row tile")?;
    check(&[16, 32, 64], bn, "the column tile")?;
    Ok(kernels::jit::symbol(&format!(
        "affine_qmm_t{form}_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}"
    )))
}

fn qmm_precast_name(before: &str, after: &str, bm: i32, bn: i32) -> Result<&'static str, Refusal> {
    check(&[16, 32, 64], bm, "the row tile")?;
    check(&[16, 32, 64], bn, "the column tile")?;
    Ok(kernels::jit::symbol(&format!(
        "affine_qmm_t{before}_fp16_precast{after}_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

fn qmv_wide_strided_name(bits: i32) -> Result<&'static str, Refusal> {
    check(&[4, 8], bits, "the bit width")?;
    Ok(kernels::jit::symbol(&format!(
        "affine_qmv_wide_strided_bfloat16_gs_64_b_{bits}_v_4_kl_8"
    )))
}

fn qmv_name(form: &str, group: i32, bits: i32) -> Result<&'static str, Refusal> {
    check(&[32, 64, 128], group, "the group size")?;
    check(&[4, 8], bits, "the bit width")?;
    Ok(kernels::jit::symbol(&format!(
        "affine_qmv_{form}_bfloat16_gs_{group}_b_{bits}"
    )))
}

fn check(points: &[i32], v: i32, what: &'static str) -> Result<(), Refusal> {
    points.contains(&v).then_some(()).ok_or(Refusal::Narrow {
        what,
        at: i64::from(v),
    })
}

fn qmm_grid(n: i32, bn: i32, m: i32, bm: i32, split_k: i32) -> Result<[u32; 3], Refusal> {
    let tiles = |extent: i32, tile: i32, what: &'static str| -> Result<u32, Refusal> {
        if extent <= 0 {
            return Err(Refusal::Empty { what });
        }
        if tile <= 0 {
            return Err(Refusal::Empty { what: "the tile" });
        }
        u32::try_from(extent)
            .map(|e| e.div_ceil(tile.unsigned_abs()))
            .map_err(|_| Refusal::Grid {
                what,
                at: i64::from(extent),
            })
    };
    if split_k <= 0 {
        return Err(Refusal::Empty {
            what: "the k split",
        });
    }
    let x = tiles(n, bn, "the column count")?;
    let y = tiles(m, bm, "the row count")?;
    let z = split_k.unsigned_abs();
    let lanes = |groups: u32, local: u32, what: &'static str| -> Result<u32, Refusal> {
        groups.checked_mul(local).ok_or(Refusal::Grid {
            what,
            at: i64::from(groups),
        })
    };
    Ok([
        lanes(x, 32, "the column tiles")?,
        lanes(y, 2, "the row tiles")?,
        lanes(z, 2, "the k splits")?,
    ])
}

fn quarters(m: i32) -> i32 {
    if m <= 0 {
        m
    } else {
        m / 4 + i32::from(m % 4 != 0)
    }
}

fn qmv_grid(vecs: i32, out_vec_size: i32) -> Result<[u32; 3], Refusal> {
    if vecs <= 0 {
        return Err(Refusal::Empty {
            what: "the vectors",
        });
    }
    if out_vec_size <= 0 {
        return Err(Refusal::Empty {
            what: "the output vector",
        });
    }
    let x = vecs.unsigned_abs().checked_mul(64).ok_or(Refusal::Grid {
        what: "the vectors",
        at: i64::from(vecs),
    })?;
    let y = out_vec_size
        .unsigned_abs()
        .div_ceil(8)
        .checked_mul(2)
        .ok_or(Refusal::Grid {
            what: "the output rows",
            at: i64::from(out_vec_size),
        })?;
    Ok([x, y, 1])
}

#[routine]
pub fn qmm_t(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    bn: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_name("", *group, *bits, *bm, *bn)?, ctx.best()),
            qmm_name("", *group, *bits, *bm, *bn)?,
        )
        .apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    bn: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_name("_bias", *group, *bits, *bm, *bn)?, ctx.best()),
            qmm_name("_bias", *group, *bits, *bm, *bn)?,
        )
        .apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            bias.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_residual(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    bn: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                qmm_name("_residual", *group, *bits, *bm, *bn)?,
                ctx.best(),
            ),
            qmm_name("_residual", *group, *bits, *bm, *bn)?,
        )
        .apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            residual.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>,
    bn: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_precast_name("", "", *bm, *bn)?, ctx.best()),
            qmm_precast_name("", "", *bm, *bn)?,
        )
        .apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_bias_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>,
    bn: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_precast_name("_bias", "", *bm, *bn)?, ctx.best()),
            qmm_precast_name("_bias", "", *bm, *bn)?,
        )
        .apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            bias.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_residual_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    residual: In<Tensor<bf16>>,
    bm: Const<i32>,
    bn: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_precast_name("_residual", "", *bm, *bn)?, ctx.best()),
            qmm_precast_name("_residual", "", *bm, *bn)?,
        )
        .apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            residual.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_splitk(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = out.width;

    let row_stride = 0i32;

    let k_partition_size = ctx.param(3)?;

    let split_k_partition_stride = ctx.param(4)?;

    let split_k = ctx.param(5)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_name("_splitk", *group, *bits, *bm, 32)?, ctx.best()),
            qmm_name("_splitk", *group, *bits, *bm, 32)?,
        )
        .apply(qmm_grid(n, 32, m, *bm, split_k)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            out.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_splitk_f32(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = out.width;

    let row_stride = 0i32;

    let k_partition_size = ctx.param(3)?;

    let split_k_partition_stride = ctx.param(4)?;

    let split_k = ctx.param(5)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                qmm_name("_splitk_f32", *group, *bits, *bm, 32)?,
                ctx.best(),
            ),
            qmm_name("_splitk_f32", *group, *bits, *bm, 32)?,
        )
        .apply(qmm_grid(n, 32, m, *bm, split_k)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            out.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_splitk_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = out.width;

    let row_stride = 0i32;

    let k_partition_size = ctx.param(3)?;

    let split_k_partition_stride = ctx.param(4)?;

    let split_k = ctx.param(5)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_precast_name("_splitk", "", *bm, 32)?, ctx.best()),
            qmm_precast_name("_splitk", "", *bm, 32)?,
        )
        .apply(qmm_grid(n, 32, m, *bm, split_k)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            out.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_splitk_fp16_precast_f32(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = out.width;

    let row_stride = 0i32;

    let k_partition_size = ctx.param(3)?;

    let split_k_partition_stride = ctx.param(4)?;

    let split_k = ctx.param(5)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_precast_name("_splitk", "_f32", *bm, 32)?, ctx.best()),
            qmm_precast_name("_splitk", "_f32", *bm, 32)?,
        )
        .apply(qmm_grid(n, 32, m, *bm, split_k)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            out.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_strided(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;

    let row_stride = ctx.param(2)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_name("_strided", *group, *bits, *bm, 32)?, ctx.best()),
            qmm_name("_strided", *group, *bits, *bm, 32)?,
        )
        .apply(qmm_grid(n, 32, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_strided_residual(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;

    let row_stride = ctx.param(2)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                qmm_name("_strided_residual", *group, *bits, *bm, 32)?,
                ctx.best(),
            ),
            qmm_name("_strided_residual", *group, *bits, *bm, 32)?,
        )
        .apply(qmm_grid(n, 32, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            residual.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_strided_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = y.width;

    let row_stride = ctx.param(2)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmm_precast_name("_strided", "", *bm, 32)?, ctx.best()),
            qmm_precast_name("_strided", "", *bm, 32)?,
        )
        .apply(qmm_grid(n, 32, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_strided_fp16_precast_residual(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    residual: In<Tensor<bf16>>,
    bm: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = y.width;

    let row_stride = ctx.param(2)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                qmm_precast_name("_strided", "_residual", *bm, 32)?,
                ctx.best(),
            ),
            qmm_precast_name("_strided", "_residual", *bm, 32)?,
        )
        .apply(qmm_grid(n, 32, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            residual.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
        ],
    )
}

#[routine]
pub fn qmm_splitk_reduce(
    ctx: &Ctx<'_>,
    y: Out<Tensor<bf16>>,
    partial: In<Tensor<bf16>>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = 0i32;
    let n = y.width;

    let row_stride = 0i32;

    let split_k_partition_stride = ctx.param(3)?;

    let split_k = ctx.param(4)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("qmm_splitk_reduce_bfloat16", ctx.best()),
            "qmm_splitk_reduce_bfloat16",
        )
        .apply(elementwise_rows(n, m)?),
        &[
            y.arg(),
            partial.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

#[routine]
pub fn qmm_splitk_reduce_f32(
    ctx: &Ctx<'_>,
    y: Out<Tensor<bf16>>,
    partial: In<Tensor<f32>>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = 0i32;
    let n = y.width;

    let row_stride = 0i32;

    let split_k_partition_stride = ctx.param(3)?;

    let split_k = ctx.param(4)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("qmm_splitk_reduce_f32_bfloat16", ctx.best()),
            "qmm_splitk_reduce_f32_bfloat16",
        )
        .apply(elementwise_rows(n, m)?),
        &[
            y.arg(),
            partial.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

#[routine]
pub fn cast_qmm_input_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    cast_in: In<Tensor<bf16>>,
    half_out: Out<Tensor<f16>>,
) -> Result<(), Refusal> {
    let k = 0i32;
    let n = 0i32;

    let row_stride = 0i32;

    let count = ctx.param(3)?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("cast_qmm_input_bfloat16_to_float16", ctx.best()),
            "cast_qmm_input_bfloat16_to_float16",
        )
        .apply(elementwise(count, 1)?),
        &[
            cast_in.arg(),
            half_out.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            count.arg(),
        ],
    )
}

#[routine]
pub fn cast_qmm_input_strided_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    cast_in: In<Tensor<bf16>>,
    half_out: Out<Tensor<f16>>,
    row_stride: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let k = cast_in.width;
    let n = half_out.width;
    let rows = *rows;

    let count = rows.saturating_mul(*row_stride);
    ctx.fire(
        Fire::at(
            crate::routine::module_path("cast_qmm_input_strided_bfloat16_to_float16", ctx.best()),
            "cast_qmm_input_strided_bfloat16_to_float16",
        )
        .apply(elementwise_rows(k, rows)?),
        &[
            cast_in.arg(),
            half_out.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            count.arg(),
        ],
    )
}

#[routine]
pub fn qmv_fast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    vecs: Const<i32>,
) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let vecs = *vecs;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmv_name("fast", *group, *bits)?, ctx.best()),
            qmv_name("fast", *group, *bits)?,
        )
        .apply(qmv_grid(vecs, out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
        ],
    )
}

#[routine]
pub fn qmv_fast_residual(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    vecs: Const<i32>,
) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let vecs = *vecs;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmv_name("fast_residual", *group, *bits)?, ctx.best()),
            qmv_name("fast_residual", *group, *bits)?,
        )
        .apply(qmv_grid(vecs, out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            residual.arg(),
        ],
    )
}

#[routine]
pub fn qmv_tail(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bits: Const<i32>,
    vecs: Const<i32>,
) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let vecs = *vecs;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmv_name("tail", 64, *bits)?, ctx.best()),
            qmv_name("tail", 64, *bits)?,
        )
        .apply(qmv_grid(vecs, out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
        ],
    )
}

#[routine]
pub fn qmv_tail_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    bits: Const<i32>,
    vecs: Const<i32>,
) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let vecs = *vecs;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmv_name("tail_bias", 64, *bits)?, ctx.best()),
            qmv_name("tail_bias", 64, *bits)?,
        )
        .apply(qmv_grid(vecs, out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            bias.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
        ],
    )
}

#[routine]
pub fn qmv_wide_strided(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bits: Const<i32>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;

    let row_stride = ctx.param(2)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(qmv_wide_strided_name(*bits)?, ctx.best()),
            qmv_wide_strided_name(*bits)?,
        )
        .apply(qmv_grid(quarters(m), out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            row_stride.arg(),
            m.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
                ctx.best(),
            ),
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
        )
        .apply(qmm_grid(n, 32, m, 128, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
                ctx.best(),
            ),
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
        )
        .apply(qmm_grid(n, 32, m, 32, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
                ctx.best(),
            ),
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
        )
        .apply(qmm_grid(n, 32, m, 64, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
                ctx.best(),
            ),
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
        )
        .apply(qmm_grid(n, 32, m, 64, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    m: Const<i32>,
) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = *m;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
                ctx.best(),
            ),
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
        )
        .apply(qmm_grid(n, 64, m, 64, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn encode_u4_bf16(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    codes: Out<Tensor<u32>>,
    scales: Out<Tensor<bf16>>,
    biases: Out<Tensor<bf16>>,
    group_size: Const<i32>,
    groups: Const<i32>,
) -> Result<(), Refusal> {
    let groups = *groups;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("affine_encode_u4_bf16", ctx.best()),
            "affine_encode_u4_bf16",
        )
        .apply(elementwise(groups, 1)?),
        &[
            input.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            groups.arg(),
            group_size.arg(),
        ],
    )
}

#[routine]
pub fn encode_u4_f32(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    codes: Out<Tensor<u32>>,
    scales: Out<Tensor<bf16>>,
    biases: Out<Tensor<bf16>>,
    group_size: Const<i32>,
    groups: Const<i32>,
) -> Result<(), Refusal> {
    let groups = *groups;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("affine_encode_u4_f32", ctx.best()),
            "affine_encode_u4_f32",
        )
        .apply(elementwise(groups, 1)?),
        &[
            input.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            groups.arg(),
            group_size.arg(),
        ],
    )
}

#[routine]
pub fn mxfp4_dequant_bf16(
    ctx: &Ctx<'_>,
    payload: In<Tensor<u8>>,
    exponents: In<Tensor<u8>>,
    out: Out<Tensor<bf16>>,
    block_size: Const<i32>,
    blocks: Const<i32>,
) -> Result<(), Refusal> {
    let blocks = *blocks;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("mxfp4_dequant_bf16", ctx.best()),
            "mxfp4_dequant_bf16",
        )
        .apply(elementwise(blocks, 1)?),
        &[
            payload.arg(),
            exponents.arg(),
            out.arg(),
            blocks.arg(),
            block_size.arg(),
        ],
    )
}
