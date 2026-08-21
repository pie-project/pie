use kernels::Grid;
use kernels::routine::Refusal;
use kernels_macros::routine;

use crate::routine::{
    Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows, f16,
};

/// Every `(file, entrypoint)` this crate STAMPS rather than finds declared.
///
/// # Why the crate answers this at all
///
/// Because something downstream asks *"is this string an entrypoint this tree
/// can produce"*, and for these four forms the shader no longer knows. It
/// declares what it CAN stamp; the host decides which points to stamp, so the
/// host is the only party that can enumerate them. `kernels_metal::kernel_of`
/// is the asker: it maps an instantiated symbol out of a model text back to
/// the routine that fires it, and its census used to come entirely from
/// `build.rs` expanding the `.metal`'s own instantiation lists.
///
/// # It is not a table
///
/// It is the product of the axes through [`qmm_point`] -- the same function a
/// fire calls -- so a name here is a name a fire can reach, by construction.
/// That is the difference between this and the fifty-four-line list it
/// replaced in `moe.rs`.
///
/// The forms NOT here are the ones `quant/qmm_t.metal` still instantiates
/// itself (`_splitk`, `_strided`, the `_fp16_precast` family) and the matvecs.
/// Those stay in `build.rs`'s census until they move too.
#[must_use]
pub fn composed() -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::new();
    for form in ["", "_bias", "_residual", "_routed"] {
        for &gs in &[32, 64, 128] {
            for &b in &[4, 8] {
                for &bm in &[16, 32, 64] {
                    for &bn in &[16, 32, 64] {
                        let p = qmm_point(form, "", gs, b, bm, bn)
                            .expect("an axis point, by construction");
                        out.push(("quant/qmm_t.metal", p.entry));
                    }
                }
            }
        }
    }
    out
}

/// An entrypoint this crate can reach, and the line that makes it exist.
///
/// Two renderings of one set of numbers. [`Self::stamp`] is composed by the
/// same call that composes [`Self::entry`] and embeds it, so the name a fire
/// asks for and the name the shader exports cannot disagree -- which is the
/// whole of what `quant/qmm_t.metal`'s deleted instantiation list, `moe.rs`'s
/// table of the same names, and a fixture comparing the two were for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    /// The symbol a fire names.
    pub entry: &'static str,
    /// The `PIE_STAMP_*(..)` call that declares it, or empty where the file
    /// still declares the form itself. See [`kernels::routine::Fire::stamp`].
    pub stamp: &'static str,
}

/// The K tile every point of this family is stamped at.
///
/// Not an axis: `instantiate_qmm_t`'s `bk` argument was 32 at all fifty-four
/// call sites. A parameter for a constant reads as a choice a caller has.
const QMM_BK: i32 = 32;

/// The affine matmul's point on its four axes.
///
/// `form` is the variant's own infix -- `""`, `"_bias"`, `"_residual"`,
/// `"_routed"` -- and `stamp` names the `#define` in `quant/qmm_t.metal` that
/// stamps that form. **An empty `stamp` means the file still declares the form
/// itself**, which is the four `_splitk` and `_strided` variants: they have
/// their own `#define instantiate_*` blocks with their own call lists, and
/// they keep working untouched. That is what makes this migration a family at
/// a time rather than a file at once.
///
/// # The axis checks stay, and mean something sharper now
///
/// They used to mirror the shader's own call list -- a second copy of it, in
/// Rust. There is no list to mirror any more, so what they are is the bound on
/// what this host will ASK to be stamped: a tile the template cannot serve is
/// now a Metal compile error at load rather than a `Refusal` at the fire, and
/// this is what keeps that from being reachable by a stray number.
pub(crate) fn qmm_point(
    form: &str,
    stamp: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<Point, Refusal> {
    check(&[32, 64, 128], group, "the group size")?;
    check(&[4, 8], bits, "the bit width")?;
    check(&[16, 32, 64], bm, "the row tile")?;
    check(&[16, 32, 64], bn, "the column tile")?;
    let entry = kernels::jit::symbol(&format!(
        "affine_qmm_t{form}_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}"
    ));
    Ok(Point {
        entry,
        stamp: if stamp.is_empty() {
            ""
        } else {
            kernels::jit::symbol(&format!(
                "{stamp}(\"{entry}\", {group}, {bits}, {bm}, {QMM_BK}, {bn})"
            ))
        },
    })
}

/// [`qmm_point`] for a form the file still declares, discarding the stamp.
fn qmm_name(form: &str, group: i32, bits: i32, bm: i32, bn: i32) -> Result<&'static str, Refusal> {
    Ok(qmm_point(form, "", group, bits, bm, bn)?.entry)
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
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "the column count",
        });
    }
    if m <= 0 {
        return Err(Refusal::Empty {
            what: "the row count",
        });
    }
    if bn <= 0 || bm <= 0 {
        return Err(Refusal::Empty { what: "the tile" });
    }
    if split_k <= 0 {
        return Err(Refusal::Empty {
            what: "the k split",
        });
    }
    if m % bm != 0 {
        return Err(Refusal::Misaligned {
            what: "the row count, which the tile must divide because no \
                   entrypoint takes m and the shader reads it from the grid",
        });
    }
    if n % bn != 0 {
        return Err(Refusal::Misaligned {
            what: "the column count, which the tile must divide: `qmm_t.metal` \
                   states `M % BM == 0, N % BN == 0 and K % BK == 0` as the \
                   condition under which the driver may select it at all, and \
                   `load_unsafe` is the only path its hot loop takes",
        });
    }
    let lanes = |groups: u32, local: u32, what: &'static str| -> Result<u32, Refusal> {
        groups.checked_mul(local).ok_or(Refusal::Grid {
            what,
            at: i64::from(groups),
        })
    };
    Ok([
        lanes(
            n.unsigned_abs().div_ceil(bn.unsigned_abs()),
            32,
            "the column tiles",
        )?,
        lanes(m.unsigned_abs() / bm.unsigned_abs(), 2, "the row tiles")?,
        lanes(split_k.unsigned_abs(), 2, "the k splits")?,
    ])
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
    let x = vecs.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
        what: "the vectors",
        at: i64::from(vecs),
    })?;
    Ok([x, out_vec_size.unsigned_abs().div_ceil(4), 1])
}

fn quarters(m: i32) -> i32 {
    if m <= 0 {
        m
    } else {
        m / 4 + i32::from(m % 4 != 0)
    }
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
    let point = qmm_point("", "PIE_STAMP_qmm_t", *group, *bits, *bm, *bn)?;
    ctx.fire(
        Fire::at("quant/qmm_t.metal", point.entry)
            .stamp(point.stamp)
            .apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, [32, 2, 2])),
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
    let point = qmm_point("_bias", "PIE_STAMP_qmm_t_bias", *group, *bits, *bm, *bn)?;
    ctx.fire(
        Fire::at("quant/qmm_t.metal", point.entry)
            .stamp(point.stamp)
            .apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            bias.arg(),
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
    let point = qmm_point(
        "_residual",
        "PIE_STAMP_qmm_t_residual",
        *group,
        *bits,
        *bm,
        *bn,
    )?;
    ctx.fire(
        Fire::at("quant/qmm_t.metal", point.entry)
            .stamp(point.stamp)
            .apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, [32, 2, 2])),
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
        Fire::at("quant/qmm_t.metal", qmm_precast_name("", "", *bm, *bn)?)
            .apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
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
            "quant/qmm_t.metal",
            qmm_precast_name("_bias", "", *bm, *bn)?,
        )
        .apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            bias.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
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
            "quant/qmm_t.metal",
            qmm_precast_name("_residual", "", *bm, *bn)?,
        )
        .apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            residual.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
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

    let k_partition_size = ctx.param(3)?;

    let split_k_partition_stride = ctx.param(4)?;

    let split_k = ctx.param(5)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            "quant/qmm_t.metal",
            qmm_name("_splitk", *group, *bits, *bm, 32)?,
        )
        .apply(Grid::of(qmm_grid(n, 32, m, *bm, split_k)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            w.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            out.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
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

    let k_partition_size = ctx.param(3)?;

    let split_k_partition_stride = ctx.param(4)?;

    let split_k = ctx.param(5)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            "quant/qmm_t.metal",
            qmm_name("_splitk_f32", *group, *bits, *bm, 32)?,
        )
        .apply(Grid::of(qmm_grid(n, 32, m, *bm, split_k)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            w.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            out.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
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

    let k_partition_size = ctx.param(3)?;

    let split_k_partition_stride = ctx.param(4)?;

    let split_k = ctx.param(5)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            "quant/qmm_t.metal",
            qmm_precast_name("_splitk", "", *bm, 32)?,
        )
        .apply(Grid::of(qmm_grid(n, 32, m, *bm, split_k)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            w.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            out.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            w.arg(),
            half_in.arg(),
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

    let k_partition_size = ctx.param(3)?;

    let split_k_partition_stride = ctx.param(4)?;

    let split_k = ctx.param(5)?;
    let m = *m;
    ctx.fire(
        Fire::at(
            "quant/qmm_t.metal",
            qmm_precast_name("_splitk", "_f32", *bm, 32)?,
        )
        .apply(Grid::of(qmm_grid(n, 32, m, *bm, split_k)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            w.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            out.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            w.arg(),
            half_in.arg(),
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
            "quant/qmm_t.metal",
            qmm_name("_strided", *group, *bits, *bm, 32)?,
        )
        .apply(Grid::of(qmm_grid(n, 32, m, *bm, 1)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
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
            "quant/qmm_t.metal",
            qmm_name("_strided_residual", *group, *bits, *bm, 32)?,
        )
        .apply(Grid::of(qmm_grid(n, 32, m, *bm, 1)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            residual.arg(),
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
            "quant/qmm_t.metal",
            qmm_precast_name("_strided", "", *bm, 32)?,
        )
        .apply(Grid::of(qmm_grid(n, 32, m, *bm, 1)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            row_stride.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
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
            "quant/qmm_t.metal",
            qmm_precast_name("_strided", "_residual", *bm, 32)?,
        )
        .apply(Grid::of(qmm_grid(n, 32, m, *bm, 1)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            residual.arg(),
            row_stride.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
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
    let n = y.width;

    let split_k_partition_stride = ctx.param(3)?;

    let split_k = ctx.param(4)?;
    let m = *m;
    ctx.fire(
        Fire::at("quant/qmm_t.metal", "qmm_splitk_reduce_bfloat16")
            .apply(Grid::of(elementwise_rows(n, m)?, [256, 1, 1])),
        &[
            partial.arg(),
            partial.arg(),
            partial.arg(),
            partial.arg(),
            y.arg(),
            partial.arg(),
            n.arg(),
            partial.arg(),
            partial.arg(),
            partial.arg(),
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
    let n = y.width;

    let split_k_partition_stride = ctx.param(3)?;

    let split_k = ctx.param(4)?;
    let m = *m;
    ctx.fire(
        Fire::at("quant/qmm_t.metal", "qmm_splitk_reduce_f32_bfloat16")
            .apply(Grid::of(elementwise_rows(n, m)?, [256, 1, 1])),
        &[
            partial.arg(),
            partial.arg(),
            partial.arg(),
            partial.arg(),
            y.arg(),
            partial.arg(),
            n.arg(),
            partial.arg(),
            partial.arg(),
            partial.arg(),
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
    let count = ctx.param(3)?;
    ctx.fire(
        Fire::at("quant/qmm_t.metal", "cast_qmm_input_bfloat16_to_float16")
            .apply(Grid::of(elementwise(count, 1)?, [256, 1, 1])),
        &[
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            half_out.arg(),
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
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "quant/qmm_t.metal",
            "cast_qmm_input_strided_bfloat16_to_float16",
        )
        .apply(Grid::of(elementwise_rows(k, rows)?, [256, 1, 1])),
        &[
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            k.arg(),
            cast_in.arg(),
            cast_in.arg(),
            row_stride.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            half_out.arg(),
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
        Fire::at("quant/qmv.metal", qmv_name("fast", *group, *bits)?)
            .apply(Grid::of(qmv_grid(vecs, out_vec_size)?, [32, 2, 1])),
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
        Fire::at("quant/qmv.metal", qmv_name("fast_residual", *group, *bits)?)
            .apply(Grid::of(qmv_grid(vecs, out_vec_size)?, [32, 2, 1])),
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
        Fire::at("quant/qmv.metal", qmv_name("tail", 64, *bits)?)
            .apply(Grid::of(qmv_grid(vecs, out_vec_size)?, [32, 2, 1])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            w.arg(),
            w.arg(),
            in_vec_size.arg(),
            in_vec_size.arg(),
            1_i32.arg(),
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
        Fire::at("quant/qmv.metal", qmv_name("tail_bias", 64, *bits)?)
            .apply(Grid::of(qmv_grid(vecs, out_vec_size)?, [32, 2, 1])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            bias.arg(),
            w.arg(),
            in_vec_size.arg(),
            in_vec_size.arg(),
            1_i32.arg(),
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
        Fire::at("quant/qmm_t.metal", qmv_wide_strided_name(*bits)?)
            .apply(Grid::of(qmv_grid(quarters(m), out_vec_size)?, [32, 2, 1])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            w.arg(),
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
            "quant/qmm_t.metal",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
        )
        .apply(Grid::of(qmm_grid(n, 32, m, 128, 1)?, [32, 2, 2])),
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
            "quant/qmm_t.metal",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
        )
        .apply(Grid::of(qmm_grid(n, 32, m, 32, 1)?, [32, 2, 2])),
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
            "quant/qmm_t.metal",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
        )
        .apply(Grid::of(qmm_grid(n, 32, m, 64, 1)?, [32, 2, 2])),
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
            "quant/qmm_t.metal",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
        )
        .apply(Grid::of(qmm_grid(n, 32, m, 64, 1)?, [32, 2, 2])),
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
            "quant/qmm_t.metal",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
        )
        .apply(Grid::of(qmm_grid(n, 64, m, 64, 1)?, [32, 2, 2])),
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
        Fire::at("quant/transcode.metal", "affine_encode_u4_bf16")
            .apply(Grid::of(elementwise(groups, 1)?, [256, 1, 1])),
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
        Fire::at("quant/transcode.metal", "affine_encode_u4_f32")
            .apply(Grid::of(elementwise(groups, 1)?, [256, 1, 1])),
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
        Fire::at("quant/transcode.metal", "mxfp4_dequant_bf16")
            .apply(Grid::of(elementwise(blocks, 1)?, [256, 1, 1])),
        &[
            payload.arg(),
            exponents.arg(),
            out.arg(),
            blocks.arg(),
            block_size.arg(),
        ],
    )
}

#[cfg(test)]
mod stamping {
    use super::*;
    use std::collections::BTreeSet;

    /// A STAMP CARRIES THE NAME IT DECLARES, which is the one thing the two
    /// renderings must share.
    ///
    /// The whole argument for composing the stamp beside the entry is that
    /// they cannot drift. This is that claim, checked rather than asserted in
    /// a comment -- and it is checked over the PRODUCT, because a fold that
    /// was right at one point and wrong at another is the defect the
    /// fifty-four-line tables actually shipped.
    #[test]
    fn every_stamp_declares_the_entry_it_is_paired_with() {
        for form in ["", "_bias", "_residual", "_routed"] {
            for &gs in &[32, 64, 128] {
                for &b in &[4, 8] {
                    for &bm in &[16, 32, 64] {
                        for &bn in &[16, 32, 64] {
                            let p = qmm_point(form, "PIE_STAMP_x", gs, b, bm, bn)
                                .expect("an axis point");
                            assert!(
                                p.stamp.contains(&format!("\"{}\"", p.entry)),
                                "{} does not declare {}",
                                p.stamp,
                                p.entry
                            );
                        }
                    }
                }
            }
        }
    }

    /// TWO POINTS ARE TWO NAMES. An entry that two coordinates share is an
    /// aliasing: the grid is built for one shape and the pipeline computes
    /// another, which is how gemma4's logits came back all zero.
    #[test]
    fn distinct_points_compose_distinct_entries() {
        let composed = composed();
        let distinct: BTreeSet<&str> = composed.iter().map(|(_, name)| *name).collect();
        assert_eq!(distinct.len(), composed.len(), "two points share one name");
        assert_eq!(
            composed.len(),
            4 * 3 * 2 * 3 * 3,
            "the product is the census"
        );
    }

    /// THE STAMP IS THE FILE'S OWN LANGUAGE, not a signature restated here.
    ///
    /// What the host composes is a macro CALL; the `#define` it names holds
    /// the device parameter list, in `quant/qmm_t.metal`, written once. A
    /// stamp that spelled `const device uint32_t*` would have moved the ABI
    /// into Rust, which is the thing the instantiation lists were deleted for.
    #[test]
    fn a_stamp_is_a_macro_call_and_carries_no_signature() {
        let p = qmm_point("", "PIE_STAMP_qmm_t", 64, 4, 32, 32).expect("an axis point");
        assert_eq!(
            p.stamp,
            "PIE_STAMP_qmm_t(\"affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32\", 64, 4, 32, 32, 32)"
        );
    }

    /// AN UNMIGRATED FORM STAMPS NOTHING, and says so with an empty string --
    /// `Fire::stamp`'s "the file already declares it". `_splitk` and
    /// `_strided` keep their own `instantiate_*` lists in the shader.
    #[test]
    fn a_form_the_file_still_declares_carries_no_stamp() {
        let p = qmm_point("_splitk", "", 64, 4, 32, 32).expect("an axis point");
        assert!(p.stamp.is_empty());
        assert_eq!(
            p.entry,
            "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_32_bn_32"
        );
    }
}
