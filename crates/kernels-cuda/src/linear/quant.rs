use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::gemm;
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

const FILE: &str = "linear/quant.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const DECODED_WEIGHT: &str = "linear.quant.decoded_weight";

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

fn extent(op: &'static str, n: u64) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OffsetKind {
    Post,
    PreInt,
    PreReal,
    PreConst,
}

impl OffsetKind {
    const fn axis(self) -> &'static str {
        match self {
            Self::Post => "::pie::linear::kOffPost",
            Self::PreInt => "::pie::linear::kOffPreInt",
            Self::PreReal => "::pie::linear::kOffPreReal",
            Self::PreConst => "::pie::linear::kOffPreConst",
        }
    }

    const fn spelling(self) -> &'static str {
        match self {
            Self::Post => "a post-offset arm (`s·c + b`)",
            Self::PreInt => "an integer pre-offset arm (`s·(c − z)`)",
            Self::PreReal => "a real pre-offset arm (`s·(c − z)`)",
            Self::PreConst => "a constant pre-offset arm (`s·(c − 2^(bits−1))`)",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

#[derive(Clone, Copy)]
struct Affine {
    bits: u32,
    group: u32,
    n: u32,
    k: u32,
}

fn affine(
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    y: &Tensor,
) -> Result<Affine, Error> {
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert!(
        biases.is_none_or(|b| b.dtype == Dtype::U8),
        "a packed plane binds as bytes"
    );
    match (offset, biases) {
        (OffsetKind::PreConst, Some(_)) => {
            return Err(refuse(
                op,
                format!(
                    "{} reads no offset plane, and one was bound",
                    offset.spelling()
                ),
            ));
        }
        (OffsetKind::Post | OffsetKind::PreInt | OffsetKind::PreReal, None) => {
            return Err(refuse(
                op,
                format!("{} was declared with no offset plane to read", offset.spelling()),
            ));
        }
        _ => {}
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    if scales.width == 0 || !scales.width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "a {}-byte factor row is not a whole number of two-byte factors",
                scales.width
            ),
        ));
    }
    let groups = scales.width / 2;
    if !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else if codes.width * 4 == k {
        2
    } else {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at neither two, four nor eight bits",
                codes.width
            ),
        ));
    };
    let per_word = 32 / bits;
    if !group.is_multiple_of(per_word) {
        return Err(refuse(
            op,
            format!("a {group}-code group is not a whole number of {per_word}-code words"),
        ));
    }
    if let Some(plane) = biases {
        let want = if offset == OffsetKind::PreInt {
            groups
        } else {
            scales.width
        };
        if plane.width != want {
            return Err(refuse(
                op,
                format!(
                    "{} states a {want}-byte offset row over a {k}-wide row, and the \
                     plane holds {}",
                    offset.spelling(),
                    plane.width
                ),
            ));
        }
    }
    Ok(Affine { bits, group, n, k })
}

#[allow(clippy::too_many_arguments)]
fn dense_affine(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const ROWS_PER_WARP: u32 = 4;
    const BLOCK_LANES: u32 = 128;

    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let Affine { bits, group, n, k } = affine(op, act, codes, scales, offset, biases, y)?;
    if y.rows == 0 {
        return Ok(());
    }
    let tile = (BLOCK_LANES / WARP) * ROWS_PER_WARP;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::matmul_affine<{t}, {f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group}), ::pie::i32({ROWS_PER_WARP})>",
                offset.axis()
            )),
        )
        .apply(Launch::grid(
            [y.rows, n.div_ceil(tile), 1],
            [BLOCK_LANES, 1, 1],
        )),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            y.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            ArgValue::Ptr(seat.cell),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decoded_plane(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    n: u32,
    k: u32,
    seat: GroupSeat,
) -> Result<Tensor, Error> {
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if seat.streams() {
        return Err(refuse(
            op,
            "a decoded plane serves resident projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    if codes.rows != n {
        return Err(refuse(
            op,
            format!("the code plane holds {} rows and the entry states {n}", codes.rows),
        ));
    }
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else if codes.width * 4 == k {
        2
    } else {
        return Err(refuse(
            op,
            format!("a {}-byte code row stores a {k}-wide row at neither two, four nor eight bits", codes.width),
        ));
    };
    let groups = scales.width / 2;
    if groups == 0 || !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch(op, DECODED_WEIGHT, bytes)? as usize as u64;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::dequant_affine<{f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group})>",
                offset.axis()
            )),
        )
        .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            ArgValue::Ptr(tile),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    Ok(Tensor::new(tile, n, k, Dtype::Bf16))
}

#[allow(clippy::too_many_arguments)]
pub fn decode_into(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    dst: u64,
    n: u32,
    k: u32,
) -> Result<Tensor, Error> {
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if codes.rows != n {
        return Err(refuse(
            op,
            format!("the code plane holds {} rows and the entry states {n}", codes.rows),
        ));
    }
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else if codes.width * 4 == k {
        2
    } else {
        return Err(refuse(
            op,
            format!("a {}-byte code row stores a {k}-wide row at neither two, four nor eight bits", codes.width),
        ));
    };
    let groups = scales.width / 2;
    if groups == 0 || !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::dequant_affine<{f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group})>",
                offset.axis()
            )),
        )
        .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            ArgValue::Ptr(dst),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    Ok(Tensor::new(dst, n, k, Dtype::Bf16))
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_via_dense(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine_via_dense(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lm_head_via_dense(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine_via_dense(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_affine_via_dense(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if seat.streams() {
        return Err(refuse(
            op,
            "the decoded-tile arm serves resident dense projections only, and these \
             planes are seated by a streaming tier",
        ));
    }
    let Affine { bits, group, n, k } = affine(op, act, codes, scales, offset, biases, y)?;
    if y.rows == 0 {
        return Ok(());
    }
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch(op, DECODED_WEIGHT, bytes)? as usize as u64;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::dequant_affine<{f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group})>",
                offset.axis()
            )),
        )
        .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            ArgValue::Ptr(tile),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    gemm::act_x_wt(ctx, op, act, Tensor::new(tile, n, k, Dtype::Bf16), y)
}

pub fn cast_fp32_to(ctx: &Ctx, src: Tensor, dst: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_cast_fp32_to";
    debug_assert_eq!(src.dtype, Dtype::F32, "`{OP}` casts an f32 source");
    let t = dtype_dispatch!(OP, dst.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let n = dst.elements();
    let lanes = nonzero(OP, "the cast's element count", extent(OP, n)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::cast_f32_to<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[src.arg(), dst.arg(), n.arg()],
    )
}

pub fn scale_rows(ctx: &Ctx, l: Tensor, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_scale_rows";
    let t = dtype_dispatch!(OP, buf.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", buf.rows)?;
    let width = stated(OP, nonzero(OP, "width", buf.width)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::scale_rows<{t}>")))
            .apply(route_rows(buf.rows, buf.width)),
        &[buf.arg(), l.arg(), width.arg()],
    )
}
