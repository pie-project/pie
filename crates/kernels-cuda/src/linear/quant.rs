//! Dtype casts and the weight quantizers the loader drives.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::gemm;
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

const FILE: &str = "linear/quant.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

/// Scratch key for the decoded bf16 `[n, k]` tile the prefill arm projects
/// through; grown to the widest projection requested, never shrunk.
const DECODED_WEIGHT: &str = "linear.quant.decoded_weight";

/// One block per row, sized to the row in whole warps.
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

/// A 32-bit launch extent; refused rather than truncated.
fn extent(op: &'static str, n: u64) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })
}

/// The offset arm: declared by the caller, checked against the bound planes
/// rather than inferred (a `Post` bias plane and a `PreReal` zero plane are
/// the same byte rectangle, so only the caller knows which is which).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OffsetKind {
    /// `s·c + b`, `b` a factor-dtype real per group (value-domain offset).
    Post,
    /// `s·(c − z)`, `z` an unsigned code-domain integer per group (GPTQ/AWQ).
    PreInt,
    /// `s·(c − z)`, `z` a factor-dtype real per group (HQQ).
    PreReal,
    /// `s·(c − 2^(bits−1))`, no offset plane (excess-binary symmetric rows).
    PreConst,
}

impl OffsetKind {
    /// The `kOffset` constant this arm is stamped as.
    const fn axis(self) -> &'static str {
        match self {
            Self::Post => "::pie::linear::kOffPost",
            Self::PreInt => "::pie::linear::kOffPreInt",
            Self::PreReal => "::pie::linear::kOffPreReal",
            Self::PreConst => "::pie::linear::kOffPreConst",
        }
    }

    /// The arm's name in a refusal, in the algebra's own words.
    const fn spelling(self) -> &'static str {
        match self {
            Self::Post => "a post-offset arm (`s·c + b`)",
            Self::PreInt => "an integer pre-offset arm (`s·(c − z)`)",
            Self::PreReal => "a real pre-offset arm (`s·(c − z)`)",
            Self::PreConst => "a constant pre-offset arm (`s·(c − 2^(bits−1))`)",
        }
    }
}

/// `linear.matmul` over a weight stored as codes plus a per-group factor
/// plane. The caller states the offset arm and the scales dtype (the scales
/// plane binds as raw `U8` bytes, so its dtype can't be read back out); the
/// bit width and group size are inferred from the codes/factor row widths.
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

/// [`matmul`] under the head's own op name, `linear::gemm`'s pairing kept.
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

/// Shared plane geometry, measured once for both [`dense_affine`] and
/// [`dense_affine_via_dense`] so the two arms can't disagree about it.
#[derive(Clone, Copy)]
struct Affine {
    /// Four or eight, off the codes row against `k`.
    bits: u32,
    /// The codes under one factor, off the factor row against `k`.
    group: u32,
    /// The rows this projection lands, which are the weight's rows.
    n: u32,
    /// The contraction it walks, which is the weight's width.
    k: u32,
}

/// Measures and checks [`Affine`]. `y`'s row count is not read here: an
/// empty fire is each caller's own no-op.
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
    // The arm and the bound plane must agree: `PreConst` reads no offset
    // plane, the other three arms require one.
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
    // Group count comes off the factor plane: it is `[n, k / group]`
    // two-byte factors per row, so half its byte width is the group count.
    if scales.width == 0 || scales.width % 2 != 0 {
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
    // The kernel reads codes a 32-bit word at a time, so a group is a whole
    // number of words or its second half would be read against the next
    // group's factor.
    let per_word = 32 / bits;
    if !group.is_multiple_of(per_word) {
        return Err(refuse(
            op,
            format!("a {group}-code group is not a whole number of {per_word}-code words"),
        ));
    }
    // `PreInt`'s zero is a `u4` (GPTQ/AWQ) but travels widened to one byte
    // per group, so the offset plane is the same shape at four and eight bits.
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

/// The one launch behind both dense entries, over all four offset arms.
/// A fire with no rows is a no-op, matching the dense gemm's behavior.
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
            // Live-rows word when a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// Prefill arm of [`matmul`]: decodes the stored planes once into a scratch
/// bf16 tile, then projects through dense cuBLAS instead of the fused
/// kernel. Faster over many rows; resident weights only, not streamed seats
/// (refused rather than silently falling back to the fused arm).
///
/// The fused arm rounds in f32 inside the dot; this arm rounds to bf16
/// once during decode, so the two agree in value but not bit-for-bit.
/// One affine plane decoded to a bf16 `[n, k]` rectangle in fire scratch,
/// for an entry that reads a dense weight (the MLA absorbs). Resident planes
/// only; the bit width and group come off the plane widths.
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

/// [`matmul_via_dense`] under the head's own op name — [`lm_head`]'s prefill
/// twin, and the same pairing `linear::gemm` keeps.
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

/// The two launches behind both prefill entries: `dequant_affine` into the
/// slab, then `linear::gemm`'s dense point over it.
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
    // Bf16 activations only; refused here, before any decode work.
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
    // Tile is bf16: two bytes an element.
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch(op, DECODED_WEIGHT, bytes)? as usize as u64;
    // One thread per code word (each writes `32 / bits` elements).
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

/// Scales each row of `buf` by the matching row of `l`, in place.
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
