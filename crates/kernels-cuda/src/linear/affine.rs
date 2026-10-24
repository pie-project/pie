//! Affine-quantized projections (`linear/affine.cu`): codes, per-group scales
//! and offsets, read at bf16. `tiled` is the repacked 4-bit tile and its
//! matvec; the decode-then-gemm forms are `dequant`'s.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::linear::moe::GroupSeat;
#[cfg(feature = "cuda")]
use crate::linear::w8a16::affine_w4a16_mma;
use crate::tensor::Tensor;

pub mod tiled;

const AFFINE: &str = "linear/affine.cu";

const WARP: u32 = 32;

pub(crate) fn extent(op: &'static str, n: u64) -> Result<u32, Error> {
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
    pub(crate) const fn axis(self) -> &'static str {
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

#[derive(Clone, Copy)]
pub(crate) struct Affine {
    pub(crate) bits: u32,
    pub(crate) group: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
}

pub(crate) fn affine(
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
                format!(
                    "{} was declared with no offset plane to read",
                    offset.spelling()
                ),
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

/// The most rows the 4-bit affine GEMV on mma.sync (`affine_w4a16_mma`)
/// takes: one 16-row tile of the mma.
pub const AFFINE_MMA_UNTIL: u32 = 16;

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
    // A decode's few rows of a 4-bit MLX plane on the tensor cores, its
    // codes read once for every row.
    #[cfg(feature = "cuda")]
    if let Some(biases) = biases
        && bits == 4
        && group == 64
        && matches!(offset, OffsetKind::Post)
        && act.dtype == Dtype::Bf16
        && factor == Dtype::Bf16
        && y.dtype == Dtype::Bf16
        && !seat.streams()
        && y.rows <= AFFINE_MMA_UNTIL
        && k.is_multiple_of(64)
        && ctx.tuning().affine_mma
    {
        return affine_w4a16_mma(ctx, op, act, codes, scales, biases, y, n, k);
    }
    let tile = (BLOCK_LANES / WARP) * ROWS_PER_WARP;
    ctx.fire(
        op,
        Fire::at(kernel(
            AFFINE,
            &format!(
                "::pie::linear::matmul_affine<{t}, {f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group}), ::pie::i32({ROWS_PER_WARP})>",
                offset.axis()
            ),
        ))
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
