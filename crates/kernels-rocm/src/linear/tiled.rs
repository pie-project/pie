use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::moe::GroupSeat;
use crate::linear::quant::GemmSplit;
use crate::tensor::Tensor;

const FILE: &str = "linear/tiled.hip";

const BLOCK: u32 = 256;

const TILE_K: u32 = 64;

const BAND: u32 = 16;

/// The prefill GEMM's tile: `m` activation rows (32, 64 or 128) by `n`
/// output columns, `threads` a block. The kernel lands 128 columns with
/// 256 threads, single-staged in LDS with the next stage's reads in flight.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tuple {
    pub m: u32,
    pub n: u32,
    pub threads: u32,
    pub stages: u32,
}

pub const LONG: Tuple = Tuple {
    m: 128,
    n: 128,
    threads: 256,
    stages: 1,
};

pub const SHORT: Tuple = Tuple {
    m: 32,
    n: 128,
    threads: 256,
    stages: 1,
};

/// The tile `rows` rows take: the long one from `long_rows`
/// (`kernel.qmm.long_rows`), else the short one.
#[must_use]
pub const fn tuple_for(rows: u32, long_rows: u32) -> Tuple {
    if rows >= long_rows { LONG } else { SHORT }
}

#[derive(Clone, Copy)]
struct Tiled {
    group: u32,
    n_pad: u32,
}

const fn padded(n: u32) -> u32 {
    n.div_ceil(BAND) * BAND
}

fn tiled(
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    n: u32,
    k: u32,
) -> Result<Tiled, Error> {
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(biases.dtype, Dtype::U8, "a packed plane binds as bytes");
    if !k.is_multiple_of(TILE_K) {
        return Err(refuse(
            op,
            format!("a {k}-wide row is not a whole number of {TILE_K}-wide contraction steps"),
        ));
    }
    if codes.width * 2 != k {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at something other than four bits",
                codes.width
            ),
        ));
    }
    if scales.width == 0 || !scales.width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "a {}-byte factor row is not a whole number of two-byte factors",
                scales.width
            ),
        ));
    }
    if biases.width != scales.width {
        return Err(refuse(
            op,
            format!(
                "the post-offset arm reads a {}-byte bias row beside a {}-byte scale row",
                biases.width, scales.width
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
    if !group.is_multiple_of(BAND) {
        return Err(refuse(
            op,
            format!("a {group}-code group is not a whole number of {BAND}-wide mma k tiles"),
        ));
    }
    Ok(Tiled {
        group,
        n_pad: padded(n),
    })
}

pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    let tuple = tuple_for(y.rows, ctx.tuning().qmm_long_rows);
    tiled_launch(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        biases,
        y,
        seat,
        tuple,
    )
}

pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    let tuple = tuple_for(y.rows, ctx.tuning().qmm_long_rows);
    tiled_launch(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        biases,
        y,
        seat,
        tuple,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_with(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    tuple: Tuple,
) -> Result<(), Error> {
    tiled_launch(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        biases,
        y,
        seat,
        tuple,
    )
}

#[allow(clippy::too_many_arguments)]
fn tiled_launch(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    tuple: Tuple,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    if seat.streams() {
        return Err(refuse(
            op,
            "the tiled arm serves resident dense projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    let Tiled { group, n_pad } = tiled(op, codes, scales, biases, n, k)?;
    if codes.rows != n_pad || scales.rows != n_pad || biases.rows != n_pad {
        return Err(refuse(
            op,
            format!(
                "a {n}-column projection repacks into {n_pad} rows, and the planes hold \
                 ({}, {}, {})",
                codes.rows, scales.rows, biases.rows
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    let Tuple {
        m,
        n: tile_n,
        threads,
        ..
    } = tuple;
    if tile_n != 128 || threads != 256 || ![32, 64, 128].contains(&m) {
        return Err(refuse(
            op,
            format!(
                "the tiled GEMM lands 128 columns with 256 threads and 32, 64 or 128 rows, \
                 not {tuple:?}"
            ),
        ));
    }
    let cus = ctx.multiprocessors().unwrap_or(304).max(1);
    let blocks = y.rows.div_ceil(m) * n.div_ceil(tile_n);
    let split = GemmSplit::plan(ctx, op, y.rows, n, k, blocks, cus)?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::matmul_affine_tiled<::pie::i32({group}), ::pie::i32({m})>"
            )),
        )
        .apply(Launch::grid(
            [y.rows.div_ceil(m), n.div_ceil(tile_n), split.splits],
            [threads, 1, 1],
        )),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            split.part_arg(),
            split.per_arg(op)?,
            stated(op, y.rows)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            ctx.stage(),
        ],
    )?;
    split.finalize(ctx, op, FILE, "::pie::bf16", y)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Carve {
    pub bands: u32,
    pub split: u32,
}

impl Carve {
    /// A block is one band's `split` wavefronts.
    #[must_use]
    pub const fn threads(self) -> u32 {
        64 * self.bands * self.split
    }
}

/// Wavefronts a block at most.
const MAX_SPLIT: u32 = 16;

pub const MIN_SPLIT: u32 = 8;

pub const THIN_ROWS: u32 = 8;

/// Split-K depth for `rows` x `n`: aims for `kernel.qmm.split_warps_per_sm`
/// warps on every SM, capped by the thin (8 rows or fewer) or wide maximum.
fn carve_for(n: u32, rows: u32, ctx: &Ctx) -> Carve {
    let tuning = ctx.tuning();
    // A context that cannot say its SM count aims as an L40S would.
    let sms = ctx.multiprocessors().unwrap_or(142);
    // The knob counts 32-lane warps; a 64-lane wavefront is two.
    let warps = (sms.saturating_mul(tuning.qmm_split_warps_per_sm) / 2).max(1);
    let bands = n.div_ceil(BAND);
    let deepest = if rows <= THIN_ROWS {
        tuning.qmm_split_thin_max
    } else {
        tuning.qmm_split_wide_max
    };
    #[allow(clippy::manual_checked_ops)]
    let want = if bands == 0 {
        deepest
    } else {
        (warps / bands).next_power_of_two()
    };
    let split = if want < MIN_SPLIT {
        MIN_SPLIT
    } else if want > deepest {
        deepest
    } else {
        want
    };
    Carve {
        bands: 1,
        split: split.min(MAX_SPLIT),
    }
}

pub fn matmul_gemv(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    gemv_with(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        biases,
        y,
        seat,
        carve_for(y.width, y.rows, ctx),
    )
}

pub fn lm_head_gemv(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    gemv_with(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        biases,
        y,
        seat,
        carve_for(y.width, y.rows, ctx),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gemv_with(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    carve: Carve,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    if seat.streams() {
        return Err(refuse(
            op,
            "the tiled arm serves resident dense projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    let Tiled { group, n_pad } = tiled(op, codes, scales, biases, n, k)?;
    if codes.rows != n_pad || scales.rows != n_pad || biases.rows != n_pad {
        return Err(refuse(
            op,
            format!(
                "a {n}-column projection repacks into {n_pad} rows, and the planes hold \
                 ({}, {}, {})",
                codes.rows, scales.rows, biases.rows
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    if y.rows > BAND {
        return Err(refuse(
            op,
            format!(
                "the decode point holds {} rows in registers, and a {}-row fire is the \
                 tiled GEMM's shape",
                BAND, y.rows
            ),
        ));
    }
    let Carve { bands, split } = carve;
    if bands != 1 || !(1..=MAX_SPLIT).contains(&split) {
        return Err(refuse(
            op,
            format!("a block is one band of 1 to {MAX_SPLIT} wavefronts, not {carve:?}"),
        ));
    }
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::gemv_affine_tiled<::pie::i32({group}), ::pie::i32({split})>"
            )),
        )
        .apply(Launch::grid([n_pad / BAND, 1, 1], [carve.threads(), 1, 1])),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            stated(op, y.rows)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn repack(
    ctx: &Ctx,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    out_codes: &mut Tensor,
    out_scales: &mut Tensor,
    out_biases: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.repack";

    let n = nonzero(OP, "N, the rows of the weight", codes.rows)?;
    let k = nonzero(OP, "K, the width of the weight", codes.width)? * 2;
    let Tiled { group, n_pad } = tiled(OP, codes, scales, biases, n, k)?;
    if scales.rows != n || biases.rows != n {
        return Err(refuse(
            OP,
            format!(
                "an {n}-row code plane sits beside a {}-row scale plane and a {}-row bias \
                 plane",
                scales.rows, biases.rows
            ),
        ));
    }
    if out_codes.rows != n_pad
        || out_codes.width != codes.width
        || out_scales.rows != n_pad
        || out_scales.width != scales.width
        || out_biases.rows != n_pad
        || out_biases.width != biases.width
    {
        return Err(refuse(
            OP,
            format!(
                "the repack of a [{n}, {k}] weight is a [{n_pad}, {}] code plane and two \
                 [{n_pad}, {}] factor planes",
                codes.width, scales.width
            ),
        ));
    }
    let groups = k / group;
    let words = n_pad / BAND * (k / BAND) * 32;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::repack_affine_tiled<::pie::i32(4)>")
            .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            out_codes.arg(),
            stated(OP, n)?.arg(),
            stated(OP, k)?.arg(),
        ],
    )?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::repack_factors_tiled")
            .apply(Launch::flat(n_pad * groups, BLOCK)),
        &[
            scales.arg(),
            biases.arg(),
            out_scales.arg(),
            out_biases.arg(),
            stated(OP, n)?.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

#[must_use]
pub const fn repacked_rows(n: u32) -> u32 {
    padded(n)
}
