use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

const FILE: &str = "linear/tiled.cuh";

const BLOCK: u32 = 256;

const TILE_K: u32 = 64;

const LD_A: u32 = TILE_K + 8;

const BAND: u32 = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tuple {
    pub m: u32,
    pub n: u32,
    pub threads: u32,
    pub stages: u32,
}

impl Tuple {
    const fn smem(self) -> u32 {
        let staging = self.stages * self.m * LD_A;
        let epilogue = self.m * (self.n + 8);
        if staging > epilogue { staging * 2 } else { epilogue * 2 }
    }
}

pub const LONG: Tuple = Tuple {
    m: 64,
    n: 128,
    threads: 256,
    stages: 2,
};

pub const SHORT: Tuple = Tuple {
    m: 32,
    n: 128,
    threads: 256,
    stages: 4,
};

pub const LONG_PREFILL_ROWS: u32 = 512;

#[must_use]
pub const fn tuple_for(rows: u32) -> Tuple {
    if rows >= LONG_PREFILL_ROWS { LONG } else { SHORT }
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
    let tuple = tuple_for(y.rows);
    tiled_launch(ctx, "linear.matmul", act, codes, scales, biases, y, seat, tuple)
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
    let tuple = tuple_for(y.rows);
    tiled_launch(ctx, "linear.lm_head", act, codes, scales, biases, y, seat, tuple)
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
    tiled_launch(ctx, "linear.matmul", act, codes, scales, biases, y, seat, tuple)
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
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16" });
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
        stages,
    } = tuple;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::matmul_affine_tiled<{t}, ::pie::i32(4), ::pie::i32({group}), \
                 ::pie::i32({m}), ::pie::i32({tile_n}), ::pie::i32({threads}), \
                 ::pie::i32({stages})>"
            )),
        )
        .apply(
            Launch::grid([y.rows.div_ceil(m), n.div_ceil(tile_n), 1], [threads, 1, 1])
                .smem(tuple.smem()),
        ),
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Carve {
    pub bands: u32,
    pub split: u32,
}

impl Carve {
    #[must_use]
    pub const fn threads(self) -> u32 {
        32 * self.bands * self.split
    }

    const fn smem(self, rows: u32) -> u32 {
        if self.split > 1 {
            self.bands * self.split * rows * BAND * 4
        } else {
            0
        }
    }
}

pub const TARGET_WARPS: u32 = 8 * 1024;

pub const MIN_SPLIT: u32 = 8;

pub const THIN_SPLIT: u32 = 32;

pub const WIDE_SPLIT: u32 = 16;

pub const THIN_ROWS: u32 = 8;

#[must_use]
pub const fn carve_for(n: u32, rows: u32) -> Carve {
    let bands = n.div_ceil(BAND);
    let deepest = if rows <= THIN_ROWS { THIN_SPLIT } else { WIDE_SPLIT };
    #[allow(clippy::manual_checked_ops)]
    let want = if bands == 0 {
        deepest
    } else {
        (TARGET_WARPS / bands).next_power_of_two()
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
        split,
    }
}

const fn bucket(rows: u32) -> u32 {
    if rows <= 1 {
        1
    } else if rows <= 2 {
        2
    } else if rows <= 4 {
        4
    } else if rows <= 8 {
        8
    } else {
        16
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
        carve_for(y.width, y.rows),
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
        carve_for(y.width, y.rows),
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
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16" });
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
    let rows = bucket(y.rows);
    let Carve { bands, split } = carve;
    let threads = carve.threads();
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::gemv_affine_tiled<{t}, ::pie::i32(4), ::pie::i32({group}), \
                 ::pie::i32({rows}), ::pie::i32({bands}), ::pie::i32({split})>"
            )),
        )
        .apply(
            Launch::grid([(n_pad / BAND).div_ceil(bands), 1, 1], [threads, 1, 1])
                .smem(carve.smem(rows)),
        ),
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
        Fire::at(
            FILE,
            "::pie::linear::repack_affine_tiled<::pie::i32(4)>",
        )
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
