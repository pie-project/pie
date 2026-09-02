//! Tiled W4A16 GEMM: reads a repacked weight plane once into mma B
//! fragments, folding the post-affine dequant in registers.

use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

const FILE: &str = "linear/tiled.cuh";

/// One thread per output word; the relabelling passes are unshaped.
const BLOCK: u32 = 256;

/// Contraction step per mainloop iteration. Matches `kTiledK` in
/// `linear/tiled.cuh`; the repacked layout is grouped by it.
const TILE_K: u32 = 64;

/// Staged activation row stride in bf16: `TILE_K` plus 8 padding elements
/// to spread one `ldmatrix`'s addresses across shared-memory banks.
const LD_A: u32 = TILE_K + 8;

/// The mma tile's n extent and one warp's column span. A plane whose `n`
/// is not a whole band is padded with zero codes/factors in the tail.
const BAND: u32 = 16;

/// Mirrors the template parameters in `linear/tiled.cuh`; keep the two in
/// sync — a launch that doesn't match the kernel's tile is not refused,
/// it just answers wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tuple {
    /// The rows one block lands.
    pub m: u32,
    /// Columns one block lands; `warps * 16`, one warp per 16-column band.
    pub n: u32,
    pub threads: u32,
    /// cp.async pipeline depth over the activation tile.
    pub stages: u32,
}

impl Tuple {
    /// Shared memory needed: the larger of the staged activation tile and
    /// the epilogue tile it's reused as. The weight never lands in smem.
    const fn smem(self) -> u32 {
        let staging = self.stages * self.m * LD_A;
        let epilogue = self.m * (self.n + 8);
        if staging > epilogue { staging * 2 } else { epilogue * 2 }
    }
}

/// Long-prefill tile: 64 rows by 128 columns, 8 warps, 2-stage pipeline.
pub const LONG: Tuple = Tuple {
    m: 64,
    n: 128,
    threads: 256,
    stages: 2,
};

/// Short-prefill tile: 32 rows by 128 columns, 8 warps, 4-stage pipeline.
/// Half the row tile of `LONG`, to double the grid when there aren't
/// enough row tiles to fill the SMs.
pub const SHORT: Tuple = Tuple {
    m: 32,
    n: 128,
    threads: 256,
    stages: 4,
};

/// Row count above which `LONG`'s taller tile wins; below it `SHORT`'s
/// doubled grid keeps more SMs busy. Same crossover in both projection
/// directions.
pub const LONG_PREFILL_ROWS: u32 = 512;

/// Picks `LONG` or `SHORT` by row count alone, not by the n-vs-k aspect
/// ratio: the row-tile crossover sits at the same place in both
/// projection directions.
#[must_use]
pub const fn tuple_for(rows: u32) -> Tuple {
    if rows >= LONG_PREFILL_ROWS { LONG } else { SHORT }
}

/// The planes this point was handed, measured and checked.
#[derive(Clone, Copy)]
struct Tiled {
    /// Codes per factor (`k / groups`).
    group: u32,
    /// `n` rounded up to a whole 16-column band; the row count the
    /// repacked planes carry.
    n_pad: u32,
}

/// `n` rounded up to a whole band.
const fn padded(n: u32) -> u32 {
    n.div_ceil(BAND) * BAND
}

/// Validates a repacked plane's layout: four-bit codes, a bf16
/// (scale, bias) pair, post-offset fold only. Anything else is refused.
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
    // `k` must be a whole number of `TILE_K`-wide steps; the mainloop has
    // no tail stage.
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
    if scales.width == 0 || scales.width % 2 != 0 {
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
    // A 16-wide k tile is one lane's whole B fragment; it must sit inside
    // one group, or the lane would fold two contraction positions against
    // the wrong factor.
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

/// `y = act · W^T` over repacked planes, with `W[n][k] = s*code + b` folded
/// in registers at the mma. Requires `repack`'s layout, not row-major — an
/// un-repacked plane is not refusable and answers nonsense. Refuses a
/// streamed seat: a repacked plane is a load-time artifact of a resident one.
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

/// `matmul` under the head's own op name.
///
/// # Errors
///
/// Same as `matmul`.
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

/// `matmul` with the config tuple named explicitly, for the timing sweep
/// that justifies `tuple_for`'s pick. Not a runtime tuning hook.
///
/// # Errors
///
/// `matmul`'s plane ladder, plus a tuple whose row tile doesn't divide the
/// block's thread count.
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

/// Shared launch path for `matmul`, `lm_head` and `matmul_with`.
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
    // Repacked planes carry `n_pad` rows, not `n`: a band starting inside
    // `n` still reads all 16 of its columns.
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
            // Staged-geometry seat: live-rows word when a body replay
            // armed one, else the null seat.
            ctx.stage(),
        ],
    )
}

/// One warp per (band, split) pair. `bands` warps cover adjacent
/// 16-column bands; `split` warps share one band and reduce through
/// shared memory. `bands * split` is the block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Carve {
    /// 16-column bands one block covers, one per warp.
    pub bands: u32,
    /// Contraction slices one band is split across.
    pub split: u32,
}

impl Carve {
    #[must_use]
    pub const fn threads(self) -> u32 {
        32 * self.bands * self.split
    }

    /// Cross-slice reduction buffer; empty when `split == 1`, since a
    /// warp then owns its band's whole contraction.
    const fn smem(self, rows: u32) -> u32 {
        if self.split > 1 {
            self.bands * self.split * rows * BAND * 4
        } else {
            0
        }
    }
}

/// Target concurrent warps to cover the decode step's memory latency.
pub const TARGET_WARPS: u32 = 8 * 1024;

/// Shallowest split. A band's warps are also this point's only prefetch
/// depth: one superword in flight per warp.
pub const MIN_SPLIT: u32 = 8;

/// Deepest split a thin batch can afford: one block is `32 * split`
/// threads.
pub const THIN_SPLIT: u32 = 32;

/// Deepest split a full mma tile of rows can afford. Accumulators are 2
/// registers per row per lane; occupancy falls off above this.
pub const WIDE_SPLIT: u32 = 16;

/// Row count threshold between `THIN_SPLIT` and `WIDE_SPLIT`.
pub const THIN_ROWS: u32 = 8;

/// Picks a carve: one band per block (a decode step shares no activation
/// across bands, so a wider block buys nothing) and a contraction split
/// sized to keep around `TARGET_WARPS` busy.
#[must_use]
pub const fn carve_for(n: u32, rows: u32) -> Carve {
    let bands = n.div_ceil(BAND);
    let deepest = if rows <= THIN_ROWS { THIN_SPLIT } else { WIDE_SPLIT };
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

/// Rounds up to the next power of two, so the jit's kernel-name cache
/// sees at most 5 buckets instead of one per row count. The kernel masks
/// against the live count.
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

/// Decode-step reader for `repack`'s planes, for row counts too small for
/// a tensor-core tile.
///
/// # Errors
///
/// `matmul`'s plane ladder, a streamed seat, or more than 16 rows.
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

/// `matmul_gemv` under the head's own op name.
///
/// # Errors
///
/// Same as `matmul_gemv`.
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

/// `matmul_gemv` with the carve named explicitly, for the timing sweep
/// that justifies `carve_for`'s pick.
///
/// # Errors
///
/// Same as `matmul_gemv`.
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
    // Row cap is this kernel's own register-accumulator limit, not a
    // dispatch rule: a caller above it wanted `matmul`.
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
            // Staged-geometry seat: live-rows word when a body replay
            // armed one, else the null seat.
            ctx.stage(),
        ],
    )
}

/// Load-time relabelling into `matmul`'s fragment order: shifts, masks
/// and a permutation only, no arithmetic touches a value. Destination
/// rows are `n` padded to a whole band; the tail is zero codes/factors,
/// which decodes to a zero weight.
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

/// Row count `repack`'s destinations carry for an `n`-row weight.
#[must_use]
pub const fn repacked_rows(n: u32) -> u32 {
    padded(n)
}
