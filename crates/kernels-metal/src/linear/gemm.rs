//! `Gemm`: dense projections against a transposed weight. The narrow-vs-wide
//! choice lives here, inside the entry (decision #13) — a dispatch arm never
//! sees it.
//!
//! # THREE RUNGS, KEYED ON THE FIRE'S ROWS
//!
//! ```text
//!   rows < VECTOR_MAX_ROWS (4)   dense_gemv_t_bfloat16
//!   rows < TILE_M (32)           dense_gemm_t_bfloat16_bm_8_bk_64_bn_32
//!   otherwise                    dense_gemm_t_bfloat16_bm_32_bk_32_bn_32
//! ```
//!
//! Measured on an M1 Max over qwen35-d0.8b, one decode fire, the vector rung
//! against the 8-row tile at every width:
//!
//! ```text
//!   lanes    1      2      4      8     16     31
//!   vector  10.3   12.9   22.1   40.3   76.9  145.1  ms
//!   tile    13.5   13.8   16.7   22.2   37.7   66.7  ms
//! ```
//!
//! The vector rung gives each output column one simdgroup and no reuse of the
//! weight across the rows sharing it, so a fleet of decodes walks the whole
//! weight table once per ROW — it climbs with the batch. The tile reads the
//! table once for eight rows of arithmetic and is nearly flat in it. The lines
//! cross between two rows and four, which is [`VECTOR_MAX_ROWS`], and the
//! third rung is what keeps BOTH ends: the vector point's one-lane latency and
//! the 8-row block's 1.3-2.2x from four lanes up.
//!
//! # THE ARM MOVES WITH THE COMPOSITION, AND THAT IS THE RULING
//!
//! A fire's rows are however many rows the composition put in it, so the arm
//! this entry picks — and a lane's low-order bits with it — is a function of
//! who else was being served at that instant. The two arms round differently:
//! `dense_gemv_t` splits K across a simdgroup's 32 lanes and folds the pieces
//! with `simd_sum`, the tiles walk K in ascending 8-wide chunks on the matrix
//! unit. Measured on qwen35-d0.8b-bf16, a 20-row prompt alone against the same
//! prompt in a crowd: 0.45 of a logit at the prefill readout, which a greedy
//! continuation can turn into a different sentence several tokens later.
//!
//! That drift is ACCEPTED:
//!
//! > We do NOT need bit-level identity. If a much faster path has small
//! > numerical drift from nondeterminism, that is obviously acceptable.
//!
//! A release deleted the vector rung to buy bit-identity across compositions,
//! at 24% of a one-lane decode; this is that decision reversed on the owner's
//! ruling, with the 8-row rung that release introduced KEPT, because it wins
//! on its own merits at the widths between. `a_lane_answers_the_same_in_a_crowd`
//! is the gate that holds what is still promised: the same TOKEN at every step
//! the model actually decided, with near-ties excused by a stated margin.

use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::tensor::Tensor;

pub const TILE_M: u32 = 32;

pub const TILE_N: u32 = 32;

/// The wide rung — [`TILE_M`] rows of output per threadgroup, over the four
/// simdgroups `TILE_GROUP` launches.
const TILE_ENTRY: &str = "dense_gemm_t_bfloat16_bm_32_bk_32_bn_32";

const TILE_GROUP: [u32; 3] = [32, 2, 2];

/// The narrow rung: the same kernel at the smallest row block the matrix
/// unit has (an 8x8 fragment is what it multiplies), over one simdgroup of
/// rows rather than two.
const NARROW_M: u32 = 8;

const NARROW_ENTRY: &str = "dense_gemm_t_bfloat16_bm_8_bk_64_bn_32";

const NARROW_GROUP: [u32; 3] = [32, 1, 2];

/// The floor: one simdgroup per output column, the contraction split across
/// its thirty-two lanes and folded with `simd_sum`. The fold IS where the
/// parallelism at one row comes from, and is also why this rung's bits are
/// not the tiles' — see this module's header.
const VECTOR_ENTRY: &str = "dense_gemv_t_bfloat16";

const VECTOR_GROUP: u32 = 128;

const LANES_PER_COLUMN: u32 = 32;

/// The fire width at which the 8-row tile overtakes the vector point, from
/// the table in this module's header: the vector rung wins at one and two
/// rows (10.3 against 13.5, 12.9 against 13.8) and loses at four (22.1
/// against 16.7). Four is the first measured width where it loses, so four is
/// where the ladder leaves it.
///
/// **IT IS A DENSE NUMBER AND NOT [`crate::tuning`]'s.**
/// `DeviceTuning::qmm_min_batch` is the same crossover for the QUANTIZED
/// ladder and reads five; the two GEMMs are different kernels over different
/// operands and there is no measurement saying they cross at the same width.
/// A device sweep that re-draws this curve is what would earn it a tuned
/// field, and none has been taken.
pub const VECTOR_MAX_ROWS: u32 = 4;

const FILE: &str = "linear/gemm_dense.metal";

const LANES_PER_TILE: u32 = 32;

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

/// `y = act x w^T`. The vector point under [`VECTOR_MAX_ROWS`] rows, the
/// 8-row block under [`TILE_M`], the 32x32 tile at or above it — see this
/// module's header for the measurement behind each threshold.
///
/// A fire with no rows lands nothing and encodes nothing: see [`extent`] for
/// which of the three extents is allowed to be zero and why.
pub fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let (rows, columns, contraction) = extent(op, act, y)?;
    if rows == 0 {
        return Ok(());
    }
    let (entry, grid) = if rows < VECTOR_MAX_ROWS {
        (VECTOR_ENTRY, vector_grid(op, rows, columns)?)
    } else if rows < TILE_M {
        (
            NARROW_ENTRY,
            tile_grid(op, rows, columns, NARROW_M, NARROW_GROUP)?,
        )
    } else {
        (
            TILE_ENTRY,
            tile_grid(op, rows, columns, TILE_M, TILE_GROUP)?,
        )
    };
    ctx.fire(
        Fire::at(FILE, entry).apply(grid),
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            stated(op, rows)?.arg(),
            stated(op, columns)?.arg(),
            stated(op, contraction)?.arg(),
        ],
    )
}

/// The three extents, of which exactly one may be zero.
///
/// The two WIDTHS are the weight's — fixed by the checkpoint, the same in
/// every fire this artifact runs — so a zero in either is a malformed weight
/// row that would land nothing forever, and is refused. The ROWS are the
/// composition's: a guarded region can legitimately compose to none, and a
/// refusal there would kill a whole fire over a node that simply had nothing
/// to do, so a zero row count comes back as a zero and the caller no-ops on
/// it. The widths are checked first, so a malformed weight answers the same on
/// every fire and not only on the ones that had rows to project.
fn extent(op: &'static str, act: Tensor, y: Tensor) -> Result<(u32, u32, u32), Error> {
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    Ok((y.rows, y.width, act.width))
}

/// The grid one rung launches: column tiles across, row blocks down, and the
/// rung's own threadgroup. `block` is the rung's `BM`; `group` is the shape
/// the kernel was instantiated at, whose `y` is how many simdgroups share the
/// block's rows.
fn tile_grid(
    op: &'static str,
    rows: u32,
    columns: u32,
    block: u32,
    group: [u32; 3],
) -> Result<Grid, Error> {
    let tiles = |extent: u32, tile: u32, per: u32, what: &'static str| -> Result<u32, Error> {
        extent
            .div_ceil(tile)
            .checked_mul(per)
            .ok_or_else(|| refuse(op, format!("{what} will not launch at {extent}")))
    };
    Ok(Grid::of(
        [
            tiles(columns, TILE_N, LANES_PER_TILE, "the column tiles")?,
            tiles(rows, block, group[1], "the row tiles")?,
            group[2],
        ],
        group,
    ))
}

/// The grid the vector rung launches: one simdgroup per output column, packed
/// [`VECTOR_GROUP`] lanes to a threadgroup, and one row of the grid's `y` per
/// row of the fire. The kernel guards both `n >= N` and `m >= M`, so a
/// column count the threadgroup does not divide is launched over and
/// discarded rather than refused.
fn vector_grid(op: &'static str, rows: u32, columns: u32) -> Result<Grid, Error> {
    let lanes = columns
        .div_ceil(VECTOR_GROUP / LANES_PER_COLUMN)
        .checked_mul(VECTOR_GROUP)
        .ok_or_else(|| {
            refuse(
                op,
                format!("the {columns} columns, one simdgroup each, will not launch"),
            )
        })?;
    Ok(Grid::of([lanes, rows, 1], [VECTOR_GROUP, 1, 1]))
}
