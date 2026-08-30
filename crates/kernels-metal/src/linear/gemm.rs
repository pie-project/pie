//! `Gemm`: dense projections against a transposed weight. The narrow-vs-wide
//! choice lives here, inside the entry (decision #13) — a dispatch arm never
//! sees it.
//!
//! # A LANE'S LOGITS MUST NOT DEPEND ON HOW MANY LANES RODE WITH IT
//!
//! The row count this entry keys its arm on is the FIRE'S, and a fire's rows
//! are however many rows the composition put in it — which is to say, who
//! else was being served at that instant. So an arm chosen on that number is
//! a lane's answer chosen on its neighbours, and the only thing that makes
//! that legal is two arms that land the same bits.
//!
//! **THEY DID NOT.** Until this was fixed the narrow rung was
//! `dense_gemv_t_bfloat16`, which splits K across a simdgroup's 32 lanes and
//! folds the pieces with `simd_sum`; the wide rung walks K in ascending
//! 8-wide chunks on the matrix unit. Same arithmetic, different order,
//! different rounding — and a 31-token prompt is 31 rows alone (the vector
//! kernel) and 248 rows eight ways at once (the tile), so the same prompt at
//! temperature 0 answered differently in a crowd. Measured on an M1 Max over
//! qwen35-d0.8b-bf16: 0.45 of a logit at the prefill readout, and a different
//! sentence by the ninth token. `test_curated.py`'s
//! `greedy-decoding-is-the-same-alone-and-in-a-crowd` is the gate that says
//! so at the serving door; `engine-metal`'s own
//! `a_lane_answers_the_same_in_a_crowd` is the same claim with the feedback
//! loop cut.
//!
//! The fix is `engine-cuda`'s, in this plane's terms: the narrow rung is now
//! the SAME kernel at an 8-row block. `BM`, `BK`, `WM` and `WN` decide who
//! holds which element, never the order k is walked in, so the two rungs are
//! bit-identical and the threshold between them is invisible in the answer.
//! The vector kernel is gone rather than demoted — an arm nobody may take is
//! not an arm.
//!
//! **WHAT IT COST, STATED.** The matrix unit multiplies 8x8 fragments, so the
//! narrow rung's floor is eight rows and a one-row decode pays for eight.
//! One decode fire of qwen35-d0.8b on an M1 Max, before against after:
//!
//! ```text
//!   lanes    1      2      4      8     16     31
//!   before  10.3   12.9   22.1   40.3   76.9  145.1  ms  (the vector rung)
//!   after   13.5   13.8   16.7   22.2   37.7   66.7  ms  (the 8-row block)
//! ```
//!
//! One lane is 24% slower, two are 6% slower, and every width above that is
//! 1.3x to 2.2x FASTER — because the vector rung gave each output column one
//! simdgroup and no reuse of the weight across the rows sharing it, so a
//! fleet of decodes walked the whole weight table once per ROW. The correctness
//! this was done for is not paid for out of throughput; it is paid for out of
//! one-lane latency, and only there.

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

const FILE: &str = "linear/gemm_dense.metal";

const LANES_PER_TILE: u32 = 32;

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

/// `y = act x w^T`. Skinny fires (fewer than [`TILE_M`] rows — the decode
/// path) take the 8-row block; everything else takes the 32x32 tile. The two
/// are one kernel at two row blocks and land the same bits, which is what
/// makes a threshold on a number the composition owns legal at all — see this
/// module's header. A fire with no rows lands nothing and encodes nothing: see
/// [`extent`] for which of the three extents is allowed to be zero and why.
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
    let (entry, grid) = if rows < TILE_M {
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
