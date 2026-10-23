//! The staged tile (`linear/qmm_staged.metal`): eight rows a tile, the
//! contraction split in partials when the columns alone do not fill the
//! device; wide enough, the register-fed cooperative tile instead.

use dtype::Dtype;

use crate::Tuning;
use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};
use crate::tuning::Qmm;

use super::sg::Segment;
use super::tiled::splitk_reduce;
use super::{Scratch, symbol};

const STAGED_FILE: &str = "linear/qmm_staged.metal";

/// The rows of one staged tile.
pub(super) const TILE: i32 = 8;

const STAGED_SPLIT_WIDE: i32 = 8;

const STAGED_SPLIT: i32 = 4;

const STAGED_DEEP: i32 = 8192;

const STAGED_FEW_TILES: i32 = 32;

/// The contraction span the register-fed tile permutes within: four quant groups.
const COOP_SPAN: i32 = 256;

/// Narrower than this the staged tile is as quick (a 1024-column projection is 32 tiles).
const COOP_LEAST_COLUMNS: i32 = 4096;

/// From this many columns the register-fed tile fills the device without a k cut.
const COOP_WHOLE_COLUMNS: i32 = 12288;

const COOP_SPLIT: i32 = 4;

#[allow(clippy::too_many_arguments)]
pub(super) fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    (act, w, biases, y): (Tensor, Bank, Tensor, Tensor),
    scratch: &Scratch<'_>,
    (m, n, k, cols, ld): (i32, i32, i32, i32, i32),
) -> Result<(), Error> {
    let tiles = cols / 64;
    if cols >= COOP_LEAST_COLUMNS && k % COOP_SPAN == 0 {
        // Wide enough to fill the device from its columns alone, it runs whole; narrower, the
        // contraction is cut in four, within each column tile's threadgroup, so a 5120-column
        // projection is not 160 simdgroups.
        let split = if cols >= COOP_WHOLE_COLUMNS || (k / COOP_SPAN) % COOP_SPLIT != 0 {
            1
        } else {
            COOP_SPLIT
        };
        return coop(ctx, act, w, biases, y, (m, n, k, cols, ld), split);
    }
    let mut split = if k >= STAGED_DEEP || tiles < STAGED_FEW_TILES {
        STAGED_SPLIT_WIDE
    } else {
        STAGED_SPLIT
    };
    while split > 1 && (k / 64) % split != 0 {
        split /= 2;
    }
    let partial = if split > 1 {
        (scratch.partials)(m.unsigned_abs() * split.unsigned_abs(), n.unsigned_abs())
    } else {
        None
    };
    let split = if partial.is_some() { split } else { 1 };
    let entry = symbol(&format!("affine_qmm_staged_m{TILE}_k{split}"));
    let stamp = symbol(&format!("PIE_STAGED_POINT(\"{entry}\",{TILE},{split})"));
    ctx.fire(
        Fire::at(STAGED_FILE, entry).stamp(stamp).apply(Grid::of(
            [64 * tiles.unsigned_abs(), split.unsigned_abs(), 1],
            [64, 1, 1],
        )),
        &[
            w.codes.arg(),
            w.scales.arg(),
            biases.arg(),
            act.arg(),
            y.arg_mut(),
            k.arg(),
            n.arg(),
            match partial {
                Some(partial) => partial.arg_mut(),
                None => ctx.absent()?,
            },
            m.arg(),
            ld.arg(),
        ],
    )?;
    match partial {
        Some(partial) => splitk_reduce(ctx, op, y, partial, (m, n, cols), split),
        None => Ok(()),
    }
}

/// One register-fed (Apple10 cooperative) tile matmul of `act` into `y`: `(m, n, k, cols, ld)`
/// its rows, row stride, contraction, columns computed and the bank's row stride; `split`
/// partitions of the contraction, each a pair of simdgroups of the column tile's threadgroup,
/// which sums them in place.
#[allow(clippy::too_many_arguments)]
pub fn coop(
    ctx: &Ctx<'_>,
    act: Tensor,
    w: Bank,
    biases: Tensor,
    y: Tensor,
    (m, n, k, cols, ld): (i32, i32, i32, i32, i32),
    split: i32,
) -> Result<(), Error> {
    let tiles = (cols / 64).unsigned_abs();
    let entry = symbol(&format!("affine_qmm_coop_m{TILE}_k{split}"));
    let stamp = symbol(&format!("PIE_COOP_POINT(\"{entry}\",{split})"));
    let threads = 64 * split.unsigned_abs();
    ctx.fire(
        Fire::at(STAGED_FILE, entry)
            .stamp(stamp)
            .apply(Grid::of([threads * tiles, 1, 1], [threads, 1, 1])),
        &[
            w.codes.arg(),
            w.scales.arg(),
            biases.arg(),
            act.arg(),
            y.arg_mut(),
            k.arg(),
            n.arg(),
            ctx.absent()?,
            m.arg(),
            ld.arg(),
        ],
    )
}

/// Two or three matmuls of one `m`-row, `k`-wide activation in one register-fed (Apple10
/// cooperative) launch, each segment's columns over the whole contraction as [`coop`] walks
/// them unsplit (`Segment::split` is not read).
pub fn coop_segments(
    ctx: &Ctx<'_>,
    segments: &[Segment],
    act: Tensor,
    m: i32,
    k: i32,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_segments";
    if !(2..=3).contains(&segments.len()) {
        return Err(refuse(OP, "a segment launch takes two or three matmuls"));
    }
    if !(1..=8).contains(&m) || k % COOP_SPAN != 0 {
        return Err(refuse(
            OP,
            "the register-fed tile takes up to 8 rows of whole 256-input spans",
        ));
    }
    let mut tiles = 0u32;
    let mut args = Vec::with_capacity(24);
    for s in segments {
        let Some(biases) = s.w.biases else {
            return Err(refuse(OP, "a segment's bank holds no biases"));
        };
        let ld = stated(OP, s.w.ld)?;
        if s.w.bits != 4
            || s.w.group != 64
            || !s.y.width.is_multiple_of(32)
            || !s.first.is_multiple_of(32)
            || s.first + s.y.width > s.w.codes.rows
            || ld % 64 != 0
            || ld < k
            || s.y.rows != m.unsigned_abs()
        {
            return Err(refuse(OP, "a segment the register-fed tile does not walk"));
        }
        tiles += s.y.width / 32;
        args.extend([
            s.w.codes.arg(),
            s.w.scales.arg(),
            biases.arg(),
            s.y.arg_mut(),
        ]);
    }
    for _ in segments.len()..3 {
        let absent = ctx.absent()?;
        args.extend([absent; 4]);
    }
    args.extend([act.arg(), k.arg(), m.arg()]);
    let at = |i: usize, of: &dyn Fn(&Segment) -> i32| segments.get(i).map_or(0, of);
    for of in [
        &(|s: &Segment| s.y.width.cast_signed()) as &dyn Fn(&Segment) -> i32,
        &|s: &Segment| s.w.ld.cast_signed(),
        &|s: &Segment| s.first.cast_signed(),
    ] {
        for i in 0..3 {
            args.push(at(i, of).arg());
        }
    }
    ctx.fire(
        Fire::at(STAGED_FILE, "affine_qmm_coop_segments")
            .apply(Grid::of([64 * tiles.div_ceil(2), 1, 1], [64, 1, 1])),
        &args,
    )
}

/// Whether [`coop_segments`] takes matmuls of `rows` rows by 4-bit, 64-group banks of `widths`
/// columns over a contraction of `k`: the staged rung runs the widest of them first, every
/// segment is whole simdgroups of columns, and together they fill the device unsplit.
#[must_use]
pub fn coop_segments_fit(tuning: &Tuning, rows: u32, widths: &[u32], k: u32) -> bool {
    let (Some(&widest), total) = (widths.iter().max(), widths.iter().sum::<u32>()) else {
        return false;
    };
    rows > 0
        && rows <= TILE.unsigned_abs()
        && k.is_multiple_of(COOP_SPAN.unsigned_abs())
        && widths.iter().all(|w| w.is_multiple_of(32))
        && total >= COOP_WHOLE_COLUMNS.unsigned_abs()
        && tuning.qmm(&tuning.qmm_at(rows, widest, k, 4, 64)).first() == Some(&Qmm::Staged)
}

/// The gate and up of a packed `[gate | up]` bank and their swiglu into `h` in one
/// register-fed (Apple10 cooperative) launch, where the matmul of `act` by the whole bank
/// takes that tile with no contraction split: `h` lands as [`super::matmul`] then
/// [`crate::linear::mlp::swiglu`] land it, bit for bit, without the packed plane between them.
/// `Ok(false)`, and nothing fired, where the matmul takes another tile.
pub fn coop_swiglu(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    act: Tensor,
    w: Bank,
    intermediate: u32,
    h: Tensor,
    capacity_rows: u32,
) -> Result<bool, Error> {
    const OP: &str = "linear.glu";
    let Some(biases) = w.biases else {
        return Ok(false);
    };
    let (rows, k) = (act.rows, act.width);
    let cols = intermediate.saturating_mul(2);
    if act.dtype != Dtype::Bf16
        || h.dtype != Dtype::Bf16
        || w.codes.dtype != Dtype::U4g64
        || w.bits != 4
        || w.group != 64
        || h.rows != rows
        || h.width != intermediate
        || w.codes.rows < cols
        || w.codes.width != k
        || rows == 0
        || !intermediate.is_multiple_of(32)
        || !k.is_multiple_of(COOP_SPAN.unsigned_abs())
        || !w.ld.is_multiple_of(64)
        || w.ld < k
        || TILE.unsigned_abs() > capacity_rows
    {
        return Ok(false);
    }
    let at = tuning.qmm_at(rows, cols, k, w.bits, w.group);
    let (m, k, ld, n) = (
        stated(OP, rows)?,
        stated(OP, k)?,
        stated(OP, w.ld)?,
        stated(OP, intermediate)?,
    );
    // The matmul's own choice: the staged rung first, and on it the whole cooperative tile.
    if tuning.qmm(&at).first() != Some(&Qmm::Staged)
        || m > TILE
        || 2 * n < COOP_LEAST_COLUMNS
        || (2 * n < COOP_WHOLE_COLUMNS && (k / COOP_SPAN) % COOP_SPLIT == 0)
    {
        return Ok(false);
    }
    ctx.fire(
        Fire::at(STAGED_FILE, "affine_qmm_coop_swiglu")
            .apply(Grid::of([64 * (intermediate / 32), 1, 1], [64, 1, 1])),
        &[
            w.codes.arg(),
            w.scales.arg(),
            biases.arg(),
            act.arg(),
            h.arg_mut(),
            k.arg(),
            n.arg(),
            ctx.absent()?,
            m.arg(),
            ld.arg(),
        ],
    )?;
    Ok(true)
}
