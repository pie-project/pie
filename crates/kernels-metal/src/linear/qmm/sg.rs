//! The register tile (`linear/qmm_sg.metal`): up to eight rows against 64
//! columns a threadgroup, the activation laid once as an input table, the
//! contraction split across partitions; and its segment launch, two or three
//! projections of one table.

use crate::Tuning;
use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

use super::{Scratch, symbol};

const REGISTER_FILE: &str = "linear/qmm_sg.metal";

/// The columns of one threadgroup of the register tile: four simdgroups of 16.
const REGISTER_COLUMNS: i32 = 64;

/// The contraction span the register tile stages and splits by.
const REGISTER_SPAN: i32 = 256;

/// The register tile's contraction partitions, as Splash's Apple9 register
/// tile takes them: doubled (to 8) while the grid stays under 4 threadgroups
/// a core with 256 inputs a partition, or under 32 a core with 1024. Partitions
/// take whole 256-input units, as many as fall to them, so an uneven count splits too.
#[must_use]
pub fn register_split(cols: i32, k: i32, cores: Option<u32>) -> i32 {
    let cores = i64::from(cores.unwrap_or(32));
    let grid = i64::from(cols / REGISTER_COLUMNS);
    let spans = k / REGISTER_SPAN;
    let mut split = 1i32;
    while split < 8 && spans >= 2 * split {
        let s = i64::from(split);
        let holds = |tg: i64, inputs: i32| grid * s < tg * cores && k / (2 * split) >= inputs;
        if !(holds(4, 256) || holds(32, 1024)) {
            break;
        }
        split *= 2;
    }
    split
}

/// Lays the register tile's input table for the `m` rows of `act` (`k`
/// wide) in `table`, with its first `counters` arrival counters zeroed.
pub fn register_table(
    ctx: &Ctx<'_>,
    act: Tensor,
    table: Tensor,
    m: i32,
    k: i32,
    counters: i32,
) -> Result<(), Error> {
    ctx.fire(
        Fire::at(REGISTER_FILE, "qmm_sg_table")
            .apply(Grid::of([32, (k / 64).unsigned_abs(), 1], [32, 4, 1])),
        &[act.arg(), table.arg_mut(), k.arg(), m.arg(), counters.arg()],
    )
}

/// One projection of a register-tile segment launch: the columns of bank
/// `w` from its row `first`, the output it lands them in whole (its width
/// the row stride and the columns' count) and its contraction's partitions.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub w: Bank,
    pub first: u32,
    pub y: Tensor,
    pub split: i32,
}

/// The arrival counters and partial sums (f32) a segment launch takes: one
/// counter per 64 columns and `split` rows of sums per row of every split
/// segment.
#[must_use]
pub fn segments_scratch(segments: &[Segment], m: u32) -> (u32, u32) {
    segments
        .iter()
        .filter(|s| s.split > 1)
        .fold((0, 0), |(counters, floats), s| {
            (
                counters + s.y.width / REGISTER_COLUMNS.unsigned_abs(),
                floats + s.split.unsigned_abs() * m * s.y.width,
            )
        })
}

/// Two or three register-tile matmuls of one laid input `table` (the `m`
/// rows of a `k`-wide activation) in one launch, each segment's columns
/// summed as its bank's own register matmul at the segment's split sums
/// them, so they land that matmul's bits. `partials` holds
/// [`segments_scratch`]'s sums and the table its counters.
pub fn register_segments(
    ctx: &Ctx<'_>,
    segments: &[Segment],
    table: Tensor,
    m: i32,
    k: i32,
    partials: Option<Tensor>,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_segments";
    if !(2..=3).contains(&segments.len()) {
        return Err(refuse(OP, "a segment launch takes two or three matmuls"));
    }
    if m <= 0 || m > 8 || k % REGISTER_SPAN != 0 {
        return Err(refuse(
            OP,
            "the register tile takes up to 8 rows of whole 256-input units",
        ));
    }
    let mut groups = 0u32;
    let mut args = Vec::with_capacity(28);
    for s in segments {
        let Some(biases) = s.w.biases else {
            return Err(refuse(OP, "a segment's bank holds no biases"));
        };
        let cols = stated(OP, s.y.width)?;
        let ld = stated(OP, s.w.ld)?;
        if s.w.bits != 4
            || s.w.group != 64
            || cols % REGISTER_COLUMNS != 0
            || s.first % REGISTER_COLUMNS.unsigned_abs() != 0
            || s.first + s.y.width > s.w.codes.rows
            || ld % REGISTER_SPAN != 0
            || ld < k
            || s.y.rows != m.unsigned_abs()
            || !(1..=8).contains(&s.split)
            || (s.split > 1 && partials.is_none())
        {
            return Err(refuse(OP, "a segment the register tile does not walk"));
        }
        groups += (cols / REGISTER_COLUMNS).unsigned_abs() * s.split.unsigned_abs();
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
    args.extend([
        table.arg(),
        match partials {
            Some(partials) => partials.arg_mut(),
            None => ctx.absent()?,
        },
        k.arg(),
        m.arg(),
    ]);
    let at = |i: usize, of: &dyn Fn(&Segment) -> i32| segments.get(i).map_or(0, of);
    for of in [
        &(|s: &Segment| s.y.width.cast_signed()) as &dyn Fn(&Segment) -> i32,
        &|s: &Segment| s.split,
        &|s: &Segment| s.w.ld.cast_signed(),
        &|s: &Segment| s.first.cast_signed(),
    ] {
        for i in 0..3 {
            args.push(at(i, of).arg());
        }
    }
    let count = segments.len();
    let entry = symbol(&format!("affine_qmm_sg_segments_s{count}"));
    let stamp = symbol(&format!("PIE_SG_SEGMENTS(\"{entry}\",{count})"));
    ctx.fire(
        Fire::at(REGISTER_FILE, entry)
            .stamp(stamp)
            .apply(Grid::of([128 * groups, 1, 1], [128, 1, 1])),
        &args,
    )
}

/// The `Qmm::Register` rung: whether it fired.
pub(super) fn fire(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    (act, w, biases, y): (Tensor, Bank, Tensor, Tensor),
    scratch: &Scratch<'_>,
    contraction: u32,
    (m, n, k, cols, ld): (i32, i32, i32, i32, i32),
) -> Result<bool, Error> {
    if cols % REGISTER_COLUMNS == 0 && m <= 8 {
        // Without room for the partials it runs whole.
        let split = register_split(cols, k, tuning.cores());
        let partial = (split > 1)
            .then(|| (scratch.partials)(m.unsigned_abs() * split.unsigned_abs(), n.unsigned_abs()))
            .flatten();
        let split = if partial.is_some() { split } else { 1 };
        // The rows' inputs as the lanes take them (8 x k bf16), their sums (k / 4
        // f32) and, split, one arrival counter per 64 columns, in 2k-byte rows.
        let counters = if split > 1 {
            cols / REGISTER_COLUMNS
        } else {
            0
        };
        let bytes = 16 * k + k + 4 * counters;
        let table_rows = bytes.unsigned_abs().div_ceil(2 * k.unsigned_abs());
        // A table another matmul of this activation laid is read as it stands.
        let table = match scratch.table {
            Some(kept) => kept(table_rows, contraction, counters.unsigned_abs()),
            None => (scratch.precast)(table_rows, contraction).map(|t| (t, false)),
        };
        let Some((table, laid)) = table else {
            return Ok(false);
        };
        if !laid {
            ctx.fire(
                Fire::at(REGISTER_FILE, "qmm_sg_table")
                    .apply(Grid::of([32, (k / 64).unsigned_abs(), 1], [32, 4, 1])),
                &[act.arg(), table.arg_mut(), k.arg(), m.arg(), counters.arg()],
            )?;
        }
        let entry = symbol(&format!("affine_qmm_sg_k{split}"));
        let stamp = symbol(&format!("PIE_SG_POINT(\"{entry}\",{split})"));
        return ctx
            .fire(
                Fire::at(REGISTER_FILE, entry).stamp(stamp).apply(Grid::of(
                    [
                        128 * (cols / REGISTER_COLUMNS).unsigned_abs(),
                        split.unsigned_abs(),
                        1,
                    ],
                    [128, 1, 1],
                )),
                &[
                    w.codes.arg(),
                    w.scales.arg(),
                    biases.arg(),
                    table.arg(),
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
            )
            .map(|()| true);
    }
    Ok(false)
}
