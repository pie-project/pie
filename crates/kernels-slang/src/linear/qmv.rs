//! The vector points of an affine-quantized matmul: `kernel.qmm`'s `qmv`
//! rung from `kernels/linear/qmv.slang` (four 32-lane strips a workgroup,
//! two output columns a strip, `vecs` activation rows a workgroup) and its
//! `one_row` rung from `kernels/linear/qmv_row.slang` (one activation row,
//! `lanes` subgroup lanes a column folding by shuffles, SPIR-V's subgroup
//! tier).

use crate::encode::{Arg, Ctx, Fire, Target, Tier, instantiated, refuse, stated};
use crate::error::Error;
use crate::linear::qmm::Bounds;

const QMV: &str = "linear/qmv.slang";

const QMV_ROW: &str = "linear/qmv_row.slang";

/// Output columns a workgroup lands: four strips of two.
const COLUMNS: u32 = 8;

/// `y = dequant(w) x` over the bank `bounds` checked, `vecs` activation rows
/// (1, 2, 4 or 8) a workgroup.
pub(crate) fn fire(ctx: &Ctx<'_>, op: &'static str, b: &Bounds, vecs: u32) -> Result<(), Error> {
    if !matches!(vecs, 1 | 2 | 4 | 8) {
        return Err(refuse(
            op,
            format!("no vector point reads {vecs} rows a workgroup"),
        ));
    }
    let chunk = if b.bits == 8 { 16 } else { 32 };
    if !b.k.is_multiple_of(chunk) {
        return Err(refuse(
            op,
            format!(
                "the vector point walks the contraction in {chunk}-code chunks; this batch is \
                 {} x {}",
                b.m, b.k
            ),
        ));
    }
    if ctx.target() == Target::Wgsl && !b.n.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "the projection lands {} columns, odd: on WGSL every output pair is one bf16 word",
                b.n
            ),
        ));
    }
    let entry = instantiated(
        op,
        format!(
            "linear_qmv_bf16_gs_{}_b_{}{}",
            b.group,
            b.bits,
            if vecs == 4 {
                String::new()
            } else {
                format!("_v{vecs}")
            }
        ),
    )?;
    ctx.fire(
        Fire::at(QMV, entry)
            .groups([b.n.div_ceil(COLUMNS), b.m.div_ceil(vecs), 1])
            .group([32, 4, 1]),
        &[
            b.w.codes.arg(),
            b.w.scales.arg(),
            b.biases.arg(),
            b.act.arg(),
            b.y.arg_mut(),
            stated(op, b.k)?.arg(),
            stated(op, b.n)?.arg(),
            stated(op, b.m)?.arg(),
        ],
    )
}

/// Whether the one-row point serves a bank: it loads 128 bits of codes at
/// once, which must stay inside one scale group.
#[must_use]
pub fn row_fits(group: u32, bits: u32) -> bool {
    matches!(bits, 4 | 8) && 128 / bits <= group
}

/// `y = dequant(w) x` for one activation row: `lanes` (8, 16 or 32) subgroup
/// lanes a column, `rows` (1 or 2) columns a lane.
pub(crate) fn fire_row(
    ctx: &Ctx<'_>,
    op: &'static str,
    b: &Bounds,
    lanes: u32,
    rows: u32,
) -> Result<(), Error> {
    if b.m != 1 {
        return Err(refuse(
            op,
            format!(
                "the one-row point reads one activation row; this batch has {}",
                b.m
            ),
        ));
    }
    if !row_fits(b.group, b.bits) {
        return Err(refuse(
            op,
            format!(
                "the one-row point loads 128 bits of codes inside one scale group; {} bits in \
                 groups of {} straddle one",
                b.bits, b.group
            ),
        ));
    }
    if !matches!((lanes, rows), (8 | 16 | 32, 1 | 2)) {
        return Err(refuse(
            op,
            format!("no one-row point runs {lanes} lanes a column and {rows} columns a lane"),
        ));
    }
    let entry = instantiated(
        op,
        if (lanes, rows) == (32, 1) {
            format!("linear_qmv_row_bf16_gs_{}_b_{}", b.group, b.bits)
        } else {
            format!(
                "linear_qmv_row_bf16_gs_{}_b_{}_l{lanes}_r{rows}",
                b.group, b.bits
            )
        },
    )?;
    // 256 invocations land 256 / lanes columns, `rows` each.
    ctx.fire(
        Fire::at(QMV_ROW, entry)
            .tier(Tier::Subgroup)
            .groups([b.n.div_ceil(256 / lanes * rows), 1, 1])
            .group([64, 4, 1]),
        &[
            b.w.codes.arg(),
            b.w.scales.arg(),
            b.biases.arg(),
            b.act.arg(),
            b.y.arg_mut(),
            stated(op, b.k)?.arg(),
            stated(op, b.n)?.arg(),
            stated(op, b.m)?.arg(),
        ],
    )
}
