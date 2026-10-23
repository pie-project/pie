//! The vector point of an affine-quantized matmul (`kernel.qmm`'s `qmv`
//! rung) from `kernels/linear/qmv.slang`: four 32-lane strips a workgroup,
//! two output columns a strip, `vecs` activation rows a workgroup.

use crate::encode::{Arg, Ctx, Fire, Target, instantiated, refuse, stated};
use crate::error::Error;
use crate::linear::qmm::Bounds;

const QMV: &str = "linear/qmv.slang";

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
