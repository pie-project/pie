//! The ternary PTQ1_0 bank's decode-in-dot matrix-vector kernel
//! (`linear/qmv_ptq1_0.metal`).

use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

use super::{QMV_GROUP, qmv_grid};

/// The PTQ1_0 ternary decode-in-dot kernel (`Dtype::Ptq1_0`, `g128_t3_f16_n`).
const PTQ1_0_FILE: &str = "linear/qmv_ptq1_0.metal";

/// Production qmv entrypoint: bf16 activation in, bf16 out.
const PTQ1_0_QMV: &str = "ptq1_0_qmv_bfloat16";

/// Weights per PTQ1_0 block; the scale is INLINE per block (single plane).
const PTQ1_0_BLOCK: u32 = 128;

/// Serve a PTQ1_0 (ternary, `g128_t3_f16_n`) projection with the decode-in-dot
/// kernel. The bank is single-plane: `w.codes` carries the 28-byte blocks with
/// their inline fp16 scale, and `w.scales`/`w.biases` are unused (the loader
/// leaves them aliased at the codes plane and this path never reads them). One
/// kernel walks every row count `m`; decode is small and matrix-vector shaped,
/// which is the path pie's `qwen_3 d27b` decode takes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Bank,
    y: Tensor,
    rows: u32,
    columns: u32,
    contraction: u32,
) -> Result<(), Error> {
    if !contraction.is_multiple_of(PTQ1_0_BLOCK) {
        return Err(refuse(
            op,
            format!(
                "the contraction is {contraction}, not a whole number of {PTQ1_0_BLOCK}-weight \
                 ternary blocks: a PTQ1_0 row is `ceil(K/{PTQ1_0_BLOCK}) * 28` bytes and the \
                 kernel reads one 28-byte block per {PTQ1_0_BLOCK} contracted weights"
            ),
        ));
    }
    let (m, n) = (stated(op, rows)?, stated(op, columns)?);
    let k = stated(op, contraction)?;
    ctx.fire(
        Fire::at(PTQ1_0_FILE, PTQ1_0_QMV).apply(Grid::of(qmv_grid(op, m, n)?, QMV_GROUP)),
        &[w.codes.arg(), act.arg(), y.arg_mut(), k.arg(), n.arg()],
    )
}
