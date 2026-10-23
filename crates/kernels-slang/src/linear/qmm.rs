//! Affine-quantized matmuls (`linear.matmul`, `linear.lm_head` over a bank):
//! the engine's `kernel.qmm` ladder, its rungs in order, and the first that
//! fits the shape runs.

use crate::encode::{Ctx, dtype_dispatch, refuse};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

/// A `kernel.qmm` rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qmm {
    /// The vector kernel at `vecs` activation rows a workgroup; it fits
    /// every shape.
    Qmv { vecs: u32 },
    /// The one-row kernel (SPIR-V's subgroup tier): `lanes` subgroup lanes a
    /// column, `rows` columns a lane.
    OneRow { lanes: u32, rows: u32 },
}

/// A matmul's operands and extents, checked once for every rung.
pub(crate) struct Bounds {
    pub act: Tensor,
    pub w: Bank,
    pub biases: Tensor,
    pub y: Tensor,
    /// Activation rows, output columns, contraction.
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub group: u32,
    pub bits: u32,
}

fn bounds(op: &'static str, act: Tensor, w: Bank, y: Tensor) -> Result<Bounds, Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let Some(biases) = w.biases else {
        return Err(refuse(
            op,
            format!(
                "the weight is a symmetric {}-bit bank in groups of {}, and no point is \
                 instantiated for one",
                w.bits, w.group
            ),
        ));
    };
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
    if !matches!(w.group, 32 | 64 | 128) || !matches!(w.bits, 2 | 4 | 8) {
        return Err(refuse(
            op,
            format!(
                "no point is instantiated at group size {} and bit width {}",
                w.group, w.bits
            ),
        ));
    }
    if !act.width.is_multiple_of(w.group) {
        return Err(refuse(
            op,
            format!(
                "the contraction is {}, not a whole number of {}-code groups: every point \
                 indexes its scales as `k / group`",
                act.width, w.group
            ),
        ));
    }
    Ok(Bounds {
        act,
        w,
        biases,
        y,
        m: y.rows,
        n: y.width,
        k: act.width,
        group: w.group,
        bits: w.bits,
    })
}

pub fn matmul(ctx: &Ctx<'_>, ladder: &[Qmm], act: Tensor, w: Bank, y: Tensor) -> Result<(), Error> {
    fire(ctx, "linear.matmul", ladder, act, w, y)
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    ladder: &[Qmm],
    act: Tensor,
    w: Bank,
    y: Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", ladder, act, w, y)
}

fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    ladder: &[Qmm],
    act: Tensor,
    w: Bank,
    y: Tensor,
) -> Result<(), Error> {
    let b = bounds(op, act, w, y)?;
    if b.m == 0 {
        return Ok(());
    }
    match ladder.first() {
        Some(Qmm::Qmv { vecs }) => crate::linear::qmv::fire(ctx, op, &b, *vecs),
        Some(Qmm::OneRow { lanes, rows }) => {
            crate::linear::qmv::fire_row(ctx, op, &b, *lanes, *rows)
        }
        None => Err(refuse(op, "the `kernel.qmm` ladder is empty")),
    }
}
