use std::cmp::Ordering;

use crate::encode::{Arg, Ctx, Fire, dtype_dispatch, elementwise, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "norm/clip.wgsl";

pub fn clamp(ctx: &Ctx<'_>, lo: f32, hi: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "clamp_bf16" });

    if !matches!(lo.partial_cmp(&hi), Some(Ordering::Less | Ordering::Equal)) {
        return Err(refuse(
            OP,
            format!(
                "the bounds {lo} and {hi} cross, and a clamp between them is the constant {hi}"
            ),
        ));
    }
    let lanes = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(super::norm::flat_words(lanes)),
        &[x.arg_mut(), lo.arg(), hi.arg(), lanes[0].arg()],
    )
}

pub fn clamp_learned(ctx: &Ctx<'_>, lo: Tensor, hi: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp_learned";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "clamp_learned_bf16" });
    for (what, bound) in [("lower", lo), ("upper", hi)] {
        if bound.dtype != x.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is {:?} and the rows it clamps are {:?}; a learned bound \
                     rides the activation's element",
                    bound.dtype, x.dtype
                ),
            ));
        }
        if bound.rows.checked_mul(bound.width) != Some(1) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is a {} x {} plane, and this clamp reads one scalar",
                    bound.rows, bound.width
                ),
            ));
        }
    }
    let lanes = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(super::norm::flat_words(lanes)),
        &[x.arg_mut(), lo.arg(), hi.arg(), lanes[0].arg()],
    )
}
