//! `elemwise.clamp` and `elemwise.clamp_learned` from
//! `kernels/elemwise/clamp.slang`: a tensor clamped in place between stated
//! bounds or bounds read from one-element tensors.

use std::cmp::Ordering;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const CLAMP: &str = "elemwise/clamp.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

pub fn clamp(ctx: &Ctx<'_>, lo: f32, hi: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.clamp";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_clamp_bf16" });
    if !matches!(lo.partial_cmp(&hi), Some(Ordering::Less | Ordering::Equal)) {
        return Err(refuse(
            OP,
            format!(
                "the bounds {lo} and {hi} cross, and a clamp between them is the constant {hi}"
            ),
        ));
    }
    let [n, _, _] = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(CLAMP, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg_mut(), lo.arg(), hi.arg(), n.arg()],
    )
}

pub fn clamp_learned(ctx: &Ctx<'_>, lo: Tensor, hi: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.clamp_learned";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_clamp_learned_bf16" });
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
    let [n, _, _] = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(CLAMP, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg_mut(), lo.arg(), hi.arg(), n.arg()],
    )
}
