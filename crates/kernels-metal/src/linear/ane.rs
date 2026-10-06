//! The GPU's side of an MLP split with the Neural Engine: stage the input
//! as fp16 for it, then add its half of the output back in. Both kernels
//! carry the shared-event value the encoder fences on beside them.

use crate::encode::{Arg, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::Tensor;

pub const FILE: &str = "linear/ane.metal";
pub const STAGE: &str = "ane_stage_bfloat16";
pub const JOIN: &str = "ane_join_bfloat16";

const GROUP: u32 = 256;

fn count(x: Tensor) -> u32 {
    x.rows.saturating_mul(x.width)
}

pub fn stage(ctx: &Ctx<'_>, x: Tensor, staged: Tensor, stamp: u32) -> Result<(), Error> {
    let n = count(x);
    ctx.fire(
        Fire::at(FILE, STAGE).apply(Grid::of([n, 1, 1], [GROUP, 1, 1])),
        &[x.arg(), staged.arg_mut(), n.arg(), stamp.arg()],
    )
}

pub fn join(ctx: &Ctx<'_>, y: Tensor, other: Tensor, stamp: u32) -> Result<(), Error> {
    let n = count(y);
    ctx.fire(
        Fire::at(FILE, JOIN).apply(Grid::of([n, 1, 1], [GROUP, 1, 1])),
        &[y.arg_mut(), other.arg(), n.arg(), stamp.arg()],
    )
}
