use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "probe/nan_check.metal";

pub fn nan_check(
    ctx: &Ctx<'_>,
    x: Tensor,
    flags: Tensor,
    slot: u32,
    limit: f32,
) -> Result<(), Error> {
    const OP: &str = "probe.nan_check";
    let entry = dtype_dispatch!(OP, x.dtype, {
        F32 => "nan_check_float32",
        Bf16 => "nan_check_bfloat16",
        F16 => "nan_check_float16",
    });
    if flags.dtype != Dtype::U32 {
        return Err(refuse(
            OP,
            format!("the flag plane is {:?}, not u32", flags.dtype),
        ));
    }
    if slot >= flags.rows.max(flags.rows.saturating_mul(flags.width)) {
        return Err(refuse(
            OP,
            format!("slot {slot} is past a {} word flag plane", flags.rows),
        ));
    }
    let n = x.rows.saturating_mul(x.width);
    if n == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(elementwise(OP, n, 1)?, [256, 1, 1])),
        &[x.arg(), flags.arg_mut(), slot.arg(), n.arg(), limit.arg()],
    )
}
