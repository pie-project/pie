use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "elemwise/pointwise.slang";

const GROUP: [u32; 3] = [256, 1, 1];

pub fn add(ctx: &Ctx<'_>, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add";
    if x.dtype != z.dtype || y.dtype != z.dtype {
        return Err(refuse(
            OP,
            format!("{:?} + {:?} into {:?}", x.dtype, y.dtype, z.dtype),
        ));
    }
    let entry =
        dtype_dispatch!(OP, z.dtype, { Bf16 => "residual_add_bf16", F32 => "residual_add_f32" });
    let lanes = elementwise(OP, z.width, z.rows)?;
    ctx.fire(
        Fire::at("norm/residual_add.slang", entry).apply(Grid::of(lanes, GROUP)),
        &[x.arg(), y.arg(), z.arg_mut(), lanes[0].arg()],
    )
}

pub fn silu(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.silu";
    let entry =
        dtype_dispatch!(OP, o.dtype, { Bf16 => "silu_scaled_bf16", F32 => "silu_scaled_f32" });
    let lanes = elementwise(OP, o.width, o.rows)?;
    ctx.fire(
        Fire::at("norm/layer_scalar.slang", entry).apply(Grid::of(lanes, GROUP)),
        &[x.arg(), o.arg_mut(), 1.0f32.arg(), lanes[0].arg()],
    )
}

pub fn tanh(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.tanh";
    let entry = dtype_dispatch!(OP, o.dtype, { Bf16 => "tanh_bf16", F32 => "tanh_f32" });
    let lanes = elementwise(OP, o.width, o.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, GROUP)),
        &[x.arg(), o.arg_mut(), lanes[0].arg()],
    )
}

pub fn cast_f32_to_bf16(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.cast";
    if x.dtype != Dtype::F32 || o.dtype != Dtype::Bf16 {
        return Err(refuse(OP, format!("{:?} into {:?}", x.dtype, o.dtype)));
    }
    let lanes = elementwise(OP, o.width, o.rows)?;
    ctx.fire(
        Fire::at(FILE, "cast_f32_to_bf16").apply(Grid::of(lanes, GROUP)),
        &[x.arg(), o.arg_mut(), lanes[0].arg()],
    )
}
