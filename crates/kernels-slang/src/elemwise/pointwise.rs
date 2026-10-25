//! Elementwise kernels with no algorithm of their own (`elemwise.binary`,
//! `residual_add`, `add_bias`, `mul_scalar`, `scale`, `silu`, `silu_scaled`,
//! `tanh`, `sqrt`, `cast`, `clamp`, `clamp_learned`, `gate_sigmoid_mul`) from
//! `kernels/elemwise/pointwise.slang`, a pair of elements a lane. A
//! whole-tensor kernel takes an odd count; an in-place one binds its tensor
//! once, writable, as WebGPU asks.

use std::cmp::Ordering;

use dtype::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, elementwise_rows, refuse, stated,
};
use crate::error::Error;
use crate::tensor::Tensor;

const POINTWISE: &str = "elemwise/pointwise.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// The `x.rows * x.width` elements of `x`, and the grid that walks them a
/// pair a lane.
fn flat(op: &'static str, x: Tensor) -> Result<(u32, Grid), Error> {
    let [n, _, _] = elementwise(op, x.width, x.rows)?;
    Ok((n, Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Div,
    /// `x - y·floor(x / y)`: floored, as Python's `%`.
    Mod,
}

/// `z = x + y`.
pub fn add(ctx: &Ctx<'_>, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    binary(ctx, BinaryOp::Add, x, y, z)
}

/// `z = x op y`, the planes the same shape and element.
pub fn binary(ctx: &Ctx<'_>, op: BinaryOp, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.binary";
    if x.dtype != z.dtype || y.dtype != z.dtype {
        return Err(refuse(
            OP,
            format!("{:?} {op:?} {:?} into {:?}", x.dtype, y.dtype, z.dtype),
        ));
    }
    let entry = match op {
        BinaryOp::Add => dtype_dispatch!(OP, z.dtype, {
            Bf16 => "elemwise_pointwise_add_bf16",
            F32 => "elemwise_pointwise_add_f32",
        }),
        BinaryOp::Mul => dtype_dispatch!(OP, z.dtype, {
            Bf16 => "elemwise_pointwise_mul_bf16",
            F32 => "elemwise_pointwise_mul_f32",
        }),
        BinaryOp::Div => dtype_dispatch!(OP, z.dtype, {
            Bf16 => "elemwise_pointwise_div_bf16",
            F32 => "elemwise_pointwise_div_f32",
        }),
        BinaryOp::Mod => dtype_dispatch!(OP, z.dtype, {
            Bf16 => "elemwise_pointwise_mod_bf16",
            F32 => "elemwise_pointwise_mod_f32",
        }),
    };
    let (n, grid) = flat(OP, z)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg(), y.arg(), z.arg_mut(), n.arg()],
    )
}

/// `y += x`.
pub fn residual_add(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.residual_add";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "elemwise_pointwise_residual_add_bf16",
        F32 => "elemwise_pointwise_residual_add_f32",
    });
    let (n, grid) = flat(OP, y)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg(), y.arg_mut(), n.arg()],
    )
}

/// `out[r, c] += bias[c]`; the bias rides the row's element, or is bf16
/// under f32 rows.
pub fn add_bias(ctx: &Ctx<'_>, bias: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.add_bias";
    let entry = match (out.dtype, bias.dtype) {
        (Dtype::Bf16, Dtype::Bf16) => "elemwise_pointwise_add_bias_bf16",
        (Dtype::F32, Dtype::F32) => "elemwise_pointwise_add_bias_f32",
        (Dtype::F32, Dtype::Bf16) => "elemwise_pointwise_add_bias_f32_bf16",
        (row, bank) => return Err(refuse(OP, format!("a {bank:?} bias on a {row:?} row"))),
    };
    let [width, rows, _] = elementwise_rows(OP, out.width, out.rows)?;
    if out.dtype == Dtype::Bf16 && !width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {width}-wide bf16 row is odd; rows land two values a word"),
        ));
    }
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(Grid::of([width.div_ceil(2), rows, 1], [GROUP, 1, 1])),
        &[out.arg_mut(), bias.arg(), stated(OP, width)?.arg()],
    )
}

/// `x *= s`, `s` rounded as `x` holds it.
pub fn mul_scalar(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.act";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_pointwise_scale_bf16",
        F32 => "elemwise_pointwise_scale_f32",
    });
    let (n, grid) = flat(OP, x)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg_mut(), s.arg(), n.arg()],
    )
}

/// `x *= s[0]`.
pub fn scale(ctx: &Ctx<'_>, s: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.act";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_pointwise_scale_read_bf16" });
    let (n, grid) = flat(OP, x)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg_mut(), s.arg(), n.arg()],
    )
}

/// `x = silu(x * s)`.
pub fn silu_scaled(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    silu_at(ctx, "elemwise.act", s, x)
}

/// `x = silu(x)`.
pub fn silu(ctx: &Ctx<'_>, x: Tensor) -> Result<(), Error> {
    silu_at(ctx, "elemwise.act", 1.0, x)
}

fn silu_at(ctx: &Ctx<'_>, op: &'static str, s: f32, x: Tensor) -> Result<(), Error> {
    let entry = dtype_dispatch!(op, x.dtype, {
        Bf16 => "elemwise_pointwise_silu_scaled_bf16",
        F32 => "elemwise_pointwise_silu_scaled_f32",
    });
    let (n, grid) = flat(op, x)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg_mut(), s.arg(), n.arg()],
    )
}

pub fn sqrt(ctx: &Ctx<'_>, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.act";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_pointwise_sqrt_bf16",
        F32 => "elemwise_pointwise_sqrt_f32",
    });
    let (n, grid) = flat(OP, x)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg_mut(), n.arg()],
    )
}

/// `x = tanh(x)`.
pub fn tanh(ctx: &Ctx<'_>, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.act";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_pointwise_tanh_bf16",
        F32 => "elemwise_pointwise_tanh_f32",
    });
    let (n, grid) = flat(OP, x)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg_mut(), n.arg()],
    )
}

/// f32 values rounded into bf16 ones.
pub fn cast_f32_to_bf16(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.cast";
    if x.dtype != Dtype::F32 || o.dtype != Dtype::Bf16 {
        return Err(refuse(OP, format!("{:?} into {:?}", x.dtype, o.dtype)));
    }
    let (n, grid) = flat(OP, o)?;
    ctx.fire(
        Fire::at(POINTWISE, "elemwise_pointwise_cast_f32_to_bf16").apply(grid),
        &[x.arg(), o.arg_mut(), n.arg()],
    )
}

/// `x = min(max(x, lo), hi)`, the bounds rounded as `x` holds them.
pub fn clamp(ctx: &Ctx<'_>, lo: f32, hi: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.clamp";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_pointwise_clamp_bf16" });
    if !matches!(lo.partial_cmp(&hi), Some(Ordering::Less | Ordering::Equal)) {
        return Err(refuse(
            OP,
            format!(
                "the bounds {lo} and {hi} cross, and a clamp between them is the constant {hi}"
            ),
        ));
    }
    let (n, grid) = flat(OP, x)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg_mut(), lo.arg(), hi.arg(), n.arg()],
    )
}

/// `x` clamped between one-element bound tensors of its element.
pub fn clamp_learned(ctx: &Ctx<'_>, lo: Tensor, hi: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.clamp";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_pointwise_clamp_learned_bf16" });
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
    let (n, grid) = flat(OP, x)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg_mut(), lo.arg(), hi.arg(), n.arg()],
    )
}

/// `x *= sigmoid(gate)`, the gate a tensor of `x`'s shape.
pub fn sigmoid_mul(ctx: &Ctx<'_>, gate: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.gate_sigmoid_mul";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_pointwise_sigmoid_mul_bf16" });
    if (gate.rows, gate.width, gate.dtype) != (x.rows, x.width, x.dtype) {
        return Err(refuse(
            OP,
            format!(
                "a {} x {} {:?} gate on a {} x {} {:?} rectangle; the gate rides the \
                 rectangle it gates",
                gate.rows, gate.width, gate.dtype, x.rows, x.width, x.dtype
            ),
        ));
    }
    let (n, grid) = flat(OP, x)?;
    ctx.fire(
        Fire::at(POINTWISE, entry).apply(grid),
        &[x.arg_mut(), gate.arg(), n.arg()],
    )
}
