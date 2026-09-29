use dtype::Dtype;

use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::tensor::Tensor;

const FLOATS: &[Dtype] = &[Dtype::Bf16, Dtype::F16, Dtype::F32];

/// `x *= σ(gate)`, element for element. Reference: kernels-wgpu
/// `norm/gate.wgsl`.
pub fn sigmoid_mul(ctx: &Ctx<'_>, gate: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.gate_sigmoid_mul";
    expect(OP, x, FLOATS)?;
    if gate.rows != x.rows || gate.width != x.width {
        return Err(refuse(OP, "the gate plane rides the rectangle it gates"));
    }
    ctx.emit(&mut |cx| {
        let g = cx.read_f32(gate)?;
        let s = cx.sigmoid(g);
        let v = cx.read_f32(x)?;
        let v = cx.mul(v, s)?;
        cx.write(x, v)
    })
}

/// `x[n, h·d + i] *= scale · σ(gate[n, h])`: one logit per head. Reference:
/// kernels-cuda / kernels-metal `gate_sigmoid_mul_heads` (wgpu refuses it).
/// The gate plane may hold more rows than `x`; the first `x.rows` are read.
pub fn sigmoid_mul_heads(
    ctx: &Ctx<'_>,
    gate: Tensor,
    head_dim: u32,
    scale: f32,
    x: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gate_sigmoid_mul_heads";
    expect(OP, x, FLOATS)?;
    if head_dim == 0 || x.width == 0 || !x.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("the {}-wide row is not a whole number of {head_dim}-wide heads", x.width),
        ));
    }
    let heads = x.width / head_dim;
    if gate.width != heads || gate.rows < x.rows {
        return Err(refuse(
            OP,
            format!(
                "the gate plane is {} x {}, and this gate reads one logit per head ({heads}) for each of {} rows",
                gate.rows, gate.width, x.rows
            ),
        ));
    }
    let (rows, hs, d) = (i64::from(x.rows), i64::from(heads), i64::from(head_dim));
    ctx.emit(&mut |cx| {
        let g = cx.read_f32(gate)?;
        let g = cx.slice_axis(g, 0, 0, rows)?;
        let s = cx.sigmoid(g);
        let s = cx.scale(s, f64::from(scale))?;
        let s = cx.broadcast(s, &[rows, hs, d], &[0, 1])?;
        let s = cx.reshape(s, &[rows, hs * d])?;
        let v = cx.read_f32(x)?;
        let v = cx.mul(v, s)?;
        cx.write(x, v)
    })
}
