use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Func, Val};
use crate::tensor::Operand;

const FLOATS: &[Dtype] = &[Dtype::Bf16, Dtype::F16, Dtype::F32];

/// `x *= σ(gate)`, element for element. Reference: kernels-slang
/// `norm/gate.wgsl`.
pub fn sigmoid_mul(f: &mut Func, gate: Operand, x: Operand) -> Result<Val, Error> {
    const OP: &str = "elemwise.gate_sigmoid_mul";
    expect(OP, x, FLOATS)?;
    if gate.rows != x.rows || gate.width != x.width {
        return Err(refuse(OP, "the gate plane rides the rectangle it gates"));
    }
    let g = gate.f32(f);
    let s = f.sigmoid(g);
    let v = x.f32(f);
    let v = f.mul(v, s)?;
    Ok(v)
}

/// `x[n, h·d + i] *= scale · σ(gate[n, h])`: one logit per head. Reference:
/// kernels-cuda / kernels-metal `gate_sigmoid_mul_heads` (wgpu refuses it).
/// The gate plane may hold more rows than `x`; the first `x.rows` are read.
pub fn sigmoid_mul_heads(
    f: &mut Func,
    gate: Operand,
    head_dim: u32,
    scale: f32,
    x: Operand,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.gate_sigmoid_mul_heads";
    expect(OP, x, FLOATS)?;
    if head_dim == 0 || x.width == 0 || !x.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide row is not a whole number of {head_dim}-wide heads",
                x.width
            ),
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
    let g = gate.f32(f);
    let g = f.slice_axis(g, 0, 0, rows)?;
    let s = f.sigmoid(g);
    let s = f.scale(s, f64::from(scale))?;
    let s = f.broadcast(s, &[rows, hs, d], &[0, 1])?;
    let s = f.reshape(s, &[rows, hs * d])?;
    let v = x.f32(f);
    let v = f.mul(v, s)?;
    Ok(v)
}
