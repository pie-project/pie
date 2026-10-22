use crate::error::Error;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, head_grid, head_group, nonzero, refuse,
    stated,
};
use crate::tensor::Tensor;

pub fn sigmoid_mul_heads(
    ctx: &Ctx<'_>,
    gate: Tensor,
    head_dim: u32,
    scale: f32,
    x: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.gate_sigmoid_mul_heads";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "gate_sigmoid_mul_heads_bfloat16" });
    let head_dim = nonzero(OP, "the head width", head_dim)?;
    if !x.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "a {}-wide row is not a whole number of {head_dim}-wide heads",
                x.width
            ),
        ));
    }
    let heads = x.width / head_dim;
    if gate.width != heads || gate.rows < x.rows {
        return Err(refuse(
            OP,
            format!(
                "the gate plane is {} x {} and this fold reads one logit per head of \
                 {heads} for each of {} rows",
                gate.rows, gate.width, x.rows
            ),
        ));
    }
    ctx.fire(
        Fire::at("elemwise/gate.metal", entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [256.min(x.width), 1, 1],
        )),
        &[
            x.arg_mut(),
            gate.arg(),
            heads.arg(),
            head_dim.arg(),
            scale.arg(),
        ],
    )
}

pub fn sigmoid_mul(ctx: &Ctx<'_>, gate: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.gate_sigmoid_mul";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "gate_bfloat16" });
    debug_assert!(
        gate.rows == x.rows && gate.width == x.width && gate.dtype == x.dtype,
        "the gate plane rides the rectangle it gates"
    );
    ctx.fire(
        Fire::at("elemwise/gate.metal", entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [256, 1, 1],
        )),
        &[x.arg_mut(), gate.arg(), stated(OP, x.width)?.arg()],
    )
}

/// [`sigmoid_mul`] over at most eight rows that also lays the register tile's
/// input table of the gated rows in `table`, with `counters` arrival counters
/// zeroed: the table `linear::quant`'s register matmul lays over `x`, bit for
/// bit, so the out projection runs no table pass.
pub fn sigmoid_mul_table(
    ctx: &Ctx<'_>,
    gate: Tensor,
    x: Tensor,
    table: Tensor,
    counters: u32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.gate_sigmoid_mul";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    debug_assert!(
        gate.rows == x.rows && gate.width == x.width && gate.dtype == x.dtype,
        "the gate plane rides the rectangle it gates"
    );
    if x.rows > 8 || !x.width.is_multiple_of(64) || x.width > 8192 {
        return Err(refuse(
            OP,
            format!(
                "a register table holds at most 8 rows of whole 64-input groups, and this \
                 one stages at most 8192, not {} x {}",
                x.rows, x.width
            ),
        ));
    }
    ctx.fire(
        Fire::at("elemwise/gate.metal", "gate_table_bfloat16").apply(Grid::of(
            [x.width.min(1024), 8, 1],
            [x.width.min(1024), 1, 1],
        )),
        &[
            x.arg_mut(),
            gate.arg(),
            stated(OP, x.width)?.arg(),
            table.arg_mut(),
            x.rows.arg(),
            counters.arg(),
        ],
    )
}

pub fn split_q_gate(
    ctx: &Ctx<'_>,
    packed: Tensor,
    head_dim: u32,
    q: Tensor,
    gate: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_q_gate";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "q_gate_split_bfloat16" });
    nonzero(OP, "the head width this cut walks", head_dim)?;
    if q.width == 0 || !q.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide query half does not divide by the stated head width {head_dim}",
                q.width
            ),
        ));
    }
    let lanes = head_grid(OP, head_dim, q.width / head_dim, packed.rows)?;
    ctx.fire(
        Fire::at("elemwise/gate.metal", entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            packed.arg(),
            q.arg_mut(),
            gate.arg_mut(),
            stated(OP, head_dim)?.arg(),
            stated(OP, packed.width)?.arg(),
            stated(OP, q.width)?.arg(),
        ],
    )
}
