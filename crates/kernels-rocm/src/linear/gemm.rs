use crate::error::Error;

use crate::jit::{Ctx, count, dtype_dispatch, stated};
use crate::tensor::Tensor;

#[cfg(feature = "rocm")]
use super::dense;
#[cfg(feature = "rocm")]
use dtype::Dtype;

pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, None, y)
}

/// `act . w^T` over bf16 operands landing f32 (a router's logits).
pub fn matmul_f32(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.matmul_f32";
    dtype_dispatch!(OP, act.dtype, { Bf16 => () });
    dtype_dispatch!(OP, w.dtype, { Bf16 => () });
    dtype_dispatch!(OP, y.dtype, { F32 => () });
    let n = count(OP, "the columns this projection lands", y.width)?;
    let k = count(OP, "the contraction this projection walks", act.width)?;
    if w.rows != y.width || w.width != act.width {
        return Err(crate::jit::refuse(
            OP,
            format!(
                "the weight is {} x {} and this projection contracts {k} into {n}",
                w.rows, w.width
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    let m = ctx.opaque_rows(stated(OP, y.rows)?);
    #[cfg(feature = "rocm")]
    {
        dense::act_x_wt_f32(ctx, OP, act.ptr, w.ptr, y.ptr, m, n, k)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, m, n, k);
        Err(crate::jit::runtimeless(OP))
    }
}

pub fn matmul_bias(
    ctx: &Ctx,
    act: Tensor,
    w: Tensor,
    bias: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul_bias", act, w, Some(bias), y)
}

/// `h = silu(act · gateᵀ) · (act · upᵀ)` over a packed `[gate | up]` bank in
/// one fire, for the few rows a decode carries. `Ok(false)` when the fused
/// fire does not cover the shape, leaving the matmul and the swiglu to the
/// caller.
pub fn matmul_swiglu(
    ctx: &Ctx,
    act: Tensor,
    gate_up: Tensor,
    h: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "linear.matmul_swiglu";
    dtype_dispatch!(OP, act.dtype, { Bf16 => () });
    if gate_up.dtype != dtype::Dtype::Bf16 || gate_up.rows != 2 * h.width || h.rows == 0 {
        return Ok(false);
    }
    let i = count(OP, "the intermediate width", h.width)?;
    let k = count(OP, "the contraction", act.width)?;
    let m = ctx.opaque_rows(stated(OP, h.rows)?);
    #[cfg(feature = "rocm")]
    {
        dense::act_x_wt_swiglu(ctx, act.ptr, gate_up.ptr, h.ptr, m, i, k)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, m, i, k);
        Err(crate::jit::runtimeless(OP))
    }
}

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, None, y)
}

/// [`lm_head`] over a body's padded readouts, `live` a device word holding
/// how many of act's rows are live: past `kernel.gemm.live_from` padded rows
/// only those are multiplied (the rest of y keeps what it held).
pub fn lm_head_live(ctx: &Ctx, act: Tensor, w: Tensor, live: u64, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.lm_head";
    #[cfg(feature = "rocm")]
    {
        let from = ctx.tuning().gemm_live_from;
        if live != 0
            && from > 0
            && y.rows >= from
            && act.dtype == Dtype::Bf16
            && w.dtype == Dtype::Bf16
            && y.dtype == Dtype::Bf16
            && act.rows == y.rows
            && w.rows == y.width
            && w.width == act.width
            && act.width.is_multiple_of(8)
            && dense::act_x_wt_live(ctx, OP, act.ptr, w.ptr, live, y.ptr, y.rows, y.width, act.width)?
        {
            return Ok(());
        }
    }
    let _ = live;
    act_x_wt(ctx, OP, act, w, None, y)
}

pub fn act_x_wt(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    y: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = count(op, "the columns this projection lands", y.width)?;
    let k = count(op, "the contraction this projection walks", act.width)?;
    if y.rows == 0 {
        return Ok(());
    }
    let m = ctx.opaque_rows(stated(op, y.rows)?);

    #[cfg(feature = "rocm")]
    {
        // The gemv folds a bf16 bias into its store; any other tactic leaves it
        // to the add.
        let folded = bias
            .filter(|bias| bias.dtype == Dtype::Bf16)
            .map_or(0, |bias| bias.ptr);
        if dense::act_x_wt(ctx, op, act.ptr, w.ptr, folded, y.ptr, m, n, k)? {
            return Ok(());
        }
        match bias {
            Some(bias) => crate::elemwise::norm::add_bias(ctx, bias, y),
            None => Ok(()),
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, w, bias, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}
