use crate::error::Error;

use crate::jit::{Ctx, count, dtype_dispatch, stated};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use super::dense;
#[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
    {
        dense::act_x_wt_f32(ctx, OP, act.ptr, w.ptr, y.ptr, m, n, k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = m;
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

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, None, y)
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

    #[cfg(feature = "cuda")]
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
            Some(bias) => crate::elemwise::pointwise::add_bias(ctx, bias, y),
            None => Ok(()),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, w, bias, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}
