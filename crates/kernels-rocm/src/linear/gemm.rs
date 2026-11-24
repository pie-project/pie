use crate::error::Error;

use crate::jit::{Ctx, count, dtype_dispatch, stated};
use crate::tensor::Tensor;

#[cfg(feature = "rocm")]
use super::dense;
use dtype::Dtype;

/// The workgroups (blocks) of the skinny kernel, ROCm's few-rows kernel,
/// walking a plain bf16 product of `m x n x k` (`m` up to its 8 rows) one
/// compute unit of the current device holds at once, at its launch's
/// threads and LDS: HIP's occupancy calculator
/// (`hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`) over the kernel's
/// registers and LDS against the compute unit's.
pub fn few_rows_resident_blocks(ctx: &Ctx, m: u32, n: u32, k: u32) -> Result<u32, Error> {
    const OP: &str = "linear.few_rows_occupancy";
    let (m, n, k) = (
        count(OP, "the rows", m)?,
        count(OP, "the columns", n)?,
        count(OP, "the contraction", k)?,
    );
    #[cfg(feature = "rocm")]
    {
        super::gemv::skinny_resident_blocks(ctx, m, n, k)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, m, n, k);
        Err(crate::jit::runtimeless(OP))
    }
}

/// The waves hipBLASLt's first heuristic algorithm for a dense bf16 product
/// of `m x n x k` fills on the current device, as its library states the
/// algorithm's kernel (`hip::lt::waves`). `None` where it offers no
/// algorithm or states no waves; where its solution could not be read, the
/// reason is kept for `hip::lt_kernel::pricing`.
#[cfg(feature = "rocm")]
#[must_use]
pub fn waves(m: u32, n: u32, k: u32) -> Option<f64> {
    crate::hip::lt::waves(crate::hip::lt::Shape {
        m,
        n,
        k,
        bias: false,
        groups: 1,
        f32_out: false,
    })
    .unwrap_or_else(|why| {
        crate::hip::lt_kernel::unread(why);
        None
    })
}

/// The most rows ROCm's few-rows kernel (the skinny kernel) walks.
pub const FEW_ROWS: u32 = 8;

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
    act_x_wt(ctx, "fused.matmul_bias", act, w, Some(bias), y)
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
    const OP: &str = "fused.matmul_glu";
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
    act_x_wt(ctx, "linear.matmul", act, w, None, y)
}

/// [`lm_head`] over a body's padded readouts, `live` a device word holding
/// how many of act's rows are live: past `kernel.gemm.live_from` padded rows
/// only those are multiplied (the rest of y keeps what it held).
pub fn lm_head_live(
    ctx: &Ctx,
    act: Tensor,
    w: Tensor,
    live: u64,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul";
    #[cfg(feature = "rocm")]
    {
        let from = ctx.tuning().gemm_live_from();
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
            && dense::act_x_wt_live(
                ctx, OP, act.ptr, w.ptr, live, y.ptr, y.rows, y.width, act.width,
            )?
        {
            return Ok(());
        }
    }
    let _ = live;
    act_x_wt(ctx, OP, act, w, None, y)
}

/// `z = rmsnorm(x + y) · wᵀ` in one launch where that pays
/// (`ResidualAddRmsnormMatmul`): `x + y` lands in `y_out` and the normed
/// rows in `out`, each block norming its own copy where it stages them for
/// the product. `Ok(false)`, landing nothing, where the product would not take
/// the skinny kernel or the blocks' redundant norms cost more than the
/// launch they save; the caller then runs the norm and the product apart.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rmsnorm_matmul(
    ctx: &Ctx,
    x: Tensor,
    y: Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    w: Tensor,
    y_out: &mut Tensor,
    out: &mut Tensor,
    z: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "fused.residual_add_rmsnorm_matmul";
    if [x.dtype, y.dtype, weight.dtype, w.dtype, y_out.dtype, out.dtype, z.dtype]
        .iter()
        .any(|&d| d != Dtype::Bf16)
        || x.rows != y.rows
        || x.rows != z.rows
        || x.width != y.width
        // The norm's weight as a load lays it (k rows of one) or one row.
        || u64::from(weight.rows) * u64::from(weight.width) != u64::from(x.width)
        || w.width != x.width
        || w.rows != z.width
        || z.rows == 0
    {
        return Ok(false);
    }
    let n = count(OP, "the columns this projection lands", z.width)?;
    let k = count(OP, "the contraction this projection walks", x.width)?;
    let m = ctx.opaque_rows(stated(OP, z.rows)?);
    #[cfg(feature = "rocm")]
    {
        dense::act_x_wt_normed(
            ctx,
            w.ptr,
            z.ptr,
            m,
            n,
            k,
            super::gemv::SkinnyNorm {
                x: x.ptr,
                y: y.ptr,
                w: weight.ptr,
                y_out: y_out.ptr,
                out: out.ptr,
                eps,
                plus_one: i32::from(plus_one),
            },
        )
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, plus_one, eps, y_out, out, m, n, k);
        Err(crate::jit::runtimeless(OP))
    }
}

/// `h = swiglu(rmsnorm(x + y) · gate_upᵀ)` in one launch where that pays
/// (`ResidualAddRmsnormMatmulSwigluMatmul`'s first half): `x + y` lands in
/// `y_out` and the normed rows in `out`, each block norming its own copy
/// where it stages them for the packed `[gate | up]` product. `Ok(false)`,
/// landing nothing, where the gated product would not take the skinny
/// kernel or the blocks' redundant norms cost more than the launch they
/// save; the caller then runs the norm and the MLP apart.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rmsnorm_matmul_swiglu(
    ctx: &Ctx,
    x: Tensor,
    y: Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    gate_up: Tensor,
    y_out: &mut Tensor,
    out: &mut Tensor,
    h: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "fused.residual_add_rmsnorm_matmul_swiglu_matmul";
    if [x.dtype, y.dtype, weight.dtype, gate_up.dtype, y_out.dtype, out.dtype, h.dtype]
        .iter()
        .any(|&d| d != Dtype::Bf16)
        || x.rows != y.rows
        || x.rows != h.rows
        || x.width != y.width
        || u64::from(weight.rows) * u64::from(weight.width) != u64::from(x.width)
        || gate_up.width != x.width
        || gate_up.rows != 2 * h.width
        || h.rows == 0
    {
        return Ok(false);
    }
    let i = count(OP, "the intermediate width", h.width)?;
    let k = count(OP, "the contraction this projection walks", x.width)?;
    let m = ctx.opaque_rows(stated(OP, h.rows)?);
    #[cfg(feature = "rocm")]
    {
        dense::act_x_wt_swiglu_normed(
            ctx,
            gate_up.ptr,
            h.ptr,
            m,
            i,
            k,
            super::gemv::SkinnyNorm {
                x: x.ptr,
                y: y.ptr,
                w: weight.ptr,
                y_out: y_out.ptr,
                out: out.ptr,
                eps,
                plus_one: i32::from(plus_one),
            },
        )
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, plus_one, eps, y_out, out, m, i, k);
        Err(crate::jit::runtimeless(OP))
    }
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
            Some(bias) => crate::elemwise::pointwise::add_bias(ctx, bias, y),
            None => Ok(()),
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, w, bias, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}
