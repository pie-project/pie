//! `Gemm`: dense projections against a transposed weight. Everything that
//! used to be a choice upstream — gemv vs cuBLAS vs cuBLASLt, algorithm
//! index, unroll depth per architecture — lives below this file's entries
//! (decision #13): a dispatch arm never sees it.

use new_kernels::KernelError;

use crate::jit::{Ctx, dtype_dispatch, stated};
use crate::tensor::Tensor;

#[cfg(feature = "_cuda")]
pub mod dense;

#[cfg(feature = "_cuda")]
pub(crate) mod gemv;

/// A handle's address as the raw device pointer the cuBLAS host programs
/// take.
#[cfg(feature = "_cuda")]
pub(crate) const fn dev(ptr: u64) -> *mut core::ffi::c_void {
    ptr as usize as *mut core::ffi::c_void
}

pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), KernelError> {
    act_x_wt(ctx, "gemm.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), KernelError> {
    act_x_wt(ctx, "gemm.lm_head", act, w, y)
}

/// `layer` names the o-proj slice on planes that stack them; this plane's
/// weights arrive pre-sliced, so it is stated and unused — as before.
pub fn attention_landing(
    ctx: &Ctx,
    act: Tensor,
    w: Tensor,
    layer: u32,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    let _ = layer;
    act_x_wt(ctx, "gemm.attention_landing", act, w, y)
}

/// `y = act x w^T`. An empty projection (any extent zero) is a silent no-op,
/// as before — a conditioned batch may legitimately land nothing, and a
/// refusal here would kill the whole fire under graph capture.
pub fn act_x_wt(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let m = stated(op, y.rows)?;
    let n = stated(op, y.width)?;
    let k = stated(op, act.width)?;
    let handle = ctx.cublas(op)?;

    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16(handle, dev(act.ptr), dev(w.ptr), dev(y.ptr), m, n, k, 0.0);
        Ok(())
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (handle, w, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}
