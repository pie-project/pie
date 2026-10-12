#![allow(clippy::too_many_arguments)]

//! `y = act · wᵀ` for a bf16 projection: the gemv kernel at one row, rocBLAS
//! everywhere else. rocBLAS on gfx94x already routes a bf16 gemm to its
//! hipBLASLt kernels, so this layer keeps no tactic table of its own yet.

use crate::error::Error;
use crate::hip::blas::{self, Operand};
use crate::hip::sys;
use crate::jit::{Ctx, refuse};

use super::gemv::gemv_bf16;

/// Returns whether the bias was folded into the store (only the gemv folds
/// one); `false` leaves the bias for the caller to add.
pub(crate) fn act_x_wt(
    ctx: &Ctx,
    op: &'static str,
    act: u64,
    w: u64,
    bias: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<bool, Error> {
    if m <= 0 || n <= 0 || k <= 0 {
        return Ok(false);
    }
    if m == 1 && gemv_bf16(ctx, w, act, bias, y, n, k).is_ok() {
        return Ok(bias != 0);
    }
    let handle = ctx.blas(op)?;
    // Row-major `y[m, n] = act[m, k] · w[n, k]ᵀ` is column-major
    // `yᵀ[n, m] = w[n, k] · actᵀ`, which is BLAS's `op(A)=Aᵀ, op(B)=B` with
    // A = w (k × n, ld k) and B = act (k × m, ld k).
    // SAFETY: the context's handle is live; the three planes are device
    // ranges the caller sized at [n, k], [m, k] and [m, n].
    unsafe {
        blas::bind(handle, ctx.stream())?;
        blas::gemm(
            handle,
            sys::BLAS_OP_T,
            sys::BLAS_OP_N,
            n,
            m,
            k,
            Operand::new(w, sys::BLAS_BF16, k),
            Operand::new(act, sys::BLAS_BF16, k),
            Operand::new(y, sys::BLAS_BF16, n),
        )
    }
    .map_err(|why| refuse(op, format!("{why} at M={m} N={n} K={k}")))?;
    Ok(false)
}
