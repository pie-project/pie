#![allow(clippy::too_many_arguments)]

//! `y = act · wᵀ` for a bf16 projection: the skinny kernel for a decode's
//! few rows (the gemv where it refuses), hipBLASLt's heuristic pick above
//! them (what torch and vLLM run on ROCm), and rocBLAS where hipBLASLt is
//! absent or offers nothing.

use crate::error::Error;
use crate::hip::blas::{self, Operand};
use crate::hip::{lt, sys};
use crate::jit::{Ctx, refuse};

use super::dense_batch::dense_batch_bf16;
use super::gemv::{gemv_bf16, skinny_bf16, skinny_swiglu_bf16};
use super::mgemm::mgemm_bf16;

/// Up to four rows the skinny kernel streams the weight faster than
/// hipBLASLt tiles it; up to eight it still does for a small weight.
const SKINNY_ALWAYS: i32 = 4;
const SKINNY_SMALL: i32 = 8;
const SKINNY_SMALL_WEIGHT: i64 = 1 << 21;

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
    let skinny = m <= SKINNY_ALWAYS
        || (m <= SKINNY_SMALL && i64::from(n) * i64::from(k) <= SKINNY_SMALL_WEIGHT);
    if bias == 0 && skinny && skinny_bf16(ctx, w, act, y, m, n, k).is_ok() {
        return Ok(false);
    }
    if m == 1 && gemv_bf16(ctx, w, act, bias, y, n, k).is_ok() {
        return Ok(bias != 0);
    }
    if bias == 0 && batch_on(ctx, m, n, k) && dense_batch_bf16(ctx, w, act, y, m, n, k, false)? {
        return Ok(false);
    }
    if bias == 0 && mgemm_on() && mgemm_bf16(ctx, w, act, y, m, n, k, false)? {
        return Ok(false);
    }
    let shape = lt::Shape {
        m: m.unsigned_abs(),
        n: n.unsigned_abs(),
        k: k.unsigned_abs(),
        bias: false,
        groups: 1,
    };
    let workspace = ctx.scratch_shared(op, "linear.lt_workspace", lt::WORKSPACE)? as u64;
    // SAFETY: the three planes are device ranges the caller sized at [m, k],
    // [n, k] and [m, n]; the workspace is this stream's.
    let start = crate::prof::on()
        .then(|| crate::prof::begin(ctx.stream()))
        .flatten();
    let landed = unsafe { lt::matmul(shape, act, w, 0, y, workspace, ctx.stream()) };
    crate::prof::end(ctx.stream(), start, || {
        format!("{op} hipblaslt m={m} n={n} k={k}")
    });
    if landed.map_err(|why| refuse(op, format!("{why} at M={m} N={n} K={k}")))? {
        return Ok(false);
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

/// `h = silu(act · gateᵀ) · (act · upᵀ)` from a packed `[gate | up]` bank,
/// in one fire where the skinny kernel covers the rows; `Ok(false)` leaves
/// the projection and the activation to the caller.
pub(crate) fn act_x_wt_swiglu(
    ctx: &Ctx,
    act: u64,
    gate_up: u64,
    h: u64,
    m: i32,
    intermediate: i32,
    k: i32,
) -> Result<bool, Error> {
    if m <= 0 || intermediate <= 0 || k <= 0 {
        return Ok(false);
    }
    let skinny = m <= SKINNY_ALWAYS
        || (m <= SKINNY_SMALL && 2 * i64::from(intermediate) * i64::from(k) <= SKINNY_SMALL_WEIGHT);
    if skinny && skinny_swiglu_bf16(ctx, gate_up, act, h, m, intermediate, k).is_ok() {
        return Ok(true);
    }
    if batch_on(ctx, m, 2 * intermediate, k)
        && dense_batch_bf16(ctx, gate_up, act, h, m, intermediate, k, true)?
    {
        return Ok(true);
    }
    Ok(mgemm_on() && mgemm_bf16(ctx, gate_up, act, h, m, intermediate, k, true)?)
}

/// Whether the batch engine takes `m` rows over an n x k weight
/// (`kernel.dense.batch_*`).
fn batch_on(ctx: &Ctx, m: i32, n: i32, k: i32) -> bool {
    let t = ctx.tuning();
    let weight = i64::from(n) * i64::from(k);
    (t.dense_batch_from as i32..=t.dense_batch_until as i32).contains(&m)
        && (i64::from(t.dense_batch_weight_mi) << 20..=i64::from(t.dense_batch_weight_max_mi) << 20)
            .contains(&weight)
}

/// The decode-batch kernel streams the weight at ~1.7 TB/s against
/// hipBLASLt's ~2.2-2.9 (its 16-rows-a-read operand layout, and the read
/// counter's issue order, cap it), so it serves only under `PIE_MGEMM`
/// until its weight goes through LDS.
fn mgemm_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_MGEMM").is_some())
}
