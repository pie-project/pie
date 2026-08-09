#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Root, Routine};
use crate::routine;
use crate::x::abi::bf16;
use kernels::Refusal;

use core::ffi::c_void;

/// `gemm/gemv.cuh` — the root the GEMV routines compile a symbol out of.
///
/// The dense forms have no root: they are `cublasGemmEx` and its batched
/// siblings, so their "device text" is the library's.
pub static ROOT: Root =
    Root::new("gemm/gemv", include_str!("../../csrc/src/gemm/gemv.cuh"), "gemm/gemv.cuh");

/// The dense matmul's host program: the autotuner, the plan cache and the
#[cfg(feature = "_cuda")]
pub mod dense;
/// The GEMV's host program: the four instantiations' selection and launch.
///
/// Feature-free, unlike [`dense`]: it selects among four instantiations and
/// launches through `Ctx`, which is the same path every other family takes.
pub mod gemv;

/// `gemm::act_x_wt_bf16` — the dense matmul, tactic-selected.
///
/// # Safety
///
/// `act`, `w` and `y` must address `M*K`, `N*K` and `M*N` live bf16 elements
/// and outlive the launch — asynchronous on the handle's stream, so
/// "outlive" ends at the next synchronisation and not at this call's return.
pub fn act_x_wt_bf16(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- the three matrices address live device
    // memory of the extents `m`, `n` and `k` describe.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16(ctx.cublas()?, act, w, y, m, n, k, beta);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act, w, y, m, n, k, beta);
    Ok(())
}

/// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
///
/// # Safety
///
/// `act` and `w` must address `M*K` and `N*K` live bf16 elements, `y` must
/// address `M*N` live floats, and all three must outlive the launch.
pub fn act_x_wt_bf16_out_fp32(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    y: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Refusal> {
    // SAFETY: as [`act_x_wt_bf16`]'s, with `y` addressing `m * n` floats.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16_out_fp32(ctx.cublas()?, act, w, y, m, n, k);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act, w, y, m, n, k);
    Ok(())
}

/// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
///
/// # Safety
///
/// The three pointer arrays must be HOST arrays of `group_count` device
/// addresses (cuBLAS reads them on the host for the grouped form), and
/// `m_array_host` a host array of `group_count` row counts.
pub fn grouped_act_x_wt_bf16(
    ctx: &Ctx,
    act_ptrs_host: *const *const c_void,
    w_ptrs_host: *const *const c_void,
    y_ptrs_host: *const *mut c_void,
    m_array_host: *const i32,
    group_count: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    if group_count <= 0 {
        return Err(Refusal::Empty { what: "group_count" });
    }
    let handle = ctx.cublas()?;
    // SAFETY: the three pointer arrays are HOST arrays of `group_count`
    // device addresses -- cuBLAS reads them on the host for the grouped form.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::grouped_act_x_wt_bf16(
            handle,
            act_ptrs_host,
            w_ptrs_host,
            y_ptrs_host,
            m_array_host,
            group_count,
            n,
            k,
            beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (handle, act_ptrs_host, w_ptrs_host, y_ptrs_host, m_array_host, n, k, beta);
    Ok(())
}

/// `gemm::act_x_wt_bias_bf16` — TWO KERNELS IN ONE BODY.
///
/// # Safety
///
/// `act`, `w`, `bias` and `y` must address live device memory of the extents
/// `M`, `N` and `K` describe, and `y` must be writable.
pub fn act_x_wt_bias_bf16(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    bias: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    // SAFETY: as [`act_x_wt_bf16`]'s, plus `bias` addressing `n` bf16
    // elements when it is not null.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16(ctx.cublas()?, act, w, y, m, n, k, beta);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act, w, y, m, n, k, beta);
    if bias.is_null() {
        return Ok(());
    }
    // The second kernel is `norm`'s, and this is the one routine that fires
    // another family's: the bias add reads exactly what the GEMM above wrote,
    // on the same stream, so splitting them would be two statements for one
    // operation.
    crate::x::norm::add_bias_bf16(ctx, y.cast::<bf16>(), bias.cast::<bf16>(), m, n)
}

/// The GEMV, re-exported at family level so `routine!` can name it.
///
/// A routine is a concrete `fn` path and `gemv::gemv_bf16` lives a module
/// down; the re-export keeps its NAME, because `routine!` stringifies the
/// identifier it is handed and that name is half the trace symbol.
pub use gemv::gemv_bf16;

/// This family's routines: the four dense entry points and the GEMV.
///
/// Five of the twelve contracts below are NCCL/P2P collectives with no device
/// text in this tree at all, and three name quantised GEMMs whose host program
/// is `driver-cuda`'s. Those have no routine and the test names each one.
pub static ROUTINES: &[Routine] = &[
    routine!(act_x_wt_bf16),
    routine!(act_x_wt_bf16_out_fp32),
    routine!(grouped_act_x_wt_bf16, whole),
    routine!(act_x_wt_bias_bf16),
    routine!(gemv_bf16),
];

/// `gemm`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "gemm", routines: ROUTINES };
