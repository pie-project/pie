#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Routine};
use crate::{driver_bound, routine};
use crate::jit::abi::bf16;
use kernels::Refusal;

use core::ffi::c_void;

/// MLA's absorb pair: two `cublasGemmStridedBatchedEx` over the head axis.
///
/// Feature-free, like [`gemv`] and unlike [`dense`], because its two entry
/// points are in [`ROUTINES`] and a routine table may not change shape with a
/// feature. The cuBLAS call inside it is gated; the refusals around it are
/// not.
pub mod absorb;
/// The dense matmul's host program: the autotuner, the plan cache and the
#[cfg(feature = "_cuda")]
pub mod dense;
/// The GEMV's host program: the four instantiations' selection and launch.
///
/// Feature-free, unlike [`dense`]: it selects among four instantiations and
/// launches through `Ctx`, which is the same path every other family takes.
pub mod gemv;
/// The LoRA adapter correction's launch half: three passes of matmul over a
/// staged lane set.
///
/// Feature-free, and more thoroughly than [`absorb`] — it names no `cudarc`
/// type at all, because every launch it makes is one of this module's own
/// feature-free entry points or `quant`'s.
pub mod lora;
/// The quantised matmuls' host program: `gemm.cpp`'s router on the weight's
/// dtype, its cuBLASLt recipes and its two caches.
///
/// `_cuda`, for [`dense`]'s reason and not for a new one: the file names
/// `cudarc::cublaslt` types in its struct fields, so there is nothing left of
/// it once the binding is gone. The three entry points below are feature-free
/// anyway, because they are in [`ROUTINES`] and a routine table may not change
/// shape with a feature.
#[cfg(feature = "_cuda")]
pub mod quant;

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
    let _ =
        (handle, act_ptrs_host, w_ptrs_host, y_ptrs_host, m_array_host, group_count, n, k, beta);
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
    crate::norm::add_bias::<bf16>(ctx, y.cast::<bf16>(), bias.cast::<bf16>(), m, n)
}

// ── the quantised three, which a DRIVER fires and no statement binds ──
//
// The bodies are `quant`'s and the shape here is [`act_x_wt_bf16`]'s: a
// feature-free `fn` that takes `ctx.cublas()?` and hands it to the gated
// half. What is different is who calls them, and it is the whole reason the
// three lines at the bottom of [`ROUTINES`] are `driver_bound!` rather than
// `routine!`.
//
// **A statement cannot supply these arguments.** Six of the fifteen describe
// the weight's REPRESENTATION -- its dtype, its byte count, the scale's dtype
// and element count, the zero point, the channel axis or the group extent --
// and a trace states a value, not a layout. The dense `gemm::act_x_wt_bf16`
// takes none of them and crossed as a `routine!` for exactly that reason: the
// difference between the two is not the arithmetic, it is that one of them
// has to be told how to READ its operand.
//
// The view is assembled from a `WeightRepr` (`model-dsl`'s, three variants:
// `Bf16`, `Scaled`, `Mxfp4Marlin`), which is the COMPILER's vocabulary, and
// it does not cross the ABI: what crosses is the `i32` dtype code these
// parameters carry. So `WeightRepr` stays where it is. Moving it down would
// put a compiler type in a kernel crate to explain six integers that are
// already self-describing, and the six integers are what `gemm.hpp`'s inlines
// took.

/// `gemm::act_x_wt_channel_scaled` — one scale per output channel of `W`.
///
/// # Errors
///
/// [`Refusal::Absent`] if this context carries no cuBLAS handle. Everything
/// the router itself refuses, it refuses by PANICKING, and `quant`'s header
/// is the argument for why -- the C++ threw and the shim's `catch` aborted,
/// so a value here would be a fallback the archive never had.
///
/// # Safety
///
/// Every pointer must address live device memory of the extents `m`, `n` and
/// `k` describe, `w` holding at least `n * k` elements of `w_dtype` and
/// `scale` at least `n` values.
pub fn act_x_wt_channel_scaled(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    zero_point: *const c_void,
    channel_axis: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: the caller's obligation, forwarded.
    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_channel_scaled(
            handle, act, w, w_dtype, w_nbytes, scale, scale_dtype, scale_numel, zero_point,
            channel_axis, y, m, n, k, beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (
        handle, act, w, w_dtype, w_nbytes, scale, scale_dtype, scale_numel, zero_point,
        channel_axis, y, m, n, k, beta,
    );
    Ok(())
}

/// `gemm::act_x_wt_grouped_scaled` — one scale per group along `K`, and for
/// FP8 per 2-D block.
///
/// # Errors
///
/// As [`act_x_wt_channel_scaled`].
///
/// # Safety
///
/// As [`act_x_wt_channel_scaled`], with the scale count `group_size` implies.
pub fn act_x_wt_grouped_scaled(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    zero_point: *const c_void,
    group_size: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: the caller's obligation, forwarded.
    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_grouped_scaled(
            handle, act, w, w_dtype, w_nbytes, scale, scale_dtype, scale_numel, zero_point,
            group_size, y, m, n, k, beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (
        handle, act, w, w_dtype, w_nbytes, scale, scale_dtype, scale_numel, zero_point, group_size,
        y, m, n, k, beta,
    );
    Ok(())
}

/// `gemm::act_x_wt_mxfp4_marlin` — nibble-packed MXFP4 with E8M0 block
/// scales, dequanted and run through the classic GEMM.
///
/// # Errors
///
/// As [`act_x_wt_channel_scaled`].
///
/// # Safety
///
/// `w` must hold at least `ceil(n * k / 2)` bytes and `scale` at least
/// `n * ceil(k / 32)` bytes.
pub fn act_x_wt_mxfp4_marlin(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    w_nbytes: usize,
    scale: *const c_void,
    scale_numel: usize,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: the caller's obligation, forwarded.
    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_mxfp4_marlin(
            handle, act, w, w_nbytes, scale, scale_numel, y, m, n, k, beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (handle, act, w, w_nbytes, scale, scale_numel, y, m, n, k, beta);
    Ok(())
}

/// The GEMV, re-exported at family level so `routine!` can name it.
///
/// A routine is a concrete `fn` path and `gemv::gemv_bf16` lives a module
/// down; the re-export keeps its NAME, because `routine!` stringifies the
/// identifier it is handed and that name is half the trace symbol.
pub use gemv::gemv_bf16;

/// MLA's absorb pair, re-exported at family level for [`gemv_bf16`]'s reason:
/// `routine!` stringifies the identifier it is handed, and that identifier is
/// half the trace symbol.
pub use absorb::{mla_absorb_latent_to_v_bf16, mla_absorb_q_to_latent_bf16};

/// The adapter correction, re-exported for the same reason as the two above.
pub use lora::lora_qkv_correction;

/// This family's routines: the four dense entry points, the GEMV, MLA's
/// absorb pair, and the four forms the DRIVER fires by path.
///
/// The two absorbs are the family's only members whose CALLER is another
/// family's — MLA's lane fires them either side of its attention — and they
/// are here rather than in `attn` because a routine's symbol is its
/// family's namespace plus its name, and a trace states them as `gemm::`.
/// `absorb`'s own header is the full argument.
pub static ROUTINES: &[Routine] = &[
    routine!(act_x_wt_bf16),
    routine!(act_x_wt_bf16_out_fp32),
    routine!(grouped_act_x_wt_bf16, whole),
    routine!(act_x_wt_bias_bf16),
    routine!(gemv_bf16),
    routine!(mla_absorb_q_to_latent_bf16),
    routine!(mla_absorb_latent_to_v_bf16),
    // ── what the DRIVER fires, by path ──────────────────────────────────
    //
    // `driver_bound!` and not `routine!`, and the fact is per symbol rather
    // than per line: **no statement supplies a weight REPRESENTATION.** Six
    // of each signature's parameters describe how to read the operand rather
    // than which operand it is, and the extractor that builds a `routine!`'s
    // row from a `&[Value]` has nothing to recover them from.
    //
    // They were three rows in `not_yet_crossed.rs`, hand-transcribing columns
    // — all three of which were `false` and `&[]` — off `fn`s that did not
    // exist yet because their bodies were a crate up. The bodies are here
    // now and the rows derive.
    driver_bound!(act_x_wt_channel_scaled),
    driver_bound!(act_x_wt_grouped_scaled),
    driver_bound!(act_x_wt_mxfp4_marlin),
    // The adapter correction, for a DIFFERENT reason, and the difference is
    // worth one line: its arguments are not unstatable, they are a `Staged`
    // — a borrow of two `Vec`s the driver's per-fire staging built out of an
    // arena `Ctx` cannot offer. A trace states `q` and `v` and the guard that
    // wraps them; everything else here is the fire's.
    //
    // It was `pie_lora_qkv_correction`, a bare symbol that no `Family` could
    // ever have produced. `lora`'s header is the whole of that argument.
    driver_bound!(lora_qkv_correction),
];

/// `gemm`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);
