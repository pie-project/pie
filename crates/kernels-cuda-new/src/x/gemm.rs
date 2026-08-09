#![allow(clippy::too_many_arguments)]

use crate::x::abi::{MaybeConst, bf16};
use crate::x::contract::{Fired, Refusal};
use crate::x::launch::Launch;
use crate::{contract, unit};

use core::ffi::c_void;

/// The dense matmul's host program: the autotuner, the plan cache and the
#[cfg(feature = "_cuda")]
pub mod dense;
/// The GEMV's host program: the four instantiations' selection and launch.
#[cfg(feature = "_cuda")]
pub mod gemv;

unit! {
    /// `gemm`'s device text: the row-per-warp GEMV and its split-K twin.
    unit GEMV = "gemm/gemv",
        text = include_str!("../../csrc/src/gemm/gemv.cuh"),
        file = "gemm/gemv.cuh";

    /// `gemv.cuh` — the SPLIT-K form: one block per output row, `kWarps`
    fn gemv_splitk = "gemm::device::gemv_splitk_bf16_kernel" (
        weight: *const bf16,
        act: *const bf16,
        bias: MaybeConst<bf16>,
        out: *mut bf16,
        n: i32,
        k: i32,
        beta: f32,
    ) {
        /// `gemm/gemv.cu:344-346`:
        "gemm::gemv_splitk_bf16_w4_u2" => "device::i32(4), 2",
        /// `gemm/gemv.cu:355-357`:
        "gemm::gemv_splitk_bf16_w8_u1" => "device::i32(8), 1",
    }

    /// `gemv.cuh` — the ROW-PER-WARP form: one warp per output row, `kWarps`
    fn gemv = "gemm::device::gemv_bf16_kernel" (
        weight: *const bf16,
        act: *const bf16,
        bias: MaybeConst<bf16>,
        out: *mut bf16,
        n: i32,
        k: i32,
        beta: f32,
    ) {
        /// `gemm/gemv.cu:372-374`:
        "gemm::gemv_bf16_w4_u2" => "device::i32(4), 2",
        /// `gemm/gemv.cu:382-383`:
        "gemm::gemv_bf16_w4_u4" => "device::i32(4), 4",
    }
}

/// `gemm::act_x_wt_bf16` — the dense matmul, tactic-selected.
///
/// # Safety
///
/// `act`, `w` and `y` must address `M*K`, `N*K` and `M*N` live bf16 elements
/// and outlive the launch — asynchronous on the handle's stream, so
/// "outlive" ends at the next synchronisation and not at this call's return.
#[cfg(feature = "_cuda")]
pub unsafe fn act_x_wt_bf16(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Fired {
    // SAFETY: the caller's obligation, above.
    unsafe { dense::act_x_wt_bf16(handle, act, w, y, m, n, k, beta) }
    Fired::Launched
}

/// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
///
/// # Safety
///
/// `act` and `w` must address `M*K` and `N*K` live bf16 elements, `y` must
/// address `M*N` live floats, and all three must outlive the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn act_x_wt_bf16_out_fp32(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Fired {
    // SAFETY: the caller's obligation, above.
    unsafe { dense::act_x_wt_bf16_out_fp32(handle, act, w, y, m, n, k) }
    Fired::Launched
}

/// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
///
/// # Safety
///
/// The three pointer arrays must be HOST arrays of `group_count` device
/// addresses (cuBLAS reads them on the host for the grouped form), and
/// `m_array_host` a host array of `group_count` row counts.
#[cfg(feature = "_cuda")]
pub unsafe fn grouped_act_x_wt_bf16(
    handle: *mut c_void,
    act_ptrs_host: *const *const c_void,
    w_ptrs_host: *const *const c_void,
    y_ptrs_host: *const *mut c_void,
    m_array_host: *const i32,
    group_count: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Fired {
    if group_count <= 0 {
        return Fired::Declined(Refusal::Empty { what: "group_count" });
    }
    // SAFETY: the caller's obligation, above.
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
    Fired::Launched
}

/// `gemm::act_x_wt_bias_bf16` — TWO KERNELS IN ONE BODY.
///
/// # Safety
///
/// `act`, `w`, `bias` and `y` must address live device memory of the extents
/// `M`, `N` and `K` describe, and `y` must be writable.
#[cfg(feature = "_cuda")]
pub unsafe fn act_x_wt_bias_bf16(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    bias: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
    beta: f32,
) -> Fired {
    // SAFETY: the caller's obligation, above.
    unsafe {
        dense::act_x_wt_bf16(handle, act, w, y, m, n, k, beta);
    }
    if bias.is_null() {
        return Fired::Launched;
    }
    let block = n.max(0).unsigned_abs().div_ceil(32).max(1).saturating_mul(32).min(1024);
    // SAFETY: `y` was just written by the GEMM above and is `m * n` bf16
    unsafe {
        crate::x::norm::add_bias::raw::add_bias(
            "norm::add_bias_bf16",
            Launch::per_row(m.max(0).unsigned_abs(), block),
            y.cast::<bf16>(),
            bias.cast::<bf16>(),
            n,
            stream,
        );
    }
    Fired::Launched
}

contract! {
    /// The in-place NCCL all-reduce.
    ALL_REDUCE = "dist::all_reduce_bf16" as all_reduce {
        whole: true,
        in_place: &[(0, 0)],
    }

    /// The OUT-OF-PLACE sum. Same collective, a separate destination —
    ALL_REDUCE_OUT = "dist::all_reduce_bf16_out" as all_reduce_out {
        whole: true,
    }

    /// The all-gather.
    ALL_GATHER = "dist::all_gather_bf16" as all_gather {
        whole: true,
    }

    /// The peer-to-peer all-reduce — `kernels::comm::CustomAllReduce`'s, not
    ALL_REDUCE_P2P = "comm::all_reduce_bf16" as all_reduce_p2p {
        whole: true,
    }

    /// The FUSED landing: sum, add the residual, norm. Two results — the
    ALL_REDUCE_RESIDUAL_RMSNORM = "comm::all_reduce_residual_rmsnorm_bf16"
        as all_reduce_residual_rmsnorm {
        whole: true,
        in_place: &[(0, 1)],
    }

    /// The plain x·Wᵀ, which every family fires.
    GEMM_XWT = "gemm::act_x_wt_bf16" as gemm_xwt {
        lowered_as: Some("gemm::act_x_w"),
    }

    /// `y[M, N] = act[M, K] x W[N, K]^T` with `W` quantized per output
    GEMM_XWT_CHANNEL_SCALED = "gemm::act_x_wt_channel_scaled" as gemm_xwt_channel_scaled

    /// The same, with `W` quantized per GROUP along K.
    GEMM_XWT_GROUPED_SCALED = "gemm::act_x_wt_grouped_scaled" as gemm_xwt_grouped_scaled

    /// MXFP4 through the Marlin kernels.
    GEMM_XWT_MXFP4_MARLIN = "gemm::act_x_wt_mxfp4_marlin" as gemm_xwt_mxfp4_marlin

    /// bf16 in, fp32 out — the one whose destination is not bf16.
    GEMM_OUT_FP32 = "gemm::act_x_wt_bf16_out_fp32" as gemm_out_fp32

    /// The grouped form. `whole` because the group boundaries (`M_array`)
    GEMM_GROUPED = "gemm::grouped_act_x_wt_bf16" as gemm_grouped {
        whole: true,
    }

    /// A projection with its bias in the EPILOGUE — one statement where a
    GEMM_BIAS = "gemm::act_x_wt_bias_bf16" as gemm_bias
}
