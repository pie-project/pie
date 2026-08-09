//! The dense bf16 GEMM and its autotuner, in Rust.
//!
//! # What this is
//!
//! The last of `crates/kernels-cuda/csrc/src/gemm/gemm.cpp`. That file began
//! at 2,216 lines with **zero `__global__`, zero `<<<>>>` and 138
//! cuBLAS/cuBLASLt calls**; §45 took its four pure-cuBLAS bodies into
//! `driver_cuda::bind::service`, §45's continuation took the quantized router
//! into `driver_cuda::bind::quant_gemm`, and this module takes what was left:
//! `gemm_bf16_impl`, `gemm_batched_bf16_impl`, the cuBLASLt plan cache and
//! the dense tactic autotuner around them.
//!
//! It was never a kernel file. It is a *host program* — a shape ladder, three
//! per-device caches, a private measurement stream and a disk memo — and the
//! rule it fell to is not "kernels compile through NVRTC" but the stronger
//! one: **every piece of CPU-side code is Rust.**
//!
//! # Why it went last, and why that reason is now spent
//!
//! `gemm_bf16_impl` called `gemv_bf16`, whose `bool` return meant *"I did not
//! launch"* — and §45.5 records that a row cannot decline. That kept the
//! whole tuner in C++, because the tuner has to be able to *ask* a candidate
//! whether it ran.
//!
//! The answer was never to make the row decline. It was that a
//! **driver-owned launch is not a row**: [`gemv_bf16`] is
//! Rust, its two `__global__`s are the `gemm/gemv` JIT unit, and its refusal
//! is a type — [`Gemv::Declined`] carrying *which* of the
//! four tests refused. So the tuner's `GemmKind::Gemv` arm is now
//! `matches!(gemv_bf16(..), Gemv::Launched)`, in the same short-circuiting
//! position the C++ put it in, and the ambiguity that blocked this file for
//! three arcs is gone by construction.
//!
//! # The four things this file is, in order
//!
//! 1. **A cuBLASLt plan cache** ([`lt_plan_for`]). Descriptor creation and
//!    the heuristic query are host work that would otherwise repeat per call.
//!    Per device, because a `cublasLtMatmulAlgo_t` is selected for one handle
//!    on one device and must not be replayed on another.
//! 2. **A shape ladder** ([`lt_algo_index_for_shape`], [`lt_min_n`]). Which
//!    heuristic index a shape prefers, measured per model. **Every one of
//!    those measurements is carried verbatim onto the function that made it**
//!    — they name specific checkpoints and specific regressions, and a port
//!    that dropped them would be deleting the only record.
//! 3. **A tactic autotuner** ([`tune_dense`]). The ladder is a list of
//!    measurements someone took once, on models that are not the ones being
//!    served today; so take the measurement here instead, on the real shape,
//!    and remember it — in memory and on disk.
//! 4. **The fallback ladder** ([`act_x_wt_bf16`]). Everything after the
//!    tuner is what serves shapes the tuner declined or could not run, and it
//!    is the C++'s own order: GEMV, then the Lt ladder, then `cublasGemmEx`
//!    with its two documented retries.
//!
//! # The tactic cache is the fourth thing beside streams, graphs and dispatch
//!
//! §51.4 predicted the driver would need one before this file could follow,
//! and it does: [`DenseGemmTuner`] holds a per-device memo, a per-device
//! recurrence counter and one process-wide [`DiskCache`]. That is the third
//! implementation of `tuning_cache.hpp`'s file format in this tree — after
//! the C++ and after `driver_cuda::fire::flashinfer_moe`'s — and it is a second
//! implementation rather than a shared one deliberately: `flashinfer_moe` is
//! behind `#[cfg(feature = "bridge")]`, and the dense GEMM is not optional.
//!
//! # Why nothing new links
//!
//! `cudarc`'s `fallback-dynamic-loading` resolves every cuBLAS and cuBLASLt
//! symbol with `dlopen` on first use, so `driver-cuda` still builds with no
//! CUDA toolkit installed — the hard, long-standing gate. `cublaslt` is in
//! the crate's `cudarc` feature list for `driver_cuda::bind::quant_gemm` already;
//! it is a binding-generation feature, not a link flag.

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmBatchedEx, cublasGemmEx,
    cublasGemmGroupedBatchedEx, cublasGetStream_v2, cublasGetVersion_v2, cublasHandle_t,
    cublasOperation_t, cublasSetStream_v2, cublasStatus_t, cudaDataType,
};
use cudarc::cublaslt::sys as lt;
use cudarc::runtime::sys::{
    cudaDeviceProp, cudaError, cudaEventCreate, cudaEventDestroy, cudaEventElapsedTime,
    cudaEventRecord, cudaEventSynchronize, cudaEvent_t, cudaFree, cudaGetDevice,
    cudaGetDeviceProperties_v2, cudaGetErrorName, cudaGetLastError, cudaMalloc, cudaMemGetInfo,
    cudaMemsetAsync, cudaPeekAtLastError, cudaStreamCaptureStatus, cudaStreamCreateWithFlags,
    cudaStreamDestroy, cudaStreamIsCapturing, cudaStreamNonBlocking, cudaStreamSynchronize,
};

use super::gemv::{Gemv, gemv_bf16};

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:82` — the compute type, and why it is not the fast one
// ───────────────────────────────────────────────────────────────────────────

/// `cublasComputeType_t bf16_compute_type() { return CUBLAS_COMPUTE_32F; }`.
///
/// **Never `CUBLAS_COMPUTE_32F_FAST_16BF`**, and the reason is a measurement
/// that no signature carries. Quoting `gemm.cpp:70-81` because the line that
/// held it is being deleted:
///
/// > `CUBLAS_COMPUTE_32F_FAST_16BF` exists to let a matmul over *fp32*
/// > operands round them to bf16 for the tensor cores. Operands that are
/// > already bf16 gain nothing from it, and cuBLASLt has no algorithm at all
/// > for many bf16 shapes under it — the MLA absorb batches and the MoE
/// > expert batch among them. Its heuristic query then fails on every call,
/// > and cuBLAS silently retries the matmul in `CUBLAS_COMPUTE_32F`. That
/// > internal retry is not reliable when eight rank threads take it at the
/// > same instant: when it loses the race the call returns `NOT_SUPPORTED` or
/// > `INTERNAL_ERROR`, and if it happened inside a graph capture the failure
/// > also invalidates the capture, so the next GEMM dies far from the cause.
/// > **That is what killed roughly one boot in ten at tp > 1.**
/// > `CUBLAS_COMPUTE_32F` is what bf16 operands should have been asking for
/// > all along: same tensor cores, same fp32 accumulate, no fallback to race.
const COMPUTE: cublasComputeType_t = cublasComputeType_t::CUBLAS_COMPUTE_32F;

/// `CUBLAS_GEMM_DEFAULT_TENSOR_OP` — the pin every call starts with.
const ALGO_TENSOR_OP: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

/// `CUBLAS_GEMM_DEFAULT` — the un-pinned retry. See [`act_x_wt_bf16`]'s
/// `NOT_SUPPORTED` arm for the shape that needs it.
const ALGO_DEFAULT: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT;

/// `gemm.cpp:92` — `cublaslt_bf16_workspace_bytes()`, 64 MiB.
///
/// The heuristics are queried with this number, so [`DenseTuneArena`] must
/// allocate exactly it: an algorithm that needs the full amount and is handed
/// less is a failure the tuner would misread as "this candidate is slow".
const LT_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

/// `gemm.cpp:85` — throw on a non-success status.
fn check(status: cublasStatus_t, what: &str) {
    assert!(
        status == cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "cuBLAS error ({status:?}): {what}"
    );
}

/// Clears a sticky CUDA error, the C++'s bare `cudaGetLastError();`.
fn clear_error() {
    let _ = unsafe { cudaGetLastError() };
}

/// The device this thread is bound to, or 0.
fn current_device() -> i32 {
    let mut dev: i32 = 0;
    let _ = unsafe { cudaGetDevice(&raw mut dev) };
    dev
}

/// `gemm.cpp:195` — the handle's stream, or `None` if cuBLAS will not say.
///
/// **Never guess.** Falling back to the null stream would run the GEMV
/// outside the caller's ordering and race whatever produced its input.
fn cublas_stream(handle: cublasHandle_t) -> Option<*mut c_void> {
    let mut stream: cudarc::cublas::sys::cudaStream_t = std::ptr::null_mut();
    if unsafe { cublasGetStream_v2(handle, &raw mut stream) } == cublasStatus_t::CUBLAS_STATUS_SUCCESS
    {
        Some(stream.cast())
    } else {
        None
    }
}

/// `cudaStreamIsCapturing`, with a failed query reported as "unknown".
fn capture_status(stream: *mut c_void) -> Option<cudaStreamCaptureStatus> {
    let mut status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
    if unsafe { cudaStreamIsCapturing(stream.cast(), &raw mut status) } != cudaError::cudaSuccess {
        clear_error();
        return None;
    }
    Some(status)
}

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:104` — per_device_singleton<T>
// ───────────────────────────────────────────────────────────────────────────
//
// Tensor parallelism runs every rank inside this one process, each bound to
// its own device. A plain process-global would therefore hand rank 1 state
// that belongs to rank 0's device: cuBLASLt would run both ranks' matmuls
// against a single scratch buffer (a live data race), and any algorithm that
// zeroes its workspace bakes a memset of that foreign pointer into the
// captured decode graph, which makes `cudaGraphInstantiate` reject the graph
// on every rank but rank 0.
//
// The C++ had a `thread_local` fast path in front of a `static` mutex-guarded
// map. Rust spells the map the same way and drops the thread-local cache: the
// contents are behind a `Mutex` either way, and a `thread_local` holding a
// `&mut` into a mutex-guarded map is exactly the aliasing this language
// exists to refuse. The cost is one uncontended lock per call on a path that
// is already about to enter cuBLAS.

/// The cuBLASLt handle and shared workspace for one device.
struct Bf16LtCtx {
    handle: lt::cublasLtHandle_t,
    workspace: *mut c_void,
    workspace_bytes: usize,
}

// SAFETY: `handle` and `workspace` are device-side resources reached only
// under the map's `Mutex`, and the pointers are never dereferenced by Rust.
unsafe impl Send for Bf16LtCtx {}

impl Bf16LtCtx {
    /// `gemm.cpp:130` — `ensure()`. Idempotent; both halves are separately
    /// guarded because the C++'s were.
    fn ensure(&mut self) {
        if self.handle.is_null() {
            check(
                unsafe { lt::cublasLtCreate(&raw mut self.handle) },
                "cublasLtCreate",
            );
        }
        if self.workspace.is_null() {
            let code = unsafe { cudaMalloc(&raw mut self.workspace, self.workspace_bytes) };
            assert!(
                code == cudaError::cudaSuccess && !self.workspace.is_null(),
                "cudaMalloc({}) for the cuBLASLt bf16 workspace failed with {code:?}",
                self.workspace_bytes
            );
        }
    }
}

/// The three fields of this device's [`Bf16LtCtx`], copied out.
///
/// Copied rather than borrowed so the lock is never held across a cuBLASLt
/// call — the tuner reaches this from inside a matmul loop, and a `&mut`
/// held that long would deadlock the moment two ranks shared a device.
fn lt_ctx() -> (lt::cublasLtHandle_t, *mut c_void, usize) {
    static CTXS: OnceLock<Mutex<HashMap<i32, Bf16LtCtx>>> = OnceLock::new();
    let mut map = CTXS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("dense-GEMM cuBLASLt context mutex poisoned");
    let ctx = map.entry(current_device()).or_insert_with(|| Bf16LtCtx {
        handle: std::ptr::null_mut(),
        workspace: std::ptr::null_mut(),
        workspace_bytes: LT_WORKSPACE_BYTES,
    });
    ctx.ensure();
    (ctx.handle, ctx.workspace, ctx.workspace_bytes)
}

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:157` — Bf16LtPlan
// ───────────────────────────────────────────────────────────────────────────

/// The descriptors for one `(M, N, K)`, plus every algorithm the heuristic
/// offered for it.
///
/// The heuristic list is kept whole — not just `heuristics[0]` — so the
/// autotuner can time them against each other instead of trusting the order
/// they came back in.
struct Bf16LtPlan {
    op_desc: lt::cublasLtMatmulDesc_t,
    a_desc: lt::cublasLtMatrixLayout_t,
    b_desc: lt::cublasLtMatrixLayout_t,
    c_desc: lt::cublasLtMatrixLayout_t,
    heuristics: Vec<lt::cublasLtMatmulHeuristicResult_t>,
}

// SAFETY: the four descriptors are opaque cuBLASLt handles, never
// dereferenced by Rust; `cublasLtMatmul` is documented as safe to call from
// several threads with one descriptor set. The `Arc` these live behind is
// what makes the plan outlive the cache lock, which is the C++'s `shared_ptr`
// exactly.
unsafe impl Send for Bf16LtPlan {}
// SAFETY: as above — shared access is read-only after `build_lt_plan`.
unsafe impl Sync for Bf16LtPlan {}

impl Drop for Bf16LtPlan {
    /// Reverse of creation, matching `gemm.cpp:168-173`.
    fn drop(&mut self) {
        unsafe {
            if !self.c_desc.is_null() {
                let _ = lt::cublasLtMatrixLayoutDestroy(self.c_desc);
            }
            if !self.b_desc.is_null() {
                let _ = lt::cublasLtMatrixLayoutDestroy(self.b_desc);
            }
            if !self.a_desc.is_null() {
                let _ = lt::cublasLtMatrixLayoutDestroy(self.a_desc);
            }
            if !self.op_desc.is_null() {
                let _ = lt::cublasLtMatmulDescDestroy(self.op_desc);
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:199` and `:222` — THE SHAPE LADDER
// ───────────────────────────────────────────────────────────────────────────

/// Which returned cuBLASLt heuristic a shape prefers.
///
/// **This function is a list of measurements, and each comment names the
/// checkpoint it was taken on.** It is kept even though [`tune_dense`] now
/// measures the same choice directly, for two reasons: it orders the
/// candidate list, so a shape where nothing measurably wins keeps doing what
/// it did before; and it is still the answer for every shape the tuner
/// declines (too large to allocate a probe output for) or cannot run.
fn lt_algo_index_for_shape(n: i32, k: i32) -> i32 {
    // Qwen3-0.6B's lm_head shape (K=1024, very wide N) consistently prefers
    // the third returned Lt heuristic. Larger hidden sizes regress on that
    // choice, so keep the old default for them.
    if k < 2048 && n >= 12288 {
        return 2;
    }
    // Qwen3.6-35B-A3B's MTP/lm_head shape (K=2048, very wide vocab) is a
    // small but repeatable win on the second returned heuristic.
    if k == 2048 && n >= 200_000 {
        return 1;
    }
    // Qwen3.6-27B's H=5120 projections and lm_head consistently prefer the
    // first returned heuristic. `lt_min_n` already keeps smaller GEMMs on the
    // regular cuBLAS path.
    if k == 5120 {
        return 0;
    }
    // Qwen3.6-35B-A3B's hidden-size projections (for example GDN qkv and
    // full-attention q/gate, N≈8k) are faster on the first heuristic. The old
    // generic index 5 regresses the MTP verifier by several percent.
    if k == 2048 && n >= 6144 {
        return 0;
    }
    // Gemma4 E4B's target lm_head (K=2560, very wide vocab) is slightly
    // faster with the first returned Lt heuristic; keep this narrow so the
    // MTP assistant scorer (K=256) and other projection GEMMs stay unchanged.
    if k == 2560 && n >= 100_000 {
        return 0;
    }
    5
}

/// The narrowest output width at which the Lt ladder is worth taking.
///
/// Two measurements, and the second one is a fault rather than a slowdown:
///
/// * Small hidden-size models (H=1024) only benefited from cuBLASLt on the
///   very wide lm_head; routing their 2k/6k projection GEMMs through Lt was
///   consistently slower. H=2048 keeps the previous threshold because the
///   1.7B-class models still prefer Lt for their 6k-wide MLP projection.
/// * For large hidden sizes, the current Lt heuristic can select kernels that
///   **fault** on compact multi-row lm_head shapes such as Kimi TP greedy
///   prefill (M small, N ≈ 20k, K ≈ 7k). The classic cuBLAS path is stable
///   for those shapes and is already used for M=1 decode, so keep Lt out of
///   the large-H wide-output path by default.
fn lt_min_n(k: i32) -> i32 {
    if k >= 4096 {
        return 32768;
    }
    if k < 2048 {
        12288
    } else if k == 2048 {
        6144
    } else {
        12288
    }
}

/// `gemm.cpp:236-238` — the other three gates on the Lt ladder. `MAX_N == 0`
/// means "no upper bound", which is how the C++ spelled a disabled ceiling.
const LT_MIN_K: i32 = 1024;
/// See [`LT_MIN_K`]. M=1 is the GEMV's shape, not Lt's.
const LT_MIN_M: i32 = 2;
/// See [`LT_MIN_K`].
const LT_MAX_N: i32 = 0;

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:245` — running one Lt algorithm
// ───────────────────────────────────────────────────────────────────────────

/// One `cublasLtMatmul`. `true` iff the status was success.
///
/// `workspace` is `None` for the context's shared scratch. It is overridable
/// because **the autotuner runs matmuls on a stream of its own**, concurrently
/// with whatever the caller's stream has in flight, and two matmuls
/// scribbling on one scratch buffer silently corrupt each other's results.
#[allow(clippy::too_many_arguments)]
fn run_lt_algo(
    plan: &Bf16LtPlan,
    algo: *const lt::cublasLtMatmulAlgo_t,
    stream: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    beta: f32,
    workspace: Option<(*mut c_void, usize)>,
) -> bool {
    let alpha = 1.0f32;
    let (handle, ctx_ws, ctx_ws_bytes) = lt_ctx();
    let (ws, ws_bytes) = workspace.unwrap_or((ctx_ws, ctx_ws_bytes));
    // SAFETY: descriptors belong to `plan`, which outlives the call; the four
    // pointers are the caller's device addresses. `y` is passed as both C and
    // D, which is the in-place form the C++ used and what `beta = 1` needs.
    let status = unsafe {
        lt::cublasLtMatmul(
            handle,
            plan.op_desc,
            std::ptr::from_ref(&alpha).cast(),
            w,
            plan.a_desc,
            act,
            plan.b_desc,
            std::ptr::from_ref(&beta).cast(),
            y.cast_const(),
            plan.c_desc,
            y,
            plan.c_desc,
            algo,
            ws,
            ws_bytes,
            stream.cast(),
        )
    };
    status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

/// [`run_lt_algo`] against the stream `cublas_handle` is bound to.
#[allow(clippy::too_many_arguments)]
fn run_lt_plan(
    plan: &Bf16LtPlan,
    algo: &lt::cublasLtMatmulAlgo_t,
    cublas_handle: cublasHandle_t,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    beta: f32,
    workspace: Option<(*mut c_void, usize)>,
) -> bool {
    // The C++ ignored this status and passed a null stream on failure, which
    // is the legacy default stream — carried, because changing it here would
    // change which stream a failed query runs on inside a capture.
    let stream = cublas_stream(cublas_handle).unwrap_or(std::ptr::null_mut());
    run_lt_algo(plan, std::ptr::from_ref(algo), stream, act, w, y, beta, workspace)
}

/// `gemm.cpp:296` — create the descriptors for a shape and ask cuBLASLt which
/// algorithms it would consider.
///
/// **Nothing is run.** The caller decides which of `heuristics` to use, either
/// by [`lt_algo_index_for_shape`] or by measuring them.
fn build_lt_plan(m: i32, n: i32, k: i32) -> Option<Arc<Bf16LtPlan>> {
    let (handle, _, workspace_bytes) = lt_ctx();
    let mut plan = Bf16LtPlan {
        op_desc: std::ptr::null_mut(),
        a_desc: std::ptr::null_mut(),
        b_desc: std::ptr::null_mut(),
        c_desc: std::ptr::null_mut(),
        heuristics: Vec::new(),
    };
    let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;

    // The C++ chained every step on `st == SUCCESS`, so the first failure
    // skips the rest and the descriptors built so far are still destroyed by
    // `~Bf16LtPlan`. A closure gives the same short-circuit with `?`, and
    // `plan`'s `Drop` gives the same cleanup.
    let transa = cublasOperation_t::CUBLAS_OP_T;
    let transb = cublasOperation_t::CUBLAS_OP_N;
    let mut st = unsafe {
        lt::cublasLtMatmulDescCreate(
            &raw mut plan.op_desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType::CUDA_R_32F,
        )
    };
    if st == ok {
        st = unsafe {
            lt::cublasLtMatmulDescSetAttribute(
                plan.op_desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                std::ptr::from_ref(&transa).cast(),
                std::mem::size_of::<lt::cublasOperation_t>(),
            )
        };
    }
    if st == ok {
        st = unsafe {
            lt::cublasLtMatmulDescSetAttribute(
                plan.op_desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                std::ptr::from_ref(&transb).cast(),
                std::mem::size_of::<lt::cublasOperation_t>(),
            )
        };
    }
    // `A` is the weight [K, N] column-major (row-major [N, K]), `B` the
    // activation [K, M], `C`/`D` the output [N, M] — the transposed view the
    // module header describes.
    if st == ok {
        st = unsafe {
            lt::cublasLtMatrixLayoutCreate(
                &raw mut plan.a_desc,
                lt::cudaDataType::CUDA_R_16BF,
                u64::try_from(k).unwrap_or(0),
                u64::try_from(n).unwrap_or(0),
                i64::from(k),
            )
        };
    }
    if st == ok {
        st = unsafe {
            lt::cublasLtMatrixLayoutCreate(
                &raw mut plan.b_desc,
                lt::cudaDataType::CUDA_R_16BF,
                u64::try_from(k).unwrap_or(0),
                u64::try_from(m).unwrap_or(0),
                i64::from(k),
            )
        };
    }
    if st == ok {
        st = unsafe {
            lt::cublasLtMatrixLayoutCreate(
                &raw mut plan.c_desc,
                lt::cudaDataType::CUDA_R_16BF,
                u64::try_from(n).unwrap_or(0),
                u64::try_from(m).unwrap_or(0),
                i64::from(n),
            )
        };
    }
    if st == ok {
        st = unsafe { lt::cublasLtMatmulPreferenceCreate(&raw mut pref) };
    }
    if st == ok {
        st = unsafe {
            lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                std::ptr::from_ref(&workspace_bytes).cast(),
                std::mem::size_of::<usize>(),
            )
        };
    }
    // BAR SPLIT-K ALGORITHMS THAT REDUCE IN PLACE. Those accumulate the
    // partial products straight into the output buffer, serialised by
    // counters in the workspace -- so every partial lands exactly once, but in
    // the order the CTAs happen to arrive. Floating-point addition is not
    // associative, so the last bit of the result depends on GPU scheduling,
    // and a greedy decode will silently pick a different token from one run to
    // the next whenever two logits are close. It is not hypothetical: enabling
    // the fused MoE changed the occupancy enough to flip GLM-5.2's step-13
    // argmax about half the time, purely through the LM head's split-K order.
    // The other two schemes stage their partials in the workspace and reduce
    // them in a fixed order, which is reproducible; measured cost of the
    // restriction is nil.
    if st == ok {
        let deterministic: u32 = (lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_MASK
            as u32)
            & !(lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_INPLACE as u32);
        st = unsafe {
            lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                std::ptr::from_ref(&deterministic).cast(),
                std::mem::size_of::<u32>(),
            )
        };
    }

    let mut heuristics: [lt::cublasLtMatmulHeuristicResult_t; 8] =
        unsafe { core::mem::zeroed() };
    let mut returned: i32 = 0;
    if st == ok {
        st = unsafe {
            lt::cublasLtMatmulAlgoGetHeuristic(
                handle,
                plan.op_desc,
                plan.a_desc,
                plan.b_desc,
                plan.c_desc,
                plan.c_desc,
                pref,
                8,
                heuristics.as_mut_ptr(),
                &raw mut returned,
            )
        };
    }
    if !pref.is_null() {
        let _ = unsafe { lt::cublasLtMatmulPreferenceDestroy(pref) };
    }
    if st != ok || returned <= 0 {
        return None;
    }
    let count = (returned as usize).min(heuristics.len());
    plan.heuristics.extend_from_slice(&heuristics[..count]);
    Some(Arc::new(plan))
}

/// `gemm.cpp:370` — the plan for a shape, built once and shared.
///
/// Per device: a cached `cublasLtMatmulAlgo_t` is selected by heuristics for
/// one handle on one device and must not be replayed on another.
///
/// A failed build is **not** memoised, matching the C++: `build_lt_plan`
/// returning null leaves the map untouched, so a shape whose heuristic query
/// failed once is retried. That is deliberate — the query can fail for a
/// transient reason (a sticky error from an unrelated call) and latching it
/// would strand the shape on the fallback ladder for the process's life.
fn lt_plan_for(m: i32, n: i32, k: i32) -> Option<Arc<Bf16LtPlan>> {
    static PLANS: OnceLock<Mutex<HashMap<(i32, i32, i32, i32), Arc<Bf16LtPlan>>>> = OnceLock::new();
    let key = (current_device(), m, n, k);
    let cell = PLANS.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let map = cell.lock().expect("dense-GEMM Lt plan cache poisoned");
        if let Some(plan) = map.get(&key) {
            return Some(Arc::clone(plan));
        }
    }
    let plan = build_lt_plan(m, n, k)?;
    let mut map = cell.lock().expect("dense-GEMM Lt plan cache poisoned");
    Some(Arc::clone(map.entry(key).or_insert(plan)))
}

/// `gemm.cpp:384` — the Lt ladder: preferred index first, then every other
/// index in order, and `false` when none of them ran.
fn gemm_bf16_lt(
    cublas_handle: cublasHandle_t,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> bool {
    let Some(plan) = lt_plan_for(m, n, k) else {
        return false;
    };
    let returned = plan.heuristics.len() as i32;
    let preferred = lt_algo_index_for_shape(n, k);
    let begin = preferred.min((returned - 1).max(0));
    for pass in 0..2 {
        let (first, last) = if pass == 0 {
            (begin, begin + 1)
        } else {
            (0, returned)
        };
        for i in first..last {
            if pass == 1 && i == begin {
                continue;
            }
            let Some(h) = plan.heuristics.get(usize::try_from(i).unwrap_or(usize::MAX)) else {
                continue;
            };
            if run_lt_plan(&plan, &h.algo, cublas_handle, act, w, y, beta, None) {
                return true;
            }
        }
    }
    false
}

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:443` — DENSE bf16 GEMM AUTOTUNING
// ───────────────────────────────────────────────────────────────────────────
//
// Every linear layer in the model ends up here, and which kernel is fastest
// for a given (M, N, K) is not something anyone can predict: it depends on the
// shape, the architecture, and the cuBLAS build. This used to be encoded as a
// ladder of hand-written special cases -- "Qwen3.6-27B's H=5120 projections
// prefer the first heuristic", "keep cuBLASLt out of the large-H wide-output
// path" -- which is a list of measurements someone took once, on models that
// are not the ones being served today. Take the measurement here instead.
//
// The candidates are the same three things the ladder was choosing between:
// the warp-per-row GEMV (M=1 only), classic `cublasGemmEx`, and each algorithm
// cuBLASLt's heuristic offers. They are ordered so that the incumbent choice
// comes first, and ties are broken towards the front of the list, so a shape
// where nothing measurably wins keeps doing what it did before.

/// A candidate must beat the incumbent by this much to displace it; anything
/// closer is treated as a tie.
///
/// Below this the difference is timing noise, and switching on noise would
/// make the kernel choice — and so the last bit of every result — vary
/// between runs.
const TACTIC_MARGIN: f32 = 0.98;

/// `gemm.cpp:466` — which family a tactic names. The integers are ON DISK, in
/// `dense_gemm.txt`, so they may not be renumbered without a signature bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmKind {
    GemmEx = 0,
    Lt = 1,
    Gemv = 2,
}

impl GemmKind {
    /// The disk's integer back to a kind. `None` for anything outside the
    /// enum, which is how a corrupt or newer cache line is rejected — the
    /// C++'s `tactic.kind < 0 || tactic.kind > GemmKind::Gemv`.
    fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::GemmEx),
            1 => Some(Self::Lt),
            2 => Some(Self::Gemv),
            _ => None,
        }
    }

    /// The three spellings `PIE_GEMM_TUNE_LOG` prints.
    fn label(self) -> &'static str {
        match self {
            Self::GemmEx => "gemmex",
            Self::Lt => "lt",
            Self::Gemv => "gemv",
        }
    }
}

/// `gemm.cpp:469` — one candidate: a family and, for `Lt`, an index into the
/// plan's heuristic list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DenseTactic {
    kind: GemmKind,
    algo: i32,
}

/// `gemm.cpp:474` — the classic path, `cublasGemmEx` with the tensor-op pin.
fn run_gemm_ex(
    handle: cublasHandle_t,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> bool {
    let alpha = 1.0f32;
    // SAFETY: the caller's device pointers, of the extents its own contract
    // states. Row-major `y[M,N] = act[M,K] @ w[N,K]^T` is the column-major
    // `w * act^T`, which is where `OP_T/OP_N` and the `m=N, n=M` swap come
    // from.
    let status = unsafe {
        cublasGemmEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w,
            cudaDataType::CUDA_R_16BF,
            k,
            act,
            cudaDataType::CUDA_R_16BF,
            k,
            std::ptr::from_ref(&beta).cast(),
            y,
            cudaDataType::CUDA_R_16BF,
            n,
            COMPUTE,
            ALGO_TENSOR_OP,
        )
    };
    status == cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

/// `gemm.cpp:485` — run `t` on `handle`'s stream.
///
/// `false` means the kernel is not usable for this shape, which lets the
/// caller fall back rather than fail. **That is the whole reason this
/// function returns a `bool` and [`gemv_bf16`] does not:**
/// here `false` is one of several tactics declining and the caller has
/// another to try, so there is nothing to disambiguate; there, `false` was
/// the *only* answer and had to be told apart from "ran and wrote nothing".
#[allow(clippy::too_many_arguments)]
fn run_dense_tactic(
    handle: cublasHandle_t,
    t: DenseTactic,
    plan: Option<&Bf16LtPlan>,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    lt_workspace: Option<(*mut c_void, usize)>,
    bias: *const c_void,
) -> bool {
    // Only the GEMV epilogue can absorb a bias. Anything else must decline
    // rather than silently drop it.
    if !bias.is_null() && t.kind != GemmKind::Gemv {
        return false;
    }
    match t.kind {
        GemmKind::Gemv => {
            if !(beta == 0.0 || beta == 1.0) || m != 1 {
                return false;
            }
            let Some(stream) = cublas_stream(handle) else {
                return false;
            };
            // The C++'s `bool` with the ambiguity removed: `Declined` names
            // WHICH of the four tests refused, and every arm of it enqueues
            // nothing, so `y` is exactly as this call found it.
            matches!(
                gemv_bf16(w, act, bias, y, n, k, stream, beta),
                Gemv::Launched
            )
        }
        GemmKind::Lt => {
            let Some(plan) = plan else { return false };
            let Ok(idx) = usize::try_from(t.algo) else {
                return false;
            };
            let Some(h) = plan.heuristics.get(idx) else {
                return false;
            };
            run_lt_plan(plan, &h.algo, handle, act, w, y, beta, lt_workspace)
        }
        GemmKind::GemmEx => run_gemm_ex(handle, act, w, y, m, n, k, beta),
    }
}

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:531` — DenseTuneArena
// ───────────────────────────────────────────────────────────────────────────

/// Everything the probes need that must not end up in a captured graph.
///
/// Tuning has to be able to run while the caller's stream is mid graph
/// capture: decode shapes are only ever seen inside `cudaStreamBeginCapture`,
/// so a tuner that refused to run there would never see them. Capture is
/// opened in `cudaStreamCaptureModeRelaxed`, which permits allocation and
/// cross-stream synchronisation from the capturing thread, so the way to stay
/// out of the graph is to own everything that carries work: a private stream,
/// private events, and private activation and output buffers. The weights are
/// shared, but they are read-only and were written long before.
///
/// **The one thing this deliberately does NOT own is the cuBLAS handle.**
/// Creating one mid-capture invalidates the capture — `cublasCreate`
/// initialises on the legacy default stream, which implicitly synchronises
/// every blocking stream including the one being captured. So borrow the
/// caller's handle and point it at the private stream for the duration,
/// restoring it on the way out. Nothing else can be using it: we are inside
/// one of its own calls.
struct DenseTuneArena {
    stream: *mut c_void,
    start: cudaEvent_t,
    stop: cudaEvent_t,
    handle: cublasHandle_t,
    caller_stream: *mut c_void,
    act: *mut c_void,
    y: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
}

impl Drop for DenseTuneArena {
    fn drop(&mut self) {
        unsafe {
            // Restoring the stream also returns the borrowed handle to
            // cuBLAS's own workspace pool, undoing anything the probes did to
            // it.
            if !self.handle.is_null() {
                let _ = cublasSetStream_v2(self.handle, self.caller_stream.cast());
            }
            if !self.start.is_null() {
                let _ = cudaEventDestroy(self.start);
            }
            if !self.stop.is_null() {
                let _ = cudaEventDestroy(self.stop);
            }
            if !self.act.is_null() {
                let _ = cudaFree(self.act);
            }
            if !self.y.is_null() {
                let _ = cudaFree(self.y);
            }
            if !self.workspace.is_null() {
                let _ = cudaFree(self.workspace);
            }
            if !self.stream.is_null() {
                let _ = cudaStreamDestroy(self.stream.cast());
            }
        }
        clear_error();
    }
}

impl DenseTuneArena {
    /// An arena with nothing acquired. Every failure path in [`Self::init`]
    /// leaves this droppable, which is what the C++ got from its destructor
    /// running on an aggregate-initialised object.
    fn empty() -> Self {
        Self {
            stream: std::ptr::null_mut(),
            start: std::ptr::null_mut(),
            stop: std::ptr::null_mut(),
            handle: std::ptr::null_mut(),
            caller_stream: std::ptr::null_mut(),
            act: std::ptr::null_mut(),
            y: std::ptr::null_mut(),
            workspace: std::ptr::null_mut(),
            workspace_bytes: 0,
        }
    }

    /// `gemm.cpp:565` — acquire everything, or answer `false` having acquired
    /// what it could (which `Drop` then releases).
    fn init(&mut self, caller: cublasHandle_t, m: i32, n: i32, k: i32) -> bool {
        let act_bytes = (m as usize) * (k as usize) * 2;
        let y_bytes = (m as usize) * (n as usize) * 2;
        // Must match what the heuristics were queried with, or an algorithm
        // that needs the full amount will be handed less than it asked for.
        self.workspace_bytes = lt_ctx().2;
        let acquired = unsafe {
            cudaMalloc(&raw mut self.act, act_bytes.max(1)) == cudaError::cudaSuccess
                && cudaMalloc(&raw mut self.y, y_bytes.max(1)) == cudaError::cudaSuccess
                && cudaMalloc(&raw mut self.workspace, self.workspace_bytes)
                    == cudaError::cudaSuccess
                && cudaStreamCreateWithFlags(
                    std::ptr::from_mut(&mut self.stream).cast(),
                    cudaStreamNonBlocking,
                ) == cudaError::cudaSuccess
                && cudaEventCreate(&raw mut self.start) == cudaError::cudaSuccess
                && cudaEventCreate(&raw mut self.stop) == cudaError::cudaSuccess
        };
        if !acquired {
            clear_error();
            return false;
        }
        let Some(caller_stream) = cublas_stream(caller) else {
            return false;
        };
        self.caller_stream = caller_stream;
        // The probes run beside the caller's stream, not behind it, so
        // anything still in flight there would overlap them. During capture
        // that cannot happen -- capture records, it does not execute, and
        // synchronising a capturing stream is an error -- but everywhere else
        // drain it first. This is once per shape, at the cost of a stall the
        // tuning sync would have imposed anyway.
        let Some(capture) = capture_status(caller_stream) else {
            return false;
        };
        if capture == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone
            && unsafe { cudaStreamSynchronize(caller_stream.cast()) } != cudaError::cudaSuccess
        {
            clear_error();
            return false;
        }
        if unsafe { cublasSetStream_v2(caller, self.stream.cast()) }
            != cublasStatus_t::CUBLAS_STATUS_SUCCESS
        {
            return false;
        }
        self.handle = caller;
        // A GEMM's cost does not depend on its values, only that they are
        // finite. 0x3C3C is a small positive bf16.
        let filled = unsafe {
            cudaMemsetAsync(self.act, 0x3C, act_bytes, self.stream.cast())
                == cudaError::cudaSuccess
                && cudaMemsetAsync(self.y, 0x3C, y_bytes, self.stream.cast())
                    == cudaError::cudaSuccess
                && cudaStreamSynchronize(self.stream.cast()) == cudaError::cudaSuccess
        };
        if !filled {
            clear_error();
            return false;
        }
        true
    }
}

/// `gemm.cpp:606` — elapsed time of the fastest of seven runs, or `None` if
/// the candidate cannot run this shape.
///
/// Failures are expected — cuBLAS rejects some kernels for skinny shapes — so
/// they drop the candidate rather than propagate.
fn time_dense_tactic(
    arena: &DenseTuneArena,
    t: DenseTactic,
    plan: Option<&Bf16LtPlan>,
    w: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Option<f32> {
    const WARMUP: i32 = 3;
    const ITERS: i32 = 7;
    let ws = Some((arena.workspace, arena.workspace_bytes));
    let mut fire = || {
        run_dense_tactic(
            arena.handle,
            t,
            plan,
            arena.act.cast_const(),
            w,
            arena.y,
            m,
            n,
            k,
            beta,
            ws,
            std::ptr::null(),
        )
    };
    for _ in 0..WARMUP {
        if !fire() {
            let _ = unsafe { cudaStreamSynchronize(arena.stream.cast()) };
            clear_error();
            return None;
        }
    }
    if unsafe { cudaStreamSynchronize(arena.stream.cast()) } != cudaError::cudaSuccess {
        clear_error();
        return None;
    }
    let mut best: Option<f32> = None;
    for _ in 0..ITERS {
        let _ = unsafe { cudaEventRecord(arena.start, arena.stream.cast()) };
        if !fire() {
            let _ = unsafe { cudaStreamSynchronize(arena.stream.cast()) };
            clear_error();
            return None;
        }
        let _ = unsafe { cudaEventRecord(arena.stop, arena.stream.cast()) };
        if unsafe { cudaEventSynchronize(arena.stop) } != cudaError::cudaSuccess {
            clear_error();
            return None;
        }
        let mut ms = 0.0f32;
        if unsafe { cudaEventElapsedTime(&raw mut ms, arena.start, arena.stop) }
            != cudaError::cudaSuccess
        {
            clear_error();
            return None;
        }
        if best.is_none_or(|b| ms < b) {
            best = Some(ms);
        }
    }
    best
}

/// `gemm.cpp:653` — the ballot.
///
/// Ordered by what the shape would have used without tuning, because ties
/// resolve to the first entry.
///
/// `beta = 1` is on the ballot too: the GEMV folds the accumulate into its
/// epilogue, and excluding it meant every projection that adds into a
/// residual — `o_proj` on every model here — was decided without its fastest
/// candidate.
fn dense_candidates(plan: Option<&Bf16LtPlan>, m: i32, n: i32, k: i32, beta: f32) -> Vec<DenseTactic> {
    let mut out = Vec::new();
    if m == 1 && (beta == 0.0 || beta == 1.0) {
        out.push(DenseTactic {
            kind: GemmKind::Gemv,
            algo: 0,
        });
    }
    out.push(DenseTactic {
        kind: GemmKind::GemmEx,
        algo: 0,
    });
    if let Some(plan) = plan {
        let preferred = lt_algo_index_for_shape(n, k);
        let count = plan.heuristics.len() as i32;
        if preferred < count {
            out.push(DenseTactic {
                kind: GemmKind::Lt,
                algo: preferred,
            });
        }
        for i in 0..count {
            if i == preferred {
                continue;
            }
            out.push(DenseTactic {
                kind: GemmKind::Lt,
                algo: i,
            });
        }
    }
    out
}

// ───────────────────────────────────────────────────────────────────────────
// The on-disk tactic cache — `tuning_cache.hpp`
// ───────────────────────────────────────────────────────────────────────────

/// `tuning_cache.hpp:34` — mixes `v` into hash `h`.
///
/// **That header is deleted.** `gemm/gemm.cpp` was its last includer, so it
/// and `cache_root.hpp` went with it; the file format lives in this module
/// and in `driver_cuda::fire::flashinfer_moe`, and nowhere else. The line is
/// quoted rather than cited: `h ^= v + 0x9e3779b97f4a7c15 + (h << 6) + (h >>
/// 2)`, boost's `hash_combine` over the golden-ratio word.
///
/// Spelled here rather than reused from `driver_cuda::fire::flashinfer_moe`
/// because that module is `#[cfg(feature = "bridge")]` and the dense GEMM is
/// not optional. Identical constant, identical shifts; the keys are on disk
/// and must not move.
#[must_use]
pub const fn tuning_hash(h: u64, v: u64) -> u64 {
    h ^ (v
        .wrapping_add(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(h << 6)
        .wrapping_add(h >> 2))
}

/// `gemm.cpp:713` — the cache key for a dense shape.
///
/// `beta` folds in as a **bit**, not a value: the tactic only depends on
/// whether the GEMM accumulates, and the GEMV arm is on the ballot for both 0
/// and 1.
fn dense_key(m: i32, n: i32, k: i32, beta: f32) -> u64 {
    let mut h = 0u64;
    h = tuning_hash(h, m as u64);
    h = tuning_hash(h, n as u64);
    h = tuning_hash(h, k as u64);
    h = tuning_hash(h, u64::from(beta != 0.0));
    h
}

/// The C++'s `TuningCache`, for this one file's use.
///
/// `tuning_cache.hpp` is deleted — `gemm/gemm.cpp` was its last includer —
/// so this and `driver_cuda::fire::flashinfer_moe`'s copy are the only remaining
/// descriptions of the format. Both carry it in full for that reason.
///
/// Every observable is preserved: the signature is line 1, entries are
/// `%016llx %d %d`, writes APPEND (so concurrent ranks cannot truncate each
/// other, and a key written twice is harmless because the last line read
/// wins), and a file whose first line does not match is discarded AND removed
/// rather than replayed.
struct DiskCache {
    signature: String,
    path: Option<PathBuf>,
    entries: HashMap<u64, (i32, i32)>,
}

impl DiskCache {
    fn new(name: &str, signature: String) -> Self {
        let path = cache_path(name);
        let mut cache = Self {
            signature,
            path,
            entries: HashMap::new(),
        };
        if !cache.signature.is_empty() && cache.path.is_some() {
            cache.load();
        }
        cache
    }

    fn lookup(&self, key: u64) -> Option<(i32, i32)> {
        self.entries.get(&key).copied()
    }

    fn store(&mut self, key: u64, a: i32, b: i32) {
        self.entries.insert(key, (a, b));
        let (Some(path), false) = (self.path.as_ref(), self.signature.is_empty()) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        else {
            return;
        };
        // `std::ftell` on a freshly `fopen(.., "a")`d file answers the file's
        // size, so the C++'s `ftell(f) == 0` is "this file is new".
        let empty = file.metadata().is_ok_and(|meta| meta.len() == 0);
        if empty {
            let _ = writeln!(file, "{}", self.signature);
        }
        let _ = writeln!(file, "{key:016x} {a} {b}");
    }

    fn load(&mut self) {
        let Some(path) = self.path.clone() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let mut lines = text.lines();
        let matches = lines
            .next()
            .is_some_and(|first| first.trim_end_matches(['\n', '\r']) == self.signature);
        if matches {
            // `fscanf("%llx %d %d")` skips whitespace freely and stops at the
            // first token that does not parse, which is what this is.
            let mut fields = lines.flat_map(str::split_whitespace);
            while let (Some(k), Some(a), Some(b)) = (fields.next(), fields.next(), fields.next()) {
                let (Ok(k), Ok(a), Ok(b)) = (
                    u64::from_str_radix(k, 16),
                    a.parse::<i32>(),
                    b.parse::<i32>(),
                ) else {
                    break;
                };
                self.entries.insert(k, (a, b));
            }
        } else {
            // Entries measured against a different GPU or cuBLAS build do not
            // name the kernels we would run today, so the file is worse than
            // nothing.
            self.entries.clear();
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// `cache_root.hpp`'s derivation, carried because that header is deleted too:
/// XDG, else `$HOME/.cache`.
///
/// `None` when neither is set, which is a real configuration on a locked-down
/// host and is why the C++ returned an empty path rather than guessing.
fn cache_path(name: &str) -> Option<PathBuf> {
    if let Some(xdg) = std::env::var("XDG_CACHE_HOME").ok().filter(|s| !s.is_empty()) {
        return Some(Path::new(&xdg).join("pie").join(name));
    }
    if let Some(home) = std::env::var("HOME").ok().filter(|s| !s.is_empty()) {
        return Some(Path::new(&home).join(".cache").join("pie").join(name));
    }
    None
}

/// `gemm.cpp:676` — `# pie-dense-gemm v1 sm<major><minor> cublas=<n> dev=<name>`.
///
/// Empty when the device cannot be identified, which disables the cache.
fn dense_cache_signature() -> String {
    let mut device: i32 = 0;
    if unsafe { cudaGetDevice(&raw mut device) } != cudaError::cudaSuccess {
        clear_error();
        return String::new();
    }
    let mut prop = unsafe { core::mem::zeroed::<cudaDeviceProp>() };
    if unsafe { cudaGetDeviceProperties_v2(&raw mut prop, device) } != cudaError::cudaSuccess {
        clear_error();
        return String::new();
    }
    let mut version: i32 = 0;
    // The C++ passed a null handle, which `cublasGetVersion` accepts.
    let _ = unsafe { cublasGetVersion_v2(std::ptr::null_mut(), &raw mut version) };
    let name = unsafe { CStr::from_ptr(prop.name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    format!(
        "# pie-dense-gemm v1 sm{}{} cublas={version} dev={name}",
        prop.major, prop.minor
    )
}

/// The tactic file's basename. Unchanged from the C++, so a machine that has
/// tuned once keeps its answers across this port.
const CACHE_FILE: &str = "dense_gemm.txt";

/// `gemm.cpp:693` — the per-device memo, the recurrence counter and the disk.
struct DenseGemmTuner {
    chosen: HashMap<u64, DenseTactic>,
    seen: HashMap<u64, i32>,
    disk: DiskCache,
}

/// Ceiling on how many shapes will ever be measured, so a workload with an
/// unbounded spread of shapes cannot spend unbounded time tuning or grow the
/// on-disk cache without limit. The decode lattice is a few dozen shapes per
/// model; this is far above it.
const MAX_TUNED_SHAPES: usize = 1024;

/// The per-device tuner map.
///
/// The C++ made the whole `DenseGemmTuner` — including its `TuningCache` — a
/// `per_device_singleton`, so a two-GPU process built the disk cache twice
/// with two signatures over one file. **That is a finding carried, not
/// fixed**: the signatures differ only if the two devices differ, and
/// `DiskCache::load` deletes a file whose first line does not match, so a
/// heterogeneous box degrades to "no disk cache" rather than to wrong
/// answers. Homogeneous multi-GPU — every deployment in this tree — writes
/// one signature from both.
fn with_tuner<R>(f: impl FnOnce(&mut DenseGemmTuner) -> R) -> R {
    static TUNERS: OnceLock<Mutex<HashMap<i32, DenseGemmTuner>>> = OnceLock::new();
    let mut map = TUNERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("dense-GEMM tuner mutex poisoned");
    let tuner = map.entry(current_device()).or_insert_with(|| DenseGemmTuner {
        chosen: HashMap::new(),
        seen: HashMap::new(),
        disk: DiskCache::new(CACHE_FILE, dense_cache_signature()),
    });
    f(tuner)
}

/// `PIE_GEMM_TUNE_LOG`, read once per process.
fn tune_log() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_GEMM_TUNE_LOG").is_some())
}

/// `gemm.cpp:725` — measure every candidate and pick one.
fn tune_dense(
    caller: cublasHandle_t,
    plan: Option<&Bf16LtPlan>,
    w: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> DenseTactic {
    let candidates = dense_candidates(plan, m, n, k, beta);
    // `candidates` is never empty: `GemmEx` is pushed unconditionally.
    let mut best = candidates[0];

    let mut arena = DenseTuneArena::empty();
    if !arena.init(caller, m, n, k) {
        return best;
    }

    let mut timings: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
    let mut fastest: Option<f32> = None;
    for (i, cand) in candidates.iter().enumerate() {
        let Some(ms) = time_dense_tactic(&arena, *cand, plan, w, m, n, k, beta) else {
            continue;
        };
        if ms <= 0.0 {
            continue;
        }
        timings.push((i, ms));
        if fastest.is_none_or(|f| ms < f) {
            fastest = Some(ms);
        }
    }
    // PIE_GEMM_TUNE_LOG also dumps every candidate's measured time, not just
    // the winner: knowing that the GEMV lost is not the same as knowing by how
    // much, and the gap is what says whether a better kernel is worth writing.
    if tune_log() {
        for &(i, ms) in &timings {
            eprintln!(
                "[gemm-cand] M={m} N={n} K={k} {}(algo={}) {:.1} us",
                candidates[i].kind.label(),
                candidates[i].algo,
                ms * 1000.0
            );
        }
    }
    let Some(fastest) = fastest.filter(|f| *f > 0.0) else {
        return best;
    };

    let cutoff = fastest / TACTIC_MARGIN;
    for &(i, ms) in &timings {
        if ms > cutoff {
            continue;
        }
        best = candidates[i];
        break;
    }
    best
}

/// `gemm.cpp:775` — choose (and on first sight of a shape, measure) the kernel
/// for this shape.
///
/// `None` means no measured choice is available, leaving the caller on its
/// original path. The plan is returned regardless of whether a tactic was:
/// it is the source of the cuBLASLt candidates, and the tactic names one of
/// them by index.
fn dense_tactic_for(
    caller: cublasHandle_t,
    w: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    capturing: cudaStreamCaptureStatus,
) -> (Option<Arc<Bf16LtPlan>>, Option<DenseTactic>) {
    // The arena allocates an M x N output. Tuning a shape whose output alone
    // would rival the KV cache is not worth the memory; those shapes are large
    // enough that cuBLAS's own choice is close to optimal anyway.
    const MAX_TUNE_OUTPUT_BYTES: usize = 256 * 1024 * 1024;
    if (m as usize) * (n as usize) * 2 > MAX_TUNE_OUTPUT_BYTES {
        return (None, None);
    }

    // Built for every shape, tuned or not.
    let plan = lt_plan_for(m, n, k);
    let key = dense_key(m, n, k, beta);

    // The C++ held `tuner.mu` across the memo lookup, the disk lookup, the
    // sweep and the insert. Same here: `with_tuner` holds the map's lock for
    // the whole closure, and `tune_dense` runs inside it. That serialises two
    // threads that meet a new shape at once, which is the point — the loser
    // finds the winner's answer in `chosen` rather than measuring it twice on
    // a device they are both driving.
    let tactic = with_tuner(|tuner| {
        if let Some(t) = tuner.chosen.get(&key) {
            return Some(*t);
        }
        if tuner.chosen.len() >= MAX_TUNED_SHAPES {
            return None;
        }
        // Measuring a shape costs ~10 kernel launches per candidate plus a
        // stall, which is only worth paying for a shape that will come back.
        // Decode shapes are seen exactly once here -- during graph capture --
        // and then replayed forever from the graph, so those must be tuned on
        // sight or never. Everything else is prefill, whose M is the token
        // count and so is effectively arbitrary; make it prove it recurs
        // before spending anything on it.
        if capturing == cudaStreamCaptureStatus::cudaStreamCaptureStatusNone {
            let seen = tuner.seen.entry(key).or_insert(0);
            *seen += 1;
            if *seen < 2 {
                return None;
            }
        }

        let tactic = match tuner.disk.lookup(key) {
            Some((kind, algo)) => match GemmKind::from_i32(kind) {
                Some(kind) => DenseTactic { kind, algo },
                None => {
                    let t = tune_dense(caller, plan.as_deref(), w, m, n, k, beta);
                    tuner.disk.store(key, t.kind as i32, t.algo);
                    t
                }
            },
            None => {
                let t = tune_dense(caller, plan.as_deref(), w, m, n, k, beta);
                tuner.disk.store(key, t.kind as i32, t.algo);
                t
            }
        };
        tuner.chosen.insert(key, tactic);
        // PIE_GEMM_TUNE_LOG: which kernel a shape ended up on. Logged HERE
        // rather than inside the tuner because the choice is cached on disk,
        // so on any machine that has run the model once the tuner never
        // executes again and a log inside it prints nothing.
        if tune_log() {
            eprintln!(
                "[gemm-tune] M={m} N={n} K={k} -> {}(algo={})",
                tactic.kind.label(),
                tactic.algo
            );
        }
        Some(tactic)
    });
    (plan, tactic)
}

/// `gemm.cpp:849` — side-effect-free peek at the tuner's verdict.
///
/// Deliberately does *not* call [`dense_tactic_for`]: that bumps the `seen`
/// counter and can trigger a tune, and asking *"would you have picked the
/// GEMV?"* must not change the answer to *"what will you pick?"*.
///
/// **It had no consumer in the archive** — `dense_tactic_already_gemv` was
/// swept across every `.cu`, `.cuh`, `.cpp`, `.hpp` and `.rs` in the worktree
/// and called from nowhere, its last caller having left with
/// `act_x_wt_bias_bf16`'s `M == 1` fused arm. It is carried rather than
/// dropped because the property it encodes — that a peek must not perturb —
/// is the kind a re-implementation gets wrong, and it is four lines.
#[must_use]
pub fn dense_tactic_is_gemv(m: i32, n: i32, k: i32, beta: f32) -> bool {
    let key = dense_key(m, n, k, beta);
    with_tuner(|tuner| {
        tuner
            .chosen
            .get(&key)
            .is_some_and(|t| t.kind == GemmKind::Gemv)
    })
}

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:862` — PIE_GEMM_PATH_TRACE
// ───────────────────────────────────────────────────────────────────────────

/// One line per dense bf16 GEMM naming the shape, the capture status and the
/// branch that served it — the discriminating probe for *"what did the boot
/// lattice bake"*.
///
/// The budget is 40,000 calls, process-wide. (The C++'s comment said "first
/// 200 calls only" and its counter said 40,000; the counter is the behaviour
/// and is what is carried, with the stale comment corrected rather than
/// copied.)
fn path_trace_take() -> bool {
    use std::sync::atomic::{AtomicI32, Ordering};
    static ON: OnceLock<bool> = OnceLock::new();
    static BUDGET: AtomicI32 = AtomicI32::new(40000);
    let on = *ON.get_or_init(|| {
        std::env::var("PIE_GEMM_PATH_TRACE")
            .is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
    });
    if !on {
        return false;
    }
    BUDGET.fetch_sub(1, Ordering::Relaxed) > 0
}

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:875` — gemm_bf16_impl, the dense entry point
// ───────────────────────────────────────────────────────────────────────────

/// `gemm::act_x_wt_bf16` — `y[M, N] = act[M, K] @ W[N, K]^T + beta * y`.
///
/// All bf16, fp32 accumulation. `beta = 0` overwrites; `beta = 1` fuses a
/// residual add.
///
/// # The order, which is the whole body
///
/// 1. **The tuner.** Which of the three kernel families is fastest for this
///    shape is a measurement, not a rule — so take it, once per shape, and
///    remember it. Everything after this point is the fallback for shapes the
///    tuner declined (too large to allocate a probe output for) or could not
///    run.
/// 2. **The GEMV**, at `M == 1 && beta == 0`. M=1 is the decode shape: a
///    single activation row against the whole weight, so there is no reuse for
///    a tiled GEMM to exploit and the call is a pure streaming read. cuBLAS
///    picks kernels sized for an M worth filling and reaches roughly half of
///    HBM bandwidth on these; a warp-per-row GEMV nearly doubles it.
/// 3. **The Lt ladder**, behind [`lt_min_n`], [`LT_MIN_M`], [`LT_MIN_K`] and
///    [`LT_MAX_N`].
/// 4. **`cublasGemmEx`**, with the two retries below.
///
/// # Safety
///
/// `act`, `w` and `y` must address `M*K`, `N*K` and `M*N` live bf16 elements
/// and outlive the launch — which is asynchronous on the handle's stream, so
/// "outlive" ends at the next synchronisation and not at this call's return.
/// `handle` must be a live `cublasHandle_t`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn act_x_wt_bf16(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let handle: cublasHandle_t = handle.cast::<cublasContext>();
    let path_trace = path_trace_take();
    if path_trace {
        let capture = cublas_stream(handle)
            .and_then(capture_status)
            .unwrap_or(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone);
        clear_error();
        eprintln!("[gemm-path] M={m} N={n} K={k} beta={beta} capturing={capture:?}");
    }

    // ── 1. the tuned tactic ────────────────────────────────────────────
    {
        let caller_stream = cublas_stream(handle);
        let capturing = caller_stream.and_then(capture_status);
        if let (Some(_), Some(capturing)) = (caller_stream, capturing) {
            let (plan, tactic) = dense_tactic_for(handle, w, m, n, k, beta, capturing);
            if let Some(tactic) = tactic
                && run_dense_tactic(
                    handle,
                    tactic,
                    plan.as_deref(),
                    act,
                    w,
                    y,
                    m,
                    n,
                    k,
                    beta,
                    None,
                    std::ptr::null(),
                )
            {
                if path_trace {
                    eprintln!(
                        "[gemm-path]   -> tuned kind={} algo={}",
                        tactic.kind as i32, tactic.algo
                    );
                }
                return;
            }
        }
        if path_trace {
            eprintln!("[gemm-path]   -> tuner declined/failed");
        }
        clear_error();
    }

    // ── 2. the warp-per-row GEMV ───────────────────────────────────────
    if m == 1 && beta == 0.0 {
        if let Some(stream) = cublas_stream(handle)
            && matches!(
                gemv_bf16(w, act, std::ptr::null(), y, n, k, stream, 0.0),
                Gemv::Launched
            )
        {
            if path_trace {
                eprintln!("[gemm-path]   -> gemv");
            }
            return;
        }
    }

    // ── 3. the cuBLASLt ladder ─────────────────────────────────────────
    if m >= LT_MIN_M
        && n >= lt_min_n(k)
        && k >= LT_MIN_K
        && (LT_MAX_N == 0 || n <= LT_MAX_N)
        && gemm_bf16_lt(handle, act, w, y, m, n, k, beta)
    {
        if path_trace {
            eprintln!("[gemm-path]   -> lt-ladder");
        }
        return;
    }

    // ── 4. cublasGemmEx, and its two retries ───────────────────────────
    let alpha = 1.0f32;
    // SAFETY: the caller's obligation, above.
    let status = unsafe {
        cublasGemmEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w,
            cudaDataType::CUDA_R_16BF,
            k,
            act,
            cudaDataType::CUDA_R_16BF,
            k,
            std::ptr::from_ref(&beta).cast(),
            y,
            cudaDataType::CUDA_R_16BF,
            n,
            COMPUTE,
            ALGO_TENSOR_OP,
        )
    };
    if status == cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED {
        // `CUBLAS_GEMM_DEFAULT_TENSOR_OP` pins the tensor-core kernel family,
        // and cuBLAS has no member of it for some skinny shapes: the packed
        // q/k/v projection at TP=2 (N = Hq/T + 2*Hk/T = 2048, K = 1024) is
        // rejected at M=1, while the same M=1 succeeds at the TP=1 width
        // (N=4096) and at the packed gate/up width (N=3072). M=1 is not a
        // serving shape — it is the R=1 rung of the graph lattice, so this
        // surfaces only during upfront capture, and under TP it surfaced as a
        // HANG rather than an error: rank 0 threw out of capture while the
        // follower sat in `tp_graph_capture_barrier` waiting for a peer that
        // was never going to arrive.
        //
        // Retry without the tensor-op pin. cuBLAS then picks whatever kernel
        // fits; at M=1 there is no tensor-core throughput to lose anyway.
        let retry = unsafe {
            cublasGemmEx(
                handle,
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                std::ptr::from_ref(&alpha).cast(),
                w,
                cudaDataType::CUDA_R_16BF,
                k,
                act,
                cudaDataType::CUDA_R_16BF,
                k,
                std::ptr::from_ref(&beta).cast(),
                y,
                cudaDataType::CUDA_R_16BF,
                n,
                COMPUTE,
                ALGO_DEFAULT,
            )
        };
        if retry == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return;
        }
        // Neither GemmEx algorithm family covers the shape. cuBLASLt does —
        // it is normally skipped here by the `min_m`/`min_n` heuristics, which
        // exist to pick the FASTER path, not the only working one.
        if gemm_bf16_lt(handle, act, w, y, m, n, k, beta) {
            return;
        }
        panic!(
            "cuBLAS error ({retry:?}) after non-tensor-op and cuBLASLt retries: \
             cublasGemmEx[bf16] M={m} N={n} K={k}"
        );
    }
    check(status, &format!("cublasGemmEx[bf16] M={m} N={n} K={k}"));
}

// ───────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1005` — the batched twin
// ───────────────────────────────────────────────────────────────────────────

/// Whether `cublasGemmGroupedBatchedEx` can serve a shape, per device.
///
/// Only discoverable by calling it and looking at the status. That is fine on
/// a plain stream, but **a failed cuBLAS call inside a stream capture
/// INVALIDATES the capture**, and the next GEMM then dies with an unrelated
/// `INTERNAL_ERROR` far from the cause — intermittently, because which shapes
/// reach a capture first depends on rank timing. So speculate only outside
/// capture, remember the answer per shape, and while capturing an untried
/// shape go straight to the batched path.
fn grouped_support(key: u64) -> Option<bool> {
    static KNOWN: OnceLock<Mutex<HashMap<(i32, u64), bool>>> = OnceLock::new();
    let map = KNOWN
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("grouped-batched support map poisoned");
    map.get(&(current_device(), key)).copied()
}

/// Records what [`grouped_support`] could not answer. `emplace`, not
/// `insert_or_assign`: the C++ used `emplace`, so the FIRST answer for a
/// shape is the one that sticks.
fn store_grouped_support(key: u64, supported: bool) {
    static KNOWN: OnceLock<Mutex<HashMap<(i32, u64), bool>>> = OnceLock::new();
    let mut map = KNOWN
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("grouped-batched support map poisoned");
    map.entry((current_device(), key)).or_insert(supported);
}

/// `gemm::batched_act_x_wt_bf16` — per-batch `act`/`W`/`y` pointers, all
/// sharing one `(M, N, K)`.
///
/// **This symbol has no row.** `table::gemm` struck
/// `gemm::batched_act_x_wt_bf16` (§38) because its whole consumer set was one
/// unreachable inline, and `model-compiler`'s `dsl.rs:3722` still spells the
/// name for a lowering nothing emits. It is ported anyway, for the reason
/// §45.2 gives — *"porting them unfaithfully is how you get 99.83% of the
/// right answer"* — and because the capture latch above is a measurement that
/// would otherwise be deleted with the only body that records it. Re-stating
/// the row is then an eight-line edit rather than a re-derivation.
///
/// # Safety
///
/// The three pointer arrays are **device** arrays of `batch_count` device
/// pointers each, and cuBLAS does not consume them synchronously.
#[allow(clippy::too_many_arguments)]
pub unsafe fn batched_act_x_wt_bf16(
    handle: *mut c_void,
    act_ptrs_dev: *const *const c_void,
    w_ptrs_dev: *const *const c_void,
    y_ptrs_dev: *const *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    batch_count: i32,
    beta: f32,
) {
    if batch_count <= 0 {
        return;
    }
    let handle: cublasHandle_t = handle.cast::<cublasContext>();
    let alpha = 1.0f32;
    let grouped_key = dense_key(m, n, k, beta)
        ^ (batch_count as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let known = grouped_support(grouped_key);
    let capturing = cublas_stream(handle)
        // A handle that will not name its stream, or a stream that will not
        // report its capture state, is treated as CAPTURING — the C++'s two
        // `return true` early exits. Speculating is the dangerous direction.
        .map_or(true, |s| {
            capture_status(s)
                .is_none_or(|c| c != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone)
        });
    let try_grouped = known == Some(true) || (known.is_none() && !capturing);
    if try_grouped {
        let transa = [cublasOperation_t::CUBLAS_OP_T];
        let transb = [cublasOperation_t::CUBLAS_OP_N];
        let m_array = [n];
        let n_array = [m];
        let k_array = [k];
        let lda = [k];
        let ldb = [k];
        let ldc = [n];
        let group_size = [batch_count];
        // SAFETY: the caller's obligation. One group, so every array is one
        // element wide.
        let status = unsafe {
            cublasGemmGroupedBatchedEx(
                handle,
                transa.as_ptr(),
                transb.as_ptr(),
                m_array.as_ptr(),
                n_array.as_ptr(),
                k_array.as_ptr(),
                std::ptr::from_ref(&alpha).cast(),
                w_ptrs_dev,
                cudaDataType::CUDA_R_16BF,
                lda.as_ptr(),
                act_ptrs_dev,
                cudaDataType::CUDA_R_16BF,
                ldb.as_ptr(),
                std::ptr::from_ref(&beta).cast(),
                y_ptrs_dev,
                cudaDataType::CUDA_R_16BF,
                ldc.as_ptr(),
                1,
                group_size.as_ptr(),
                COMPUTE,
            )
        };
        if known.is_none() {
            store_grouped_support(grouped_key, status == cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return;
        }
    }
    // SAFETY: as above.
    let status = unsafe {
        cublasGemmBatchedEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w_ptrs_dev,
            cudaDataType::CUDA_R_16BF,
            k,
            act_ptrs_dev,
            cudaDataType::CUDA_R_16BF,
            k,
            std::ptr::from_ref(&beta).cast(),
            y_ptrs_dev,
            cudaDataType::CUDA_R_16BF,
            n,
            batch_count,
            COMPUTE,
            ALGO_TENSOR_OP,
        )
    };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        // cuBLAS reports INTERNAL_ERROR for anything it cannot explain,
        // including a CUDA error that was already sticky before the call.
        // Report the surrounding CUDA state so the message names the real
        // fault instead of the call that noticed it.
        let device = current_device();
        let pending = unsafe { cudaPeekAtLastError() };
        let pending_name = unsafe { CStr::from_ptr(cudaGetErrorName(pending)) }
            .to_string_lossy()
            .into_owned();
        let capture = cublas_stream(handle).map_or_else(
            || "unknown".to_owned(),
            |s| match capture_status(s) {
                Some(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive) => "active".to_owned(),
                Some(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated) => {
                    "INVALIDATED".to_owned()
                }
                Some(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) => "none".to_owned(),
                _ => "unknown".to_owned(),
            },
        );
        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;
        let _ = unsafe { cudaMemGetInfo(&raw mut free_bytes, &raw mut total_bytes) };
        panic!(
            "cuBLAS error ({status:?}): cublasGemmBatchedEx[bf16] M={m} N={n} K={k} \
             batch={batch_count} device={device} capture={capture} \
             pending_cuda={pending_name} free_mib={}",
            free_bytes >> 20
        );
    }
}

// ───────────────────────────────────────────────────────────────────────────
// The two cuBLAS entry points that came from `bind/service.rs`
// ───────────────────────────────────────────────────────────────────────────
//
// §5 step 5. They are here rather than in `x/gemm.rs` because `COMPUTE`,
// `ALGO_TENSOR_OP` and `check` are here, and a second copy of the compute
// type is a second place for it to drift — which is the exact defect
// `COMPUTE`'s own doc records having been measured.
//
// Both took a `&DispatchCtx` in the driver and take a handle here: there is
// no `DispatchCtx` in this crate, and the only field either read off it was
// `ctx.cublas`.

/// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
///
/// Ported from `gemm.cpp:1030-1058` (`gemm_bf16_out_fp32_impl`, reached
/// through the one-line `act_x_wt_bf16_out_fp32` at `:2327`), and moved here
/// from `driver-cuda/src/bind/service.rs:120`. Row-major
/// `y[M, N] = act[M, K] @ W[N, K]^T`, written column-major as the transpose,
/// which is where `OP_T/OP_N` and the `m=N, n=M` swap come from.
///
/// # Safety
///
/// `act` and `w` must address `M*K` and `N*K` live bf16 elements, `y` must
/// address `M*N` live floats, and all three must outlive the launch — which
/// is asynchronous on the handle's stream, so "outlive" ends at the next
/// synchronisation and not at this call's return.
pub unsafe fn act_x_wt_bf16_out_fp32(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: the caller's obligation, above. The handle is the engine's,
    // created once at boot by `driver_cuda::device::cublas`.
    let status = unsafe {
        cublasGemmEx(
            handle.cast::<cublasContext>(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w,
            cudaDataType::CUDA_R_16BF,
            k,
            act,
            cudaDataType::CUDA_R_16BF,
            k,
            std::ptr::from_ref(&beta).cast(),
            y.cast(),
            cudaDataType::CUDA_R_32F,
            n,
            COMPUTE,
            ALGO_TENSOR_OP,
        )
    };
    check(
        status,
        &format!("cublasGemmEx[bf16->fp32] M={m} N={n} K={k}"),
    );
}

/// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
///
/// Ported from `gemm.cpp:1242-1294` (`gemm_grouped_bf16_impl`, reached
/// through `grouped_act_x_wt_bf16` at `:1632`), and moved here from
/// `driver-cuda/src/bind/service.rs:181`. Every group shares `N`, `K` and
/// the three leading dimensions; only `M` differs, which is why the arrays
/// are filled from one scalar each and `n[]` from `m_array_host`.
///
/// `group_count <= 0` returns silently here and is a `Refusal::Empty` at
/// [`crate::x::gemm::grouped_act_x_wt_bf16`], which is the caller a bind and
/// `fire::lora` both reach. The guard is kept in both places because this is
/// the one that indexes the arrays.
///
/// # Safety
///
/// The three pointer arrays must be HOST arrays of `group_count` device
/// addresses (cuBLAS reads them on the host for the grouped form), and
/// `m_array_host` a host array of `group_count` row counts.
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
) {
    if group_count <= 0 {
        return;
    }
    let groups = group_count as usize;
    let transa = vec![cublasOperation_t::CUBLAS_OP_T; groups];
    let transb = vec![cublasOperation_t::CUBLAS_OP_N; groups];
    let m_arr = vec![n; groups];
    // SAFETY: `m_array_host` is a host array of `group_count` ints, per the
    // caller's obligation above.
    let n_arr = unsafe { std::slice::from_raw_parts(m_array_host, groups) }.to_vec();
    let k_arr = vec![k; groups];
    let lda = vec![k; groups];
    let ldb = vec![k; groups];
    let ldc = vec![n; groups];
    let group_size = vec![1i32; groups];
    let alpha = vec![1.0f32; groups];
    let beta_values = vec![beta; groups];

    // No `CUBLAS_COMPUTE_32F_FAST_16BF` attempt first: it has no algorithm
    // for these shapes, and a failed call inside a graph capture invalidates
    // the capture. (`gemm.cpp:1288`, kept verbatim because the reason is not
    // obvious from the code.)
    // SAFETY: every array above is `group_count` long and lives across the
    // call; cuBLAS reads them synchronously.
    let status = unsafe {
        cublasGemmGroupedBatchedEx(
            handle.cast::<cublasContext>(),
            transa.as_ptr(),
            transb.as_ptr(),
            m_arr.as_ptr(),
            n_arr.as_ptr(),
            k_arr.as_ptr(),
            alpha.as_ptr().cast(),
            w_ptrs_host,
            cudaDataType::CUDA_R_16BF,
            lda.as_ptr(),
            act_ptrs_host,
            cudaDataType::CUDA_R_16BF,
            ldb.as_ptr(),
            beta_values.as_ptr().cast(),
            y_ptrs_host,
            cudaDataType::CUDA_R_16BF,
            ldc.as_ptr(),
            group_count,
            group_size.as_ptr(),
            COMPUTE,
        )
    };
    check(
        status,
        &format!("cublasGemmGroupedBatchedEx[bf16] groups={group_count} N={n} K={k}"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tuning_cache.hpp:34`'s first mix, which pins the constant.
    #[test]
    fn tuning_hash_matches_the_cpp() {
        assert_eq!(tuning_hash(0, 0), 0x9e37_79b9_7f4a_7c15);
    }

    /// The disk stores `kind` as an integer, so the discriminants are an ABI.
    #[test]
    fn gemm_kind_discriminants_are_the_disk_format() {
        assert_eq!(GemmKind::GemmEx as i32, 0);
        assert_eq!(GemmKind::Lt as i32, 1);
        assert_eq!(GemmKind::Gemv as i32, 2);
        assert_eq!(GemmKind::from_i32(3), None);
        assert_eq!(GemmKind::from_i32(-1), None);
    }

    /// `beta` folds in as a bit, so 1.0 and 2.0 must key the same and 0.0
    /// must key differently.
    #[test]
    fn dense_key_reads_beta_as_a_bit() {
        assert_eq!(dense_key(1, 2, 3, 1.0), dense_key(1, 2, 3, 2.0));
        assert_ne!(dense_key(1, 2, 3, 0.0), dense_key(1, 2, 3, 1.0));
    }

    /// The five ladder rungs, each named for the checkpoint that measured it.
    #[test]
    fn the_shape_ladder_is_the_cpp_ladder() {
        assert_eq!(lt_algo_index_for_shape(12288, 1024), 2);
        assert_eq!(lt_algo_index_for_shape(200_000, 2048), 1);
        assert_eq!(lt_algo_index_for_shape(4096, 5120), 0);
        assert_eq!(lt_algo_index_for_shape(8192, 2048), 0);
        assert_eq!(lt_algo_index_for_shape(100_000, 2560), 0);
        assert_eq!(lt_algo_index_for_shape(4096, 4096), 5);
    }

    /// The large-hidden-size rung is a FAULT guard, not a speed one.
    #[test]
    fn lt_min_n_keeps_large_k_off_the_lt_path() {
        assert_eq!(lt_min_n(7168), 32768);
        assert_eq!(lt_min_n(1024), 12288);
        assert_eq!(lt_min_n(2048), 6144);
        assert_eq!(lt_min_n(2560), 12288);
    }

    /// A `beta` the GEMV cannot fold keeps it off the ballot; `M > 1` too.
    #[test]
    fn the_gemv_is_balloted_only_where_it_can_run() {
        let gemv = |m, beta| {
            dense_candidates(None, m, 4096, 4096, beta)
                .first()
                .map(|t| t.kind)
        };
        assert_eq!(gemv(1, 0.0), Some(GemmKind::Gemv));
        assert_eq!(gemv(1, 1.0), Some(GemmKind::Gemv));
        assert_eq!(gemv(1, 0.5), Some(GemmKind::GemmEx));
        assert_eq!(gemv(2, 0.0), Some(GemmKind::GemmEx));
    }
}
