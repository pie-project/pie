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
    cudaError, cudaEvent_t, cudaEventCreate, cudaEventDestroy,
    cudaEventElapsedTime, cudaEventRecord, cudaEventSynchronize, cudaFree, cudaGetDevice,
    cudaGetErrorName, cudaGetLastError, cudaMalloc, cudaMemGetInfo, cudaMemsetAsync,
    cudaPeekAtLastError, cudaStreamCaptureStatus, cudaStreamCreateWithFlags, cudaStreamDestroy,
    cudaStreamIsCapturing, cudaStreamNonBlocking, cudaStreamSynchronize,
};

// `get_device_properties` STOOD HERE, and it stood in `driver-cuda`'s
// `bind/quant_gemm.rs` as well -- two private copies of one `#[cfg]` pair,
// byte-identical in the body, because the two callers were in different
// crates and neither could see the other. The quantised router's descent put
// them in one directory and the duplicate had to answer for itself. It is
// `jit::device::properties` now, which is the module whose subject is the
// device facilities a body needs that are not a launch.

use super::gemv::gemv_bf16;

/// `cublasComputeType_t bf16_compute_type() { return CUBLAS_COMPUTE_32F; }`.
const COMPUTE: cublasComputeType_t = cublasComputeType_t::CUBLAS_COMPUTE_32F;

/// `CUBLAS_GEMM_DEFAULT_TENSOR_OP` — the pin every call starts with.
const ALGO_TENSOR_OP: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

/// `CUBLAS_GEMM_DEFAULT` — the un-pinned retry. See [`act_x_wt_bf16`]'s
const ALGO_DEFAULT: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT;

/// `gemm.cpp:92` — `cublaslt_bf16_workspace_bytes()`, 64 MiB.
const LT_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

/// `gemm.cpp:85` — throw on a non-success status.
/// `check`, for the cuBLASLt handle's own status type.
fn check_lt(status: lt::cublasStatus_t, what: &str) {
    assert!(
        status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "cuBLASLt error ({status:?}): {what}"
    );
}

fn check(status: cublasStatus_t, what: &str) {
    assert!(status == cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS error ({status:?}): {what}");
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
fn cublas_stream(handle: cublasHandle_t) -> Option<*mut c_void> {
    let mut stream: cudarc::cublas::sys::cudaStream_t = std::ptr::null_mut();
    if unsafe { cublasGetStream_v2(handle, &raw mut stream) }
        == cublasStatus_t::CUBLAS_STATUS_SUCCESS
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

/// The cuBLASLt handle and shared workspace for one device.
struct Bf16LtCtx {
    handle: lt::cublasLtHandle_t,
    workspace: *mut c_void,
    workspace_bytes: usize,
}

// SAFETY: `handle` and `workspace` are device-side resources reached only
unsafe impl Send for Bf16LtCtx {}

impl Bf16LtCtx {
    /// `gemm.cpp:130` — `ensure()`. Idempotent; both halves are separately
    fn ensure(&mut self) {
        if self.handle.is_null() {
            check_lt(unsafe { lt::cublasLtCreate(&raw mut self.handle) }, "cublasLtCreate");
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

/// The descriptors for one `(M, N, K)`, plus every algorithm the heuristic
struct Bf16LtPlan {
    op_desc: lt::cublasLtMatmulDesc_t,
    a_desc: lt::cublasLtMatrixLayout_t,
    b_desc: lt::cublasLtMatrixLayout_t,
    c_desc: lt::cublasLtMatrixLayout_t,
    heuristics: Vec<lt::cublasLtMatmulHeuristicResult_t>,
}

// SAFETY: the four descriptors are opaque cuBLASLt handles, never
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

/// Which returned cuBLASLt heuristic a shape prefers.
fn lt_algo_index_for_shape(n: i32, k: i32) -> i32 {
    if k < 2048 && n >= 12288 {
        return 2;
    }
    if k == 2048 && n >= 200_000 {
        return 1;
    }
    if k == 5120 {
        return 0;
    }
    if k == 2048 && n >= 6144 {
        return 0;
    }
    if k == 2560 && n >= 100_000 {
        return 0;
    }
    5
}

/// The narrowest output width at which the Lt ladder is worth taking.
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
const LT_MIN_K: i32 = 1024;
/// See [`LT_MIN_K`]. M=1 is the GEMV's shape, not Lt's.
const LT_MIN_M: i32 = 2;
/// See [`LT_MIN_K`].
const LT_MAX_N: i32 = 0;

/// One `cublasLtMatmul`. `true` iff the status was success.
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
    let stream = cublas_stream(cublas_handle).unwrap_or(std::ptr::null_mut());
    run_lt_algo(plan, std::ptr::from_ref(algo), stream, act, w, y, beta, workspace)
}

/// `gemm.cpp:296` — create the descriptors for a shape and ask cuBLASLt which
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
                std::mem::size_of::<cublasOperation_t>(),
            )
        };
    }
    if st == ok {
        st = unsafe {
            lt::cublasLtMatmulDescSetAttribute(
                plan.op_desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                std::ptr::from_ref(&transb).cast(),
                std::mem::size_of::<cublasOperation_t>(),
            )
        };
    }
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

    let mut heuristics: [lt::cublasLtMatmulHeuristicResult_t; 8] = unsafe { core::mem::zeroed() };
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
        let (first, last) = if pass == 0 { (begin, begin + 1) } else { (0, returned) };
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

/// A candidate must beat the incumbent by this much to displace it; anything
const TACTIC_MARGIN: f32 = 0.98;

/// `gemm.cpp:466` — which family a tactic names. The integers are ON DISK, in
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmKind {
    GemmEx = 0,
    Lt = 1,
    Gemv = 2,
}

impl GemmKind {
    /// The disk's integer back to a kind. `None` for anything outside the
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
    // `&& t.kind != GemmKind::Gemv` IS GONE, AND DELETING `gemv_bf16`'s
    // `bias` IS WHAT REQUIRED IT.
    //
    // The exemption was honest while it lasted: every OTHER tactic refused
    // a non-null bias because it had nowhere to put one, and `Gemv` was
    // exempt because it did -- it forwarded the pointer straight into
    // `gemv_bf16`'s fourth argument. *"Refuses a non-null bias for every
    // tactic BUT this one, which is why the parameter exists"*, as `gemv.rs`
    // put it.
    //
    // §3.9 deleted that argument, so the reason for the exemption is gone
    // and only the exemption would have been left. THAT IS THE DANGEROUS
    // RESIDUE OF A DELETION AND IT IS NOT THE DELETION'S OWN FILE. A
    // non-null bias with `GemmKind::Gemv` would have gone from FORWARDED to
    // SILENTLY DROPPED, still returning `true`, still reporting the tactic
    // ran -- and the caller would have got an unbiased result reported as a
    // biased one.
    //
    // Behaviour-neutral today, which is the only reason it is a one-line
    // change rather than a bug report: both call sites (`dense.rs`'s tuner
    // and its capture path) pass `std::ptr::null()`, so no branch changes
    // for any input the tree produces. It is the FUTURE caller this stops,
    // and the deletion is what made it possible to stop cheaply -- with the
    // parameter gone there is no longer any way to express a bias here, so
    // refusing one is now the whole truth rather than a policy.
    if !bias.is_null() {
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
            // SAFETY: the tuner's arena, live across the launch.
            let ctx = unsafe { crate::jit::Ctx::on(stream) };
            // `Weight(w)`, because the leg's parameter is `Weight<..>` now
            // (`gemv.rs:53`) and this is the call the mark describes: `w`
            // arrives here as `act_x_wt_bf16`'s named bank and goes on as
            // one.
            gemv_bf16(
                &ctx,
                kernels::Const { v: w },
                kernels::In { ptr: act, rows: 0, width: k },
                kernels::Out { ptr: y, rows: 0, width: n },
                beta,
            )
            .is_ok()
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

/// Everything the probes need that must not end up in a captured graph.
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
    fn init(&mut self, caller: cublasHandle_t, m: i32, n: i32, k: i32) -> bool {
        let act_bytes = (m as usize) * (k as usize) * 2;
        let y_bytes = (m as usize) * (n as usize) * 2;
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
        let filled = unsafe {
            cudaMemsetAsync(self.act, 0x3C, act_bytes, self.stream.cast()) == cudaError::cudaSuccess
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
    let fire = || {
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
fn dense_candidates(
    plan: Option<&Bf16LtPlan>,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Vec<DenseTactic> {
    let mut out = Vec::new();
    if m == 1 && (beta == 0.0 || beta == 1.0) {
        out.push(DenseTactic { kind: GemmKind::Gemv, algo: 0 });
    }
    out.push(DenseTactic { kind: GemmKind::GemmEx, algo: 0 });
    if let Some(plan) = plan {
        let preferred = lt_algo_index_for_shape(n, k);
        let count = plan.heuristics.len() as i32;
        if preferred < count {
            out.push(DenseTactic { kind: GemmKind::Lt, algo: preferred });
        }
        for i in 0..count {
            if i == preferred {
                continue;
            }
            out.push(DenseTactic { kind: GemmKind::Lt, algo: i });
        }
    }
    out
}

/// `tuning_cache.hpp:34` — mixes `v` into hash `h`.
#[must_use]
pub const fn tuning_hash(h: u64, v: u64) -> u64 {
    h ^ (v.wrapping_add(0x9e37_79b9_7f4a_7c15).wrapping_add(h << 6).wrapping_add(h >> 2))
}

/// `gemm.cpp:713` — the cache key for a dense shape.
fn dense_key(m: i32, n: i32, k: i32, beta: f32) -> u64 {
    let mut h = 0u64;
    h = tuning_hash(h, m as u64);
    h = tuning_hash(h, n as u64);
    h = tuning_hash(h, k as u64);
    h = tuning_hash(h, u64::from(beta != 0.0));
    h
}

/// The C++'s `TuningCache`, for this one file's use.
struct DiskCache {
    signature: String,
    path: Option<PathBuf>,
    entries: HashMap<u64, (i32, i32)>,
}

impl DiskCache {
    fn new(name: &str, signature: String) -> Self {
        let path = cache_path(name);
        let mut cache = Self { signature, path, entries: HashMap::new() };
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
        let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) else {
            return;
        };
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
            let mut fields = lines.flat_map(str::split_whitespace);
            while let (Some(k), Some(a), Some(b)) = (fields.next(), fields.next(), fields.next()) {
                let (Ok(k), Ok(a), Ok(b)) =
                    (u64::from_str_radix(k, 16), a.parse::<i32>(), b.parse::<i32>())
                else {
                    break;
                };
                self.entries.insert(k, (a, b));
            }
        } else {
            self.entries.clear();
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// `cache_root.hpp`'s derivation, carried because that header is deleted too:
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
fn dense_cache_signature() -> String {
    let mut device: i32 = 0;
    if unsafe { cudaGetDevice(&raw mut device) } != cudaError::cudaSuccess {
        clear_error();
        return String::new();
    }
    let Some(prop) = crate::jit::device::properties(device) else {
        clear_error();
        return String::new();
    };
    let mut version: i32 = 0;
    let _ = unsafe { cublasGetVersion_v2(std::ptr::null_mut(), &raw mut version) };
    let name = unsafe { CStr::from_ptr(prop.name.as_ptr()) }.to_string_lossy().into_owned();
    format!("# pie-dense-gemm v1 sm{}{} cublas={version} dev={name}", prop.major, prop.minor)
}

/// The tactic file's basename. Unchanged from the C++, so a machine that has
const CACHE_FILE: &str = "dense_gemm.txt";

/// `gemm.cpp:693` — the per-device memo, the recurrence counter and the disk.
struct DenseGemmTuner {
    chosen: HashMap<u64, DenseTactic>,
    seen: HashMap<u64, i32>,
    disk: DiskCache,
}

/// Ceiling on how many shapes will ever be measured, so a workload with an
const MAX_TUNED_SHAPES: usize = 1024;

/// The per-device tuner map.
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
fn dense_tactic_for(
    caller: cublasHandle_t,
    w: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    capturing: cudaStreamCaptureStatus,
) -> (Option<Arc<Bf16LtPlan>>, Option<DenseTactic>) {
    const MAX_TUNE_OUTPUT_BYTES: usize = 256 * 1024 * 1024;
    if (m as usize) * (n as usize) * 2 > MAX_TUNE_OUTPUT_BYTES {
        return (None, None);
    }

    let plan = lt_plan_for(m, n, k);
    let key = dense_key(m, n, k, beta);

    let tactic = with_tuner(|tuner| {
        if let Some(t) = tuner.chosen.get(&key) {
            return Some(*t);
        }
        if tuner.chosen.len() >= MAX_TUNED_SHAPES {
            return None;
        }
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
#[must_use]
pub fn dense_tactic_is_gemv(m: i32, n: i32, k: i32, beta: f32) -> bool {
    let key = dense_key(m, n, k, beta);
    with_tuner(|tuner| tuner.chosen.get(&key).is_some_and(|t| t.kind == GemmKind::Gemv))
}

/// One line per dense bf16 GEMM naming the shape, the capture status and the
fn path_trace_take() -> bool {
    use std::sync::atomic::{AtomicI32, Ordering};
    static ON: OnceLock<bool> = OnceLock::new();
    static BUDGET: AtomicI32 = AtomicI32::new(40000);
    let on = *ON.get_or_init(|| {
        std::env::var("PIE_GEMM_PATH_TRACE").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
    });
    if !on {
        return false;
    }
    BUDGET.fetch_sub(1, Ordering::Relaxed) > 0
}

/// `gemm::act_x_wt_bf16` — `y[M, N] = act[M, K] @ W[N, K]^T + beta * y`.
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

    if m == 1 && beta == 0.0 {
        // SAFETY: the caller's matrices, live across the launch.
        if let Some(stream) = cublas_stream(handle)
            && gemv_bf16(
                &unsafe { crate::jit::Ctx::on(stream) },
                kernels::Const { v: w },
                kernels::In { ptr: act, rows: 0, width: k },
                kernels::Out { ptr: y, rows: 0, width: n },
                0.0,
            )
            .is_ok()
        {
            if path_trace {
                eprintln!("[gemm-path]   -> gemv");
            }
            return;
        }
    }

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

/// Whether `cublasGemmGroupedBatchedEx` can serve a shape, per device.
fn grouped_support(key: u64) -> Option<bool> {
    static KNOWN: OnceLock<Mutex<HashMap<(i32, u64), bool>>> = OnceLock::new();
    let map = KNOWN
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("grouped-batched support map poisoned");
    map.get(&(current_device(), key)).copied()
}

/// Records what [`grouped_support`] could not answer. `emplace`, not
fn store_grouped_support(key: u64, supported: bool) {
    static KNOWN: OnceLock<Mutex<HashMap<(i32, u64), bool>>> = OnceLock::new();
    let mut map = KNOWN
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("grouped-batched support map poisoned");
    map.entry((current_device(), key)).or_insert(supported);
}

/// `gemm::batched_act_x_wt_bf16` — per-batch `act`/`W`/`y` pointers, all
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
    let grouped_key =
        dense_key(m, n, k, beta) ^ (batch_count as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let known = grouped_support(grouped_key);
    let capturing = cublas_stream(handle).map_or(true, |s| {
        capture_status(s).is_none_or(|c| c != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone)
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
        let device = current_device();
        let pending = unsafe { cudaPeekAtLastError() };
        let pending_name =
            unsafe { CStr::from_ptr(cudaGetErrorName(pending)) }.to_string_lossy().into_owned();
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

/// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
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
    check(status, &format!("cublasGemmEx[bf16->fp32] M={m} N={n} K={k}"));
}

/// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
///
/// # Safety
///
/// The three pointer arrays must be **DEVICE** arrays of `group_count` device
/// addresses, and `m_array_host` a **host** array of `group_count` row counts.
/// The split is real and is cuBLAS's: the scalar arrays (`m`/`n`/`k`, the
/// leading dimensions, alpha and beta, the group sizes) are read on the host,
/// and the `Aarray`/`Barray`/`Carray` pointer arrays are dereferenced on the
/// device like every other batched form.
///
/// This said HOST for the pointer arrays, and said cuBLAS "reads them on the
/// host for the grouped form". It does not. Handing it host addresses is
/// `cudaErrorIllegalAddress` at the next synchronize on CUDA 13 / sm_120 --
/// measured, not inferred, and the same call is clean when the arrays are
/// staged to the device. It evidently went unpunished on whatever card
/// recorded `tests/oracle/gemm_service/golden.txt`, since those rows hash to
/// real products rather than to the untouched output buffer, which is what
/// let a wrong sentence survive as a correct-looking one.
///
/// The only caller that matters already gets this right: `lora.rs` passes a
/// slot in `staged.ptr_slab`, whose own doc calls it "the device pointer
/// slab". The names were the bug, not the behaviour.
pub unsafe fn grouped_act_x_wt_bf16(
    handle: *mut c_void,
    act_ptrs_dev: *const *const c_void,
    w_ptrs_dev: *const *const c_void,
    y_ptrs_dev: *const *mut c_void,
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
    let n_arr = unsafe { std::slice::from_raw_parts(m_array_host, groups) }.to_vec();
    let k_arr = vec![k; groups];
    let lda = vec![k; groups];
    let ldb = vec![k; groups];
    let ldc = vec![n; groups];
    let group_size = vec![1i32; groups];
    let alpha = vec![1.0f32; groups];
    let beta_values = vec![beta; groups];

    // SAFETY: every array above is `group_count` long and lives across the
    let status = unsafe {
        cublasGemmGroupedBatchedEx(
            handle.cast::<cublasContext>(),
            transa.as_ptr(),
            transb.as_ptr(),
            m_arr.as_ptr(),
            n_arr.as_ptr(),
            k_arr.as_ptr(),
            alpha.as_ptr().cast(),
            w_ptrs_dev,
            cudaDataType::CUDA_R_16BF,
            lda.as_ptr(),
            act_ptrs_dev,
            cudaDataType::CUDA_R_16BF,
            ldb.as_ptr(),
            beta_values.as_ptr().cast(),
            y_ptrs_dev,
            cudaDataType::CUDA_R_16BF,
            ldc.as_ptr(),
            group_count,
            group_size.as_ptr(),
            COMPUTE,
        )
    };
    check(status, &format!("cublasGemmGroupedBatchedEx[bf16] groups={group_count} N={n} K={k}"));
}
