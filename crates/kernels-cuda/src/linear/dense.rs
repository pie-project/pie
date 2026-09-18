#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::Error;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmEx, cublasGetVersion_v2,
    cublasHandle_t, cublasOperation_t, cublasSetStream_v2, cublasSetWorkspace_v2, cublasStatus_t,
    cudaDataType,
};
use cudarc::cublaslt::sys as lt;
use cudarc::runtime::sys::{
    cudaError, cudaEvent_t, cudaEventCreate, cudaEventDestroy, cudaEventElapsedTime,
    cudaEventRecordExternal, cudaEventRecordWithFlags, cudaEventSynchronize, cudaFree,
    cudaGetDevice, cudaGetLastError, cudaGraph_t, cudaGraphDestroy, cudaGraphExec_t,
    cudaGraphExecDestroy, cudaGraphInstantiateWithFlags, cudaGraphLaunch, cudaMalloc,
    cudaMemsetAsync, cudaStreamBeginCapture, cudaStreamCaptureMode, cudaStreamCaptureStatus,
    cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamEndCapture, cudaStreamNonBlocking,
    cudaStreamSynchronize,
};

use super::gemv::gemv_bf16;
use crate::jit::device::capture_status;
use crate::jit::{Ctx, refuse};

const fn dev(address: u64) -> *const c_void {
    address as usize as *const c_void
}

const fn dev_mut(address: u64) -> *mut c_void {
    address as usize as *mut c_void
}

fn clear_error() {
    let _ = unsafe { cudaGetLastError() };
}

fn current_device() -> i32 {
    let mut device: i32 = 0;
    let _ = unsafe { cudaGetDevice(&raw mut device) };
    device
}

#[derive(Clone, Copy, Debug)]
struct Call {
    act: u64,
    w: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tactic {
    Gemv,
    GemmEx,
    Lt(usize),
}

impl Tactic {
    fn encode(self) -> (i32, i32) {
        match self {
            Self::GemmEx => (0, 0),
            Self::Lt(algo) => (1, i32::try_from(algo).unwrap_or(0)),
            Self::Gemv => (2, 0),
        }
    }

    fn decode(kind: i32, algo: i32) -> Option<Self> {
        match kind {
            0 => Some(Self::GemmEx),
            1 => usize::try_from(algo).ok().map(Self::Lt),
            2 => Some(Self::Gemv),
            _ => None,
        }
    }
}

pub(crate) fn act_x_wt(
    ctx: &Ctx,
    op: &'static str,
    act: u64,
    w: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    if m <= 0 || n <= 0 || k <= 0 {
        return Ok(());
    }
    let handle: cublasHandle_t = ctx.cublas(op)?.cast::<cublasContext>();
    let stream = ctx.stream();
    let call = Call { act, w, y, m, n, k };
    let capturing = capture_status(stream);
    let (plan, tactic, lt_handle, want) = with_device(|device| {
        let plan = device.plan_for(call);
        let tactic = capturing
            .and_then(|status| device.tactic_for(handle, stream, plan.as_deref(), call, status));
        (plan, tactic, device.lt.handle, device.lt.workspace_bytes)
    });
    let (ws, ws_bytes) = match ctx.scratch_shared(op, LT_WORKSPACE, want) {
        Ok(ws) if !ws.is_null() => (ws, want),
        _ => (std::ptr::null_mut(), 0),
    };
    // SAFETY: `handle` is this context's live cuBLAS handle, bound to
    // `stream`; `ws` is a device slab of `ws_bytes` the context owns.
    unsafe {
        cublasSetWorkspace_v2(handle, ws, ws_bytes);
    }
    if let Some(tactic) = tactic
        && run_tactic(
            handle,
            stream,
            lt_handle,
            plan.as_deref(),
            tactic,
            call,
            ws,
            ws_bytes,
        )
    {
        return Ok(());
    }
    clear_error();

    if m == 1 && gemv_bf16(&unsafe { Ctx::on(stream) }, w, act, y, n, k).is_ok() {
        return Ok(());
    }
    if gemm_ex(
        handle,
        call,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    ) == cublasStatus_t::CUBLAS_STATUS_SUCCESS
    {
        return Ok(());
    }
    let status = gemm_ex(handle, call, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT);
    if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Ok(());
    }
    if let Some(plan) = plan {
        for heuristic in &plan.heuristics {
            if run_lt(
                lt_handle,
                &plan,
                &heuristic.algo,
                stream,
                call,
                ws,
                ws_bytes,
            ) {
                return Ok(());
            }
        }
    }
    Err(refuse(
        op,
        format!(
            "`cublasGemmEx` answered {status:?} at M={m} N={n} K={k}, and the \
             non-tensor-op and cuBLASLt retries failed too"
        ),
    ))
}

fn run_tactic(
    handle: cublasHandle_t,
    stream: *mut c_void,
    lt_handle: lt::cublasLtHandle_t,
    plan: Option<&LtPlan>,
    tactic: Tactic,
    call: Call,
    ws: *mut c_void,
    ws_bytes: usize,
) -> bool {
    match tactic {
        Tactic::Gemv => {
            call.m == 1
                && gemv_bf16(
                    &unsafe { Ctx::on(stream) },
                    call.w,
                    call.act,
                    call.y,
                    call.n,
                    call.k,
                )
                .is_ok()
        }
        Tactic::GemmEx => {
            gemm_ex(
                handle,
                call,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            ) == cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        Tactic::Lt(index) => {
            let (Some(plan), false) = (plan, lt_handle.is_null()) else {
                return false;
            };
            let Some(heuristic) = plan.heuristics.get(index) else {
                return false;
            };
            run_lt(lt_handle, plan, &heuristic.algo, stream, call, ws, ws_bytes)
        }
    }
}

fn gemm_ex(handle: cublasHandle_t, call: Call, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        cublasGemmEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            call.n,
            call.m,
            call.k,
            std::ptr::from_ref(&alpha).cast(),
            dev(call.w),
            cudaDataType::CUDA_R_16BF,
            call.k,
            dev(call.act),
            cudaDataType::CUDA_R_16BF,
            call.k,
            std::ptr::from_ref(&beta).cast(),
            dev_mut(call.y),
            cudaDataType::CUDA_R_16BF,
            call.n,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            algo,
        )
    }
}

const LT_WORKSPACE: &str = "linear.lt_workspace";

struct LtCtx {
    handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
}

unsafe impl Send for LtCtx {}

impl LtCtx {
    fn ensure(&mut self) -> bool {
        if self.handle.is_null()
            && unsafe { lt::cublasLtCreate(&raw mut self.handle) }
                != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
        {
            return false;
        }
        !self.handle.is_null()
    }
}

struct LtPlan {
    op_desc: lt::cublasLtMatmulDesc_t,
    a_desc: lt::cublasLtMatrixLayout_t,
    b_desc: lt::cublasLtMatrixLayout_t,
    c_desc: lt::cublasLtMatrixLayout_t,
    heuristics: Vec<lt::cublasLtMatmulHeuristicResult_t>,
}

unsafe impl Send for LtPlan {}

unsafe impl Sync for LtPlan {}

impl Drop for LtPlan {
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

const HEURISTICS: usize = 8;

fn build_lt_plan(
    lt_handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
    m: i32,
    n: i32,
    k: i32,
) -> Option<LtPlan> {
    struct Pref(lt::cublasLtMatmulPreference_t);
    impl Drop for Pref {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let _ = unsafe { lt::cublasLtMatmulPreferenceDestroy(self.0) };
            }
        }
    }

    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let mut plan = LtPlan {
        op_desc: std::ptr::null_mut(),
        a_desc: std::ptr::null_mut(),
        b_desc: std::ptr::null_mut(),
        c_desc: std::ptr::null_mut(),
        heuristics: Vec::new(),
    };
    if unsafe {
        lt::cublasLtMatmulDescCreate(
            &raw mut plan.op_desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType::CUDA_R_32F,
        )
    } != ok
    {
        return None;
    }
    for (attribute, operation) in [
        (
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            cublasOperation_t::CUBLAS_OP_T,
        ),
        (
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            cublasOperation_t::CUBLAS_OP_N,
        ),
    ] {
        if unsafe {
            lt::cublasLtMatmulDescSetAttribute(
                plan.op_desc,
                attribute,
                std::ptr::from_ref(&operation).cast(),
                std::mem::size_of::<cublasOperation_t>(),
            )
        } != ok
        {
            return None;
        }
    }
    for (desc, rows, cols, ld) in [
        (&raw mut plan.a_desc, k, n, k),
        (&raw mut plan.b_desc, k, m, k),
        (&raw mut plan.c_desc, n, m, n),
    ] {
        if unsafe {
            lt::cublasLtMatrixLayoutCreate(
                desc,
                lt::cudaDataType::CUDA_R_16BF,
                u64::try_from(rows).ok()?,
                u64::try_from(cols).ok()?,
                i64::from(ld),
            )
        } != ok
        {
            return None;
        }
    }
    let mut pref = Pref(std::ptr::null_mut());
    if unsafe { lt::cublasLtMatmulPreferenceCreate(&raw mut pref.0) } != ok {
        return None;
    }
    if unsafe {
        lt::cublasLtMatmulPreferenceSetAttribute(
            pref.0,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            std::ptr::from_ref(&workspace_bytes).cast(),
            std::mem::size_of::<usize>(),
        )
    } != ok
    {
        return None;
    }
    let mut heuristics: Vec<lt::cublasLtMatmulHeuristicResult_t> =
        vec![unsafe { core::mem::zeroed() }; HEURISTICS];
    let mut returned: i32 = 0;
    if unsafe {
        lt::cublasLtMatmulAlgoGetHeuristic(
            lt_handle,
            plan.op_desc,
            plan.a_desc,
            plan.b_desc,
            plan.c_desc,
            plan.c_desc,
            pref.0,
            i32::try_from(HEURISTICS).unwrap_or(8),
            heuristics.as_mut_ptr(),
            &raw mut returned,
        )
    } != ok
        || returned <= 0
    {
        return None;
    }
    heuristics.truncate((returned as usize).min(HEURISTICS));
    let mut augmented: Vec<lt::cublasLtMatmulHeuristicResult_t> = Vec::new();
    for heuristic in &heuristics {
        let mut supports: i32 = 0;
        let mut written: usize = 0;
        let asked = unsafe {
            lt::cublasLtMatmulAlgoCapGetAttribute(
                std::ptr::from_ref(&heuristic.algo),
                lt::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                std::ptr::from_mut(&mut supports).cast(),
                std::mem::size_of::<i32>(),
                &raw mut written,
            )
        };
        if asked != ok || supports == 0 {
            continue;
        }
        for split in [2i32, 4, 8] {
            let mut algo = heuristic.algo;
            let scheme = lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            let set_split = unsafe {
                lt::cublasLtMatmulAlgoConfigSetAttribute(
                    &raw mut algo,
                    lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    std::ptr::from_ref(&split).cast(),
                    std::mem::size_of::<i32>(),
                )
            };
            let set_scheme = unsafe {
                lt::cublasLtMatmulAlgoConfigSetAttribute(
                    &raw mut algo,
                    lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                    std::ptr::from_ref(&scheme).cast(),
                    std::mem::size_of::<lt::cublasLtReductionScheme_t>(),
                )
            };
            if set_split != ok || set_scheme != ok {
                continue;
            }
            let mut result: lt::cublasLtMatmulHeuristicResult_t = unsafe { core::mem::zeroed() };
            let checked = unsafe {
                lt::cublasLtMatmulAlgoCheck(
                    lt_handle,
                    plan.op_desc,
                    plan.a_desc,
                    plan.b_desc,
                    plan.c_desc,
                    plan.c_desc,
                    &raw const algo,
                    &raw mut result,
                )
            };
            if checked != ok || result.state != ok || result.workspaceSize > workspace_bytes {
                clear_error();
                continue;
            }
            result.algo = algo;
            augmented.push(result);
        }
    }
    heuristics.extend(augmented);
    plan.heuristics = heuristics;
    Some(plan)
}

fn run_lt(
    lt_handle: lt::cublasLtHandle_t,
    plan: &LtPlan,
    algo: &lt::cublasLtMatmulAlgo_t,
    stream: *mut c_void,
    call: Call,
    ws: *mut c_void,
    ws_bytes: usize,
) -> bool {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let status = unsafe {
        lt::cublasLtMatmul(
            lt_handle,
            plan.op_desc,
            std::ptr::from_ref(&alpha).cast(),
            dev(call.w),
            plan.a_desc,
            dev(call.act),
            plan.b_desc,
            std::ptr::from_ref(&beta).cast(),
            dev(call.y),
            plan.c_desc,
            dev_mut(call.y),
            plan.c_desc,
            std::ptr::from_ref(algo),
            ws,
            ws_bytes,
            stream.cast(),
        )
    };
    status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

struct Device {
    lt: LtCtx,
    plans: HashMap<(i32, i32, i32), Arc<LtPlan>>,
    chosen: HashMap<u64, Tactic>,
    seen: HashMap<u64, u32>,
    disk: DiskCache,
}

fn with_device<R>(f: impl FnOnce(&mut Device) -> R) -> R {
    static DEVICES: OnceLock<Mutex<HashMap<i32, Arc<Mutex<Device>>>>> = OnceLock::new();
    let device = {
        let mut map = DEVICES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(map.entry(current_device()).or_insert_with(|| {
            Arc::new(Mutex::new(Device {
                lt: LtCtx {
                    handle: std::ptr::null_mut(),
                    workspace_bytes: 8 * 1024 * 1024,
                },
                plans: HashMap::new(),
                chosen: HashMap::new(),
                seen: HashMap::new(),
                disk: DiskCache::new(),
            }))
        }))
    };
    let mut device = device
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&mut device)
}

impl Device {
    fn plan_for(&mut self, call: Call) -> Option<Arc<LtPlan>> {
        let (m, n, k) = (call.m, call.n, call.k);
        if !self.lt.ensure() {
            return None;
        }
        if let Some(plan) = self.plans.get(&(m, n, k)) {
            return Some(Arc::clone(plan));
        }
        let plan = Arc::new(build_lt_plan(self.lt.handle, self.lt.workspace_bytes, m, n, k)?);
        self.plans.insert((m, n, k), Arc::clone(&plan));
        Some(plan)
    }

    fn tactic_for(
        &mut self,
        handle: cublasHandle_t,
        stream: *mut c_void,
        plan: Option<&LtPlan>,
        call: Call,
        capturing: cudaStreamCaptureStatus,
    ) -> Option<Tactic> {
        let key = shape_key(call);
        if let Some(tactic) = self.chosen.get(&key) {
            return Some(*tactic);
        }
        if self.chosen.len() >= 1024 {
            return None;
        }
        if let Some(tactic) = self
            .disk
            .lookup(key)
            .and_then(|(kind, algo)| Tactic::decode(kind, algo))
        {
            self.chosen.insert(key, tactic);
            return Some(tactic);
        }
        if capturing != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone {
            return None;
        }
        if (call.m as usize) * (call.n as usize) * 2 > 256 * 1024 * 1024 {
            return None;
        }
        let seen = self.seen.entry(key).or_insert(0);
        *seen += 1;
        let tactic = tune(handle, stream, &self.lt, plan, call);
        self.disk.store(key, tactic.encode());
        self.chosen.insert(key, tactic);
        tracing::info!(
            m = call.m,
            n = call.n,
            k = call.k,
            ?tactic,
            "dense gemm shape tuned"
        );
        Some(tactic)
    }
}

fn shape_key(call: Call) -> u64 {
    let mut bytes = [0u8; 12];
    bytes[..4].copy_from_slice(&call.m.to_le_bytes());
    bytes[4..8].copy_from_slice(&call.n.to_le_bytes());
    bytes[8..].copy_from_slice(&call.k.to_le_bytes());
    crate::source::fnv1a64(&bytes)
}

fn tune(
    handle: cublasHandle_t,
    caller_stream: *mut c_void,
    lt: &LtCtx,
    plan: Option<&LtPlan>,
    call: Call,
) -> Tactic {
    let mut candidates = Vec::new();
    if call.m == 1 {
        candidates.push(Tactic::Gemv);
    }
    if let Some(plan) = plan {
        candidates.extend((0..plan.heuristics.len()).map(Tactic::Lt));
    }
    candidates.push(Tactic::GemmEx);

    let mut arena = TuneArena::empty();
    if !arena.init(handle, caller_stream, lt.workspace_bytes, call) {
        return candidates[0];
    }
    let mut timings = Vec::with_capacity(candidates.len());
    for &candidate in &candidates {
        let Some(ms) = arena
            .time(candidate, lt.handle, plan, call)
            .filter(|ms| *ms > 0.0)
        else {
            continue;
        };
        tracing::debug!(
            m = call.m,
            n = call.n,
            k = call.k,
            ?candidate,
            us = ms * 1000.0,
            "dense gemm bench"
        );
        timings.push((candidate, ms));
    }
    let fastest = timings
        .iter()
        .fold(f32::INFINITY, |best, &(_, ms)| best.min(ms));
    for &(candidate, ms) in &timings {
        if ms <= fastest / 0.98 {
            return candidate;
        }
    }
    candidates[0]
}

struct TuneArena {
    stream: *mut c_void,
    start: cudaEvent_t,
    stop: cudaEvent_t,
    handle: cublasHandle_t,
    caller_stream: *mut c_void,
    act: *mut c_void,
    y: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    flush: *mut c_void,
    flush_bytes: usize,
}

impl Drop for TuneArena {
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
            if !self.flush.is_null() {
                let _ = cudaFree(self.flush);
            }
            if !self.stream.is_null() {
                let _ = cudaStreamDestroy(self.stream.cast());
            }
        }
        clear_error();
    }
}

impl TuneArena {
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
            flush: std::ptr::null_mut(),
            flush_bytes: 0,
        }
    }

    fn init(
        &mut self,
        caller: cublasHandle_t,
        caller_stream: *mut c_void,
        workspace_bytes: usize,
        call: Call,
    ) -> bool {
        let act_bytes = ((call.m as usize) * (call.k as usize) * 2).max(1);
        let y_bytes = ((call.m as usize) * (call.n as usize) * 2).max(1);
        self.workspace_bytes = workspace_bytes;
        self.caller_stream = caller_stream;
        self.flush_bytes = flush_bytes();
        let acquired = unsafe {
            cudaMalloc(&raw mut self.act, act_bytes) == cudaError::cudaSuccess
                && cudaMalloc(&raw mut self.y, y_bytes) == cudaError::cudaSuccess
                && cudaMalloc(&raw mut self.workspace, self.workspace_bytes)
                    == cudaError::cudaSuccess
                && cudaMalloc(&raw mut self.flush, self.flush_bytes) == cudaError::cudaSuccess
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
        if unsafe { cudaStreamSynchronize(caller_stream.cast()) } != cudaError::cudaSuccess {
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

    fn time(
        &self,
        tactic: Tactic,
        lt_handle: lt::cublasLtHandle_t,
        plan: Option<&LtPlan>,
        call: Call,
    ) -> Option<f32> {
        let bench = Call {
            act: self.act.addr() as u64,
            y: self.y.addr() as u64,
            ..call
        };
        let fire = || {
            run_tactic(
                self.handle,
                self.stream,
                lt_handle,
                plan,
                tactic,
                bench,
                self.workspace,
                self.workspace_bytes,
            )
        };
        let stream = self.stream.cast();
        if !fire() {
            let _ = unsafe { cudaStreamSynchronize(stream) };
            clear_error();
            return None;
        }
        if unsafe { cudaStreamSynchronize(stream) } != cudaError::cudaSuccess {
            clear_error();
            return None;
        }

        let mut graph: cudaGraph_t = std::ptr::null_mut();
        let mut exec: cudaGraphExec_t = std::ptr::null_mut();
        let captured = unsafe {
            cudaStreamBeginCapture(stream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal)
                == cudaError::cudaSuccess
                && cudaMemsetAsync(self.flush, 0, self.flush_bytes, stream) == cudaError::cudaSuccess
                && cudaEventRecordWithFlags(self.start, stream, cudaEventRecordExternal)
                    == cudaError::cudaSuccess
                && fire()
                && cudaEventRecordWithFlags(self.stop, stream, cudaEventRecordExternal)
                    == cudaError::cudaSuccess
        };
        let ended =
            unsafe { cudaStreamEndCapture(stream, &raw mut graph) } == cudaError::cudaSuccess;
        if !captured || !ended || graph.is_null() {
            if !graph.is_null() {
                let _ = unsafe { cudaGraphDestroy(graph) };
            }
            let _ = unsafe { cudaStreamSynchronize(stream) };
            clear_error();
            return None;
        }
        let instantiated = unsafe { cudaGraphInstantiateWithFlags(&raw mut exec, graph, 0) }
            == cudaError::cudaSuccess;
        let _ = unsafe { cudaGraphDestroy(graph) };
        if !instantiated || exec.is_null() {
            clear_error();
            return None;
        }

        let mut best: Option<f32> = None;
        for round in 0..10 {
            if unsafe { cudaGraphLaunch(exec, stream) } != cudaError::cudaSuccess
                || unsafe { cudaEventSynchronize(self.stop) } != cudaError::cudaSuccess
            {
                let _ = unsafe { cudaStreamSynchronize(stream) };
                let _ = unsafe { cudaGraphExecDestroy(exec) };
                clear_error();
                return None;
            }
            if round < 3 {
                continue;
            }
            let mut ms = 0.0f32;
            if unsafe { cudaEventElapsedTime(&raw mut ms, self.start, self.stop) }
                != cudaError::cudaSuccess
            {
                let _ = unsafe { cudaGraphExecDestroy(exec) };
                clear_error();
                return None;
            }
            if best.is_none_or(|b| ms < b) {
                best = Some(ms);
            }
        }
        let _ = unsafe { cudaStreamSynchronize(stream) };
        let _ = unsafe { cudaGraphExecDestroy(exec) };
        best
    }
}

fn flush_bytes() -> usize {
    const FALLBACK: usize = 64 << 20;
    let mut device: i32 = 0;
    if unsafe { cudaGetDevice(&raw mut device) } != cudaError::cudaSuccess {
        clear_error();
        return FALLBACK;
    }
    crate::jit::device::properties(device)
        .and_then(|prop| usize::try_from(prop.l2CacheSize).ok())
        .filter(|&l2| l2 > 0)
        .map_or(FALLBACK, |l2| (3 * l2).max(FALLBACK))
}

struct DiskCache {
    signature: String,
    path: Option<PathBuf>,
    entries: HashMap<u64, (i32, i32)>,
}

impl DiskCache {
    fn new() -> Self {
        let mut cache = Self {
            signature: signature(),
            path: cache_path(),
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

    fn store(&mut self, key: u64, spelled: (i32, i32)) {
        let (kind, algo) = spelled;
        self.entries.insert(key, spelled);
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
        let empty = file.metadata().is_ok_and(|meta| meta.len() == 0);
        if empty {
            let _ = writeln!(file, "{}", self.signature);
        }
        let _ = writeln!(file, "{key:016x} {kind} {algo}");
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
            while let (Some(key), Some(kind), Some(algo)) =
                (fields.next(), fields.next(), fields.next())
            {
                let (Ok(key), Ok(kind), Ok(algo)) = (
                    u64::from_str_radix(key, 16),
                    kind.parse::<i32>(),
                    algo.parse::<i32>(),
                ) else {
                    break;
                };
                self.entries.insert(key, (kind, algo));
            }
        } else {
            self.entries.clear();
            let _ = std::fs::remove_file(&path);
        }
    }
}

fn cache_path() -> Option<PathBuf> {
    Some(crate::disk::dir(crate::disk::GEMM_ALGOS)?.join("dense.txt"))
}

fn signature() -> String {
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
    let name = unsafe { CStr::from_ptr(prop.name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    format!(
        "# pie-dense-gemm v6 sm{}{} cublas={version} dev={name}",
        prop.major, prop.minor
    )
}
