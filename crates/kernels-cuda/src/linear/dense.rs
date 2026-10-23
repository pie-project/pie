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

use super::gemv::{gemm_narrow_bf16, gemm_rows_bf16, gemv_bf16, rows_cluster};
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
    bias: u64,
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
    /// `gemm_narrow_bf16`: a block a row, at most eight columns.
    Narrow,
    /// `gemm_rows_bf16`: a few rows in one launch, warps streaming the
    /// weight's rows.
    Rows,
}

impl Tactic {
    fn encode(self) -> (i32, i32) {
        match self {
            Self::GemmEx => (0, 0),
            Self::Lt(algo) => (1, i32::try_from(algo).unwrap_or(0)),
            Self::Gemv => (2, 0),
            Self::Narrow => (3, 0),
            Self::Rows => (4, 0),
        }
    }

    fn decode(kind: i32, algo: i32) -> Option<Self> {
        match kind {
            0 => Some(Self::GemmEx),
            1 => usize::try_from(algo).ok().map(Self::Lt),
            2 => Some(Self::Gemv),
            3 => Some(Self::Narrow),
            4 => Some(Self::Rows),
            _ => None,
        }
    }
}

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
    let handle: cublasHandle_t = ctx.cublas(op)?.cast::<cublasContext>();
    let stream = ctx.stream();
    let call = Call {
        act,
        w,
        bias,
        y,
        m,
        n,
        k,
    };
    let capturing = capture_status(stream);
    let tuning = ctx.tuning();
    let (plan, tactic, lt_handle, want) = with_device(|device| {
        device.configure(
            (tuning.gemm_workspace_mb as usize) << 20,
            tuning.gemm_heuristics as usize,
            tuning.gemm_narrow_n,
            tuning.gemm_split_launch_ns,
            tuning.gemm_rows_m,
            tuning.gemm_split_launch_from,
        );
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
    // A cuBLASLt tactic lands the bias from its epilogue where the profile
    // says (`kernel.gemm.bias_epilogue`); one it refuses runs bare and the
    // bias is added after.
    if let (Some(Tactic::Lt(index)), true, Some(plan)) = (
        tactic,
        bias != 0 && tuning.gemm_bias_epilogue,
        plan.as_deref(),
    ) && let Some(heuristic) = plan.heuristics.get(index)
        && !lt_handle.is_null()
        && run_lt_bias(lt_handle, plan, &heuristic.algo, stream, call, ws, ws_bytes)
    {
        return Ok(true);
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
            tuning,
        )
    {
        return Ok(matches!(tactic, Tactic::Gemv | Tactic::Rows) && bias != 0);
    }
    clear_error();

    if m == 1
        && gemv_bf16(
            &unsafe { Ctx::on(stream) }.with_tuning(tuning),
            w,
            act,
            bias,
            y,
            n,
            k,
        )
        .is_ok()
    {
        return Ok(bias != 0);
    }
    if gemm_ex(
        handle,
        call,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    ) == cublasStatus_t::CUBLAS_STATUS_SUCCESS
    {
        return Ok(false);
    }
    let status = gemm_ex(handle, call, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT);
    if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Ok(false);
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
                return Ok(false);
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
    tuning: crate::tuning::Tuning,
) -> bool {
    // The kernels' own launches read this dispatch's decisions (`kernel.pdl`,
    // `kernel.gemv.bf16`), not a bare stream's defaults.
    let pdl = tuning.pdl;
    match tactic {
        Tactic::Gemv => {
            call.m == 1
                && gemv_bf16(
                    &unsafe { Ctx::on(stream) }.with_tuning(tuning),
                    call.w,
                    call.act,
                    call.bias,
                    call.y,
                    call.n,
                    call.k,
                )
                .is_ok()
        }
        Tactic::Rows => gemm_rows_bf16(
            &unsafe { Ctx::on(stream) }.with_tuning(tuning),
            call.w,
            call.act,
            call.bias,
            call.y,
            call.m,
            call.n,
            call.k,
            pdl,
        )
        .is_ok(),
        Tactic::Narrow => gemm_narrow_bf16(
            &unsafe { Ctx::on(stream) }.with_tuning(tuning),
            call.w,
            call.act,
            call.bias,
            call.y,
            call.m,
            call.n,
            call.k,
        )
        .is_ok(),
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

/// `y = act . w^T` over bf16 operands landing f32, one cuBLAS call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn act_x_wt_f32(
    ctx: &Ctx,
    op: &'static str,
    act: u64,
    w: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    let handle: cublasHandle_t = ctx.cublas(op)?.cast::<cublasContext>();
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: `handle` is this context's live cuBLAS handle; the operands are
    // device rows of the stated extents.
    let status = unsafe {
        cublasGemmEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            std::ptr::from_ref(&alpha).cast(),
            dev(w),
            cudaDataType::CUDA_R_16BF,
            k,
            dev(act),
            cudaDataType::CUDA_R_16BF,
            k,
            std::ptr::from_ref(&beta).cast(),
            dev_mut(y),
            cudaDataType::CUDA_R_32F,
            n,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(crate::jit::refuse(
            op,
            format!("cuBLAS answered {status:?}"),
        ))
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

fn build_lt_plan(
    lt_handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
    heuristics_wanted: usize,
    m: i32,
    n: i32,
    k: i32,
    in_place: bool,
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
    // A split's partials are summed in f32 or not split at all: the in-place
    // and output-type schemes add them in bf16, and over a cancelling K that
    // is off by several percent (sm89 offers them at a hundred rows).
    let reductions = lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32;
    if unsafe {
        lt::cublasLtMatmulPreferenceSetAttribute(
            pref.0,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            std::ptr::from_ref(&reductions).cast(),
            std::mem::size_of::<u32>(),
        )
    } != ok
    {
        return None;
    }
    let mut heuristics: Vec<lt::cublasLtMatmulHeuristicResult_t> =
        vec![unsafe { core::mem::zeroed() }; heuristics_wanted.max(1)];
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
            i32::try_from(heuristics_wanted.max(1)).unwrap_or(8),
            heuristics.as_mut_ptr(),
            &raw mut returned,
        )
    } != ok
        || returned <= 0
    {
        return None;
    }
    heuristics.truncate((returned as usize).min(heuristics_wanted.max(1)));
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
        // Where a split's launch is charged, also each split reduced in
        // place: one launch, its partials serialized through the output.
        let schemes: &[lt::cublasLtReductionScheme_t] = if in_place {
            &[
                lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_INPLACE,
            ]
        } else {
            &[lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE]
        };
        for (split, &scheme) in [2i32, 4, 8]
            .into_iter()
            .flat_map(|split| schemes.iter().map(move |scheme| (split, scheme)))
        {
            let mut algo = heuristic.algo;
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

/// [`run_lt`] with the bias (bf16, one a column of `y`) added in the
/// epilogue, through a descriptor of its own for the call. False where the
/// algorithm refuses the epilogue.
fn run_lt_bias(
    lt_handle: lt::cublasLtHandle_t,
    plan: &LtPlan,
    algo: &lt::cublasLtMatmulAlgo_t,
    stream: *mut c_void,
    call: Call,
    ws: *mut c_void,
    ws_bytes: usize,
) -> bool {
    struct Desc(lt::cublasLtMatmulDesc_t);
    impl Drop for Desc {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let _ = unsafe { lt::cublasLtMatmulDescDestroy(self.0) };
            }
        }
    }
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let mut desc = Desc(std::ptr::null_mut());
    if unsafe {
        lt::cublasLtMatmulDescCreate(
            &raw mut desc.0,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType::CUDA_R_32F,
        )
    } != ok
    {
        return false;
    }
    let set = |attribute: lt::cublasLtMatmulDescAttributes_t,
               value: *const c_void,
               bytes: usize| unsafe {
        lt::cublasLtMatmulDescSetAttribute(desc.0, attribute, value, bytes) == ok
    };
    let (ta, tb) = (
        cublasOperation_t::CUBLAS_OP_T,
        cublasOperation_t::CUBLAS_OP_N,
    );
    let epilogue = lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
    let bias_ptr = dev(call.bias);
    let bias_type = lt::cudaDataType::CUDA_R_16BF;
    let size = std::mem::size_of::<cublasOperation_t>();
    if !(set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        std::ptr::from_ref(&ta).cast(),
        size,
    ) && set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        std::ptr::from_ref(&tb).cast(),
        size,
    ) && set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
        std::ptr::from_ref(&epilogue).cast(),
        std::mem::size_of::<lt::cublasLtEpilogue_t>(),
    ) && set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        std::ptr::from_ref(&bias_ptr).cast(),
        std::mem::size_of::<*const c_void>(),
    ) && set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
        std::ptr::from_ref(&bias_type).cast(),
        std::mem::size_of::<lt::cudaDataType>(),
    )) {
        return false;
    }
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let status = unsafe {
        lt::cublasLtMatmul(
            lt_handle,
            desc.0,
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
    if status != ok {
        clear_error();
        return false;
    }
    true
}

struct Device {
    lt: LtCtx,
    heuristics: usize,
    /// `kernel.gemm.narrow_n`: products this many columns or fewer may time
    /// the narrow kernel.
    narrow_n: u32,
    /// `kernel.gemm.split_launch_ns`: what a K-splitting algorithm's own
    /// reduce launch costs a captured step past its timing, or 0.
    split_ns: u32,
    /// `kernel.gemm.split_launch_from`: the fewest rows charged `split_ns`.
    split_from: u32,
    /// `kernel.gemm.rows_m`: products of 2 to this many rows may time the
    /// few-rows kernel.
    rows_m: u32,
    plans: HashMap<(i32, i32, i32), Arc<LtPlan>>,
    chosen: HashMap<u64, Tactic>,
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
                heuristics: 8,
                narrow_n: 0,
                split_ns: 0,
                split_from: 0,
                rows_m: 0,
                plans: HashMap::new(),
                chosen: HashMap::new(),
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
    /// Lay a load's GEMM settings on this device (`kernel.gemm.workspace_mb`,
    /// `kernel.gemm.heuristics`): plans and tactics chosen under other
    /// settings are dropped, since an algorithm may need more workspace.
    fn configure(
        &mut self,
        workspace_bytes: usize,
        heuristics: usize,
        narrow_n: u32,
        split_ns: u32,
        rows_m: u32,
        split_from: u32,
    ) {
        if self.lt.workspace_bytes != workspace_bytes
            || self.heuristics != heuristics
            || self.narrow_n != narrow_n
            || self.split_ns != split_ns
            || self.rows_m != rows_m
            || self.split_from != split_from
        {
            self.split_from = split_from;
            self.lt.workspace_bytes = workspace_bytes;
            self.heuristics = heuristics;
            self.narrow_n = narrow_n;
            self.split_ns = split_ns;
            self.rows_m = rows_m;
            self.plans.clear();
            self.chosen.clear();
        }
    }

    /// What a split algorithm's reduce launch is charged at `m` rows.
    fn charge(&self, m: i32) -> u32 {
        if i64::from(m) >= i64::from(self.split_from) {
            self.split_ns
        } else {
            0
        }
    }

    fn plan_for(&mut self, call: Call) -> Option<Arc<LtPlan>> {
        let (m, n, k) = (call.m, call.n, call.k);
        if !self.lt.ensure() {
            return None;
        }
        if let Some(plan) = self.plans.get(&(m, n, k)) {
            return Some(Arc::clone(plan));
        }
        let plan = Arc::new(build_lt_plan(
            self.lt.handle,
            self.lt.workspace_bytes,
            self.heuristics,
            m,
            n,
            k,
            self.charge(m) > 0,
        )?);
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
        // A product below `split_from` rows keys, plans and times as with
        // no charge, so its tactic is what it was before one.
        let charge = self.charge(call.m);
        let key = shape_key(
            call,
            self.lt.workspace_bytes,
            self.heuristics,
            self.narrow_n,
            charge,
        );
        // A shape the few-rows kernel may serve keys apart from the one it
        // may not, and one it serves over a cluster apart again.
        let key = if rows_tried(call, self.rows_m) {
            let cluster = rows_cluster(call.n, call.k);
            key ^ 0x726f_7773_0000_0000 ^ u64::from(self.rows_m) ^ (u64::from(cluster > 1) << 8)
        } else {
            key
        };
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
        let tactic = tune(
            handle,
            stream,
            &self.lt,
            plan,
            call,
            self.narrow_n,
            charge,
            self.rows_m,
        );
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

/// Whether the tuner times the few-rows kernel for `call`
/// (`kernel.gemm.rows_m`).
fn rows_tried(call: Call, rows_m: u32) -> bool {
    call.m >= 2 && call.m <= rows_m.min(8) as i32 && call.k % 8 == 0
}

fn shape_key(
    call: Call,
    workspace_bytes: usize,
    heuristics: usize,
    narrow_n: u32,
    split_ns: u32,
) -> u64 {
    let mut bytes = [0u8; 28];
    bytes[..4].copy_from_slice(&call.m.to_le_bytes());
    bytes[4..8].copy_from_slice(&call.n.to_le_bytes());
    bytes[8..12].copy_from_slice(&call.k.to_le_bytes());
    bytes[12..20].copy_from_slice(&(workspace_bytes as u64).to_le_bytes());
    bytes[20..].copy_from_slice(&(heuristics as u64).to_le_bytes());
    // A shape the narrow kernel may serve keys apart from the one it may
    // not, and one timed with a split's launch charged from one timed
    // without; every other key is what it was.
    let narrow = narrow_n > 0 && call.n <= narrow_n as i32;
    let whole = split_ns > 0;
    if !narrow && !whole {
        return crate::source::fnv1a64(&bytes);
    }
    let mut wider = [0u8; 36];
    wider[..28].copy_from_slice(&bytes);
    if narrow {
        wider[28..32].copy_from_slice(&narrow_n.to_le_bytes());
    }
    if !whole {
        return crate::source::fnv1a64(&wider[..32]);
    }
    wider[32..].copy_from_slice(&split_ns.to_le_bytes());
    crate::source::fnv1a64(&wider)
}

fn tune(
    handle: cublasHandle_t,
    caller_stream: *mut c_void,
    lt: &LtCtx,
    plan: Option<&LtPlan>,
    call: Call,
    narrow_n: u32,
    split_ns: u32,
    rows_m: u32,
) -> Tactic {
    let mut candidates = Vec::new();
    if call.m == 1 {
        candidates.push(Tactic::Gemv);
    }
    if let Some(plan) = plan {
        candidates.extend((0..plan.heuristics.len()).map(Tactic::Lt));
    }
    candidates.push(Tactic::GemmEx);
    // Last, so a tie keeps cuBLAS.
    if call.n <= narrow_n as i32 && call.k % 8 == 0 {
        candidates.push(Tactic::Narrow);
    }
    if rows_tried(call, rows_m) {
        candidates.push(Tactic::Rows);
    }

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
        // A captured step pays a split's reduce its own launch, which a
        // lone timing hides (`kernel.gemm.split_launch_ns`).
        // `cublasGemmEx` picks its own algorithm, a split one as likely, so
        // it is charged too.
        let two = match candidate {
            Tactic::Lt(i) => plan.is_some_and(|p| splits_of(&p.heuristics[i].algo) > 1),
            Tactic::GemmEx => true,
            Tactic::Gemv | Tactic::Narrow | Tactic::Rows => false,
        };
        let charge = if two { split_ns as f32 * 1e-6 } else { 0.0 };
        timings.push((candidate, ms + charge, charge));
    }
    let fastest = timings
        .iter()
        .fold(f32::INFINITY, |best, &(_, ms, _)| best.min(ms));
    // One pass leaves a plateau of near-equal tactics ordered by noise: the
    // first few within reach of the fastest are timed again in turns (a
    // drift in the device's speed falling on each alike), and of those whose
    // medians tie the earliest (cuBLAS's own ranking) is kept.
    let mut finals: Vec<(Tactic, f32, Vec<f32>)> = timings
        .iter()
        .filter(|&&(_, ms, _)| ms <= fastest * FINAL_REACH)
        .take(FINALISTS)
        .map(|&(candidate, ms, charge)| (candidate, charge, vec![ms]))
        .collect();
    if finals.len() > 1 {
        for _ in 0..FINAL_ROUNDS {
            for (candidate, charge, samples) in &mut finals {
                if let Some(ms) = arena.time(*candidate, lt.handle, plan, call) {
                    samples.push(ms + *charge);
                }
            }
        }
    }
    let medians: Vec<(Tactic, f32)> = finals
        .into_iter()
        .map(|(candidate, _, mut samples)| {
            samples.sort_by(f32::total_cmp);
            (candidate, samples[samples.len() / 2])
        })
        .collect();
    let best = medians.iter().fold(f32::INFINITY, |b, &(_, ms)| b.min(ms));
    medians
        .iter()
        .find(|&&(_, ms)| ms <= best * FINAL_TIE)
        .map_or(candidates[0], |&(candidate, _)| candidate)
}

/// A tactic within this factor of the fastest's first timing is a finalist,
/// the first this many such, each timed this many times more; finalists'
/// medians within the last factor of the best tie.
const FINAL_REACH: f32 = 1.05;
const FINALISTS: usize = 8;
const FINAL_ROUNDS: usize = 4;
const FINAL_TIE: f32 = 1.02;

/// An algorithm's K splits (1 when it does not split K).
fn splits_of(algo: &lt::cublasLtMatmulAlgo_t) -> i32 {
    let mut splits: i32 = 1;
    let mut written: usize = 0;
    let asked = unsafe {
        lt::cublasLtMatmulAlgoConfigGetAttribute(
            std::ptr::from_ref(algo),
            lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            std::ptr::from_mut(&mut splits).cast(),
            std::mem::size_of::<i32>(),
            &raw mut written,
        )
    };
    if asked != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        clear_error();
        return 1;
    }
    splits.max(1)
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
    /// The evictions' span alone, timed once.
    alone: std::cell::Cell<Option<f32>>,
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
            alone: std::cell::Cell::new(None),
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
                // Once outside a capture, its kernel loaded here.
                && self.evict()
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

    /// Reads the flush slab through a cuBLAS product of one row (its rows,
    /// each dot landing past them), so the timed product starts on an L2 of
    /// clean lines another kernel's weights left, as in a step: a memset
    /// leaves it dirty, every eviction a write-back the product pays.
    fn evict(&self) -> bool {
        const K: usize = 8192;
        let rows = self.flush_bytes / (2 * K + 2);
        let at = self.flush.addr() as u64;
        let call = Call {
            act: at,
            w: at,
            bias: 0,
            y: at + (rows * 2 * K) as u64,
            m: 1,
            n: rows as i32,
            k: K as i32,
        };
        gemm_ex(
            self.handle,
            call,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        ) == cublasStatus_t::CUBLAS_STATUS_SUCCESS
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
            bias: 0,
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
                crate::tuning::Tuning::DEFAULT,
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
        // A decode's product streams a weight no cache holds and runs a
        // few microseconds, which the difference of two spans of evictions
        // cannot resolve (one read 0.2 us for a 7 MB weight, A100): one
        // eviction, then a run of products back to back, each over a
        // different slice of the slab, timed alone.
        if call.m == 1
            && let Some(ms) = self.streamed(tactic, lt_handle, plan, bench)
        {
            return Some(ms);
        }
        // The product's share of a run of evicted products, past the
        // evictions alone: the device's timer steps some 2 us, coarser than
        // the tactics' gaps, so one product a span reads as a tie.
        let alone = match self.alone.get() {
            Some(alone) => alone,
            None => {
                let alone = self.span(&|| true)?;
                self.alone.set(Some(alone));
                alone
            }
        };
        let with = self.span(&fire)?;
        Some((with - alone).max(0.0) / REPS as f32)
    }

    /// The least of a few timings (ms) of one product, as up to `STREAMED`
    /// of them run back to back after one eviction, each reading its weight
    /// from its own slice of the slab's first half (the eviction leaves its
    /// last third cached); `None` when that half holds fewer than four.
    fn streamed(
        &self,
        tactic: Tactic,
        lt_handle: lt::cublasLtHandle_t,
        plan: Option<&LtPlan>,
        bench: Call,
    ) -> Option<f32> {
        let weight = (bench.n as usize) * (bench.k as usize) * 2;
        let cold = self.flush_bytes / 2;
        let runs = (cold / weight.next_multiple_of(256).max(1)).min(STREAMED);
        if weight == 0 || runs < 4 {
            return None;
        }
        let stride = (cold / runs) & !255;
        let at = self.flush.addr() as u64;
        let stream = self.stream.cast();
        let mut graph: cudaGraph_t = std::ptr::null_mut();
        let mut exec: cudaGraphExec_t = std::ptr::null_mut();
        let captured = unsafe {
            cudaStreamBeginCapture(
                stream,
                cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
            ) == cudaError::cudaSuccess
                && self.evict()
                && cudaEventRecordWithFlags(self.start, stream, cudaEventRecordExternal)
                    == cudaError::cudaSuccess
                && (0..runs).all(|i| {
                    let call = Call {
                        w: at + (i * stride) as u64,
                        ..bench
                    };
                    run_tactic(
                        self.handle,
                        self.stream,
                        lt_handle,
                        plan,
                        tactic,
                        call,
                        self.workspace,
                        self.workspace_bytes,
                        crate::tuning::Tuning::DEFAULT,
                    )
                })
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
        for round in 0..ROUNDS {
            let mut ms = 0.0f32;
            let timed = unsafe {
                cudaGraphLaunch(exec, stream) == cudaError::cudaSuccess
                    && cudaEventSynchronize(self.stop) == cudaError::cudaSuccess
                    && (round == 0
                        || cudaEventElapsedTime(&raw mut ms, self.start, self.stop)
                            == cudaError::cudaSuccess)
            };
            if !timed {
                let _ = unsafe { cudaStreamSynchronize(stream) };
                let _ = unsafe { cudaGraphExecDestroy(exec) };
                clear_error();
                return None;
            }
            if round > 0 && best.is_none_or(|b| ms < b) {
                best = Some(ms);
            }
        }
        let _ = unsafe { cudaStreamSynchronize(stream) };
        let _ = unsafe { cudaGraphExecDestroy(exec) };
        best.map(|ms| ms / runs as f32)
    }

    /// The least of a few timings (ms) of `REPS` evictions, each followed by
    /// `body`, captured as one graph.
    fn span(&self, body: &dyn Fn() -> bool) -> Option<f32> {
        let stream = self.stream.cast();
        let mut graph: cudaGraph_t = std::ptr::null_mut();
        let mut exec: cudaGraphExec_t = std::ptr::null_mut();
        let captured = unsafe {
            cudaStreamBeginCapture(
                stream,
                cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
            ) == cudaError::cudaSuccess
                && cudaEventRecordWithFlags(self.start, stream, cudaEventRecordExternal)
                    == cudaError::cudaSuccess
                && (0..REPS).all(|_| self.evict() && body())
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
        for round in 0..ROUNDS {
            if unsafe { cudaGraphLaunch(exec, stream) } != cudaError::cudaSuccess
                || unsafe { cudaEventSynchronize(self.stop) } != cudaError::cudaSuccess
            {
                let _ = unsafe { cudaStreamSynchronize(stream) };
                let _ = unsafe { cudaGraphExecDestroy(exec) };
                clear_error();
                return None;
            }
            if round == 0 {
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

/// Evicted products a timed span, and the spans timed (the first a warm-up).
const REPS: usize = 4;
const ROUNDS: usize = 3;

/// Products of one row a streamed span runs back to back.
const STREAMED: usize = 16;

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
        "# pie-dense-gemm v9 sm{}{} cublas={version} dev={name}",
        prop.major, prop.minor
    )
}
