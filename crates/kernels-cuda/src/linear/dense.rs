//! Dense host programs: `cublasGemmEx`, the `cublasLtMatmul` plan cache, the
//! per-weight family algorithm that makes a row's arithmetic independent of
//! the fire's width, and the per-shape autotuner (gemv/GemmEx/Lt heuristics)
//! for shapes no family covers, cached in memory and on disk by device and
//! cuBLAS version.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::Error;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmEx, cublasGetVersion_v2,
    cublasHandle_t, cublasOperation_t, cublasSetStream_v2, cublasStatus_t, cudaDataType,
};
use cudarc::cublaslt::sys as lt;
use cudarc::runtime::sys::{
    cudaError, cudaEvent_t, cudaEventCreate, cudaEventDestroy, cudaEventElapsedTime,
    cudaEventRecord, cudaEventSynchronize, cudaFree, cudaGetDevice, cudaGetLastError, cudaMalloc,
    cudaMemcpyAsync, cudaMemcpyKind, cudaMemsetAsync, cudaStreamCaptureStatus,
    cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamNonBlocking, cudaStreamSynchronize,
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

/// One projection call: `y = act x w^T`, addresses and extents together so
/// the tuner can re-aim the same call at its synthetic operands.
#[derive(Clone, Copy, Debug)]
struct Call {
    act: u64,
    w: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
}

/// One way to run the projection, in ladder order: the skinny jit kernel,
/// plain GemmEx, or one of the Lt heuristics by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tactic {
    Gemv,
    GemmEx,
    Lt(usize),
}

impl Tactic {
    /// The two-int disk spelling.
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

// ─── the entry and its ladder ───────────────────────────────────────────────

/// `y = act x w^T`, bf16 throughout. An empty projection (any extent zero)
/// is a silent no-op — a conditioned batch may legitimately land nothing,
/// and a refusal here would kill the whole fire under graph capture.
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

    // The settle is eager-only: it blocks the host, which would take a
    // capture down with it. A captured first sighting leaves the family
    // unsettled; the next eager fire settles it.
    let eager = matches!(
        capturing,
        Some(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone)
    );
    let (plan, tactic, lt_handle, want) = with_device(|device| {
        let plan = device.plan_for(call, stream, eager);
        // The tuner races tactics per (m, n, k); a shape with a family
        // algorithm never reaches it, since that would answer one lane
        // differently depending on who else rode in the same fire.
        let tuned = plan.as_deref().is_none_or(|plan| plan.invariant.is_none());
        let tactic = capturing
            .filter(|_| tuned)
            .and_then(|status| device.tactic_for(handle, stream, plan.as_deref(), call, status));
        (plan, tactic, device.lt.handle, device.lt.workspace_bytes)
    });
    // The Lt workspace is a per-(arena, name, stream) slab, not a
    // per-device buffer: two concurrent matmuls sharing one workspace would
    // silently corrupt each other. Absent is not fatal — Lt takes a null
    // workspace at zero bytes.
    let (ws, ws_bytes) = match ctx.scratch(op, LT_WORKSPACE, want) {
        Ok(ws) if !ws.is_null() => (ws, want),
        _ => (std::ptr::null_mut(), 0),
    };
    // The batch-invariant rung, above every other: the weight's own
    // algorithm, walking the contraction unsplit regardless of this fire's
    // row count, so a lane's logits depend only on the lane.
    if let Some(plan) = plan.as_deref()
        && let Some((algo, needs)) = plan.invariant
        && ws_bytes >= needs
        && run_lt(lt_handle, plan, &algo, stream, call, ws, ws_bytes)
    {
        return Ok(());
    }
    clear_error();
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

    // The untuned ladder: the skinny kernel for a single row, GemmEx in its
    // tensor-op then plain spelling, then any Lt heuristic. First success
    // wins; a shape walks this only until the tuner has seen it twice.
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

/// One attempt under a chosen tactic; `false` sends the caller down the
/// ladder. `stream` must be the stream `handle` is bound to — the tuner
/// rebinds both to its private bench.
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

// ─── the cuBLASLt plumbing ──────────────────────────────────────────────────

/// The name the Lt workspace slab is keyed by, in the same namespace every
/// other scratch entry uses.
const LT_WORKSPACE: &str = "linear.lt_workspace";

/// The Lt handle (created on first use) and the workspace byte count — the
/// single source the heuristic preference, tuner and fire path all read. The
/// buffer itself lives in a per-`(arena, stream)` slab, not here.
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

/// One shape's Lt descriptors and the heuristics Lt offered for it, best
/// first by Lt's own estimate. Cached per shape, shared out under `Arc`.
struct LtPlan {
    op_desc: lt::cublasLtMatmulDesc_t,
    a_desc: lt::cublasLtMatrixLayout_t,
    b_desc: lt::cublasLtMatrixLayout_t,
    c_desc: lt::cublasLtMatrixLayout_t,
    heuristics: Vec<lt::cublasLtMatmulHeuristicResult_t>,
    /// The weight's algorithm accepted for this M, and the workspace it
    /// needs. `None` means the family had nothing this M would take, and the
    /// caller walks the ladder below it. Bytes are carried so the rung is
    /// taken or skipped before the call, never failed inside a graph capture.
    invariant: Option<(lt::cublasLtMatmulAlgo_t, usize)>,
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

/// How many heuristics a plan keeps by default: the ladder's last rungs, and
/// the tuner's `Tactic::Lt` indices, which the disk cache spells by position.
const HEURISTICS: usize = 8;

fn build_lt_plan(
    lt_handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
    m: i32,
    n: i32,
    k: i32,
) -> Option<LtPlan> {
    build_lt_plan_wide(lt_handle, workspace_bytes, m, n, k, HEURISTICS)
}

/// The same plan, asking Lt for `wanted` heuristics rather than [`HEURISTICS`].
/// The extra ones are for [`settle_small`]'s race, not the ladder: Lt's
/// estimated order is not a measurement, so the actual winner can sit well
/// past [`HEURISTICS`].
fn build_lt_plan_wide(
    lt_handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
    m: i32,
    n: i32,
    k: i32,
    wanted: usize,
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
        invariant: None,
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
    // In cuBLAS's column-major eyes: w^T is A (k x n), act is B (k x m),
    // y is C (n x m).
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
    // Masking out in-place reduction keeps a fixed shape summing the same
    // way run to run.
    let deterministic: u32 = (lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_MASK as u32)
        & !(lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_INPLACE as u32);
    if unsafe {
        lt::cublasLtMatmulPreferenceSetAttribute(
            pref.0,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            std::ptr::from_ref(&deterministic).cast(),
            std::mem::size_of::<u32>(),
        )
    } != ok
    {
        return None;
    }
    let wanted = wanted.clamp(1, 64);
    let mut heuristics: Vec<lt::cublasLtMatmulHeuristicResult_t> =
        vec![unsafe { core::mem::zeroed() }; wanted];
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
            i32::try_from(wanted).unwrap_or(8),
            heuristics.as_mut_ptr(),
            &raw mut returned,
        )
    } != ok
        || returned <= 0
    {
        return None;
    }
    heuristics.truncate((returned as usize).min(wanted));
    plan.heuristics = heuristics;
    Some(plan)
}

// ─── the batch-invariant rung ───────────────────────────────────────────────

/// The row count every shape family's `(n, k)` algorithm is chosen at; a
/// performance choice, not a correctness one. Set large: a wide tile reused
/// on a narrow (weight-bound) fire wastes little, but the converse re-reads
/// weights.
const FAMILY_ROWS: i32 = 128;

/// The algorithms a `(n, k)` family offers, in Lt's own estimated order, each
/// pinned to a split-free K walk: at small M, Lt's heuristic may split the
/// contraction across CTAs for parallelism, and a differently-split walk sums
/// bf16 in a different order (different bits). Pinning
/// `CUBLASLT_ALGO_CONFIG_SPLITK_NUM = 1` makes one CTA own the whole walk.
fn family_algos(
    lt_handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
    n: i32,
    k: i32,
) -> Vec<lt::cublasLtMatmulAlgo_t> {
    let Some(plan) = build_lt_plan(lt_handle, workspace_bytes, FAMILY_ROWS, n, k) else {
        return Vec::new();
    };
    plan.heuristics
        .iter()
        .map(|heuristic| split_free(heuristic.algo))
        .filter(|algo| {
            accepted(lt_handle, &plan, std::slice::from_ref(algo), workspace_bytes).is_some()
        })
        .collect()
}

/// One algorithm, pinned to a split-free K walk. See [`family_algos`].
fn split_free(mut algo: lt::cublasLtMatmulAlgo_t) -> lt::cublasLtMatmulAlgo_t {
    let splits: i32 = 1;
    let scheme = lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as u32;
    // A setter that refuses means the algorithm has no such knob, and so
    // never splits anyway.
    let _ = unsafe {
        lt::cublasLtMatmulAlgoConfigSetAttribute(
            &raw mut algo,
            lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            std::ptr::from_ref(&splits).cast(),
            std::mem::size_of::<i32>(),
        )
    };
    let _ = unsafe {
        lt::cublasLtMatmulAlgoConfigSetAttribute(
            &raw mut algo,
            lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
            std::ptr::from_ref(&scheme).cast(),
            std::mem::size_of::<u32>(),
        )
    };
    algo
}

/// The first family algorithm cuBLASLt accepts for this M and workspace. A
/// tile is a rectangle over `(m, n)`, so the family's best is not always
/// available at every M; `None` sends the caller down the untuned ladder.
fn accepted(
    lt_handle: lt::cublasLtHandle_t,
    plan: &LtPlan,
    family: &[lt::cublasLtMatmulAlgo_t],
    workspace_bytes: usize,
) -> Option<(lt::cublasLtMatmulAlgo_t, usize)> {
    for algo in family {
        let mut checked: lt::cublasLtMatmulHeuristicResult_t = unsafe { core::mem::zeroed() };
        let status = unsafe {
            lt::cublasLtMatmulAlgoCheck(
                lt_handle,
                plan.op_desc,
                plan.a_desc,
                plan.b_desc,
                plan.c_desc,
                plan.c_desc,
                std::ptr::from_ref(algo),
                &raw mut checked,
            )
        };
        if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
            && checked.workspaceSize <= workspace_bytes
        {
            return Some((*algo, checked.workspaceSize));
        }
    }
    None
}

/// The row counts below [`FAMILY_ROWS`] that a padded fire actually lands
/// on (the deployment's lattice, rounded); the rungs [`settle_small`]
/// measures the crossover at. A fire on some other lattice is still served
/// conservatively, since the ceiling is a row count and the test is `m <=
/// ceiling`.
const SMALL_RUNGS: [i32; 4] = [8, 16, 32, 64];

/// The narrowest of [`SMALL_RUNGS`], where the candidate pool is raced: the
/// smallest fire this tree ever hands cuBLASLt, so a candidate that can't
/// win here has nothing to offer at any width.
const SMALL_ROWS: i32 = SMALL_RUNGS[0];

/// One weight's algorithms: the family, and the second one that serves below
/// [`FAMILY_ROWS`]. cuBLASLt makes no contractual promise that two
/// algorithms sum a contraction in the same order, so [`settle_small`]
/// checks it empirically against this weight's real bytes and admits a
/// candidate only if it lands identical bits; a driver change that breaks
/// the assumption just falls the family back to one algorithm.
struct Family {
    /// Algorithms accepted at [`FAMILY_ROWS`], in Lt's own order: the
    /// primary, taken by every fire at or above `FAMILY_ROWS`.
    algos: Vec<lt::cublasLtMatmulAlgo_t>,
    /// The algorithm serving narrow fires (`m <= ceiling`), and that
    /// ceiling; bit-identical to the primary at every rung it serves. See
    /// [`settle_small`].
    small: Option<(lt::cublasLtMatmulAlgo_t, i32)>,
    /// Whether the settle has run; `false` means the first sighting was a
    /// capturing fire, which cannot bench.
    settled: bool,
}

/// A bf16 pattern with a spread of exponents, so fp32 accumulation of `k`
/// products is inexact — a constant fill would make every product identical
/// and pass the identity check blind regardless of K order.
fn inexact_bf16(len: usize) -> Vec<u16> {
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            let unit = f32::from(bits as u16) / 65536.0 - 0.5;
            let binade = ((bits >> 16) % 7) as i32 - 3;
            let value = unit * (2.0_f32).powi(binade) / 3.0;
            // bf16 is the top half of the f32, rounded to nearest even.
            let raw = value.to_bits();
            ((raw.wrapping_add(0x7fff + ((raw >> 16) & 1))) >> 16) as u16
        })
        .collect()
}

/// The second algorithm for this weight, or `None`. A candidate must land
/// the primary's bits AND be faster at [`SMALL_ROWS`]; the reference is the
/// primary's own output over the first `SMALL_ROWS` rows of a `FAMILY_ROWS`
/// fire, which share inputs with the narrow fire and differ only in how many
/// other rows rode along.
fn settle_small(
    lt: &LtCtx,
    caller_stream: *mut c_void,
    n: i32,
    k: i32,
    w: u64,
    algos: &[lt::cublasLtMatmulAlgo_t],
) -> Option<(lt::cublasLtMatmulAlgo_t, i32)> {
    if w == 0 || algos.is_empty() || n <= 0 || k <= 0 {
        return None;
    }
    let workspace_bytes = lt.workspace_bytes;
    // Candidate pool: Lt's estimate at SMALL_ROWS (deeper than the ladder
    // keeps) plus the family itself, whose second/third entries are often
    // the small-M winner.
    let narrow = build_lt_plan_wide(lt.handle, workspace_bytes, SMALL_ROWS, n, k, 16)?;
    let wide = build_lt_plan(lt.handle, workspace_bytes, FAMILY_ROWS, n, k)?;
    let lone = build_lt_plan(lt.handle, workspace_bytes, 1, n, k)?;
    let above = SMALL_RUNGS[1..]
        .iter()
        .map(|&m| build_lt_plan(lt.handle, workspace_bytes, m, n, k))
        .collect::<Option<Vec<_>>>()?;
    let ladder: Vec<&LtPlan> = std::iter::once(&narrow).chain(above.iter()).collect();

    let (primary_wide, _) = accepted(lt.handle, &wide, algos, workspace_bytes)?;
    let primary: Vec<Option<lt::cublasLtMatmulAlgo_t>> = ladder
        .iter()
        .map(|plan| accepted(lt.handle, plan, algos, workspace_bytes).map(|(algo, _)| algo))
        .collect();

    let mut bench = PairBench::empty();
    if !bench.init(caller_stream, workspace_bytes, n, k) {
        return None;
    }
    let rows = (SMALL_ROWS as usize) * (n as usize);
    let reference = bench.rows(lt.handle, &wide, &primary_wide, w, rows)?;
    // The primary must already agree with itself across widths, or there is
    // nothing for a second algorithm to be identical to.
    let anchor = primary[0]?;
    if bench.rows(lt.handle, &narrow, &anchor, w, rows)? != reference {
        tracing::warn!(
            n,
            k,
            "the family gemm algorithm does not agree with itself across widths on this \
             device; no second algorithm admitted"
        );
        return None;
    }
    // What the family costs at every rung: the bar a candidate must clear to
    // serve a rung, and the price of every rung it doesn't.
    let standing: Vec<Option<f32>> = ladder
        .iter()
        .zip(&primary)
        .map(|(plan, algo)| algo.and_then(|algo| bench.time(lt.handle, plan, &algo, w)))
        .collect();
    let bar = standing[0]?;

    // Pass one: who is even in the race, at SMALL_ROWS (the narrowest width
    // a fire lands on). A candidate that can't win there has nothing to
    // offer at any rung, since the primary only gets better as M climbs
    // toward FAMILY_ROWS.
    let mut divergent = 0_u32;
    let mut survivors: Vec<(lt::cublasLtMatmulAlgo_t, f32)> = Vec::new();
    let mut fastest = bar;
    let pool = narrow
        .heuristics
        .iter()
        .map(|heuristic| split_free(heuristic.algo))
        .chain(algos.iter().copied());
    for candidate in pool {
        if accepted(
            lt.handle,
            &narrow,
            std::slice::from_ref(&candidate),
            workspace_bytes,
        )
        .is_none()
        {
            continue;
        }
        let Some(ms) = bench.time(lt.handle, &narrow, &candidate, w) else {
            continue;
        };
        // Only an actual improvement is worth the (expensive) identity check.
        if ms >= fastest * 0.98 {
            continue;
        }
        if bench.rows(lt.handle, &narrow, &candidate, w, rows)? != reference {
            divergent += 1;
            continue;
        }
        fastest = ms;
        survivors.push((candidate, ms));
    }

    // Pass two: the ceiling is measured, per family, at every rung a padded
    // fire actually lands on. A fixed ceiling was a bug: a candidate can be
    // faster at SMALL_ROWS and markedly slower by FAMILY_ROWS. The candidate
    // serves up to the last rung it's still winning at, walking up from
    // SMALL_ROWS and stopping at the first loss (contiguous from the bottom,
    // so no rung is served by an algorithm whose curve isn't understood).
    // The winner is the one with the lowest total across the whole ladder,
    // counting the family's own time for every rung it declined.
    let mut best: Option<(lt::cublasLtMatmulAlgo_t, usize, f32)> = None;
    for (candidate, ms) in survivors {
        let mut top = 0_usize;
        let mut total = ms;
        for rung in 1..ladder.len() {
            let (Some(theirs), true) = (standing[rung], primary[rung].is_some()) else {
                break;
            };
            if accepted(
                lt.handle,
                ladder[rung],
                std::slice::from_ref(&candidate),
                workspace_bytes,
            )
            .is_none()
            {
                break;
            }
            let Some(ours) = bench.time(lt.handle, ladder[rung], &candidate, w) else {
                break;
            };
            if ours >= theirs * 0.98 {
                break;
            }
            // It will serve this rung, so it owes this rung's bits too.
            if bench.rows(lt.handle, ladder[rung], &candidate, w, rows)? != reference {
                divergent += 1;
                break;
            }
            top = rung;
            total += ours;
        }
        // Every declined rung costs the family's own time.
        total += standing[top + 1..]
            .iter()
            .filter_map(|rung| *rung)
            .sum::<f32>();
        if best.is_none_or(|(_, _, standing)| total < standing) {
            best = Some((candidate, top, total));
        }
    }
    if divergent > 0 {
        tracing::warn!(
            n,
            k,
            divergent,
            "faster small-batch gemm algorithms exist for this weight but do not land the \
             family algorithm's bits; the family algorithm serves those widths"
        );
    }
    let (algo, top, _) = best?;
    let ceiling = SMALL_RUNGS[top];

    // Belt: check one row below the range and the first rung the family
    // takes back — a ceiling is a claim about a boundary, and a boundary
    // neither side of which was checked is a guess.
    let overshoot: &LtPlan = if top + 1 < ladder.len() {
        ladder[top + 1]
    } else {
        &wide
    };
    for (plan, count) in [(&lone, n as usize), (overshoot, rows)] {
        if accepted(
            lt.handle,
            plan,
            std::slice::from_ref(&algo),
            workspace_bytes,
        )
        .is_none()
        {
            continue;
        }
        if bench.rows(lt.handle, plan, &algo, w, count)? != reference[..count] {
            tracing::warn!(
                n,
                k,
                ceiling,
                "the faster small-batch gemm algorithm for this weight agrees with the \
                 family's over the rungs it would serve and not on the boundary of them; \
                 the family algorithm serves every width"
            );
            return None;
        }
    }
    tracing::info!(
        n,
        k,
        ceiling,
        us = fastest * 1000.0,
        was_us = bar * 1000.0,
        "a second gemm algorithm serves this weight up to its measured ceiling, \
         bit-identical to the family's"
    );
    Some((algo, ceiling))
}

/// The settle's bench: a stream of its own, timing events, one synthetic
/// activation block and one output block, sized at [`FAMILY_ROWS`]. The
/// weight itself is not synthetic — it's the caller's own resident `w`, so
/// the identity check is over the bytes the deployment actually contracts.
/// Blocks the host; only an eager fire reaches this.
struct PairBench {
    stream: *mut c_void,
    start: cudaEvent_t,
    stop: cudaEvent_t,
    act: *mut c_void,
    y: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    /// bf16 count the identity check reads back: the first [`SMALL_ROWS`]
    /// columns of the column-major result (contiguous).
    compared: usize,
    n: i32,
    k: i32,
}

impl Drop for PairBench {
    fn drop(&mut self) {
        unsafe {
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

impl PairBench {
    fn empty() -> Self {
        Self {
            stream: std::ptr::null_mut(),
            start: std::ptr::null_mut(),
            stop: std::ptr::null_mut(),
            act: std::ptr::null_mut(),
            y: std::ptr::null_mut(),
            workspace: std::ptr::null_mut(),
            workspace_bytes: 0,
            compared: 0,
            n: 0,
            k: 0,
        }
    }

    fn init(&mut self, caller_stream: *mut c_void, workspace_bytes: usize, n: i32, k: i32) -> bool {
        let rows = FAMILY_ROWS as usize;
        let act_bytes = rows * (k as usize) * 2;
        let y_bytes = rows * (n as usize) * 2;
        // A synthetic output past this size is not worth the malloc.
        if y_bytes > 256 * 1024 * 1024 {
            return false;
        }
        self.workspace_bytes = workspace_bytes;
        self.compared = (SMALL_ROWS as usize) * (n as usize);
        self.n = n;
        self.k = k;
        let acquired = unsafe {
            cudaMalloc(&raw mut self.act, act_bytes) == cudaError::cudaSuccess
                && cudaMalloc(&raw mut self.y, y_bytes) == cudaError::cudaSuccess
                && cudaMalloc(&raw mut self.workspace, workspace_bytes) == cudaError::cudaSuccess
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
        // One sync of the caller's stream, so the weight's own writes are
        // landed and the timings do not race its queued work.
        if unsafe { cudaStreamSynchronize(caller_stream.cast()) } != cudaError::cudaSuccess {
            clear_error();
            return false;
        }
        let pattern = inexact_bf16(rows * (k as usize));
        let staged = unsafe {
            cudaMemcpyAsync(
                self.act,
                pattern.as_ptr().cast(),
                act_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream.cast(),
            ) == cudaError::cudaSuccess
                && cudaStreamSynchronize(self.stream.cast()) == cudaError::cudaSuccess
        };
        if !staged {
            clear_error();
            return false;
        }
        true
    }

    fn call(&self, w: u64) -> Call {
        Call {
            act: self.act.addr() as u64,
            w,
            y: self.y.addr() as u64,
            m: FAMILY_ROWS,
            n: self.n,
            k: self.k,
        }
    }

    /// One fire, and the first `count` bf16 of what it landed (the first
    /// `count / n` rows of the column-major `[n, m]` result).
    fn rows(
        &self,
        lt_handle: lt::cublasLtHandle_t,
        plan: &LtPlan,
        algo: &lt::cublasLtMatmulAlgo_t,
        w: u64,
        count: usize,
    ) -> Option<Vec<u16>> {
        let call = self.call(w);
        let mut landed = vec![0_u16; count];
        let read = unsafe {
            cudaMemsetAsync(self.y, 0, count * 2, self.stream.cast())
                == cudaError::cudaSuccess
                && run_lt(
                    lt_handle,
                    plan,
                    algo,
                    self.stream,
                    call,
                    self.workspace,
                    self.workspace_bytes,
                )
                && cudaMemcpyAsync(
                    landed.as_mut_ptr().cast(),
                    self.y,
                    count * 2,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    self.stream.cast(),
                ) == cudaError::cudaSuccess
                && cudaStreamSynchronize(self.stream.cast()) == cudaError::cudaSuccess
        };
        if !read {
            clear_error();
            return None;
        }
        Some(landed)
    }

    /// The best of seven event-timed fires, in milliseconds.
    fn time(
        &self,
        lt_handle: lt::cublasLtHandle_t,
        plan: &LtPlan,
        algo: &lt::cublasLtMatmulAlgo_t,
        w: u64,
    ) -> Option<f32> {
        let call = self.call(w);
        let fire = || {
            run_lt(
                lt_handle,
                plan,
                algo,
                self.stream,
                call,
                self.workspace,
                self.workspace_bytes,
            )
        };
        for _ in 0..3 {
            if !fire() {
                let _ = unsafe { cudaStreamSynchronize(self.stream.cast()) };
                clear_error();
                return None;
            }
        }
        if unsafe { cudaStreamSynchronize(self.stream.cast()) } != cudaError::cudaSuccess {
            clear_error();
            return None;
        }
        let mut best: Option<f32> = None;
        for _ in 0..7 {
            let _ = unsafe { cudaEventRecord(self.start, self.stream.cast()) };
            if !fire() {
                let _ = unsafe { cudaStreamSynchronize(self.stream.cast()) };
                clear_error();
                return None;
            }
            let _ = unsafe { cudaEventRecord(self.stop, self.stream.cast()) };
            if unsafe { cudaEventSynchronize(self.stop) } != cudaError::cudaSuccess {
                clear_error();
                return None;
            }
            let mut ms = 0.0f32;
            if unsafe { cudaEventElapsedTime(&raw mut ms, self.start, self.stop) }
                != cudaError::cudaSuccess
            {
                clear_error();
                return None;
            }
            if best.is_none_or(|b| ms < b) {
                best = Some(ms);
            }
        }
        best.filter(|ms| *ms > 0.0)
    }
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

// ─── the per-device state ───────────────────────────────────────────────────

/// One device's whole memory: the Lt context, the plan cache, and what the
/// tuner has chosen or seen.
struct Device {
    lt: LtCtx,
    plans: HashMap<(i32, i32, i32), Arc<LtPlan>>,
    /// One algorithm list per weight `(n, k)`, chosen at [`FAMILY_ROWS`] and
    /// reused at every M, plus the second algorithm that serves below it.
    families: HashMap<(i32, i32), Arc<Family>>,
    chosen: HashMap<u64, Tactic>,
    seen: HashMap<u64, u32>,
    disk: DiskCache,
}

fn with_device<R>(f: impl FnOnce(&mut Device) -> R) -> R {
    static DEVICES: OnceLock<Mutex<HashMap<i32, Device>>> = OnceLock::new();
    let mut map = DEVICES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let device = map.entry(current_device()).or_insert_with(|| Device {
        lt: LtCtx {
            handle: std::ptr::null_mut(),
            workspace_bytes: 64 * 1024 * 1024,
        },
        plans: HashMap::new(),
        families: HashMap::new(),
        chosen: HashMap::new(),
        seen: HashMap::new(),
        disk: DiskCache::new(),
    });
    f(device)
}

impl Device {
    fn plan_for(&mut self, call: Call, stream: *mut c_void, eager: bool) -> Option<Arc<LtPlan>> {
        let (m, n, k) = (call.m, call.n, call.k);
        if !self.lt.ensure() {
            return None;
        }
        // Family first: a settle here evicts this weight's cached plans, so
        // the lookup below must not have answered from one of them.
        let family = self.family_for(n, k, call.w, stream, eager);
        if let Some(plan) = self.plans.get(&(m, n, k)) {
            return Some(Arc::clone(plan));
        }
        let mut plan = build_lt_plan(self.lt.handle, self.lt.workspace_bytes, m, n, k)?;
        // Threshold is measured per weight (by `settle_small`), not assumed
        // at FAMILY_ROWS: the two algorithms land the same bits, which is
        // the only reason the split is allowed to exist.
        plan.invariant = family
            .small
            .filter(|(_, ceiling)| m <= *ceiling)
            .and_then(|(algo, _)| {
                accepted(
                    self.lt.handle,
                    &plan,
                    std::slice::from_ref(&algo),
                    self.lt.workspace_bytes,
                )
            })
            .or_else(|| {
                accepted(
                    self.lt.handle,
                    &plan,
                    &family.algos,
                    self.lt.workspace_bytes,
                )
            });
        if plan.invariant.is_none() {
            tracing::warn!(
                m,
                n,
                k,
                "no batch-invariant gemm algorithm for this shape; this fire's \
                 answer will depend on its width"
            );
        }
        let plan = Arc::new(plan);
        self.plans.insert((m, n, k), Arc::clone(&plan));
        Some(plan)
    }

    /// The `(n, k)` family, built on first sighting and settled on the first
    /// EAGER sighting — which may be a later one, because the settle benches
    /// and a capturing fire cannot.
    fn family_for(
        &mut self,
        n: i32,
        k: i32,
        w: u64,
        stream: *mut c_void,
        eager: bool,
    ) -> Arc<Family> {
        if let Some(family) = self.families.get(&(n, k))
            && (family.settled || !eager)
        {
            return Arc::clone(family);
        }
        let algos = match self.families.get(&(n, k)) {
            Some(family) => family.algos.clone(),
            None => family_algos(self.lt.handle, self.lt.workspace_bytes, n, k),
        };
        let small = eager
            .then(|| settle_small(&self.lt, stream, n, k, w, &algos))
            .flatten();
        let family = Arc::new(Family {
            algos,
            small,
            settled: eager,
        });
        self.families.insert((n, k), Arc::clone(&family));
        // A plan built before the settle carries the pre-settle choice.
        self.plans.retain(|&(_, cached_n, cached_k), _| {
            (cached_n, cached_k) != (n, k)
        });
        family
    }

    /// The remembered tactic for this shape, tuning it on the second eager
    /// sighting. `None` while the shape is unknown — the caller walks the
    /// ladder.
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
        // A runaway shape population would make the bench the workload; a
        // thousand tuned shapes is already generous.
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
        // A capturing stream never reaches the bench's host syncs: this
        // fire walks the ladder, and a later eager fire tunes the shape.
        if capturing != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone {
            return None;
        }
        // A bench output past this size is not worth the synthetic malloc.
        if (call.m as usize) * (call.n as usize) * 2 > 256 * 1024 * 1024 {
            return None;
        }
        let seen = self.seen.entry(key).or_insert(0);
        *seen += 1;
        if *seen < 2 {
            return None;
        }
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

// ─── the autotuner ──────────────────────────────────────────────────────────

/// Race every candidate on the bench and keep the first within 2% of the
/// fastest — ties go to the earlier, simpler tactic, and the Lt candidates
/// come in Lt's own estimated order. Falls back to the first candidate
/// untimed when the bench will not build.
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
    candidates.push(Tactic::GemmEx);
    if let Some(plan) = plan {
        candidates.extend((0..plan.heuristics.len()).map(Tactic::Lt));
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

/// The autotuner's private bench: a non-blocking stream of its own, timing
/// events, and synthetic operands to race the tactics over. `init`/`time`
/// block the host, guarded by the `cudaStreamIsCapturing` check in
/// [`Device::tactic_for`]: a captured fire never tunes, an eager fire may
/// block once per untuned shape (then cached in memory and on disk).
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
        }
    }

    /// Builds the bench: synthetic operands, the caller's handle rebound to
    /// the private stream (the drop restores it), and one sync of the
    /// caller's stream so the timings do not race its queued work.
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
        let acquired = unsafe {
            cudaMalloc(&raw mut self.act, act_bytes) == cudaError::cudaSuccess
                && cudaMalloc(&raw mut self.y, y_bytes) == cudaError::cudaSuccess
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
        // 0x3C bytes spell a small positive bf16: real work, no NaNs.
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

    /// Times one tactic on the bench: warmup fires, then the best of the
    /// event-timed fires. Blocks the host between phases.
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
        for _ in 0..3 {
            if !fire() {
                let _ = unsafe { cudaStreamSynchronize(self.stream.cast()) };
                clear_error();
                return None;
            }
        }
        if unsafe { cudaStreamSynchronize(self.stream.cast()) } != cudaError::cudaSuccess {
            clear_error();
            return None;
        }
        let mut best: Option<f32> = None;
        for _ in 0..7 {
            let _ = unsafe { cudaEventRecord(self.start, self.stream.cast()) };
            if !fire() {
                let _ = unsafe { cudaStreamSynchronize(self.stream.cast()) };
                clear_error();
                return None;
            }
            let _ = unsafe { cudaEventRecord(self.stop, self.stream.cast()) };
            if unsafe { cudaEventSynchronize(self.stop) } != cudaError::cudaSuccess {
                clear_error();
                return None;
            }
            let mut ms = 0.0f32;
            if unsafe { cudaEventElapsedTime(&raw mut ms, self.start, self.stop) }
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
}

// ─── the disk cache ─────────────────────────────────────────────────────────

/// The tuner's on-disk memory: one line per shape under a signature naming
/// everything able to invalidate a timing — device, arch, cuBLAS version.
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

/// The measured table, under the deployment's stated cache root. `None` is a
/// process that stated none: the disk half is off, the in-memory half still
/// works.
fn cache_path() -> Option<PathBuf> {
    Some(crate::disk::dir(crate::disk::GEMM_ALGOS)?.join("dense.txt"))
}

/// An empty answer disables the disk half of the cache.
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
        "# pie-dense-gemm v2 sm{}{} cublas={version} dev={name}",
        prop.major, prop.minor
    )
}
