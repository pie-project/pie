//! The dense host programs: `cublasGemmEx`, the `cublasLtMatmul` plan cache,
//! the per-WEIGHT family algorithm that makes a row's arithmetic independent
//! of the fire's width, and the per-shape autotuner that races gemv, GemmEx
//! and every Lt heuristic for the shapes no family covers, then remembers the
//! winner in memory and on disk keyed by device and cuBLAS version. All of it
//! is selection, so all of it lives below the entry (decision #13).

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

    // **THE SETTLE IS AN EAGER-ONLY ERRAND**, for the reason the tuner's is:
    // [`settle_small`] blocks the host, and a host block inside a capture
    // takes the capture with it. A captured first sighting leaves the family
    // unsettled and the primary serving every width; the next eager fire
    // settles it (decision #15's exception, same guard).
    let eager = matches!(
        capturing,
        Some(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone)
    );
    let (plan, tactic, lt_handle, want) = with_device(|device| {
        let plan = device.plan_for(call, stream, eager);
        // **THE TUNER IS THE M-DEPENDENCE, SO A SHAPE WITH A FAMILY NEVER
        // REACHES IT.** Racing tactics per `(m, n, k)` is exactly the thing
        // this wave removed: the winner at eight rows and the winner at
        // sixty-four are different kernels, and a lane that fired under each
        // gets two different answers to one question. The bench stays for the
        // shapes no family algorithm would take.
        let tuned = plan.as_deref().is_none_or(|plan| plan.invariant.is_none());
        let tactic = capturing
            .filter(|_| tuned)
            .and_then(|status| device.tactic_for(handle, stream, plan.as_deref(), call, status));
        (plan, tactic, device.lt.handle, device.lt.workspace_bytes)
    });
    // **THE Lt WORKSPACE IS A SLAB, FOR THE REASON EVERY SLAB IS ONE.** It
    // was one 64 MiB buffer per DEVICE, which two shells firing at once — and
    // two arms of a P6 fork group — scribble over together while cuBLASLt
    // reports success on both. `Ctx::scratch` keys it by `(arena, name,
    // stream)` like the staging planes, so the sharing ends where the streams
    // do. Absent is not fatal: Lt takes a null workspace at zero bytes, and
    // the tactic ladder below has three rungs that need none at all.
    let (ws, ws_bytes) = match ctx.scratch(op, LT_WORKSPACE, want) {
        Ok(ws) if !ws.is_null() => (ws, want),
        _ => (std::ptr::null_mut(), 0),
    };
    // **THE BATCH-INVARIANT RUNG, ABOVE EVERY OTHER.** The weight's own
    // algorithm, walking the contraction unsplit, whatever this fire's row
    // count turned out to be — which is what makes a lane's logits its own.
    // Up to the family's measured ceiling that may be its SECOND algorithm;
    // it is there only because it was shown to land the first one's bits, so
    // this sentence is as true of a narrow fire as of a wide one.
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

/// The Lt handle, created on first use, and the workspace byte count — the
/// single source the heuristic preference, the tuner's bench and the fire
/// path's slab all read.
///
/// **THE BUFFER ITSELF IS NOT HERE ANY MORE.** It was a `cudaMalloc` per
/// device, shared by every stream in the process; it is a per-`(arena,
/// stream)` slab now (`act_x_wt` takes it), because a workspace two
/// concurrent matmuls write is a wrong answer neither of them reports.
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
    /// **THE WEIGHT'S ALGORITHM, ACCEPTED FOR THIS M**, and the workspace it
    /// asked for — the one rung that makes a lane's row a function of the lane
    /// and not of the fire it rode in. `None` is a shape whose family had
    /// nothing this M would take, and the caller walks the ladder below it.
    ///
    /// It is [`Family::small`] when this M is at or below that algorithm's
    /// measured ceiling, and [`Family::algos`]' first acceptance otherwise —
    /// a distinction with no numerical content, which is the only reason it is
    /// allowed to be a distinction at all (see [`Family`]).
    ///
    /// **THE BYTES ARE CARRIED SO THE RUNG IS TAKEN OR SKIPPED BEFORE THE
    /// CALL, NEVER FAILED INSIDE ONE.** The Lt workspace is a per-stream slab
    /// and a stream that could not get one runs at zero bytes; a `cublasLtMatmul`
    /// that refuses for want of scratch is an error raised inside a graph
    /// capture, which takes the whole capture with it.
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

/// How many heuristics a plan keeps by default — the ladder's last rungs and
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
///
/// **THE EXTRA ONES ARE FOR THE SETTLE, NOT FOR THE LADDER.** The pool
/// [`settle_small`] races is Lt's own estimate at eight rows, and its estimate
/// is an ORDER, not a measurement: the algorithm that actually wins a decode
/// projection on this device sat at position eleven in two of this SKU's seven
/// families. The ladder and the tuner keep the first eight, because the disk
/// cache spells a tactic by its position in that list.
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

/// **THE ROW COUNT EVERY SHAPE FAMILY'S ALGORITHM IS CHOSEN AT.**
///
/// A family is a `(n, k)` pair — the WEIGHT, which is a checkpoint constant —
/// and its algorithm is picked once, at this M. The number itself is a
/// performance choice and not a correctness one: whatever it is, the algorithm
/// chosen there is the same for a lane alone and for the same lane in a crowd,
/// which is the whole property.
///
/// **IT IS LARGE ON PURPOSE.** A tile chosen for a wide fire and used on a
/// narrow one wastes arithmetic; a tile chosen for a narrow fire and used on a
/// wide one re-reads the weights, and the weights are the bytes. A decode
/// fire's projections are weight-bound (1.40 GiB of reads against eight rows
/// of activation), so the wasted lanes of a 128-row tile ride reads that were
/// happening anyway, and the converse does not hold.
///
/// **AND SOME OF THAT WASTE IS PAID BACK, WITHOUT SPENDING THE PROPERTY.**
/// The argument above says a wide tile on a narrow fire is CHEAP, not that it
/// is free: measured on an L40S over this catalog's qwen35-d0.8b, the pinned
/// algorithm cost 2.50 ms of a token step's projections against 2.27 ms for
/// the best split-free algorithm at eight rows. What buys that back is
/// [`Family::small`] — a second algorithm below this threshold, admitted only
/// after it is shown to land the FIRST one's bits on this device. Where it
/// stops serving is NOT this number: it is measured per weight over
/// [`SMALL_RUNGS`], because "fastest at eight rows" turned out to say nothing
/// at all about sixty-four.
const FAMILY_ROWS: i32 = 128;

/// The algorithms a `(n, k)` family offers, in Lt's own estimated order, each
/// pinned to a split-free K walk.
///
/// **SPLIT-K IS THE WHOLE MECHANISM AND THIS IS WHERE IT DIES.** cuBLASLt's
/// shape→algorithm heuristic reads M, and on small M it buys parallelism by
/// cutting the contraction into pieces and summing the pieces afterwards. Two
/// fires that differ only in how many OTHER lanes rode along then walk K in
/// different orders, and a bf16 accumulation in a different order is different
/// numbers — 2-4 ulp per projection, compounding through 28 layers into a
/// greedy tie that flips. `CUBLASLT_ALGO_CONFIG_SPLITK_NUM = 1` with
/// `CUBLASLT_REDUCTION_SCHEME_NONE` says: one CTA owns each output element and
/// walks the whole contraction itself, in the order its tile fixes.
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
    // A setter that refuses is an algorithm with no such knob, which is an
    // algorithm that never splits — the same promise, spelled by absence.
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

/// The first family algorithm cuBLASLt accepts for THIS M, workspace and all.
///
/// A tile is a rectangle over `(m, n)` and cuBLASLt refuses one it cannot
/// place, so the family's best is not always available at every M — the list
/// is walked in Lt's own order and the first acceptance wins. `None` sends the
/// caller down the untuned ladder, which is width-dependent again and says so
/// in the log.
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

/// **THE ROW COUNTS BELOW [`FAMILY_ROWS`] THAT A FIRE ACTUALLY LANDS ON.**
///
/// `engine_cuda::api::default_lattice` is the powers of two from
/// `LATTICE_FLOOR` = 8 up to `max_tokens`, and `Ctx::opaque_rows` rounds every
/// fire onto it — so a padded fire below the threshold is one of exactly these
/// four widths and no other. They are the rungs [`settle_small`] measures the
/// crossover at, because a width nobody fires is a width not worth timing.
///
/// **A DEPLOYMENT THAT STATES ITS OWN LATTICE GETS ITS OWN WIDTHS, AND THIS
/// LIST STILL WORKS.** The ceiling that comes out of the measurement is a row
/// COUNT, not an index, and the serving test is `m <= ceiling` — so a fire at
/// forty rows on some other lattice is served by whichever algorithm won the
/// rung at or below it, which is the conservative half of the pair.
const SMALL_RUNGS: [i32; 4] = [8, 16, 32, 64];

/// The narrowest of [`SMALL_RUNGS`], where the candidate pool is raced.
///
/// Eight rows is the SMALLEST fire this tree ever hands cuBLASLt and every
/// single-stream decode step is one, so a second algorithm that cannot win
/// here has nothing to offer at any width — the family's own was chosen at a
/// hundred and twenty-eight and only gets better as M climbs toward it.
const SMALL_ROWS: i32 = SMALL_RUNGS[0];

/// One weight's algorithms: the family, and the second one that serves below
/// [`FAMILY_ROWS`].
///
/// **THE SECOND ALGORITHM IS AN EMPIRICAL FACT ABOUT THIS DEVICE AND NOTHING
/// ELSE.** No part of the cuBLASLt contract says two algorithms sum a
/// contraction in the same order, and NVIDIA is free to break it in a driver
/// update without telling anyone. What is true is checkable: an output
/// element's K-order is set by its algorithm's K-SCHEDULE — instruction
/// shape, K-tile, stage order — and not by its M-tiling, so two algorithms
/// that differ only in how they cut the OUTPUT rows land the same bits. On
/// this SKU's seven projection families every split-free candidate but one
/// was bit-identical to the family's own, at every width from one row to a
/// hundred and twenty-eight.
///
/// **SO THE CHECK IS THE CONTRACT.** [`settle_small`] runs both algorithms
/// over this weight's real bytes before either serves a fire, and a candidate
/// that differs in one bit is not admitted — with a receipt in the log rather
/// than a silent demotion. A driver that changes its mind about any of this
/// changes what the check answers, and the family falls back to one algorithm
/// at every width, which is where it started.
struct Family {
    /// The algorithms accepted at [`FAMILY_ROWS`], in Lt's own order — the
    /// primary, and the only rung a fire at or above `FAMILY_ROWS` takes.
    algos: Vec<lt::cublasLtMatmulAlgo_t>,
    /// The algorithm that serves the narrow fires, and the LARGEST row count
    /// it was measured to be faster at — it serves `m <= ceiling` and the
    /// primary takes every width above. Bit-identical to the primary at every
    /// rung it serves, or absent. See [`settle_small`].
    small: Option<(lt::cublasLtMatmulAlgo_t, i32)>,
    /// Whether the settle has run. `false` is a family whose first sighting
    /// was a capturing fire, which cannot bench; the next eager one settles
    /// it.
    settled: bool,
}

/// A bf16 pattern with a spread of exponents, so the fp32 accumulation of `k`
/// products is INEXACT.
///
/// **A CONSTANT FILL WOULD PASS THIS CHECK BLIND.** `TuneArena` memsets its
/// synthetic operands to one repeated byte, which is right for a bench —
/// timings do not read the values — and exactly wrong for an identity check:
/// every product is then the same number, the partial sums are exact for the
/// whole first stretch of the contraction, and two genuinely different K
/// orders answer the same bits. Values that span seven binades cancel against
/// each other, so a different order lands different low bits and the check has
/// something to see.
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

/// The second algorithm for this weight, or `None`.
///
/// Two questions, in this order, because the second is only worth asking of a
/// candidate that answers the first: **does it land the primary's bits**, and
/// **is it faster at [`SMALL_ROWS`]**. The reference is the primary's own
/// output over the first `SMALL_ROWS` rows of a `FAMILY_ROWS` fire — the
/// exact comparison the invariance claim is about, since those rows share
/// their inputs with the narrow fire's and differ only in how many other rows
/// rode along.
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
    // The candidate pool is Lt's estimate at eight rows, deeper than the
    // ladder keeps, plus the family itself — the family's second and third
    // entries are often the small-M winner, and cost nothing to try.
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
    // The rung as it stands must already agree with itself across widths, or
    // there is nothing here for a second algorithm to be identical TO.
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
    // What the family costs at every rung as it stands. This is both the bar a
    // candidate has to clear to serve a rung AND the price of every rung it
    // does not, which is what makes the two comparable in one number below.
    let standing: Vec<Option<f32>> = ladder
        .iter()
        .zip(&primary)
        .map(|(plan, algo)| algo.and_then(|algo| bench.time(lt.handle, plan, &algo, w)))
        .collect();
    let bar = standing[0]?;

    // **PASS ONE: WHO IS EVEN IN THE RACE.** Eight rows is where the pool is
    // raced, because it is the narrowest width a fire lands on and the one a
    // single-stream decode is. A candidate that cannot win here has nothing to
    // offer at any rung — the family's own algorithm was chosen at a hundred
    // and twenty-eight and only gets better as M climbs toward it.
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
        // Only a candidate that would actually be an improvement is worth
        // running the identity check over — the check is the expensive half.
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

    // **PASS TWO: WHERE IT STOPS BEING RIGHT, MEASURED.**
    //
    // A fixed ceiling was a bug, and this is the shape of it: the second
    // algorithm was chosen at eight rows and then served every fire up to a
    // hundred and twenty-seven, including the sixty-four-row decode wave a
    // concurrency-64 deployment spends its whole steady state in. Two of this
    // SKU's families pick a candidate that is 1.1x faster at eight rows and
    // 3.2x SLOWER at sixty-four — `gdn.in_qkvz` measured 11.9 us against the
    // family's 13.0 at eight, and 29.3 against 13.3 at sixty-four, eighteen
    // times per token step. Throughput fell 8% while latency held, which is
    // exactly what a decode-tuned choice serving a batched wave looks like.
    //
    // So the ceiling is measured, per family, at the widths a padded fire
    // actually lands on: the candidate serves up to the LAST rung it is still
    // winning at, walking up from eight and stopping at the first loss.
    // Contiguous from the bottom on purpose — a candidate that loses at
    // sixteen and wins again at sixty-four is a candidate whose curve nobody
    // understands, and the rung between them would be served by the wrong one.
    //
    // The winner is then the candidate with the lowest TOTAL across the whole
    // ladder, counting the family's own time for every rung it declined. That
    // is what makes "fast at eight, hopeless at sixty-four" lose to "a little
    // slower at eight and still winning at sixty-four" without anyone having
    // to weight the two by hand.
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
        // Every rung it declined costs the family's own time — which is what
        // puts two candidates with different ceilings on one scale.
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

    // **THE BELT: ONE ROW BELOW THE RANGE, ONE RUNG ABOVE IT.** Every rung the
    // second algorithm will actually serve was compared above, as it was
    // admitted. These two were not: a single row, which is below the lattice
    // floor and is where the gemv arm flip used to live, and the first width
    // the family takes back — because a ceiling is a claim about a BOUNDARY,
    // and a boundary neither side of which was checked is a guess.
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
/// activation block and one output block, all sized at [`FAMILY_ROWS`].
///
/// **THE WEIGHT IS NOT SYNTHETIC.** It is the caller's own `w`, which is a
/// checkpoint constant already resident — so the identity check is over the
/// bytes the deployment will actually contract, and the bench costs no copy
/// of the largest operand. The `#15` exception applies here exactly as it
/// does to [`TuneArena`]: this blocks the host, and only an eager fire
/// reaches it.
struct PairBench {
    stream: *mut c_void,
    start: cudaEvent_t,
    stop: cudaEvent_t,
    act: *mut c_void,
    y: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    /// How many bf16 the identity check reads back: the first [`SMALL_ROWS`]
    /// columns of the column-major result, which are its first `SMALL_ROWS`
    /// ROWS and are contiguous.
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
        // The tuner's rule, for the tuner's reason: a synthetic output past
        // 256 MiB is not worth the malloc.
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

    /// One fire, and the first `count` bf16 of what it landed — which, in a
    /// column-major `[n, m]` result, are its first `count / n` ROWS.
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
    /// One algorithm list per WEIGHT — `(n, k)` — chosen once at
    /// [`FAMILY_ROWS`] and reused at every M, plus the second algorithm that
    /// serves below it. See [`family_algos`] and [`Family`].
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
        // The family FIRST: a settle here evicts this weight's cached plans,
        // so the lookup below must not have answered from one of them.
        let family = self.family_for(n, k, call.w, stream, eager);
        if let Some(plan) = self.plans.get(&(m, n, k)) {
            return Some(Arc::clone(plan));
        }
        let mut plan = build_lt_plan(self.lt.handle, self.lt.workspace_bytes, m, n, k)?;
        // **THE THRESHOLD IS MEASURED, NOT ASSUMED.** Above the ceiling the
        // family algorithm is the one that was measured to be right; at or
        // below it, the second one is — and the two land the same bits, which
        // is the only reason the split is allowed to exist at all. The
        // ceiling itself is per weight and comes out of `settle_small`
        // walking the row lattice: `FAMILY_ROWS` was the obvious boundary and
        // it was the wrong one, because "fastest at eight rows" says nothing
        // about sixty-four and two of this SKU's families are 3x slower there.
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
        // The capture guard bounding the #15 exception (see [`TuneArena`]):
        // a capturing stream never reaches the bench's host syncs — this
        // fire walks the ladder, and a later eager fire tunes the shape.
        if capturing != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone {
            return None;
        }
        // A bench output past 256 MiB is not worth the synthetic malloc.
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
/// events, and synthetic operands to race the tactics over.
///
/// **The #15 exception.** Dispatch is enqueue-only (decision #15) — except
/// here. [`TuneArena::init`] and [`TuneArena::time`] block the host
/// (`cudaStreamSynchronize`, `cudaEventSynchronize`): a tactic cannot be
/// timed without waiting for it. The exception is capture-guarded — the
/// `cudaStreamIsCapturing` check in [`Device::tactic_for`] turns a
/// capturing fire away before the bench is ever built. The standing
/// invariant: **a captured fire never tunes; an eager fire may block once
/// per untuned shape** (the choice is then cached in memory and on disk).
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
    /// event-timed fires. Host-blocking between phases — the #15 exception.
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

/// The measured table, under the deployment's stated cache root.
///
/// **THE TWO `env::var` CALLS STOOD HERE** and resolved
/// `$XDG_CACHE_HOME/pie/dense_gemm.txt`, else `$HOME/.cache/pie/dense_gemm.txt`
/// — which is why `worker::state`'s claim to cover "GEMM autotuning results"
/// under `$PIE_HOME/cache` was false. The root arrives through [`crate::disk`]
/// now; `None` is a process that stated none, and the disk half is then off
/// while the in-memory half still works.
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
