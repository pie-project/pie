#![allow(clippy::too_many_arguments)]
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::c_void;
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
    cudaGraphExecDestroy, cudaGraphInstantiateWithFlags, cudaGraphLaunch, cudaMalloc, cudaMemcpyAsync,
    cudaMemcpyKind, cudaMemsetAsync, cudaStreamBeginCapture, cudaStreamCaptureMode, cudaStreamCaptureStatus,
    cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamEndCapture, cudaStreamNonBlocking,
    cudaStreamSynchronize,
};

use crate::jit::device::capture_status;
use crate::jit::{Ctx, refuse};
use crate::linear::gemv::{gemm_narrow_bf16, gemm_rows_bf16, gemv_bf16, rows_cluster};
use crate::linear::skeleton::{
    Epilogue, In, LtEpilogue, MAX_INS, MAX_OUTS, Paging, ROWS_M, RowLeg, Skeleton,
};
use crate::linear::skinny;

const fn dev(address: u64) -> *const c_void {
    address as usize as *const c_void
}

const fn dev_mut(address: u64) -> *mut c_void {
    address as usize as *mut c_void
}

fn clear_error() {
    // SAFETY: takes no arguments; it only clears this thread's sticky error.
    let _ = unsafe { cudaGetLastError() };
}

fn current_device() -> i32 {
    let mut device: i32 = 0;
    // SAFETY: `device` is a local out-pointer; on an error it keeps its 0.
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
    /// An anchored region's epilogue and planes; `y` is then its products'
    /// plane where its leg lands them unfused.
    anchor: Option<Anchor>,
}

/// An anchored region's planes past the contraction's: its epilogue, its
/// inputs and results, its unfused leg's row skeleton where one carries it,
/// and the plain product's tactic that leg runs (its `n` every part's).
#[derive(Clone, Copy, Debug)]
struct Anchor {
    epi: &'static Epilogue,
    ins: [In; MAX_INS],
    n_ins: usize,
    outs: [u64; MAX_OUTS],
    n_outs: usize,
    leg: Option<RowLeg>,
    inner: Option<Tactic>,
    /// Where its paged results' cache rows live.
    paging: Option<Paging>,
}

impl Anchor {
    fn ins(&self) -> &[In] {
        &self.ins[..self.n_ins]
    }

    fn outs(&self) -> &[u64] {
        &self.outs[..self.n_outs]
    }

    fn parts(&self) -> i32 {
        i32::try_from(self.epi.parts()).unwrap_or(1)
    }
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
    /// The skinny skeleton: up to 64 rows on mma.sync, the weight streamed once
    /// past the activation's rows in shared memory.
    Skinny,
    /// An anchored region fused: a contraction skeleton with its epilogue
    /// filled in.
    Skeleton(Skeleton),
    /// An anchored region unfused: the plain product's own tactic, then the
    /// epilogue on the row skeleton.
    Unfused,
    /// An anchored region on cuBLASLt's own epilogue (its
    /// [`Epilogue::lt`]): the plain product's heuristic `index` with it.
    LtEpilogue(usize),
}

impl Tactic {
    fn encode(self) -> (i32, i32) {
        match self {
            Self::GemmEx => (0, 0),
            Self::Lt(algo) => (1, i32::try_from(algo).unwrap_or(0)),
            Self::Gemv => (2, 0),
            Self::Narrow => (3, 0),
            Self::Rows => (4, 0),
            Self::Skinny => (5, 0),
            Self::Skeleton(skeleton) => (
                6,
                Skeleton::ALL
                    .iter()
                    .position(|f| *f == skeleton)
                    .and_then(|i| i32::try_from(i).ok())
                    .unwrap_or(0),
            ),
            Self::Unfused => (7, 0),
            Self::LtEpilogue(algo) => (8, i32::try_from(algo).unwrap_or(0)),
        }
    }

    fn decode(kind: i32, algo: i32) -> Option<Self> {
        match kind {
            0 => Some(Self::GemmEx),
            1 => usize::try_from(algo).ok().map(Self::Lt),
            2 => Some(Self::Gemv),
            3 => Some(Self::Narrow),
            4 => Some(Self::Rows),
            5 => Some(Self::Skinny),
            6 => usize::try_from(algo)
                .ok()
                .and_then(|i| Skeleton::ALL.get(i))
                .map(|f| Self::Skeleton(*f)),
            7 => Some(Self::Unfused),
            8 => usize::try_from(algo).ok().map(Self::LtEpilogue),
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
    let call = Call {
        act,
        w,
        y,
        m,
        n,
        k,
        anchor: None,
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
            tuning.gemm_skinny_rows,
        );
        let plan = device.plan_for(call);
        let tactic = capturing
            .and_then(|status| device.tactic_for(handle, stream, plan.as_deref(), call, status))
            .filter(|_| false);
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
            tuning,
        )
    {
        return Ok(());
    }
    clear_error();

    if m == 1
        && gemv_bf16(
            // SAFETY: `stream` is `ctx`'s, live by `Ctx::on`'s contract for `ctx`.
            &unsafe { Ctx::on(stream) }.with_tuning(tuning),
            w,
            act,
            y,
            n,
            k,
        )
        .is_ok()
    {
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

/// `outs = epi(act . w^T)`, an anchored region over a dense bf16 bank of
/// `parts · n` rows, on whichever of its candidates the shape's timing
/// picked: each contraction skeleton that covers it with the epilogue filled
/// in, and the plain product's own tactic followed by the epilogue on the
/// row skeleton where the row skeleton carries it. Where no skeleton covers the
/// shape, or a capture meets it untimed, the unfused leg serves it untimed;
/// false, landing nothing, only where the row skeleton does not carry that
/// leg either, which the fold never forms.
pub(crate) fn anchored(
    ctx: &Ctx,
    op: &'static str,
    epi: &'static Epilogue,
    act: u64,
    w: u64,
    ins: &[In],
    outs: &[u64],
    paging: Option<Paging>,
    m: i32,
    n: i32,
    k: i32,
) -> Result<bool, Error> {
    if m <= 0 || n <= 0 || k <= 0 || epi.plain() || ins.len() > MAX_INS || outs.len() > MAX_OUTS {
        return Ok(false);
    }
    let tuning = ctx.tuning();
    let skeletons: Vec<Skeleton> = Skeleton::ALL
        .into_iter()
        .filter(|f| f.covers(epi, m, n, k, &tuning))
        .collect();
    if skeletons.is_empty() && epi.lt().is_none() {
        return unfused(ctx, op, epi, act, w, ins, outs, paging, m, n, k);
    }
    let mut anchor = Anchor {
        epi,
        ins: [In::one(0); MAX_INS],
        n_ins: ins.len(),
        outs: [0; MAX_OUTS],
        n_outs: outs.len(),
        leg: None,
        inner: None,
        paging,
    };
    anchor.ins[..ins.len()].copy_from_slice(ins);
    anchor.outs[..outs.len()].copy_from_slice(outs);
    let parts = anchor.parts();
    let handle: cublasHandle_t = ctx.cublas(op)?.cast::<cublasContext>();
    let stream = ctx.stream();
    let capturing = capture_status(stream);
    let covering = skeletons
        .iter()
        .map(|f| Skeleton::ALL.iter().position(|g| g == f).unwrap_or(0))
        .fold(0u64, |mask, i| mask | 1 << i);
    let chosen = with_device(|device| {
        device.configure(
            (tuning.gemm_workspace_mb as usize) << 20,
            tuning.gemm_heuristics as usize,
            tuning.gemm_narrow_n,
            tuning.gemm_split_launch_ns,
            tuning.gemm_rows_m,
            tuning.gemm_split_launch_from,
            tuning.gemm_skinny_rows,
        );
        let call = Call {
            act,
            w,
            y: 0,
            m,
            n,
            k,
            anchor: None,
        };
        // The candidates key it: the skeletons that cover the shape, and
        // whether cuBLASLt's own epilogue is one.
        let key = shape_key(call, device.lt.workspace_bytes, device.heuristics, 0, 0)
            ^ epi.hash().rotate_left(17)
            ^ 0x616e_6368_0000_0000
            ^ (covering << 40)
            ^ (u64::from(epi.lt().is_some()) << 48);
        // The plain product's plan: its heuristics are the unfused leg's
        // and, with its epilogue, cuBLASLt's fused candidates.
        let plain = Call {
            n: n * parts,
            ..call
        };
        let lt = (device.lt.handle, device.lt.workspace_bytes);
        let planned = |device: &mut Device| epi.lt().and_then(|_| device.plan_for(plain));
        if let Some(tactic) = device.chosen.get(&key).copied() {
            return (Some(tactic), planned(device), lt);
        }
        if let Some(tactic) = device
            .disk
            .lookup(key)
            .and_then(|(kind, algo)| Tactic::decode(kind, algo))
        {
            device.chosen.insert(key, tactic);
            return (Some(tactic), planned(device), lt);
        }
        let Some(status) = capturing else {
            return (None, planned(device), lt);
        };
        if status != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone
            || device.chosen.len() >= 1024
            || (m as usize) * (n as usize) * 2 * (parts as usize + outs.len()) > 256 << 20
        {
            return (None, planned(device), lt);
        }
        // The unfused leg: the plain product's tactic, timed (or read) as
        // the plain product's own, then the row skeleton.
        let plan = device.plan_for(plain);
        anchor.leg = RowLeg::of(epi, n.unsigned_abs());
        if anchor.leg.is_some() && epi.rounds() {
            anchor.inner = device.tactic_for(handle, stream, plan.as_deref(), plain, status);
        }
        let mut candidates = Vec::new();
        if anchor.leg.is_some() && (anchor.inner.is_some() || !epi.rounds()) {
            candidates.push(Tactic::Unfused);
        }
        candidates.extend(skeletons.iter().map(|f| Tactic::Skeleton(*f)));
        // cuBLASLt's own epilogue, at each of the plain product's heuristics.
        if let (Some(_), Some(plan)) = (epi.lt(), plan.as_deref()) {
            candidates.extend((0..plan.heuristics.len()).map(Tactic::LtEpilogue));
        }
        if candidates.is_empty() {
            return (None, plan, lt);
        }
        let charge = device.charge(m);
        let tactic = pick(
            handle,
            stream,
            &device.lt,
            plan.as_deref(),
            Call {
                anchor: Some(anchor),
                ..call
            },
            &candidates,
            charge,
        );
        device.disk.store(key, tactic.encode());
        device.chosen.insert(key, tactic);
        tracing::info!(
            m,
            n,
            k,
            parts,
            epilogue = epi.hash(),
            ?tactic,
            "anchored gemm shape tuned"
        );
        (Some(tactic), plan, lt)
    });
    let (chosen, plan, (lt_handle, want)) = chosen;
    // Untimed (a capture, a shape past the tuning arena), a region cuBLASLt's
    // epilogue lands runs on it at cuBLASLt's own first heuristic: one pass
    // over the product, where the unfused leg is two.
    let chosen = match chosen {
        None if epi.lt().is_some() && outs.len() == 1 && plan.is_some() => {
            Some(Tactic::LtEpilogue(0))
        }
        chosen => chosen,
    };
    match chosen {
        Some(Tactic::LtEpilogue(index)) => {
            let (Some(plan), Some(epilogue), Some(heuristic)) = (
                plan.as_deref(),
                epi.lt(),
                plan.as_deref().and_then(|p| p.heuristics.get(index)),
            ) else {
                return unfused(ctx, op, epi, act, w, ins, outs, paging, m, n, k);
            };
            let (ws, ws_bytes) = match ctx.scratch_shared(op, LT_WORKSPACE, want) {
                Ok(ws) if !ws.is_null() => (ws, want),
                _ => (std::ptr::null_mut(), 0),
            };
            let call = Call {
                act,
                w,
                y: outs[0],
                m,
                n,
                k,
                anchor: None,
            };
            let bias = ins.first().map_or(0, |b| b.base());
            if !lt_handle.is_null()
                && run_lt_epilogue(
                    lt_handle,
                    plan,
                    &heuristic.algo,
                    stream,
                    call,
                    epilogue,
                    bias,
                    ws,
                    ws_bytes,
                )
            {
                return Ok(true);
            }
            clear_error();
            unfused(ctx, op, epi, act, w, ins, outs, paging, m, n, k)
        }
        Some(Tactic::Skeleton(skeleton)) => {
            skeleton.fire(ctx, epi, w, act, ins, outs, paging, m, n, k)?;
            Ok(true)
        }
        Some(Tactic::Unfused) => {
            let Some(leg) = RowLeg::of(epi, n.unsigned_abs()) else {
                return Ok(false);
            };
            let bytes = (m as usize) * (n as usize) * (parts as usize) * epi.product_bytes();
            let products = ctx.scratch_stream(op, ANCHOR_PRODUCTS, bytes)?.addr() as u64;
            product(ctx, op, epi, act, w, products, m, n * parts, k)?;
            leg.fire(epi, ctx, products, ins, outs, paging, m.unsigned_abs(), n.unsigned_abs())?;
            Ok(true)
        }
        _ => unfused(ctx, op, epi, act, w, ins, outs, paging, m, n, k),
    }
}

/// `out = epi(act . w^T)` on cuBLASLt's own epilogue alone, at the first of
/// the plain product's heuristics that takes it: the fused candidate
/// [`anchored`] times, run as it is. False, landing nothing, where the
/// epilogue is none of cuBLASLt's or no heuristic takes it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_lt(
    ctx: &Ctx,
    op: &'static str,
    epi: &'static Epilogue,
    act: u64,
    w: u64,
    ins: &[In],
    out: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<bool, Error> {
    let Some(epilogue) = epi.lt() else {
        return Ok(false);
    };
    let tuning = ctx.tuning();
    let call = Call {
        act,
        w,
        y: out,
        m,
        n,
        k,
        anchor: None,
    };
    let (plan, lt_handle, want) = with_device(|device| {
        device.configure(
            (tuning.gemm_workspace_mb as usize) << 20,
            tuning.gemm_heuristics as usize,
            tuning.gemm_narrow_n,
            tuning.gemm_split_launch_ns,
            tuning.gemm_rows_m,
            tuning.gemm_split_launch_from,
            tuning.gemm_skinny_rows,
        );
        (
            device.plan_for(call),
            device.lt.handle,
            device.lt.workspace_bytes,
        )
    });
    let Some(plan) = plan else {
        return Ok(false);
    };
    if lt_handle.is_null() {
        return Ok(false);
    }
    let (ws, ws_bytes) = match ctx.scratch_shared(op, LT_WORKSPACE, want) {
        Ok(ws) if !ws.is_null() => (ws, want),
        _ => (std::ptr::null_mut(), 0),
    };
    let bias = ins.first().map_or(0, |b| b.base());
    for heuristic in &plan.heuristics {
        if run_lt_epilogue(
            lt_handle,
            &plan,
            &heuristic.algo,
            ctx.stream(),
            call,
            epilogue,
            bias,
            ws,
            ws_bytes,
        ) {
            return Ok(true);
        }
    }
    clear_error();
    Ok(false)
}

/// `outs = epi(act . w^T)` untimed on the unfused leg: the plain product
/// (its own tactic, as a bare product picks it) into scratch, then the
/// epilogue over it on the row skeleton, a run of rows at a time so the
/// scratch stays within [`UNFUSED_BYTES`]. False, landing nothing, where
/// the row skeleton does not carry the leg.
#[allow(clippy::too_many_arguments)]
fn unfused(
    ctx: &Ctx,
    op: &'static str,
    epi: &'static Epilogue,
    act: u64,
    w: u64,
    ins: &[In],
    outs: &[u64],
    paging: Option<Paging>,
    m: i32,
    n: i32,
    k: i32,
) -> Result<bool, Error> {
    let Some(leg) = RowLeg::of(epi, n.unsigned_abs()) else {
        return Ok(false);
    };
    let wide = n as usize * epi.parts() as usize;
    let row_bytes = wide * epi.product_bytes();
    let run = (UNFUSED_BYTES / row_bytes).clamp(1, m as usize);
    let products = ctx
        .scratch_stream(op, ANCHOR_PRODUCTS, run * row_bytes)?
        .addr() as u64;
    let mut from = 0usize;
    while from < m as usize {
        let rows = run.min(m as usize - from);
        let at = |base: u64, width: usize| base + (from * width * 2) as u64;
        let ins: Vec<In> = ins.iter().map(|input| input.from_row(from)).collect();
        let outs: Vec<u64> = (0..outs.len())
            .zip(outs)
            .map(|(r, &out)| {
                // a cache's rows are its page's and slot's, not the run's
                if epi.paged(r).is_some() {
                    out
                } else {
                    at(out, epi.window(r, n.unsigned_abs()).width as usize)
                }
            })
            .collect();
        let rows_i = i32::try_from(rows).unwrap_or(i32::MAX);
        product(
            ctx,
            op,
            epi,
            at(act, k as usize),
            w,
            products,
            rows_i,
            wide as i32,
            k,
        )?;
        // a paged result's live rows from the run's first
        let paging = paging.map(|p| Paging {
            valid: if p.valid == 0 { 0 } else { p.valid + from as u64 },
            ..p
        });
        leg.fire(
            epi,
            ctx,
            products,
            &ins,
            &outs,
            paging,
            rows_i.unsigned_abs(),
            n.unsigned_abs(),
        )?;
        from += rows;
    }
    Ok(true)
}

/// `y = act . w^T` as `epi` reads its products: rounded to bf16 on the plain
/// product's own tactic, or its f32 sums.
#[allow(clippy::too_many_arguments)]
fn product(
    ctx: &Ctx,
    op: &'static str,
    epi: &Epilogue,
    act: u64,
    w: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    if epi.rounds() {
        act_x_wt(ctx, op, act, w, y, m, n, k)
    } else {
        act_x_wt_f32(ctx, op, act, w, y, m, n, k)
    }
}

/// The most scratch the untimed unfused leg's products take at once.
const UNFUSED_BYTES: usize = 64 << 20;

/// The plane an anchored region's unfused leg lands its products in.
const ANCHOR_PRODUCTS: &str = "linear.anchor_products";

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
                    // SAFETY: `stream` is a live `Ctx`'s stream or the tune arena's own; `Ctx::on`
                    // only records it.
                    &unsafe { Ctx::on(stream) }.with_tuning(tuning),
                    call.w,
                    call.act,
                    call.y,
                    call.n,
                    call.k,
                )
                .is_ok()
        }
        Tactic::Rows => gemm_rows_bf16(
            // SAFETY: `stream` is a live `Ctx`'s stream or the tune arena's own; `Ctx::on`
            // only records it.
            &unsafe { Ctx::on(stream) }.with_tuning(tuning),
            call.w,
            call.act,
            call.y,
            call.m,
            call.n,
            call.k,
            pdl,
        )
        .is_ok(),
        Tactic::Skinny => {
            skinny::covers(call.m, call.n, call.k, 1, skinny::ROWS)
                && skinny::skinny(
                    // SAFETY: `stream` is a live `Ctx`'s stream or the tune arena's own; `Ctx::on`
                    // only records it.
                    &unsafe { Ctx::on(stream) }.with_tuning(tuning),
                    Epilogue::store(),
                    call.w,
                    call.act,
                    &[],
                    &[call.y],
                    None,
                    call.m,
                    call.n,
                    call.k,
                )
                .is_ok()
        }
        Tactic::Skeleton(skeleton) => call.anchor.is_some_and(|a| {
            skeleton
                .fire(
                    // SAFETY: `stream` is a live `Ctx`'s stream or the tune arena's own; `Ctx::on`
                    // only records it.
                    &unsafe { Ctx::on(stream) }.with_tuning(tuning),
                    a.epi,
                    call.w,
                    call.act,
                    a.ins(),
                    a.outs(),
                    a.paging,
                    call.m,
                    call.n,
                    call.k,
                )
                .is_ok()
        }),
        Tactic::Unfused => {
            let Some(a) = call.anchor else {
                return false;
            };
            let Some(leg) = a.leg else {
                return false;
            };
            let plain = Call {
                n: call.n * a.parts(),
                anchor: None,
                ..call
            };
            // The products as the epilogue reads them: rounded to bf16 by
            // the plain product's own tactic, or its f32 sums.
            let landed = if a.epi.rounds() {
                a.inner.is_some_and(|inner| {
                    run_tactic(
                        handle, stream, lt_handle, plan, inner, plain, ws, ws_bytes, tuning,
                    )
                })
            } else {
                gemm_ex_f32(handle, plain) == cublasStatus_t::CUBLAS_STATUS_SUCCESS
            };
            landed
                && leg
                    .fire(
                        a.epi,
                        // SAFETY: as above.
                        &unsafe { Ctx::on(stream) }.with_tuning(tuning),
                        call.y,
                        a.ins(),
                        a.outs(),
                        a.paging,
                        call.m.unsigned_abs(),
                        call.n.unsigned_abs(),
                    )
                    .is_ok()
        }
        Tactic::Narrow => gemm_narrow_bf16(
            // SAFETY: `stream` is a live `Ctx`'s stream or the tune arena's own; `Ctx::on`
            // only records it.
            &unsafe { Ctx::on(stream) }.with_tuning(tuning),
            call.w,
            call.act,
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
        Tactic::LtEpilogue(index) => {
            let (Some(plan), false, Some(a)) = (plan, lt_handle.is_null(), call.anchor) else {
                return false;
            };
            let (Some(heuristic), Some(epilogue), [out]) =
                (plan.heuristics.get(index), a.epi.lt(), a.outs())
            else {
                return false;
            };
            let bias = a.ins().first().map_or(0, |b| b.base());
            run_lt_epilogue(
                lt_handle,
                plan,
                &heuristic.algo,
                stream,
                Call {
                    y: *out,
                    anchor: None,
                    ..call
                },
                epilogue,
                bias,
                ws,
                ws_bytes,
            )
        }
    }
}

fn gemm_ex(handle: cublasHandle_t, call: Call, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: `handle` is a live cuBLAS handle (a `Ctx`'s, or the arena's
    // borrowed one) and `alpha`/`beta` are host f32 locals, as the default
    // pointer mode and the 32F compute type take. `call` names an `m x k`
    // act, an `n x k` weight and an `m x n` y, all bf16: the caller's
    // tensors, or the arena's slabs allocated at those sizes.
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
    let call = Call {
        act,
        w,
        y,
        m,
        n,
        k,
        anchor: None,
    };
    let status = gemm_ex_f32(handle, call);
    if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(crate::jit::refuse(
            op,
            format!("cuBLAS answered {status:?}"),
        ))
    }
}

/// `call`'s product landing f32 (`call.y` an `m x n` f32 plane) on
/// `handle`'s stream: an anchored region's products where it reads them as
/// they are summed.
fn gemm_ex_f32(handle: cublasHandle_t, call: Call) -> cublasStatus_t {
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: `handle` is a live cuBLAS handle bound to the caller's stream;
    // the operands are device rows of the call's extents.
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
            cudaDataType::CUDA_R_32F,
            call.n,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
}

const LT_WORKSPACE: &str = "linear.lt_workspace";

struct LtCtx {
    handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
}

// SAFETY: the cuBLASLt handle holds no stream or workspace (each matmul
// names its own) and is only created or read under the device's mutex;
// nothing in it is tied to the creating thread.
unsafe impl Send for LtCtx {}

impl LtCtx {
    fn ensure(&mut self) -> bool {
        if self.handle.is_null()
            // SAFETY: `self.handle` is null here, so this writes a fresh handle and leaks none.
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

// SAFETY: the descriptors are host objects cuBLASLt allocates with no
// thread affinity; a plan is built whole before it is shared and freed
// once, by whichever thread drops the last `Arc`.
unsafe impl Send for LtPlan {}

// SAFETY: after `build_lt_plan` returns nothing writes a plan's
// descriptors: `run_lt` and `cublasLtMatmulAlgoCheck` only read them,
// and `run_lt_epilogue` sets its epilogue on a descriptor of its own.
unsafe impl Sync for LtPlan {}

impl Drop for LtPlan {
    fn drop(&mut self) {
        // SAFETY: each handle is one `build_lt_plan` created for this plan alone;
        // null ones (a build that failed part way) are skipped, and each is
        // freed once.
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
                // SAFETY: a non-null preference this guard alone owns, freed once.
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
    // SAFETY: `plan.op_desc` is a local out-pointer.
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
        // SAFETY: `op_desc` was just created; `operation` is a local of the
        // attribute's type and size, which cuBLASLt copies.
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
        // SAFETY: `desc` points at a field of the local `plan`; the dims are plain integers.
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
    // SAFETY: `pref.0` is a local out-pointer, freed by the guard.
    if unsafe { lt::cublasLtMatmulPreferenceCreate(&raw mut pref.0) } != ok {
        return None;
    }
    // SAFETY: `pref.0` was just created; `workspace_bytes` is a usize local of the size passed.
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
    // SAFETY: `pref.0` was just created; `reductions` is a u32 local of the size passed.
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
        // SAFETY: the heuristic result is a C struct of opaque algo words,
        // integers, a status whose zero is SUCCESS and a float; all-zero bytes are a
        // valid value.
        vec![unsafe { core::mem::zeroed() }; heuristics_wanted.max(1)];
    let mut returned: i32 = 0;
    // SAFETY: `lt_handle` is the device's live handle, the descriptors and
    // `pref.0` were created above, and `heuristics` has room for the count
    // asked for; `returned` is a local out-pointer.
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
        // SAFETY: `heuristic.algo` is an algorithm cuBLASLt just returned;
        // `supports` is an i32 local of the size passed and `written` a local.
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
            // SAFETY: `algo` is a local copy of a returned algorithm; `split` is an i32 local of the size passed.
            let set_split = unsafe {
                lt::cublasLtMatmulAlgoConfigSetAttribute(
                    &raw mut algo,
                    lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    std::ptr::from_ref(&split).cast(),
                    std::mem::size_of::<i32>(),
                )
            };
            // SAFETY: as above, `scheme` a local of the enum's size.
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
            // SAFETY: as for `heuristics` above: all-zero bytes are a valid value.
            let mut result: lt::cublasLtMatmulHeuristicResult_t = unsafe { core::mem::zeroed() };
            // SAFETY: `lt_handle` and the plan's descriptors are live, `algo` is a
            // local the calls above configured, and `result` a local out-pointer.
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
    // SAFETY: `lt_handle` is the device's live handle and `plan`'s
    // descriptors were built for `call`'s `m`, `n` and `k`; `algo` is one of
    // its heuristics; `alpha`/`beta` are host f32 locals. `ws` is a device slab
    // of `ws_bytes` (or null and 0) and `stream` live, as `run_tactic`'s callers
    // hold. The operands are the caller's `call`, as for `gemm_ex`.
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

/// [`run_lt`] with cuBLASLt's own `epilogue` landing `y`: a bias (`bias`,
/// bf16, one a column of `y`, where the epilogue reads one) or a ReLU, through a descriptor of its own for the call. False where the
/// algorithm refuses the epilogue.
#[allow(clippy::too_many_arguments)]
fn run_lt_epilogue(
    lt_handle: lt::cublasLtHandle_t,
    plan: &LtPlan,
    algo: &lt::cublasLtMatmulAlgo_t,
    stream: *mut c_void,
    call: Call,
    epilogue: LtEpilogue,
    bias: u64,
    ws: *mut c_void,
    ws_bytes: usize,
) -> bool {
    if epilogue.biased() && bias == 0 {
        return false;
    }
    struct Desc(lt::cublasLtMatmulDesc_t);
    impl Drop for Desc {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: a non-null descriptor this guard alone owns, freed once.
                let _ = unsafe { lt::cublasLtMatmulDescDestroy(self.0) };
            }
        }
    }
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let mut desc = Desc(std::ptr::null_mut());
    // SAFETY: `desc.0` is a local out-pointer, freed by the guard.
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
               // SAFETY: `desc.0` was just created; every caller below passes a
               // pointer to a local of exactly `bytes` bytes, of the attribute's type,
               // which cuBLASLt copies.
               bytes: usize| unsafe {
        lt::cublasLtMatmulDescSetAttribute(desc.0, attribute, value, bytes) == ok
    };
    let (ta, tb) = (
        cublasOperation_t::CUBLAS_OP_T,
        cublasOperation_t::CUBLAS_OP_N,
    );
    let epilogue = match epilogue {
        LtEpilogue::Bias => lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        LtEpilogue::Relu => lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        LtEpilogue::ReluBias => lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
    };
    let biased = bias != 0;
    let bias_ptr = dev(bias);
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
    ) && (!biased
        || set(
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            std::ptr::from_ref(&bias_ptr).cast(),
            std::mem::size_of::<*const c_void>(),
        ) && set(
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
            std::ptr::from_ref(&bias_type).cast(),
            std::mem::size_of::<lt::cudaDataType>(),
        )))
    {
        return false;
    }
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: as in `run_lt`, with `desc.0` the descriptor just set up;
    // `bias`, where the epilogue reads one, is the caller's `n`-long bf16
    // bias, non-zero by the test above.
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
    /// `kernel.gemm.skinny_rows`: products of 2 to this many rows may time
    /// the skinny skeleton where it covers them.
    skinny_rows: u32,
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
            .expect("lock poisoned");
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
                skinny_rows: 0,
                plans: HashMap::new(),
                chosen: HashMap::new(),
                disk: DiskCache::new(),
            }))
        }))
    };
    let mut device = device.lock().expect("device lock poisoned");
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
        skinny_rows: u32,
    ) {
        if self.lt.workspace_bytes != workspace_bytes
            || self.heuristics != heuristics
            || self.narrow_n != narrow_n
            || self.split_ns != split_ns
            || self.rows_m != rows_m
            || self.split_from != split_from
            || self.skinny_rows != skinny_rows
        {
            self.skinny_rows = skinny_rows;
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
        // And one the skinny skeleton may serve apart from one it may not.
        let key = if skinny_tried(call, self.skinny_rows) {
            key ^ 0x736b_696e_0000_0000 ^ (u64::from(self.skinny_rows) << 16)
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
            self.skinny_rows,
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
    call.m >= 2 && call.m <= rows_m.min(ROWS_M) as i32 && call.k % 8 == 0
}

/// Whether the tuner times the skinny skeleton for `call`
/// (`kernel.gemm.skinny_rows`): from 2 rows (the gemv serves one), where it
/// covers the shape.
fn skinny_tried(call: Call, skinny_rows: u32) -> bool {
    call.m >= 2 && skinny::covers(call.m, call.n, call.k, 1, skinny_rows)
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
    skinny_rows: u32,
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
    if skinny_tried(call, skinny_rows) {
        candidates.push(Tactic::Skinny);
    }
    pick(handle, caller_stream, lt, plan, call, &candidates, split_ns)
}

/// Whether a captured step pays `tactic` a split's reduce launch of its own,
/// which a lone timing hides (`kernel.gemm.split_launch_ns`):
/// `cublasGemmEx` picks its own algorithm, a split one as likely, so it is
/// charged too, and an unfused leg as its product's tactic is.
fn charged(tactic: Tactic, plan: Option<&LtPlan>, call: Call) -> bool {
    match tactic {
        Tactic::Lt(i) | Tactic::LtEpilogue(i) => {
            plan.is_some_and(|p| splits_of(&p.heuristics[i].algo) > 1)
        }
        Tactic::GemmEx => true,
        Tactic::Unfused => call
            .anchor
            .and_then(|a| a.inner)
            .is_some_and(|inner| charged(inner, plan, call)),
        Tactic::Gemv | Tactic::Narrow | Tactic::Rows | Tactic::Skinny | Tactic::Skeleton(_) => {
            false
        }
    }
}

/// The fastest of `candidates` for `call`, each timed in the arena; of
/// those within a tie, the earliest.
fn pick(
    handle: cublasHandle_t,
    caller_stream: *mut c_void,
    lt: &LtCtx,
    plan: Option<&LtPlan>,
    call: Call,
    candidates: &[Tactic],
    split_ns: u32,
) -> Tactic {
    let mut arena = TuneArena::empty();
    if !arena.init(handle, caller_stream, lt.workspace_bytes, call) {
        return candidates[0];
    }
    let mut timings = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
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
        let charge = if charged(candidate, plan, call) {
            split_ns as f32 * 1e-6
        } else {
            0.0
        };
        timings.push((candidate, ms + charge, charge));
    }
    let fastest = timings
        .iter()
        .fold(f32::INFINITY, |best, &(_, ms, _)| best.min(ms));
    // One pass leaves a plateau of near-equal tactics ordered by noise: the
    // fastest few within reach of the fastest are timed again in turns (a
    // drift in the device's speed falling on each alike), and of those whose
    // medians tie the earliest (cuBLAS's own ranking) is kept.
    // The finalists are the fastest few within reach, not the first few:
    // a plateau of cuBLASLt heuristics would otherwise crowd out the
    // candidates timed after them. They keep the candidates' order, so a tie
    // still keeps the earliest.
    let mut reach: Vec<usize> = (0..timings.len())
        .filter(|&i| timings[i].1 <= fastest * FINAL_REACH)
        .collect();
    reach.sort_by(|&a, &b| timings[a].1.total_cmp(&timings[b].1));
    reach.truncate(FINALISTS);
    reach.sort_unstable();
    let mut finals: Vec<(Tactic, f32, Vec<f32>)> = reach
        .into_iter()
        .map(|i| {
            let (candidate, ms, charge) = timings[i];
            (candidate, charge, vec![ms])
        })
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
    // SAFETY: `algo` is an algorithm cuBLASLt returned; `splits` is an i32
    // local of the size passed and `written` a local.
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
    /// A paged epilogue's rows as the arena lands them: row `r` at page `r`
    /// and slot 0 of pages one row long (`0..m`, then `m` zeros, i32), so its
    /// caches are the arena's own tiles.
    ids: *mut c_void,
    /// The evictions' span alone, timed once.
    alone: std::cell::Cell<Option<f32>>,
}

impl Drop for TuneArena {
    fn drop(&mut self) {
        // SAFETY: `handle` is non-null only once `init` rebound the caller's
        // handle to this arena's stream, so this puts it back. Every other handle
        // is one `init` created and only this `Drop` frees; null ones never were.
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
            if !self.ids.is_null() {
                let _ = cudaFree(self.ids);
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
            ids: std::ptr::null_mut(),
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
        // An anchored region's products, then its results.
        let tiles = call.anchor.map_or(1, |a| {
            a.epi.parts() as usize * a.epi.product_bytes() / 2 + a.n_outs
        });
        let y_bytes = ((call.m as usize) * (call.n as usize) * 2 * tiles).max(1);
        self.workspace_bytes = workspace_bytes;
        self.caller_stream = caller_stream;
        self.flush_bytes = flush_bytes();
        // SAFETY: each call writes one field of `self` (all null from `empty`)
        // with a fresh allocation, stream or event that this arena alone owns and
        // `Drop` frees; the `cast` reinterprets `*mut *mut c_void` as the
        // `*mut cudaStream_t` it is.
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
        // SAFETY: `caller_stream` is the live stream of the `Ctx` that asked for this product.
        if unsafe { cudaStreamSynchronize(caller_stream.cast()) } != cudaError::cudaSuccess {
            clear_error();
            return false;
        }
        // SAFETY: `caller` is that `Ctx`'s live handle and `self.stream` the
        // stream just created; `Drop` puts the caller's stream back.
        if unsafe { cublasSetStream_v2(caller, self.stream.cast()) }
            != cublasStatus_t::CUBLAS_STATUS_SUCCESS
        {
            return false;
        }
        self.handle = caller;
        // SAFETY: `act` and `y` are the allocations just made, `act_bytes` and
        // `y_bytes` long, filled on the arena's own stream.
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
        if call.anchor.is_some_and(|a| a.epi.pages()) {
            let rows = call.m as usize;
            let mut ids: Vec<i32> = (0..i32::try_from(rows).unwrap_or(i32::MAX)).collect();
            ids.extend(std::iter::repeat_n(0, rows));
            // SAFETY: `ids` is a local out-pointer for a fresh allocation this
            // arena owns and `Drop` frees, `ids.len()` i32 long, filled from the
            // host vector (which outlives the copy: the stream is drained
            // before it drops) before anything reads it. The copy is the
            // arena's stream's, not the legacy stream's, which waits for every
            // blocking stream and so invalidates a capture another thread has
            // open on one (a model's arming beside this load's tuning).
            let landed = unsafe {
                cudaMalloc(&raw mut self.ids, ids.len() * 4) == cudaError::cudaSuccess
                    && cudaMemcpyAsync(
                        self.ids,
                        ids.as_ptr().cast(),
                        ids.len() * 4,
                        cudaMemcpyKind::cudaMemcpyHostToDevice,
                        self.stream.cast(),
                    ) == cudaError::cudaSuccess
                    && cudaStreamSynchronize(self.stream.cast()) == cudaError::cudaSuccess
            };
            if !landed {
                clear_error();
                return false;
            }
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
            y: at + (rows * 2 * K) as u64,
            m: 1,
            n: rows as i32,
            k: K as i32,
            anchor: None,
        };
        gemm_ex(
            self.handle,
            call,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        ) == cublasStatus_t::CUBLAS_STATUS_SUCCESS
    }

    /// `call` on the arena's planes: its activation, its product (an
    /// anchored region's products, then its results).
    fn bench(&self, call: Call) -> Call {
        let y = self.y.addr() as u64;
        let tile = (call.m as u64) * (call.n as u64) * 2;
        let anchor = call.anchor.map(|mut a| {
            let products = u64::from(a.epi.parts()) * tile * a.epi.product_bytes() as u64 / 2;
            for (r, out) in (0u64..).zip(a.outs.iter_mut().take(a.n_outs)) {
                *out = y + products + r * tile;
            }
            // A paged result's cache is its tile: row `r` at page `r`, slot 0.
            if a.epi.pages() {
                let ids = self.ids.addr() as u64;
                for k in 0..a.n_outs {
                    if let Some(paged) = a.epi.paged(k) {
                        a.ins[paged.page as usize] = In::row(ids).i32();
                        a.ins[paged.slot as usize] = In::row(ids + call.m as u64 * 4).i32();
                    }
                }
                a.paging = Some(Paging {
                    valid: 0,
                    page_size: 1,
                });
            }
            a
        });
        Call {
            act: self.act.addr() as u64,
            y,
            anchor,
            ..call
        }
    }

    fn time(
        &self,
        tactic: Tactic,
        lt_handle: lt::cublasLtHandle_t,
        plan: Option<&LtPlan>,
        call: Call,
    ) -> Option<f32> {
        let bench = self.bench(call);
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
            // SAFETY: the arena's own stream, created in `init` and live until `Drop`.
            let _ = unsafe { cudaStreamSynchronize(stream) };
            clear_error();
            return None;
        }
        // SAFETY: the arena's own stream, created in `init` and live until `Drop`.
        if unsafe { cudaStreamSynchronize(stream) } != cudaError::cudaSuccess {
            clear_error();
            return None;
        }
        // A decode's product streams a weight no cache holds and runs a
        // few microseconds, which the difference of two spans of evictions
        // cannot resolve (one read 0.2 us for a 7 MB weight, A100), and over
        // an L2 as large as an L40S's (96 MB) the span's weight stays
        // cached between its products, timing a split algorithm at half
        // the copy rate's floor: one eviction, then a run of products back
        // to back, each over a different slice of the slab, timed alone, up
        // to a decode's `STREAMED_ROWS` rows.
        if call.m <= STREAMED_ROWS
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
    /// last third cached); `None` when that half holds fewer than two.
    fn streamed(
        &self,
        tactic: Tactic,
        lt_handle: lt::cublasLtHandle_t,
        plan: Option<&LtPlan>,
        bench: Call,
    ) -> Option<f32> {
        let parts = bench.anchor.map_or(1, |a| a.parts()) as usize;
        let weight = parts * (bench.n as usize) * (bench.k as usize) * 2;
        let cold = self.flush_bytes / 2;
        let runs = (cold / weight.next_multiple_of(256).max(1)).min(STREAMED);
        if weight == 0 || runs < 2 {
            return None;
        }
        let stride = (cold / runs) & !255;
        let at = self.flush.addr() as u64;
        let stream = self.stream.cast();
        let mut graph: cudaGraph_t = std::ptr::null_mut();
        let mut exec: cudaGraphExec_t = std::ptr::null_mut();
        // SAFETY: `stream` and both events are the arena's own. Each run reads its
        // weight from `flush`: `runs * stride <= flush_bytes / 2` and `stride` is
        // at least the weight rounded up, so every slice is inside the slab.
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
            // SAFETY: ends the capture begun on the arena's stream; `graph` is a local out-pointer.
            unsafe { cudaStreamEndCapture(stream, &raw mut graph) } == cudaError::cudaSuccess;
        if !captured || !ended || graph.is_null() {
            if !graph.is_null() {
                // SAFETY: a non-null graph the capture above just returned, destroyed once.
                let _ = unsafe { cudaGraphDestroy(graph) };
            }
            // SAFETY: the arena's own stream, created in `init` and live until `Drop`.
            let _ = unsafe { cudaStreamSynchronize(stream) };
            clear_error();
            return None;
        }
        // SAFETY: `graph` is the live graph just captured; `exec` is a local out-pointer.
        let instantiated = unsafe { cudaGraphInstantiateWithFlags(&raw mut exec, graph, 0) }
            == cudaError::cudaSuccess;
        // SAFETY: the instantiated `exec` no longer needs `graph`, destroyed once here.
        let _ = unsafe { cudaGraphDestroy(graph) };
        if !instantiated || exec.is_null() {
            clear_error();
            return None;
        }
        let mut best: Option<f32> = None;
        for round in 0..ROUNDS {
            let mut ms = 0.0f32;
            // SAFETY: `exec` and `stream` are live, the events are the arena's own,
            // and `ms` is a local out-pointer.
            let timed = unsafe {
                cudaGraphLaunch(exec, stream) == cudaError::cudaSuccess
                    && cudaEventSynchronize(self.stop) == cudaError::cudaSuccess
                    && (round == 0
                        || cudaEventElapsedTime(&raw mut ms, self.start, self.stop)
                            == cudaError::cudaSuccess)
            };
            if !timed {
                // SAFETY: the arena's own stream, created in `init` and live until `Drop`.
                let _ = unsafe { cudaStreamSynchronize(stream) };
                // SAFETY: `exec` is idle after the sync and is destroyed once, here, before the return.
                let _ = unsafe { cudaGraphExecDestroy(exec) };
                clear_error();
                return None;
            }
            if round > 0 && best.is_none_or(|b| ms < b) {
                best = Some(ms);
            }
        }
        // SAFETY: the arena's own stream, created in `init` and live until `Drop`.
        let _ = unsafe { cudaStreamSynchronize(stream) };
        // SAFETY: `exec` is idle after the sync and is destroyed once, here.
        let _ = unsafe { cudaGraphExecDestroy(exec) };
        best.map(|ms| ms / runs as f32)
    }

    /// The least of a few timings (ms) of `REPS` evictions, each followed by
    /// `body`, captured as one graph.
    fn span(&self, body: &dyn Fn() -> bool) -> Option<f32> {
        let stream = self.stream.cast();
        let mut graph: cudaGraph_t = std::ptr::null_mut();
        let mut exec: cudaGraphExec_t = std::ptr::null_mut();
        // SAFETY: `stream` and both events are the arena's own, live until `Drop`.
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
            // SAFETY: ends the capture begun on the arena's stream; `graph` is a local out-pointer.
            unsafe { cudaStreamEndCapture(stream, &raw mut graph) } == cudaError::cudaSuccess;
        if !captured || !ended || graph.is_null() {
            if !graph.is_null() {
                // SAFETY: a non-null graph the capture above just returned, destroyed once.
                let _ = unsafe { cudaGraphDestroy(graph) };
            }
            // SAFETY: the arena's own stream, created in `init` and live until `Drop`.
            let _ = unsafe { cudaStreamSynchronize(stream) };
            clear_error();
            return None;
        }
        // SAFETY: `graph` is the live graph just captured; `exec` is a local out-pointer.
        let instantiated = unsafe { cudaGraphInstantiateWithFlags(&raw mut exec, graph, 0) }
            == cudaError::cudaSuccess;
        // SAFETY: the instantiated `exec` no longer needs `graph`, destroyed once here.
        let _ = unsafe { cudaGraphDestroy(graph) };
        if !instantiated || exec.is_null() {
            clear_error();
            return None;
        }
        let mut best: Option<f32> = None;
        for round in 0..ROUNDS {
            // SAFETY: `exec` was instantiated above and `stream` is the arena's own.
            if unsafe { cudaGraphLaunch(exec, stream) } != cudaError::cudaSuccess
                // SAFETY: `stop` is the arena's own event.
                || unsafe { cudaEventSynchronize(self.stop) } != cudaError::cudaSuccess
            {
                // SAFETY: the arena's own stream, created in `init` and live until `Drop`.
                let _ = unsafe { cudaStreamSynchronize(stream) };
                // SAFETY: `exec` is idle after the sync and is destroyed once, before the return.
                let _ = unsafe { cudaGraphExecDestroy(exec) };
                clear_error();
                return None;
            }
            if round == 0 {
                continue;
            }
            let mut ms = 0.0f32;
            // SAFETY: both events are the arena's own and `stop` has completed; `ms` is a local.
            if unsafe { cudaEventElapsedTime(&raw mut ms, self.start, self.stop) }
                != cudaError::cudaSuccess
            {
                // SAFETY: `exec` is idle (`stop`, its last node, completed) and is destroyed once, before the return.
                let _ = unsafe { cudaGraphExecDestroy(exec) };
                clear_error();
                return None;
            }
            if best.is_none_or(|b| ms < b) {
                best = Some(ms);
            }
        }
        // SAFETY: the arena's own stream, created in `init` and live until `Drop`.
        let _ = unsafe { cudaStreamSynchronize(stream) };
        // SAFETY: `exec` is idle after the sync and is destroyed once, here.
        let _ = unsafe { cudaGraphExecDestroy(exec) };
        best
    }
}

/// Evicted products a timed span, and the spans timed (the first a warm-up).
const REPS: usize = 4;
const ROUNDS: usize = 3;

/// Products a streamed span runs back to back.
const STREAMED: usize = 16;

/// The most rows a product is timed streamed: a decode's, its weight read
/// from DRAM in a step.
const STREAMED_ROWS: i32 = 64;

fn flush_bytes() -> usize {
    const FALLBACK: usize = 64 << 20;
    let mut device: i32 = 0;
    // SAFETY: `device` is a local out-pointer.
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
    // SAFETY: `device` is a local out-pointer.
    if unsafe { cudaGetDevice(&raw mut device) } != cudaError::cudaSuccess {
        clear_error();
        return String::new();
    }
    let Some(prop) = crate::jit::device::properties(device) else {
        clear_error();
        return String::new();
    };
    let mut version: i32 = 0;
    // SAFETY: `version` is a local out-pointer. The handle is null: this
    // leans on cuBLAS reading the library's version without a handle.
    let _ = unsafe { cublasGetVersion_v2(std::ptr::null_mut(), &raw mut version) };
    // The driver's name, read up to its NUL without trusting one is there.
    let name: Vec<u8> = prop
        .name
        .iter()
        .map(|&c| u8::from_ne_bytes(c.to_ne_bytes()))
        .take_while(|&b| b != 0)
        .collect();
    let name = String::from_utf8_lossy(&name).into_owned();
    format!(
        "# pie-dense-gemm v10 sm{}{} cublas={version} dev={name}",
        prop.major, prop.minor
    )
}
