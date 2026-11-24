#![allow(clippy::too_many_arguments)]
#![allow(unsafe_code)]

//! The dense bf16 products, nothing timed and no table read: few rows a
//! streaming skeleton covers run on it, the epilogue fused (the gemv one
//! row; the rows skeleton up to its tile of 16 rows of one part, a
//! structural bound), the kernel streaming the weight once through every
//! row; every other
//! runs cuBLASLt's first heuristic algorithm for its shape
//! (its own ranking, deterministic for the shape, the workspace and the
//! library), an anchored region with an epilogue cuBLASLt lands on that
//! algorithm with it, else the product then its epilogue on the row
//! skeleton.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::Error;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmEx, cublasHandle_t,
    cublasOperation_t, cublasSetWorkspace_v2, cublasStatus_t, cudaDataType,
};
use cudarc::cublaslt::sys as lt;
use cudarc::runtime::sys::{cudaGetDevice, cudaGetLastError};

use crate::jit::{Ctx, refuse};
use crate::linear::skeleton::LtEpilogue;
use crate::linear::skeleton::{Epilogue, In, MAX_INS, MAX_OUTS, Paging, RowLeg, Skeleton};

/// The skeleton that serves `m x n x k` with `epi`, its epilogue fused: the
/// gemv (one row) or the rows skeleton (up to its 16-row tile, one part),
/// the first that covers it, each a warp streaming a weight row through
/// every activation row; past them the skinny skeleton where it covers the
/// product and tiles its rows tighter than cuBLASLt would ([`skinny_rows`]).
/// `None` where none does: the product runs on cuBLASLt.
fn streamed(ctx: &Ctx, epi: &Epilogue, w: u64, m: i32, n: i32, k: i32) -> Option<Skeleton> {
    let tuning = ctx.tuning();
    [Skeleton::Gemv, Skeleton::Rows]
        .into_iter()
        .find(|s| s.covers(epi, m, n, k, &tuning))
        .or_else(|| {
            let rows = u32::try_from(m).ok()?;
            let cols = u32::try_from(n).ok()?.checked_mul(epi.parts())?;
            let depth = u32::try_from(k).ok()?;
            if !Skeleton::Skinny.covers(epi, m, n, k, &tuning) {
                return None;
            }
            let stated = skinny_rows(&tuning, rows, cols, depth).is_some();
            // Where the device's measured product times are laid (a served
            // load's), they choose among the candidates, the tile rule's
            // stated first; unmeasured, the stated one serves, noted.
            let Some(routes) = ctx.routes() else {
                return stated.then_some(Skeleton::Skinny);
            };
            let listed = candidates(&tuning, rows, cols, depth, stated);
            let codes: Vec<u32> = listed.iter().map(|c| c.code).collect();
            let shape = crate::linear::routes::Shape {
                kind: crate::linear::routes::Kind::Plain,
                rows: cols,
                k: depth,
                m: rows,
            };
            let picked = routes.pick(shape, w, &codes);
            let chosen = listed.iter().find(|c| c.code == picked)?;
            match chosen.serves {
                Serves::Skinny => Some(Skeleton::Skinny),
                Serves::Lt(heuristic) => {
                    lay_lt_pick(rows, cols, depth, heuristic);
                    None
                }
            }
        })
}

/// The rows the skinny skeleton pads `m` to, where it runs a dense product
/// of `m x n x k` in place of cuBLASLt: past the rows skeleton's 16-row tile
/// both stream the weight once over whole mma tiles of rows, so the one
/// whose tile pads the rows least does the least work. The skinny skeleton
/// tiles `m` in mma.sync's 16 rows within its one 64-row block; cuBLASLt's
/// first heuristic algorithm ([`tile`]) pads to its tile's rows. `Some`
/// where the skinny tile is the tighter (or cuBLASLt offers none), derived
/// from the two tiles: nothing timed. Up to the rows skeleton's tile a
/// product stays the streaming skeletons' or the library's few-row kernels'
/// (which split K to fill the device); the skeleton's coverage
/// (`kernel.gemm.skinny_rows`, the chip's own) bounds whether it runs at all.
#[must_use]
pub fn skinny_rows(tuning: &crate::tuning::Tuning, m: u32, n: u32, k: u32) -> Option<u32> {
    let skinny = m.next_multiple_of(16);
    if m <= crate::linear::skeleton::ROWS_M
        || skinny > crate::linear::skinny::ROWS.min(tuning.gemm_skinny_rows())
    {
        return None;
    }
    match tile_rows(tuning, m, n, k) {
        Some(rows) if m.next_multiple_of(rows) <= skinny => None,
        _ => Some(skinny),
    }
}

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
}

/// `y = act . w^T` over bf16 operands, on cuBLASLt's first heuristic
/// algorithm for the shape.
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
    let call = Call { act, w, y, m, n, k };
    let tuning = ctx.tuning();
    // A product bound by its weight's read streams it once.
    if let Some(skeleton) = streamed(ctx, Epilogue::store(), w, m, n, k) {
        return skeleton.fire(ctx, Epilogue::store(), w, act, &[], &[y], None, m, n, k);
    }
    // A prefill's wide product on cuBLAS's own algorithm with a split's
    // partials summed in bf16, as torch runs it by default
    // (`kernel.gemm.bf16_split_from`); the handle is set back at once.
    if tuning.gemm_bf16_split_from() > 0
        && m >= i32::try_from(tuning.gemm_bf16_split_from()).unwrap_or(i32::MAX)
    {
        // SAFETY: `handle` is this context's live cuBLAS handle.
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                handle,
                cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
            )
        };
        let status = gemm_ex(
            handle,
            call,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        );
        // SAFETY: as above.
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                handle,
                cudarc::cublas::sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(refuse(
                op,
                format!("`cublasGemmEx` answered {status:?} at M={m} N={n} K={k}"),
            ));
        }
        return Ok(());
    }
    let (plan, lt_handle, want) = with_device(|device| {
        device.configure((tuning.gemm_workspace_mb() as usize) << 20);
        (
            device.plan_for(call),
            device.lt.handle,
            device.lt.workspace_bytes,
        )
    });
    let (ws, ws_bytes) = lt_workspace(ctx, op, want);
    // cuBLASLt's first heuristic where it offers one this context's
    // workspace holds, else cuBLAS's own algorithm: chosen up front, its
    // failure refused by name.
    if let Some(plan) = plan.as_deref()
        && let Some(heuristic) = plan.heuristics.first()
        && !lt_handle.is_null()
        && heuristic.workspaceSize <= ws_bytes
    {
        if !run_lt(
            lt_handle,
            plan,
            &heuristic.algo,
            ctx.stream(),
            call,
            ws,
            ws_bytes,
        ) {
            clear_error();
            return Err(refuse(
                op,
                format!("cuBLASLt's first heuristic failed at M={m} N={n} K={k}"),
            ));
        }
        return Ok(());
    }
    // SAFETY: `handle` is this context's live cuBLAS handle, bound to its
    // stream; `ws` is a device slab of `ws_bytes` the context owns.
    unsafe {
        cublasSetWorkspace_v2(handle, ws, ws_bytes);
    }
    let status = gemm_ex(
        handle,
        call,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(refuse(
            op,
            format!(
                "cuBLASLt has no algorithm and `cublasGemmEx` answered {status:?} at M={m} N={n} K={k}"
            ),
        ));
    }
    Ok(())
}

/// The cuBLASLt workspace a product runs with: the stream's slab of `want`
/// bytes, or none on a context that lays no such slab (a heuristic that
/// needs one is then not chosen).
fn lt_workspace(ctx: &Ctx, op: &'static str, want: usize) -> (*mut c_void, usize) {
    match ctx.scratch_stream(op, LT_WORKSPACE, want) {
        Ok(ws) if !ws.is_null() => (ws, want),
        _ => (std::ptr::null_mut(), 0),
    }
}

/// `outs = epi(act . w^T)`, an anchored region over a dense bf16 bank of
/// `parts · n` rows: on cuBLASLt's own epilogue where it lands it (one
/// result), else the plain product then the epilogue on the row skeleton.
/// False, landing nothing, only where the row skeleton does not carry that
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
    if let Some(skeleton) = streamed(ctx, epi, w, m, n, k) {
        skeleton.fire(ctx, epi, w, act, ins, outs, paging, m, n, k)?;
        return Ok(true);
    }
    if let [out] = outs
        && epi.parts() == 1
        && on_lt(ctx, op, epi, act, w, ins, *out, m, n, k)?
    {
        return Ok(true);
    }
    unfused(ctx, op, epi, act, w, ins, outs, paging, m, n, k)
}

/// `out = epi(act . w^T)` on cuBLASLt's own epilogue alone, at cuBLASLt's
/// first heuristic algorithm for the product with that epilogue within the
/// workspace the context holds. False, landing nothing, where the epilogue
/// is none of cuBLASLt's or cuBLASLt offers no algorithm for it; the chosen
/// one failing is a fault.
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
    };
    let (plan, lt_handle, want) = with_device(|device| {
        device.configure((tuning.gemm_workspace_mb() as usize) << 20);
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
    let (ws, ws_bytes) = lt_workspace(ctx, op, want);
    let bias = ins.first().map_or(0, |b| b.base());
    run_lt_epilogue(
        lt_handle,
        &plan,
        ctx.stream(),
        call,
        epilogue,
        bias,
        ws,
        ws_bytes,
    )
    .map_err(|why| refuse(op, format!("cuBLASLt's epilogue {why} at M={m} N={n} K={k}")))
}

/// `outs = epi(act . w^T)` on the unfused leg: the plain product into
/// scratch, then the
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
        let ins: Vec<In> = ins.iter().map(|input| input.at_row(from)).collect();
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
            valid: if p.valid == 0 {
                0
            } else {
                p.valid + from as u64
            },
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

/// `y = act . w^T` as `epi` reads its products: rounded to bf16, or its f32
/// sums.
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

/// The most scratch the unfused leg's products take at once.
const UNFUSED_BYTES: usize = 256 << 20;

/// The plane an anchored region's unfused leg lands its products in.
const ANCHOR_PRODUCTS: &str = "linear.anchor_products";

fn gemm_ex(handle: cublasHandle_t, call: Call, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: `handle` is a live cuBLAS handle (a `Ctx`'s, or the arena's
    // borrowed one) and `alpha`/`beta` are host f32 locals, as the default
    // pointer mode and the 32F compute type take. `call` names an `m x k`
    // act, an `n x k` weight and an `m x n` y, all bf16: the caller's
    // tensors, or the arena's slabs allocated at those sizes.
    if crate::sizing::active() {
        // A size-only walk (`crate::sizing`) runs no product.
        crate::sizing::counted();
        cublasStatus_t::CUBLAS_STATUS_SUCCESS
    } else {
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
    let call = Call { act, w, y, m, n, k };
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
    if crate::sizing::active() {
        // A size-only walk (`crate::sizing`) runs no product.
        crate::sizing::counted();
        cublasStatus_t::CUBLAS_STATUS_SUCCESS
    } else {
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

/// `m x n x k`'s plan: its descriptors and cuBLASLt's first heuristic
/// algorithm for it (cuBLASLt's own ranking, deterministic for the shape,
/// the workspace and the library), its K splits' partials summed in f32.
fn build_lt_plan(
    lt_handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
    m: i32,
    n: i32,
    k: i32,
) -> Option<LtPlan> {
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
    plan.heuristics = vec![first_heuristic(
        lt_handle,
        workspace_bytes,
        plan.op_desc,
        &plan,
    )?];
    Some(plan)
}

/// cuBLASLt's first heuristic algorithm for the product `op_desc` states
/// (the plain product, or one with its epilogue set) over `plan`'s layouts,
/// within `workspace_bytes`, a K split's partials summed in f32: its own
/// ranking, deterministic for the shape, the epilogue, the workspace and the
/// library. `None` where it offers none.
fn first_heuristic(
    lt_handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
    op_desc: lt::cublasLtMatmulDesc_t,
    plan: &LtPlan,
) -> Option<lt::cublasLtMatmulHeuristicResult_t> {
    heuristics(lt_handle, workspace_bytes, op_desc, plan, 1)?.into_iter().next()
}

/// cuBLASLt's first `count` heuristic algorithms for the product, in its own
/// ranking (as [`first_heuristic`]).
fn heuristics(
    lt_handle: lt::cublasLtHandle_t,
    workspace_bytes: usize,
    op_desc: lt::cublasLtMatmulDesc_t,
    plan: &LtPlan,
    count: usize,
) -> Option<Vec<lt::cublasLtMatmulHeuristicResult_t>> {
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
        vec![unsafe { core::mem::zeroed() }; count.max(1)];
    let mut returned: i32 = 0;
    // SAFETY: `lt_handle` is the device's live handle, the descriptors and
    // `pref.0` were created above, and `heuristics` has room for the count
    // asked for; `returned` is a local out-pointer.
    if unsafe {
        lt::cublasLtMatmulAlgoGetHeuristic(
            lt_handle,
            op_desc,
            plan.a_desc,
            plan.b_desc,
            plan.c_desc,
            plan.c_desc,
            pref.0,
            i32::try_from(count.max(1)).unwrap_or(1),
            heuristics.as_mut_ptr(),
            &raw mut returned,
        )
    } != ok
        || returned <= 0
    {
        return None;
    }
    heuristics.truncate(usize::try_from(returned).unwrap_or(0));
    Some(heuristics)
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
    // of `ws_bytes` (or null and 0) and `stream` live, as `act_x_wt`'s callers
    // hold. The operands are the caller's `call`, as for `gemm_ex`.
    let status = if crate::sizing::active() {
        // A size-only walk (`crate::sizing`) runs no product.
        crate::sizing::counted();
        lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
    } else {
        unsafe {
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
        }
    };
    status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

/// [`run_lt`] with cuBLASLt's own `epilogue` landing `y`: a bias (`bias`,
/// bf16, one a column of `y`, where the epilogue reads one) or a ReLU,
/// through a descriptor of its own for the call. False, landing nothing,
/// where no bias is given for a biased epilogue or cuBLASLt offers no
/// algorithm for it; the why where the chosen one fails.
#[allow(clippy::too_many_arguments)]
fn run_lt_epilogue(
    lt_handle: lt::cublasLtHandle_t,
    plan: &LtPlan,
    stream: *mut c_void,
    call: Call,
    epilogue: LtEpilogue,
    bias: u64,
    ws: *mut c_void,
    ws_bytes: usize,
) -> Result<bool, String> {
    if epilogue.biased() && bias == 0 {
        return Ok(false);
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
        return Err("descriptor could not be created".into());
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
        return Err("descriptor could not be set".into());
    }
    // cuBLASLt's first heuristic for the product with its epilogue, asked
    // once a shape and kept: the plain product's may not take it.
    let key = (
        call.m,
        call.n,
        call.k,
        epilogue_code(epilogue),
        biased,
        ws_bytes,
    );
    let Some(algo) = with_device(|device| {
        *device
            .epilogues
            .entry(key)
            .or_insert_with(|| first_heuristic(lt_handle, ws_bytes, desc.0, plan).map(|h| h.algo))
    }) else {
        clear_error();
        return Ok(false);
    };
    let algo = &algo;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: as in `run_lt`, with `desc.0` the descriptor just set up;
    // `bias`, where the epilogue reads one, is the caller's `n`-long bf16
    // bias, non-zero by the test above.
    let status = if crate::sizing::active() {
        // A size-only walk (`crate::sizing`) runs no product.
        crate::sizing::counted();
        lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
    } else {
        unsafe {
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
        }
    };
    if status != ok {
        clear_error();
        return Err(format!("answered {status:?}"));
    }
    Ok(true)
}

struct Device {
    lt: LtCtx,
    plans: HashMap<(i32, i32, i32), Arc<LtPlan>>,
    /// cuBLASLt's first heuristic for a product with an epilogue, by its
    /// shape, its epilogue ([`epilogue_code`]), whether it reads a bias and
    /// the workspace it was asked within; `None` where it offers none.
    epilogues: HashMap<(i32, i32, i32, u32, bool, usize), Option<lt::cublasLtMatmulAlgo_t>>,
    /// The cuBLASLt algorithm the routes picked for a plain product, by its
    /// shape (`streamed`): its plan runs it in place of the first.
    lt_picks: HashMap<(i32, i32, i32), lt::cublasLtMatmulHeuristicResult_t>,
    /// The candidates of a decode product, by shape (`candidates`).
    listed: HashMap<(u32, u32, u32, bool), Arc<Vec<Candidate>>>,
}

/// An epilogue as the heuristic cache keys it.
fn epilogue_code(epilogue: lt::cublasLtEpilogue_t) -> u32 {
    epilogue as u32
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
                plans: HashMap::new(),
                epilogues: HashMap::new(),
                lt_picks: HashMap::new(),
                listed: HashMap::new(),
            }))
        }))
    };
    let mut device = device.lock().expect("device lock poisoned");
    f(&mut device)
}

impl Device {
    /// Lay a load's workspace on this device (`kernel.gemm.workspace_mb`):
    /// plans chosen under another are dropped, since an algorithm may need
    /// more.
    fn configure(&mut self, workspace_bytes: usize) {
        if self.lt.workspace_bytes != workspace_bytes {
            self.lt.workspace_bytes = workspace_bytes;
            self.plans.clear();
            self.epilogues.clear();
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
        let mut plan = build_lt_plan(self.lt.handle, self.lt.workspace_bytes, m, n, k)?;
        // The algorithm the routes picked for the shape, where they picked
        // one past the first.
        if let Some(&heuristic) = self.lt_picks.get(&(m, n, k)) {
            plan.heuristics = vec![heuristic];
        }
        let plan = Arc::new(plan);
        self.plans.insert((m, n, k), Arc::clone(&plan));
        Some(plan)
    }
}

/// The rows of the tile a dense bf16 product of `m x n x k` (`act . w^T`)
/// runs on this device under `tuning`'s workspace: cuBLASLt's first
/// heuristic algorithm's ([`tile`]). `None` where cuBLASLt offers none, or
/// names a tile this does not know.
#[must_use]
pub fn tile_rows(tuning: &crate::tuning::Tuning, m: u32, n: u32, k: u32) -> Option<u32> {
    tile(tuning, m, n, k).map(|t| t.rows)
}

/// The tile cuBLASLt's first heuristic algorithm runs a dense bf16 product
/// on, and the waves it fills.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tile {
    /// The tile's rows of the product (cuBLASLt's tile is `n`-major: its
    /// `N` is the product's rows).
    pub rows: u32,
    /// The tile's columns of the product.
    pub cols: u32,
    /// The slices K is split into (1: none).
    pub split_k: u32,
    /// The waves its grid fills on this device, as cuBLASLt's heuristic
    /// states them for the algorithm at the shape (`wavesCount`: its tiles
    /// over the blocks the device holds at once of its kernel).
    pub waves: f32,
}

/// [`Tile`] of a dense bf16 product of `m x n x k` on this device under
/// `tuning`'s workspace. `None` where cuBLASLt offers no algorithm, or names
/// a tile this does not know.
#[must_use]
pub fn tile(tuning: &crate::tuning::Tuning, m: u32, n: u32, k: u32) -> Option<Tile> {
    let call = Call {
        act: 0,
        w: 0,
        y: 0,
        m: i32::try_from(m).ok()?,
        n: i32::try_from(n).ok()?,
        k: i32::try_from(k).ok()?,
    };
    let plan = with_device(|device| {
        device.configure((tuning.gemm_workspace_mb() as usize) << 20);
        device.plan_for(call)
    })?;
    let first = plan.heuristics.first()?;
    let algo = &first.algo;
    let config = |attribute| -> Option<u32> {
        let mut value: u32 = 0;
        let mut written: usize = 0;
        // SAFETY: `algo` is an algorithm cuBLASLt returned; `value` is a
        // u32 local of the size passed (each attribute asked is a u32 or
        // an enum of one) and `written` a local.
        let asked = unsafe {
            lt::cublasLtMatmulAlgoConfigGetAttribute(
                std::ptr::from_ref(algo),
                attribute,
                std::ptr::from_mut(&mut value).cast(),
                std::mem::size_of::<u32>(),
                &raw mut written,
            )
        };
        if asked == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Some(value)
        } else {
            clear_error();
            None
        }
    };
    let id = config(lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_TILE_ID)?;
    // `cublasLtMatmulTile_t` is `repr(u32)` over 0 to its END, every value
    // between a variant, so an id below END is one; its name spells the tile.
    if id >= lt::cublasLtMatmulTile_t::CUBLASLT_MATMUL_TILE_END as u32 {
        return None;
    }
    // SAFETY: `id` is below END, and every value from 0 to END is a variant.
    let named: lt::cublasLtMatmulTile_t = unsafe { std::mem::transmute(id) };
    let name = format!("{named:?}");
    let (cols, rows) = name
        .strip_prefix("CUBLASLT_MATMUL_TILE_")?
        .split_once('x')?;
    let split_k =
        config(lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?;
    Some(Tile {
        rows: rows.parse().ok().filter(|&r| r > 0)?,
        cols: cols.parse().ok().filter(|&c| c > 0)?,
        split_k: split_k.max(1),
        waves: first.wavesCount,
    })
}

// ---------------------------------------------------------------------------
// A decode product's candidates (`crate::linear::routes`): the skinny
// skeleton and cuBLASLt's first algorithms, each under its code.

/// cuBLASLt's ranked algorithms a decode product's candidates hold.
const LT_CANDIDATES: usize = 2;

/// What a candidate runs.
#[derive(Clone, Copy)]
enum Serves {
    Skinny,
    Lt(lt::cublasLtMatmulHeuristicResult_t),
}

/// One kernel that can serve a decode product, under its code.
#[derive(Clone, Copy)]
struct Candidate {
    code: u32,
    serves: Serves,
}

// SAFETY: a heuristic result is plain data (an algorithm's opaque words, a
// status, sizes and a float); nothing in it points anywhere.
unsafe impl Send for Candidate {}
unsafe impl Sync for Candidate {}

/// The device and driver every code folds in, besides the profile's own
/// device: an entry measured under another driver or arch is not this one's.
fn platform() -> &'static str {
    static PLATFORM: OnceLock<String> = OnceLock::new();
    PLATFORM.get_or_init(|| {
        let mut driver: i32 = 0;
        // SAFETY: a local out-parameter.
        let _ = unsafe { cudarc::runtime::sys::cudaDriverGetVersion(&raw mut driver) };
        // SAFETY: no arguments.
        let lt_version = unsafe { lt::cublasLtGetVersion() };
        format!(
            "{}|driver {driver}|cublaslt {lt_version}",
            crate::jit::cache::arch().unwrap_or("?")
        )
    })
}

/// The skinny skeleton's code: its unit's compile key, every source byte
/// and option the kernel is built from.
fn skinny_code() -> u32 {
    static CODE: OnceLock<u32> = OnceLock::new();
    *CODE.get_or_init(|| {
        let kernel = crate::linear::skinny::kernel(Epilogue::store());
        let fingerprint = kernel.root.as_ref().map_or_else(
            || format!("skinny {}", kernel.name),
            |root| root.key(kernel.name, crate::jit::cache::arch().unwrap_or("?")),
        );
        crate::linear::routes::code(platform(), &fingerprint)
    })
}

/// The words that name a cuBLASLt algorithm's code: its id and the config
/// its kernel is built from.
fn algo_words(algo: &lt::cublasLtMatmulAlgo_t) -> String {
    use lt::cublasLtMatmulAlgoConfigAttributes_t as A;
    let mut words = Vec::new();
    for attribute in [
        A::CUBLASLT_ALGO_CONFIG_ID,
        A::CUBLASLT_ALGO_CONFIG_TILE_ID,
        A::CUBLASLT_ALGO_CONFIG_STAGES_ID,
        A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        A::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        A::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        A::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
        A::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
    ] {
        let mut value: u32 = 0;
        let mut written: usize = 0;
        // SAFETY: `algo` is an algorithm cuBLASLt returned; `value` and
        // `written` are locals of the sizes passed.
        let asked = unsafe {
            lt::cublasLtMatmulAlgoConfigGetAttribute(
                std::ptr::from_ref(algo),
                attribute,
                std::ptr::from_mut(&mut value).cast(),
                std::mem::size_of::<u32>(),
                &raw mut written,
            )
        };
        if asked == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            words.push(value.to_string());
        } else {
            clear_error();
            words.push("-".to_owned());
        }
    }
    words.join(".")
}

/// A plain dense product's candidates at `m x n x k`, the stated one first
/// (the skinny skeleton where `skinny_first`, the tile rule's choice, else
/// cuBLASLt's first algorithm), asked once a shape and kept.
fn candidates(
    tuning: &crate::tuning::Tuning,
    m: u32,
    n: u32,
    k: u32,
    skinny_first: bool,
) -> Arc<Vec<Candidate>> {
    with_device(|device| {
        if let Some(listed) = device.listed.get(&(m, n, k, skinny_first)) {
            return Arc::clone(listed);
        }
        device.configure((tuning.gemm_workspace_mb() as usize) << 20);
        let mut lts = Vec::new();
        if device.lt.ensure()
            && let (Ok(mi), Ok(ni), Ok(ki)) = (i32::try_from(m), i32::try_from(n), i32::try_from(k))
            && let Some(plan) = build_lt_plan(device.lt.handle, device.lt.workspace_bytes, mi, ni, ki)
        {
            lts = heuristics(
                device.lt.handle,
                device.lt.workspace_bytes,
                plan.op_desc,
                &plan,
                LT_CANDIDATES,
            )
            .unwrap_or_default();
        }
        let skinny = Candidate { code: skinny_code(), serves: Serves::Skinny };
        let lt: Vec<Candidate> = lts
            .into_iter()
            .map(|heuristic| Candidate {
                code: crate::linear::routes::code(
                    platform(),
                    &format!("cublaslt {}", algo_words(&heuristic.algo)),
                ),
                serves: Serves::Lt(heuristic),
            })
            .collect();
        let mut listed = Vec::with_capacity(lt.len() + 1);
        if skinny_first || lt.is_empty() {
            listed.push(skinny);
            listed.extend(lt);
        } else {
            listed.push(lt[0]);
            listed.push(skinny);
            listed.extend(lt.into_iter().skip(1));
        }
        let listed = Arc::new(listed);
        device.listed.insert((m, n, k, skinny_first), Arc::clone(&listed));
        listed
    })
}

/// The plain product of `m x n x k` runs on `heuristic` from here on (the
/// routes' pick): its plan is laid anew where it ran another.
fn lay_lt_pick(m: u32, n: u32, k: u32, heuristic: lt::cublasLtMatmulHeuristicResult_t) {
    let (Ok(m), Ok(n), Ok(k)) = (i32::try_from(m), i32::try_from(n), i32::try_from(k)) else {
        return;
    };
    with_device(|device| {
        let same = device
            .lt_picks
            .get(&(m, n, k))
            .is_some_and(|held| algo_words(&held.algo) == algo_words(&heuristic.algo));
        if !same {
            device.lt_picks.insert((m, n, k), heuristic);
            device.plans.remove(&(m, n, k));
        }
    });
}
