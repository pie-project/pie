#![allow(clippy::too_many_arguments)]
#![allow(unsafe_code)]

//! `y = act · wᵀ` for a bf16 projection: the skinny kernel for a decode's
//! few rows (the gemv where it refuses), hipBLASLt's heuristic pick above
//! them (what torch and vLLM run on ROCm), and rocBLAS where hipBLASLt is
//! absent or offers nothing.

use crate::error::Error;
use crate::hip::blas::{self, Operand};
use crate::hip::{lt, sys};
use crate::jit::{Ctx, refuse};

use super::dense_batch::dense_batch_bf16;
use super::gemv::{
    SkinnyEpilogue, gemv_bf16, skinny_bf16, skinny_f32, skinny_swiglu_bf16, skinnyt, turn_plan,
};

/// Up to eight rows the skinny kernels always take a small weight.
const SKINNY_SMALL: i32 = 8;
const SKINNY_SMALL_WEIGHT: i64 = 1 << 21;
/// The LDS the skinny kernels stage their activation rows in, and the
/// fewest bank rows their whole-product kernel takes (`linear/gemv.rs`).
const SKINNY_LDS: i64 = 64 << 10;
const SKINNY_QUAD_ROWS: i32 = super::gemv::QUAD_MIN_ROWS as i32;

/// Who takes `m` rows over a bank of `rows` weight rows (`2n` for a packed
/// `[gate | up]` bank): the plain projection and its fused GLU ask the same,
/// so the two land the same bits.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    /// The turned kernel (`kernel.gemm.turn_*`; past 16 rows, an lm head).
    Turn,
    /// The skinny kernels (`linear/gemv.hip`).
    Skinny,
    /// The decode-batch engine (`kernel.dense.batch_*`, or up to eight rows
    /// over a bank at least `kernel.gemm.skinny_batch_min_n` wide).
    Batch,
    /// What follows: the gemv, hipBLASLt, rocBLAS.
    Other,
}

fn route(ctx: &Ctx, m: i32, rows: i32, k: i32) -> Route {
    let t = ctx.tuning();
    let until = if rows.unsigned_abs() >= t.gemm_skinny_batch_min_n {
        t.gemm_turn_wide_until
    } else {
        t.gemm_turn_until.min(16)
    };
    if t.gemm_turn_from > 0
        && (t.gemm_turn_from as i32..=until as i32).contains(&m)
        && rows.unsigned_abs() >= t.gemm_turn_min_n
        && turn_plan(m.unsigned_abs(), rows.unsigned_abs(), k.unsigned_abs()).is_some()
    {
        return Route::Turn;
    }
    if m <= SKINNY_SMALL && i64::from(rows) * i64::from(k) <= SKINNY_SMALL_WEIGHT {
        return Route::Skinny;
    }
    // Past two rows an activation beyond LDS (a long K) costs the diagonal
    // kernel its L2 rereads, where hipBLASLt still tiles a short bank faster;
    // a bank of the whole-product kernel's width keeps it.
    let reread = m > 2 && 2 * i64::from(m) * i64::from(k) > SKINNY_LDS && rows < SKINNY_QUAD_ROWS;
    if m <= t.gemm_skinny_dense_rows as i32 && !reread {
        return Route::Skinny;
    }
    // Past the skinny rows an lm_head-wide bank streams best on the batch
    // engine; others take
    // hipBLASLt unless the batch window holds them.
    if batch_on(ctx, m, rows, k)
        || (m <= SKINNY_SMALL && rows.unsigned_abs() >= t.gemm_skinny_batch_min_n)
    {
        Route::Batch
    } else {
        Route::Other
    }
}

/// Returns whether the bias was folded into the store (only the gemv folds
/// one); `false` leaves the bias for the caller to add.
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
    let way = route(ctx, m, n, k);
    if bias == 0
        && way == Route::Turn
        && skinnyt(ctx, w, act, y, m, n, k, SkinnyEpilogue::Store).is_ok()
    {
        return Ok(false);
    }
    if bias == 0 && way == Route::Skinny && skinny_bf16(ctx, w, act, y, m, n, k).is_ok() {
        return Ok(false);
    }
    if m == 1 && gemv_bf16(ctx, w, act, bias, y, n, k).is_ok() {
        return Ok(bias != 0);
    }
    if bias == 0 && way == Route::Batch && dense_batch_bf16(ctx, w, act, y, m, n, k, false)? {
        return Ok(false);
    }
    let shape = lt::Shape {
        m: m.unsigned_abs(),
        n: n.unsigned_abs(),
        k: k.unsigned_abs(),
        bias: false,
        groups: 1,
        f32_out: false,
    };
    let workspace = ctx.scratch_shared(op, "linear.lt_workspace", lt::WORKSPACE)? as u64;
    // SAFETY: the three planes are device ranges the caller sized at [m, k],
    // [n, k] and [m, n]; the workspace is this stream's.
    let landed = unsafe { lt::matmul(shape, act, w, 0, y, workspace, ctx.stream()) };
    if landed.map_err(|why| refuse(op, format!("{why} at M={m} N={n} K={k}")))? {
        return Ok(false);
    }
    let handle = ctx.blas(op)?;
    // Row-major `y[m, n] = act[m, k] · w[n, k]ᵀ` is column-major
    // `yᵀ[n, m] = w[n, k] · actᵀ`, which is BLAS's `op(A)=Aᵀ, op(B)=B` with
    // A = w (k × n, ld k) and B = act (k × m, ld k).
    // SAFETY: the context's handle is live; the three planes are device
    // ranges the caller sized at [n, k], [m, k] and [m, n].
    unsafe {
        blas::bind(handle, ctx.stream())?;
        blas::gemm(
            handle,
            sys::BLAS_OP_T,
            sys::BLAS_OP_N,
            n,
            m,
            k,
            Operand::new(w, sys::BLAS_BF16, k),
            Operand::new(act, sys::BLAS_BF16, k),
            Operand::new(y, sys::BLAS_BF16, n),
        )
    }
    .map_err(|why| refuse(op, format!("{why} at M={m} N={n} K={k}")))?;
    Ok(false)
}

/// `y = act · wᵀ` over bf16 operands landing f32: the skinny kernel for a
/// decode's few rows, hipBLASLt above them, rocBLAS where hipBLASLt offers
/// nothing.
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
    let way = route(ctx, m, n, k);
    if way == Route::Turn && skinnyt(ctx, w, act, y, m, n, k, SkinnyEpilogue::F32).is_ok() {
        return Ok(());
    }
    if way == Route::Skinny && skinny_f32(ctx, w, act, y, m, n, k).is_ok() {
        return Ok(());
    }
    let shape = lt::Shape {
        m: m.unsigned_abs(),
        n: n.unsigned_abs(),
        k: k.unsigned_abs(),
        bias: false,
        groups: 1,
        f32_out: true,
    };
    let workspace = ctx.scratch_shared(op, "linear.lt_workspace", lt::WORKSPACE)? as u64;
    // SAFETY: the three planes are device ranges the caller sized at [m, k],
    // [n, k] and [m, n] (f32); the workspace is this stream's.
    let landed = unsafe { lt::matmul(shape, act, w, 0, y, workspace, ctx.stream()) };
    if landed.map_err(|why| refuse(op, format!("{why} at M={m} N={n} K={k}")))? {
        return Ok(());
    }
    let handle = ctx.blas(op)?;
    // SAFETY: the context's handle is live; the three planes are device
    // ranges the caller sized at [n, k], [m, k] and [m, n].
    unsafe {
        blas::bind(handle, ctx.stream())?;
        blas::gemm(
            handle,
            sys::BLAS_OP_T,
            sys::BLAS_OP_N,
            n,
            m,
            k,
            Operand::new(w, sys::BLAS_BF16, k),
            Operand::new(act, sys::BLAS_BF16, k),
            Operand::new(y, sys::BLAS_F32, n),
        )
    }
    .map_err(|why| refuse(op, format!("{why} at M={m} N={n} K={k}")))
}

/// `h = silu(act · gateᵀ) · (act · upᵀ)` from a packed `[gate | up]` bank,
/// in one fire where the skinny kernel covers the rows; `Ok(false)` leaves
/// the projection and the activation to the caller.
pub(crate) fn act_x_wt_swiglu(
    ctx: &Ctx,
    act: u64,
    gate_up: u64,
    h: u64,
    m: i32,
    intermediate: i32,
    k: i32,
) -> Result<bool, Error> {
    if m <= 0 || intermediate <= 0 || k <= 0 {
        return Ok(false);
    }
    let way = route(ctx, m, 2 * intermediate, k);
    if way == Route::Turn
        && skinnyt(
            ctx,
            gate_up,
            act,
            h,
            m,
            intermediate,
            k,
            SkinnyEpilogue::Swiglu,
        )
        .is_ok()
    {
        return Ok(true);
    }
    if way == Route::Skinny && skinny_swiglu_bf16(ctx, gate_up, act, h, m, intermediate, k).is_ok()
    {
        return Ok(true);
    }
    if way == Route::Batch && dense_batch_bf16(ctx, gate_up, act, h, m, intermediate, k, true)? {
        return Ok(true);
    }
    Ok(false)
}

/// Whether the batch engine takes `m` rows over an n x k weight
/// (`kernel.dense.batch_*`).
fn batch_on(ctx: &Ctx, m: i32, n: i32, k: i32) -> bool {
    let t = ctx.tuning();
    let weight = i64::from(n) * i64::from(k);
    (t.dense_batch_from as i32..=t.dense_batch_until as i32).contains(&m)
        && (i64::from(t.dense_batch_weight_mi) << 20..=i64::from(t.dense_batch_weight_max_mi) << 20)
            .contains(&weight)
}

/// `y = act · wᵀ` over the first `*live` of act's `m` rows (`live` a device
/// word), the rest of y left as it was: hipBLASLt's grouped kernels with M
/// read on the device, a launch for each range of live rows
/// (`kernel.gemm.live_tile_*`). `Ok(false)` where the library or a tile is
/// missing.
pub(crate) fn act_x_wt_live(
    ctx: &Ctx,
    op: &'static str,
    act: u64,
    w: u64,
    live: u64,
    y: u64,
    m: u32,
    n: u32,
    k: u32,
) -> Result<bool, Error> {
    use crate::hip::lt_grouped::{USER_ARGS_BYTES, library};
    use crate::jit::{ArgValue, Fire, Launch, symbol};
    let Ok(lib) = library(dtype::Dtype::Bf16) else {
        return Ok(false);
    };
    let tiles = ctx.tuning().gemm_live_tiles;
    let Some(at) = tiles
        .iter()
        .map(|&t| lib.pick(t))
        .collect::<Option<Vec<usize>>>()
    else {
        return Ok(false);
    };
    // Launch r takes live rows up to its tile's MT1 (the last, up to m).
    let bound = |r: usize| {
        if r == 2 {
            m
        } else {
            lib.solutions[at[r]].mt[1].min(m)
        }
    };
    let args = ctx.scratch(op, "linear.live_args", 6 * USER_ARGS_BYTES)? as u64;
    ctx.fire(
        op,
        Fire::at(
            "linear/moe.hip",
            symbol("::pie::linear::dense_lt_live_args<::pie::bf16>"),
        )
        .apply(Launch::grid([1, 1, 1], [64, 1, 1])),
        &[
            ArgValue::Ptr(live),
            ArgValue::Ptr(args),
            ArgValue::Ptr(w),
            ArgValue::Ptr(act),
            ArgValue::Ptr(y),
            ArgValue::I32(n as i32),
            ArgValue::I32(k as i32),
            ArgValue::I32(m as i32),
            ArgValue::I32(bound(0) as i32),
            ArgValue::I32(bound(1) as i32),
        ],
    )?;
    for (r, &solution) in at.iter().enumerate() {
        let grid = lib.solutions[solution].worst_grid(n, bound(r), 1);
        let record = args + (2 * r * USER_ARGS_BYTES) as u64;
        // SAFETY: the records were written on this stream, naming act, w and
        // y, live device planes of [m, k], [n, k] and [m, n] bf16.
        unsafe { lib.launch(solution, 2, record, grid, ctx.stream()) }
            .map_err(|why| refuse(op, why))?;
    }
    Ok(true)
}
