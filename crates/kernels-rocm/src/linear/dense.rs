#![allow(clippy::too_many_arguments)]
#![allow(unsafe_code)]

//! `y = act · wᵀ` for a bf16 projection: the skinny kernel for a decode's
//! few rows (the gemv where it refuses), hipBLASLt's heuristic pick above
//! them (what torch and vLLM run on ROCm), and rocBLAS for a shape hipBLASLt
//! offers no algorithm for.

use crate::error::Error;
use crate::hip::blas::{self, Operand};
use crate::hip::{lt, sys};
use crate::jit::{Ctx, aligned16, kernel, refuse};

use super::dense_batch::dense_batch_bf16;
use super::routes::{Kernel, Kind, Shape};
use super::skinny;
use super::gemv::{
    SkinnyEpilogue, gemv_bf16, skinny_bf16, skinny_f32, skinny_swiglu_bf16, skinnyt, turn_fits,
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
    /// The turned kernel (`kernel.gemm.turn_*`'s rows; over the skinny
    /// kernel where the device's measured times say, `super::routes`).
    Turn,
    /// The skinny kernels (`linear/gemv.hip`).
    Skinny,
    /// The decode-batch engine (`kernel.dense.batch_*`, or up to eight rows
    /// over a bank at least `kernel.gemm.skinny_batch_min_n` wide).
    Batch,
    /// What follows: the gemv, hipBLASLt, rocBLAS.
    Other,
}

/// Whether the diagonal kernels' vector reads take these planes: each one
/// there, and the weight and the activation 16-byte aligned.
/// [`act_x_wt`] over rows normed where the skinny or turned kernel stages
/// them (`super::gemv::skinny_norm_bf16`, `skinnyt_norm`), where the product
/// takes either; `Ok(false)` otherwise or where the fold does not pay.
pub(crate) fn act_x_wt_normed(
    ctx: &Ctx,
    w: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
    norm: super::gemv::SkinnyNorm,
) -> Result<bool, Error> {
    if m <= 0 || n <= 0 || k <= 0 {
        return Ok(false);
    }
    match route(ctx, Kind::Normed, m, n, k, w, vector_operands(w, norm.x, y)) {
        Route::Skinny => super::gemv::skinny_norm_bf16(ctx, w, y, m, n, k, norm, SkinnyEpilogue::Store),
        Route::Turn => super::gemv::skinnyt_norm(ctx, w, y, m, n, k, norm, SkinnyEpilogue::Store),
        _ => Ok(false),
    }
}

/// [`act_x_wt_swiglu`] over rows normed where the skinny or turned kernel
/// stages them, where the gated product takes either; `Ok(false)` otherwise or
/// where the fold does not pay.
pub(crate) fn act_x_wt_swiglu_normed(
    ctx: &Ctx,
    gate_up: u64,
    h: u64,
    m: i32,
    intermediate: i32,
    k: i32,
    norm: super::gemv::SkinnyNorm,
) -> Result<bool, Error> {
    if m <= 0 || intermediate <= 0 || k <= 0 {
        return Ok(false);
    }
    let (i, e) = (intermediate, SkinnyEpilogue::Swiglu);
    match route(ctx, Kind::NormedGated, m, 2 * intermediate, k, gate_up, vector_operands(gate_up, norm.x, h)) {
        Route::Skinny => super::gemv::skinny_norm_bf16(ctx, gate_up, h, m, i, k, norm, e),
        Route::Turn => super::gemv::skinnyt_norm(ctx, gate_up, h, m, i, k, norm, e),
        _ => Ok(false),
    }
}

fn vector_operands(w: u64, act: u64, y: u64) -> bool {
    w != 0 && act != 0 && y != 0 && aligned16(w) && aligned16(act)
}

/// The route for `m` rows over a bank of `rows` weight rows of K `k` (the
/// bank `w`, a product of `kind`): a kernel is named only where it serves
/// the product, its shape and its planes (`vectors`, [`vector_operands`]),
/// so a route named is a route that runs, and one that refuses is a fault.
/// Where the turned kernel's rows hold `m` and the skinny kernel would take
/// it too, the device's measured times choose (`super::routes`); against
/// hipBLASLt or the batch engine the turned kernel takes them.
fn route(ctx: &Ctx, kind: Kind, m: i32, rows: i32, k: i32, w: u64, vectors: bool) -> Route {
    let t = ctx.tuning();
    let until = if rows.unsigned_abs() >= t.gemm_skinny_batch_min_n() {
        t.gemm_turn_wide_until()
    } else {
        t.gemm_turn_until().min(16)
    };
    let apart = route_apart(ctx, m, rows, k, vectors);
    let turns = t.gemm_turn_from() > 0
        && (t.gemm_turn_from() as i32..=until as i32).contains(&m)
        && vectors
        && turn_fits(m.unsigned_abs(), rows.unsigned_abs(), k.unsigned_abs());
    if !turns {
        return apart;
    }
    if apart != Route::Skinny {
        return Route::Turn;
    }
    let shape = Shape { kind, rows: rows.unsigned_abs(), k: k.unsigned_abs(), m: m.unsigned_abs() };
    match ctx.routes().map(|routes| super::routes::pick(routes, shape, w)) {
        Some(Kernel::Turned) => Route::Turn,
        Some(Kernel::Skinny) | None => Route::Skinny,
    }
}

/// [`route`] where the turned kernel takes nothing.
fn route_apart(ctx: &Ctx, m: i32, rows: i32, k: i32, vectors: bool) -> Route {
    let t = ctx.tuning();
    let skinny = vectors && skinny::fits(m, rows, k);
    if skinny && m <= SKINNY_SMALL && i64::from(rows) * i64::from(k) <= SKINNY_SMALL_WEIGHT {
        return Route::Skinny;
    }
    // Past two rows an activation beyond LDS (a long K) costs the diagonal
    // kernel its L2 rereads, where hipBLASLt still tiles a short bank faster;
    // a bank of the whole-product kernel's width keeps it.
    let reread = m > 2 && 2 * i64::from(m) * i64::from(k) > SKINNY_LDS && rows < SKINNY_QUAD_ROWS;
    if skinny && m <= t.gemm_skinny_dense_rows() as i32 && !reread {
        return Route::Skinny;
    }
    // Past the skinny rows an lm_head-wide bank streams best on the batch
    // engine; others take
    // hipBLASLt unless the batch window holds them.
    if batch_on(ctx, m, rows, k)
        || (m <= SKINNY_SMALL && rows.unsigned_abs() >= t.gemm_skinny_batch_min_n())
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
    let vectors = vector_operands(w, act, y);
    let way = route(ctx, Kind::Plain, m, n, k, w, vectors);
    if bias == 0 && way == Route::Turn {
        skinnyt(ctx, w, act, y, m, n, k, SkinnyEpilogue::Store)?;
        return Ok(false);
    }
    if bias == 0 && way == Route::Skinny {
        skinny_bf16(ctx, w, act, y, m, n, k)?;
        return Ok(false);
    }
    // The gemv: one row, K in whole eights, over vector-read planes.
    if m == 1 && vectors && k % 8 == 0 {
        gemv_bf16(ctx, w, act, bias, y, n, k)?;
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
            &mut ctx.blas_workspace(op),
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
    let way = route(ctx, Kind::PlainF32, m, n, k, w, vector_operands(w, act, y));
    if way == Route::Turn {
        return skinnyt(ctx, w, act, y, m, n, k, SkinnyEpilogue::F32);
    }
    if way == Route::Skinny {
        return skinny_f32(ctx, w, act, y, m, n, k);
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
            &mut ctx.blas_workspace(op),
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
    let way = route(ctx, Kind::Gated, m, 2 * intermediate, k, gate_up, vector_operands(gate_up, act, h));
    if way == Route::Turn {
        skinnyt(
            ctx,
            gate_up,
            act,
            h,
            m,
            intermediate,
            k,
            SkinnyEpilogue::Swiglu,
        )?;
        return Ok(true);
    }
    if way == Route::Skinny {
        skinny_swiglu_bf16(ctx, gate_up, act, h, m, intermediate, k)?;
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
    (t.dense_batch_from() as i32..=t.dense_batch_until() as i32).contains(&m)
        && (i64::from(t.dense_batch_weight_mi()) << 20..=i64::from(t.dense_batch_weight_max_mi()) << 20)
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
    use crate::jit::{ArgValue, Fire, Launch};
    let lib = library(dtype::Dtype::Bf16).map_err(|why| refuse(op, why))?;
    let tiles = ctx.tuning().gemm_live_tiles();
    let at = tiles
        .iter()
        .map(|&t| {
            lib.pick(t).ok_or_else(|| {
                refuse(
                    op,
                    format!("the profile's live-row tile {t} is not in hipBLASLt's grouped library"),
                )
            })
        })
        .collect::<Result<Vec<usize>, Error>>()?;
    // Launch r takes live rows up to its tile's MT1 (the last, up to m).
    let bound = |r: usize| {
        if r == 2 {
            m
        } else {
            lib.solutions[at[r]].mt[1].min(m)
        }
    };
    // Three records a launch: its whole MT0 tiles, the weight rows past
    // them, the sentinel (`dense_lt_live_args`).
    let args = ctx.scratch(op, "linear.live_args", 9 * USER_ARGS_BYTES)? as u64;
    let tile = |r: usize| lib.solutions[at[r]].mt[0] as i32;
    ctx.fire(
        op,
        Fire::at(kernel(
            "linear/moe.hip",
            "::pie::linear::dense_lt_live_args<::pie::bf16>",
        ))
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
            ArgValue::I32(tile(0)),
            ArgValue::I32(tile(1)),
            ArgValue::I32(tile(2)),
        ],
    )?;
    for (r, &solution) in at.iter().enumerate() {
        // Two groups split the weight rows: the worst grid over both.
        let grid = lib.solutions[solution].worst_grid(n, bound(r), 2);
        let record = args + (3 * r * USER_ARGS_BYTES) as u64;
        // SAFETY: the records were written on this stream, naming act, w and
        // y, live device planes of [m, k], [n, k] and [m, n] bf16, and
        // column bands of w and y inside them.
        unsafe { lib.launch(solution, 3, record, grid, ctx.stream()) }
            .map_err(|why| refuse(op, why))?;
    }
    Ok(true)
}

