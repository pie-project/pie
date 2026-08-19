//! The GEMV host program: `out[n] = sum_k W[n][k] * x[k] + beta * out[n]`
//! in bf16, for the single-row (decode) shape.
//!
//! [`gemv_bf16`] picks one of four instantiations — split-K under 4096
//! output rows, row-per-warp above it, each at the unroll depth this device
//! was measured for (2 on Blackwell and later, 4 below) — and refuses a K
//! that is not a whole multiple of eight, or an operand that is not 16-byte
//! aligned.

use kernels::{Fire};
use kernels_macros::routine;
use crate::jit::{ArgValue, Ctx, Launch, aligned16};
use crate::jit::abi::Tensor;
use kernels::Refusal;
use kernels::routine::{Const, In, Out};
// `#[routine]` is written fully-qualified (this file imports
// no `routine!`, so nothing would collide, but a reader arriving from
// `mod.rs` should not have to check). The derived column is discussed once,
// in `gemm/mod.rs`, where `ROUTINES` is; the launcher below's own marks are
// argued locally because they describe this signature's own shape.

/// Warps per block in the row-per-warp form — `gemv.cu:329`'s
const WARPS: u32 = 4;

/// A warp, which is the first block axis of all four launches.
const WARP_LANES: u32 = 32;

/// The largest grid the row-per-warp form will open — `gemv.cu:381`'s
const MAX_BLOCKS: i64 = 2_147_483_647;

/// How deep to unroll the row walk: 2 on Blackwell and later, 4 below.
///
/// An unknown compute capability falls back to the conservative default (4).
fn unroll_depth(ctx: &Ctx<'_>) -> i32 {
    if ctx.compute_capability_major().is_some_and(|major| major >= 10) { 2 } else { 4 }
}

/// Single-row bf16 GEMV: `out[n] = sum_k W[n][k] * x[k] + bias[n] + beta * out[n]`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[routine]
pub fn gemv_bf16(
    ctx: &Ctx<'_>,
    // The only launcher in this family where `weight` precedes `act`;
    // without this explicit `Weight<0, _>` mark, positional inference would
    // derive `In(0)` for `weight` and `In(1)` for `act` -- backwards from
    // every other signature here.
    weight: Const<Tensor<std::ffi::c_void>>,
    // `InRow`/`OutRow` (width only, not a full `In`/`Out` region): this leg's
    // row count is not a real operand extent (`dense.rs` refuses unless
    // `m == 1`) and its caller is a tuner that never holds a `Facts` to
    // resolve a region against, so only the width -- a real extent -- is
    // taken. `n`/`k` below mirror the parent's `OutWidth(0)`/`InWidth(0)`
    // marks in `gemm/mod.rs`.
    act: In<Tensor<std::ffi::c_void>>,
    // Same reason as `act`. `alias()` never touches an output, so stating
    // `Out(0)` here carries no risk even in a family that declares
    // `in_place`.
    out: Out<Tensor<std::ffi::c_void>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    beta: f32) -> Result<(), Refusal> {
    /// The row count below which K is split INSIDE the block — `gemv.cu:317`'s
    const SPLIT_K_MAX_ROWS: i32 = 4096;

    /// Warps per block in the split-K form everywhere else — `gemv.cu:352`'s
    const SPLIT_WARPS: u32 = 8;

    /// Warps per block in the split-K form on Blackwell — `gemv.cu:342`'s
    const SPLIT_WARPS_B: u32 = 4;

    let n = out.width;
    let k = act.width;

    if n <= 0 || k <= 0 || k % 8 != 0 {
        return Err(Refusal::Narrow { what: "n, k, or k in whole eights", at: i64::from(k) });
    }
    for (p, what) in [(weight.v, "weight"), (act.ptr, "act"), (out.ptr.cast_const(), "out")] {
        if p.is_null() {
            return Err(Refusal::Null { what });
        }
    }
    for (p, what) in [(weight.v, "weight"), (act.ptr, "act")] {
        if !aligned16(p) {
            return Err(Refusal::Misaligned { what });
        }
    }

    let values = [
        ArgValue::Ptr(weight.v.cast_mut()),
        ArgValue::Ptr(act.ptr.cast_mut()),
        // Always null: `gemv_bf16` takes no bias parameter, but the kernel
        // ABI still expects a pointer argument in this slot.
        ArgValue::Ptr(std::ptr::null_mut()),
        ArgValue::Ptr(out.ptr),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::F32(beta),
    ];

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    if n <= SPLIT_K_MAX_ROWS {
        // Split-K: four warps and unroll 2 on Blackwell and later, eight and
        // unroll 1 below it.
        let (instantiation, warps) = if unroll_depth(ctx) == 2 {
            ("::pie::gemm::gemv_splitk_bf16_kernel<::pie::i32(4), 2>", SPLIT_WARPS_B)
        } else {
            ("::pie::gemm::gemv_splitk_bf16_kernel<::pie::i32(8), 1>", SPLIT_WARPS)
        };
        return ctx.fire(Fire::at("gemm/gemv.cuh", instantiation).apply(Launch::grid([n.unsigned_abs(), 1, 1], [WARP_LANES, warps, 1])), &values);
    }

    let warps = i64::from(WARPS);
    let blocks = (i64::from(n) + warps - 1) / warps;
    let Ok(grid_x) = u32::try_from(blocks.min(MAX_BLOCKS + 1)) else {
        return Err(Refusal::Grid { what: "x", at: blocks });
    };
    if blocks > MAX_BLOCKS {
        return Err(Refusal::Grid { what: "x", at: blocks });
    }

    // Row-per-warp, four warps, at the unroll this device was measured for.
    let instantiation = if unroll_depth(ctx) == 2 {
        "::pie::gemm::gemv_bf16_kernel<::pie::i32(4), 2>"
    } else {
        "::pie::gemm::gemv_bf16_kernel<::pie::i32(4), 4>"
    };
    ctx.fire(Fire::at("gemm/gemv.cuh", instantiation).apply(Launch::grid([grid_x, 1, 1], [WARP_LANES, WARPS, 1])), &values)
}

