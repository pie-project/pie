
use kernels::{Fire};
use kernels_macros::routine;
use crate::jit::{ArgValue, Ctx, Launch, aligned16};
use crate::jit::abi::Tensor;
use kernels::Refusal;
use kernels::routine::{Const, In, Out};

const WARPS: u32 = 4;

const WARP_LANES: u32 = 32;

const MAX_BLOCKS: i64 = 2_147_483_647;

fn unroll_depth(ctx: &Ctx<'_>) -> i32 {
    if ctx.compute_capability_major().is_some_and(|major| major >= 10) { 2 } else { 4 }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[routine(internal)]
pub fn gemv_bf16(
    ctx: &Ctx<'_>,
    weight: Const<Tensor<std::ffi::c_void>>,
    act: In<Tensor<std::ffi::c_void>>,
    out: Out<Tensor<std::ffi::c_void>>,
    // THE CALLER'S NUMBER, AND `Const` IS HOW THIS ROUTINE'S CALLERS ALREADY
    // HAND ONE OVER: `dense.rs` builds `Const { v: w }`, `In { .. }` and
    // `Out { .. }` by hand at both call sites, because a path-fired launch has
    // no statement to place them. One site passes its own `beta` and the other
    // a literal `0.0` under a `beta == 0.0` guard, so the value is decided
    // per call and no fact could answer it.
    beta: Const<f32>) -> Result<(), Refusal> {
    const SPLIT_K_MAX_ROWS: i32 = 4096;

    const SPLIT_WARPS: u32 = 8;

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
        ArgValue::Ptr(std::ptr::null_mut()),
        ArgValue::Ptr(out.ptr),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::F32(beta.v),
    ];

    if n <= SPLIT_K_MAX_ROWS {

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

    let instantiation = if unroll_depth(ctx) == 2 {
        "::pie::gemm::gemv_bf16_kernel<::pie::i32(4), 2>"
    } else {
        "::pie::gemm::gemv_bf16_kernel<::pie::i32(4), 4>"
    };
    ctx.fire(Fire::at("gemm/gemv.cuh", instantiation).apply(Launch::grid([grid_x, 1, 1], [WARP_LANES, WARPS, 1])), &values)
}
