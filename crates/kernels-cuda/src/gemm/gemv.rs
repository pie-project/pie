use crate::jit::{ArgValue, Ctx, Launch, aligned16};
use kernels::Refusal;

/// Warps per block in the row-per-warp form — `gemv.cu:329`'s
const WARPS: u32 = 4;

/// A warp, which is the first block axis of all four launches.
const WARP_LANES: u32 = 32;

/// The largest grid the row-per-warp form will open — `gemv.cu:381`'s
const MAX_BLOCKS: i64 = 2_147_483_647;

/// How deep to unroll the row walk: 2 on Blackwell and later, 4 below.
///
/// The device answers; an unknown one gets the conservative 4, which is what
/// every pre-Blackwell part wants and what the probe used to fall back to at
/// each of its four failure points.
fn unroll_depth(ctx: &Ctx) -> i32 {
    if ctx.compute_capability_major().is_some_and(|major| major >= 10) { 2 } else { 4 }
}

/// Single-row bf16 GEMV: `out[n] = sum_k W[n][k] * x[k] + bias[n] + beta * out[n]`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn gemv_bf16(
    ctx: &Ctx,
    weight: *const std::ffi::c_void,
    act: *const std::ffi::c_void,
    bias: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    /// The row count below which K is split INSIDE the block — `gemv.cu:317`'s
    const SPLIT_K_MAX_ROWS: i32 = 4096;

    /// Warps per block in the split-K form everywhere else — `gemv.cu:352`'s
    const SPLIT_WARPS: u32 = 8;

    /// Warps per block in the split-K form on Blackwell — `gemv.cu:342`'s
    const SPLIT_WARPS_B: u32 = 4;

    if n <= 0 || k <= 0 || k % 8 != 0 {
        return Err(Refusal::Narrow { what: "n, k, or k in whole eights", at: i64::from(k) });
    }
    for (p, what) in [(weight, "weight"), (act, "act"), (out.cast_const(), "out")] {
        if p.is_null() {
            return Err(Refusal::Null { what });
        }
    }
    for (p, what) in [(weight, "weight"), (act, "act")] {
        if !aligned16(p) {
            return Err(Refusal::Misaligned { what });
        }
    }

    let values = [
        ArgValue::Ptr(weight.cast_mut()),
        ArgValue::Ptr(act.cast_mut()),
        ArgValue::Ptr(bias.cast_mut()),
        ArgValue::Ptr(out),
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
        return unsafe {
            ctx.launch(
                "gemm/gemv.cuh",
                instantiation,
                Launch::grid([n.unsigned_abs(), 1, 1], [WARP_LANES, warps, 1]),
                &values,
            )
        };
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
    unsafe {
        ctx.launch(
            "gemm/gemv.cuh",
            instantiation,
            Launch::grid([grid_x, 1, 1], [WARP_LANES, WARPS, 1]),
            &values,
        )
    }
}

