//! The skinny fires: one output vector against the whole weight, split over
//! k when the column count allows it. The unroll depth follows the
//! architecture probe — Blackwell prefers shallow unrolls with more warps in
//! flight.

use core::ffi::c_void;

use new_kernels::KernelError;

use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, refuse};

const OP: &str = "gemm.gemv";

const FILE: &str = "gemm/gemv.cuh";

const WARPS: u32 = 4;

const WARP_LANES: u32 = 32;

const MAX_BLOCKS: i64 = 2_147_483_647;

fn unroll_depth(ctx: &Ctx) -> i32 {
    if ctx
        .compute_capability_major()
        .is_some_and(|major| major >= 10)
    {
        2
    } else {
        4
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemv_bf16(
    ctx: &Ctx,
    weight: *const c_void,
    act: *const c_void,
    out: *mut c_void,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), KernelError> {
    const SPLIT_K_MAX_ROWS: i32 = 4096;

    const SPLIT_WARPS: u32 = 8;

    const SPLIT_WARPS_B: u32 = 4;

    if n <= 0 || k <= 0 || k % 8 != 0 {
        return Err(refuse(
            OP,
            format!("needs n > 0 and k > 0 in whole eights, and was handed n={n}, k={k}"),
        ));
    }
    for (p, what) in [
        (weight, "the weight"),
        (act, "the activation"),
        (out.cast_const(), "the output"),
    ] {
        if p.is_null() {
            return Err(refuse(OP, format!("{what} is null")));
        }
    }
    for (p, what) in [(weight, "the weight"), (act, "the activation")] {
        if !aligned16(p.addr() as u64) {
            return Err(refuse(
                OP,
                format!("{what} is not 16-byte aligned, and the vectorised loads demand it"),
            ));
        }
    }

    let values = [
        ArgValue::Ptr(weight.addr() as u64),
        ArgValue::Ptr(act.addr() as u64),
        ArgValue::ABSENT,
        ArgValue::Ptr(out.addr() as u64),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::F32(beta),
    ];

    if n <= SPLIT_K_MAX_ROWS {
        let (instantiation, warps) = if unroll_depth(ctx) == 2 {
            (
                "::pie::gemm::gemv_splitk_bf16_kernel<::pie::i32(4), 2>",
                SPLIT_WARPS_B,
            )
        } else {
            (
                "::pie::gemm::gemv_splitk_bf16_kernel<::pie::i32(8), 1>",
                SPLIT_WARPS,
            )
        };
        return ctx.fire(
            OP,
            Fire::at(FILE, instantiation).apply(Launch::grid(
                [n.unsigned_abs(), 1, 1],
                [WARP_LANES, warps, 1],
            )),
            &values,
        );
    }

    let warps = i64::from(WARPS);
    let blocks = (i64::from(n) + warps - 1) / warps;
    if blocks > MAX_BLOCKS {
        return Err(refuse(
            OP,
            format!("{blocks} blocks do not fit the grid's x axis"),
        ));
    }
    let Ok(grid_x) = u32::try_from(blocks) else {
        return Err(refuse(
            OP,
            format!("{blocks} blocks do not fit the grid's x axis"),
        ));
    };

    let instantiation = if unroll_depth(ctx) == 2 {
        "::pie::gemm::gemv_bf16_kernel<::pie::i32(4), 2>"
    } else {
        "::pie::gemm::gemv_bf16_kernel<::pie::i32(4), 4>"
    };
    ctx.fire(
        OP,
        Fire::at(FILE, instantiation).apply(Launch::grid([grid_x, 1, 1], [WARP_LANES, WARPS, 1])),
        &values,
    )
}
