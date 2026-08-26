//! The skinny fires: one output vector against the whole weight, split over
//! k when the column count allows it. The unroll depth follows the
//! architecture probe — Blackwell prefers shallow unrolls with more warps in
//! flight.

use kernels::KernelError;

use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, refuse};

pub(crate) fn gemv_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    n: i32,
    k: i32,
) -> Result<(), KernelError> {
    if n <= 0 || k <= 0 || k % 8 != 0 {
        return Err(refuse(
            "linear.gemv",
            format!("needs n > 0 and k > 0 in whole eights, and was handed n={n}, k={k}"),
        ));
    }
    for (address, what) in [(weight, "the weight"), (act, "the activation"), (out, "the output")] {
        if address == 0 {
            return Err(refuse("linear.gemv", format!("{what} is null")));
        }
    }
    for (address, what) in [(weight, "the weight"), (act, "the activation")] {
        if !aligned16(address) {
            return Err(refuse(
                "linear.gemv",
                format!("{what} is not 16-byte aligned, and the vectorised loads demand it"),
            ));
        }
    }

    let values = [
        ArgValue::Ptr(weight),
        ArgValue::Ptr(act),
        ArgValue::ABSENT,
        ArgValue::Ptr(out),
        ArgValue::I32(n),
        ArgValue::I32(k),
        // beta: nothing ever accumulates into the output on this plane.
        ArgValue::F32(0.0),
    ];
    let blackwell = ctx.compute_capability_major().is_some_and(|major| major >= 10);

    // Split over k while the row count is skinny enough to leave SMs idle:
    // one block per row, the warps sharing the row's k walk.
    if n <= 4096 {
        let (instantiation, warps) = if blackwell {
            ("::pie::linear::gemv_splitk_bf16_kernel<::pie::i32(4), 2>", 4)
        } else {
            ("::pie::linear::gemv_splitk_bf16_kernel<::pie::i32(8), 1>", 8)
        };
        return ctx.fire(
            "linear.gemv",
            Fire::at("linear/gemv.cuh", instantiation)
                .apply(Launch::grid([n.unsigned_abs(), 1, 1], [32, warps, 1])),
            &values,
        );
    }

    let instantiation = if blackwell {
        "::pie::linear::gemv_bf16_kernel<::pie::i32(4), 2>"
    } else {
        "::pie::linear::gemv_bf16_kernel<::pie::i32(4), 4>"
    };
    // Four warps per block, one row per warp.
    ctx.fire(
        "linear.gemv",
        Fire::at("linear/gemv.cuh", instantiation)
            .apply(Launch::grid([n.unsigned_abs().div_ceil(4), 1, 1], [32, 4, 1])),
        &values,
    )
}
