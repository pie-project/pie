use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, refuse};

pub(crate) fn gemv_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    bias: u64,
    out: u64,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    if n <= 0 || k <= 0 || k % 8 != 0 {
        return Err(refuse(
            "linear.gemv",
            format!("needs n > 0 and k > 0 in whole eights, and was handed n={n}, k={k}"),
        ));
    }
    for (address, what) in [
        (weight, "the weight"),
        (act, "the activation"),
        (out, "the output"),
    ] {
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
        if bias == 0 {
            ArgValue::ABSENT
        } else {
            ArgValue::Ptr(bias)
        },
        ArgValue::Ptr(out),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::F32(0.0),
    ];
    // Four wavefronts a block: a row each, or (for a short N, which would
    // not fill the device a wavefront per row) one row split four ways on K.
    const WAVES: u32 = 4;
    let rows = n.unsigned_abs();
    let (instantiation, grid) = if rows <= ctx.tuning().gemv_splitk_max_n {
        ("::pie::linear::gemv_bf16<4, true>", rows)
    } else {
        ("::pie::linear::gemv_bf16<4, false>", rows.div_ceil(WAVES))
    };
    ctx.fire(
        "linear.gemv",
        Fire::at("linear/gemv.hip", instantiation).apply(Launch::grid([grid, 1, 1], [64, WAVES, 1])),
        &values,
    )
}
