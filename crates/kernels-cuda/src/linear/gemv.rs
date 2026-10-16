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
    // `kernel.gemv.bf16`: Blackwell profiles pick `narrow`.
    let tuning = ctx.tuning();
    let narrow = tuning.gemv_narrow;

    if n.unsigned_abs() <= tuning.gemv_splitk_max_n {
        let (instantiation, warps) = if narrow {
            (
                "::pie::linear::gemv_splitk_bf16_kernel<::pie::i32(4), 2>",
                4,
            )
        } else {
            (
                "::pie::linear::gemv_splitk_bf16_kernel<::pie::i32(8), 1>",
                8,
            )
        };
        return ctx.fire(
            "linear.gemv",
            Fire::at("linear/gemv.cuh", instantiation)
                .apply(Launch::grid([n.unsigned_abs(), 1, 1], [32, warps, 1])),
            &values,
        );
    }

    let instantiation = if narrow {
        "::pie::linear::gemv_bf16_kernel<::pie::i32(4), 2>"
    } else {
        "::pie::linear::gemv_bf16_kernel<::pie::i32(4), 4>"
    };
    ctx.fire(
        "linear.gemv",
        Fire::at("linear/gemv.cuh", instantiation).apply(Launch::grid(
            [n.unsigned_abs().div_ceil(4), 1, 1],
            [32, 4, 1],
        )),
        &values,
    )
}

/// `out[m][n] = act[m] . weight[n]` for a few rows (2..=8, a decode's) in
/// one launch: the tactic `kernel.gemm.rows_m` lets the GEMM tuner try. A
/// block's eight warps stream weight rows, as many warps a row as keep some
/// 120 blocks busy (each a slice of K, summed in slice order).
#[allow(clippy::too_many_arguments)]
pub fn gemm_rows_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    bias: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    const OP: &str = "linear.gemm_rows";
    if !(1..=8).contains(&m) || n <= 0 || k <= 0 || k % 8 != 0 {
        return Err(refuse(
            OP,
            format!("needs 1..=8 rows and k in whole eights, and was handed m={m}, k={k}"),
        ));
    }
    if weight == 0 || act == 0 || out == 0 || !aligned16(weight) || !aligned16(act) {
        return Err(refuse(
            OP,
            "the weight and the activation are 16-byte aligned, and nothing is null",
        ));
    }
    let vectors = k / 8;
    // Eight warps a row where K divides so: the most blocks and the fewest
    // dependent loads a lane (best at every DeepSeek-V4-Flash shape measured).
    let split = [8, 4, 2, 1]
        .into_iter()
        .find(|s| vectors % s == 0)
        .unwrap_or(1);
    let rows = 16 / split;
    ctx.fire(
        OP,
        Fire::at(
            "linear/gemv.cuh",
            crate::jit::symbol(&format!(
                "::pie::linear::gemm_rows_bf16_kernel<{m}, {split}>"
            )),
        )
        .apply(Launch::grid(
            [n.unsigned_abs().div_ceil(rows as u32), 1, 1],
            [256, 1, 1],
        )),
        &[
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
        ],
    )
}

/// `out[m][n] = act[m] . weight[n]` for a product at most eight columns wide,
/// a block a row: the tactic `kernel.gemm.narrow_n` lets the GEMM tuner try.
#[allow(clippy::too_many_arguments)]
pub fn gemm_narrow_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    bias: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    const OP: &str = "linear.gemm_narrow";
    if m <= 0 || !(1..=8).contains(&n) || k <= 0 || k % 8 != 0 {
        return Err(refuse(
            OP,
            format!("needs 1..=8 columns and k in whole eights, and was handed n={n}, k={k}"),
        ));
    }
    if weight == 0 || act == 0 || out == 0 || !aligned16(weight) || !aligned16(act) {
        return Err(refuse(
            OP,
            "the weight and the activation are 16-byte aligned, and nothing is null",
        ));
    }
    ctx.fire(
        OP,
        Fire::at(
            "linear/gemv.cuh",
            crate::jit::symbol(&format!("::pie::linear::gemm_narrow_bf16_kernel<{n}>")),
        )
        .apply(Launch::grid([m.unsigned_abs(), 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(weight),
            ArgValue::Ptr(act),
            if bias == 0 {
                ArgValue::ABSENT
            } else {
                ArgValue::Ptr(bias)
            },
            ArgValue::Ptr(out),
            ArgValue::I32(k),
        ],
    )
}
