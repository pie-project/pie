use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, refuse};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Epilogue {
    Store,
    Softcap(f32),
    Geglu,
}

pub const ROWS: u32 = 64;

const WARPS: u32 = 4;
const STAGES: u32 = 2;
const STEP: u32 = 128;

const SMEM_BYTES: u32 = STAGES * (WARPS * 16 + ROWS) * (STEP + 8) * 2;

const fn cols(epilogue: Epilogue) -> u32 {
    match epilogue {
        Epilogue::Geglu => WARPS * 8,
        Epilogue::Store | Epilogue::Softcap(_) => WARPS * 16,
    }
}

#[must_use]
pub fn covers(m: i32, n: i32, k: i32, epilogue: Epilogue) -> bool {
    let cols = cols(epilogue) as i32;
    m >= 1 && m <= ROWS as i32 && n > 0 && n % cols == 0 && k > 0 && k % STEP as i32 == 0
}

#[allow(clippy::too_many_arguments)]
pub fn skinny_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: Epilogue,
) -> Result<(), Error> {
    if !covers(m, n, k, epilogue) {
        return Err(refuse(
            "linear.skinny",
            format!(
                "covers 1 <= m <= 64, n in whole {}s and k in whole {STEP}s, and was handed \
                 m={m}, n={n}, k={k} for {epilogue:?}",
                cols(epilogue)
            ),
        ));
    }
    for (address, what) in [
        (weight, "the weight"),
        (act, "the activation"),
        (out, "the output"),
    ] {
        if address == 0 {
            return Err(refuse("linear.skinny", format!("{what} is null")));
        }
        if !aligned16(address) {
            return Err(refuse(
                "linear.skinny",
                format!("{what} is not 16-byte aligned, and cp.async demands it"),
            ));
        }
    }
    let cap = match epilogue {
        Epilogue::Softcap(cap) => {
            if !(cap.is_finite() && cap > 0.0) {
                return Err(refuse(
                    "linear.skinny",
                    format!("a softcap of {cap} is not a positive number"),
                ));
            }
            cap
        }
        Epilogue::Store | Epilogue::Geglu => 0.0,
    };
    let entrypoint = match epilogue {
        Epilogue::Store => "::pie::linear::skinny_bf16_kernel<4, 2, 128, 0>",
        Epilogue::Softcap(_) => "::pie::linear::skinny_bf16_kernel<4, 2, 128, 1>",
        Epilogue::Geglu => "::pie::linear::skinny_bf16_kernel<4, 2, 128, 2>",
    };
    let values = [
        ArgValue::Ptr(weight),
        ArgValue::Ptr(act),
        ArgValue::Ptr(out),
        ArgValue::I32(m),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::F32(cap),
    ];
    ctx.fire(
        "linear.skinny",
        Fire::at("linear/skinny.cuh", entrypoint).apply(
            Launch::grid(
                [(n / cols(epilogue) as i32) as u32, 1, 1],
                [32 * WARPS, 1, 1],
            )
            .smem(SMEM_BYTES),
        ),
        &values,
    )
}
