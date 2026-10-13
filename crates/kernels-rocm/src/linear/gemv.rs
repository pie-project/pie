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
        Fire::at("linear/gemv.hip", instantiation)
            .apply(Launch::grid([grid, 1, 1], [64, WAVES, 1])),
        &values,
    )
}

/// The current device's compute units, asked once per process (every device
/// a process drives is the same part).
pub(crate) fn compute_units() -> u32 {
    static CUS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CUS.get_or_init(|| {
        let mut device = 0i32;
        // SAFETY: writes one int.
        unsafe { crate::hip::rt::hipGetDevice(&raw mut device) };
        crate::hip::multiprocessors(device.unsigned_abs())
            .unwrap_or(304)
            .max(1)
    })
}

/// The LDS the skinny kernel may give the activation rows (one block a CU).
const SKINNY_LDS: u32 = 64 << 10;

/// The skinny kernel's wavefronts a block.
const SKINNY_WAVES: u32 = 16;

/// `y[m, n] = act[m, k] · w[n, k]ᵀ` for 1 ≤ m ≤ 8, the decode projection:
/// the weight streamed once past the activation rows in LDS. Refuses what it
/// does not cover.
pub(crate) fn skinny_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    skinny(ctx, weight, act, out, m, n, k, false)
}

/// [`skinny_bf16`] over a packed `[gate | up]` bank of 2n rows, landing
/// `silu(gate)·up` (n columns): the decode MLP's first half in one fire.
pub(crate) fn skinny_swiglu_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    skinny(ctx, weight, act, out, m, n, k, true)
}

#[allow(clippy::too_many_arguments)]
fn skinny(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    swiglu: bool,
) -> Result<(), Error> {
    const OP: &str = "linear.skinny";
    let (mu, nu, ku) = (m.unsigned_abs(), n.unsigned_abs(), k.unsigned_abs());
    if !(1..=8).contains(&m) || n <= 0 || k <= 0 || k % 8 != 0 || mu * ku * 2 > SKINNY_LDS {
        return Err(refuse(
            OP,
            format!("covers 1..=8 rows of K in whole eights within LDS, not m={m} n={n} k={k}"),
        ));
    }
    for (address, what) in [
        (weight, "the weight"),
        (act, "the activation"),
        (out, "the output"),
    ] {
        if address == 0 || (what != "the output" && !aligned16(address)) {
            return Err(refuse(OP, format!("{what} is null or not 16-byte aligned")));
        }
    }
    let cus = compute_units();
    // Rows a wavefront: as many as keep every SIMD of every CU busy, then
    // as few wavefronts as finish in the same number of rounds.
    let per_round = cus * 4;
    let yt_want = nu.div_ceil(per_round);
    let (yt, unrl) = match yt_want {
        0 | 1 => (1, 4),
        2..=8 => (2, 2),
        _ => (4, 1),
    };
    // No deeper than K: a round past the row's end only rereads its tail.
    let unrl = unrl.min((ku / 8).div_ceil(64).max(1));
    // Past four rows the activation's registers leave room for one round.
    let (yt, unrl) = if mu > 4 { (yt.min(2), 1) } else { (yt, unrl) };
    // The swiglu streams two weight rows per output column.
    let (yt, unrl) = if swiglu {
        (yt.min(2), unrl.min(2))
    } else {
        (yt, unrl)
    };
    let (yt, unrl) = if swiglu && mu > 4 { (1, 1) } else { (yt, unrl) };
    let rounds = |waves: u32| nu.div_ceil(cus * waves * yt);
    let target = rounds(SKINNY_WAVES);
    let waves = (1..=SKINNY_WAVES)
        .find(|&w| rounds(w) == target)
        .unwrap_or(SKINNY_WAVES);
    let symbol = format!("::pie::linear::skinny_bf16<{m}, {yt}, {unrl}, {swiglu}>");
    ctx.fire(
        OP,
        Fire::at("linear/gemv.hip", crate::jit::symbol(&symbol))
            .apply(Launch::grid([cus, 1, 1], [64, SKINNY_WAVES, 1]).smem(mu * ku * 2)),
        &[
            ArgValue::Ptr(weight),
            ArgValue::Ptr(act),
            ArgValue::Ptr(out),
            ArgValue::I32(n),
            ArgValue::I32(k),
            ArgValue::I32(waves as i32),
        ],
    )
}
