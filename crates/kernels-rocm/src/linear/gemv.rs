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
    skinny(ctx, weight, act, out, m, n, k, SkinnyEpilogue::Store)
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
    skinny(ctx, weight, act, out, m, n, k, SkinnyEpilogue::Swiglu)
}

/// What the skinny kernel lands of each product: it as is, the gated
/// activation of a packed `[gate | up]` bank of 2n rows (`n` columns), or
/// `cap · tanh(v / cap)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SkinnyEpilogue {
    Store,
    Swiglu,
    Geglu,
    Softcap(f32),
}

impl SkinnyEpilogue {
    const fn code(self) -> u32 {
        match self {
            Self::Store => 0,
            Self::Swiglu => 1,
            Self::Geglu => 2,
            Self::Softcap(_) => 3,
        }
    }
}

/// How the skinny kernel walks one shape: weight rows a wavefront (`yt`),
/// 1 KB reads a row in flight (`unrl`), the wavefronts that work in each
/// block of `block_waves`, `grid` blocks, and whether the activation is read
/// from L2 (`act_global`) rather than staged in LDS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkinnyPlan {
    pub yt: u32,
    pub unrl: u32,
    pub waves: u32,
    pub block_waves: u32,
    pub grid: u32,
    pub act_global: bool,
    /// The whole-product kernel (`skinny4_bf16`): `yt` counts groups of
    /// four rows, and `ks` wavefronts split each tile's K (`waves` a
    /// multiple of it; the block is `waves` wavefronts).
    pub quad: bool,
    pub ks: u32,
}

/// The LDS the skinny kernel stages its activation rows in.
pub(crate) const SKINNY_LDS: u32 = 64 << 10;

/// The whole-product kernel's banks: at least this many weight rows (a
/// short bank leaves its four-row tiles too few to fill the card, where the
/// diagonal kernel still streams).
pub(crate) const QUAD_MIN_ROWS: u32 = 8192;
/// A bank this wide (an lm_head) keeps the general walk at one or two rows.
const WIDE_ROWS: u32 = 65536;
/// The K from which its tiles split across two wavefronts.
const QUAD_SPLIT_K: u32 = 12288;

/// The walk for `m` rows over a bank of `rows` weight rows (`2n` for a
/// packed `[gate | up]` bank) of K `k`, the whole-product kernel from
/// `quad_from` rows. The summation order depends only on `m`, `rows`, `k`
/// and `quad_from` (never on `n` or the epilogue), so a fused GLU lands the
/// bits its bank's plain projection would.
pub(crate) fn skinny_plan(m: u32, rows: u32, k: u32, swiglu: bool, quad_from: u32) -> SkinnyPlan {
    let cus = compute_units();
    let n = if swiglu { rows / 2 } else { rows };
    let act_global = m * k * 2 > SKINNY_LDS;
    if m >= quad_from && rows >= QUAD_MIN_ROWS {
        // A long K splits across two wavefronts (the slices meet in LDS).
        let ks = if k >= QUAD_SPLIT_K { 2 } else { 1 };
        // The activation (rows padded a chunk) in LDS where it fits: then
        // one block of twelve wavefronts a CU, a group of four rows each
        // with four chunks of a row in flight (two for a GLU's two groups).
        let staged = m * (k / 8 + 1) * 16;
        let red = |waves: u32, groups: u32| {
            if ks > 1 {
                waves * m.div_ceil(4) * groups * 64
            } else {
                0
            }
        };
        let unrl = if swiglu { 2 } else { 4 };
        if staged + red(12, if swiglu { 2 } else { 1 }) <= SKINNY_LDS {
            return SkinnyPlan {
                yt: 1,
                unrl,
                waves: 12,
                block_waves: 12,
                grid: cus,
                act_global: false,
                quad: true,
                ks,
            };
        }
        // Read from L2: two blocks of eight wavefronts a CU, four groups of
        // four rows a tile (two for a GLU) with two chunks of each in flight
        // (one for a GLU).
        return SkinnyPlan {
            yt: if swiglu { 2 } else { 4 },
            unrl: if swiglu { 1 } else { 2 },
            waves: 8,
            block_waves: 8,
            grid: 2 * cus,
            act_global: true,
            quad: true,
            ks,
        };
    }
    if m <= 2 && !swiglu && !act_global && rows < WIDE_ROWS {
        // A row a wavefront, four 1 KB reads of it in flight, twelve
        // wavefronts a CU (an lm_head-wide bank keeps four rows a wavefront).
        return SkinnyPlan {
            yt: 1,
            unrl: 4.min((k / 8).div_ceil(64).max(1)),
            waves: 12,
            block_waves: 12,
            grid: cus,
            act_global,
            quad: false,
            ks: 1,
        };
    }
    // Rows a wavefront: as many as keep every SIMD of every CU busy, then
    // as few wavefronts as finish in the same number of rounds.
    let per_round = cus * 4;
    let yt_want = n.div_ceil(per_round);
    let (yt, unrl) = match yt_want {
        0 | 1 => (1, 4),
        2..=8 => (2, 2),
        _ => (4, 1),
    };
    // No deeper than K: a round past the row's end only rereads its tail.
    let unrl = unrl.min((k / 8).div_ceil(64).max(1));
    // Past four rows the activation's registers leave room for one round.
    let (yt, unrl) = if m > 4 { (yt.min(2), 1) } else { (yt, unrl) };
    // The swiglu streams two weight rows per output column.
    let (yt, unrl) = if swiglu {
        (yt.min(2), unrl.min(2))
    } else {
        (yt, unrl)
    };
    let (yt, unrl) = if swiglu && m > 4 { (1, 1) } else { (yt, unrl) };
    let rounds = |waves: u32| n.div_ceil(cus * waves * yt);
    let target = rounds(SKINNY_WAVES);
    let waves = (1..=SKINNY_WAVES)
        .find(|&w| rounds(w) == target)
        .unwrap_or(SKINNY_WAVES);
    SkinnyPlan {
        yt,
        unrl,
        waves,
        block_waves: SKINNY_WAVES,
        grid: cus,
        act_global,
        quad: false,
        ks: 1,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn skinny(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: SkinnyEpilogue,
) -> Result<(), Error> {
    let swiglu = matches!(epilogue, SkinnyEpilogue::Swiglu | SkinnyEpilogue::Geglu);
    let rows = n.unsigned_abs() * if swiglu { 2 } else { 1 };
    let plan = skinny_plan(
        m.unsigned_abs(),
        rows,
        k.unsigned_abs(),
        swiglu,
        ctx.tuning().gemm_skinny_quad_from,
    );
    skinny_planned(ctx, weight, act, out, m, n, k, epilogue, plan)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn skinny_planned(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: SkinnyEpilogue,
    plan: SkinnyPlan,
) -> Result<(), Error> {
    const OP: &str = "linear.skinny";
    let (mu, ku) = (m.unsigned_abs(), k.unsigned_abs());
    if !super::skinny::fits(m, n, k) {
        return Err(refuse(
            OP,
            format!("covers 1..=8 rows of K in whole eights, not m={m} n={n} k={k}"),
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
    let SkinnyPlan {
        yt,
        unrl,
        waves,
        block_waves,
        grid,
        act_global,
        quad,
        ks,
    } = plan;
    let swiglu = matches!(epilogue, SkinnyEpilogue::Swiglu | SkinnyEpilogue::Geglu);
    // The whole-product kernel pads each LDS row a chunk, and its K slices
    // meet in LDS past the activation.
    let (block_waves, staged, red) = if quad {
        let ks = ks.max(1);
        if !(1..=16).contains(&waves) || waves % ks != 0 {
            return Err(refuse(
                OP,
                format!("{waves} wavefronts do not split {ks} ways"),
            ));
        }
        let groups = if swiglu { 2 * yt } else { yt };
        let red = if ks > 1 {
            waves * mu.div_ceil(4) * groups * 64
        } else {
            0
        };
        (waves, mu * (ku / 8 + 1) * 16, red)
    } else {
        (block_waves, mu * ku * 2, 0)
    };
    let act_global = act_global || staged + red > SKINNY_LDS;
    let symbol = if quad {
        format!(
            "::pie::linear::skinny4_bf16<{m}, {yt}, {unrl}, {}, {act_global}, {}>",
            epilogue.code(),
            ks.max(1)
        )
    } else {
        format!(
            "::pie::linear::skinny_bf16<{m}, {yt}, {unrl}, {}, {act_global}>",
            epilogue.code()
        )
    };
    let cap = match epilogue {
        SkinnyEpilogue::Softcap(cap) => cap,
        _ => 0.0,
    };
    let smem = if act_global { red } else { staged + red };
    ctx.fire(
        OP,
        Fire::at("linear/gemv.hip", crate::jit::symbol(&symbol))
            .apply(Launch::grid([grid, 1, 1], [64, block_waves, 1]).smem(smem)),
        &[
            ArgValue::Ptr(weight),
            ArgValue::Ptr(act),
            ArgValue::Ptr(out),
            ArgValue::I32(n),
            ArgValue::I32(k),
            ArgValue::I32(waves.min(block_waves) as i32),
            ArgValue::F32(cap),
        ],
    )
}
