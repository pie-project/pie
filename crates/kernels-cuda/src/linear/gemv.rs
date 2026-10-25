use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Kernel, Launch, aligned16, kernel, refuse};
use crate::linear::frame::{Epilogue, In};

const GEMV: &str = "linear/gemv.cu";

/// `out[n] = act . weight[n]` for one row: the gemv frame with the plain
/// store filled in. The GEMM tuner's `Gemv` tactic.
pub(crate) fn gemv_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    gemv(ctx, Epilogue::store(), weight, act, &[], &[out], n, k)
}

/// The gemv frame's kernel for `epi` over `n` columns, and whether it splits
/// each column's K over a block (`kernel.gemv.splitk_max_n` columns at
/// most) and its warps a block. `kernel.gemv.bf16`: Blackwell profiles pick
/// `narrow`.
pub(crate) fn gemv_kernel(
    epi: &Epilogue,
    n: i32,
    tuning: &crate::tuning::Tuning,
) -> (&'static Kernel, bool, u32) {
    let narrow = tuning.gemv_narrow;
    if n.unsigned_abs() <= tuning.gemv_splitk_max_n {
        let (class, warps) = if narrow { ("4, 2", 4) } else { ("8, 1", 8) };
        (epi.kernel("gemv_splitk_frame", class), true, warps)
    } else {
        let class = if narrow { "4, 2" } else { "4, 4" };
        (epi.kernel("gemv_frame", class), false, 4)
    }
}

/// `outs = epi(act . weight^T)` for one row on the gemv frame: a warp a
/// column (each part's weight row in turn), or a block a column split over
/// K up to `kernel.gemv.splitk_max_n` columns; `ins` the epilogue's inputs
/// and `outs` its results, each `n` wide.
#[allow(clippy::too_many_arguments)]
pub fn gemv(
    ctx: &Ctx,
    epi: &Epilogue,
    weight: u64,
    act: u64,
    ins: &[In],
    outs: &[u64],
    n: i32,
    k: i32,
) -> Result<(), Error> {
    const OP: &str = "linear.gemv";
    if n <= 0 || k <= 0 || k % 8 != 0 {
        return Err(refuse(
            OP,
            format!("needs n > 0 and k > 0 in whole eights, and was handed n={n}, k={k}"),
        ));
    }
    for (address, what) in [(weight, "the weight"), (act, "the activation")] {
        if address == 0 || !aligned16(address) {
            return Err(refuse(
                OP,
                format!(
                    "{what} is null or not 16-byte aligned, and the vectorised loads demand it"
                ),
            ));
        }
    }
    let planes = epi.planes(ins, outs, n.unsigned_abs())?;
    let tuning = ctx.tuning();
    let (kernel, split, warps) = gemv_kernel(epi, n, &tuning);
    let grid = if split {
        n.unsigned_abs()
    } else {
        n.unsigned_abs().div_ceil(warps)
    };
    ctx.fire(
        OP,
        Fire::at(kernel).apply(Launch::grid([grid, 1, 1], [32, warps, 1]).programmatic(tuning.pdl)),
        &[
            ArgValue::Ptr(weight),
            ArgValue::Ptr(act),
            ArgValue::of_slice(&planes),
            ArgValue::of(&NO_FIELDS),
            ArgValue::I32(n),
            ArgValue::I32(k),
        ],
    )
}

/// A generated body holds no fields; C++ gives the empty struct one byte.
const NO_FIELDS: u8 = 0;

/// The warps a row of the few-rows kernel splits K over: eight where K
/// divides so, the most blocks and the fewest dependent loads a lane.
fn rows_split(k: i32) -> i32 {
    [8, 4, 2, 1]
        .into_iter()
        .find(|s| (k / 8) % s == 0)
        .unwrap_or(1)
}

/// The blocks a cluster of the few-rows kernel splits a row's K over besides:
/// four for a product of a few columns (a router's), whose few blocks alone
/// keep too few loads in flight; else, or below sm_90 (no clusters), 1.
pub(crate) fn rows_cluster(n: i32, k: i32) -> u32 {
    // Thread block clusters arrive with sm_90.
    if crate::jit::device::compute_capability_major().is_some_and(|major| major < 9) {
        return 1;
    }
    let split = rows_split(k);
    let blocks = n.unsigned_abs().div_ceil(16 / split as u32);
    if blocks <= 32 && (k / 8) % (split * 4) == 0 {
        4
    } else {
        1
    }
}

/// The few-rows frame's kernel for `epi` at `m x n x k`, its blocks and its
/// cluster.
pub(crate) fn rows_kernel(epi: &Epilogue, m: i32, n: i32, k: i32) -> (&'static Kernel, u32, u32) {
    let split = rows_split(k);
    let rows = 16 / split;
    let cl = rows_cluster(n, k);
    (
        epi.kernel("rows_frame", &format!("{m}, {split}, {cl}")),
        n.unsigned_abs().div_ceil(rows as u32),
        cl,
    )
}

/// `out[m][n] = act[m] . weight[n]` for a few rows: the few-rows frame with
/// the plain store filled in. The GEMM tuner's `Rows` tactic.
#[allow(clippy::too_many_arguments)]
pub fn gemm_rows_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    pdl: bool,
) -> Result<(), Error> {
    gemm_rows(
        ctx,
        Epilogue::store(),
        weight,
        act,
        &[],
        &[out],
        m,
        n,
        k,
        pdl,
    )
}

/// `outs = epi(act . weight^T)` for a few rows (1..=16, a decode's) in one
/// launch, one dense bank: a block's eight warps stream weight rows, as many
/// warps a row as keep some 120 blocks busy (each a slice of K, summed in
/// slice order). Under `pdl` it launches as the previous kernel's
/// programmatic dependent, waiting on it in-kernel (sm_90 and later).
#[allow(clippy::too_many_arguments)]
pub fn gemm_rows(
    ctx: &Ctx,
    epi: &Epilogue,
    weight: u64,
    act: u64,
    ins: &[In],
    outs: &[u64],
    m: i32,
    n: i32,
    k: i32,
    pdl: bool,
) -> Result<(), Error> {
    const OP: &str = "linear.gemm_rows";
    let most = crate::linear::frame::ROWS_M as i32;
    if !(1..=most).contains(&m) || n <= 0 || k <= 0 || k % 8 != 0 || epi.parts() != 1 {
        return Err(refuse(
            OP,
            format!(
                "needs 1..={most} rows, k in whole eights and one dense bank, and was handed \
                 m={m}, k={k}, {} part(s)",
                epi.parts()
            ),
        ));
    }
    if weight == 0 || act == 0 || !aligned16(weight) || !aligned16(act) {
        return Err(refuse(
            OP,
            "the weight and the activation are 16-byte aligned, and nothing is null",
        ));
    }
    let planes = epi.planes(ins, outs, n.unsigned_abs())?;
    let (kernel, blocks, cl) = rows_kernel(epi, m, n, k);
    ctx.fire(
        OP,
        Fire::at(kernel).apply(
            Launch::grid([blocks * cl, 1, 1], [256, 1, 1])
                .programmatic(pdl)
                .cluster(cl),
        ),
        &[
            ArgValue::Ptr(weight),
            ArgValue::Ptr(act),
            ArgValue::of_slice(&planes),
            ArgValue::of(&NO_FIELDS),
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
        Fire::at(kernel(
            GEMV,
            &format!("::pie::linear::gemm_narrow_bf16_kernel<{n}>"),
        ))
        .apply(Launch::grid([m.unsigned_abs(), 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(weight),
            ArgValue::Ptr(act),
            ArgValue::Ptr(out),
            ArgValue::I32(k),
        ],
    )
}
