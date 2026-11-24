#![allow(unsafe_code)]
use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, kernel, refuse};

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
    let (instantiation, grid) = if rows <= ctx.tuning().gemv_splitk_max_n() {
        ("::pie::linear::gemv_bf16<4, true>", rows)
    } else {
        ("::pie::linear::gemv_bf16<4, false>", rows.div_ceil(WAVES))
    };
    ctx.fire(
        "linear.gemv",
        Fire::at(kernel("linear/gemv.hip", instantiation))
            .apply(Launch::grid([grid, 1, 1], [64, WAVES, 1])),
        &values,
    )
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

/// [`skinny_bf16`] landing the f32 products (a router's logits).
pub(crate) fn skinny_f32(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    skinny(ctx, weight, act, out, m, n, k, SkinnyEpilogue::F32)
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

/// `pie::linear::SkinnyNorm`, field for field: the rows a skinny kernel
/// multiplies normed where it stages them (`x + y` landed in `y_out`, then
/// the RMS norm by `w`), all null for a plain activation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SkinnyNorm {
    pub x: u64,
    pub y: u64,
    pub w: u64,
    pub y_out: u64,
    pub out: u64,
    pub eps: f32,
    pub plus_one: i32,
}

/// `out[m, n] = rmsnorm(x + y) · w[n, k]ᵀ` on the skinny kernel, the norm
/// computed by every block where it stages the rows (`x + y` landed in
/// `norm.y_out`, the normed rows in `norm.out`, by the first): one launch
/// where the norm and the product were two. Under a gated `epilogue` the
/// bank is a packed `[gate | up]` of 2n rows and `out` the n gated columns. `Ok(false)`, launching nothing, where that does not pay: the
/// blocks' redundant reads (`grid · m · k` rows of x, y and the norm's
/// weight) take longer at the device's copy rate than the launch they save
/// (the profile's measured `[rates]`), or the plan stages no rows in LDS.
#[allow(clippy::too_many_arguments)]
pub(crate) fn skinny_norm_bf16(
    ctx: &Ctx,
    weight: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    norm: SkinnyNorm,
    epilogue: SkinnyEpilogue,
) -> Result<bool, Error> {
    const OP: &str = "linear.skinny";
    let gated = matches!(epilogue, SkinnyEpilogue::Swiglu | SkinnyEpilogue::Geglu);
    let Some(plan) = skinny_norm_plan(ctx, m, n, k, gated)? else {
        return Ok(false);
    };
    norm_aligned(OP, weight, &norm)?;
    skinny_launch(ctx, weight, 0, out, m, n, k, epilogue, plan, Some(norm))?;
    Ok(true)
}

/// The skinny kernel's walk of a normed product, where it stages the rows
/// and the fold pays ([`fold_pays`]); `None` where it does not.
pub(crate) fn skinny_norm_plan(ctx: &Ctx, m: i32, n: i32, k: i32, gated: bool) -> Result<Option<SkinnyPlan>, Error> {
    if !super::skinny::fits(m, n, k) {
        return Ok(None);
    }
    let plan = skinny_plan(
        m.unsigned_abs(),
        n.unsigned_abs() * if gated { 2 } else { 1 },
        k.unsigned_abs(),
        gated,
        ctx.tuning().gemm_skinny_quad_from(),
        ctx.multiprocessors("linear.skinny")?,
    );
    // The staged rows and, beside them in the same LDS, the prologue's
    // per-row partial sums (`gemv.hip`'s `sums[ROWS * kNormGroups]` floats):
    // a fold whose rows fill the LDS on their own has no room for those, and
    // its launch asks the device for more than a block holds.
    let staged = m.unsigned_abs() * k.unsigned_abs() * 2 + m.unsigned_abs() * NORM_GROUPS * 4;
    // The prologue sums a row as `residual_add_rmsnorm`'s packed path does,
    // a row of at most NORM_ROW_PACKS packs (`gemv.hip`'s kNormBlock x
    // kNormPpt); a wider one stays apart.
    if plan.quad || plan.act_global || staged > SKINNY_LDS || k / 8 > NORM_ROW_PACKS {
        return Ok(None);
    }
    Ok(fold_pays(ctx, "skinny", m, n, k, gated, plan.grid).then_some(plan))
}

/// Whether a norm folded into a product of `grid` blocks pays: every block
/// reads the rows' addends and residuals and the norm's weight again, the
/// same few kilobytes each, which the last-level caches serve after the
/// first block (priced at device memory's rate where the profile states no
/// cached one), against the norm's own launch.
#[allow(clippy::too_many_arguments)]
fn fold_pays(ctx: &Ctx, kernel: &'static str, m: i32, n: i32, k: i32, gated: bool, grid: u32) -> bool {
    let t = ctx.tuning();
    let redundant = f64::from(grid) * f64::from(2 * m + 1) * f64::from(k) * 2.0;
    let gbps = if t.cached_gbps > 0 { t.cached_gbps } else { t.hbm_gbps };
    // GB/s is bytes a nanosecond.
    let folds =
        t.launch_ns > 0 && gbps > 0 && redundant / f64::from(gbps) < f64::from(t.launch_ns);
    told_fold(kernel, m, n, k, gated, grid, redundant, t.launch_ns, gbps, folds);
    folds
}

/// Refuses a fold whose operands are null or not 16-byte aligned.
fn norm_aligned(op: &'static str, weight: u64, norm: &SkinnyNorm) -> Result<(), Error> {
    for (address, what) in [
        (norm.x, "the residual's addend"),
        (norm.y, "the residual"),
        (norm.w, "the norm's weight"),
        (norm.y_out, "the residual's sum"),
        (norm.out, "the normed rows"),
        (weight, "the weight"),
    ] {
        if address == 0 || !aligned16(address) {
            return Err(refuse(op, format!("{what} is null or not 16-byte aligned")));
        }
    }
    Ok(())
}

/// The fold's choice for a shape, said once.
#[allow(clippy::too_many_arguments)]
fn told_fold(
    kernel: &'static str,
    m: i32,
    n: i32,
    k: i32,
    gated: bool,
    grid: u32,
    bytes: f64,
    launch_ns: u32,
    gbps: u32,
    folds: bool,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static TOLD: Mutex<Option<HashSet<(&'static str, i32, i32, i32, bool, u32)>>> = Mutex::new(None);
    let mut told = TOLD.lock().expect("told lock poisoned");
    if told.get_or_insert_with(HashSet::new).insert((kernel, m, n, k, gated, grid)) {
        eprintln!(
            "kernels-rocm: the norm {} the {m}x{n}x{k} {kernel}{} product: {grid} blocks' redundant {:.2} MB at {gbps} GB/s against a {launch_ns} ns launch",
            if folds { "folds into" } else { "stays apart from" },
            if gated { " gated" } else { "" },
            bytes / 1e6
        );
    }
}

/// What the skinny kernel lands of each product: it as is, the gated
/// activation of a packed `[gate | up]` bank of 2n rows (`n` columns),
/// `cap · tanh(v / cap)`, or the f32 product unrounded into an f32 plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SkinnyEpilogue {
    Store,
    Swiglu,
    Geglu,
    Softcap(f32),
    F32,
}

impl SkinnyEpilogue {
    const fn code(self) -> u32 {
        match self {
            Self::Store => 0,
            Self::Swiglu => 1,
            Self::Geglu => 2,
            Self::Softcap(_) => 3,
            Self::F32 => 4,
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

/// The registers a lane of the diagonal skinny kernel holds its sums and
/// reads in: of the 128 a lane of a 1024-thread block has, the rest for its
/// row pointers and indices. Past them the kernel spills into scratch.
const SKINNY_REGISTERS: u32 = 112;

/// The registers a lane holds for `m` activation rows, `wr` weight rows and
/// `unrl` reads of each in flight: a 4-float sum per row pair, a 16-byte
/// read per row and round.
const fn skinny_registers(m: u32, wr: u32, unrl: u32) -> u32 {
    4 * (m * wr + (wr + m) * unrl)
}

/// The LDS the skinny kernel stages its activation rows in.
pub(crate) const SKINNY_LDS: u32 = 64 << 10;

/// The most packs (eight bf16) a row a normed skinny kernel's prologue sums:
/// `gemv.hip`'s kNormBlock virtual threads of kNormPpt packs, the unfused
/// `residual_add_rmsnorm`'s packed path, whose sums it reproduces.
const NORM_ROW_PACKS: i32 = 256 * 4;

/// The partial sums a row the normed prologue keeps in LDS beside the staged
/// rows: `gemv.hip`'s kNormGroups, kNormBlock (256) over the wavefront (64).
const NORM_GROUPS: u32 = 256 / 64;

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
pub(crate) fn skinny_plan(
    m: u32,
    rows: u32,
    k: u32,
    swiglu: bool,
    quad_from: u32,
    cus: u32,
) -> SkinnyPlan {
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
    // Within the registers a lane of a 1024-thread block has: fewer reads in
    // flight first, then fewer rows a wavefront (neither moves the order a
    // lane sums its packs in).
    let (mut yt, mut unrl) = (yt, unrl);
    while skinny_registers(m, if swiglu { 2 * yt } else { yt }, unrl) > SKINNY_REGISTERS {
        if unrl > 1 {
            unrl -= 1;
        } else if yt > 1 {
            yt /= 2;
        } else {
            break;
        }
    }
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
        ctx.tuning().gemm_skinny_quad_from(),
        ctx.multiprocessors("linear.skinny")?,
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
    skinny_launch(ctx, weight, act, out, m, n, k, epilogue, plan, None)
}

/// The skinny kernel's launch, its rows normed where it stages them under
/// `norm` (`act` then unread).
#[allow(clippy::too_many_arguments)]
fn skinny_launch(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: SkinnyEpilogue,
    plan: SkinnyPlan,
    norm: Option<SkinnyNorm>,
) -> Result<(), Error> {
    const OP: &str = "linear.skinny";
    let (kernel, block_waves, smem) = skinny_kernel(m, k, epilogue, plan, norm.is_some())?;
    let SkinnyPlan { waves, grid, .. } = plan;
    let cap = match epilogue {
        SkinnyEpilogue::Softcap(cap) => cap,
        _ => 0.0,
    };
    let norm = norm.unwrap_or_default();
    let mut args = vec![
        ArgValue::Ptr(weight),
        ArgValue::Ptr(act),
        ArgValue::Ptr(out),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::I32(waves.min(block_waves) as i32),
        ArgValue::F32(cap),
    ];
    // The whole-product kernel takes no norm.
    if !plan.quad {
        args.push(ArgValue::of(&norm));
    }
    ctx.fire(
        OP,
        Fire::at(kernel).apply(Launch::grid([grid, 1, 1], [64, block_waves, 1]).smem(smem)),
        &args,
    )
}

/// The skinny kernel `plan` walks `m` rows of K `k` on with `epilogue`, its
/// block's wavefronts and its dynamic LDS bytes.
fn skinny_kernel(
    m: i32,
    k: i32,
    epilogue: SkinnyEpilogue,
    plan: SkinnyPlan,
    normed: bool,
) -> Result<(&'static crate::jit::Kernel, u32, u32), Error> {
    const OP: &str = "linear.skinny";
    let (mu, ku) = (m.unsigned_abs(), k.unsigned_abs());
    let SkinnyPlan {
        yt,
        unrl,
        waves,
        block_waves,
        act_global,
        quad,
        ks,
        ..
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
            "::pie::linear::skinny_bf16<{m}, {yt}, {unrl}, {}, {act_global}, {normed}>",
            epilogue.code()
        )
    };
    let smem = if act_global { red } else { staged + red };
    Ok((kernel("linear/gemv.hip", &symbol), block_waves, smem))
}

/// The blocks of the skinny kernel serving walks `m` rows (1..=8) of a
/// plain `n x k` bank with one compute unit of the current device holds at
/// once, at its launch's threads (64 a wavefront, its block's wavefronts)
/// and LDS: HIP's occupancy calculator over the kernel's registers and LDS
/// against the compute unit's (`hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`).
pub(crate) fn skinny_resident_blocks(ctx: &Ctx, m: i32, n: i32, k: i32) -> Result<u32, Error> {
    const OP: &str = "linear.skinny";
    if !super::skinny::fits(m, n, k) {
        return Err(refuse(
            OP,
            format!("covers 1..=8 rows of K in whole eights, not m={m} n={n} k={k}"),
        ));
    }
    let plan = skinny_plan(
        m.unsigned_abs(),
        n.unsigned_abs(),
        k.unsigned_abs(),
        false,
        ctx.tuning().gemm_skinny_quad_from(),
        ctx.multiprocessors(OP)?,
    );
    let (kernel, block_waves, smem) = skinny_kernel(m, k, SkinnyEpilogue::Store, plan, false)?;
    kernel.resident_blocks(64 * block_waves, smem)
}

/// The turned kernel's LDS a wavefront: a 16 x 256 bf16 turn.
const TURN_WAVE_LDS: u32 = 8 << 10;

/// The turned kernel's walk of a bank of `rows` rows (2n for a packed
/// `[gate | up]` one) over K `k`: wavefronts a block (`ks`, a 256-element
/// slice of K each, up to four), slices a wavefront (`r`, up to two) and
/// blocks (as many as the LDS keeps resident, at most a 16-row tile each).
/// `None` past K 2048 ([`turn_fits`]).
pub(crate) fn turn_plan(m: u32, rows: u32, k: u32, cus: u32) -> Option<(u32, u32, u32)> {
    if !turn_fits(m, rows, k) {
        return None;
    }
    let slices = k / 256;
    let ks = slices.min(if m > 16 { 8 } else { 4 });
    let r = slices.div_ceil(ks);
    let resident = cus * ((64 << 10) / (ks * TURN_WAVE_LDS));
    Some((ks, r, rows.div_ceil(16).min(resident)))
}

/// Whether the turned kernel covers `m` rows over `rows` weight rows of K
/// `k`: 1 to 32 rows, K in whole 256s up to 2048.
pub(crate) fn turn_fits(m: u32, rows: u32, k: u32) -> bool {
    (1..=32).contains(&m) && k != 0 && k.is_multiple_of(256) && k <= 2048 && rows != 0
}

/// `y[m, n] = act[m, k] · w[n, k]ᵀ` for 1 ≤ m ≤ 32 on the turned kernel
/// (`skinnyt_bf16`), K up to 2048 in whole 256s. Refuses what it does not
/// cover.
#[allow(clippy::too_many_arguments)]
pub(crate) fn skinnyt(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: SkinnyEpilogue,
) -> Result<(), Error> {
    turn_launch(ctx, weight, act, out, m, n, k, epilogue, None)
}

/// [`skinnyt`] over rows normed where the turned kernel stages them (in its
/// turn's LDS with their partial sums, within its wavefronts' 8 KB each: 15
/// rows at K 1024 under four), the rows' sums and
/// normed rows landed as `norm` says; `Ok(false)` where it does not cover
/// the shape or the fold does not pay ([`fold_pays`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn skinnyt_norm(
    ctx: &Ctx,
    weight: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    norm: SkinnyNorm,
    epilogue: SkinnyEpilogue,
) -> Result<bool, Error> {
    const OP: &str = "linear.skinnyt";
    let gated = matches!(epilogue, SkinnyEpilogue::Swiglu | SkinnyEpilogue::Geglu);
    if !skinnyt_norm_covers(ctx, m, n, k, gated)? {
        return Ok(false);
    }
    norm_aligned(OP, weight, &norm)?;
    turn_launch(ctx, weight, 0, out, m, n, k, epilogue, Some(norm))?;
    Ok(true)
}

/// Whether the turned kernel stages a normed product's rows (they and their
/// sums within its turn) and the fold pays ([`fold_pays`]).
pub(crate) fn skinnyt_norm_covers(ctx: &Ctx, m: i32, n: i32, k: i32, gated: bool) -> Result<bool, Error> {
    if m <= 0 || n <= 0 || k <= 0 || k / 8 > NORM_ROW_PACKS {
        return Ok(false);
    }
    let Some((ks, _, grid)) = turn_plan(
        m.unsigned_abs(),
        n.unsigned_abs() * if gated { 2 } else { 1 },
        k.unsigned_abs(),
        ctx.multiprocessors("linear.skinnyt")?,
    ) else {
        return Ok(false);
    };
    // The normed rows (each padded a 16-byte pack, spread over the LDS
    // banks) and their partial sums (four floats a row) in the turn.
    Ok(m.unsigned_abs() * (k.unsigned_abs() * 2 + 32) <= ks * TURN_WAVE_LDS
        && fold_pays(ctx, "turned", m, n, k, gated, grid))
}

#[allow(clippy::too_many_arguments)]
fn turn_launch(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: SkinnyEpilogue,
    norm: Option<SkinnyNorm>,
) -> Result<(), Error> {
    const OP: &str = "linear.skinnyt";
    let glu = matches!(epilogue, SkinnyEpilogue::Swiglu | SkinnyEpilogue::Geglu);
    let plan = turn_plan(
        m.unsigned_abs(),
        n.unsigned_abs() * if glu { 2 } else { 1 },
        k.unsigned_abs(),
        ctx.multiprocessors(OP)?,
    );
    let (Some((ks, r, grid)), true) = (plan, n > 0 && k > 0) else {
        return Err(refuse(
            OP,
            format!("covers 1..=32 rows of K up to 2048 in whole 256s, not m={m} n={n} k={k}"),
        ));
    };
    for (address, what) in [
        (weight, "the weight"),
        (if norm.is_some() { weight } else { act }, "the activation"),
        (out, "the output"),
    ] {
        if address == 0 || (what != "the output" && !aligned16(address)) {
            return Err(refuse(OP, format!("{what} is null or not 16-byte aligned")));
        }
    }
    let cap = match epilogue {
        SkinnyEpilogue::Softcap(cap) => cap,
        _ => 0.0,
    };
    let symbol = format!(
        "::pie::linear::skinnyt_bf16<{m}, {r}, {ks}, {}, {}>",
        epilogue.code(),
        norm.is_some()
    );
    ctx.fire(
        OP,
        Fire::at(kernel("linear/gemv.hip", &symbol)).apply(Launch::grid([grid, 1, 1], [64, ks, 1])),
        &[
            ArgValue::Ptr(weight),
            ArgValue::Ptr(act),
            ArgValue::Ptr(out),
            ArgValue::I32(n),
            ArgValue::I32(k),
            ArgValue::F32(cap),
            ArgValue::of(&norm.unwrap_or_default()),
        ],
    )
}

/// The rates step's probe of the on-chip read rate (`[rates] cached_gbps`,
/// engine-cuda `rates`): `blocks` blocks each read one `set_packs`-pack
/// (16-byte) set of the `sets` at `sets_base` once, block b the set b %
/// `sets`, as a normed skinny product's blocks read its rows. `sink` holds
/// `blocks` floats and is written by none.
pub fn cached_read_probe(
    ctx: &Ctx,
    sets_base: u64,
    set_packs: u32,
    sets: u32,
    blocks: u32,
    sink: u64,
) -> Result<(), Error> {
    ctx.fire(
        "linear.cached_read_probe",
        Fire::at(kernel("linear/gemv.hip", "::pie::linear::cached_read_probe"))
            .apply(Launch::grid([blocks, 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(sets_base),
            ArgValue::I32(set_packs as i32),
            ArgValue::I32(sets as i32),
            ArgValue::Ptr(sink),
        ],
    )
}
