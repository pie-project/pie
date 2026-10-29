#![allow(unsafe_code)]
#[cfg(feature = "cuda")]
use super::*;
#[cfg(feature = "cuda")]
use crate::error::Error;
#[cfg(feature = "cuda")]
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, kernel, nonzero, refuse, stated};
#[cfg(feature = "cuda")]
use crate::tensor::Tensor;
#[cfg(feature = "cuda")]
use dtype::Dtype;

/// Whether this device has the FP8 MMA the W8A8 kernels run on (sm_89 on):
/// where it has no tcgen05, a block-FP8 deployment's batches quantize their
/// rows as the deployment says rather than read them bf16 (`w8a16`).
#[cfg(feature = "cuda")]
#[must_use]
pub fn fp8_w8a8_capable() -> bool {
    crate::jit::cache::arch()
        .and_then(|a| {
            a.trim_start_matches("sm_")
                .trim_end_matches(char::is_alphabetic)
                .parse::<u32>()
                .ok()
        })
        .is_some_and(|cc| cc >= 89)
}

/// The fewest routes a block-FP8 select takes [`select_fp8_w8a8`] at, and
/// the rows of a dense plane [`matmul_fp8_w8a8`] takes, between a decode's
/// register-held W8A16 kernels and a prefill's widened cuBLAS GEMM.
pub const FP8_W8A8_FROM: u32 = 64;

pub const FP8_W8A8_DENSE: core::ops::RangeInclusive<u32> = 33..=256;

/// `x`'s rows as e4m3 in 1x128 blocks (codes, a biased exponent a block) in
/// stream scratch, for [`select_fp8_w8a8`] and [`matmul_fp8_w8a8`].
#[cfg(feature = "cuda")]
pub(super) fn quantize_rows_e4m3b128(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
) -> Result<(u64, u64), Error> {
    let (rows, k) = (x.rows as usize, x.width as usize);
    let codes = ctx.scratch_stream(op, "linear.fp8_w8a8_x", rows * k)? as u64;
    let exps = ctx.scratch_stream(op, "linear.fp8_w8a8_xe", rows * (k / 128))? as u64;
    let lanes = u32::try_from(rows * (k / 128) * 16).map_err(|_| refuse(op, "too many rows"))?;
    ctx.fire(
        op,
        Fire::at(kernel(ACT_QUANT, "::pie::linear::quantize_rows_e4m3b128"))
            .apply(Launch::grid([lanes.div_ceil(128), 1, 1], [128, 1, 1])),
        &[
            x.arg(),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(exps),
            stated(op, x.rows)?.arg(),
            stated(op, x.width)?.arg(),
        ],
    )?;
    Ok((codes, exps))
}

/// Whether the mma.sync fp8 kernels take a product of `x` against a
/// 128x128-block FP8 plane `n` rows wide: bf16 rows, K in whole 512s (as the
/// load lays the exponents), N in whole 128-row blocks.
#[cfg(feature = "cuda")]
#[must_use]
pub fn fp8_w8a8_takes(x: Tensor, n: u32) -> bool {
    x.dtype == Dtype::Bf16 && x.width > 0 && x.width.is_multiple_of(512) && n.is_multiple_of(128)
}

/// `y[route] = q(x[route or its token]) . W[routes[route]]^T` for a
/// 128x128-block FP8 bank (W8A8, `q` the rows' e4m3 1x128 quantization) on
/// mma.sync, off tcgen05: the rows quantized once, routes laid out by expert
/// on-device (graph-safe), each expert's codes read once a slice.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn select_fp8_w8a8(
    ctx: &Ctx,
    x: Tensor,
    codes: u64,
    atoms: u64,
    experts: u32,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select_fp8_w8a8";
    let fan = selected(OP, x, routes, y)?;
    let n = stated(OP, nonzero(OP, "N, the bank's output width", y.width)?)?;
    if !fp8_w8a8_takes(x, y.width) || experts == 0 {
        return Err(refuse(
            OP,
            "the fp8 select takes bf16 rows, K in 512s and N in 128s",
        ));
    }
    let (xq, xe) = quantize_rows_e4m3b128(ctx, OP, x)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let per = fan.route_count.div_ceil(experts);
    // On Hopper, wgmma (sm_90a): an expert's slice 32 routes, or 64 past an
    // average of 16, at about the card's rate either way, so a busy expert's
    // routes rarely spill into a second read of its codes.
    if hopper_wgmma() {
        let slice: u32 = if per <= 16 { 32 } else { 64 };
        let RouteOrder {
            order,
            offsets,
            work,
            work_cap,
        } = route_order_sliced(
            ctx,
            OP,
            routes,
            &fan,
            experts,
            slice,
            true,
            if KEEP_NEXT.with(core::cell::Cell::get) {
                Share::Keep
            } else {
                Share::Take
            },
        )?;
        // `kWgStages` stages of 128 code rows and the slice's route rows, a
        // 128-code K step each, past the 1024-byte alignment.
        let smem = 3 * (128 + slice) * 128 + 1024;
        return ctx.fire(
            OP,
            Fire::at(kernel(
                SELECT_SM90,
                &format!("::pie::linear::moe_select_fp8a8_wgmma<{slice}>"),
            ))
            .apply(Launch::grid([work_cap, y.width / 128, 1], [128, 1, 1]).smem(smem)),
            &[
                ArgValue::Ptr(xq),
                ArgValue::Ptr(xe),
                ArgValue::Ptr(order),
                ArgValue::Ptr(offsets),
                ArgValue::Ptr(work),
                ArgValue::Ptr(codes),
                ArgValue::Ptr(atoms),
                y.arg(),
                act_div.arg(),
                n.arg(),
                stated(OP, x.width)?.arg(),
                stated(OP, experts)?.arg(),
            ],
        );
    }
    // As the bf16 select: an expert's slice 8, 16, 32 or 64 routes.
    let nb: u32 = match per {
        0..=8 => 1,
        9..=16 => 2,
        17..=40 => 4,
        41..=64 => 8,
        _ => 16,
    };
    let wn: u32 = if nb >= 8 { 2 } else { 1 };
    let RouteOrder {
        order,
        offsets,
        work,
        work_cap,
    } = route_order_sliced(
        ctx,
        OP,
        routes,
        &fan,
        experts,
        8 * nb,
        true,
        if KEEP_NEXT.with(core::cell::Cell::get) {
            Share::Keep
        } else {
            Share::Take
        },
    )?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            SELECT_SM80,
            &format!("::pie::linear::moe_select_fp8a8_mma<{nb}, false, {wn}>"),
        ))
        .apply(
            Launch::grid([work_cap, y.width.div_ceil(128), 1], [128 * wn, 1, 1])
                .smem(fp8_mma_smem(nb)),
        ),
        &[
            ArgValue::Ptr(xq),
            ArgValue::Ptr(xe),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            y.arg(),
            ArgValue::ABSENT,
            act_div.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            stated(OP, experts)?.arg(),
            stated(OP, x.rows)?.arg(),
        ],
    )
}

/// `y = q(x) . W^T` for one 128x128-block FP8 plane (W8A8) on mma.sync, off
/// tcgen05: [`select_fp8_w8a8`]'s kernel over the rows in order.
#[cfg(feature = "cuda")]
pub fn matmul_fp8_w8a8(
    ctx: &Ctx,
    x: Tensor,
    codes: u64,
    atoms: u64,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_fp8_w8a8";
    if !fp8_w8a8_takes(x, y.width) || y.rows != x.rows || y.dtype != Dtype::Bf16 {
        return Err(refuse(
            OP,
            "the fp8 matmul takes bf16 rows, K in 512s and N in 128s",
        ));
    }
    if x.rows == 0 {
        return Ok(());
    }
    let (xq, xe) = quantize_rows_e4m3b128(ctx, OP, x)?;
    let nb: u32 = match x.rows {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    };
    // K split where the plane's row blocks and the row slices leave the
    // multiprocessors short of two CTAs each: every code still read once.
    let sms = ctx.multiprocessors().unwrap_or(132);
    let ctas = x.rows.div_ceil(8 * nb) * y.width.div_ceil(128);
    let splits = (2 * sms / ctas.max(1)).clamp(1, 8).min(x.width / 128);
    let count = x.rows as usize * y.width as usize;
    let part = if splits > 1 {
        ctx.scratch_stream(OP, "linear.fp8_w8a8_part", splits as usize * count * 4)? as u64
    } else {
        0
    };
    ctx.fire(
        OP,
        Fire::at(kernel(
            SELECT_SM80,
            &format!("::pie::linear::moe_select_fp8a8_mma<{nb}, true>"),
        ))
        .apply(
            Launch::grid(
                [x.rows.div_ceil(8 * nb), y.width.div_ceil(128), splits],
                [128, 1, 1],
            )
            .smem(fp8_mma_smem(nb)),
        ),
        &[
            ArgValue::Ptr(xq),
            ArgValue::Ptr(xe),
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            y.arg(),
            ArgValue::Ptr(part),
            1i32.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, x.width)?.arg(),
            1i32.arg(),
            stated(OP, x.rows)?.arg(),
        ],
    )?;
    if splits == 1 {
        return Ok(());
    }
    let quads = u32::try_from(count / 4).map_err(|_| refuse(OP, "too many outputs"))?;
    ctx.fire(
        OP,
        Fire::at(kernel(SELECT_SM80, "::pie::linear::fp8a8_fold"))
            .apply(Launch::grid([quads.div_ceil(256), 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(part),
            y.arg(),
            (count as i64).arg(),
            stated(OP, splits)?.arg(),
        ],
    )
}

/// Three stages of 128 weight rows and 8 nb route rows, each a 128-code K
/// step padded by 16 bytes (`kFp8a8Stages`).
#[cfg(feature = "cuda")]
pub(super) const fn fp8_mma_smem(nb: u32) -> u32 {
    (if nb == 16 { 2 } else { 3 }) * (128 + 8 * nb) * (128 + 16)
}

/// A routed select over a block-FP8 bank (`experts x n x k` e4m3 codes at
/// `codes`, one ue8m0 exponent a weight row a `gran` columns in atoms at
/// `atoms`) against bf16 activations (W8A16, `linear/moe_w8.cuh`), with the
/// GLU in its epilogue where `glu` says: the select where there are no FP8
/// tensor cores.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn select_fp8_w8a16(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: u64,
    atoms: u64,
    experts: u32,
    gran: u32,
    routes: Tensor,
    y: &mut Tensor,
    glu: Option<crate::linear::gemm_sm100::Glu>,
) -> Result<(), Error> {
    const THREADS: u32 = 128;
    const DEPTH: u32 = 4;
    let fan = selected(op, x, routes, y)?;
    if x.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(refuse(op, "a W8A16 select reads and lands bf16 rows"));
    }
    if !matches!(gran, 32 | 128) || !x.width.is_multiple_of(128) {
        return Err(refuse(
            op,
            format!(
                "a W8A16 select walks whole 128s of K in {gran}-wide blocks, not K {}",
                x.width
            ),
        ));
    }
    let (code, limit, alpha) = glu_code(op, glu)?;
    let bank_rows = if glu.is_some() { 2 * y.width } else { y.width };
    let experts = experts.clamp(1, EXPERT_CAP);
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", bank_rows)?)?;
    let inter = stated(op, y.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let SelectPlan {
        nb,
        tiled,
        tiles,
        ksplit,
        order,
    } = select_plan(
        ctx,
        op,
        routes,
        &fan,
        experts,
        y,
        glu.is_some(),
        fan.route_count >= FP8_GEMM_FROM,
        4,
    )?;
    let fire = if tiled {
        // Three stages of 64 routes' activations and 128 rows' codes, 64 of
        // K, and the widened tile: two CTAs an SM.
        const STAGES: u32 = 3;
        const WN: u32 = 2;
        let cols = if glu.is_some() { 64 } else { 128 };
        Fire::at(kernel(
            SELECT_SM80,
            &format!("::pie::linear::moe_select_fp8_gemm<{STAGES}, {code}, {gran}, {WN}>"),
        ))
        .apply(
            Launch::grid([order.work_cap, y.width.div_ceil(cols), 1], [64 * WN, 1, 1])
                .smem(STAGES * (64 * 128 + 128 * 64) + 128 * 128),
        )
    } else {
        let cols = if glu.is_some() { 64 } else { 64 * tiles };
        Fire::at(kernel(SELECT_SM80, &format!(
                "::pie::linear::moe_select_fp8_mma<{nb}, {DEPTH}, {code}, {gran}, {tiles}, {ksplit}>"
            )))
        .apply(Launch::grid(
            [order.work_cap, y.width.div_ceil(cols), 1],
            [THREADS * ksplit, 1, 1],
        ))
    };
    ctx.fire(
        op,
        fire,
        &[
            x.arg(),
            ArgValue::Ptr(order.order),
            ArgValue::Ptr(order.offsets),
            ArgValue::Ptr(order.work),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            inter.arg(),
            limit.arg(),
            alpha.arg(),
        ],
    )
}

/// The most rows [`matmul_grouped_fp8`] takes: the MMA's eight columns.
/// Past them cuBLAS's bf16 leads (mma.sync's fp8 at 16, 32 and 64 rows
/// took 20, 30 and 58 us to its 14 to 15, staged or split as it might be).
pub const GROUPED_FP8_ROWS: u32 = 8;

/// A grouped projection on fp8 planes (DeepSeek-V4's `wo_a`):
/// `y[r, g*n ..] = q(x[r, g*k .. g*k + k]) . w_g^T`, each plane e4m3 under a
/// ue8m0 exponent a 128x128 block, `q` the rows' e4m3 1x128 quantization, in
/// one launch on mma.sync. At most [`GROUPED_FP8_ROWS`] rows; `k` whole 512s,
/// `n` whole 64s.
#[cfg(feature = "cuda")]
pub fn matmul_grouped_fp8(
    ctx: &Ctx,
    x: Tensor,
    codes: u64,
    scales: u64,
    y: &mut Tensor,
    groups: u32,
) -> Result<(), Error> {
    // Four 16-row tiles a CTA, K in quarters a tile, two steps of codes in
    // flight a warp (measured against cuBLAS's bf16 at DeepSeek-V4-Flash).
    matmul_grouped_fp8_with(ctx, x, codes, scales, y, groups, Fp8Shape::GROUPED)
}

/// How [`matmul_grouped_fp8_with`] lays a plane's rows and K over a CTA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8Shape {
    /// 16-row tiles of the plane a CTA.
    pub tiles: u32,
    /// Warps a tile, each a share of K (`k` whole `128 split`s).
    pub split: u32,
    /// 128-wide steps of codes in flight a warp.
    pub depth: u32,
}

impl Fp8Shape {
    /// DeepSeek-V4's grouped output projection's.
    pub const GROUPED: Fp8Shape = Fp8Shape {
        tiles: 4,
        split: 4,
        depth: 2,
    };
}

/// The CTA shape a dense fp8b128 projection (`n x k`) at `rows` decode rows
/// runs [`matmul_grouped_fp8_with`] in where it leads the tcgen05 GEMM and the
/// fp8 GEMV; `None` where one of those does (DeepSeek-V4-Flash, measured).
#[must_use]
pub fn dense_fp8_shape(rows: u32, n: u32, k: u32) -> Option<Fp8Shape> {
    let big = u64::from(n) * u64::from(k) >= 32 << 20;
    let shape = |tiles, split, depth| Fp8Shape {
        tiles,
        split,
        depth,
    };
    let picked = match rows {
        0 | 1 => return None,
        2..=4 if n <= 1024 => return None,
        2..=4 if big && n >= 16384 => shape(2, 4, 2),
        _ if rows > GROUPED_FP8_ROWS => return None,
        5.. if big => return None,
        _ if n >= 8192 => shape(4, 4, 2),
        5.. if n <= 1024 => shape(1, 8, 4),
        _ => shape(2, 8, 2),
    };
    (k.is_multiple_of(128 * picked.split) && n.is_multiple_of(16 * picked.tiles)).then_some(picked)
}

/// [`matmul_grouped_fp8`] in the CTA shape `shape` (`k` whole `128
/// shape.split`s, `n` whole `16 shape.tiles`s).
#[cfg(feature = "cuda")]
pub fn matmul_grouped_fp8_with(
    ctx: &Ctx,
    x: Tensor,
    codes: u64,
    scales: u64,
    y: &mut Tensor,
    groups: u32,
    shape: Fp8Shape,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_grouped";
    let Fp8Shape {
        tiles,
        split,
        depth,
    } = shape;
    if x.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(refuse(OP, "the fp8 grouped leg reads and lands bf16 rows"));
    }
    if groups == 0
        || !x.width.is_multiple_of(groups)
        || !y.width.is_multiple_of(groups)
        || x.rows != y.rows
    {
        return Err(refuse(OP, "the groups do not divide the rows' widths"));
    }
    let (k, n) = (x.width / groups, y.width / groups);
    if !k.is_multiple_of(128 * split) || !n.is_multiple_of(16 * tiles) || y.rows > GROUPED_FP8_ROWS
    {
        return Err(refuse(
            OP,
            format!(
                "the fp8 grouped leg takes K in whole {}s, N in whole {}s and at most \
                 {GROUPED_FP8_ROWS} rows, not {} rows of [{n}, {k}]",
                128 * split,
                16 * tiles,
                y.rows
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    // The rows' codes (padded 16 bytes a row) and exponents, or the
    // tiles' sums of K handed across, whichever is more.
    let smem = (y.rows * (k + 16) + y.rows * (k / 128) * 4).max(tiles * (split - 1) * 4 * 32 * 4);
    ctx.fire(
        OP,
        Fire::at(kernel(
            SELECT_SM80,
            &format!(
                "::pie::linear::grouped_mma_fp8b128<::pie::i32({tiles}), ::pie::i32({split}), \
                 ::pie::i32({depth})>"
            ),
        ))
        .apply(Launch::grid([n / (16 * tiles), groups, 1], [32 * split * tiles, 1, 1]).smem(smem)),
        &[
            ArgValue::Ptr(codes),
            ArgValue::Ptr(scales),
            x.arg(),
            y.arg(),
            stated(OP, y.rows)?.arg(),
            stated(OP, n)?.arg(),
            stated(OP, k)?.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

/// A bf16 plane `[rows, k]` as fp8b128: e4m3 codes at `codes`, each 128x128
/// block's ue8m0 exponent spread in tcgen05's atoms at `atoms`. The exponent
/// is the least that holds the block's amax (DeepSeek's rule).
#[cfg(feature = "cuda")]
pub fn quantize_fp8b128(ctx: &Ctx, src: Tensor, codes: u64, atoms: u64) -> Result<(), Error> {
    const OP: &str = "linear.quantize_fp8b128";
    if src.dtype != Dtype::Bf16
        || !src.rows.is_multiple_of(128)
        || !src.width.is_multiple_of(128)
        || src.rows == 0
    {
        return Err(refuse(
            OP,
            format!(
                "quantizes a bf16 plane of whole 128x128 blocks, not {} x {} {:?}",
                src.rows, src.width, src.dtype
            ),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(kernel(ACT_QUANT, "::pie::linear::quantize_fp8b128")).apply(Launch::grid(
            [src.width / 128, src.rows / 128, 1],
            [256, 1, 1],
        )),
        &[
            src.arg(),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            stated(OP, src.rows)?.arg(),
            stated(OP, src.width)?.arg(),
        ],
    )
}

/// An MXFP4 bank (`rows` rows of `k` codes, an e8m0 scale a 32) as fp8b128
/// at `codes` and `atoms` ([`quantize_fp8b128`]) through bf16 at `wide`
/// (`rows * k * 2` bytes): each value exactly where its block's scales lie
/// within 2^14 of each other.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_to_fp8b128(
    ctx: &Ctx,
    src_codes: u64,
    src_scales: u64,
    rows: u32,
    k: u32,
    wide: u64,
    codes: u64,
    atoms: u64,
) -> Result<(), Error> {
    const OP: &str = "linear.mxfp4_to_fp8b128";
    let count = u64::from(rows) * u64::from(k);
    let threads = u32::try_from(count.div_ceil(8))
        .map_err(|_| refuse(OP, "the bank's 8-code runs pass a 32-bit launch"))?;
    ctx.fire(
        OP,
        Fire::at(kernel(DEQUANT, "::pie::linear::dequantize_mxfp4"))
            .apply(Launch::flat(threads, 256)),
        &[
            ArgValue::Ptr(src_codes),
            ArgValue::Ptr(src_scales),
            ArgValue::Ptr(wide),
            (count as i64).arg(),
        ],
    )?;
    quantize_fp8b128(ctx, Tensor::new(wide, rows, k, Dtype::Bf16), codes, atoms)
}

/// An fp8b128 plane (`codes` `[rows, k]`, each 128x128 block's exponent
/// spread to its rows in tcgen05's atoms at `atoms`) as bf16 at `out`, each
/// value exactly: the bf16 [`quantize_fp8b128`] read.
#[cfg(feature = "cuda")]
pub fn dequantize_fp8b128(
    ctx: &Ctx,
    codes: u64,
    atoms: u64,
    out: u64,
    rows: u32,
    k: u32,
) -> Result<(), Error> {
    const OP: &str = "linear.dequantize_fp8b128";
    if !k.is_multiple_of(128) {
        return Err(refuse(OP, format!("K {k} is not whole 128-wide blocks")));
    }
    let threads = u32::try_from((u64::from(rows) * u64::from(k)).div_ceil(16))
        .map_err(|_| refuse(OP, "the plane's 16-code runs pass a 32-bit launch"))?;
    ctx.fire(
        OP,
        Fire::at(kernel(DEQUANT, "::pie::linear::dequantize_fp8b128"))
            .apply(Launch::flat(threads, 256)),
        &[
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            ArgValue::Ptr(out),
            stated(OP, rows)?.arg(),
            stated(OP, k)?.arg(),
        ],
    )
}

/// A grouped projection whose routes send row group `g` to bank `g`: one
/// strided batched bf16 GEMM, `x` `[rows, groups * k]`, `w`
/// `[groups * n, k]`, `y` `[rows, groups * n]`.
#[cfg(feature = "cuda")]
pub fn matmul_grouped_strided(
    ctx: &Ctx,
    x: Tensor,
    w: Tensor,
    y: &mut Tensor,
    groups: u32,
) -> Result<(), Error> {
    use cudarc::cublas::sys::{
        cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmStridedBatchedEx,
        cublasOperation_t, cublasStatus_t, cudaDataType,
    };
    const OP: &str = "linear.matmul_grouped";
    if x.dtype != Dtype::Bf16 || w.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(refuse(OP, "the strided grouped leg reads and lands bf16"));
    }
    if groups == 0
        || !x.width.is_multiple_of(groups)
        || !y.width.is_multiple_of(groups)
        || x.rows != y.rows
    {
        return Err(refuse(OP, "the groups do not divide the rows' widths"));
    }
    let k = x.width / groups;
    let n = y.width / groups;
    if u64::from(w.rows) * u64::from(w.width) != u64::from(groups) * u64::from(n) * u64::from(k) {
        return Err(refuse(
            OP,
            format!(
                "the plane is [{}, {}], not {groups} groups of [{n}, {k}]",
                w.rows, w.width
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    // One row (a decode's) as one GEMV over every group's rows, where the
    // profile says (`kernel.gemm.grouped_gemv`): batched cuBLAS spends its
    // few CTAs on a plane a GEMV streams at full bandwidth.
    if ctx.tuning().grouped_gemv
        && y.rows == 1
        && k.is_multiple_of(8)
        && [x.ptr, w.ptr].iter().all(|p| p.is_multiple_of(16))
    {
        let rows = groups * n;
        return ctx.fire(
            OP,
            Fire::at(kernel(
                GEMV,
                "::pie::linear::gemv_bf16_grouped_kernel<::pie::i32(4)>",
            ))
            .apply(Launch::grid([rows.div_ceil(4), 1, 1], [32, 4, 1])),
            &[
                w.arg(),
                x.arg(),
                y.arg(),
                stated(OP, rows)?.arg(),
                stated(OP, k)?.arg(),
                stated(OP, n)?.arg(),
            ],
        );
    }
    let int =
        |v: u32| i32::try_from(v).map_err(|_| refuse(OP, "an extent past the int cuBLAS takes"));
    let handle = ctx.cublas(OP)?.cast::<cublasContext>();
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // Column-major: Y_g^T = W_g (op T) x X_g^T.
    // SAFETY: `handle` is this context's live cuBLAS handle bound to its stream;
    // the planes hold what the extents say.
    let status = unsafe {
        cublasGemmStridedBatchedEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            int(n)?,
            int(y.rows)?,
            int(k)?,
            core::ptr::from_ref(&alpha).cast(),
            w.ptr as usize as *const core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            int(k)?,
            i64::from(n) * i64::from(k),
            x.ptr as usize as *const core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            int(x.width)?,
            i64::from(k),
            core::ptr::from_ref(&beta).cast(),
            y.ptr as usize as *mut core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            int(y.width)?,
            i64::from(n),
            int(groups)?,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    };
    if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(refuse(
            OP,
            format!("`cublasGemmStridedBatchedEx` answered {}", status as i32),
        ))
    }
}
