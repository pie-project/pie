use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::linear::dequant::{Planes, planes};
use crate::tensor::Tensor;
use dtype::Dtype;

/// The group of an affine 4-bit bank of bf16 factors that the tcgen05
/// GEMM takes (32, 64 or 128 codes), from its planes' widths.
#[cfg(feature = "cuda")]
pub(crate) fn affine_u4_group(x: Tensor, codes: Tensor, scales: Tensor, n: u32) -> Option<u32> {
    let (k, n) = (u64::from(x.width), u64::from(n));
    if x.dtype != Dtype::Bf16 || u64::from(codes.width) * 2 != k * n || k % 64 != 0 {
        return None;
    }
    let groups = u64::from(scales.width).checked_div(2 * n)?;
    let group = k.checked_div(groups)?;
    (groups * group == k && matches!(group, 32 | 64 | 128)).then_some(group as u32)
}

/// Whether the register-dequant MMA kernel takes a fire of `routes` routes
/// over `experts`: `kernel.moe.mxfp4_mma_from ..= kernel.moe.mxfp4_mma_until`,
/// its experts averaging `kernel.moe.mxfp4_mma_per_from` routes or more.
#[cfg(feature = "cuda")]
pub(super) fn mxfp4_mma_spans(ctx: &Ctx, routes: u32, experts: u32) -> bool {
    let t = ctx.tuning();
    t.mxfp4_mma_from <= routes
        && routes <= t.mxfp4_mma_until
        && routes.div_ceil(experts.clamp(1, EXPERT_CAP)) >= t.mxfp4_mma_per_from
}

#[cfg(feature = "cuda")]
pub(super) fn nvfp4_mma_spans(ctx: &Ctx, routes: u32) -> bool {
    let t = ctx.tuning();
    t.nvfp4_mma_from <= routes && routes <= t.nvfp4_mma_until
}

/// Whether a routed NVFP4 select of bf16 activations (W4A16) runs the
/// register-dequant MMA kernel: resident experts, whole 32-code groups, and
/// a width `kernel.moe.nvfp4_mma_from` to `_until` takes.
#[cfg(feature = "cuda")]
pub(crate) fn nvfp4_mma_worth(
    ctx: &Ctx,
    x: Tensor,
    y: &Tensor,
    experts: ExpertTable,
    fan: &Selected,
) -> bool {
    nvfp4_mma_spans(ctx, fan.route_count)
        && !experts.streams()
        && x.dtype == Dtype::Bf16
        && y.dtype == Dtype::Bf16
        && x.width > 0
        && x.width.is_multiple_of(32)
}

/// The routed NVFP4 select of bf16 activations on the register-dequant MMA
/// kernel: `scales` row-major e4m3 a 16 codes, `tensor_scales` an f32 a row.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_nvfp4_mma(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<bool, Error> {
    let fan = selected(op, x, routes, y)?;
    if !nvfp4_mma_worth(ctx, x, y, experts, &fan) {
        return Ok(false);
    }
    select_mxfp4_mma(
        ctx,
        op,
        x,
        codes,
        scales,
        None,
        routes,
        y,
        &fan,
        None,
        Some(tensor_scales),
        false,
    )?;
    Ok(true)
}

/// Whether a routed mxfp4 select runs the register-dequant MMA kernel: bf16
/// activations, resident experts, whole 32-code blocks, and a width
/// [`mxfp4_mma_spans`] takes.
#[cfg(feature = "cuda")]
pub(crate) fn mxfp4_mma_worth(
    ctx: &Ctx,
    x: Tensor,
    y: &Tensor,
    seat: GroupSeat,
    fan: &Selected,
    experts: u32,
) -> bool {
    mxfp4_mma_spans(ctx, fan.route_count, experts)
        && !seat.streams()
        && x.dtype == Dtype::Bf16
        && y.dtype == Dtype::Bf16
        && x.width > 0
        && x.width.is_multiple_of(32)
}

/// The most routes an expert averages that the register-held mxfp4
/// select takes; past it, the shared-memory-staged one.
#[cfg(feature = "cuda")]
pub(super) const STAGED_FROM: u32 = 16;

/// The fewest routes an expert averages that Hopper's staged wgmma select
/// takes from the register-held one (DeepSeek-V4-Flash's miniature at 24 a
/// 64-row decode: 11% less time a fire).
#[cfg(feature = "cuda")]
pub(super) const WGMMA_FROM: u32 = 13;

/// The fewest routes an expert averages that the staged slice takes on
/// `moe_select_mxfp4_mma_pipe`: below it the fire's 256-row CTAs fill the
/// card too thinly (DeepSeek-V4.1's miniature at 128 tokens: 7-30% slower,
/// at 512 even; 9-27% faster from 1024).
#[cfg(feature = "cuda")]
const PIPE_FROM: u32 = 192;

/// The stages of `moe_select_mxfp4_mma_pipe`'s ring, and a stage's bytes:
/// 64 routes' 128-code runs, 256 rows' codes and their scales.
#[cfg(feature = "cuda")]
const PIPE_STAGES: u32 = 3;
#[cfg(feature = "cuda")]
const PIPE_STAGE: u32 = 64 * 256 + 256 * 64 + 256 * 4;

/// The 128-code chunks a warp of the register-held mxfp4 select keeps in
/// flight: one, five CTAs an SM; a decode's few CTAs take more warps from
/// the K split instead.
#[cfg(feature = "cuda")]
pub(super) const MMA_DEPTH: u32 = 1;

/// The deepest K split the register-held mxfp4 select's CTA takes.
#[cfg(feature = "cuda")]
pub(super) const MMA_KSPLIT_MAX: u32 = 4;

/// The routed mxfp4 select on mma.sync, each expert's rows dequantized in
/// registers (`linear/moe_w4.cuh`); with `glu`, `y` is the GLU of the bank's
/// gate rows and the up rows after them, half the bank's width.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn select_mxfp4_mma(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    glu: Option<crate::linear::gemm_sm100::Glu>,
    tensor_scales: Option<Tensor>,
    scale_atoms: bool,
) -> Result<(), Error> {
    const THREADS: u32 = 128;
    // The converts read e8m0 scales a 32 codes and no tensor scale, so an
    // NVFP4 bank (e4m3 a 16 codes, a tensor scale a row) keeps the bf16 path.
    let cvt =
        ctx.tuning().mxfp4_cvt && crate::linear::gemm_sm100::supported() && tensor_scales.is_none();
    if scale_atoms && !cvt {
        return Err(refuse(
            op,
            "scales in tcgen05's atoms are read by the e2m1-convert select alone",
        ));
    }
    let (code, limit, alpha) = glu_code(op, glu)?;
    let bank_rows = if glu.is_some() { 2 * y.width } else { y.width };
    let experts = codes.rows.clamp(1, EXPERT_CAP);
    // An expert's slice is 8, 16 or 32 routes: as few 8-route blocks as the
    // average expert fills.
    let per = fan.route_count.div_ceil(experts);
    // Past `STAGED_FROM` routes an expert, its activations staged in shared
    // memory a CTA of 64 routes (`moe_select_mxfp4_mma_staged`), so a wide
    // fire streams each expert's codes once a 64.
    // Hopper's wgmma slice from `WGMMA_FROM`: the register-held kernel's
    // 32-route block is ALU-bound past 16 routes.
    let staged = !cvt
        && (per > STAGED_FROM
            || (hopper_wgmma() && per >= WGMMA_FROM && x.width.is_multiple_of(64)));
    let nb: u32 = match per {
        _ if staged => 8,
        0..=4 => 1,
        5..=12 => 2,
        _ => 4,
    };
    // A CTA lands 128 rows, or 64 GLU columns (their gate and up rows).
    let cols = if glu.is_some() { 64 } else { 128 };
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", bank_rows)?)?;
    let inter = stated(op, y.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    // A wide fire of a K of whole 128s takes the staged slice's eight-warp
    // ring of codes and routes (`moe_select_mxfp4_mma_pipe`), mxfp4 alone.
    let pipe = staged
        && per >= PIPE_FROM
        && !hopper_wgmma()
        && tensor_scales.is_none()
        && k % 128 == 0
        && ctx.smem_fits(PIPE_STAGES * PIPE_STAGE);
    let RouteOrder {
        order,
        offsets,
        work,
        work_cap,
    } = route_order_sliced(
        ctx,
        op,
        routes,
        fan,
        experts,
        8 * nb,
        true,
        if glu.is_some() || KEEP_NEXT.with(core::cell::Cell::get) {
            Share::Keep
        } else {
            Share::Take
        },
    )?;
    // A fire of a few routes splits K across a cluster of two CTAs
    // (`kernel.moe.mxfp4_cvt_ksplit_until`), each streaming half the codes.
    let ksplit = if cvt && fan.route_count <= ctx.tuning().mxfp4_cvt_ksplit_until {
        2
    } else {
        1
    };
    let grid = [work_cap * ksplit, y.width.div_ceil(cols), 1];
    // The e2m1 converts exist on the tcgen05 Blackwells alone. Two chunks in
    // flight a warp, a window of 8 chunks of one 8-route block: four CTAs an SM.
    let fire = if cvt {
        const DEPTH: u32 = 2;
        let win = 8 / nb;
        let blocks = match nb {
            1 => 4,
            2 => 3,
            _ => 2,
        };
        Fire::at(kernel(
            SELECT_FP4_SM100,
            &format!(
                "::pie::linear::moe_select_mxfp4_cvt<{nb}, {code}, {DEPTH}, {win}, {blocks}, \
                 {ksplit}>"
            ),
        ))
        .apply(
            Launch::grid(grid, [THREADS, 1, 1])
                .smem(nb * win * 4096)
                .cluster(ksplit),
        )
    } else if staged && hopper_wgmma() && x.width.is_multiple_of(64) {
        // Hopper: the staged slice on wgmma (`moe_select_fp4_wgmma`), the
        // codes dequantized to its A registers, a warpgroup's 128 rows (64
        // GLU columns) a CTA; a K of whole 64s (gpt-oss's 2880) steps its
        // last chunk's half. Two stages where the fire's slices fill the
        // card three CTAs an SM deep (their shared memory), else three.
        let nv = tensor_scales.is_some();
        let sms = ctx.multiprocessors().unwrap_or(132);
        let ctas = fan.route_count.div_ceil(64) * grid[1];
        let stages: u32 = if ctas >= 3 * sms - sms / 2 { 2 } else { 3 };
        let smem = stages * (64 * 256 + 128 * (64 + 8)) + 128 * 32 + 1024;
        Fire::at(kernel(
            SELECT_SM90,
            &format!("::pie::linear::moe_select_fp4_wgmma<{code}, {nv}, {stages}>"),
        ))
        .apply(Launch::grid(grid, [THREADS, 1, 1]).smem(smem))
    } else if pipe {
        let cols = if glu.is_some() { 128 } else { 256 };
        let tiles = y.width.div_ceil(cols);
        Fire::at(kernel(
            SELECT_FP4_SM80,
            &format!("::pie::linear::moe_select_mxfp4_mma_pipe<{code}, {PIPE_STAGES}>"),
        ))
        .apply(
            Launch::grid(
                if act_div == 1 {
                    [tiles, grid[0], 1]
                } else {
                    [grid[0], tiles, 1]
                },
                [256, 1, 1],
            )
            .smem(PIPE_STAGES * PIPE_STAGE),
        )
    } else if staged {
        const STAGES: u32 = 3;
        const WARPS: u32 = 8;
        let nv = tensor_scales.is_some();
        let grid = [grid[0], y.width.div_ceil(cols * WARPS / 4), 1];
        Fire::at(kernel(
            SELECT_FP4_SM80,
            &format!(
                "::pie::linear::moe_select_mxfp4_mma_staged<{nb}, {STAGES}, {code}, {nv}, {WARPS}>"
            ),
        ))
        .apply(
            Launch::grid(
                if act_div == 1 {
                    [grid[1], grid[0], 1]
                } else {
                    grid
                },
                [32 * WARPS, 1, 1],
            )
            .smem(STAGES * 8 * nb * 16 * 16),
        )
    } else {
        let depth = MMA_DEPTH;
        // A fire of few CTAs (a decode's routes) splits K across groups of
        // the CTA's warps, so the card's memory has warps enough to stream
        // the bank: as many as bring it to four CTAs' worth an SM.
        let sms = ctx.multiprocessors().unwrap_or(108);
        let ctas = (grid[0] * grid[1]).max(1);
        let ksplit = if nb == 1 {
            (4 * sms)
                .div_ceil(ctas)
                .next_power_of_two()
                .clamp(1, MMA_KSPLIT_MAX)
        } else {
            1
        };
        let nv = tensor_scales.is_some();
        Fire::at(kernel(
            SELECT_FP4_SM80, // NVFP4 (`nv`): an e4m3 scale a 16 codes, a tensor scale a row.
            &format!("::pie::linear::moe_select_mxfp4_mma<{nb}, {depth}, {code}, {nv}, {ksplit}>"),
        ))
        .apply(Launch::grid(grid, [THREADS * ksplit, 1, 1]))
    };
    let mut args = vec![
        x.arg(),
        ArgValue::Ptr(order),
        ArgValue::Ptr(offsets),
        ArgValue::Ptr(work),
        codes.arg(),
        scales.arg(),
        bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
        y.arg(),
        act_div.arg(),
        n.arg(),
        k.arg(),
        stated(op, experts)?.arg(),
        inter.arg(),
        limit.arg(),
        alpha.arg(),
    ];
    // The e2m1-convert kernel reads where its scales lie; the bf16 one, an
    // NVFP4 bank's tensor scales.
    if cvt {
        args.push(i32::from(scale_atoms).arg());
    } else {
        args.push(tensor_scales.map_or(ArgValue::ABSENT, |t| t.arg()));
    }
    ctx.fire(op, fire, &args)
}

/// Whether [`select_mxfp4_atoms`] takes a fire of `routes` over `x`'s
/// width (asked before its activations are read back for it).
#[cfg(feature = "cuda")]
#[must_use]
pub fn select_mxfp4_atoms_takes(ctx: &Ctx, routes: Tensor, x: Tensor) -> bool {
    let t = ctx.tuning();
    let count = routes.rows * routes.width;
    t.mxfp4_cvt
        && crate::linear::gemm_sm100::supported()
        && t.mxfp4_a8_from <= count
        && count <= t.mxfp4_a8_until
        && x.width > 0
        && x.width.is_multiple_of(32)
}

/// A routed mxfp4 select with tcgen05-atom scales (a W4A8 bank) on the
/// e2m1-convert register-dequant kernel, against bf16 activations.
/// `Ok(false)` where the kernel does not take the fire.
#[cfg(feature = "cuda")]
pub fn select_mxfp4_atoms(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    atoms: u64,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "linear.moe_select_mxfp4_atoms";
    let fan = selected(OP, x, routes, y)?;
    if !(select_mxfp4_atoms_takes(ctx, routes, x) && y.dtype == Dtype::Bf16) {
        return Ok(false);
    }
    let scales = Tensor::new(atoms, codes.rows, 0, Dtype::U8);
    select_mxfp4_mma(
        ctx, OP, x, codes, scales, None, routes, y, &fan, None, None, true,
    )?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_select_mxfp4(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const MXFP4_BLOCK: u32 = 32;

    const ROWS_PER_WARP: u32 = 4;

    const DECODE_BLOCK: u32 = 128;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "an mxfp4 bank's codes are u8");
    debug_assert_eq!(scales.dtype, Dtype::U8, "an mxfp4 bank's scales are u8");
    debug_assert!(
        bias.is_none_or(|bias| bias.dtype == x.dtype),
        "the expert bias rides the activation's dtype"
    );
    let fan = selected(op, x, routes, y)?;
    #[cfg(feature = "cuda")]
    if mxfp4_mma_worth(ctx, x, y, seat, &fan, codes.rows) {
        return select_mxfp4_mma(
            ctx, op, x, codes, scales, bias, routes, y, &fan, None, None, false,
        );
    }
    #[cfg(feature = "cuda")]
    if !seat.streams()
        && tc5_worth(
            ctx,
            x,
            (Dtype::Bf16, codes.rows),
            y,
            ExpertTable::RESIDENT,
            routes.rows,
            fan.route_count,
        )
    {
        use crate::linear::gemm_sm100;
        return select_tc5(
            ctx,
            op,
            x,
            gemm_sm100::Operand::scaled(codes.ptr, gemm_sm100::Elem::E2m1, scales.ptr, MXFP4_BLOCK),
            codes.rows,
            routes,
            y,
            gemm_sm100::Epilogue {
                bias: bias.map_or(0, |b| b.ptr),
                ..gemm_sm100::Epilogue::STORE
            },
        );
    }
    if x.width == 0 || !x.width.is_multiple_of(MXFP4_BLOCK) {
        return Err(refuse(
            op,
            format!(
                "K is {}, not a whole number of {MXFP4_BLOCK}-code mxfp4 blocks",
                x.width
            ),
        ));
    }
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    // A lean select (`kernel.moe.mxfp4_select_lean`) runs blocks of two warps.
    let lean = ctx.tuning().mxfp4_select_lean;
    let block = if lean { 2 * WARP } else { DECODE_BLOCK };
    let tile = (block / WARP) * ROWS_PER_WARP;
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    let experts = codes.rows.clamp(1, EXPERT_CAP);
    // Grouping pays once an expert serves more than
    // `kernel.moe.mxfp4_grouped_reuse` routes on average.
    if fan.route_count > ctx.tuning().mxfp4_grouped_reuse * experts && ctx.tuning().mxfp4_grouped {
        let RouteOrder {
            order,
            offsets,
            work,
            work_cap,
        } = route_order(ctx, op, routes, &fan, experts)?;
        let wmma = x.dtype == Dtype::Bf16 && ctx.tuning().mxfp4_wmma;
        let entry = if wmma {
            "::pie::linear::moe_matmul_select_mxfp4_wmma".to_string()
        } else {
            format!("::pie::linear::moe_matmul_select_mxfp4_grouped<{t}>")
        };
        let mut args = vec![x.arg(), ArgValue::Ptr(order), ArgValue::Ptr(offsets)];
        if wmma {
            args.push(ArgValue::Ptr(work));
        }
        args.extend([
            codes.arg(),
            scales.arg(),
            bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
        ]);
        return ctx.fire(
            op,
            Fire::at(kernel(if wmma { SELECT_SM80 } else { SELECT }, &entry)).apply(Launch::grid(
                [
                    if wmma { work_cap } else { experts },
                    y.width.div_ceil(GROUPED_TILE_N),
                    1,
                ],
                [GROUPED_BLOCK, 1, 1],
            )),
            &args,
        );
    }
    ctx.fire(
        op,
        Fire::at(kernel(
            SELECT,
            &format!(
                "::pie::linear::moe_matmul_select_mxfp4<{t}, ::pie::i32({ROWS_PER_WARP}){}>",
                if lean { ", true" } else { "" }
            ),
        ))
        .apply(Launch::grid(
            [fan.route_count, y.width.div_ceil(tile), 1],
            [block, 1, 1],
        )),
        &[
            x.arg(),
            routes.arg(),
            codes.arg(),
            scales.arg(),
            bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            y.arg(),
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_select_mlxu4(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const ROWS_PER_WARP: u32 = 4;

    const DECODE_BLOCK: u32 = 128;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(
        codes.dtype,
        Dtype::U8,
        "a packed bank's planes bind as bytes"
    );
    debug_assert_eq!(
        scales.dtype,
        Dtype::U8,
        "a packed bank's planes bind as bytes"
    );
    debug_assert_eq!(
        biases.dtype,
        Dtype::U8,
        "a packed bank's planes bind as bytes"
    );
    let fan = selected(op, x, routes, y)?;
    // A 4-bit bank of bf16 factors on the grouped tcgen05 GEMM.
    #[cfg(feature = "cuda")]
    if let Some(group) = affine_u4_group(x, codes, scales, y.width)
        && !seat.streams()
        && tc5_worth(
            ctx,
            x,
            (Dtype::Bf16, codes.rows),
            y,
            ExpertTable::RESIDENT,
            routes.rows,
            fan.route_count,
        )
    {
        use crate::linear::gemm_sm100;
        return select_tc5(
            ctx,
            op,
            x,
            gemm_sm100::Operand::affine(codes.ptr, scales.ptr, biases.ptr, group),
            codes.rows,
            routes,
            y,
            gemm_sm100::Epilogue::STORE,
        );
    }
    let k = stated(op, nonzero(op, "K, the bank's contracted width", x.width)?)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let elems = u64::from(x.width) * u64::from(y.width);
    let code_bits = u64::from(codes.width) * 8;
    let bits: u32 = match code_bits {
        b if b == elems * 8 => 8,
        b if b == elems * 4 => 4,
        b if b == elems * 2 => 2,
        _ => {
            return Err(refuse(
                op,
                format!(
                    "a {}-byte expert code row stores {elems} codes at neither two, four nor eight bits",
                    codes.width
                ),
            ));
        }
    };
    let factor_bytes = u64::from(scales.width);
    let groups_per_row = if factor_bytes > 0 && factor_bytes % (2 * u64::from(y.width)) == 0 {
        factor_bytes / (2 * u64::from(y.width))
    } else {
        0
    };
    let group = if groups_per_row > 0 && u64::from(x.width) % groups_per_row == 0 {
        u64::from(x.width) / groups_per_row
    } else {
        0
    };
    let group = u32::try_from(group).unwrap_or(0);
    if !matches!(group, 32 | 64 | 128) || !group.is_multiple_of(32 / bits) {
        return Err(refuse(
            op,
            format!(
                "{groups_per_row} factors over a {}-wide row is not a 32-, 64- or 128-code affine group",
                x.width
            ),
        ));
    }
    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP;
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    let experts = codes.rows.clamp(1, EXPERT_CAP);
    let tuning = ctx.tuning();
    if tuning.affine_grouped && fan.route_count >= tuning.affine_grouped_from {
        let RouteOrder {
            order,
            offsets,
            work,
            work_cap,
        } = route_order(ctx, op, routes, &fan, experts)?;
        let wmma = x.dtype == Dtype::Bf16 && tuning.affine_wmma;
        let entry = if wmma {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_wmma<::pie::i32({bits}), ::pie::i32({group})>"
            )
        } else {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_grouped<{t}, ::pie::i32({bits}), \
                 ::pie::i32({group})>"
            )
        };
        let mut args = vec![x.arg(), ArgValue::Ptr(order), ArgValue::Ptr(offsets)];
        if wmma {
            args.push(ArgValue::Ptr(work));
        }
        args.extend([
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
        ]);
        return ctx.fire(
            op,
            Fire::at(kernel(if wmma { SELECT_SM80 } else { SELECT }, &entry)).apply(Launch::grid(
                [
                    if wmma { work_cap } else { experts },
                    y.width.div_ceil(GROUPED_TILE_N),
                    1,
                ],
                [GROUPED_BLOCK, 1, 1],
            )),
            &args,
        );
    }
    ctx.fire(
        op,
        Fire::at(kernel(SELECT, &format!(
                "::pie::linear::moe_matmul_select_mlxu4<{t}, ::pie::i32({bits}), ::pie::i32({group}), \
                 ::pie::i32({ROWS_PER_WARP})>"
            )))
        .apply(Launch::grid(
            [fan.route_count, y.width.div_ceil(tile), 1],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            routes.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
}

// NVFP4 selects read at bf16 (W4A16): see `crate::linear::dequant`'s
// nvfp4 planes for the layout.

const NVFP4_WARP: u32 = 32;

const NVFP4_ROWS_PER_WARP: u32 = 4;

const NVFP4_DECODE_BLOCK: u32 = 128;

/// The grouped arm's K tile.
const NVFP4_TILE_K: u32 = 128;

/// The routed product: `y[route] = x[route or its token] · W[routes[route]]ᵀ`.
#[allow(clippy::too_many_arguments)]
pub fn select_nvfp4(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = selected(OP, x, routes, y)?;
    let Planes { n, k } = planes(OP, codes, scales, tensor_scales, y.width, x.width)?;
    // A narrow fire on the register-dequant MMA kernel where the profile
    // says; a launch that fails falls to the arms below.
    #[cfg(feature = "cuda")]
    if !seat.streams()
        && matches!(
            crate::linear::moe::select_nvfp4_mma(
                ctx,
                OP,
                x,
                codes,
                scales,
                tensor_scales,
                routes,
                y,
                crate::linear::moe::ExpertTable::RESIDENT,
            ),
            Ok(true)
        )
    {
        return Ok(());
    }
    // On the grouped tcgen05 GEMM (W4A16), the tensor scale a row in its
    // epilogue.
    #[cfg(feature = "cuda")]
    if !seat.streams()
        && k.is_multiple_of(64)
        && crate::linear::moe::tc5_worth(
            ctx,
            x,
            (Dtype::Bf16, codes.rows),
            y,
            crate::linear::moe::ExpertTable::RESIDENT,
            routes.rows,
            fan.route_count,
        )
    {
        use crate::linear::gemm_sm100;
        // A tcgen05 launch that fails falls back to the arm below.
        if crate::linear::moe::select_tc5(
            ctx,
            OP,
            x,
            gemm_sm100::Operand::scaled(codes.ptr, gemm_sm100::Elem::E2m1, scales.ptr, 16),
            codes.rows,
            routes,
            y,
            gemm_sm100::Epilogue {
                weight_scale: tensor_scales.ptr,
                ..gemm_sm100::Epilogue::STORE
            },
        )
        .is_ok()
        {
            return Ok(());
        }
    }
    nonzero(OP, "N, the bank's output width", n)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let experts = codes.rows.clamp(1, EXPERT_CAP);
    // Grouping pays once an expert serves more routes than the mxfp4 bank's
    // knob says, the same trade at the same width of code.
    if x.dtype == Dtype::Bf16
        && k.is_multiple_of(NVFP4_TILE_K)
        && ctx.tuning().mxfp4_grouped
        && fan.route_count > ctx.tuning().mxfp4_grouped_reuse * experts
    {
        return select_nvfp4_grouped(
            ctx,
            OP,
            x,
            codes,
            scales,
            tensor_scales,
            routes,
            y,
            seat,
            &fan,
            experts,
        );
    }
    select_nvfp4_routed(
        ctx,
        OP,
        t,
        x,
        Some(routes),
        codes,
        scales,
        tensor_scales,
        y,
        seat,
        fan.route_count,
        fan.top_k,
        act_div,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn select_nvfp4_grouped(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    fan: &Selected,
    experts: u32,
) -> Result<(), Error> {
    let RouteOrder {
        order,
        offsets,
        work,
        work_cap,
    } = route_order(ctx, op, routes, fan, experts)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    ctx.fire(
        op,
        Fire::at(kernel(
            SELECT_FP4_SM80,
            "::pie::linear::moe_matmul_select_nvfp4_mma",
        ))
        .apply(Launch::grid(
            [work_cap, y.width.div_ceil(GROUPED_TILE_N), 1],
            [GROUPED_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            codes.arg(),
            scales.arg(),
            tensor_scales.arg(),
            y.arg(),
            act_div.arg(),
            stated(op, y.width)?.arg(),
            stated(op, x.width)?.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn select_nvfp4_routed(
    ctx: &Ctx,
    op: &'static str,
    t: &str,
    x: Tensor,
    routes: Option<Tensor>,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    route_count: u32,
    top_k: i32,
    act_div: i32,
) -> Result<(), Error> {
    // A lean select (`kernel.moe.mxfp4_select_lean`) runs blocks of two warps.
    let lean = ctx.tuning().mxfp4_select_lean;
    let block = if lean {
        2 * NVFP4_WARP
    } else {
        NVFP4_DECODE_BLOCK
    };
    let tile = (block / NVFP4_WARP) * NVFP4_ROWS_PER_WARP;
    if route_count == 0 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(kernel(
            SELECT,
            &format!(
                "::pie::linear::moe_matmul_select_nvfp4<{t}, ::pie::i32({NVFP4_ROWS_PER_WARP}){}>",
                if lean { ", true" } else { "" }
            ),
        ))
        .apply(Launch::grid(
            [route_count, y.width.div_ceil(tile), 1],
            [block, 1, 1],
        )),
        &[
            x.arg(),
            routes.map_or(ArgValue::ABSENT, |routes| routes.arg()),
            codes.arg(),
            scales.arg(),
            tensor_scales.arg(),
            y.arg(),
            top_k.arg(),
            act_div.arg(),
            stated(op, y.width)?.arg(),
            stated(op, x.width)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            if routes.is_some() {
                ctx.stage()
            } else {
                ArgValue::ABSENT
            },
        ],
    )
}
