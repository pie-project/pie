//! The vendor-library legs: hipBLASLt's grouped GEMM and rocBLAS's
//! batched GEMM over the routes bucketed by expert.

#[allow(clippy::wildcard_imports)]
use super::*;

/// Grouped GEMMs the hipBLASLt leg has launched, for tests.
#[cfg(all(feature = "rocm", feature = "test-support"))]
pub(super) static LT_FIRES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

#[cfg(all(feature = "rocm", feature = "test-support"))]
pub fn lt_grouped_fires() -> u64 {
    LT_FIRES.load(core::sync::atomic::Ordering::Relaxed)
}

/// The most the hipBLASLt leg stages: its routes' rows in and out. The load
/// lays that staging for good (`jit::device` places every slab at load), so
/// this bounds the device memory the leg holds: a chosen bound, not a
/// measured one; a wider shape takes the grouped kernels.
#[cfg(feature = "rocm")]
pub(super) const LT_STAGE_CAP: u64 = 3 << 30;

/// Whether the hipBLASLt leg takes `per` routes an expert where the profile
/// starts it at `from` (0: never) and the shape is of the class it takes
/// (`fits`).
#[cfg(feature = "rocm")]
pub(super) fn lt_wanted(ctx: &Ctx, per: u32, from: u32, fits: bool) -> bool {
    ctx.tuning().moe_lt_grouped() && fits && from > 0 && per >= from
}

/// The library and the solution the profile names for a contraction of `k`
/// (`kernel.moe.lt_tile_deep` from `kernel.moe.lt_deep_k`, else
/// `kernel.moe.lt_tile_shallow`): a library that did not open, or a tile it
/// does not hold, is refused by name.
#[cfg(feature = "rocm")]
pub(super) fn lt_solution(
    ctx: &Ctx,
    op: &'static str,
    dtype: Dtype,
    k: u32,
) -> Result<(&'static crate::hip::lt_grouped::Library, usize), Error> {
    let lib = crate::hip::lt_grouped::library(dtype).map_err(|why| refuse(op, why))?;
    let t = ctx.tuning();
    let tile = if k >= t.moe_lt_deep_k() {
        t.moe_lt_tile_deep()
    } else {
        t.moe_lt_tile_shallow()
    };
    let at = lib.pick(tile).ok_or_else(|| {
        refuse(
            op,
            format!("the profile's tile {tile} is not in hipBLASLt's grouped library"),
        )
    })?;
    Ok((lib, at))
}

/// The hipBLASLt leg's planes: routes in expert order, the staged result
/// rows, and the `UserArguments` array (one record an expert and the
/// sentinel).
#[cfg(feature = "rocm")]
pub(super) struct LtStage {
    ordered: RouteOrder,
    staged_out: u64,
    args: u64,
}

/// Orders the routes, gathers their rows by expert into a stream slab and
/// writes the experts' arguments (`pos`, when given, takes each route's row
/// in the order).
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub(super) fn stage_lt(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    n: u32,
    experts: u32,
    fan: &Selected,
    pos: u64,
    table: ExpertTable,
) -> Result<LtStage, Error> {
    use crate::hip::lt_grouped::USER_ARGS_BYTES;
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (k, rows) = (x.width, fan.route_count);
    let ordered = route_order(ctx, op, routes, fan, experts, GROUP_ROUTES_WIDE)?;
    let staged_in =
        ctx.scratch_stream(op, "linear.moe_lt_in", rows as usize * k as usize * 2)? as usize as u64;
    let staged_out = ctx.scratch_stream(op, "linear.moe_lt_out", rows as usize * n as usize * 2)?
        as usize as u64;
    let args = ctx.scratch(
        op,
        "linear.moe_lt_args",
        (experts as usize + 1) * USER_ARGS_BYTES,
    )? as usize as u64;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &format!("::pie::linear::moe_lt_scatter<{t}>"))).apply(Launch::grid(
            [copy_blocks(rows, k, ctx.multiprocessors(op)?), 1, 1],
            [256, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
            ArgValue::Ptr(staged_in),
            ArgValue::Ptr(args),
            ArgValue::Ptr(pos),
            bank.ptr.arg(),
            staged_out.arg(),
            act_div.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ],
    )?;
    Ok(LtStage {
        ordered,
        staged_out,
        args,
    })
}

/// Blocks a 16-byte grid-stride copy of `rows` rows of `width` takes, at
/// most eight on each of the device's `cus` compute units.
#[cfg(feature = "rocm")]
pub(super) fn copy_blocks(rows: u32, width: u32, cus: u32) -> u32 {
    let pieces = u64::from(rows) * u64::from(width / 8);
    pieces.div_ceil(256).clamp(1, u64::from(cus) * 8) as u32
}

/// One grouped GEMM over the arguments at `args`: `experts` groups and the
/// sentinel, a grid for `rows` routes however they fall.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub(super) fn lt_launch(
    op: &'static str,
    lib: &crate::hip::lt_grouped::Library,
    at: usize,
    args: u64,
    n: u32,
    rows: u32,
    experts: u32,
    stream: *mut core::ffi::c_void,
) -> Result<(), Error> {
    let grid = lib.solutions[at].worst_grid(n, rows, experts);
    // SAFETY: the argument array was written on this stream, its records
    // naming staged planes and the bank, all live through the fire.
    unsafe { lib.launch(at, experts + 1, args, grid, stream) }.map_err(|why| refuse(op, why))?;
    // A size-only walk launches nothing.
    #[cfg(feature = "test-support")]
    if !crate::sizing::active() {
        LT_FIRES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }
    Ok(())
}

#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_lt(
    op: &'static str,
    lib: &crate::hip::lt_grouped::Library,
    at: usize,
    stage: &LtStage,
    n: u32,
    rows: u32,
    experts: u32,
    stream: *mut core::ffi::c_void,
) -> Result<(), Error> {
    lt_launch(op, lib, at, stage.args, n, rows, experts, stream)
}

/// The routed swiglu MLP with its rows kept by expert from the gate-up to
/// the fold, over hipBLASLt's grouped GEMM: the routing scatter gathers the
/// token rows by expert once (noting where each route's row went), the
/// swiglu runs over the rows in that order, neither GEMM scatters back, and
/// the fold reads each route's row where it sits. From
/// `kernel.moe.lt_mlp_from` routes an expert, over two deep contractions
/// (hidden and intermediate at least `kernel.moe.lt_deep_k`: on MI300X
/// hipBLASLt's grouped tiles lose to the native legs over a 640-deep down
/// projection, Qwen3.8-Flash-Next's, at every prefill); whether it ran.
#[cfg(feature = "rocm")]
pub(super) fn mlp_lt(
    ctx: &Ctx,
    op: &'static str,
    p: &RoutedMlp,
    y: &mut Tensor,
) -> Result<bool, Error> {
    use crate::hip::lt_grouped::USER_ARGS_BYTES;
    let (x, i) = (p.x, p.intermediate);
    let dtype = x.dtype;
    let (tokens, top_k) = (p.routes.rows, p.routes.width);
    let (k, out) = (x.width, y.width);
    let experts = p.gate_up.rows;
    if !matches!(dtype, Dtype::Bf16 | Dtype::F16)
        || p.gate_up.dtype != dtype
        || p.down.dtype != dtype
        || y.dtype != dtype
        || p.weights.dtype != Dtype::F32
        || p.gate_up_experts.streams()
        || p.down_experts.streams()
        || top_k == 0
        || x.rows != tokens
        || y.rows != tokens
        || experts == 0
        || experts > ORDER_EXPERT_CAP
        || p.down.rows != experts
        || u64::from(p.gate_up.width) != 2 * u64::from(i) * u64::from(k)
        || u64::from(p.down.width) != u64::from(out) * u64::from(i)
        || [k, i, out].iter().any(|w| !w.is_multiple_of(8) || *w < 64)
    {
        return Ok(false);
    }
    let Some(rows) = tokens.checked_mul(top_k) else {
        return Ok(false);
    };
    let t = ctx.tuning();
    if !t.moe_grouped()
        || u64::from(rows) * u64::from(k + 3 * i + out) * 2 > LT_STAGE_CAP
        || !lt_wanted(
            ctx,
            rows / experts,
            t.moe_lt_mlp_from(),
            k >= t.moe_lt_deep_k() && i >= t.moe_lt_deep_k(),
        )
    {
        return Ok(false);
    }
    let (lib, up_at) = lt_solution(ctx, op, dtype, k)?;
    let (_, down_at) = lt_solution(ctx, op, dtype, i)?;
    let ty = dtype_dispatch!(op, dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = Selected {
        route_count: rows,
        top_k: stated(op, top_k)?,
        by_token: true,
    };
    let stream = ctx.stream();
    let pos = ctx.scratch(op, "linear.moe_lt_pos", rows as usize * 4)? as usize as u64;
    let stage = stage_lt(
        ctx,
        op,
        x,
        p.gate_up,
        p.routes,
        2 * i,
        experts,
        &fan,
        pos,
        p.gate_up_experts,
    )?;
    gemm_lt(op, lib, up_at, &stage, 2 * i, rows, experts, stream)?;

    // The swiglu over the rows in expert order.
    let hidden = ctx.scratch_stream(op, "linear.moe_lt_hidden", rows as usize * i as usize * 2)?
        as usize as u64;
    let clamp = if p.limit.is_some() { "true" } else { "false" };
    ctx.fire(
        op,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::moe_swiglu_ordered<{ty}, {clamp}>"),
        ))
        .apply(Launch::grid([rows, i.div_ceil(8 * 256), 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(stage.staged_out),
            ArgValue::Ptr(hidden),
            ArgValue::Ptr(stage.ordered.offsets),
            stated(op, experts)?.arg(),
            stated(op, i)?.arg(),
            p.limit.unwrap_or(f32::INFINITY).arg(),
        ],
    )?;

    // The down GEMM over those rows, its products left in expert order.
    let routed = ctx.scratch_stream(op, "linear.moe_lt_down", rows as usize * out as usize * 2)?
        as usize as u64;
    let args = ctx.scratch(
        op,
        "linear.moe_lt_down_args",
        (experts as usize + 1) * USER_ARGS_BYTES,
    )? as usize as u64;
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &format!("::pie::linear::moe_lt_args<{ty}>")))
            .apply(Launch::grid([1, 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(stage.ordered.offsets),
            ArgValue::Ptr(args),
            p.down.ptr.arg(),
            hidden.arg(),
            routed.arg(),
            stated(op, out)?.arg(),
            stated(op, i)?.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(p.down_experts.table),
            ArgValue::Ptr(p.down_experts.hits),
        ],
    )?;
    lt_launch(op, lib, down_at, args, out, rows, experts, stream)?;

    // The fold, each route's row read where it sits.
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &format!("::pie::linear::moe_lt_fold<{ty}>")))
            .apply(elementwise_rows(op, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            ArgValue::Ptr(pos),
            p.routes.arg(),
            p.weights.arg(),
            stated(op, top_k)?.arg(),
            stated(op, out)?.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(true)
}

#[cfg(feature = "rocm")]
pub(super) const fn block_rows(per_expert: u32) -> u32 {
    let want = per_expert.next_power_of_two();
    if want < 16 {
        16
    } else if want > 128 {
        128
    } else {
        want
    }
}

/// The most the rocBLAS leg stages (its blocks' rows in and out), held for
/// good as [`LT_STAGE_CAP`]'s: a chosen bound, not a measured one; a wider
/// shape takes the grouped kernels.
#[cfg(feature = "rocm")]
pub(super) const GROUP_WORKSPACE_CAP: u64 = 512 << 20;

/// Whether the rocBLAS leg seats its experts and its staged workspace fits.
#[cfg(feature = "rocm")]
pub(super) fn blas_worth(x: Tensor, y: &Tensor, fan: &Selected, count: u32) -> bool {
    if count > MAX_EXPERTS {
        return false;
    }
    let block = block_rows(fan.route_count.div_ceil(count));
    let blocks = u64::from(count) + u64::from(fan.route_count.div_ceil(block));
    let staged = (blocks * u64::from(block))
        .saturating_mul(u64::from(x.width) + u64::from(y.width))
        .saturating_mul(x.dtype.bytes_ceil());
    staged <= GROUP_WORKSPACE_CAP
}

/// The rocBLAS leg: routes bucketed into blocks of one expert, gathered,
/// multiplied as one batched GEMM, and scattered back.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub(super) fn select_grouped_blas(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: u32,
    table: ExpertTable,
) -> Result<(), Error> {
    use crate::hip::blas::{self, Operand};
    use crate::hip::sys;

    let (t, scalar) = match x.dtype {
        Dtype::Bf16 => ("::pie::bf16", sys::BLAS_BF16),
        Dtype::F16 => ("::pie::f16", sys::BLAS_F16),
        other => {
            return Err(refuse(
                op,
                format!("the grouped leg hands rocBLAS a 16-bit bank and this one is {other:?}"),
            ));
        }
    };
    let handle = ctx.blas(op)?;

    let fan = selected(op, x, routes, y)?;
    let per_expert = fan.route_count.div_ceil(experts);
    let block = block_rows(per_expert);
    let blocks = experts + fan.route_count.div_ceil(block);
    let rows = blocks.checked_mul(block).ok_or_else(|| {
        refuse(
            op,
            format!("{blocks} blocks of {block} rows overflow the fire"),
        )
    })?;

    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let route_count = stated(op, fan.route_count)?;
    let aligned_rows = stated(op, rows)?;
    let block_size = stated(op, block)?;
    let max_blocks = stated(op, blocks)?;
    let expert_count = stated(op, experts)?;

    let elem = x.dtype.bytes_ceil();
    let slab = |name: &'static str, bytes: u64| -> Result<u64, Error> {
        usize::try_from(bytes)
            .map_err(|_| {
                refuse(
                    op,
                    format!("{name} wants {bytes} bytes, past this host's usize"),
                )
            })
            .and_then(|bytes| ctx.scratch(op, name, bytes))
            .map(|ptr| ptr as usize as u64)
    };
    let sorted = slab("linear.moe_group_sorted", u64::from(rows) * 4)?;
    let expert_ids = slab("linear.moe_group_experts", u64::from(blocks) * 4)?;
    let staged_in = slab(
        "linear.moe_group_in",
        u64::from(rows) * u64::from(x.width) * elem,
    )?;
    let staged_out = slab(
        "linear.moe_group_out",
        u64::from(rows) * u64::from(y.width) * elem,
    )?;
    let ptrs = slab("linear.moe_group_ptrs", u64::from(blocks) * 3 * 8)?;
    let w_ptrs = ptrs;
    let act_ptrs = ptrs + u64::from(blocks) * 8;
    let out_ptrs = ptrs + u64::from(blocks) * 16;

    ctx.fire(
        op,
        Fire::at(kernel(FILE, "::pie::linear::moe_align_decode<::pie::i32>"))
            .apply(Launch::grid([1, 1, 1], [BLOCK, 1, 1]).smem((3 * experts + 34) * 4)),
        &[
            routes.arg(),
            ArgValue::Ptr(sorted),
            ArgValue::Ptr(expert_ids),
            ArgValue::ABSENT,
            route_count.arg(),
            expert_count.arg(),
            block_size.arg(),
            max_blocks.arg(),
            ArgValue::ABSENT,
        ],
    )?;

    let gather_fan = if fan.by_token { fan.top_k } else { 1 };
    ctx.fire(
        op,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::gather_moe_aligned_inputs<{t}>"),
        ))
        .apply(Launch::grid(
            [rows, x.width.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(sorted),
            ArgValue::Ptr(staged_in),
            route_count.arg(),
            aligned_rows.arg(),
            gather_fan.arg(),
            k.arg(),
            (-1i32).arg(),
            stated(op, x.rows)?.arg(),
        ],
    )?;

    ctx.fire(
        op,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::build_moe_leg_ptrs<{t}>"),
        ))
        .apply(Launch::flat(blocks, BLOCK)),
        &[
            ArgValue::Ptr(expert_ids),
            bank.arg(),
            ArgValue::Ptr(staged_in),
            ArgValue::Ptr(staged_out),
            ArgValue::Ptr(w_ptrs),
            ArgValue::Ptr(act_ptrs),
            ArgValue::Ptr(out_ptrs),
            max_blocks.arg(),
            block_size.arg(),
            k.arg(),
            n.arg(),
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
            ArgValue::Ptr(sorted),
            route_count.arg(),
        ],
    )?;

    // SAFETY: `handle` is this context's live rocBLAS handle. The three
    // pointer arrays hold `blocks` entries each, written by the launch above
    // into slabs this context owns and holds for the fire's life.
    unsafe {
        blas::bind(handle, ctx.stream())?;
        blas::gemm_batched(
            handle,
            sys::BLAS_OP_T,
            sys::BLAS_OP_N,
            n,
            block_size,
            k,
            Operand::new(w_ptrs, scalar, k),
            Operand::new(act_ptrs, scalar, k),
            Operand::new(out_ptrs, scalar, n),
            max_blocks,
            &mut ctx.blas_workspace(op),
        )
    }
    .map_err(|why| {
        refuse(
            op,
            format!("{why} at {max_blocks} blocks of M={block_size} N={n} K={k}"),
        )
    })?;

    ctx.fire(
        op,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::reorder_moe_aligned_output<{t}>"),
        ))
        .apply(Launch::grid(
            [rows, y.width.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            ArgValue::Ptr(staged_out),
            ArgValue::Ptr(sorted),
            y.arg(),
            route_count.arg(),
            aligned_rows.arg(),
            n.arg(),
            (-1i32).arg(),
            0i32.arg(),
            ArgValue::ABSENT,
        ],
    )
}
