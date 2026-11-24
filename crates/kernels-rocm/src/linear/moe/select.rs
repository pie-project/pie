//! The selected experts' product over their routes: the batch, wide,
//! stream and GEMV legs, and the entry that picks one.

#[allow(clippy::wildcard_imports)]
use super::*;

pub fn matmul_select(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";

    let fan = selected(OP, x, routes, y)?;
    // Past 64 routes an expert at a decode batch, a bf16 bank on the matrix
    // core's GEMM tile: items of 128 routes against 128 weight rows.
    if let Some(count) = dense_experts(x, bank, y, experts)
        && x.dtype == Dtype::Bf16
        && x.width.is_multiple_of(64)
        && count <= ORDER_EXPERT_CAP
        && (65..=ctx.tuning().moe_dense_batch_until()).contains(&fan.route_count.div_ceil(count))
    {
        return select_gemm(ctx, OP, x, bank, routes, y, &fan, count, experts);
    }
    // A wide decode batch over a located bank: an item an expert's routes,
    // its weight read once.
    if let Some(count) = dense_experts(x, bank, y, experts)
        && let Some(mw) = wide_items(ctx, &fan, count, x)
    {
        return select_wide(ctx, OP, x, bank, routes, y, &fan, count, mw, experts);
    }
    // A decode batch over a located bank: each expert read once for all its
    // routes.
    if let Some(count) = dense_experts(x, bank, y, experts)
        && x.width.is_multiple_of(8)
        && let Some(mt) = batch_items(ctx, &fan, count, 64, ctx.tuning().moe_dense_batch_until())
    {
        let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
        let nbw = dense_nbw(mt);
        return select_batch(
            ctx,
            OP,
            FILE,
            &format!(
                "::pie::linear::moe_matmul_select_dense_batch<{t}, ::pie::i32({mt}), ::pie::i32({nbw})>"
            ),
            x,
            &[bank.arg()],
            routes,
            y,
            &fan,
            count,
            mt,
            nbw,
            &[
                ArgValue::Ptr(0),
                ArgValue::Ptr(0),
                ArgValue::Ptr(0),
                0i32.arg(),
                ArgValue::Ptr(experts.table),
                ArgValue::Ptr(experts.hits),
            ],
        );
    }
    if let Some(count) = grouped_worth(ctx, x, bank, y, &fan, experts) {
        if x.width.is_multiple_of(8) {
            return select_grouped(ctx, OP, x, bank, routes, y, count, &fan, experts);
        }
        // A row the native legs cannot read in 16-byte pieces: rocBLAS's
        // batched GEMM, eagerly. Where both run the native leg is several
        // times the faster.
        #[cfg(feature = "rocm")]
        if ctx.stage() == ArgValue::Ptr(0) && blas_worth(x, y, &fan, count) {
            return select_grouped_blas(ctx, OP, x, bank, routes, y, count, experts);
        }
    }
    select_gemv(ctx, OP, x, bank, routes, y, experts)
}

/// 16-row weight blocks a batch wave takes (the kernels' `NBW`), and so the
/// rows a batch block takes.
pub(super) const BATCH_NBW: u32 = 1;

/// 16-row weight blocks a dense batch wave takes for `mt`-route items: two
/// at 64, each activation piece read from LDS then feeding twice the products.
pub(super) const fn dense_nbw(mt: u32) -> u32 {
    if mt >= 64 { DENSE_WIDE_NBW } else { BATCH_NBW }
}
pub(super) const DENSE_WIDE_NBW: u32 = 2;

/// The routes a batch item takes, if the batch legs take this fire: from
/// `kernel.moe.batch_from` tokens' routes (one token's read each expert once
/// on the stream, which orders nothing), while the experts average (routes over
/// experts, rounded up) at most `until`. An item is never under 16 routes:
/// its products are one 16-route block whatever it holds, and a real
/// routing's busy experts (Qwen3.6 at 32 tokens: 98 of 256 experts, the
/// busiest past 16 routes) split into several items, each reading the
/// weight again (4-route items: 120 expert reads a layer, 13% slower). From
/// three routes an expert, 32 (at 128 tokens: 169 reads at 16, 154 at 32),
/// and past 25 the widest; an expert past its item takes several.
pub(super) fn batch_items(
    ctx: &Ctx,
    fan: &Selected,
    experts: u32,
    widest: u32,
    until: u32,
) -> Option<u32> {
    let t = ctx.tuning();
    let top_k = fan.top_k.unsigned_abs();
    if !t.moe_batch() || fan.route_count < t.moe_batch_from().saturating_mul(top_k) {
        return None;
    }
    let per = fan.route_count.div_ceil(experts);
    if per > until {
        return None;
    }
    Some(match per {
        0..=2 => 16,
        3..=25 => 32,
        _ => widest,
    })
}

/// A batch fire: routes ordered by expert into `mt`-route items, then one
/// item a block row against 64·nbw weight rows a block. `planes` are
/// the bank's arguments, between the work list and the result, and `tail`
/// (ending in the expert table and its counter) follows the extents.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_batch(
    ctx: &Ctx,
    op: &'static str,
    file: &'static str,
    entry: &str,
    x: Tensor,
    planes: &[ArgValue],
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    experts: u32,
    mt: u32,
    nbw: u32,
    tail: &[ArgValue],
) -> Result<(), Error> {
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let ordered = route_order(ctx, op, routes, fan, experts, mt)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let mut args = vec![
        x.arg(),
        ArgValue::Ptr(ordered.order),
        ArgValue::Ptr(ordered.offsets),
        ArgValue::Ptr(ordered.work),
    ];
    args.extend_from_slice(planes);
    args.extend([
        y.arg(),
        act_div.arg(),
        n.arg(),
        k.arg(),
        stated(op, experts)?.arg(),
    ]);
    args.extend_from_slice(tail);
    ctx.fire(
        op,
        Fire::at(kernel(file, entry)).apply(Launch::grid(
            [ordered.work_cap, y.width.div_ceil(64 * nbw), 1],
            [GROUP_BLOCK, 1, 1],
        )),
        &args,
    )
}

/// Weight rows a wide block takes (the kernel's `NT`).
pub(super) const WIDE_ROWS: u32 = 64;
pub(super) const WIDE_ROWS_TALL: u32 = 128;
/// Elements a wide stage takes of a row (the kernel's `KC`).
pub(super) const WIDE_K: u32 = 64;

/// The routes a wide wave takes (the kernel's `MW`; an item is four waves'),
/// if the wide leg takes this fire: from `kernel.moe.dense_wide_from` routes
/// an expert (averaged, rounded up) to `kernel.moe.dense_batch_until`, K
/// whole 64-element stages and the activation under 4 GiB. The item is the
/// least of 64, 128, 192 and 256 routes past the average by two of its
/// square roots and one (as `batch_items`), so each expert's routes are one
/// item.
pub(super) fn wide_items(ctx: &Ctx, fan: &Selected, experts: u32, x: Tensor) -> Option<u32> {
    let t = ctx.tuning();
    let top_k = fan.top_k.unsigned_abs();
    if !t.moe_batch()
        || t.moe_dense_wide_from() == 0
        || fan.route_count < t.moe_batch_from().saturating_mul(top_k)
        || !x.width.is_multiple_of(WIDE_K)
        || u64::from(x.rows) * u64::from(x.width) * 2 >= 0xffff_f000
    {
        return None;
    }
    let per = fan.route_count.div_ceil(experts);
    if per < t.moe_dense_wide_from() || per > t.moe_dense_batch_until() {
        return None;
    }
    // per ≤ (√cap − 1)² for cap = 64, 128, 192, 256.
    Some(match per {
        0..=49 => 16,
        50..=106 => 32,
        107..=165 => 48,
        _ => 64,
    })
}

/// The routes an item would hold the busiest expert in: what they average,
/// exactly where every token takes every expert, else past the average by
/// two of its square roots and one (`batch_items`' margin).
pub(super) fn stream_items(fan: &Selected, experts: u32) -> u32 {
    let per = fan.route_count.div_ceil(experts.max(1));
    if fan.top_k.unsigned_abs() >= experts {
        return per;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let margin = (2.0 * f64::from(per).sqrt()).ceil() as u32;
    per + margin + 1
}

/// A bf16 bank's select on the GEMM tile (`linear/fp8.hip`'s eight-wave
/// tile at bf16): routes ordered by expert into items of 128, each against
/// 128 weight rows a block.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_gemm(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    experts: u32,
    table: ExpertTable,
) -> Result<(), Error> {
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let ordered = route_order(ctx, op, routes, fan, experts, 128)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    // One slice of K: slicing measured no faster.
    ctx.fire(
        op,
        Fire::at(kernel("linear/fp8.hip", "::pie::linear::moe_bf16_gemm8")).apply(Launch::grid(
            [ordered.work_cap, y.width.div_ceil(128), 1],
            [2 * GROUP_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
            ArgValue::Ptr(ordered.work),
            bank.arg(),
            y.arg(),
            ArgValue::ABSENT,
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            stated(op, y.rows)?.arg(),
            stated(op, 2 * x.width / 128)?.arg(),
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ],
    )
}

/// A wide fire: routes ordered by expert into items of up to 4·`mw`, then
/// one item a block row against 64 or 128 weight rows a block.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_wide(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    experts: u32,
    mw: u32,
    table: ExpertTable,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let ordered = route_order(ctx, op, routes, fan, experts, 4 * mw)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    #[allow(clippy::cast_possible_truncation)]
    let act_bytes = (u64::from(x.rows) * u64::from(x.width) * 2) as u32;
    // Items an XCD takes (the kernel's `per_xcd`).
    let per_xcd = ordered.work_cap.div_ceil(8);
    // 128-row tiles once the items are 128 or 192 routes (the activation
    // read half as often), their weight two stages ahead at 192 (so the
    // tile fits the registers); 64 rows otherwise, four stages ahead.
    let nt = if (32..=48).contains(&mw) {
        WIDE_ROWS_TALL
    } else {
        WIDE_ROWS
    };
    let depth = if nt == WIDE_ROWS_TALL && mw >= 48 {
        2
    } else {
        4
    };
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &format!(
                "::pie::linear::moe_matmul_select_dense_wide<{t}, ::pie::i32({mw}), ::pie::i32({nt}), \
                 ::pie::i32({WIDE_K}), ::pie::i32({depth}), ::pie::i32(2)>"
            )))
        .apply(Launch::grid(
            [8 * per_xcd * y.width.div_ceil(nt), 1, 1],
            [GROUP_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
            ArgValue::Ptr(ordered.work),
            bank.arg(),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            act_bytes.arg(),
            per_xcd.arg(),
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ],
    )
}

/// Waves a stream block runs (`kStreamWaves`).
pub(super) const STREAM_WAVES: u32 = 4;
pub(super) const STREAM_LANES: u32 = 64;
pub(super) const STREAM_BLOCK: u32 = STREAM_WAVES * STREAM_LANES;
/// The LDS a block may hold.
pub(super) const STREAM_LDS: u32 = 64 << 10;

/// Unit-dot slots a stream wave holds in LDS, short of a single row.
pub(super) const STREAM_DOTS: u32 = 1024;

/// The slots a stream row takes (`stream_slots`): one a unit for a row of
/// fewer units than lanes, else one a lane.
pub(super) const fn stream_slots(units: u32) -> u32 {
    if units < STREAM_LANES {
        units
    } else {
        STREAM_LANES
    }
}

/// A stream codec as the launch sees it: codes a unit, units a lane
/// fetches a round (`kUnroll`), and whether it stages unit sums.
#[derive(Clone, Copy)]
pub(super) struct StreamCodec {
    pub(super) codes: u32,
    pub(super) unroll: u32,
    pub(super) xsum: bool,
}

/// The rows a stream wave takes for rows of `units` units: the count (≤ 64,
/// its slots within `STREAM_DOTS`) whose run wastes the least of its last
/// round of fetches, the more rows on a tie.
pub(super) fn stream_rows(units: u32, unroll: u32) -> u32 {
    let round = STREAM_LANES * unroll;
    let most = (STREAM_DOTS / stream_slots(units).max(1)).clamp(1, STREAM_LANES);
    (1..=most)
        .max_by_key(|&r| {
            let run = r * units;
            // The share of the fetched units the run uses, in 1/1024ths.
            (
                u64::from(run) * 1024 / u64::from(run.div_ceil(round) * round),
                r,
            )
        })
        .unwrap_or(1)
}

/// The decode stream's launch over `routes` routes of `n` rows: each wave
/// takes `stream_rows` rows, and the block stages the activation (and,
/// `xsum`, its unit sums) and its waves' unit dots in LDS. The rows a wave
/// takes ride along as the kernel's `rows_per_wave`.
pub(super) fn stream_launch(
    op: &'static str,
    routes: u32,
    n: u32,
    k: u32,
    codec: StreamCodec,
) -> Result<(Launch, i32), Error> {
    let units = k / codec.codes;
    let rows = stream_rows(units, codec.unroll);
    let lds = u64::from(2 * k).next_multiple_of(16)
        + if codec.xsum {
            u64::from(units.next_multiple_of(4)) * 4
        } else {
            0
        }
        + u64::from(STREAM_WAVES * rows) * u64::from(stream_slots(units)) * 4;
    if lds > u64::from(STREAM_LDS) {
        return Err(refuse(
            op,
            format!(
                "a {k}-wide row stages {lds} bytes of LDS, past the {STREAM_LDS} a block holds"
            ),
        ));
    }
    let launch = Launch::grid(
        [routes, n.div_ceil(STREAM_WAVES * rows), 1],
        [STREAM_BLOCK, 1, 1],
    )
    .smem(lds as u32);
    Ok((launch, stated(op, rows)?))
}

pub(crate) fn select_gemv(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<(), Error> {
    const VEC_WIDTH: u32 = 8;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(bank.dtype, x.dtype, "the bank rides the activation's dtype");
    let fan = selected(op, x, routes, y)?;
    if x.width == 0 || !x.width.is_multiple_of(VEC_WIDTH) {
        return Err(refuse(
            op,
            format!(
                "K is {}, not a whole number of {VEC_WIDTH}-element 16-byte loads",
                x.width
            ),
        ));
    }
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let (launch, rows) = stream_launch(
        op,
        fan.route_count,
        y.width,
        x.width,
        StreamCodec {
            codes: VEC_WIDTH,
            unroll: 8,
            xsum: false,
        },
    )?;
    let form = if fan.by_token { "by_token" } else { "by_route" };
    ctx.fire(
        op,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::moe_matmul_select_gemv_{form}<{t}>"),
        ))
        .apply(launch),
        &[
            routes.arg(),
            x.arg(),
            bank.arg(),
            y.arg(),
            fan.top_k.arg(),
            k.arg(),
            n.arg(),
            (i64::from(n) * i64::from(k)).arg(), // the bank's expert stride
            ArgValue::Ptr(experts.table),        // any table: one plain read a route
            ArgValue::Ptr(experts.hits),
            rows.arg(),
            ctx.stage(),
        ],
    )
}
