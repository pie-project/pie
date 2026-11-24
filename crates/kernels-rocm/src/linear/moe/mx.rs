//! The quantized experts: MXFP4, the quantized and FP8 banks, MLX u4.

#[allow(clippy::wildcard_imports)]
use super::*;

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
    table: ExpertTable,
) -> Result<(), Error> {
    const MXFP4_BLOCK: u32 = 32;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "an mxfp4 bank's codes are u8");
    debug_assert_eq!(scales.dtype, Dtype::U8, "an mxfp4 bank's scales are u8");
    debug_assert!(
        bias.is_none_or(|bias| bias.dtype == x.dtype),
        "the expert bias rides the activation's dtype"
    );
    let fan = selected(op, x, routes, y)?;
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
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    // The grouped kernels read an expert once per batch of its routes, the
    // by-route one once per route but at the card's streaming rate, so grouping
    // pays only once an expert serves more than `kernel.moe.mxfp4_grouped_reuse`
    // routes on average; routes over experts is a floor on that average.
    let experts = codes.rows.clamp(1, ORDER_EXPERT_CAP);
    // A table that may name mapped host memory takes these arms too: a bank
    // streamed or held lands the same bits.
    if x.dtype == Dtype::Bf16
        && codes.rows <= ORDER_EXPERT_CAP
        && let Some(mt) = batch_items(ctx, &fan, experts, 32, ctx.tuning().moe_batch_until())
        // Short of 32 routes a lane reads its two scale bytes as one.
        && (mt == 32 || x.width.is_multiple_of(2 * MXFP4_BLOCK))
    {
        return select_batch(
            ctx,
            op,
            "linear/quant.hip",
            &format!(
                "::pie::linear::moe_matmul_select_mxfp4_batch<::pie::i32({mt}), ::pie::i32({BATCH_NBW})>"
            ),
            x,
            &[
                codes.arg(),
                scales.arg(),
                bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            ],
            routes,
            y,
            &fan,
            experts,
            mt,
            BATCH_NBW,
            &table.args(),
        );
    }
    if fan.route_count > ctx.tuning().mxfp4_grouped_reuse() * experts
        && ctx.tuning().mxfp4_grouped()
    {
        let ordered = route_order(ctx, op, routes, &fan, experts, group_routes(&fan, experts))?;
        let wmma = x.dtype == Dtype::Bf16 && ctx.tuning().mxfp4_wmma();
        let entry = if wmma {
            format!(
                "::pie::linear::moe_matmul_select_mxfp4_wmma<::pie::i32({})>",
                ordered.routes
            )
        } else {
            format!("::pie::linear::moe_matmul_select_mxfp4_grouped<{t}>")
        };
        let mut args = vec![
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
        ];
        if wmma {
            args.push(ArgValue::Ptr(ordered.work));
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
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ]);
        return ctx.fire(
            op,
            Fire::at(kernel("linear/quant.hip", &entry)).apply(if wmma {
                group_launch(ordered.work_cap, ordered.routes, y.width)
            } else {
                Launch::grid(
                    [experts, y.width.div_ceil(GROUP_TILE_N), 1],
                    [GROUP_BLOCK, 1, 1],
                )
            }),
            &args,
        );
    }
    let (launch, rows) = stream_launch(
        op,
        fan.route_count,
        y.width,
        x.width,
        StreamCodec {
            codes: MXFP4_BLOCK,
            unroll: 4,
            xsum: false,
        },
    )?;
    ctx.fire(
        op,
        Fire::at(kernel(
            "linear/quant.hip",
            &format!("::pie::linear::moe_matmul_select_mxfp4<{t}>"),
        ))
        .apply(launch),
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
            rows.arg(),
            ctx.stage(),
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ],
    )
}

/// An MXFP4 plane of `rows` x `k` (codes `[rows, k / 2]` at `codes`,
/// e8m0 exponents `[rows, k / 32]` at `scales`; `rows` a multiple of 16,
/// `k` of 128) into `out_codes` and `out_scales` in the matrix core's lane
/// order, which `matmul_select_mxfp4_lanes` reads: each 16-row strip as K /
/// 128 units of 1 KB of codes (lane l's 16 bytes one 32-code block of row
/// l % 16, each word's codes k 0 .. 3 in its bytes' low nibbles, 4 .. 7 in
/// the high ones) and 64 bytes of exponents (lane l's that block's).
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_lanes(
    ctx: &Ctx,
    op: &'static str,
    codes: u64,
    out_codes: u64,
    scales: u64,
    out_scales: u64,
    rows: u64,
    k: u64,
) -> Result<(), Error> {
    if rows == 0 || k == 0 {
        return Ok(());
    }
    if !rows.is_multiple_of(16)
        || !k.is_multiple_of(128)
        || !codes.is_multiple_of(16)
        || !out_codes.is_multiple_of(16)
    {
        return Err(refuse(
            op,
            format!("an mxfp4 plane of {rows} x {k} is not whole 16-row strips of 128-code units"),
        ));
    }
    let (r, kk) = (
        i32::try_from(rows).map_err(|_| refuse(op, "the plane's rows pass an i32"))?,
        i32::try_from(k).map_err(|_| refuse(op, "the plane's width passes an i32"))?,
    );
    let chunks = rows * k / 32;
    let grid = u32::try_from(chunks.div_ceil(256).min(16384)).unwrap_or(16384);
    ctx.fire(
        op,
        Fire::at(kernel("linear/fp8.hip", "::pie::linear::mxfp4_lanes"))
            .apply(Launch::grid([grid, 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(codes),
            ArgValue::Ptr(out_codes),
            ArgValue::Ptr(scales),
            ArgValue::Ptr(out_scales),
            ArgValue::I32(r),
            ArgValue::I32(kk),
        ],
    )
}

/// An MXFP4 bank in the matrix core's lane order (`mxfp4_lanes`, resident)
/// against the routes' bf16 rows: a token's routes each their own item
/// (`moe_mxfp4_route`), more in expert order, items of 16, 32 or 64 routes
/// holding what an expert takes (`moe_mxfp4_group`). The codes decode to
/// bf16 in registers under their block's exponent, exactly, onto the bf16
/// matrix core.
pub fn matmul_select_mxfp4_lanes(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    table: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    let op = OP;
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    // A table that may name mapped host memory reads as a located one: a
    // bank streamed lands the bits it does held.
    let fan = selected(op, x, routes, y)?;
    let experts = codes.rows;
    if !x.width.is_multiple_of(128)
        || !y.width.is_multiple_of(16)
        || experts == 0
        || experts > ORDER_EXPERT_CAP
    {
        return Err(refuse(
            op,
            "a bank in lane order takes whole 16-row strips of 128-code units",
        ));
    }
    // The tile reads the activation at 32-bit offsets from its base.
    if u64::from(x.rows) * u64::from(x.width) * 2 > u64::from(u32::MAX) {
        return Err(refuse(
            op,
            "the lane-order tile reads an activation under 4 GiB",
        ));
    }
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    if routes.rows == 1 {
        // Four waves a block, a strip each, three stages read ahead.
        let (rnw, rtn, rnb) = (4, 1, 4);
        return ctx.fire(
            op,
            Fire::at(kernel("linear/fp8.hip", &format!("::pie::linear::moe_mxfp4_route<::pie::i32({rnw}), ::pie::i32({rtn}), ::pie::i32({rnb})>")))
                .apply(Launch::grid([fan.route_count, y.width.div_ceil(16 * rnw * rtn), 1], [64 * rnw, 1, 1])),
            &[
                x.arg(),
                routes.arg(),
                codes.arg(),
                scales.arg(),
                y.arg(),
                act_div.arg(),
                n.arg(),
                k.arg(),
                stated(op, experts)?.arg(),
                ArgValue::Ptr(table.table),
                ArgValue::Ptr(table.hits),
            ],
        );
    }
    let (nw, tm, nb, tn) = mxfp4_lanes_shape(stream_items(&fan, experts));
    let ordered = route_order(ctx, op, routes, &fan, experts, 16 * tm)?;
    ctx.fire(
        op,
        Fire::at(kernel("linear/fp8.hip", &format!(
                "::pie::linear::moe_mxfp4_group<::pie::i32({nw}), ::pie::i32({tm}), ::pie::i32({nb}), ::pie::i32({tn})>"
            )))
        .apply(Launch::grid([ordered.work_cap, y.width.div_ceil(16 * tn * nw), 1], [64 * nw, 1, 1])),
        &[
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
            ArgValue::Ptr(ordered.work),
            codes.arg(),
            scales.arg(),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ],
    )
}

/// The lane-order group tile's shape for items holding `need` routes: its
/// waves, 16-route blocks, ring depth and strips a wave. Decoding a block's
/// codes costs more than its matrix-core steps, so the items grow with the
/// routes to share each decode; past 32 routes eight waves share a stage's
/// 64 rows of activations in LDS (dsv41's banks: 1024 tokens 1380 -> 686 us,
/// 8192 tokens 3895 -> 2398 us).
pub(super) fn mxfp4_lanes_shape(need: u32) -> (u32, u32, u32, u32) {
    match need {
        0..=16 => (4, 1, 2, 2),
        17..=32 => (4, 2, 2, 2),
        _ => (8, 4, 2, 2),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_select_bias(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    table: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    matmul_select_mxfp4(ctx, OP, x, codes, scales, Some(bias), routes, y, table)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_select_quant(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    table: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    // A byte a code beside four a 128x128 block: an fp8 bank (`e4m3tile128`).
    if biases.is_none() && fp8_tiles(x, codes, scales, y) {
        return Err(refuse(
            OP,
            "an fp8 bank (e4m3tile128) is read against e4m3 activations under f32 scales a 1x128 block \
             (`matmul_select_fp8_mfma`), not as the routes' rows are",
        ));
    }
    match biases {
        None => matmul_select_mxfp4(ctx, OP, x, codes, scales, None, routes, y, table),
        Some(biases) => matmul_select_mlxu4(ctx, OP, x, codes, scales, biases, routes, y, table),
    }
}

/// Whether `codes` and `scales` are an fp8 bank's planes for `x` into `y`:
/// a byte an expert's code, an f32 its 128x128 block's factor.
pub fn fp8_tiles(x: Tensor, codes: Tensor, scales: Tensor, y: &Tensor) -> bool {
    let (n, k) = (u64::from(y.width), u64::from(x.width));
    n > 0
        && k > 0
        && u64::from(codes.width) == n * k
        && u64::from(scales.width) == 4 * n.div_ceil(128) * k.div_ceil(128)
}

/// An fp8 bank (`e4m3tile128`, its codes made canonical at load) against
/// the routes' activation rows (bf16) quantized as e4m3 under f32 scales a
/// 1x128 block (`quantized_codes`), on the FP8 matrix core: at decode
/// batches items of the least of 16, 32 and 64 routes holding what an expert
/// averages; past 64 routes an expert the prefill tile, items of 128 against
/// 256 weight rows. An expert past its item takes another.
pub fn matmul_select_fp8_mfma(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    table: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    let (q, act_scales) = crate::linear::act_quant::quantized_codes(ctx, OP, x, 128, false, 1)?;
    matmul_select_fp8_quantized(ctx, x, q, act_scales, codes, scales, routes, y, table)
}

/// `matmul_select_fp8_mfma` over rows already quantized (`q`, `act_scales`
/// of `x`'s rows, `x` itself the routes' bf16 rows they came from).
#[allow(clippy::too_many_arguments)]
pub fn matmul_select_fp8_quantized(
    ctx: &Ctx,
    x: Tensor,
    q: Tensor,
    act_scales: Tensor,
    codes: Tensor,
    scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    table: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    let op = OP;
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    if q.rows != x.rows
        || q.width != x.width
        || act_scales.rows != x.rows
        || act_scales.width != x.width / 128
    {
        return Err(refuse(
            op,
            "the quantized rows are not x's codes beside a scale a 128-wide block",
        ));
    }
    // A table that streams names pool slots or the weight cache's host
    // copy (pinned, or the file's mapping): the tiles read an expert's
    // codes and factors where its entries say, device or host, as the
    // lane-order MXFP4 tiles do.
    let fan = selected(op, x, routes, y)?;
    if !x.width.is_multiple_of(128) || !fp8_tiles(x, codes, scales, y) {
        return Err(refuse(
            op,
            "the bank's planes are not e4m3 codes beside an f32 factor a 128x128 block of K",
        ));
    }
    if codes.rows > ORDER_EXPERT_CAP {
        return Err(refuse(
            op,
            format!(
                "the batch legs order at most {ORDER_EXPERT_CAP} experts, and this fp8 bank holds {}",
                codes.rows
            ),
        ));
    }
    let experts = codes.rows.max(1);
    let per = fan.route_count.div_ceil(experts);
    // The prefill tile walks whole 16-row strips and reads the activation at
    // 32-bit offsets.
    let gemm = per > 64
        && y.width.is_multiple_of(16)
        && u64::from(x.rows) * u64::from(x.width) <= u64::from(u32::MAX);
    let mt = if gemm {
        128
    } else {
        crate::linear::act_quant::fp8_item_rows(per)
    };
    let kk = stated(op, x.width)?;
    let nn = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let ordered = route_order(ctx, op, routes, &fan, experts, mt)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let mut args = vec![q.arg(), act_scales.arg()];
    let (entry, rows, threads) = if gemm {
        // The dense prefill tile (128 routes x 256 weight rows, eight waves
        // at two a SIMD), its rows gathered by route.
        (
            "::pie::linear::moe_fp8_tiles<::pie::i32(3), ::pie::i32(8), ::pie::i32(1), ::pie::i32(2), \
             ::pie::i32(8), ::pie::i32(2)>"
                .to_string(),
            256,
            512,
        )
    } else {
        (
            format!(
                "::pie::linear::moe_fp8_mfma<::pie::i32({mt}), ::pie::i32({})>",
                crate::linear::act_quant::fp8_strips()
            ),
            64 * crate::linear::act_quant::fp8_strips(),
            GROUP_BLOCK,
        )
    };
    args.extend([
        ArgValue::Ptr(ordered.order),
        ArgValue::Ptr(ordered.offsets),
        ArgValue::Ptr(ordered.work),
        codes.arg(),
        scales.arg(),
        y.arg(),
        act_div.arg(),
        nn.arg(),
        kk.arg(),
        stated(op, experts)?.arg(),
        ArgValue::Ptr(table.table),
        ArgValue::Ptr(table.hits),
    ]);
    ctx.fire(
        op,
        Fire::at(kernel("linear/fp8.hip", &entry)).apply(Launch::grid(
            [ordered.work_cap, y.width.div_ceil(rows), 1],
            [threads, 1, 1],
        )),
        &args,
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
    table: ExpertTable,
) -> Result<(), Error> {
    const UNIT: u32 = 32;

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
    if !matches!(group, 32 | 64 | 128) {
        return Err(refuse(
            op,
            format!(
                "{groups_per_row} factors over a {}-wide row is not a 32-, 64- or 128-code affine group",
                x.width
            ),
        ));
    }
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    let experts = codes.rows.clamp(1, ORDER_EXPERT_CAP);
    // The batch legs read four-bit codes, and a lane's factors of a stage
    // (an even count of groups) as one read. A table that may name mapped
    // host memory takes them too: a bank streamed or held lands the same
    // bits.
    if bits == 4
        && x.dtype == Dtype::Bf16
        && codes.rows <= ORDER_EXPERT_CAP
        && x.width.is_multiple_of(2 * group)
        && let Some(mt) = batch_items(ctx, &fan, experts, 32, ctx.tuning().moe_batch_until())
    {
        return select_batch(
            ctx,
            op,
            "linear/quant.hip",
            &format!(
                "::pie::linear::moe_matmul_select_mlxu4_batch<::pie::i32({bits}), ::pie::i32({group}), \
                 ::pie::i32({mt}), ::pie::i32({BATCH_NBW})>"
            ),
            x,
            &[codes.arg(), scales.arg(), biases.arg()],
            routes,
            y,
            &fan,
            experts,
            mt,
            BATCH_NBW,
            &table.args(),
        );
    }
    let tuning = ctx.tuning();
    if tuning.affine_grouped() && fan.route_count >= tuning.affine_grouped_from() {
        let ordered = route_order(ctx, op, routes, &fan, experts, group_routes(&fan, experts))?;
        let wmma = x.dtype == Dtype::Bf16 && tuning.affine_wmma();
        let entry = if wmma {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_wmma<::pie::i32({bits}), ::pie::i32({group}), \
                 ::pie::i32({})>",
                ordered.routes
            )
        } else {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_grouped<{t}, ::pie::i32({bits}), \
                 ::pie::i32({group})>"
            )
        };
        let mut args = vec![
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
        ];
        if wmma {
            args.push(ArgValue::Ptr(ordered.work));
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
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ]);
        return ctx.fire(
            op,
            Fire::at(kernel("linear/quant.hip", &entry)).apply(if wmma {
                group_launch(ordered.work_cap, ordered.routes, y.width)
            } else {
                Launch::grid(
                    [experts, y.width.div_ceil(GROUP_TILE_N), 1],
                    [GROUP_BLOCK, 1, 1],
                )
            }),
            &args,
        );
    }
    let (launch, rows) = stream_launch(
        op,
        fan.route_count,
        y.width,
        x.width,
        StreamCodec {
            codes: UNIT,
            unroll: if bits == 8 { 2 } else { 4 },
            xsum: true,
        },
    )?;
    ctx.fire(
        op,
        Fire::at(kernel("linear/quant.hip", &format!(
                "::pie::linear::moe_matmul_select_mlxu4<{t}, ::pie::i32({bits}), ::pie::i32({group})>"
            )))
        .apply(launch),
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
            rows.arg(),
            ctx.stage(),
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ],
    )
}
