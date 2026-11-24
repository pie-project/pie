//! The MoE MLPs and the routed rows' combines: the decode and batch MLPs,
//! the weighted and bias sums, the shared expert's gate.

#[allow(clippy::wildcard_imports)]
use super::*;

/// Each token's routed rows (already weighted) summed in f32, plus its
/// `shared` row, rounded once.
pub fn sum_add(ctx: &Ctx, routed: Tensor, shared: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.moe_combine";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "the token rows this fold lands on", y.rows)?;
    if !routed.rows.is_multiple_of(y.rows) || routed.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "the routed rectangle [{}, {}] does not fold into [{}, {}] token rows",
                routed.rows, routed.width, y.rows, y.width
            ),
        ));
    }
    if shared.rows != y.rows || shared.width != y.width || shared.dtype != routed.dtype {
        return Err(refuse(
            OP,
            "the shared row is not one token row of the routed dtype",
        ));
    }
    let top_k = routed.rows / y.rows;
    let vec = if y.width.is_multiple_of(8)
        && [y.ptr, routed.ptr, shared.ptr]
            .iter()
            .all(|p| p.is_multiple_of(16))
    {
        8
    } else {
        1
    };
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::moe_sum_add<{t}, {vec}>"),
        ))
        .apply(elementwise_rows(OP, y.rows, y.width / vec)?),
        &[
            y.arg(),
            routed.arg(),
            shared.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, y.width)?.arg(),
            ctx.stage(),
        ],
    )
}

/// A routed swiglu MLP (`linear.moe_mlp_swiglu`): `packed` = x through the
/// gate_up bank by route, `h` = its swiglu (clamped at `limit` when given),
/// `routed` = h through the down bank, y = their weighted fold. `packed`,
/// `h` and `routed` are the planes the four steps land; an expert-ordered
/// pipeline may leave them untouched.
#[derive(Clone, Copy, Debug)]
pub struct RoutedMlp {
    pub x: Tensor,
    pub routes: Tensor,
    pub weights: Tensor,
    pub gate_up: Tensor,
    pub gate_up_experts: ExpertTable,
    pub down: Tensor,
    pub down_experts: ExpertTable,
    pub intermediate: u32,
    pub limit: Option<f32>,
    /// The swiglu's fan-out over `h`'s rows (the engine's plane fan).
    pub fan: u32,
    pub packed: Tensor,
    pub h: Tensor,
    pub routed: Tensor,
}

/// A shared expert's gated add after the routed fold: `y = routed +
/// shared · sigmoid(gate)`, `gate` one logit a row. Given `banks`, `shared`
/// is not landed yet: the routed legs land it, the shared expert's swiglu
/// MLP over the routed one's x as items of their own. Given `gate_w` (one
/// row), neither is `gate`: the fold takes x · gate_w itself.
#[derive(Clone, Copy, Debug)]
pub struct SharedGate {
    pub shared: Tensor,
    pub gate: Tensor,
    pub y: Tensor,
    pub banks: Option<SharedBanks>,
    pub gate_w: Option<Tensor>,
}

/// A shared expert's packed `[gate | up]` and down banks.
#[derive(Clone, Copy, Debug)]
pub struct SharedBanks {
    pub gate_up: Tensor,
    pub down: Tensor,
}

/// The fan and expert count of a fire [`decode_mlp`] covers, if it does.
pub(super) fn decode_fan(
    ctx: &Ctx,
    p: &RoutedMlp,
    shared: Option<SharedGate>,
    hidden: u32,
) -> Result<Option<(Selected, u32)>, Error> {
    const OP: &str = "linear.moe_decode_mlp";
    let (x, i) = (p.x, p.intermediate);
    let fan = selected(OP, x, p.routes, &p.packed)?;
    let (Some(count), Some(_)) = (
        dense_experts(x, p.gate_up, &p.packed, p.gate_up_experts),
        dense_experts(p.h, p.down, &p.routed, p.down_experts),
    ) else {
        return Ok(None);
    };
    let covered = fan.by_token
        && ctx.tuning().moe_batch()
        && p.limit.is_none()
        && p.packed.width == 2 * i
        && shared.is_none_or(|s| {
            s.shared.dtype == x.dtype
                && s.gate.dtype == x.dtype
                && s.shared.width == hidden
                && s.y.width == hidden
                && s.gate.width > 0
                && s.banks.is_none_or(|b| {
                    b.gate_up.dtype == x.dtype
                        && b.down.dtype == x.dtype
                        && b.gate_up.rows == 2 * i
                        && b.gate_up.width == x.width
                        && b.down.rows == hidden
                        && b.down.width == i
                        && s.shared.rows == x.rows
                })
                && s.gate_w.is_none_or(|w| {
                    w.dtype == x.dtype
                        && w.rows == 1
                        && w.width == x.width
                        && x.width.is_multiple_of(8)
                })
        });
    Ok(covered.then_some((fan, count)))
}

/// Whether [`decode_mlp`] lands what `shared` holds back (its banks, its
/// gate's weight) itself: its batch legs take the fire. A caller that holds
/// them back for it decides here, before the legs.
pub fn absorbs_shared(ctx: &Ctx, p: &RoutedMlp, shared: SharedGate, hidden: u32) -> bool {
    (shared.banks.is_some() || shared.gate_w.is_some())
        && p.x.rows >= ctx.tuning().moe_batch_from()
        && matches!(decode_fan(ctx, p, Some(shared), hidden), Ok(Some((fan, count)))
            if batch_plan(ctx, p, &fan, count).is_some())
}

/// Rows a decode wave takes in [`decode_mlp`].
pub(super) const DECODE_RPW: u32 = 2;
pub(super) const DECODE_BLOCK: u32 = 256;

/// A decode fire's routed MLP with the swiglu in the gate-up launch and the
/// fold (and `shared`'s gated add, landing in its `y`) at the end, over
/// 16-bit banks in device memory (contiguous, or a located table). One token
/// takes two launches: fewer tokens than `kernel.moe.batch_from`, rows of
/// whole 512s. A batch takes the batch legs over one route ordering where
/// both legs would. Whether it ran.
pub fn decode_mlp(
    ctx: &Ctx,
    p: &RoutedMlp,
    shared: Option<SharedGate>,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "linear.moe_decode_mlp";
    let (x, i, hidden) = (p.x, p.intermediate, y.width);
    let t = ctx.tuning();
    let Some((fan, count)) = decode_fan(ctx, p, shared, hidden)? else {
        return Ok(false);
    };
    let top_k = fan.top_k.unsigned_abs();
    if x.rows >= t.moe_batch_from() {
        return batch_mlp(ctx, p, &fan, count, shared, y);
    }
    if shared.is_some_and(|s| s.banks.is_some() || s.gate_w.is_some()) {
        return Ok(false);
    }
    let fits = |k: u32, most: u32| k.is_multiple_of(512) && (1..=most).contains(&(k / 512));
    if !fits(x.width, 8) || !fits(i, 4) || !(1..=16).contains(&top_k) {
        return Ok(false);
    }
    let ty = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows_a_block = 4 * DECODE_RPW;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!(
                "::pie::linear::moe_decode_glu<{ty}, {}, {DECODE_RPW}>",
                x.width / 512
            ),
        ))
        .apply(Launch::grid(
            [fan.route_count, i.div_ceil(rows_a_block), 1],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            p.routes.arg(),
            x.arg(),
            p.gate_up.arg(),
            p.h.arg(),
            fan.top_k.arg(),
            stated(OP, i)?.arg(),
            (2 * i64::from(i) * i64::from(x.width)).arg(),
            ctx.stage(),
            ArgValue::Ptr(p.gate_up_experts.table),
            ArgValue::Ptr(p.gate_up_experts.hits),
        ],
    )?;
    let (out, shared_at, gate_at, gate_stride) = match shared {
        Some(s) => (s.y, s.shared.ptr, s.gate.ptr, stated(OP, s.gate.width)?),
        None => (*y, 0, 0, 0),
    };
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!(
                "::pie::linear::moe_decode_down<{ty}, {}, {top_k}, {DECODE_RPW}>",
                i / 512
            ),
        ))
        .apply(Launch::grid(
            [x.rows, hidden.div_ceil(rows_a_block), 1],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            p.routes.arg(),
            p.weights.arg(),
            p.h.arg(),
            p.down.arg(),
            out.arg(),
            ArgValue::Ptr(shared_at),
            ArgValue::Ptr(gate_at),
            gate_stride.arg(),
            stated(OP, hidden)?.arg(),
            (i64::from(hidden) * i64::from(i)).arg(),
            ctx.stage(),
            ArgValue::Ptr(p.down_experts.table),
            ArgValue::Ptr(p.down_experts.hits),
        ],
    )?;
    Ok(true)
}

/// The items [`batch_mlp`] takes, if it takes the fire.
pub(super) fn batch_plan(ctx: &Ctx, p: &RoutedMlp, fan: &Selected, count: u32) -> Option<u32> {
    let (x, i) = (p.x, p.intermediate);
    let until = ctx.tuning().moe_dense_batch_until();
    let down_fan = Selected {
        by_token: false,
        ..*fan
    };
    let tiled = x.dtype == Dtype::Bf16
        && x.width.is_multiple_of(64)
        && count <= ORDER_EXPERT_CAP
        && (65..=until).contains(&fan.route_count.div_ceil(count));
    let mt = batch_items(ctx, fan, count, 64, until)?;
    if tiled
        || !x.width.is_multiple_of(8)
        || !i.is_multiple_of(8)
        || wide_items(ctx, fan, count, x).is_some()
        || wide_items(ctx, &down_fan, count, p.h).is_some()
    {
        return None;
    }
    Some(mt)
}

/// [`decode_mlp`]'s batch: where `matmul_select` would take the batch leg
/// for both matmuls, one route ordering for both, the swiglu in the gate-up
/// leg's epilogue, and the fold with the gated add.
pub(super) fn batch_mlp(
    ctx: &Ctx,
    p: &RoutedMlp,
    fan: &Selected,
    count: u32,
    shared: Option<SharedGate>,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "linear.moe_decode_mlp";
    let (x, i, hidden) = (p.x, p.intermediate, y.width);
    let Some(mt) = batch_plan(ctx, p, fan, count) else {
        return Ok(false);
    };
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let ordered = route_order(ctx, OP, p.routes, fan, count, mt)?;
    let order = [
        ArgValue::Ptr(ordered.order),
        ArgValue::Ptr(ordered.offsets),
        ArgValue::Ptr(ordered.work),
    ];
    let experts = stated(OP, count)?;
    // The shared expert's items lead both legs, its h in a slab of its own.
    let banks = shared.and_then(|s| s.banks);
    let shared_items = banks.map_or(0, |_| x.rows.div_ceil(mt));
    let shared_h = match banks {
        Some(_) => {
            ctx.scratch_stream(OP, "linear.moe_shared_h", x.rows as usize * i as usize * 2)? as u64
        }
        None => 0,
    };
    let shared_rows = stated(OP, if banks.is_some() { x.rows } else { 0 })?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::moe_matmul_select_dense_batch_swiglu<{t}, ::pie::i32({mt})>"),
        ))
        .apply(Launch::grid(
            [shared_items + ordered.work_cap, i.div_ceil(32), 1],
            [GROUP_BLOCK, 1, 1],
        )),
        &[
            &[x.arg()],
            &order[..],
            &[
                p.gate_up.arg(),
                p.h.arg(),
                fan.top_k.arg(),
                stated(OP, i)?.arg(),
                stated(OP, x.width)?.arg(),
                experts.arg(),
                ArgValue::Ptr(banks.map_or(0, |b| b.gate_up.ptr)),
                ArgValue::Ptr(shared_h),
                shared_rows.arg(),
                ArgValue::Ptr(p.gate_up_experts.table),
                ArgValue::Ptr(p.gate_up_experts.hits),
            ],
        ]
        .concat(),
    )?;
    let nbw = dense_nbw(mt);
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!(
                "::pie::linear::moe_matmul_select_dense_batch<{t}, ::pie::i32({mt}), ::pie::i32({nbw})>"
            )))
        .apply(Launch::grid(
            [shared_items + ordered.work_cap, hidden.div_ceil(64 * nbw), 1],
            [GROUP_BLOCK, 1, 1],
        )),
        &[
            &[p.h.arg()],
            &order[..],
            &[
                p.down.arg(),
                p.routed.arg(),
                1i32.arg(),
                stated(OP, hidden)?.arg(),
                stated(OP, i)?.arg(),
                experts.arg(),
                ArgValue::Ptr(banks.map_or(0, |b| b.down.ptr)),
                ArgValue::Ptr(shared_h),
                ArgValue::Ptr(match (banks, shared) {
                    (Some(_), Some(s)) => s.shared.ptr,
                    _ => 0,
                }),
                shared_rows.arg(),
                ArgValue::Ptr(p.down_experts.table),
                ArgValue::Ptr(p.down_experts.hits),
            ],
        ]
        .concat(),
    )?;
    weighted_fold(ctx, p.routed, p.weights, shared, x, y)?;
    Ok(true)
}

/// The routed MLP: on the grouped GEMM with its routes in expert order
/// throughout where that measured faster, else the four routed steps.
pub fn mlp_swiglu(ctx: &Ctx, p: &RoutedMlp, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.moe_mlp_swiglu";
    let steps = || -> Result<(), Error> {
        let (mut packed, mut h, mut routed) = (p.packed, p.h, p.routed);
        matmul_select(
            ctx,
            p.x,
            p.gate_up,
            p.routes,
            &mut packed,
            p.gate_up_experts,
        )?;
        match p.limit {
            Some(limit) => {
                crate::linear::glu::swiglu_clamp(ctx, packed, p.intermediate, p.fan, limit, &mut h)?
            }
            None => crate::linear::glu::swiglu(ctx, packed, p.intermediate, p.fan, &mut h)?,
        }
        matmul_select(ctx, h, p.down, p.routes, &mut routed, p.down_experts)?;
        Ok(())
    };
    #[cfg(feature = "rocm")]
    if mlp_lt(ctx, OP, p, y)? || ordered::mlp(ctx, OP, p, y)? {
        return Ok(());
    }
    let _ = OP;
    steps()?;
    weighted_sum(ctx, p.routed, p.weights, y)
}

pub fn weighted_sum(
    ctx: &Ctx,
    routed: Tensor,
    weights: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    weighted_fold(ctx, routed, weights, None, routed, y)
}

/// [`weighted_sum`], and given `shared` its gated add, into `shared.y` (the
/// gate x · `shared.gate_w` where given).
pub(super) fn weighted_fold(
    ctx: &Ctx,
    routed: Tensor,
    weights: Tensor,
    shared: Option<SharedGate>,
    x: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_combine";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    nonzero(OP, "the token rows this fold lands on", y.rows)?;
    if !routed.rows.is_multiple_of(y.rows) {
        return Err(refuse(
            OP,
            format!(
                "the routed rectangle's {} rows do not fold into the {} token rows",
                routed.rows, y.rows
            ),
        ));
    }
    let top_k = routed.rows / y.rows;
    debug_assert_eq!(
        routed.width, y.width,
        "the routed row's width, which the fold does not change"
    );
    debug_assert!(
        weights.rows == y.rows && weights.width == top_k,
        "the weight plane is one weight per route"
    );
    let vec = if y.width.is_multiple_of(8) { 8 } else { 1 };
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::moe_weighted_sum<{t}, {vec}>"),
        ))
        .apply(elementwise_rows(OP, y.rows, y.width / vec)?),
        &[
            shared.map_or(*y, |s| s.y).arg(),
            routed.arg(),
            weights.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, y.width)?.arg(),
            ArgValue::Ptr(shared.map_or(0, |s| s.shared.ptr)),
            ArgValue::Ptr(shared.map_or(0, |s| s.gate.ptr)),
            stated(OP, shared.map_or(0, |s| s.gate.width))?.arg(),
            ctx.stage(),
            ArgValue::Ptr(shared.and_then(|s| s.gate_w).map_or(0, |w| w.ptr)),
            x.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

/// `table` places the bias's experts where the weight cache streams it
/// (a one-plane bank); [`ExpertTable::RESIDENT`] reads it contiguous.
pub fn bias_sum(
    ctx: &Ctx,
    x: Tensor,
    bias: Tensor,
    routes: Tensor,
    weights: Tensor,
    y: &mut Tensor,
    table: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_bias_sum";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    debug_assert_eq!(
        bias.dtype, x.dtype,
        "the expert bias rides the activation's dtype"
    );
    debug_assert!(
        x.rows == y.rows && x.width == y.width,
        "the bias lands on the activation's own rectangle"
    );
    debug_assert_eq!(
        bias.width, y.width,
        "an expert's bias row is the width it is added to"
    );
    debug_assert_eq!(
        routes.rows, y.rows,
        "the route plane is one row per token row"
    );
    debug_assert!(
        weights.rows == y.rows && weights.width == routes.width,
        "the weight plane is one weight per route"
    );
    let top_k = nonzero(OP, "the routed fan-out", routes.width)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::linear::moe_bias_sum<{t}>")))
            .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            x.arg(),
            bias.arg(),
            routes.arg(),
            weights.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, y.width)?.arg(),
            ctx.stage(),
            ArgValue::Ptr(table.table),
        ],
    )
}

pub fn sigmoid_gate_add(
    ctx: &Ctx,
    routed: Tensor,
    shared: Tensor,
    gate: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_sigmoid_gate_add";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        routed.rows == y.rows && routed.width == y.width,
        "the routed row is the combined row"
    );
    debug_assert!(
        shared.rows == y.rows && shared.width == y.width,
        "the shared expert's rectangle is the routed one"
    );
    debug_assert_eq!(gate.rows, y.rows, "the gate column is one scalar per row");
    ctx.fire(
        OP,
        Fire::at(kernel(
            "linear/glu.hip",
            &format!("::pie::linear::moe_sigmoid_gate_add<{t}>"),
        ))
        .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            shared.arg(),
            gate.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, nonzero(OP, "the gate row's pitch", gate.width)?)?.arg(),
            ctx.stage(),
        ],
    )
}
