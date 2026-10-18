use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/glu.hip";

const BLOCK: u32 = 256;

fn elementwise_rows(op: &'static str, rows: u32, width: u32) -> Result<Launch, Error> {
    nonzero(op, "rows", rows)?;
    nonzero(op, "the activation's width", width)?;
    Ok(Launch::grid(
        [rows, width.div_ceil(BLOCK), 1],
        [BLOCK, 1, 1],
    ))
}

fn packed_halves(
    op: &'static str,
    packed: Tensor,
    intermediate: u32,
    fan: u32,
    y: &Tensor,
) -> Result<(Launch, i32), Error> {
    if fan == 0 || !y.rows.is_multiple_of(fan) {
        return Err(refuse(
            op,
            format!(
                "a fan-out of {fan} does not divide the {}-row plane it is stated for, so \
                 the staged seat cannot be scaled onto this rectangle's row axis",
                y.rows
            ),
        ));
    }
    debug_assert_eq!(
        packed.width,
        intermediate.saturating_mul(2),
        "the packed `[gate | up]` row is twice the intermediate width it states"
    );
    debug_assert_eq!(
        y.width, intermediate,
        "the activation's row is the intermediate width it states"
    );
    debug_assert_eq!(
        y.rows, packed.rows,
        "the activation lands one row per packed row"
    );
    Ok((elementwise_rows(op, y.rows, y.width)?, stated(op, y.width)?))
}

pub fn swiglu(
    ctx: &Ctx,
    packed: Tensor,
    intermediate: u32,
    fan: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu";
    let t = dtype_dispatch!(OP, packed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, width) = packed_halves(OP, packed, intermediate, fan, y)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::mlp_swiglu<{t}>"))).apply(launch),
        &[
            packed.arg(),
            y.arg(),
            width.arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn swiglu_clamp(
    ctx: &Ctx,
    packed: Tensor,
    intermediate: u32,
    fan: u32,
    limit: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp";
    let t = dtype_dispatch!(OP, packed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, width) = packed_halves(OP, packed, intermediate, fan, y)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::mlp_swiglu_clamp<{t}>")),
        )
        .apply(launch),
        &[
            packed.arg(),
            y.arg(),
            width.arg(),
            limit.arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn swiglu_clamp_alpha(
    ctx: &Ctx,
    packed: Tensor,
    intermediate: u32,
    fan: u32,
    limit: f32,
    alpha: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_alpha";
    let t = dtype_dispatch!(OP, packed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, width) = packed_halves(OP, packed, intermediate, fan, y)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::mlp_swiglu_clamp_alpha<{t}>")),
        )
        .apply(launch),
        &[
            packed.arg(),
            y.arg(),
            width.arg(),
            limit.arg(),
            alpha.arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )
}

/// The clamped swiglu over a gate and an up plane held apart, or over
/// packed rows (`up` none), each row times its route's f32 weight where
/// `weights` is given.
#[allow(clippy::too_many_arguments)]
fn swiglu_clamp_lanes(
    ctx: &Ctx,
    op: &'static str,
    gate: Tensor,
    up: Option<Tensor>,
    weights: Option<Tensor>,
    fan: u32,
    limit: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, gate.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if let Some(w) = weights
        && (w.dtype != Dtype::F32 || w.elements() != u64::from(y.rows))
    {
        return Err(refuse(
            op,
            format!(
                "the route weights are {:?} [{}, {}] and the {} routed rows want one f32 each",
                w.dtype, w.rows, w.width, y.rows
            ),
        ));
    }
    let n = y.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the activation's element count", lanes)?;
    let (packed, routed) = (up.is_none(), weights.is_some());
    let absent = crate::jit::ArgValue::ABSENT;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::mlp_swiglu_clamp_lanes<{t}, {packed}, {routed}>"
            )),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            gate.arg(),
            up.map_or(absent, |u| u.arg()),
            weights.map_or(absent, |w| w.arg()),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, y.width)?.arg(),
            limit.arg(),
            stated(op, fan)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn swiglu_clamp_split(
    ctx: &Ctx,
    gate: Tensor,
    up: Tensor,
    limit: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    swiglu_clamp_lanes(
        ctx,
        "linear.mlp_swiglu_clamp_split",
        gate,
        Some(up),
        None,
        1,
        limit,
        y,
    )
}

/// [`swiglu_clamp_split`] of routed rows, each row times its route's
/// weight before it rounds: `weights` is f32, one a routed row.
pub fn swiglu_clamp_split_routed(
    ctx: &Ctx,
    gate: Tensor,
    up: Tensor,
    weights: Tensor,
    fan: u32,
    limit: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    swiglu_clamp_lanes(
        ctx,
        "linear.mlp_swiglu_clamp_split_routed",
        gate,
        Some(up),
        Some(weights),
        fan,
        limit,
        y,
    )
}

/// [`swiglu_clamp`] of routed rows (each row's `intermediate` gate columns,
/// then as many up columns), each row times its route's weight.
pub fn swiglu_clamp_routed(
    ctx: &Ctx,
    packed: Tensor,
    intermediate: u32,
    weights: Tensor,
    fan: u32,
    limit: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_routed";
    if packed.width != 2 * intermediate || y.width != intermediate || packed.rows != y.rows {
        return Err(refuse(
            OP,
            format!(
                "{} x {} packed rows are not {} rows of {intermediate} gate and up columns into {} x {}",
                packed.rows, packed.width, y.rows, y.rows, y.width
            ),
        ));
    }
    swiglu_clamp_lanes(ctx, OP, packed, None, Some(weights), fan, limit, y)
}

pub fn geglu_tanh(
    ctx: &Ctx,
    gate: Tensor,
    up: Tensor,
    fan: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_geglu_tanh";
    let t = dtype_dispatch!(OP, gate.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let n = y.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the activation's element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::mlp_geglu_tanh<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            gate.arg(),
            up.arg(),
            y.arg(),
            stated(OP, lanes)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn gelu_tanh(ctx: &Ctx, x: Tensor, fan: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_gelu_tanh";
    let t = dtype_dispatch!(
        OP,
        x.dtype,
        { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" }
    );
    let n = y.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the activation's element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::mlp_gelu_tanh<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            y.arg(),
            stated(OP, lanes)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn geglu_tanh_packed(
    ctx: &Ctx,
    packed: Tensor,
    intermediate: u32,
    fan: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_geglu_tanh_packed";
    let t = dtype_dispatch!(OP, packed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, width) = packed_halves(OP, packed, intermediate, fan, y)?;
    // Eight elements a thread in 16-byte packs where the halves allow it.
    let packs =
        intermediate.is_multiple_of(8) && packed.ptr.is_multiple_of(16) && y.ptr.is_multiple_of(16);
    let (entry, launch) = if packs {
        (
            format!("::pie::linear::mlp_geglu_tanh_packed_vec8<{t}>"),
            Launch::grid(
                [y.rows, (intermediate / 8).div_ceil(BLOCK), 1],
                [BLOCK, 1, 1],
            ),
        )
    } else {
        (format!("::pie::linear::mlp_geglu_tanh_packed<{t}>"), launch)
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&entry)).apply(launch),
        &[
            packed.arg(),
            y.arg(),
            width.arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn situ(
    ctx: &Ctx,
    packed: Tensor,
    intermediate: u32,
    fan: u32,
    beta: f32,
    up_cap: Option<f32>,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_situ";
    if beta == 0.0 {
        return Err(refuse(OP, "beta is zero, and the gate divides by it"));
    }
    let t = dtype_dispatch!(OP, packed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, width) = packed_halves(OP, packed, intermediate, fan, y)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::mlp_situ<{t}>"))).apply(launch),
        &[
            packed.arg(),
            y.arg(),
            width.arg(),
            beta.arg(),
            up_cap.unwrap_or(0.0).arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )
}
