use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const GLU: &str = "linear/glu.cu";

const FILE: &str = "linear/glu.cu";

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
    // Eight columns a thread where the halves and both planes take whole
    // 16-byte runs: the same values, element for element.
    let (entry, launch) = if intermediate.is_multiple_of(8)
        && packed.ptr.is_multiple_of(16)
        && y.ptr.is_multiple_of(16)
    {
        (
            "mlp_swiglu_vec8",
            elementwise_rows(OP, y.rows, y.width / 8)?.programmatic(ctx.tuning().pdl),
        )
    } else {
        ("mlp_swiglu", launch)
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::{entry}<{t}>"))).apply(launch),
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
            GLU,
            symbol(&format!("::pie::linear::mlp_swiglu_clamp<{t}>")),
        )
        .apply(launch.programmatic(ctx.tuning().pdl)),
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
            GLU,
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

/// The clamped SwiGLU over a gate and an up plane held apart (`up`), or
/// over packed rows; with `weights`, each routed row times its route's f32
/// weight before it rounds.
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
    if let Some(w) = weights {
        debug_assert_eq!(w.dtype, dtype::Dtype::F32, "`{op}` reads f32 route weights");
        if u64::from(w.rows) * u64::from(w.width) != u64::from(y.rows) {
            return Err(refuse(
                op,
                format!(
                    "{} x {} route weights for {} routed rows",
                    w.rows, w.width, y.rows
                ),
            ));
        }
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
            GLU,
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
    fan: u32,
    limit: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_split";
    debug_assert_eq!(
        (gate.rows, gate.width),
        (up.rows, up.width),
        "`{OP}`'s gate and up planes are one rectangle"
    );
    swiglu_clamp_lanes(ctx, OP, gate, Some(up), None, fan, limit, y)
}

/// [`swiglu_clamp_split`] of routed rows, each times its route's f32
/// weight before it rounds.
pub fn swiglu_clamp_split_routed(
    ctx: &Ctx,
    gate: Tensor,
    up: Tensor,
    weights: Tensor,
    fan: u32,
    limit: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_split_routed";
    debug_assert_eq!(
        (gate.rows, gate.width),
        (up.rows, up.width),
        "`{OP}`'s gate and up planes are one rectangle"
    );
    swiglu_clamp_lanes(ctx, OP, gate, Some(up), Some(weights), fan, limit, y)
}

/// Whether [`swiglu_clamp_routed_e4m3`] lands rows `y` (at a valid launch):
/// bf16, whole 128-column blocks.
#[must_use]
pub fn routed_e4m3_spans(y: &Tensor) -> bool {
    y.dtype == dtype::Dtype::Bf16 && y.width.is_multiple_of(128)
}

/// [`swiglu_clamp_routed`] landing its bf16 rows read back through e4m3 a
/// 128 columns (`act_quant`'s E4M3 at `gran` 128, read back): what a W4A8
/// select would read back from the plain rows, and what it reads back from
/// these again. `Ok(false)` where it does not apply (not bf16, or rows not
/// whole 128-column blocks); the caller lands the plain rows then.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_clamp_routed_e4m3(
    ctx: &Ctx,
    packed: Tensor,
    intermediate: u32,
    weights: Tensor,
    fan: u32,
    limit: f32,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_routed";
    if !routed_e4m3_spans(y)
        || packed.dtype != dtype::Dtype::Bf16
        || weights.dtype != dtype::Dtype::F32
        || packed.width != 2 * intermediate
        || y.width != intermediate
        || packed.rows != y.rows
        || u64::from(weights.rows) * u64::from(weights.width) != u64::from(y.rows)
    {
        return Ok(false);
    }
    let lanes = u32::try_from(y.elements() / 8)
        .map_err(|_| refuse(OP, "the rows' eights do not fit a 32-bit launch extent"))?;
    if lanes == 0 {
        return Ok(true);
    }
    ctx.fire(
        OP,
        Fire::at(GLU, "::pie::linear::mlp_swiglu_clamp_routed_e4m3")
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            packed.arg(),
            weights.arg(),
            y.arg(),
            stated(OP, lanes)?.arg(),
            stated(OP, y.width)?.arg(),
            limit.arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(true)
}

/// [`swiglu_clamp_split_routed`] over packed rows: each row's
/// `intermediate` gate columns, then its up columns.
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
        Fire::at(GLU, symbol(&format!("::pie::linear::mlp_geglu_tanh<{t}>")))
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
    act(ctx, "linear.mlp_gelu_tanh", false, x, fan, y)
}

pub fn relu2(ctx: &Ctx, x: Tensor, fan: u32, y: &mut Tensor) -> Result<(), Error> {
    act(ctx, "linear.mlp_relu2", true, x, fan, y)
}

fn act(
    ctx: &Ctx,
    op: &'static str,
    relu2: bool,
    x: Tensor,
    fan: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(
        op,
        x.dtype,
        { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" }
    );
    let n = y.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the activation's element count", lanes)?;
    ctx.fire(
        op,
        Fire::at(
            GLU,
            symbol(&format!("::pie::linear::mlp_act<{t}, {relu2}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, y.width)?.arg(),
            stated(op, fan)?.arg(),
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
    // Eight columns a thread where the halves and both planes take whole
    // 16-byte runs: the same values, element for element.
    let (entry, launch) = if intermediate.is_multiple_of(8)
        && packed.ptr.is_multiple_of(16)
        && y.ptr.is_multiple_of(16)
    {
        (
            "mlp_geglu_tanh_packed_vec8",
            elementwise_rows(OP, y.rows, y.width / 8)?.programmatic(ctx.tuning().pdl),
        )
    } else {
        ("mlp_geglu_tanh_packed", launch)
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::{entry}<{t}>"))).apply(launch),
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
        Fire::at(GLU, symbol(&format!("::pie::linear::mlp_situ<{t}>"))).apply(launch),
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
