use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/glu.cuh";

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

pub fn swiglu_clamp_split(
    _ctx: &Ctx,
    gate: Tensor,
    _up: Tensor,
    _limit: f32,
    _y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_split";
    Err(refuse(
        OP,
        format!(
            "the unfused swiglu-clamp pair is the 2-bit MLX expert path's combine, and \
             `{FILE}` instantiates no `swiglu_clamp_split` unit — this plane serves the \
             packed `linear.mlp_swiglu_clamp` row (halves of {} at {:?}) and no MLX \
             affine bank",
            gate.width, gate.dtype,
        ),
    ))
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
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::mlp_geglu_tanh_packed<{t}>")),
        )
        .apply(launch),
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
