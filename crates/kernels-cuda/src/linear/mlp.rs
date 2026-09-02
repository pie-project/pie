//! `Mlp`: gated activations over a packed `[gate | up]` row (and one
//! two-tensor form). One entry per IR variant; every packed form fires the
//! chunked unit, gridded one row per block-row.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/glu.cuh";

const BLOCK: u32 = 256;

/// Rows on their own grid axis, the width chunked across blocks.
fn elementwise_rows(op: &'static str, rows: u32, width: u32) -> Result<Launch, Error> {
    nonzero(op, "rows", rows)?;
    nonzero(op, "the activation's width", width)?;
    Ok(Launch::grid(
        [rows, width.div_ceil(BLOCK), 1],
        [BLOCK, 1, 1],
    ))
}

/// The geometry every packed `[gate | up]` entry shares, checked once.
fn packed_halves(
    op: &'static str,
    packed: Tensor,
    intermediate: u32,
    fan: u32,
    y: &Tensor,
) -> Result<(Launch, i32), Error> {
    // `fan` must divide the packed rectangle's rows (fan rows per token),
    // else the row axis isn't the one the caller thinks.
    if fan == 0 || y.rows % fan != 0 {
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
            // Fan: seat counts token rows, this rectangle's are `fan` per token.
            stated(OP, fan)?.arg(),
            // Staged-geometry seat: region's live-rows word, or ABSENT.
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
            // Fan: seat counts token rows, this rectangle's are `fan` per token.
            stated(OP, fan)?.arg(),
            // Staged-geometry seat: region's live-rows word, or ABSENT.
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
            // Fan: seat counts token rows, this rectangle's are `fan` per token.
            stated(OP, fan)?.arg(),
            // Staged-geometry seat: region's live-rows word, or ABSENT.
            ctx.stage(),
        ],
    )
}

/// `mlp_swiglu_clamp` over an unfused pair; refused by name. No 2-bit MLX
/// bank is served on this plane, so `glu.cuh` has no `swiglu_clamp_split`
/// unit for it. The arm exists (dispatch is exhaustive) but must not claim
/// a shape it would compute wrong.
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

pub fn geglu_tanh(ctx: &Ctx, gate: Tensor, up: Tensor, fan: u32, y: &mut Tensor) -> Result<(), Error> {
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
            // Element-form seat's width: launch is flat over `rows * width`.
            stated(OP, y.width)?.arg(),
            stated(OP, fan)?.arg(),
            // Staged-geometry seat: region's live-rows word, or ABSENT.
            ctx.stage(),
        ],
    )
}

/// Ungated GELU: `y = gelu_tanh(x)`, no `up` half.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for an
/// empty rectangle or an extent past a 32-bit launch.
pub fn gelu_tanh(ctx: &Ctx, x: Tensor, fan: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_gelu_tanh";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
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
            // Element-form seat's width: launch is flat over `rows * width`.
            stated(OP, y.width)?.arg(),
            stated(OP, fan)?.arg(),
            // Staged-geometry seat: region's live-rows word, or ABSENT.
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
            // Fan: seat counts token rows, this rectangle's are `fan` per token.
            stated(OP, fan)?.arg(),
            // Staged-geometry seat: region's live-rows word, or ABSENT.
            ctx.stage(),
        ],
    )
}

/// `up_cap: None` means uncapped; the kernel reads 0 as "no cap".
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
            // Fan: seat counts token rows, this rectangle's are `fan` per token.
            stated(OP, fan)?.arg(),
            // Staged-geometry seat: region's live-rows word, or ABSENT.
            ctx.stage(),
        ],
    )
}
