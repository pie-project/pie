//! `Rope`: neox rotations, in place on the projection they turn. One entry
//! per IR variant; every arm rotates halves, so the interleaved layouts that
//! reach a shader do so as a stated flag, never as a different loop.

use kernels::KernelError;
use model_ir::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, head_group, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/rope_neox.metal";

const UNSCALED: f32 = 1.0;

/// One thread per rotated pair, per head, per row.
fn rope_grid(
    op: &'static str,
    rotary: u32,
    width: u32,
    head_dim: u32,
    rows: u32,
) -> Result<[u32; 3], KernelError> {
    nonzero(op, "the rotated width", rotary)?;
    nonzero(op, "the head width this rotation states", head_dim)?;
    nonzero(op, "rows", rows)?;
    if rotary % 2 != 0 {
        return Err(refuse(
            op,
            format!("the rotated width {rotary} is not a whole number of pairs"),
        ));
    }
    if width == 0 || width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok([rotary / 2, width / head_dim, rows])
}

fn positions_stream(op: &'static str, positions: Tensor, x: Tensor) {
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{op}` reads an i32 position stream");
    debug_assert_eq!(
        positions.rows, x.rows,
        "the position stream is one entry per rotated row"
    );
}

/// The geometric arm: frequencies straight off `base`, full-width rotation
/// stated by its own extent.
#[allow(clippy::too_many_arguments)]
fn rotate_geometric(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    positions: Tensor,
    scale: f32,
    base: f32,
    head_dim: u32,
    rotary: u32,
) -> Result<(), KernelError> {
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "neox_mb_bfloat16" });
    positions_stream(op, positions, x);
    let lanes = rope_grid(op, rotary, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            x.arg_mut(),
            positions.arg(),
            scale.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
        ],
    )
}

/// The proportional arm: the rotated span's frequencies spread over the head
/// width, for partial rotations.
#[allow(clippy::too_many_arguments)]
fn rotate_proportional(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    positions: Tensor,
    scale: f32,
    base: f32,
    head_dim: u32,
    rotary: u32,
) -> Result<(), KernelError> {
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "neox_prop_mb_bfloat16" });
    positions_stream(op, positions, x);
    let lanes = rope_grid(op, rotary, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            x.arg_mut(),
            positions.arg(),
            scale.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
        ],
    )
}

/// The tail arm: the rotation sits over the last `rotary` lanes of each head.
#[allow(clippy::too_many_arguments)]
fn rotate_tail(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    positions: Tensor,
    base: f32,
    head_dim: u32,
    rotary: u32,
    interleaved: bool,
) -> Result<(), KernelError> {
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "neox_last_mb_bfloat16" });
    positions_stream(op, positions, x);
    let lanes = rope_grid(op, rotary, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            x.arg_mut(),
            positions.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
            i32::from(interleaved).arg(),
        ],
    )
}

/// The YaRN interpolation ramp, precomputed host-side as before.
#[derive(Clone, Copy)]
struct Ramp {
    factor: f32,

    low_dim: f32,

    high_dim: f32,

    mscale: f32,
}

#[allow(clippy::cast_precision_loss)]
fn ramp_bounds(head_dim: i32, theta: f32, beta_fast: f32, beta_slow: f32, span: i32) -> (f32, f32) {
    let ln_theta = theta.ln();
    let corr_dim = |rot: f32| -> f32 {
        head_dim as f32 * (span as f32 / (rot * core::f32::consts::TAU)).ln() / (2.0 * ln_theta)
    };
    let low_dim = corr_dim(beta_fast).floor().max(0.0);
    let high_dim = corr_dim(beta_slow)
        .ceil()
        .min((head_dim / 2) as f32 - 1.0)
        .max(low_dim);
    (low_dim, high_dim)
}

/// The ramped arm: YaRN's interpolated frequencies over the whole head.
#[allow(clippy::too_many_arguments)]
fn rotate_ramped(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    positions: Tensor,
    base: f32,
    head_dim: u32,
    ramp: Ramp,
    interleaved: bool,
) -> Result<(), KernelError> {
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "neox_yarn_mb_bfloat16" });
    positions_stream(op, positions, x);
    let lanes = rope_grid(op, head_dim, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            x.arg_mut(),
            positions.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
            ramp.factor.arg(),
            ramp.low_dim.arg(),
            ramp.high_dim.arg(),
            ramp.mscale.arg(),
            i32::from(interleaved).arg(),
        ],
    )
}

pub fn full(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Result<(), KernelError> {
    const OP: &str = "rope.full";
    if interleaved {
        return Err(refuse(
            OP,
            "interleaved pairs are stated, and every neox arm rotates halves",
        ));
    }
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    let base = theta.log2();
    rotate_geometric(ctx, OP, q, positions, UNSCALED, base, head_dim, head_dim)?;
    rotate_geometric(ctx, OP, k, positions, UNSCALED, base, head_dim, head_dim)
}

pub fn partial(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), KernelError> {
    const OP: &str = "rope.partial";
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    let base = theta.log2();
    rotate_proportional(ctx, OP, q, positions, UNSCALED, base, head_dim, rotary_dim)?;
    rotate_proportional(ctx, OP, k, positions, UNSCALED, base, head_dim, rotary_dim)
}

pub fn partial_q(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), KernelError> {
    rotate_proportional(
        ctx,
        "rope.partial_q",
        q,
        positions,
        UNSCALED,
        theta.log2(),
        head_dim,
        rotary_dim,
    )
}

/// Partial rope over the last `rotary_dim` lanes of each head.
pub fn partial_last(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Result<(), KernelError> {
    const OP: &str = "rope.partial_last";
    if rotary_dim > head_dim {
        return Err(refuse(
            OP,
            format!(
                "the rotated tail is {rotary_dim} wide, wider than the {head_dim}-wide head \
                 it sits at the end of"
            ),
        ));
    }
    rotate_tail(
        ctx,
        OP,
        q,
        positions,
        theta.log2(),
        head_dim,
        rotary_dim,
        interleaved,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn yarn(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    head_dim: u32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: u32,
    interleaved: bool,
) -> Result<(), KernelError> {
    const OP: &str = "rope.yarn";
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    let span = stated(
        OP,
        nonzero(
            OP,
            "the position span this checkpoint's YaRN block states",
            original_max_position,
        )?,
    )?;
    let width = stated(OP, nonzero(OP, "the head width this rotation states", head_dim)?)?;
    let (low_dim, high_dim) = ramp_bounds(width, theta, beta_fast, beta_slow, span);
    let ramp = Ramp {
        factor,
        low_dim,
        high_dim,
        mscale: attention_factor,
    };
    let base = theta.log2();
    rotate_ramped(ctx, OP, q, positions, base, head_dim, ramp, interleaved)?;
    rotate_ramped(ctx, OP, k, positions, base, head_dim, ramp, interleaved)
}
