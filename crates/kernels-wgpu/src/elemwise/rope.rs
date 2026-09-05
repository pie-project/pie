#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;
use dtype::Dtype;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yarn {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position: u32,
}

const FILE: &str = "rope/neox.wgsl";

const GROUP: u32 = 64;

const UNSCALED: f32 = 1.0;

fn pairs(lanes: [u32; 3]) -> [u32; 3] {
    [lanes[0] / 2, lanes[1], lanes[2]]
}

fn rope_grid(
    op: &'static str,
    rotary: u32,
    width: u32,
    head_dim: u32,
    rows: u32,
) -> Result<[u32; 3], Error> {
    nonzero(op, "the rotated width", rotary)?;
    nonzero(op, "the head width this rotation states", head_dim)?;
    nonzero(op, "rows", rows)?;
    if !rotary.is_multiple_of(4) || !head_dim.is_multiple_of(4) {
        return Err(refuse(
            op,
            format!(
                "the rotated width {rotary} over a {head_dim}-wide head is not a whole \
                 number of bf16 word pairs (both must divide by four)"
            ),
        ));
    }
    if rotary > head_dim {
        return Err(refuse(
            op,
            format!("the rotated width {rotary} is wider than the {head_dim}-wide head"),
        ));
    }
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok([rotary / 2, width / head_dim, rows])
}

fn positions_stream(op: &'static str, positions: Tensor, x: Tensor) {
    debug_assert_eq!(
        positions.dtype,
        Dtype::I32,
        "`{op}` reads an i32 position stream"
    );
    debug_assert_eq!(
        positions.rows, x.rows,
        "the position stream is one entry per rotated row"
    );
}

fn rotate(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    positions: Tensor,
    scale: f32,
    base: f32,
    head_dim: u32,
    rotary: u32,
    proportional: bool,
) -> Result<(), Error> {
    let entry = if proportional {
        dtype_dispatch!(op, x.dtype, { Bf16 => "neox_prop_mb_bf16" })
    } else {
        dtype_dispatch!(op, x.dtype, { Bf16 => "neox_mb_bf16" })
    };
    positions_stream(op, positions, x);
    let lanes = rope_grid(op, rotary, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(pairs(lanes), [GROUP, 1, 1])),
        &[
            x.arg_mut(),
            positions.arg(),
            scale.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
            stated(op, lanes[0])?.arg(),
            stated(op, lanes[1])?.arg(),
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
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_full";
    if interleaved {
        return Err(refuse(
            OP,
            "interleaved pairs are stated, and every neox arm rotates halves",
        ));
    }
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    let base = theta.log2();
    rotate(
        ctx, OP, q, positions, UNSCALED, base, head_dim, head_dim, false,
    )?;
    rotate(
        ctx, OP, k, positions, UNSCALED, base, head_dim, head_dim, false,
    )
}

pub fn partial(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_partial";
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    let base = theta.log2();
    rotate(
        ctx, OP, q, positions, UNSCALED, base, head_dim, rotary_dim, true,
    )?;
    rotate(
        ctx, OP, k, positions, UNSCALED, base, head_dim, rotary_dim, true,
    )
}

pub fn partial_q(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    rotate(
        ctx,
        "elementwise.rope_partial_q",
        q,
        positions,
        UNSCALED,
        theta.log2(),
        head_dim,
        rotary_dim,
        true,
    )
}

pub fn partial_last(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_partial_last";
    if rotary_dim > head_dim {
        return Err(refuse(
            OP,
            format!(
                "the rotated tail is {rotary_dim} wide, wider than the {head_dim}-wide head \
                 it sits at the end of"
            ),
        ));
    }
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "neox_last_mb_bf16" });
    positions_stream(OP, positions, q);
    let lanes = rope_grid(OP, rotary_dim, q.width, head_dim, q.rows)?;
    let (factor, low, high) = match yarn {
        Some(y) => {
            let max_position = nonzero(
                OP,
                "the position span this layer's YaRN ramp states",
                y.original_max_position,
            )?;
            let rotated = stated(OP, rotary_dim)?;
            let (low, high) = ramp_bounds(
                rotated,
                theta,
                y.beta_fast,
                y.beta_slow,
                stated(OP, max_position)?,
            );
            (y.factor, low, high)
        }
        None => (1.0, 0.0, 0.0),
    };
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(pairs(lanes), [GROUP, 1, 1])),
        &[
            q.arg_mut(),
            positions.arg(),
            theta.log2().arg(),
            stated(OP, head_dim)?.arg(),
            i32::from(interleaved).arg(),
            (if inverse { -1.0f32 } else { 1.0f32 }).arg(),
            factor.arg(),
            low.arg(),
            high.arg(),
            stated(OP, lanes[0])?.arg(),
            stated(OP, lanes[1])?.arg(),
        ],
    )
}

#[allow(clippy::cast_precision_loss)]
fn ramp_bounds(
    head_dim: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    const TWO_PI: f32 = core::f32::consts::TAU;
    let ln_theta = theta.ln();
    let corr_dim = |rot: f32| -> f32 {
        head_dim as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let low_dim = corr_dim(beta_fast).floor().max(0.0);
    let high_dim = corr_dim(beta_slow)
        .ceil()
        .min((head_dim / 2) as f32 - 1.0)
        .max(low_dim);
    (low_dim, high_dim)
}

fn rotate_ramped(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    positions: Tensor,
    base: f32,
    head_dim: u32,
    factor: f32,
    low_dim: f32,
    high_dim: f32,
    mscale: f32,
    interleaved: bool,
) -> Result<(), Error> {
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "neox_yarn_mb_bf16" });
    positions_stream(op, positions, x);
    let lanes = rope_grid(op, head_dim, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at("rope/yarn.wgsl", entry).apply(Grid::of(pairs(lanes), [GROUP, 1, 1])),
        &[
            x.arg_mut(),
            positions.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
            factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            mscale.arg(),
            i32::from(interleaved).arg(),
            stated(op, lanes[0])?.arg(),
            stated(op, lanes[1])?.arg(),
        ],
    )
}

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
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_yarn";
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    let max_position = stated(
        OP,
        nonzero(
            OP,
            "the position span this checkpoint's YaRN block states",
            original_max_position,
        )?,
    )?;
    let width = stated(
        OP,
        nonzero(OP, "the head width this rotation states", head_dim)?,
    )?;
    let (low_dim, high_dim) = ramp_bounds(width, theta, beta_fast, beta_slow, max_position);
    let base = theta.log2();
    for x in [q, k] {
        rotate_ramped(
            ctx,
            OP,
            x,
            positions,
            base,
            head_dim,
            factor,
            low_dim,
            high_dim,
            attention_factor,
            interleaved,
        )?;
    }
    Ok(())
}
