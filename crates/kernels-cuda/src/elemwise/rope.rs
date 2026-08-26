//! `Rope`: rotations in place on the projection they turn. One entry per IR
//! variant; interleaved layouts reach the unit as a stated flag, never as a
//! different loop. The frequency-pair cache and the heads-per-block packing
//! are the only geometry decisions, and they live here.

#![allow(clippy::too_many_arguments)]

use kernels::KernelError;
use model_ir::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/rope.cuh";

pub const ROTATE_BLOCK: u32 = 256;

pub const MAX_CACHED_PAIRS: u32 = 4096;

/// No YaRN: the unit interpolation factor.
const UNSCALED: f32 = 1.0;

#[must_use]
const fn heads_per_block(half: u32) -> u32 {
    if half >= ROTATE_BLOCK {
        1
    } else {
        ROTATE_BLOCK / half
    }
}

/// How many frequency pairs fit the shared-memory cache; 0 means uncached.
#[must_use]
const fn cache_pairs(half: u32) -> u32 {
    if half <= MAX_CACHED_PAIRS { half } else { 0 }
}

#[must_use]
const fn rotate_launch(num_tokens: u32, total_heads: u32, per_block: u32, smem: u32) -> Launch {
    Launch::grid(
        [num_tokens, total_heads.div_ceil(per_block), 1],
        [ROTATE_BLOCK, 1, 1],
    )
    .smem(smem)
}

/// The YaRN interpolation ramp, precomputed host-side as before.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ramp_bounds(
    span: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    const TWO_PI: f32 = core::f32::consts::TAU;
    let ln_theta = theta.ln();
    let corr_dim = |rot: f32| -> f32 {
        span as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let mut low_dim = corr_dim(beta_fast).floor();
    let mut high_dim = corr_dim(beta_slow).ceil();
    if low_dim < 0.0 {
        low_dim = 0.0;
    }
    let max_pair = (span / 2) as f32 - 1.0;
    if high_dim > max_pair {
        high_dim = max_pair;
    }
    if high_dim < low_dim {
        high_dim = low_dim;
    }
    (low_dim, high_dim)
}

fn heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, KernelError> {
    nonzero(op, "the head width this rotation states", head_dim)?;
    if width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}

fn q_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, KernelError> {
    nonzero(op, "the q region's width", width)?;
    heads(op, width, head_dim)
}

fn positions_stream(op: &'static str, positions: Tensor, x: &Tensor) {
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

pub fn full(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Result<(), KernelError> {
    const OP: &str = "elementwise.rope_full";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    positions_stream(OP, positions, q);
    let num_q_heads = q_heads(OP, q.width, head_dim)?;
    let num_kv_heads = heads(OP, k.width, head_dim)?;
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let per_block = heads_per_block(half);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            "::pie::elemwise::rope_full<::pie::false_type::value, false>",
        )
        .apply(rotate_launch(
            q.rows,
            num_q_heads + num_kv_heads,
            per_block,
            pairs * 2 * 4,
        )),
        // The trailing null block is the fused-append variant's optional
        // slots (kv pages, page geometry, mask), absent on the plain rotate.
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            theta.arg(),
            interleaved.arg(),
            stated(OP, pairs)?.arg(),
            stated(OP, per_block)?.arg(),
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            0_i32.arg(),
            0_i32.arg(),
        ],
    )
}

pub fn partial(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), KernelError> {
    const OP: &str = "elementwise.rope_partial";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    positions_stream(OP, positions, q);
    rope_partial(ctx, OP, *q, *k, positions, rotary_dim, head_dim, theta)
}

pub fn partial_q(
    ctx: &Ctx,
    q: &mut Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), KernelError> {
    const OP: &str = "elementwise.rope_partial_q";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    positions_stream(OP, positions, q);
    // k rides as q with a zero width: the unit reads zero kv heads.
    let k = Tensor::new(q.ptr, q.rows, 0, q.dtype);
    rope_partial(ctx, OP, *q, k, positions, rotary_dim, head_dim, theta)
}

/// Partial rope over the last `rotary_dim` lanes of each head.
pub fn partial_last(
    ctx: &Ctx,
    q: &mut Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Result<(), KernelError> {
    const OP: &str = "elementwise.rope_partial_last";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    positions_stream(OP, positions, q);
    if rotary_dim > head_dim {
        return Err(refuse(
            OP,
            format!(
                "the rotated tail is {rotary_dim} wide, wider than the {head_dim}-wide head \
                 it sits at the end of"
            ),
        ));
    }
    let num_q_heads = q_heads(OP, q.width, head_dim)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::rope_partial_last")
            .apply(Launch::per_row(q.rows, ROTATE_BLOCK)),
        &[
            q.arg(),
            q.arg(),
            positions.arg(),
            stated(OP, num_q_heads)?.arg(),
            0_i32.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, rotary_dim)?.arg(),
            theta.arg(),
            false.arg(),
            interleaved.arg(),
            UNSCALED.arg(),
            0.0_f32.arg(),
            0.0_f32.arg(),
        ],
    )
}

pub fn yarn(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
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
    const OP: &str = "elementwise.rope_yarn";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    positions_stream(OP, positions, q);
    let max_position = stated(
        OP,
        nonzero(
            OP,
            "the position span this checkpoint's YaRN block states",
            original_max_position,
        )?,
    )?;
    let num_q_heads = q_heads(OP, q.width, head_dim)?;
    let num_kv_heads = heads(OP, k.width, head_dim)?;
    let (low_dim, high_dim) = ramp_bounds(
        stated(OP, head_dim)?,
        theta,
        beta_fast,
        beta_slow,
        max_position,
    );
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let per_block = heads_per_block(half);
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::rope_yarn").apply(rotate_launch(
            q.rows,
            num_q_heads + num_kv_heads,
            per_block,
            pairs * 8,
        )),
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            theta.arg(),
            factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            attention_factor.arg(),
            interleaved.arg(),
            stated(OP, per_block)?.arg(),
            stated(OP, pairs)?.arg(),
        ],
    )
}

fn rope_partial(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), KernelError> {
    let num_q_heads = q_heads(op, q.width, head_dim)?;
    let num_kv_heads = heads(op, k.width, head_dim)?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::elemwise::rope_partial<::pie::bf16>")
            .apply(Launch::per_row(q.rows, ROTATE_BLOCK)),
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            0_i32.arg(),
            stated(op, num_q_heads)?.arg(),
            stated(op, num_kv_heads)?.arg(),
            stated(op, head_dim)?.arg(),
            stated(op, rotary_dim)?.arg(),
            theta.arg(),
        ],
    )
}
