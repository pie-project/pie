#![allow(clippy::too_many_arguments)]

use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;
use dtype::Dtype;

pub const ROTATE_BLOCK: u32 = 256;

pub const MAX_CACHED_PAIRS: u32 = 4096;

const UNSCALED: f32 = 1.0;

#[must_use]
const fn heads_per_block(half: u32) -> u32 {
    if half >= ROTATE_BLOCK {
        1
    } else {
        ROTATE_BLOCK / half
    }
}

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

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ramp_bounds(
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
    let mut low_dim = corr_dim(beta_fast).floor();
    let mut high_dim = corr_dim(beta_slow).ceil();
    if low_dim < 0.0 {
        low_dim = 0.0;
    }
    let max_pair = (head_dim / 2) as f32 - 1.0;
    if high_dim > max_pair {
        high_dim = max_pair;
    }
    if high_dim < low_dim {
        high_dim = low_dim;
    }
    (low_dim, high_dim)
}

fn q_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the q region's width", width)?;
    heads(op, width, head_dim)
}

pub fn full(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope";
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
            ROPE_UNIT,
            "::pie::elemwise::rope_full<::pie::false_type::value, false>",
        )
        .apply(rotate_launch(
            q.rows,
            num_q_heads + num_kv_heads,
            per_block,
            pairs * 2 * 4,
        )),
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
            ctx.stage(),
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
    span: u32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    positions_stream(OP, positions, q);
    rope_partial(
        ctx, OP, *q, *k, positions, rotary_dim, head_dim, theta, span,
    )
}

pub fn partial_q(
    ctx: &Ctx,
    q: &mut Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: u32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    positions_stream(OP, positions, q);
    let k = Tensor::new(q.ptr, q.rows, 0, q.dtype);
    rope_partial(ctx, OP, *q, k, positions, rotary_dim, head_dim, theta, span)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yarn {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn partial_last(
    ctx: &Ctx,
    q: &mut Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope";
    let (factor, low_dim, high_dim) = match yarn {
        Some(y) => {
            let max_position = nonzero(
                OP,
                "the position span this layer's YaRN ramp states",
                y.original_max_position,
            )?;
            let (low, high) = ramp_bounds(
                stated(OP, rotary_dim)?,
                theta,
                y.beta_fast,
                y.beta_slow,
                stated(OP, max_position)?,
            );
            (y.factor, low, high)
        }
        None => (UNSCALED, 0.0, 0.0),
    };
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
    if rotary_dim / 2 > MAX_CACHED_PAIRS {
        return Err(refuse(
            OP,
            format!("caches at most {MAX_CACHED_PAIRS} angles a row, and {rotary_dim} rotate"),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(ROPE_UNIT, "::pie::elemwise::rope_partial_last")
            .apply(Launch::per_row(q.rows, ROTATE_BLOCK).smem(8 * (rotary_dim / 2))),
        &[
            q.arg(),
            q.arg(),
            positions.arg(),
            stated(OP, num_q_heads)?.arg(),
            0_i32.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, rotary_dim)?.arg(),
            theta.arg(),
            inverse.arg(),
            interleaved.arg(),
            factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            ctx.stage(),
        ],
    )
}

/// A YaRN rotation: `head_dim`-wide heads at `theta`, the frequencies ramped
/// by `yarn`, cos and sin scaled by `attention_factor`, pairs interleaved or
/// split halves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnRope {
    pub head_dim: u32,
    pub theta: f32,
    pub yarn: Yarn,
    pub attention_factor: f32,
    pub interleaved: bool,
}

pub fn yarn(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    r: YarnRope,
) -> Result<(), Error> {
    let YarnRope {
        head_dim,
        theta,
        yarn:
            Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_max_position,
            },
        attention_factor,
        interleaved,
    } = r;
    const OP: &str = "elemwise.rope";
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
        Fire::at(ROPE_UNIT, "::pie::elemwise::rope_yarn").apply(rotate_launch(
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
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn rope_partial(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: u32,
) -> Result<(), Error> {
    let num_q_heads = q_heads(op, q.width, head_dim)?;
    let num_kv_heads = heads(op, k.width, head_dim)?;
    let pairs = cache_pairs(rotary_dim / 2);
    ctx.fire(
        op,
        Fire::at(ROPE_UNIT, "::pie::elemwise::rope_partial<::pie::bf16>")
            .apply(Launch::per_row(q.rows, ROTATE_BLOCK).smem(pairs * 2 * 4)),
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            0_i32.arg(),
            stated(op, num_q_heads)?.arg(),
            stated(op, num_kv_heads)?.arg(),
            stated(op, head_dim)?.arg(),
            stated(op, rotary_dim)?.arg(),
            stated(op, span)?.arg(),
            theta.arg(),
            stated(op, pairs)?.arg(),
            ctx.stage(),
        ],
    )
}

const ROPE_AXES_UNIT: &str = "elemwise/rope_axes.hip";

pub const MAX_AXES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeForm {
    Interleaved,
    Neox,
    Split,
    SplitLadder,
}

impl RopeForm {
    const fn stamp(self) -> &'static str {
        match self {
            RopeForm::Interleaved => "0",
            RopeForm::Neox => "1",
            RopeForm::Split => "2",
            RopeForm::SplitLadder => "3",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rope_axes(
    ctx: &Ctx,
    x: Tensor,
    positions: Tensor,
    dims: [u32; MAX_AXES],
    thetas: [f32; MAX_AXES],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_axes";
    dtype_dispatch!(OP, x.dtype, { Bf16 => (), F16 => () });
    debug_assert!(
        x.rows == o.rows && x.width == o.width && x.dtype == o.dtype,
        "`{OP}` writes the rectangle it reads"
    );
    let rows = nonzero(OP, "rows", o.rows)?;
    nonzero(OP, "the head width this rotation states", head_dim)?;
    if !head_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the rotated prefix is {rotary_dim} wide, and the head it sits at the front \
                 of is {head_dim}"
            ),
        ));
    }
    if o.width == 0 || !o.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide row is not a whole number of {head_dim}-wide heads",
                o.width
            ),
        ));
    }
    let heads = o.width / head_dim;

    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {:?}, and this rotation reads f32 coordinates \
                 (they may be fractional)",
                positions.dtype
            ),
        ));
    }
    let axes = positions.width as usize;
    if axes == 0 || axes > MAX_AXES || positions.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {} x {}, and this rotation reads one coordinate per \
                 axis (at most {MAX_AXES}) for each of {rows} rows",
                positions.rows, positions.width
            ),
        ));
    }
    if form == RopeForm::SplitLadder {
        if rotary_dim != head_dim {
            return Err(refuse(
                OP,
                format!(
                    "the ladder pairs (i, i + head_dim/2) and turns the whole head, and this \
                     call rotates {rotary_dim} of {head_dim}"
                ),
            ));
        }
        let span: u32 = dims[..axes].iter().sum();
        let row = heads * rotary_dim;
        if span == 0 || span > row || !(row - span).is_multiple_of(2) {
            return Err(refuse(
                OP,
                format!(
                    "the ladder's axes own {span} channels of a {row}-wide rotated row, which \
                     leaves no whole identity pad"
                ),
            ));
        }
        if dims[..axes].iter().any(|d| *d != dims[0]) {
            return Err(refuse(
                OP,
                format!("one ladder hands its axes out round-robin, and {dims:?} is not flat"),
            ));
        }
        return fire_axes(
            ctx, x, positions, form, dims, thetas, rotary_dim, head_dim, heads, rows, o,
        );
    }
    let mut spanned = 0u32;
    for (a, &d) in dims.iter().enumerate() {
        if a >= axes {
            if d != 0 {
                return Err(refuse(
                    OP,
                    format!(
                        "axis {a} owns {d} channels and the position stream carries {axes} \
                         axes; an axis with no coordinate is a text to fix"
                    ),
                ));
            }
            continue;
        }
        if d == 0 || !d.is_multiple_of(2) {
            return Err(refuse(
                OP,
                format!("axis {a} owns {d} channels, and an axis turns whole pairs"),
            ));
        }
        spanned += d;
    }
    if spanned != rotary_dim {
        return Err(refuse(
            OP,
            format!(
                "the axes own {spanned} channels between them and the rotated prefix is \
                 {rotary_dim} wide; the blocks tile it in axis order"
            ),
        ));
    }

    fire_axes(
        ctx, x, positions, form, dims, thetas, rotary_dim, head_dim, heads, rows, o,
    )
}

#[allow(clippy::too_many_arguments)]
fn fire_axes(
    ctx: &Ctx,
    x: Tensor,
    positions: Tensor,
    form: RopeForm,
    dims: [u32; MAX_AXES],
    thetas: [f32; MAX_AXES],
    rotary_dim: u32,
    head_dim: u32,
    heads: u32,
    rows: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_axes";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ctx.fire(
        OP,
        Fire::at(
            ROPE_AXES_UNIT,
            symbol(&format!(
                "::pie::elemwise::rope_axes<{t}, {}>",
                form.stamp()
            )),
        )
        .apply(Launch::per_row(rows, ROTATE_BLOCK)),
        &[
            x.arg(),
            positions.arg(),
            o.arg(),
            stated(OP, positions.width)?.arg(),
            stated(OP, dims[0])?.arg(),
            stated(OP, dims[1])?.arg(),
            stated(OP, dims[2])?.arg(),
            stated(OP, dims[3])?.arg(),
            thetas[0].arg(),
            thetas[1].arg(),
            thetas[2].arg(),
            thetas[3].arg(),
            stated(OP, rotary_dim)?.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, heads)?.arg(),
            ctx.stage(),
        ],
    )
}

const ROPE_MROPE_UNIT: &str = "elemwise/rope_mrope.hip";

pub const AXES: u32 = 3;

#[allow(clippy::too_many_arguments)]
pub fn interleaved(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    fire_mrope(
        ctx,
        "::pie::elemwise::rope_mrope",
        q,
        k,
        positions,
        sections,
        rotary_dim,
        head_dim,
        theta,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn blocked(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    fire_mrope(
        ctx,
        "::pie::elemwise::rope_mrope_blocked",
        q,
        k,
        positions,
        sections,
        rotary_dim,
        head_dim,
        theta,
    )
}

/// Gemma's per-block form: axis `a` owns its own `2·sections[a]`-wide block,
/// a rotate_half over that block.
#[allow(clippy::too_many_arguments)]
pub fn split(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    if sections.iter().all(|&s| s == 0) {
        return Err(refuse(
            "elemwise.rope",
            format!(
                "the sections {sections:?} name no frequency pair, and a split rotation \
                 turns each axis by the pairs its section holds"
            ),
        ));
    }
    fire_mrope(
        ctx,
        "::pie::elemwise::rope_mrope_split",
        q,
        k,
        positions,
        sections,
        rotary_dim,
        head_dim,
        theta,
    )
}

#[allow(clippy::too_many_arguments)]
fn fire_mrope(
    ctx: &Ctx,
    entry: &'static str,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");

    nonzero(OP, "the head width this rotation states", head_dim)?;
    if !head_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim {
        return Err(refuse(
            OP,
            format!(
                "the rotated prefix is {rotary_dim} wide, and the head it sits at the front \
                 of is {head_dim}"
            ),
        ));
    }
    let num_q_heads = heads_of(OP, "query", q.width, head_dim)?;
    let num_kv_heads = heads_of(OP, "key", k.width, head_dim)?;

    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {:?}, and this rotation reads i32 (t, h, w) triples",
                positions.dtype
            ),
        ));
    }
    if positions.width != AXES || positions.rows != q.rows {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {} x {}, and this rotation reads one (t, h, w) triple \
                 per one of {} rotated rows",
                positions.rows, positions.width, q.rows
            ),
        ));
    }

    let half = head_dim / 2;
    let stated_pairs: u32 = sections.iter().copied().sum();
    if stated_pairs > half {
        return Err(refuse(
            OP,
            format!(
                "the sections {sections:?} name {stated_pairs} frequency pairs and a \
                 {head_dim}-wide head has {half}"
            ),
        ));
    }

    let rows = nonzero(OP, "rows", q.rows)?;
    ctx.fire(
        OP,
        Fire::at(ROPE_MROPE_UNIT, entry).apply(Launch::per_row(rows, ROTATE_BLOCK)),
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, rotary_dim)?.arg(),
            theta.arg(),
            stated(OP, sections[0])?.arg(),
            stated(OP, sections[1])?.arg(),
            stated(OP, sections[2])?.arg(),
            ctx.stage(),
        ],
    )
}

fn heads_of(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}
