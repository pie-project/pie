//! Rotary embeddings from `kernels/elemwise/rope.slang`, in place on rows:
//! neox halves at a span (`elemwise.rope_full`, `rope_partial`,
//! `rope_partial_q`), each head's rotated tail (`rope_partial_last`),
//! whole-head YaRN (`rope_yarn`), on bf16 two pairs a lane so a lane owns
//! whole words; M-RoPE sections (`rope_mrope`) and position axes
//! (`rope_axes`), a pair a lane, bf16 halves through atomics.
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

const ROPE: &str = "elemwise/rope.slang";

const GROUP: u32 = 64;

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

/// The first `rotary/2` pairs of each head turn, pair `i` being
/// `(i, i + span)` at `theta^(-i/span)`.
fn rotate(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    positions: Tensor,
    base: f32,
    head_dim: u32,
    rotary: u32,
    span: u32,
) -> Result<(), Error> {
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "elemwise_rope_neox_bf16" });
    positions_stream(op, positions, x);
    let lanes = rope_grid(op, rotary, x.width, head_dim, x.rows)?;
    if !span.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("a pair {span} apart splits a bf16 word"),
        ));
    }
    ctx.fire(
        Fire::at(ROPE, entry).apply(Grid::of(pairs(lanes), [GROUP, 1, 1])),
        &[
            x.arg_mut(),
            positions.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
            stated(op, span)?.arg(),
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
    const OP: &str = "elemwise.rope_full";
    if interleaved {
        return Err(refuse(
            OP,
            "interleaved pairs are stated, and every neox arm rotates halves",
        ));
    }
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    let base = theta.log2();
    let span = head_dim / 2;
    rotate(ctx, OP, q, positions, base, head_dim, head_dim, span)?;
    rotate(ctx, OP, k, positions, base, head_dim, head_dim, span)
}

pub fn partial(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: u32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_partial";
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");
    let base = theta.log2();
    rotate(ctx, OP, q, positions, base, head_dim, rotary_dim, span)?;
    rotate(ctx, OP, k, positions, base, head_dim, rotary_dim, span)
}

pub fn partial_q(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: u32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_partial_q";
    rotate(
        ctx,
        OP,
        q,
        positions,
        theta.log2(),
        head_dim,
        rotary_dim,
        span,
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
    const OP: &str = "elemwise.rope_partial_last";
    if rotary_dim > head_dim {
        return Err(refuse(
            OP,
            format!(
                "the rotated tail is {rotary_dim} wide, wider than the {head_dim}-wide head \
                 it sits at the end of"
            ),
        ));
    }
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "elemwise_rope_tail_bf16" });
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
        Fire::at(ROPE, entry).apply(Grid::of(pairs(lanes), [GROUP, 1, 1])),
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
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "elemwise_rope_yarn_bf16" });
    positions_stream(op, positions, x);
    let lanes = rope_grid(op, head_dim, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at(ROPE, entry).apply(Grid::of(pairs(lanes), [GROUP, 1, 1])),
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
    const OP: &str = "elemwise.rope_yarn";
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

/// The position axes an M-RoPE row reads: t, h, w.
pub const MROPE_AXES: u32 = 3;

fn heads_in(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}

fn rotate_sections(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    x: Tensor,
    positions: Tensor,
    base: f32,
    head_dim: u32,
    pairs: u32,
    heads: u32,
    sections: [u32; MROPE_AXES as usize],
) -> Result<(), Error> {
    if heads == 0 || pairs == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(ROPE, entry).apply(Grid::of([pairs, heads, x.rows], [GROUP, 1, 1])),
        &[
            x.arg_mut(),
            positions.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
            stated(op, sections[0])?.arg(),
            stated(op, sections[1])?.arg(),
            stated(op, sections[2])?.arg(),
            stated(op, heads)?.arg(),
            stated(op, pairs)?.arg(),
        ],
    )
}

struct Heads {
    num_q_heads: u32,
    num_kv_heads: u32,
}

fn sections_fit(
    op: &'static str,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; MROPE_AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
) -> Result<Heads, Error> {
    debug_assert_eq!(k.dtype, q.dtype, "`{op}` rotates q and k in one element");
    nonzero(op, "the head width this rotation states", head_dim)?;
    if !head_dim.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim {
        return Err(refuse(
            op,
            format!(
                "the rotated prefix is {rotary_dim} wide, and the head it sits at the front of is {head_dim}"
            ),
        ));
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the rotated prefix {rotary_dim} is not a whole number of pairs"),
        ));
    }
    let num_q_heads = heads_in(op, "query", q.width, head_dim)?;
    let num_kv_heads = heads_in(op, "key", k.width, head_dim)?;
    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the position stream is {:?}, and this rotation reads i32 (t, h, w) triples",
                positions.dtype
            ),
        ));
    }
    if positions.width != MROPE_AXES || positions.rows != q.rows {
        return Err(refuse(
            op,
            format!(
                "the position stream is {} x {}, and this rotation reads one (t, h, w) triple per \
                 one of {} rotated rows",
                positions.rows, positions.width, q.rows
            ),
        ));
    }
    let half = head_dim / 2;
    let stated_pairs: u32 = sections.iter().copied().sum();
    if stated_pairs > half {
        return Err(refuse(
            op,
            format!(
                "the sections {sections:?} name {stated_pairs} frequency pairs and a {head_dim}-wide head has {half}"
            ),
        ));
    }
    nonzero(op, "rows", q.rows)?;
    Ok(Heads {
        num_q_heads,
        num_kv_heads,
    })
}

fn blocked_pairs(
    op: &'static str,
    sections: [u32; MROPE_AXES as usize],
    rotary_dim: u32,
) -> Result<u32, Error> {
    let total: u32 = sections.iter().copied().sum();
    if total == 0 {
        return Err(refuse(
            op,
            format!(
                "the sections {sections:?} name no frequency pair, and a blocked rotation divides its ladder by the pairs the sections tile"
            ),
        ));
    }
    Ok((rotary_dim / 2).min(total))
}

fn sections_both(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; MROPE_AXES as usize],
    pairs: u32,
    head_dim: u32,
    theta: f32,
    geom: Heads,
) -> Result<(), Error> {
    let base = theta.log2();
    rotate_sections(
        ctx,
        op,
        entry,
        q,
        positions,
        base,
        head_dim,
        pairs,
        geom.num_q_heads,
        sections,
    )?;
    rotate_sections(
        ctx,
        op,
        entry,
        k,
        positions,
        base,
        head_dim,
        pairs,
        geom.num_kv_heads,
        sections,
    )
}

pub fn mrope_interleaved(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; MROPE_AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_mrope";
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "elemwise_rope_mrope_interleaved_bf16" });
    let geom = sections_fit(OP, q, k, positions, sections, rotary_dim, head_dim)?;
    sections_both(
        ctx,
        OP,
        entry,
        q,
        k,
        positions,
        sections,
        rotary_dim / 2,
        head_dim,
        theta,
        geom,
    )
}

pub fn mrope_blocked(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; MROPE_AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_mrope";
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "elemwise_rope_mrope_blocked_bf16" });
    let geom = sections_fit(OP, q, k, positions, sections, rotary_dim, head_dim)?;
    let pairs = blocked_pairs(OP, sections, rotary_dim)?;
    sections_both(
        ctx, OP, entry, q, k, positions, sections, pairs, head_dim, theta, geom,
    )
}

pub fn mrope_split(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; MROPE_AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_mrope";
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "elemwise_rope_mrope_split_bf16" });
    let geom = sections_fit(OP, q, k, positions, sections, rotary_dim, head_dim)?;
    let pairs = blocked_pairs(OP, sections, rotary_dim)?;
    sections_both(
        ctx, OP, entry, q, k, positions, sections, pairs, head_dim, theta, geom,
    )
}

/// The workgroup width the axes instantiations are built at.
const AXES_GROUP: u32 = 256;

/// How an axis's channels pair up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeForm {
    /// Adjacent channels.
    Interleaved,
    /// Halves of the rotated prefix.
    Neox,
    /// Halves of the axis's own block.
    Split,
    /// Axes cycling over every head's rotated channels after a centred pad.
    SplitLadder,
}

impl RopeForm {
    const fn word(self) -> i32 {
        match self {
            Self::Interleaved => 0,
            Self::Neox => 1,
            Self::Split => 2,
            Self::SplitLadder => 3,
        }
    }
}

/// `x` turned in place by `positions` (f32, a row of axes a token), each
/// axis `dims[a]` channels at `thetas[a]`; a zero width ends the axes.
#[allow(clippy::too_many_arguments)]
pub fn axes(
    ctx: &Ctx<'_>,
    x: Tensor,
    positions: Tensor,
    dims: [u32; 4],
    thetas: [f32; 4],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_axes";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_rope_axes_bf16",
        F32 => "elemwise_rope_axes_f32",
    });
    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!("{:?} positions; a rotation reads f32", positions.dtype),
        ));
    }
    let rows = nonzero(OP, "rows", x.rows)?;
    let head_dim = nonzero(OP, "the head width", head_dim)?;
    if !x.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("a {}-wide row of {head_dim}-wide heads", x.width),
        ));
    }
    let heads = x.width / head_dim;
    let axes = u32::try_from(dims.iter().take_while(|d| **d != 0).count()).unwrap_or(0);
    if axes == 0 || dims[..axes as usize].iter().any(|d| d % 2 != 0) {
        return Err(refuse(OP, format!("axis widths {dims:?}")));
    }
    let span: u32 = dims[..axes as usize].iter().sum();
    let fits = if form == RopeForm::SplitLadder {
        rotary_dim != 0
            && rotary_dim <= head_dim
            && span <= heads * rotary_dim
            && (heads * rotary_dim - span).is_multiple_of(2)
    } else {
        span == rotary_dim && rotary_dim != 0 && rotary_dim <= head_dim
    };
    if !fits || positions.rows < rows || positions.width < axes {
        return Err(refuse(
            OP,
            format!(
                "axes {dims:?} over {rotary_dim} of {head_dim} channels, {} x {} positions \
                 for {rows} rows",
                positions.rows, positions.width
            ),
        ));
    }
    let lanes = heads * (rotary_dim / 2);
    ctx.fire(
        Fire::at(ROPE, entry).apply(Grid::of([lanes, rows, 1], [AXES_GROUP, 1, 1])),
        &[
            x.arg_mut(),
            positions.arg(),
            dims[0].arg(),
            dims[1].arg(),
            dims[2].arg(),
            dims[3].arg(),
            thetas[0].arg(),
            thetas[1].arg(),
            thetas[2].arg(),
            thetas[3].arg(),
            axes.arg(),
            rotary_dim.arg(),
            head_dim.arg(),
            heads.arg(),
            form.word().arg(),
        ],
    )
}
