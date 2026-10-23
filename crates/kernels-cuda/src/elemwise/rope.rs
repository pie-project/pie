#![allow(clippy::too_many_arguments)]

use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::{KvPool, Tensor};
use dtype::Dtype;

const ROPE: &str = "elemwise/rope.cu";

pub const ROTATE_BLOCK: u32 = 256;

pub const MAX_CACHED_PAIRS: u32 = 4096;

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
            ROPE,
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

impl YarnRope {
    /// The dims the ramp runs between, over the span the checkpoint states.
    fn ramp(&self, op: &'static str) -> Result<(f32, f32), Error> {
        let span = "the position span this checkpoint's YaRN block states";
        let max_position = stated(op, nonzero(op, span, self.yarn.original_max_position)?)?;
        let y = self.yarn;
        Ok(ramp_bounds(
            stated(op, self.head_dim)?,
            self.theta,
            y.beta_fast,
            y.beta_slow,
            max_position,
        ))
    }
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
    // A row's heads over blocks of eight (DeepSeek-V4: 64 heads, a 64-wide
    // tail), the angles worked out once a block rather than once a head and
    // pair: the same values land.
    let half = rotary_dim / 2;
    // Four pairs a thread over 16- or 8-byte runs, a block as many heads as
    // fill it, the angles the spread form's.
    if ctx.tuning().rows_vec
        && half > 0
        && half <= ROTATE_BLOCK
        && rotary_dim.is_multiple_of(8)
        && head_dim.is_multiple_of(8)
        && aligned16(q.ptr)
    {
        let heads_per_block = (ROTATE_BLOCK / (half / 4)).max(1);
        return ctx.fire(
            OP,
            Fire::at(ROPE, "::pie::elemwise::rope_partial_last_vec").apply(
                Launch::grid(
                    [q.rows, num_q_heads.div_ceil(heads_per_block), 1],
                    [ROTATE_BLOCK, 1, 1],
                )
                .smem(half * 8)
                .programmatic(ctx.tuning().pdl),
            ),
            &[
                q.arg(),
                positions.arg(),
                stated(OP, num_q_heads)?.arg(),
                stated(OP, head_dim)?.arg(),
                stated(OP, rotary_dim)?.arg(),
                theta.arg(),
                inverse.arg(),
                interleaved.arg(),
                factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
                stated(OP, heads_per_block)?.arg(),
                ctx.stage(),
            ],
        );
    }
    if half > 0 && half <= ROTATE_BLOCK {
        const HEADS: u32 = 8;
        let heads = num_q_heads;
        return ctx.fire(
            OP,
            Fire::at(ROPE, "::pie::elemwise::rope_partial_last_spread").apply(
                Launch::grid([q.rows, heads.div_ceil(HEADS), 1], [ROTATE_BLOCK, 1, 1])
                    .smem(half * 8),
            ),
            &[
                q.arg(),
                positions.arg(),
                stated(OP, num_q_heads)?.arg(),
                stated(OP, head_dim)?.arg(),
                stated(OP, rotary_dim)?.arg(),
                theta.arg(),
                inverse.arg(),
                interleaved.arg(),
                factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
                stated(OP, HEADS)?.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(ROPE, "::pie::elemwise::rope_partial_last")
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
            inverse.arg(),
            interleaved.arg(),
            factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            ctx.stage(),
        ],
    )
}

pub fn yarn(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    r: YarnRope,
) -> Result<(), Error> {
    yarn_fire(ctx, "elemwise.rope", q, k, positions, r, None)
}

/// Where a YaRN rotation also lands the rotated keys and the values in a
/// paged cache, at each row's explicit page and offset.
#[derive(Clone, Copy)]
pub struct YarnAppend<'a> {
    pub v: Tensor,
    pub pool: &'a KvPool,
    pub write_page: Tensor,
    pub write_offset: Tensor,
}

impl YarnAppend<'_> {
    /// Whether the fused write lands what `kv_append` would: a bf16 NHD
    /// pool with no quantised planes and no envelopes to keep.
    #[must_use]
    pub fn fits(&self) -> bool {
        self.pool.layout == 0
            && !self.pool.has_envelopes
            && self.pool.keys.dtype == Dtype::Bf16
            && self.pool.values.dtype == Dtype::Bf16
            && self.pool.key_scales.is_absent()
            && self.pool.value_scales.is_absent()
            && self.v.dtype == Dtype::Bf16
    }
}

/// [`yarn`], landing the rotated keys and the values in `append`'s cache in
/// the same launch. The caller checks [`YarnAppend::fits`] first.
pub fn yarn_append(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    r: YarnRope,
    append: YarnAppend<'_>,
) -> Result<(), Error> {
    yarn_fire(
        ctx,
        "fused.rope_yarn_append",
        q,
        k,
        positions,
        r,
        Some(append),
    )
}

/// [`yarn_append`] reading q, k and v straight out of `packed` (`[rows, q |
/// k | v]`, `q_width` and `kv_width` wide): the rotated q lands in `q_out`,
/// the rotated k and its v in `append`'s cache, the same bits as
/// `split_qkv`, [`yarn_append`] would land. `append.v` is not read. The
/// caller checks [`YarnAppend::fits`] first.
pub fn yarn_split_append(
    ctx: &Ctx,
    packed: Tensor,
    kv_width: u32,
    q_out: &mut Tensor,
    positions: Tensor,
    r: YarnRope,
    append: YarnAppend<'_>,
) -> Result<(), Error> {
    const OP: &str = "fused.split_rope_yarn_append";
    let YarnRope {
        head_dim,
        theta,
        yarn,
        attention_factor,
        interleaved,
    } = r;
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    if !append.fits() {
        return Err(refuse(
            OP,
            "the fused YaRN write lands a bf16 NHD cache without envelopes only",
        ));
    }
    positions_stream(OP, positions, q_out);
    let num_q_heads = q_heads(OP, q_out.width, head_dim)?;
    let num_kv_heads = heads(OP, kv_width, head_dim)?;
    if packed.width != q_out.width + 2 * kv_width || packed.rows != q_out.rows {
        return Err(refuse(
            OP,
            format!(
                "a [{} x {}] packed row is not q {} wide and k and v {} wide each over {} rows",
                packed.rows, packed.width, q_out.width, kv_width, q_out.rows
            ),
        ));
    }
    let (low_dim, high_dim) = r.ramp(OP)?;
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let per_block = heads_per_block(half);
    ctx.fire(
        OP,
        Fire::at(ROPE, "::pie::elemwise::rope_yarn_split_append").apply(rotate_launch(
            q_out.rows,
            num_q_heads + num_kv_heads,
            per_block,
            pairs * 8,
        )),
        &[
            packed.arg(),
            q_out.arg(),
            positions.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            theta.arg(),
            yarn.factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            attention_factor.arg(),
            interleaved.arg(),
            stated(OP, per_block)?.arg(),
            stated(OP, pairs)?.arg(),
            append.pool.keys.arg(),
            append.pool.values.arg(),
            append.write_page.arg(),
            append.write_offset.arg(),
            append.pool.row_valid.arg(),
            append.pool.page_size.arg(),
            ctx.stage(),
        ],
    )
}

fn yarn_fire(
    ctx: &Ctx,
    op: &'static str,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    r: YarnRope,
    append: Option<YarnAppend<'_>>,
) -> Result<(), Error> {
    let YarnRope {
        head_dim,
        theta,
        yarn,
        attention_factor,
        interleaved,
    } = r;
    dtype_dispatch!(op, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{op}` rotates q and k in one element");
    positions_stream(op, positions, q);
    let num_q_heads = q_heads(op, q.width, head_dim)?;
    let num_kv_heads = heads(op, k.width, head_dim)?;
    let (low_dim, high_dim) = r.ramp(op)?;
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let per_block = heads_per_block(half);
    let (entry, v, k_pages, v_pages, w_page, w_off, row_valid, page_size) = match append {
        Some(a) => {
            if !a.fits() {
                return Err(refuse(
                    op,
                    "the fused YaRN write lands a bf16 NHD cache without envelopes only",
                ));
            }
            (
                "::pie::elemwise::rope_yarn<::pie::true_type::value>",
                a.v.arg(),
                a.pool.keys.arg(),
                a.pool.values.arg(),
                a.write_page.arg(),
                a.write_offset.arg(),
                a.pool.row_valid.arg(),
                a.pool.page_size,
            )
        }
        None => (
            "::pie::elemwise::rope_yarn<::pie::false_type::value>",
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            0,
        ),
    };
    ctx.fire(
        op,
        Fire::at(ROPE, entry).apply(rotate_launch(
            q.rows,
            num_q_heads + num_kv_heads,
            per_block,
            pairs * 8,
        )),
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            stated(op, num_q_heads)?.arg(),
            stated(op, num_kv_heads)?.arg(),
            stated(op, head_dim)?.arg(),
            theta.arg(),
            yarn.factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            attention_factor.arg(),
            interleaved.arg(),
            stated(op, per_block)?.arg(),
            stated(op, pairs)?.arg(),
            v,
            k_pages,
            v_pages,
            w_page,
            w_off,
            row_valid,
            page_size.arg(),
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
    ctx.fire(
        op,
        Fire::at(ROPE, "::pie::elemwise::rope_partial<::pie::bf16>")
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
            stated(op, span)?.arg(),
            theta.arg(),
            ctx.stage(),
        ],
    )
}

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
            ROPE,
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
        "::pie::elemwise::rope_mrope<::pie::bf16>",
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
        "::pie::elemwise::rope_mrope_blocked<::pie::bf16>",
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
        Fire::at(ROPE, entry).apply(Launch::per_row(rows, ROTATE_BLOCK)),
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

/// The most pairs a table turns (`kRopeTablePairs`).
pub const TABLE_PAIRS: usize = 256;

#[repr(C)]
struct RopeTable {
    inv_freq: [f32; TABLE_PAIRS],
    mscale: f32,
}

/// The frequencies a rotation turns at.
#[derive(Debug, Clone, Copy)]
pub struct Freqs<'a> {
    pub inv_freq: &'a [f32],
    pub mscale: f32,
}

/// How a head's pairs sit: `(i, i + span)`, or `(2i, 2i + 1)` where
/// `interleaved`; `sections`, an interleaved M-RoPE's (t, h, w) split, where
/// the positions are triples.
#[derive(Debug, Clone, Copy)]
pub struct Pairs {
    pub head_dim: u32,
    pub span: u32,
    pub interleaved: bool,
    pub sections: Option<[u32; 3]>,
}

/// Turns q's and k's heads (`k` may be zero wide: q alone) at `freqs`, one
/// pair a stated frequency.
pub fn rope(
    ctx: &Ctx,
    op: &'static str,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    pairs: Pairs,
    freqs: Freqs<'_>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.dtype, { Bf16 => () });
    let Pairs {
        head_dim,
        span,
        interleaved,
        sections,
    } = pairs;
    nonzero(op, "the head width this rotation states", head_dim)?;
    let n = freqs.inv_freq.len();
    if n == 0 || n > TABLE_PAIRS {
        return Err(refuse(
            op,
            format!("a stated table turns 1 to {TABLE_PAIRS} pairs, not {n}"),
        ));
    }
    let turned = u32::try_from(n).unwrap_or(u32::MAX);
    let reach = if interleaved {
        2 * turned
    } else {
        span + turned
    };
    if (!interleaved && span < turned) || reach > head_dim {
        return Err(refuse(
            op,
            format!(
                "{turned} pairs {span} apart do not sit in a {head_dim}-wide head{}",
                if interleaved { " (interleaved)" } else { "" }
            ),
        ));
    }
    let heads = |what: &str, width: u32| -> Result<u32, Error> {
        if width.is_multiple_of(head_dim) {
            Ok(width / head_dim)
        } else {
            Err(refuse(
                op,
                format!(
                    "the {width}-wide {what} row is not a whole number of {head_dim}-wide heads"
                ),
            ))
        }
    };
    let num_q_heads = heads("query", q.width)?;
    let num_kv_heads = heads("key", k.width)?;
    let axes = if sections.is_some() { 3 } else { 1 };
    if positions.dtype != Dtype::I32 || positions.width != axes || positions.rows != q.rows {
        return Err(refuse(
            op,
            format!(
                "the position stream is {} x {} {:?}, and this rotation reads {axes} i32 a row \
                 of {} rows",
                positions.rows, positions.width, positions.dtype, q.rows
            ),
        ));
    }
    let [_, s1, s2] = sections.unwrap_or([0; 3]);
    let mut table = RopeTable {
        inv_freq: [0.0; TABLE_PAIRS],
        mscale: freqs.mscale,
    };
    table.inv_freq[..n].copy_from_slice(freqs.inv_freq);
    let rows = nonzero(op, "rows", q.rows)?;
    ctx.fire(
        op,
        Fire::at(ROPE, "::pie::elemwise::rope_table<::pie::bf16>")
            .apply(Launch::per_row(rows, ROTATE_BLOCK).smem(2 * turned * 4)),
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            stated(op, num_q_heads)?.arg(),
            stated(op, num_kv_heads)?.arg(),
            stated(op, head_dim)?.arg(),
            stated(op, turned)?.arg(),
            stated(op, span)?.arg(),
            interleaved.arg(),
            stated(op, s1)?.arg(),
            stated(op, s2)?.arg(),
            ArgValue::of(&table),
            ctx.stage(),
        ],
    )
}
