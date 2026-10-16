#![allow(clippy::too_many_arguments)]

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "elemwise/rope.cuh";

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

fn heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this rotation states", head_dim)?;
    if !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}

fn q_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
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
) -> Result<(), Error> {
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
    const OP: &str = "elementwise.rope_partial";
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
    const OP: &str = "elementwise.rope_partial_q";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    positions_stream(OP, positions, q);
    let k = Tensor::new(q.ptr, q.rows, 0, q.dtype);
    rope_partial(ctx, OP, *q, k, positions, rotary_dim, head_dim, theta, span)
}

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_rope_partial_q(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    positions: Tensor,
    rotary_dim: u32,
    theta: f32,
    span: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_rope_partial_q";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    positions_stream(OP, positions, &x);
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !y.width.is_multiple_of(head_dim)
        || !rotary_dim.is_multiple_of(2)
        || span < rotary_dim / 2
        || span + rotary_dim / 2 > head_dim
    {
        return Err(refuse(
            OP,
            format!(
                "wants whole even heads and an even rotary width whose pairs sit within them, \
                 and was handed width={}, head_dim={head_dim}, rotary_dim={rotary_dim}, \
                 span={span}",
                y.width
            ),
        ));
    }
    let heads = y.width / head_dim;
    if y.rows == 0 || heads == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::q_rmsnorm_rope_partial<128>")
            .apply(Launch::grid([y.rows, heads, 1], [128, 1, 1])),
        &[
            x.arg(),
            weight.arg(),
            positions.arg(),
            y.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, rotary_dim)?.arg(),
            stated(OP, span)?.arg(),
            theta.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
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
    const OP: &str = "elementwise.rope_partial_last";
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
            Fire::at(FILE, "::pie::elemwise::rope_partial_last_vec").apply(
                Launch::grid(
                    [q.rows, num_q_heads.div_ceil(heads_per_block), 1],
                    [ROTATE_BLOCK, 1, 1],
                )
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
            Fire::at(FILE, "::pie::elemwise::rope_partial_last_spread").apply(
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
            inverse.arg(),
            interleaved.arg(),
            factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            ctx.stage(),
        ],
    )
}

/// A head's RMS norm (times `weight` where given; else scale-free) into `y`,
/// then the rotation of its last `rotary_dim` dims there
/// (`partial_last`). Under `kernel.elemwise.rows_vec`, where the head is
/// 128, 256 or 512 wide, the rotation interleaved and its tail (at most 128
/// wide) whole lane runs, both in one pass a warp a head; otherwise the two
/// launches the unfused ops fire, unchanged.
#[allow(clippy::too_many_arguments)]
pub fn norm_rope_last(
    ctx: &Ctx,
    x: Tensor,
    weight: Option<Tensor>,
    head_dim: u32,
    eps: f32,
    positions: Tensor,
    rotary_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.norm_rope_last";
    let tail = Tail {
        head_dim,
        eps,
        rotary_dim,
        theta,
        interleaved,
        inverse,
        yarn,
    };
    if one_pass(ctx, OP, x, weight, positions, tail, y, None)? {
        return Ok(());
    }
    match weight {
        Some(w) => crate::elemwise::norm::rmsnorm(ctx, x, w, eps, y)?,
        None => crate::elemwise::norm::rmsnorm_no_scale(ctx, x, head_dim, eps, y)?,
    }
    partial_last(
        ctx,
        y,
        positions,
        rotary_dim,
        head_dim,
        theta,
        interleaved,
        inverse,
        yarn,
    )
}

/// Where a normed, rotated row is written as well: DeepSeek's fp8 pool and
/// each row's page and offset.
pub struct Append<'a> {
    pub pool: &'a KvPool,
    pub write_page: Tensor,
    pub write_offset: Tensor,
}

/// `norm_rope_last` of a row one 512-wide head (scaled), the rotated row
/// also written into DeepSeek's fp8 pool at its page and offset, in the one
/// pass: whether it landed (under `kernel.elemwise.rows_vec`, an fp8 pool
/// given). Nothing is fired where it did not.
#[allow(clippy::too_many_arguments)]
pub fn norm_rope_last_append(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    positions: Tensor,
    rotary_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
    y: &mut Tensor,
    append: Option<Append<'_>>,
) -> Result<bool, Error> {
    const OP: &str = "elementwise.norm_rope_last_append";
    let Some(append) = append else {
        return Ok(false);
    };
    if head_dim != 512 || y.width != head_dim || !crate::attn::dsv4::fp8_ds(append.pool) {
        return Ok(false);
    }
    let tail = Tail {
        head_dim,
        eps,
        rotary_dim,
        theta,
        interleaved,
        inverse,
        yarn,
    };
    one_pass(ctx, OP, x, Some(weight), positions, tail, y, Some(append))
}

#[derive(Clone, Copy)]
struct Tail {
    head_dim: u32,
    eps: f32,
    rotary_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
}

/// The one pass, where its shape and the profile allow: whether it fired.
#[allow(clippy::too_many_arguments)]
fn one_pass(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    weight: Option<Tensor>,
    positions: Tensor,
    t: Tail,
    y: &mut Tensor,
    append: Option<Append<'_>>,
) -> Result<bool, Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    positions_stream(op, positions, &x);
    let head_dim = t.head_dim;
    let rotary_dim = t.rotary_dim;
    let per = head_dim / 32;
    let fits = ctx.tuning().rows_vec
        && t.interleaved
        && matches!(head_dim, 128 | 256 | 512)
        && rotary_dim > 0
        && rotary_dim <= head_dim.min(128)
        && rotary_dim.is_multiple_of(per.min(8))
        && y.width.is_multiple_of(head_dim)
        && weight.is_none_or(|w| w.dtype == Dtype::Bf16 && aligned16(w.ptr))
        && [x.ptr, y.ptr].iter().all(|&at| aligned16(at));
    if !fits {
        return Ok(false);
    }
    let (factor, low_dim, high_dim) = match t.yarn {
        Some(yn) => {
            let max_position = nonzero(
                op,
                "the position span this layer's YaRN ramp states",
                yn.original_max_position,
            )?;
            let (low, high) = ramp_bounds(
                stated(op, rotary_dim)?,
                t.theta,
                yn.beta_fast,
                yn.beta_slow,
                stated(op, max_position)?,
            );
            (yn.factor, low, high)
        }
        None => (UNSCALED, 0.0, 0.0),
    };
    let heads = y.width / head_dim;
    let units = y.rows.checked_mul(heads).ok_or_else(|| {
        refuse(
            op,
            format!("{} rows of {heads} heads overflow a grid", y.rows),
        )
    })?;
    if units == 0 {
        return Ok(true);
    }
    let weighted = weight.is_some();
    let appends = append.is_some();
    let (pages, write_page, write_offset, row_valid, page_size) = match &append {
        Some(a) => (
            a.pool.keys.arg(),
            a.write_page.arg(),
            a.write_offset.arg(),
            a.pool.row_valid.arg(),
            a.pool.page_size.arg(),
        ),
        None => (
            Tensor::ABSENT.arg(),
            Tensor::ABSENT.arg(),
            Tensor::ABSENT.arg(),
            Tensor::ABSENT.arg(),
            0_i32.arg(),
        ),
    };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::norm_rope_last<{per}, {weighted}, {appends}>"
            )),
        )
        .apply(Launch::grid([units.div_ceil(8), 1, 1], [256, 1, 1])),
        &[
            x.arg(),
            weight.unwrap_or(Tensor::ABSENT).arg(),
            y.arg(),
            positions.arg(),
            stated(op, units)?.arg(),
            stated(op, heads)?.arg(),
            t.eps.arg(),
            stated(op, rotary_dim)?.arg(),
            t.theta.arg(),
            t.inverse.arg(),
            factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            pages,
            write_page,
            write_offset,
            row_valid,
            page_size,
            ctx.stage(),
        ],
    )?;
    Ok(true)
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
) -> Result<(), Error> {
    yarn_fire(
        ctx,
        "elementwise.rope_yarn",
        q,
        k,
        positions,
        head_dim,
        theta,
        factor,
        beta_fast,
        beta_slow,
        attention_factor,
        original_max_position,
        interleaved,
        None,
    )
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
    head_dim: u32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: u32,
    interleaved: bool,
    append: YarnAppend<'_>,
) -> Result<(), Error> {
    yarn_fire(
        ctx,
        "elementwise.rope_yarn_append",
        q,
        k,
        positions,
        head_dim,
        theta,
        factor,
        beta_fast,
        beta_slow,
        attention_factor,
        original_max_position,
        interleaved,
        Some(append),
    )
}

/// [`yarn_append`] reading q, k and v straight out of `packed` (`[rows, q |
/// k | v]`, `q_width` and `kv_width` wide): the rotated q lands in `q_out`,
/// the rotated k and its v in `append`'s cache, the same bits as
/// `split_qkv`, [`yarn_append`] would land. `append.v` is not read. The
/// caller checks [`YarnAppend::fits`] first.
#[allow(clippy::too_many_arguments)]
pub fn yarn_split_append(
    ctx: &Ctx,
    packed: Tensor,
    kv_width: u32,
    q_out: &mut Tensor,
    positions: Tensor,
    head_dim: u32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: u32,
    interleaved: bool,
    append: YarnAppend<'_>,
) -> Result<(), Error> {
    const OP: &str = "elementwise.split_rope_yarn_append";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    if !append.fits() {
        return Err(refuse(
            OP,
            "the fused YaRN write lands a bf16 NHD cache without envelopes only",
        ));
    }
    positions_stream(OP, positions, q_out);
    let max_position = stated(
        OP,
        nonzero(
            OP,
            "the position span this checkpoint's YaRN block states",
            original_max_position,
        )?,
    )?;
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
        Fire::at(FILE, "::pie::elemwise::rope_yarn_split_append").apply(rotate_launch(
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
            factor.arg(),
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
    head_dim: u32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: u32,
    interleaved: bool,
    append: Option<YarnAppend<'_>>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{op}` rotates q and k in one element");
    positions_stream(op, positions, q);
    let max_position = stated(
        op,
        nonzero(
            op,
            "the position span this checkpoint's YaRN block states",
            original_max_position,
        )?,
    )?;
    let num_q_heads = q_heads(op, q.width, head_dim)?;
    let num_kv_heads = heads(op, k.width, head_dim)?;
    let (low_dim, high_dim) = ramp_bounds(
        stated(op, head_dim)?,
        theta,
        beta_fast,
        beta_slow,
        max_position,
    );
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
        Fire::at(FILE, entry).apply(rotate_launch(
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
            factor.arg(),
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
            stated(op, span)?.arg(),
            theta.arg(),
            ctx.stage(),
        ],
    )
}

/// Qwen3.5's attention prologue in one launch: the q heads cut from their
/// `[q | gate]` rows (`qg`) and the k heads, each normed per head with its
/// plus-one weight and turned by `rope_partial`'s pairs, the gate copied to
/// `gate`, and the turned k with its v landed in `append`'s cache. The
/// caller checks [`YarnAppend::fits`] first.
#[allow(clippy::too_many_arguments)]
pub fn qg_qknorm_rope_append(
    ctx: &Ctx,
    qg: Tensor,
    k: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    positions: Tensor,
    head_dim: u32,
    rotary_dim: u32,
    span: u32,
    theta: f32,
    q_eps: f32,
    k_eps: f32,
    q_out: &mut Tensor,
    gate: &mut Tensor,
    append: YarnAppend<'_>,
) -> Result<(), Error> {
    const OP: &str = "elementwise.qg_qknorm_rope_append";
    dtype_dispatch!(OP, qg.dtype, { Bf16 => () });
    positions_stream(OP, positions, q_out);
    if !append.fits()
        || !(head_dim == 128 || head_dim == 256)
        || !span.is_multiple_of(head_dim / 32)
        || !rotary_dim.is_multiple_of(2)
        || span < rotary_dim / 2
        || span + rotary_dim / 2 > head_dim
        || !q_out.width.is_multiple_of(head_dim)
        || !k.width.is_multiple_of(head_dim)
    {
        return Err(refuse(
            OP,
            format!(
                "wants whole 128- or 256-wide heads, an even rotary prefix whose pairs sit \
                 within them and a bf16 NHD cache; was handed head_dim={head_dim}, \
                 rotary_dim={rotary_dim}, span={span}"
            ),
        ));
    }
    let hq = q_out.width / head_dim;
    let hk = k.width / head_dim;
    if q_out.rows == 0 {
        return Ok(());
    }
    let units = q_out
        .rows
        .checked_mul(hq + hk)
        .ok_or_else(|| refuse(OP, "the (row, head) units overflow a grid"))?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::qg_qknorm_rope_append<{}>",
                head_dim / 32
            )),
        )
        .apply(Launch::grid([units.div_ceil(8), 1, 1], [256, 1, 1])),
        &[
            qg.arg(),
            k.arg(),
            append.v.arg(),
            q_norm.arg(),
            k_norm.arg(),
            positions.arg(),
            q_out.arg(),
            gate.arg(),
            append.pool.keys.arg(),
            append.pool.values.arg(),
            append.write_page.arg(),
            append.write_offset.arg(),
            append.pool.row_valid.arg(),
            append.pool.page_size.arg(),
            stated(OP, hq)?.arg(),
            stated(OP, hk)?.arg(),
            stated(OP, units)?.arg(),
            stated(OP, rotary_dim)?.arg(),
            stated(OP, span)?.arg(),
            theta.arg(),
            q_eps.arg(),
            k_eps.arg(),
            ctx.stage(),
        ],
    )
}
