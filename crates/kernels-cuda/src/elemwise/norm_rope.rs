#![allow(clippy::too_many_arguments)]

use super::rope::{Yarn, YarnAppend, partial_last, ramp_bounds};
use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::{KvPool, Tensor};
use dtype::Dtype;

const NORM_ROPE: &str = "elemwise/norm_rope.cu";

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
    const OP: &str = "fused.norm_rope";
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
    // a warp a (row, head) over whole 16-byte runs, heads up to 1024 wide
    let runs = (head_dim / 8).div_ceil(32);
    if ctx.tuning().q_norm_rope_warp
        && head_dim.is_multiple_of(8)
        && matches!(runs, 1 | 2 | 4)
        && [x.ptr, weight.ptr, y.ptr].iter().all(|p| aligned16(*p))
    {
        // eight warps a block, as many blocks as the card seats at once, each
        // warp striding over the (row, head) items
        const WARPS: u64 = 8;
        let items = u64::from(y.rows) * u64::from(heads);
        let resident = u64::from(ctx.multiprocessors().unwrap_or(148)) * 8;
        let blocks = u32::try_from(items.div_ceil(WARPS).min(resident)).map_err(|_| {
            refuse(
                OP,
                format!("{items} (row, head) items do not fit a 32-bit launch extent"),
            )
        })?;
        return ctx.fire(
            OP,
            Fire::at(
                NORM_ROPE,
                symbol(&format!(
                    "::pie::elemwise::q_rmsnorm_rope_partial_warp<{WARPS}, {runs}>"
                )),
            )
            .apply(Launch::grid([blocks, 1, 1], [WARPS as u32 * 32, 1, 1])),
            &[
                x.arg(),
                weight.arg(),
                positions.arg(),
                y.arg(),
                stated(OP, y.rows)?.arg(),
                stated(OP, head_dim)?.arg(),
                stated(OP, heads)?.arg(),
                stated(OP, rotary_dim)?.arg(),
                stated(OP, span)?.arg(),
                theta.arg(),
                eps.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(NORM_ROPE, "::pie::elemwise::q_rmsnorm_rope_partial<128>")
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
    const OP: &str = "fused.norm_rope";
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
    const OP: &str = "fused.norm_rope_last_append";
    let Some(append) = append else {
        return Ok(false);
    };
    if head_dim != 512
        || y.width != head_dim
        || !crate::attn::compressed::decode::fp8_ds(append.pool)
    {
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
            NORM_ROPE,
            symbol(&format!(
                "::pie::elemwise::norm_rope_last<{per}, {weighted}, {appends}>"
            )),
        )
        .apply(Launch::grid(
            [strided_blocks(ctx, units), 1, 1],
            [256, 1, 1],
        )),
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
    const OP: &str = "fused.q_gate_norm_rope_append";
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
            NORM_ROPE,
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

/// One side of [`norm_rope_pair`]: the rows, the head width and the norm's
/// epsilon, where the normed and rotated rows land.
pub struct Side<'a> {
    pub x: Tensor,
    pub head_dim: u32,
    pub eps: f32,
    pub y: &'a mut Tensor,
}

/// DeepSeek-V4's q (scale-free per head) and its kv plane (scaled, one
/// 512-wide head) normed and rotated (forward, interleaved) in one launch
/// (`norm_rope_pair`), the plane also written into DeepSeek's fp8 pool where
/// `append` names one: whether it fired (under `kernel.elemwise.rows_vec`,
/// for the shapes it is built for). Nothing is fired where it did not.
#[allow(clippy::too_many_arguments)]
pub fn norm_rope_pair(
    ctx: &Ctx,
    q: Side<'_>,
    kv: Side<'_>,
    kv_weight: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    theta: f32,
    yarn: Option<Yarn>,
    append: Option<Append<'_>>,
) -> Result<bool, Error> {
    const OP: &str = "fused.norm_rope_pair";
    let qper = q.head_dim / 32;
    let fits = ctx.tuning().rows_vec
        && q.x.dtype == Dtype::Bf16
        && kv.x.dtype == Dtype::Bf16
        && kv_weight.dtype == Dtype::Bf16
        && matches!(q.head_dim, 128 | 256 | 512)
        && kv.head_dim == 512
        && kv.y.width == 512
        && rotary_dim > 0
        && rotary_dim <= 128.min(q.head_dim)
        && rotary_dim.is_multiple_of(qper.min(8))
        && q.y.width.is_multiple_of(q.head_dim)
        && q.y.rows == kv.y.rows
        && append
            .as_ref()
            .is_none_or(|a| crate::attn::compressed::decode::fp8_ds(a.pool))
        && [q.x.ptr, q.y.ptr, kv.x.ptr, kv.y.ptr, kv_weight.ptr]
            .iter()
            .all(|&at| aligned16(at));
    if !fits {
        return Ok(false);
    }
    positions_stream(OP, positions, &q.x);
    let (factor, low_dim, high_dim) = match yarn {
        Some(yn) => {
            let max_position = nonzero(
                OP,
                "the position span this layer's YaRN ramp states",
                yn.original_max_position,
            )?;
            let (low, high) = ramp_bounds(
                stated(OP, rotary_dim)?,
                theta,
                yn.beta_fast,
                yn.beta_slow,
                stated(OP, max_position)?,
            );
            (yn.factor, low, high)
        }
        None => (UNSCALED, 0.0, 0.0),
    };
    let q_heads = q.y.width / q.head_dim;
    let q_units = q.y.rows.checked_mul(q_heads).ok_or_else(|| {
        refuse(
            OP,
            format!("{} rows of {q_heads} heads overflow a grid", q.y.rows),
        )
    })?;
    let kv_units = kv.y.rows;
    let units = q_units + kv_units;
    if units == 0 {
        return Ok(true);
    }
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
        OP,
        Fire::at(
            NORM_ROPE,
            symbol(&format!(
                "::pie::elemwise::norm_rope_pair<{qper}, {appends}>"
            )),
        )
        .apply(Launch::grid(
            [strided_blocks(ctx, units), 1, 1],
            [256, 1, 1],
        )),
        &[
            q.x.arg(),
            q.y.arg(),
            stated(OP, q_units)?.arg(),
            stated(OP, q_heads)?.arg(),
            q.eps.arg(),
            kv.x.arg(),
            kv_weight.arg(),
            kv.y.arg(),
            stated(OP, kv_units)?.arg(),
            kv.eps.arg(),
            positions.arg(),
            stated(OP, rotary_dim)?.arg(),
            theta.arg(),
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

/// Blocks of eight warps for `units` warp-units, at most eight blocks a
/// multiprocessor (a full one's threads): the kernels stride over the rest,
/// working the tail's frequencies once a block rather than once a unit-eight.
fn strided_blocks(ctx: &Ctx, units: u32) -> u32 {
    let resident = ctx.multiprocessors().unwrap_or(148).saturating_mul(8);
    units.div_ceil(8).min(resident).max(1)
}
