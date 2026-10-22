//! Rotations at stated frequencies: a checkpoint's rope scaling (linear,
//! dynamic NTK, YaRN, llama3, longrope) turned into one inverse frequency a
//! pair and a scale on cos and sin, the turn otherwise `rope_full`'s,
//! `rope_partial`'s or Qwen's interleaved M-RoPE.

use dtype::Dtype;

use crate::elemwise::rope::ROTATE_BLOCK;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const ROPE: &str = "elemwise/rope.cu";

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
