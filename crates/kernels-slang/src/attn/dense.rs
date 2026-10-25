//! Dense attention over segments (`attn.dense`, `attn.ragged`) from
//! `kernels/attn/dense.slang` and, for the common bf16 head widths of the
//! ragged form, `kernels/attn/dense_tiled.slang`.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const DENSE: &str = "attn/dense.slang";
const DENSE_TILED: &str = "attn/dense_tiled.slang";

/// The workgroup width of the scalar kernel: four strips of 32.
const THREADS: u32 = 128;

/// The tiled kernel's workgroup, and the query rows it answers.
const TILED_THREADS: u32 = 128;
const TILED_ROWS: u32 = 32;

/// The head-width bounds the scalar kernel is stamped at.
const STAMPS: [u32; 3] = [64, 128, 256];

fn stamp_for(op: &'static str, head_dim: u32) -> Result<usize, Error> {
    STAMPS
        .iter()
        .position(|stamp| head_dim <= *stamp)
        .ok_or_else(|| {
            refuse(
                op,
                format!(
                    "the {head_dim}-wide head is wider than the {}-wide accumulator this kernel \
                     is stamped for",
                    STAMPS[STAMPS.len() - 1]
                ),
            )
        })
}

fn heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

/// Each patch row attends every row of its image, the images split by the
/// i32 boundaries `segments`.
#[allow(clippy::too_many_arguments)]
pub fn bidirectional(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    segments: Tensor,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.dense";
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    debug_assert_eq!(v.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    nonzero(OP, "the head width this attention states", head_dim)?;
    super::pairs(OP, head_dim)?;
    let at = stamp_for(OP, head_dim)?;
    let entry = match q.dtype {
        Dtype::Bf16 => [
            "attn_dense_bf16_d_64",
            "attn_dense_bf16_d_128",
            "attn_dense_bf16_d_256",
        ][at],
        other => {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: other,
            });
        }
    };
    let num_q_heads = heads(OP, "query", q.width, head_dim)?;
    let num_kv_heads = heads(OP, "key", k.width, head_dim)?;
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    if segments.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the patch window's segment list is {:?}, and this attention walks an i32 \
                 boundary vector",
                segments.dtype
            ),
        ));
    }
    let images = segments.rows.saturating_sub(1);
    if images == 0 {
        return Err(refuse(
            OP,
            "the patch window's segment list spells no images",
        ));
    }
    let rows = nonzero(OP, "the patch rows this attention answers", q.rows)?;
    ctx.fire(
        Fire::at(DENSE, entry).apply(Grid::of([num_q_heads * THREADS, rows, 1], [THREADS, 1, 1])),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            o.arg_mut(),
            segments.arg(),
            stated(OP, images)?.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
        ],
    )
}

/// Segment `s`'s pair `(q, k)` is kept where bit `q * kv_len + k` of `bits` (u8, bound
/// whole words) from byte `indptr[s]` (i32) is set.
#[derive(Clone, Copy, Debug)]
pub struct CustomMask {
    pub bits: Tensor,
    pub indptr: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn ragged(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    mask: Option<CustomMask>,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.ragged";
    nonzero(OP, "the head width", head_dim)?;
    super::pairs(OP, head_dim)?;
    let q_heads = heads(OP, "query", q.width, head_dim)?;
    let kv_heads = nonzero(OP, "kv heads", kv_heads)?;
    if k.width != kv_heads * head_dim || v.width != k.width || !q_heads.is_multiple_of(kv_heads) {
        return Err(refuse(
            OP,
            format!(
                "{q_heads} query heads over a {}-wide key of {kv_heads} x {head_dim}",
                k.width
            ),
        ));
    }
    for (what, t) in [("query", q_indptr), ("key", kv_indptr)] {
        if t.dtype != Dtype::I32 {
            return Err(refuse(
                OP,
                format!("the {what} CSR is {:?}, not i32", t.dtype),
            ));
        }
    }
    let entries = (q_indptr.rows * q_indptr.width).min(kv_indptr.rows * kv_indptr.width);
    let segments = entries
        .checked_sub(1)
        .filter(|s| *s > 0)
        .ok_or_else(|| refuse(OP, format!("a CSR of {entries} entries names no segment")))?;
    if let Some(mask) = mask
        && (mask.bits.dtype != Dtype::U8
            || mask.indptr.dtype != Dtype::I32
            || mask.indptr.rows * mask.indptr.width < segments + 1)
    {
        return Err(refuse(
            OP,
            format!(
                "a custom mask is u8 bits and an i32 byte offset a segment; this one is {:?} \
                 bits and {} {:?} offsets for {segments} segments",
                mask.bits.dtype,
                mask.indptr.rows * mask.indptr.width,
                mask.indptr.dtype
            ),
        ));
    }
    let custom = mask.is_some();
    let rows = nonzero(OP, "query rows", q.rows)?;
    let (file, entry, grid) = match (q.dtype, head_dim) {
        (Dtype::Bf16, 64 | 128) => (
            DENSE_TILED,
            match (head_dim, custom) {
                (64, false) => "attn_dense_tiled_ragged_bf16_d_64",
                (64, true) => "attn_dense_tiled_ragged_custom_bf16_d_64",
                (_, false) => "attn_dense_tiled_ragged_bf16_d_128",
                (_, true) => "attn_dense_tiled_ragged_custom_bf16_d_128",
            },
            Grid::of(
                [q_heads * TILED_THREADS, rows.div_ceil(TILED_ROWS), 1],
                [TILED_THREADS, 1, 1],
            ),
        ),
        (dtype, _) => {
            let at = stamp_for(OP, head_dim)?;
            let entry = match (dtype, custom) {
                (Dtype::Bf16, false) => [
                    "attn_dense_ragged_bf16_d_64",
                    "attn_dense_ragged_bf16_d_128",
                    "attn_dense_ragged_bf16_d_256",
                ][at],
                (Dtype::Bf16, true) => [
                    "attn_dense_ragged_custom_bf16_d_64",
                    "attn_dense_ragged_custom_bf16_d_128",
                    "attn_dense_ragged_custom_bf16_d_256",
                ][at],
                (Dtype::F32, false) => [
                    "attn_dense_ragged_f32_d_64",
                    "attn_dense_ragged_f32_d_128",
                    "attn_dense_ragged_f32_d_256",
                ][at],
                (Dtype::F32, true) => [
                    "attn_dense_ragged_custom_f32_d_64",
                    "attn_dense_ragged_custom_f32_d_128",
                    "attn_dense_ragged_custom_f32_d_256",
                ][at],
                (other, _) => {
                    return Err(Error::DtypeUnsupported {
                        op: OP,
                        dtype: other,
                    });
                }
            };
            (
                DENSE,
                entry,
                Grid::of([q_heads * THREADS, rows, 1], [THREADS, 1, 1]),
            )
        }
    };
    let mut args = vec![
        q.arg(),
        k.arg(),
        v.arg(),
        o.arg_mut(),
        q_indptr.arg(),
        kv_indptr.arg(),
    ];
    if let Some(mask) = mask {
        args.extend([mask.bits.arg(), mask.indptr.arg()]);
    }
    args.extend([
        stated(OP, segments)?.arg(),
        stated(OP, q_heads)?.arg(),
        stated(OP, kv_heads)?.arg(),
        stated(OP, head_dim)?.arg(),
        sm_scale.arg(),
    ]);
    ctx.fire(Fire::at(file, entry).apply(grid), &args)
}
