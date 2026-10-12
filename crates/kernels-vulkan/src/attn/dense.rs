use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "attn/dense.slang";

const THREADS: u32 = 128;

const TILED_FILE: &str = "attn/tiled.slang";

const TILED_THREADS: u32 = 128;

const TILED_ROWS: u32 = 32;

const STAMPS: [u32; 3] = [64, 128, 256];

const DENSE: [&str; 3] = [
    "dense_bidirectional_bf16_d_64",
    "dense_bidirectional_bf16_d_128",
    "dense_bidirectional_bf16_d_256",
];

fn stamp_for(head_dim: u32) -> Option<usize> {
    STAMPS.iter().position(|stamp| head_dim <= *stamp)
}

fn row_heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

fn images_of(op: &'static str, segments: Tensor) -> Result<i32, Error> {
    if segments.dtype != Dtype::I32 {
        return Err(refuse(
            op,
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
            op,
            "the patch window's segment list spells no images",
        ));
    }
    stated(op, images)
}

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
    const OP: &str = "attention.dense";
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    debug_assert_eq!(v.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    nonzero(OP, "the head width this attention states", head_dim)?;
    let Some(at) = stamp_for(head_dim) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_dim}-wide head is wider than the {}-wide accumulator this kernel \
                 is stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => DENSE[at] });
    let num_q_heads = row_heads(OP, "query", q.width, head_dim)?;
    let num_kv_heads = row_heads(OP, "key", k.width, head_dim)?;
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "`{OP}` lands one output row per query row"
    );
    debug_assert_eq!(
        k.rows, v.rows,
        "`{OP}` reads one value row per key row of the patch window"
    );
    let images = images_of(OP, segments)?;
    let rows = nonzero(OP, "the patch rows this attention answers", q.rows)?;
    let lanes = num_q_heads.checked_mul(THREADS).ok_or_else(|| {
        refuse(
            OP,
            format!("the grid will not launch: {num_q_heads} query heads"),
        )
    })?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([lanes, rows, 1], [THREADS, 1, 1])),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            o.arg_mut(),
            segments.arg(),
            images.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
        ],
    )
}

/// Full attention inside each segment of a packed fire: a query row of
/// segment `s` attends the rows `kv_indptr[s]..kv_indptr[s + 1]` of the key
/// and value rectangles. Both CSRs are padded with their last entry, so a
/// segment past the live ones is empty.
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
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ragged";
    nonzero(OP, "the head width", head_dim)?;
    let Some(at) = stamp_for(head_dim) else {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head is wider than any stamp"),
        ));
    };
    let entry = match q.dtype {
        Dtype::Bf16 => ["ragged_bf16_d_64", "ragged_bf16_d_128", "ragged_bf16_d_256"][at],
        Dtype::F32 => ["ragged_f32_d_64", "ragged_f32_d_128", "ragged_f32_d_256"][at],
        other => {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: other,
            });
        }
    };
    let q_heads = row_heads(OP, "query", q.width, head_dim)?;
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
    let rows = nonzero(OP, "query rows", q.rows)?;
    // A bf16 head of 64 or 128 takes the tiled point: 32 query rows a
    // workgroup, the keys streamed through workgroup memory.
    let (file, entry, grid) = match (q.dtype, head_dim) {
        (Dtype::Bf16, 64 | 128) => (
            TILED_FILE,
            if head_dim == 64 {
                "tiled_ragged_bf16_d_64"
            } else {
                "tiled_ragged_bf16_d_128"
            },
            Grid::of(
                [q_heads * TILED_THREADS, rows.div_ceil(TILED_ROWS), 1],
                [TILED_THREADS, 1, 1],
            ),
        ),
        _ => (
            FILE,
            entry,
            Grid::of([q_heads * THREADS, rows, 1], [THREADS, 1, 1]),
        ),
    };
    ctx.fire(
        Fire::at(file, entry).apply(grid),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            o.arg_mut(),
            q_indptr.arg(),
            kv_indptr.arg(),
            stated(OP, segments)?.arg(),
            stated(OP, q_heads)?.arg(),
            stated(OP, kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
        ],
    )
}
