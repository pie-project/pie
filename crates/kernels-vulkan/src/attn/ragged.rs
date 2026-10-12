use dtype::Dtype;

use super::dense::{row_heads, stamp_for};
use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const TILED_FILE: &str = "attn/ragged.slang";

const TILED_THREADS: u32 = 128;

const TILED_ROWS: u32 = 32;

const SCALAR_FILE: &str = "attn/dense.slang";

const SCALAR_THREADS: u32 = 128;

/// A query row of segment `s` attends key rows `kv_indptr[s]..kv_indptr[s + 1]`.
#[allow(clippy::too_many_arguments)]
pub fn forward(
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
    let (file, entry, grid) = match (q.dtype, head_dim) {
        (Dtype::Bf16, 64) => (TILED_FILE, "ragged_tiled_bf16_d_64", tiled(q_heads, rows)),
        (Dtype::Bf16, 128) => (TILED_FILE, "ragged_tiled_bf16_d_128", tiled(q_heads, rows)),
        (dtype, _) => {
            let Some(at) = stamp_for(head_dim) else {
                return Err(refuse(
                    OP,
                    format!("a {head_dim}-wide head is wider than any stamp"),
                ));
            };
            let entry = match dtype {
                Dtype::Bf16 => ["ragged_bf16_d_64", "ragged_bf16_d_128", "ragged_bf16_d_256"][at],
                Dtype::F32 => ["ragged_f32_d_64", "ragged_f32_d_128", "ragged_f32_d_256"][at],
                other => {
                    return Err(Error::DtypeUnsupported {
                        op: OP,
                        dtype: other,
                    });
                }
            };
            (
                SCALAR_FILE,
                entry,
                Grid::of([q_heads * SCALAR_THREADS, rows, 1], [SCALAR_THREADS, 1, 1]),
            )
        }
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

fn tiled(q_heads: u32, rows: u32) -> Grid {
    Grid::of(
        [q_heads * TILED_THREADS, rows.div_ceil(TILED_ROWS), 1],
        [TILED_THREADS, 1, 1],
    )
}
