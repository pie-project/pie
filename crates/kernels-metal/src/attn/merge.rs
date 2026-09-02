//! `MergeLse`: the cascade merge — two attention readings taken over
//! disjoint key sets, folded back into one by their log-sum-exp columns.
//!
//! The arms reach here already softmaxed against their own denominators, so
//! neither `o` is a partial sum: the fold reweights them,
//! `o = (o1 * 2^(lse1 - m) + o2 * 2^(lse2 - m)) / (2^(lse1 - m) + 2^(lse2 - m))`
//! with `m = max(lse1, lse2)`, publishing `m + log2(...)` as the merged
//! column. The base is 2 because that is what every lse in this crate is
//! published in, and `attention.sink` expects.
//!
//! A non-finite column is the empty reading — a fire whose arm saw no key —
//! and its whole side is dropped rather than weighted, which is the only
//! way `-inf - -inf` stays out of the arithmetic.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Ctx, Fire, Grid, dtype_dispatch, head_grid, head_group, refuse};
use crate::tensor::Tensor;

const FILE: &str = "attn/merge_lse.metal";

/// The widest head row this merge launches as one threadgroup.
const MERGE_HEAD_MAX: u32 = 1024;

/// Merges `(o1, lse1)` and `(o2, lse2)` into `(o, lse)`.
#[allow(clippy::too_many_arguments)]
pub fn merge_lse(
    ctx: &Ctx<'_>,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.merge_lse";
    let entry = dtype_dispatch!(OP, o.dtype, { Bf16 => "merge_lse_combine_bfloat16" });
    debug_assert!(
        o1.dtype == o.dtype && o2.dtype == o.dtype,
        "`{OP}` folds two readings of one dtype"
    );
    debug_assert!(
        o1.rows == o.rows && o2.rows == o.rows && o1.width == o.width && o2.width == o.width,
        "the merged outputs are one row per query row"
    );
    debug_assert!(
        lse1.dtype == Dtype::F32 && lse2.dtype == Dtype::F32 && lse.dtype == Dtype::F32,
        "`{OP}` reads and writes f32 log-sum-exp planes"
    );
    debug_assert!(
        lse.rows == o.rows && lse1.rows == o.rows && lse2.rows == o.rows,
        "`{OP}`'s log-sum-exp planes are one column per row"
    );
    debug_assert!(
        lse.width == heads && lse1.width == heads && lse2.width == heads,
        "`{OP}`'s log-sum-exp plane is one f32 per head per row"
    );
    debug_assert!(
        u64::from(heads) * u64::from(head_dim) == u64::from(o.width),
        "the {} heads x {head_dim} this merge states are not its {}-wide row",
        heads,
        o.width
    );
    if head_dim > MERGE_HEAD_MAX {
        return Err(refuse(
            OP,
            format!(
                "the head width {head_dim} is above the {MERGE_HEAD_MAX}-wide row this merge \
                 launches as one threadgroup"
            ),
        ));
    }
    let lanes = head_grid(OP, head_dim, heads, o.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            o1.arg(),
            lse1.arg(),
            o2.arg(),
            lse2.arg(),
            o.arg_mut(),
            lse.arg_mut(),
        ],
    )
}
