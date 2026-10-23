//! What follows an attention read (`attn.merge_lse`, `attn.sink`,
//! `attn.logit_softcap`), from `kernels/attn/merge.slang`.

use dtype::Dtype;

use super::{pairs, row_heads};
use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, head_grid, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const MERGE: &str = "attn/merge.slang";

/// The most workgroups one grid axis launches.
const MAX_GROUPS_PER_AXIS: u32 = 65535;

/// `o, lse` the two readings `(o1, lse1)`, `(o2, lse2)` of the same rows
/// folded by their base-2 log-sum-exps.
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
    const OP: &str = "attn.merge_lse";
    let entry = dtype_dispatch!(OP, o.dtype, { Bf16 => "attn_merge_lse_bf16" });
    debug_assert!(
        o1.dtype == o.dtype && o2.dtype == o.dtype,
        "`{OP}` folds two readings of one dtype"
    );
    debug_assert!(
        lse1.dtype == Dtype::F32 && lse2.dtype == Dtype::F32 && lse.dtype == Dtype::F32,
        "`{OP}` reads and writes f32 log-sum-exp planes"
    );
    if u64::from(heads) * u64::from(head_dim) != u64::from(o.width) {
        return Err(refuse(
            OP,
            format!(
                "the {heads} heads x {head_dim} this merge states are not its {}-wide row",
                o.width
            ),
        ));
    }
    let lanes = head_grid(OP, pairs(OP, head_dim)?, heads, o.rows)?;
    ctx.fire(
        Fire::at(MERGE, entry).apply(Grid::of(lanes, [256, 1, 1])),
        &[
            o1.arg(),
            lse1.arg(),
            o2.arg(),
            lse2.arg(),
            o.arg_mut(),
            lse.arg_mut(),
            stated(OP, head_dim)?.arg(),
            stated(OP, heads)?.arg(),
        ],
    )
}

/// Each head of `o` rescaled in place by its learned sink's share,
/// `sigmoid(lse ln2 - sink)`.
pub fn sink(
    ctx: &Ctx<'_>,
    o: Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attn.sink";
    let entry = dtype_dispatch!(OP, o.dtype, { Bf16 => "attn_sink_bf16" });
    if sink.dtype != o.dtype {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: sink.dtype,
        });
    }
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 log-sum-exp plane"
    );
    let heads = row_heads(OP, o.width, head_dim)?;
    let lanes = head_grid(OP, pairs(OP, head_dim)?, heads, o.rows)?;
    ctx.fire(
        Fire::at(MERGE, entry).apply(Grid::of(lanes, [64, 1, 1])),
        &[
            o.arg_mut(),
            lse.arg(),
            sink.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, heads)?.arg(),
        ],
    )
}

/// `x = cap * tanh(x / cap)` in place.
pub fn logit_softcap(ctx: &Ctx<'_>, x: Tensor, cap: f32) -> Result<(), Error> {
    const OP: &str = "attn.logit_softcap";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "attn_softcap_bf16" });
    let [n, _, _] = elementwise(OP, x.width, x.rows)?;
    let groups = n.div_ceil(2).div_ceil(256);
    let fold = groups.div_ceil(MAX_GROUPS_PER_AXIS);
    let row_groups = groups.div_ceil(fold);
    ctx.fire(
        Fire::at(MERGE, entry)
            .groups([row_groups, fold, 1])
            .group([256, 1, 1]),
        &[
            x.arg_mut(),
            cap.arg(),
            stated(OP, n)?.arg(),
            row_groups.arg(),
        ],
    )
}
