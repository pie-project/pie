use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, head_grid, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "attn/merge_lse.slang";

const GROUP: u32 = 256;

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
    let entry = dtype_dispatch!(OP, o.dtype, { Bf16 => "merge_lse_combine_bf16" });
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
    if u64::from(heads) * u64::from(head_dim) != u64::from(o.width) {
        return Err(refuse(
            OP,
            format!(
                "the {heads} heads x {head_dim} this merge states are not its {}-wide row",
                o.width
            ),
        ));
    }
    let lanes = head_grid(OP, head_dim, heads, o.rows)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, [GROUP, 1, 1])),
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
