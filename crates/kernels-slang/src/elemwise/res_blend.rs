//! `elemwise.res_blend` from `kernels/elemwise/res_blend.slang`: a row the
//! softmax-weighted sum of its candidate blocks and its prefix, one workgroup
//! a row.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const RES_BLEND: &str = "elemwise/res_blend.slang";

/// The workgroup width the instantiation is built at.
const GROUP: u32 = 256;

/// The most candidate blocks the shader's softmax scratch holds.
pub const MAX_BLEND_BLOCKS: usize = 32;

/// `y` blended from `prefix` and `blocks`, which the caller lays as stacked
/// planes (`blocks[0]` the first, each `y.rows x y.width` after it).
pub fn res_blend(
    ctx: &Ctx<'_>,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.res_blend";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "elemwise_res_blend_bf16" });
    if blocks.len() > MAX_BLEND_BLOCKS {
        return Err(refuse(
            OP,
            format!(
                "{} candidate blocks exceed the shader's softmax scratch bound of \
                 {MAX_BLEND_BLOCKS}",
                blocks.len()
            ),
        ));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    let hidden = nonzero(OP, "the blended row's width", y.width)?;
    if !hidden.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {hidden}-wide row is odd; rows land two values a word"),
        ));
    }
    let n_blocks = u32::try_from(blocks.len()).unwrap_or(u32::MAX);
    let lanes = rows
        .checked_mul(GROUP)
        .ok_or_else(|| refuse(OP, format!("the grid will not launch: {rows} rows")))?;
    let stack = blocks.first().copied().unwrap_or(prefix);
    ctx.fire(
        Fire::at(RES_BLEND, entry).apply(Grid::of([lanes, 1, 1], [GROUP, 1, 1])),
        &[
            prefix.arg(),
            stack.arg(),
            weight.arg(),
            proj.arg(),
            y.arg_mut(),
            stated(OP, n_blocks)?.arg(),
            stated(OP, hidden)?.arg(),
            stated(OP, rows)?.arg(),
            eps.arg(),
        ],
    )
}
