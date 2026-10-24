//! `elemwise.hc_project` from `kernels/elemwise/hc_project.slang`: the
//! hyper-connection mix logits, an f32 stream row through an f32 plane.

use super::hc::MAX_HC_MULT;
use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const HC_PROJECT: &str = "elemwise/hc_project.slang";

/// The workgroup width the instantiation is built at.
const BLOCK: u32 = 256;

pub fn project(
    ctx: &Ctx<'_>,
    normed: Tensor,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_project";
    debug_assert!(
        normed.dtype == dtype::Dtype::F32
            && hc_fn.dtype == dtype::Dtype::F32
            && mixes.dtype == dtype::Dtype::F32,
        "`{OP}` projects an f32 row through an f32 plane into an f32 row"
    );
    debug_assert_eq!(
        mixes.rows, normed.rows,
        "the projection lands one mix row per stream row"
    );
    let fan = nonzero(OP, "the stream row this projection contracts", normed.width)?;
    if hc_fn.width != fan {
        return Err(refuse(
            OP,
            format!(
                "the dynamic plane contracts {} and the stream row is {fan} wide",
                hc_fn.width
            ),
        ));
    }
    if stream_count == 0 || stream_count > MAX_HC_MULT {
        return Err(refuse(
            OP,
            format!(
                "the stream count is {stream_count}, not one of the {MAX_HC_MULT} the \
                 mixers unroll"
            ),
        ));
    }

    let mix_hc = hc_fn.rows;
    let layer_row = 2 * stream_count + stream_count * stream_count;
    if mixes.width != mix_hc || (mix_hc != layer_row && mix_hc != stream_count) {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide and a trunk collapse row \
                 is {stream_count}; the plane lands {} rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", mixes.rows)?;
    let lanes = rows
        .checked_mul(mix_hc)
        .and_then(|p| p.checked_mul(BLOCK))
        .ok_or_else(|| {
            refuse(
                OP,
                format!("the grid will not launch: {rows} rows x {mix_hc}"),
            )
        })?;
    ctx.fire(
        Fire::at(HC_PROJECT, "elemwise_hc_project_f32")
            .apply(Grid::of([lanes, 1, 1], [BLOCK, 1, 1])),
        &[
            normed.arg(),
            hc_fn.arg(),
            mixes.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, mix_hc)?.arg(),
        ],
    )
}
