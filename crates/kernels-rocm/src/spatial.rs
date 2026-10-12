pub mod attn;
pub mod cache;
pub mod conv;
pub mod norm;
pub mod resample;
pub mod rule;

pub use attn::{Segment, attention};
pub use cache::{cache_gather, cache_rows, cache_store};
pub use conv::{Conv3d, ConvPath, TimePad, conv_weight_taps_major, conv3d, conv3d_on};
pub use norm::group_norm;
pub use resample::{
    avg_down, patchify, pixel_shuffle, pixel_unshuffle, unpatchify, upsample_nearest,
};
pub use rule::{GridRule, derive_grid};

use crate::error::Error;
use crate::jit::{count, refuse};
use crate::tensor::Tensor;
use dtype::Dtype;

pub(crate) fn lanes_of(op: &'static str, what: &str, grid: Tensor) -> Result<i32, Error> {
    if grid.dtype != Dtype::I32 || grid.width != 4 {
        return Err(refuse(
            op,
            format!(
                "the {what} grid is {}x{} {:?}; a lane table is `[lanes, 4]` i32 `{{t, h, w, row_offset}}`",
                grid.rows, grid.width, grid.dtype
            ),
        ));
    }
    count(op, "the lane count", grid.rows)
}

pub(crate) fn lane_pair(op: &'static str, grid: Tensor, o_grid: Tensor) -> Result<i32, Error> {
    let lanes = lanes_of(op, "input", grid)?;
    let o_lanes = lanes_of(op, "output", o_grid)?;
    if lanes != o_lanes {
        return Err(refuse(
            op,
            format!("{lanes} input lanes against {o_lanes} output lanes"),
        ));
    }
    Ok(lanes)
}

pub(crate) fn flat_elements(op: &'static str, t: Tensor, block: u32) -> Result<(u32, i64), Error> {
    let n = t.elements();
    let blocks = n.div_ceil(u64::from(block));
    let blocks = u32::try_from(blocks).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    if blocks == 0 {
        return Err(refuse(op, "the element count is zero"));
    }
    let total = i64::try_from(n)
        .map_err(|_| refuse(op, format!("{n} elements overflow the kernel's count")))?;
    Ok((blocks, total))
}
