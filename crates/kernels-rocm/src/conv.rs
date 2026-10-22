//! The convolutions and resamplings over a voxel grid: no HIP kernels yet, so
//! every one refuses.

use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimePad {
    Zero,
    Replicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv3d {
    pub k: [u32; 3],
    pub stride: [u32; 3],
    pub pad: [u32; 3],
    pub pad_back: [u32; 3],
    pub causal_t: bool,
    pub time_pad: TimePad,
}

impl Conv3d {
    #[must_use]
    pub const fn taps(&self) -> u32 {
        self.k[0] * self.k[1] * self.k[2]
    }
}

crate::error::unsupported! {
    cache_gather = "conv.cache_gather" (slab: Tensor, slot_ids: Tensor, grid: Tensor,
        frames: u32, cache: &mut Tensor);
    #[allow(clippy::too_many_arguments)]
    cache_store = "conv.cache_store" (x: Tensor, cache: Tensor, slot_ids: Tensor,
        grid: Tensor, frames: u32, slab: &mut Tensor);
    #[allow(clippy::too_many_arguments)]
    conv3d = "conv.conv3d" (x: Tensor, grid: Tensor, w: Tensor, bias: Option<Tensor>,
        conv: Conv3d, cache: Option<Tensor>, o: &mut Tensor, o_grid: Tensor);
    conv_weight_taps_major = "conv.conv_weight_taps_major" (src: Tensor, c_in: u32,
        taps: u32, dst: &mut Tensor);
    upsample_nearest = "conv.upsample" (x: Tensor, grid: Tensor, factor: [u32; 3],
        keep_first_frame: bool, o: &mut Tensor, o_grid: Tensor);
    pixel_shuffle = "conv.pixel_shuffle" (x: Tensor, grid: Tensor, r: [u32; 3], trim_t: u32,
        o: &mut Tensor, o_grid: Tensor);
    pixel_unshuffle = "conv.pixel_unshuffle" (x: Tensor, grid: Tensor, r: [u32; 3],
        o: &mut Tensor, o_grid: Tensor);
    avg_down = "conv.avg_down" (x: Tensor, grid: Tensor, r: [u32; 3], group: u32,
        o: &mut Tensor, o_grid: Tensor);
    patchify = "conv.patchify" (x: Tensor, grid: Tensor, p: [u32; 3], o: &mut Tensor,
        o_grid: Tensor);
    unpatchify = "conv.unpatchify" (x: Tensor, grid: Tensor, p: [u32; 3], o: &mut Tensor,
        o_grid: Tensor);
}
