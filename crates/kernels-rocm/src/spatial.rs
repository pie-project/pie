//! The spatial (voxel grid) ops: no HIP kernels yet, so every one refuses.

use crate::tensor::Tensor;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    #[default]
    Lane,
    Frames(u32),
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridRule {
    Conv {
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        pad_back: [u32; 3],
        causal_t: bool,
    },
    Upsample {
        factor: [u32; 3],
        keep_first_frame: bool,
    },
    Shuffle {
        r: [u32; 3],
        trim_t: u32,
    },
    Unshuffle {
        r: [u32; 3],
    },
    AvgDown {
        factor: [u32; 3],
    },
}

crate::error::unsupported! {
    #[allow(clippy::too_many_arguments)]
    attention = "spatial.attention" (q: Tensor, k: Tensor, v: Tensor, grid: Tensor,
        segment: Segment, sm_scale: f32, o: &mut Tensor);
    cache_gather = "spatial.cache_gather" (slab: Tensor, slot_ids: Tensor, grid: Tensor,
        frames: u32, cache: &mut Tensor);
    #[allow(clippy::too_many_arguments)]
    cache_store = "spatial.cache_store" (x: Tensor, cache: Tensor, slot_ids: Tensor,
        grid: Tensor, frames: u32, slab: &mut Tensor);
    #[allow(clippy::too_many_arguments)]
    conv3d = "spatial.conv3d" (x: Tensor, grid: Tensor, w: Tensor, bias: Option<Tensor>,
        conv: Conv3d, cache: Option<Tensor>, o: &mut Tensor, o_grid: Tensor);
    conv_weight_taps_major = "spatial.conv_weight_taps_major" (src: Tensor, c_in: u32,
        taps: u32, dst: &mut Tensor);
    #[allow(clippy::too_many_arguments)]
    group_norm = "spatial.group_norm" (x: Tensor, grid: Tensor, groups: u32, weight: Tensor,
        bias: Tensor, eps: f32, silu: bool, o: &mut Tensor);
    upsample_nearest = "spatial.upsample_nearest" (x: Tensor, grid: Tensor, factor: [u32; 3],
        keep_first_frame: bool, o: &mut Tensor, o_grid: Tensor);
    pixel_shuffle = "spatial.pixel_shuffle" (x: Tensor, grid: Tensor, r: [u32; 3], trim_t: u32,
        o: &mut Tensor, o_grid: Tensor);
    pixel_unshuffle = "spatial.pixel_unshuffle" (x: Tensor, grid: Tensor, r: [u32; 3],
        o: &mut Tensor, o_grid: Tensor);
    avg_down = "spatial.avg_down" (x: Tensor, grid: Tensor, r: [u32; 3], group: u32,
        o: &mut Tensor, o_grid: Tensor);
    patchify = "spatial.patchify" (x: Tensor, grid: Tensor, p: [u32; 3], o: &mut Tensor,
        o_grid: Tensor);
    unpatchify = "spatial.unpatchify" (x: Tensor, grid: Tensor, p: [u32; 3], o: &mut Tensor,
        o_grid: Tensor);
    derive_grid = "spatial.derive_grid" (grid: Tensor, rule: GridRule, o_grid: &mut Tensor);
}
