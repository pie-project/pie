//! The voxel-grid attention, norm and grid rules: no HIP kernels yet, so every one refuses.

use crate::tensor::Tensor;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    #[default]
    Lane,
    Frames(u32),
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
    #[allow(clippy::too_many_arguments)]
    group_norm = "spatial.group_norm" (x: Tensor, grid: Tensor, groups: u32, weight: Tensor,
        bias: Tensor, eps: f32, silu: bool, o: &mut Tensor);
    derive_grid = "spatial.derive_grid" (grid: Tensor, rule: GridRule, o_grid: &mut Tensor);
}
