//! Attn over a voxel grid: no HIP kernel yet, so it refuses.

use crate::tensor::Tensor;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    #[default]
    Lane,
    Frames(u32),
}

crate::error::unsupported! {
    #[allow(clippy::too_many_arguments)]
    attention = "attn.voxel" (q: Tensor, k: Tensor, v: Tensor, grid: Tensor,
        segment: Segment, sm_scale: f32, o: &mut Tensor);
}
