//! The widths mini-dit's package declares.

pub const HIDDEN: u32 = 256;
pub const HEAD_DIM: u32 = 64;
pub const CHANNELS: u32 = 16;
pub const PATCH: u32 = 2;
pub const PATCH_FEATURES: u32 = CHANNELS * PATCH * PATCH;
pub const TEXT_WIDTH: u32 = 256;
pub const CONTEXT_WIDTH: u32 = 512;
pub const TIMESTEP_DIM: u32 = 256;
pub const TIMESTEP_MAX_PERIOD: f32 = 10_000.0;
pub const TIMESTEP_FLIP_SIN_COS: bool = false;
pub const TIMESTEP_SCALE: f32 = 1.0;
pub const ROPE_DIMS: [u32; 4] = [16, 24, 24, 0];
pub const ROPE_AXES: u8 = 3;
