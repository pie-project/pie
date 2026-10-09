//! The widths and constants Z-Image's package declares, which the tests of
//! its rows, its import and its VAE hold the package to.

pub const CHANNELS: u32 = 16;
pub const PATCH: u32 = 2;
pub const PATCH_FEATURES: u32 = CHANNELS * PATCH * PATCH;
pub const SPATIAL_COMPRESSION: u32 = 8;
pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const ROPE_AXES: u8 = 3;
pub const ROPE_THETA: f32 = 256.0;

pub const TE_HIDDEN: u32 = 2560;
pub const TE_VOCAB: u32 = 151_936;
pub const TE_Q_HEADS: u32 = 32;
pub const TE_KV_HEADS: u32 = 8;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_INTER: u32 = 9728;
pub const TE_THETA: f32 = 1_000_000.0;
pub const TE_DEPTH: u32 = 36;
pub const TE_LAYERS: u32 = TE_DEPTH - 1;

pub mod port {
    pub const PAD_IMAGE: u8 = 0;
    pub const LATENTS: u8 = 1;
    pub const CONTEXT_REFINED: u8 = 2;
    pub const PAD_CAPTION: u8 = 0;
    pub const CAPTION: u8 = 0;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
    pub const VOXELS: u8 = 0;
    pub const PIXEL_VOXELS: u8 = 1;
}

pub mod vae {
    pub const GN_GROUPS: u32 = 32;
    pub const GN_EPS: f32 = 1e-6;
    pub const RGB: u32 = 3;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub joint_layers: u32,
    pub refiner_layers: u32,
    pub cap_width: u32,
    pub rope_dims: [u32; 4],
}

impl Dims {
    #[must_use]
    pub const fn turbo() -> Dims {
        Dims {
            dim: 3840,
            heads: 30,
            head_dim: 128,
            inter: 10_240,
            joint_layers: 30,
            refiner_layers: 2,
            cap_width: TE_HIDDEN,
            rope_dims: [32, 48, 48, 0],
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            dim: 256,
            heads: 4,
            head_dim: 64,
            inter: 682,
            joint_layers: 2,
            refiner_layers: 2,
            cap_width: 64,
            rope_dims: [16, 24, 24, 0],
        }
    }
}
