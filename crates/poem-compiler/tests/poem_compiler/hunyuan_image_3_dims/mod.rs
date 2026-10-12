//! The widths and constants HunyuanImage 3's package declares.

pub const TRAIN_STEPS: u32 = 1000;
pub const FLOW_SHIFT: f32 = 3.0;
pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_AXES: u8 = 2;
pub const T_FREQ_DIM: u32 = 256;
pub const LATENT_CHANNELS: u32 = 32;
pub const SPATIAL_COMPRESSION: u32 = 16;

pub const ENCODE: u8 = 0;
pub const DENOISE: u8 = 1;
pub const IMAGE_IN: u8 = 2;
pub const IMAGE_OUT: u8 = 3;

#[must_use]
pub fn rope_x_scale(head_dim: u32) -> f32 {
    ROPE_THETA.powf(-2.0 / head_dim as f32)
}

pub mod port {
    pub const ROWS: u8 = 0;
    pub const SPECIAL: u8 = 1;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
    pub const LATENT_VOXELS: u8 = 0;
    pub const ROW_VOXELS: u8 = 1;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub experts: u32,
    pub top_k: u32,
    pub moe_inter: u32,
    pub shared_inter: u32,
    pub head_hidden: u32,
}

impl Dims {
    #[must_use]
    pub const fn flagship() -> Dims {
        Dims {
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            vocab: 133_120,
            experts: 64,
            top_k: 8,
            moe_inter: 3072,
            shared_inter: 3072,
            head_hidden: 1024,
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            hidden: 256,
            layers: 2,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            vocab: 133_120,
            experts: 8,
            top_k: 2,
            moe_inter: 256,
            shared_inter: 256,
            head_hidden: 64,
        }
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        [self.head_dim / 2, self.head_dim / 2, 0, 0]
    }
}
