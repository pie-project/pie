//! The widths and constants ltx_2's package declares.

pub const VAE_Z: u32 = 128;

pub const VAE_RGB: u32 = 3;
pub const VAE_PATCH: u32 = 4;
pub const VAE_EPS: f32 = 1e-8;
pub const VAE_MID_RESNETS: u32 = 2;
pub const VAE_UP_RESNETS: [u32; 4] = [2, 4, 6, 4];

pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_AXES: u8 = 3;
pub const AUDIO_ROPE_AXES: u8 = 1;

pub const GATE_SCALE: f32 = 2.0;

pub const TRAIN_STEPS: u32 = 1000;
pub const DISTILLED_SIGMAS: [f32; 8] = [
    1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875,
];

pub const TEXT_LAYERS: u32 = 49;

pub mod port {
    pub const LATENTS: u8 = 0;
    pub const CONTEXT: u8 = 0;
    pub const AUDIO_CONTEXT: u8 = 1;
    pub const TEXT: u8 = 1;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
    pub const TIME_POSITIONS: u8 = 1;
    pub const VOXELS: u8 = 0;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub layers: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub audio_heads: u32,
    pub audio_head_dim: u32,
    pub channels: u32,
    pub cross_dim: u32,
    pub audio_cross_dim: u32,
    pub ff_mult: u32,
    pub caption: u32,
    pub conn_layers: u32,
}

impl Dims {
    #[must_use]
    pub const fn ltx_2_5() -> Dims {
        Dims {
            layers: 48,
            heads: 32,
            head_dim: 128,
            audio_heads: 32,
            audio_head_dim: 64,
            channels: 128,
            cross_dim: 4096,
            audio_cross_dim: 2048,
            ff_mult: 4,
            caption: 3840,
            conn_layers: 8,
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            layers: 2,
            heads: 2,
            head_dim: 128,
            audio_heads: 2,
            audio_head_dim: 64,
            channels: 128,
            cross_dim: 256,
            audio_cross_dim: 128,
            ff_mult: 4,
            caption: 16,
            conn_layers: 1,
        }
    }

    #[must_use]
    pub const fn dim(&self) -> u32 {
        self.heads * self.head_dim
    }

    #[must_use]
    pub const fn audio_dim(&self) -> u32 {
        self.audio_heads * self.audio_head_dim
    }

    #[must_use]
    pub const fn av_inner(&self) -> u32 {
        self.audio_heads * self.audio_head_dim
    }

    #[must_use]
    pub const fn text_in(&self) -> u32 {
        self.caption * TEXT_LAYERS
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let f = self.dim() / (2 * ROPE_AXES as u32);
        [2 * f, 2 * f, 2 * f, 0]
    }

    #[must_use]
    pub const fn audio_rope_dims(&self) -> [u32; 4] {
        [self.audio_dim(), 0, 0, 0]
    }

    #[must_use]
    pub const fn av_rope_dims(&self) -> [u32; 4] {
        [self.av_inner(), 0, 0, 0]
    }
}

#[must_use]
pub const fn rope_pad(dim: u32, axes: u8) -> u32 {
    let axes = axes as u32;
    dim / 2 - axes * (dim / (2 * axes))
}

pub const DENOISE: u8 = 0;
pub const REFINE_VIDEO: u8 = 1;
pub const REFINE_AUDIO: u8 = 2;
pub const VAE_DECODE: u8 = 3;
