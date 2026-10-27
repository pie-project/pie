//! The widths and constants wan_2's package declares.

pub const PATCH_T: u32 = 1;
pub const PATCH_H: u32 = 2;
pub const PATCH_W: u32 = 2;
pub const PATCH_VOL: u32 = PATCH_T * PATCH_H * PATCH_W;

pub const ROPE_THETA: f32 = 10_000.0;

pub const TRAIN_STEPS: u32 = 1000;
pub const SHIFT_TI2V: f32 = 5.0;

pub const TE_HEAD_DIM: u32 = 64;
pub const TE_LAYERS: u32 = 24;
pub const TE_MAX_TOKENS: u32 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub ffn: u32,
    pub layers: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    pub text_dim: u32,
    pub freq_dim: u32,
}

impl Dims {
    #[must_use]
    pub const fn ti2v_5b() -> Dims {
        Dims {
            dim: 3072,
            heads: 24,
            head_dim: 128,
            ffn: 14_336,
            layers: 30,
            in_channels: 48,
            out_channels: 48,
            text_dim: 4096,
            freq_dim: 256,
        }
    }

    #[must_use]
    pub const fn mini_d128() -> Dims {
        Dims {
            dim: 256,
            heads: 2,
            head_dim: 128,
            ffn: 512,
            layers: 2,
            in_channels: 16,
            out_channels: 16,
            text_dim: 64,
            freq_dim: 256,
        }
    }

    #[must_use]
    pub const fn mini_nano() -> Dims {
        Dims {
            dim: 48,
            heads: 2,
            head_dim: 24,
            ffn: 128,
            layers: 2,
            in_channels: 16,
            out_channels: 16,
            text_dim: 64,
            freq_dim: 32,
        }
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let hw = 2 * (self.head_dim / 6);
        [self.head_dim - 2 * hw, hw, hw, 0]
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub const fn patch_out(&self) -> u32 {
        self.out_channels * PATCH_VOL
    }
}
