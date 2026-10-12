//! The widths and constants MiniMax H3's package declares.

use poem::Stream;

pub const LATENT_CHANNELS: u32 = 24;
pub const AUDIO_CHANNELS: u32 = 32;
pub const VIDEO_FEATURES: u32 = LATENT_CHANNELS * 2 * 2;
pub const HEAD_DIM: u32 = 128;
pub const ROPE_THETA: f32 = 10_000.0;
pub const ADALN_SLICES: u32 = 6;
pub const MODALITIES: u32 = 3;
pub const FINAL_SLICES: u32 = 2;
pub const TIMESTEP_SLOTS: u32 = 4;
pub const VIDEO_SHIFT: f32 = 12.0;
pub const AUDIO_SHIFT: f32 = 3.0;
pub const STEPS: u32 = 50;
pub const TE_HIDDEN: u32 = 5120;
pub const TE_VOCAB: u32 = 151_936;
pub const TE_Q_HEADS: u32 = 64;
pub const TE_KV_HEADS: u32 = 8;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_INTER: u32 = 25_600;
pub const TE_DEPTH: u32 = 64;
pub const TE_LAYERS: u32 = 50;

pub mod port {
    pub const LATENTS: u8 = 0;
    pub const REFERENCE: u8 = 1;
    pub const AUDIO: u8 = 2;
    pub const CONTEXT: u8 = 3;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
}

#[must_use]
pub const fn modality(stream: Stream) -> usize {
    match stream {
        Stream::Video | Stream::Reference | Stream::Image => 0,
        Stream::Audio => 2,
        Stream::Text | Stream::Context => 1,
    }
}

#[must_use]
pub const fn timestep_slot(stream: Stream) -> u32 {
    match stream {
        Stream::Video | Stream::Text | Stream::Context => 0,
        Stream::Reference | Stream::Image => 1,
        Stream::Audio => 2,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub blocks: u32,
    pub refiners: u32,
    pub text_dim: u32,
    pub t_freq: u32,
    pub t_hidden: u32,
    pub t_dim: u32,
    pub rope_freqs: u32,
}

impl Dims {
    #[must_use]
    pub const fn h3() -> Dims {
        Dims {
            dim: 5376,
            heads: 56,
            head_dim: HEAD_DIM,
            inter: 14336,
            blocks: 50,
            refiners: 2,
            text_dim: TE_HIDDEN,
            t_freq: 256,
            t_hidden: 5376,
            t_dim: 2688,
            rope_freqs: 16,
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            dim: 128,
            heads: 2,
            head_dim: 64,
            inter: 256,
            blocks: 2,
            refiners: 1,
            text_dim: 64,
            t_freq: 32,
            t_hidden: 128,
            t_dim: 64,
            rope_freqs: 8,
        }
    }

    #[must_use]
    pub const fn inner(&self) -> u32 {
        self.heads * self.head_dim
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let per = 2 * self.rope_freqs;
        [per, per, per, 0]
    }

    #[must_use]
    pub const fn rotary_dim(&self) -> u32 {
        6 * self.rope_freqs
    }

    #[must_use]
    pub const fn adaln_width(&self) -> u32 {
        ADALN_SLICES * self.dim
    }
}
