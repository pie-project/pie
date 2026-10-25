//! What a generative model states of the readings its rows run: how its
//! positions are laid out and what it reads out; the
//! latent space it denoises in and the schedule it was trained on; and, for a
//! text diffusion model, the canvas it denoises.

use crate::Stream;

#[derive(Debug, Clone, PartialEq)]
pub struct Generative {
    pub readings: Vec<ReadingFact>,
    pub latent: Option<LatentSpace>,
    pub schedule: Option<ScheduleFact>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadingFact {
    pub name: String,
    pub index: u8,
    pub has_kv: bool,
    pub takes_tokens: bool,
    pub streams: Vec<Stream>,
    pub positions: Option<PositionConvention>,
    pub readout: ReadoutKind,
    pub readout_width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisRole {
    Time,
    Height,
    Width,
    Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionConvention {
    pub axes: Vec<AxisRole>,
    pub text_axis: u32,
    pub text_origin: u32,
    pub image_follows_text: bool,
    pub reference_stride: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadoutKind {
    Logits,
    Velocity,
    Hidden,
    Pixels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentSpace {
    pub channels: u32,
    pub patch_t: u32,
    pub patch_h: u32,
    pub patch_w: u32,
    pub spatial_compression: u32,
    pub temporal_compression: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleFact {
    pub kind: ScheduleKind,
    pub shift: f32,
    pub train_steps: u32,
    pub boundary: Option<f32>,
    pub pinned_sigmas: Vec<f32>,
    pub stream_shifts: Vec<(Stream, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleKind {
    Flow,
    Epsilon,
    V,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Diffusion {
    pub canvas: u32,
    pub hidden: u32,
}
