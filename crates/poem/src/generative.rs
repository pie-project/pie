//! What a generative model states of itself: the latent space it denoises
//! in and the schedule it was trained on.

#[derive(Debug, Clone, PartialEq)]
pub struct Generative {
    pub latent: Option<LatentSpace>,
    pub schedule: Option<ScheduleFact>,
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
    /// Each `(value, shift)`: the shift a lane whose `stream` switch takes
    /// `value` advances at.
    pub stream_shifts: Vec<(String, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleKind {
    Flow,
    Epsilon,
    V,
}
