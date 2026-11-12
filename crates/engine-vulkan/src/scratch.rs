//! The shared scratch planes ([`engine_gpu::scratch`]) bound to this API.
pub use engine_gpu::scratch::*;

pub type Scratch = engine_gpu::scratch::Scratch<crate::device::Api>;
