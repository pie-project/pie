//! The shared weights ([`engine_gpu::weights`]) bound to this API.
pub use engine_gpu::weights::*;

pub type Weights = engine_gpu::weights::Weights<crate::device::Api>;
