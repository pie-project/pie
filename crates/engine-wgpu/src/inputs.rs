//! The shared input store ([`engine_gpu::inputs`]) bound to this API.
pub use engine_gpu::inputs::*;

pub type Inputs = engine_gpu::inputs::Inputs<crate::device::Api>;
