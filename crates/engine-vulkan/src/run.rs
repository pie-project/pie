//! The shared run ([`engine_gpu::run`]) bound to this API.
pub use engine_gpu::run::*;

pub type Run<'c> = engine_gpu::run::Run<'c, crate::device::Api>;
