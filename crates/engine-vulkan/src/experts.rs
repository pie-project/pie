//! The shared routed experts ([`engine_gpu::experts`]) bound to this API.
pub use engine_gpu::experts::*;

pub type HostTier = engine_gpu::experts::HostTier<crate::device::Api>;
pub type Seats = engine_gpu::experts::Seats<crate::device::Api>;
pub type Gathered = engine_gpu::experts::Gathered<crate::device::Api>;
