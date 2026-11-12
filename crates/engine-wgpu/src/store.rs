//! The shared stores ([`engine_gpu::store`]) bound to this API.
pub use engine_gpu::store::*;

pub type Pools = engine_gpu::store::Pools<crate::device::Api>;

pub mod rows {
    pub use engine_gpu::store::rows::*;

    pub type RowCaches = engine_gpu::store::rows::RowCaches<crate::device::Api>;
}

pub mod experts {
    pub use engine_gpu::store::experts::*;

    pub type ExpertCaches = engine_gpu::store::experts::ExpertCaches<crate::device::Api>;
    pub type Staged = engine_gpu::store::experts::Staged<crate::device::Api>;
    pub type Cut = engine_gpu::store::experts::Cut<crate::device::Api>;
}
