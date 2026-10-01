pub mod alloc;
pub mod ctx;
pub mod elastic;
pub mod handles;
pub mod library;
#[cfg(target_vendor = "apple")]
pub mod sparse;

pub use alloc::Buffer;
pub use ctx::{Context, Pending, present, reservations};
pub use elastic::Arena as ElasticArena;
#[cfg(target_vendor = "apple")]
pub use elastic::Elastic;
pub use handles::{Binding, Handles};
pub use library::Pipelines;
#[cfg(target_vendor = "apple")]
pub use sparse::Sparse;
