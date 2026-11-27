pub mod alloc;
pub mod ctx;
pub mod elastic;
pub mod handles;
#[cfg(target_vendor = "apple")]
mod iokit;
pub mod library;
mod models;
#[cfg(target_vendor = "apple")]
pub mod sparse;

pub use alloc::Buffer;
#[cfg(feature = "fault-injection")]
pub use ctx::fail_next_frame;
pub use ctx::{Context, CoresFrom, Pending, RESIDENT_PER_CORE, present, reservations};
#[cfg(all(target_vendor = "apple", feature = "fault-injection"))]
pub use ctx::{hold_next_frame, release_held_frame};
pub use elastic::Arena as ElasticArena;
#[cfg(target_vendor = "apple")]
pub use elastic::Elastic;
pub use handles::{Binding, Handles};
pub use library::Pipelines;
#[cfg(target_vendor = "apple")]
pub use sparse::Sparse;
