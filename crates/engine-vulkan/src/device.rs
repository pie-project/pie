#[cfg(feature = "vulkan")]
pub mod alloc;
#[cfg(feature = "vulkan")]
pub mod ctx;
#[cfg(feature = "vulkan")]
pub mod pipelines;
#[cfg(feature = "vulkan")]
pub mod spirv;

#[cfg(feature = "vulkan")]
pub use alloc::{Buffer, FileWriter, Memory};
#[cfg(feature = "vulkan")]
pub use ctx::{Context, Enabled, Frame, Kept, Pending, present, reservations};
#[cfg(feature = "vulkan")]
pub use pipelines::{Pipeline, Pipelines};

#[cfg(not(feature = "vulkan"))]
mod stub;

#[cfg(not(feature = "vulkan"))]
pub mod ctx {
    pub use super::stub::{Context, Enabled, Frame, Kept, Pending, present, reservations};
}
#[cfg(not(feature = "vulkan"))]
pub mod alloc {
    pub use super::stub::{Buffer, FileWriter, Memory};
}
#[cfg(not(feature = "vulkan"))]
pub mod pipelines {
    pub use super::stub::{Pipeline, Pipelines};
}

#[cfg(not(feature = "vulkan"))]
pub use stub::{
    Buffer, Context, Enabled, FileWriter, Frame, Kept, Memory, Pending, Pipeline, Pipelines,
    present, reservations,
};

/// The shared engine's view of this API.
#[derive(Debug)]
pub enum Api {}

mod gpu;

pub type Handles = engine_gpu::Handles<Api>;
pub type Binding = engine_gpu::Binding<Api>;
pub use engine_gpu::NIL;

pub mod handles {
    pub use super::{Binding, Handles};
    pub use engine_gpu::NIL;
}
