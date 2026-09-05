#[cfg(feature = "vulkan")]
pub mod alloc;
#[cfg(feature = "vulkan")]
pub mod ctx;
#[cfg(feature = "vulkan")]
pub mod handles;
#[cfg(feature = "vulkan")]
pub mod pipelines;
#[cfg(feature = "vulkan")]
pub mod spirv;

#[cfg(feature = "vulkan")]
pub use alloc::{Buffer, FileWriter, Memory};
#[cfg(feature = "vulkan")]
pub use ctx::{Context, Enabled, Frame, Kept, Pending, present, reservations};
#[cfg(feature = "vulkan")]
pub use handles::{Binding, Handles, NIL};
#[cfg(feature = "vulkan")]
pub use pipelines::{Pipeline, Pipelines};

#[cfg(not(feature = "vulkan"))]
mod stub;

#[cfg(not(feature = "vulkan"))]
pub mod ctx {
    pub use super::stub::{Context, Enabled, Frame, Kept, Pending, present, reservations};
}
#[cfg(not(feature = "vulkan"))]
pub mod handles {
    pub use super::stub::{Binding, Handles, NIL};
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
    Binding, Buffer, Context, Enabled, FileWriter, Frame, Handles, Kept, Memory, NIL, Pending,
    Pipeline, Pipelines, present, reservations,
};
