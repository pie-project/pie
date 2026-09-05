#[cfg(feature = "wgpu")]
pub mod alloc;
#[cfg(feature = "wgpu")]
pub mod ctx;
#[cfg(feature = "wgpu")]
pub mod handles;
#[cfg(feature = "wgpu")]
pub mod pipelines;

#[cfg(feature = "wgpu")]
pub use alloc::{Buffer, FileWriter, Memory};
#[cfg(feature = "wgpu")]
pub use ctx::{Context, Enabled, Frame, Pending, present, reservations};
#[cfg(feature = "wgpu")]
pub use handles::{Binding, Handles, NIL};
#[cfg(feature = "wgpu")]
pub use pipelines::{Pipeline, Pipelines, bind_traffic};

#[cfg(not(feature = "wgpu"))]
mod stub;

#[cfg(not(feature = "wgpu"))]
pub mod ctx {
    pub use super::stub::{Context, Enabled, Frame, Pending, present, reservations};
}
#[cfg(not(feature = "wgpu"))]
pub mod handles {
    pub use super::stub::{Binding, Handles, NIL};
}
#[cfg(not(feature = "wgpu"))]
pub mod alloc {
    pub use super::stub::{Buffer, FileWriter, Memory};
}
#[cfg(not(feature = "wgpu"))]
pub mod pipelines {
    pub use super::stub::{Pipeline, Pipelines, bind_traffic};
}

#[cfg(not(feature = "wgpu"))]
pub use stub::{
    Binding, Buffer, Context, Enabled, FileWriter, Frame, Handles, Memory, NIL, Pending, Pipeline,
    Pipelines, bind_traffic, present, reservations,
};
