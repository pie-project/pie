#[cfg(feature = "wgpu")]
pub mod alloc;
#[cfg(feature = "wgpu")]
pub mod ctx;
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
pub mod host;
#[cfg(feature = "wgpu")]
pub mod pipelines;

#[cfg(feature = "wgpu")]
pub use alloc::{Buffer, FileWriter, Memory};
#[cfg(feature = "wgpu")]
pub use ctx::{
    Context, Enabled, Frame, Handed, OnDone, Pending, present, request, request_device,
    reservations, wanted_features,
};
#[cfg(feature = "wgpu")]
pub use pipelines::{Pipeline, Pipelines, bind_traffic};

#[cfg(not(feature = "wgpu"))]
mod stub;

#[cfg(not(feature = "wgpu"))]
pub mod ctx {
    pub use super::stub::{
        Context, Enabled, Frame, Handed, OnDone, Pending, present, reservations,
    };
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
    Buffer, Context, Enabled, FileWriter, Frame, Handed, Memory, OnDone, Pending, Pipeline,
    Pipelines, bind_traffic, present, reservations,
};

/// The shared engine's view of this API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Api;

mod gpu;

pub type Handles = engine_gpu::Handles<Api>;
pub type Binding = engine_gpu::Binding<Api>;
pub use engine_gpu::NIL;

pub mod handles {
    pub use super::{Binding, Handles};
    pub use engine_gpu::NIL;
}
