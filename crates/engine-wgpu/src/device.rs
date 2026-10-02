#[cfg(feature = "wgpu")]
pub mod alloc;
#[cfg(feature = "wgpu")]
pub mod ctx;
#[cfg(feature = "wgpu")]
pub mod handles;
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
pub use handles::{Binding, Handles, NIL};
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
    Binding, Buffer, Context, Enabled, FileWriter, Frame, Handed, Handles, Memory, NIL, OnDone,
    Pending, Pipeline, Pipelines, bind_traffic, present, reservations,
};
