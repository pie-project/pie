pub mod api;
pub mod boot;
pub mod chip;
pub mod device;
pub mod encode;
pub mod guest;
mod landing;
pub mod program;
pub mod serve;
pub mod tuning;

pub use boot::open;
#[cfg(feature = "wgpu")]
pub use boot::{open_with_device, request_device};

/// The fusions this backend implements: none, so the compiler forms no
/// fused op here.
pub static FUSIONS: [poem_compiler::fuse::Rule; 0] = [];
