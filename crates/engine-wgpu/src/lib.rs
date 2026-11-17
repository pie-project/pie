pub mod api;
pub mod boot;
pub mod chip;
pub mod device;
pub mod encode;
pub mod guest;
mod landing;
pub mod program;
pub mod rates;
pub mod serve;
pub mod tune;
pub mod tuning;

/// The fire a served wgpu load is asked, `(lanes, tokens)`: the ladder's
/// upper bound, whose rungs the rates step times.
pub const FIRE: (u32, u32) = (512, 10240);

pub use boot::open;
#[cfg(feature = "wgpu")]
pub use boot::{open_with_device, request_device};

/// The fusions this backend implements: none, so the compiler forms no
/// fused op here.
pub static FUSIONS: [poem_compiler::fuse::Rule; 0] = [];
