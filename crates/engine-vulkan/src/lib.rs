pub mod api;
pub mod boot;
pub mod chip;
pub mod comm;
pub mod device;
pub mod encode;
pub mod guest;
pub mod program;
pub(crate) mod record;
pub mod serve;
pub mod tune;
pub mod tuning;

pub use boot::open;

/// The fusions this backend implements: none, so the compiler forms no
/// fused op here.
pub static FUSIONS: [poem_compiler::fuse::Rule; 0] = [];
