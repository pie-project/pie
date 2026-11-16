pub mod api;
pub mod boot;
pub mod chip;
pub mod comm;
pub mod device;
pub mod encode;
pub mod guest;
pub mod program;
pub mod rates;
pub(crate) mod record;
pub mod serve;
pub mod tune;

/// The fire a served Vulkan load is asked, `(lanes, tokens)`: the ladder's
/// upper bound, whose rungs `pie tune` times.
pub const FIRE: (u32, u32) = (512, 10240);
pub mod tuning;

pub use boot::open;

/// The fusions this backend implements: none, so the compiler forms no
/// fused op here.
pub static FUSIONS: [poem_compiler::fuse::Rule; 0] = [];
