//! The Cerebras engine: the model forward runs as a resident wafer program on a CS system (or
//! the fabric simulator), compiled from the model's trace by kernels-cerebras and driven through
//! the SDK runtime binding in [`sdk`]; guest programs run in the host interpreter
//! ([`program`]). See [`spatial`].

pub mod api;
pub mod boot;
pub mod program;
pub mod sdk;
pub mod spatial;

pub use api::ContractFor;
pub use boot::open_engine;
pub use spatial::Config;

/// The fusions this backend implements: none, so the compiler forms no
/// fused op here.
pub static FUSIONS: [poem_compiler::fuse::Rule; 0] = [];
