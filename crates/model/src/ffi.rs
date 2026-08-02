//! The forward FFI boundary, shaped like `model-loader-capi`'s.
//!
//! `types` is the published `#[repr(C)]` vocabulary — the only `#[repr(C)]`
//! code in this crate — `arena` turns a traced [`ForwardPlan`](model_compiler::trace::ForwardPlan)
//! into it, and `entry` holds the `extern "C"` functions the driver calls.
//! The generated header (`crates/model/include/pie_forward.h`) is the C view
//! of exactly those three files.
//!
//! This is the only boundary the design has, on the loader's precedent
//! (`loader/src/ffi.rs`, `loader/architecture.md` §10): there is no
//! serialized form crossing here, because a plan is traced and executed in
//! one process, so a JSON round-trip would be a second representation to
//! keep in step with no reader on the far end. (The serde impls on
//! `model_compiler::trace` serve the Rust-side goldens, not this boundary.)

pub mod arena;
pub mod entry;
pub mod types;

pub use entry::{
    PieForwardLlamaLikeFacts, PieForwardQwen35FullAttnFacts, PieForwardQwen35GdnFacts,
    PieForwardQwen35HybridFacts, PieForwardQwen35MoeMlpFacts, PieForwardStatus,
    pie_forward_lower, pie_forward_release, pie_forward_trace_llama_like, pie_forward_trace_qwen3_5_full_attn,
    pie_forward_trace_qwen3_5_gdn, pie_forward_trace_qwen3_5_hybrid,
    pie_forward_trace_qwen3_5_moe_mlp,
};
pub use types::*;

/// The tracer's fingerprint, stamped into every plan header.
///
/// The FNV-1a content hash of `crates/model-compiler/src/**.rs`, computed by
/// THAT crate's build script exactly the way `model-loader`'s fingerprints
/// the load-plan compiler: (this hash, facts) identifies a traced form, so a
/// consumer can key a cache or a golden on
/// `PieForwardPlan::compiler_version` and have it auto-invalidate when the
/// tracer changes.
///
/// Read across the crate boundary rather than recomputed. `env!` only sees
/// the environment of the crate being compiled, and hashing this crate's
/// `src/` would fold chat templates into a number that claims to be about
/// the tracer.
pub fn compiler_version() -> u64 {
    model_compiler::compiler_version()
}

#[cfg(test)]
mod tests;
