//! The GPT-OSS lineage.

#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

/// gpt-oss's forward pass.
///
/// Written in `model-compiler`'s tracing eDSL: ordinary Rust that runs at
/// model-load time with the checkpoint's facts in hand and records what one
/// pass computes. The traced form is what a driver executes.
#[cfg(feature = "forward")]
pub mod forward;
