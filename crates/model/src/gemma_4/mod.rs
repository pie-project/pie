//! Gemma 4.
//!
//! One generation, one directory. What this holds is what this
//! generation implements; what it shares with a sibling is a row in
//! `contract::HF_ROWS` or an arm of `instruct::create` naming the
//! generation that owns the implementation.

#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

/// Gemma 4's forward pass.
///
/// Written in `model-compiler`'s tracing eDSL: ordinary Rust that runs at
/// model-load time with the checkpoint's facts in hand and records what one
/// pass computes. The traced form is what a driver executes.
#[cfg(feature = "forward")]
pub mod forward;
