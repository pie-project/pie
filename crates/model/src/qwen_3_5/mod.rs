//! The Qwen3.5 lineage: the GDN hybrid (full attention interleaved with
//! Gated DeltaNet), dense and MoE.
//!
//! Chat: speaks the qwen3 lineage's ChatML — registry rows in
//! `instruct::create` point every `qwen3_5*` model type there.

#[cfg(feature = "contract")]
pub mod contract;

/// The hybrid's forward pass: GDN layers, full-attention layers, MoE or
/// dense MLP, composed by a static layer schedule.
///
/// Written in `model-compiler`'s tracing eDSL: ordinary Rust that runs at
/// model-load time with the checkpoint's facts in hand and records what one
/// pass computes. The traced form is what a driver executes.
#[cfg(feature = "forward")]
pub mod forward;
