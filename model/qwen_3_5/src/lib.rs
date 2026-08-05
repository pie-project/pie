//! The Qwen3.5 lineage: the GDN hybrid (full attention interleaved with
//! Gated DeltaNet), dense and MoE.
//!
//! Chat: speaks the qwen3 lineage's ChatML — registry rows in
//! `instruct::create` point every `qwen3_5*` model type there.

#[cfg(feature = "contract")]
pub mod contract;
