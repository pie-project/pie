//! The Nemotron-H lineage (Mamba2/attention/MoE hybrid).
//!
//! Chat: speaks ChatML with a thinking suffix — a registry row in
//! `instruct::create` points `nemotron_h` at the qwen3 implementation.

#[cfg(feature = "contract")]
pub mod contract;
