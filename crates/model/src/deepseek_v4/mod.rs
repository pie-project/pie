//! The DeepSeek-V4 lineage (hypercompressed streams).
//!
//! Chat: speaks R1 — a registry row in `instruct::create` points
//! `deepseek_v4` at the r1 implementation.

#[cfg(feature = "contract")]
pub mod contract;

/// The declared forward — hyper-connections over compressed attention and
/// an MoE stack.
#[cfg(feature = "forward")]
pub mod forward;
