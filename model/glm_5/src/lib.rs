//! The GLM-5 lineage (MLA + DSA indexer + routed/shared MoE).
//!
//! Chat: speaks ChatML — a registry row in `instruct::create` points
//! `glm_moe_dsa` at the qwen3 implementation with its own stop tokens.

#[cfg(feature = "contract")]
pub mod contract;
