//! The Qwen lineage.
//!
//! `chat/` carries the ChatML implementation and its ancestors. `QwenInstruct`
//! is the most-reused template in the tree — nemotron_h, glm_moe_dsa and the
//! unknown-architecture fallback all instantiate it with their own
//! `ChatMLConfig` — so the registry rows pointing here far outnumber the files.
//!
//! Qwen3.5 (the GDN hybrid) is a separate family directory for its *contract*
//! when that aspect lands, but speaks this lineage's chat format; that reuse is
//! a registry row, not a copy.

#[cfg(feature = "chat")]
pub mod chat;
