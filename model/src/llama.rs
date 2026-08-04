//! The Llama lineage.
//!
//! One directory per family, one file per aspect: `chat/` holds the instruct
//! templates (per release — Llama 2 and Llama 3 speak different formats),
//! `contract.rs` will hold the weight contract when it migrates out of the
//! drivers, `forward.rs` the PTIR forward pass after that.
//!
//! The aspects partition differently on purpose: phi3 and mistral3 reuse this
//! lineage's *contract* (the storage schema is the same) while shipping their
//! own *templates*. The registry rows in `instruct::create` and, later, the
//! contract registry state those N:1 mappings explicitly — a directory here is
//! organization for humans, never the key the machine dispatches on.

pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;
