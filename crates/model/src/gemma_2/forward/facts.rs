//! gemma-2's per-backend binding facts.
//!
//! The SHAPE moved to `../spec.rs` (ungated: a catalog row is written in
//! it, and a row must exist under every aspect). Nothing stayed behind:
//! gemma-2 binds no per-backend question at all — no fused bank to ask
//! about, no padded head dim, no TP width the trace reads — so this file
//! is the re-export that keeps `gemma_2::forward::facts::Gemma2Facts`
//! meaning what it has always meant for the twenty callers that spell it
//! that way.

/// The shape, re-exported so a declaration reaches its facts from the
/// same path its trace lives on.
pub use super::super::spec::{Gemma2AttnFacts, Gemma2Facts};
