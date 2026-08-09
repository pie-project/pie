//! gemma3n's per-backend binding facts.
//!
//! The SHAPE moved to `../spec.rs` (ungated: a catalog row is written in
//! it, and a row must exist under every aspect). Nothing stayed behind:
//! gemma-3n's trace binds no per-backend question — the AltUp streams,
//! the laurel rank and the per-layer embedding widths are all the
//! model's own — so this file is the re-export that keeps
//! `gemma_3n::forward::facts::Gemma3nFacts` meaning what it has always
//! meant for the callers that spell it that way.

/// The shape, re-exported so a declaration reaches its facts from the
/// same path its trace lives on.
pub use super::super::spec::{Gemma3nAltUpFacts, Gemma3nAttnFacts, Gemma3nFacts};
