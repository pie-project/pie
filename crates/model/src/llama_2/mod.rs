//! Llama 2 — the `[INST] <<SYS>>` chat template, and nothing else. Listed in
//! [`crate::catalog`]'s `NO_ROWS_YET`, so nothing here is reachable yet.
//!
//! There is no `import` here, and that is the point of the module it left
//! for. A naming table belongs to whoever PUBLISHES the names, and this
//! generation publishes none — the `llama` architecture's table now sits in
//! [`crate::shared::llama_like::import`], beside the five generations whose
//! rows it actually feeds.

#[cfg(feature = "chat")]
pub mod chat;
