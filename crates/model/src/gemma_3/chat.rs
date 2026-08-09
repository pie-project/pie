//! Gemma 3's chat template — which is also gemma-3n's, and therefore
//! lives in [`families`](crate::shared).
//!
//! The implementation moved to `shared/gemma_chat.rs` unchanged. It had
//! to: gemma-3n binds this same template, and a generation naming a
//! sibling is the one thing `tests/sibling_isolation.rs` forbids —
//! `Gemma3Variant` has carried a `Gemma3n` arm since the day it was
//! written, so the sharing was already real and only the location was
//! wrong.
//!
//! This re-export stays because `gemma_3::chat::Gemma3Instruct` is the
//! path every caller spells, and moving a file is not a reason to make
//! them spell a different one.

pub use crate::shared::gemma_chat::{Gemma3Instruct, Gemma3Variant};
