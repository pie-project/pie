//! Kimi K2/K2.5 instruct implementation.
//!
//! The words moved to [`crate::families::kimi`] the day a SECOND
//! generation bound them as its own answer rather than as a cell in
//! `instruct::create`'s table — kimi-k3 speaks this template, and a row
//! there cannot name a sibling. This is the re-export, so every existing
//! spelling still resolves.

pub use crate::families::kimi::KimiInstruct;
