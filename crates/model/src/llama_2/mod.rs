//! Llama 2.
//!
//! One generation, one directory. What this holds is what this
//! generation implements: the `[INST] <<SYS>>` chat template, and
//! nothing else.
//!
//! # No row, and what that means
//!
//! This generation is in [`crate::catalog`]'s `NO_ROWS_YET` list, so
//! nothing here is reachable — the only door into a generation is a row
//! id. The template is kept rather than deleted because it is the
//! expensive half and it is tested; a row is five numbers and a
//! `manifest()`.
//!
//! The paragraph that used to sit here said what a generation shares
//! with a sibling is *"a row in `contract::HF_ROWS` or an arm of
//! `instruct::create`"*. Both of those are gone: `HF_ROWS` was one of
//! the three tables the catalog collapsed, and `instruct::create` no
//! longer matches on anything — it asks [`crate::catalog::find`] for a
//! row and calls `row.chat`. Sharing is a `use` of [`crate::shared`]
//! now, and a generation that reaches a sibling instead fails
//! `tests/sibling_isolation.rs`.

#[cfg(feature = "chat")]
pub mod chat;
