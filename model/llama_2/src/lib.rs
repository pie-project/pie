//! Llama 2.
//!
//! One generation, one directory. What this holds is what this
//! generation implements; what it shares with a sibling is a row in
//! `contract::HF_ROWS` or an arm of `instruct::create` naming the
//! generation that owns the implementation.

#[cfg(feature = "chat")]
pub mod chat;
