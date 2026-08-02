//! Gemma 2.
//!
//! One generation, one directory. What this holds is what this
//! generation implements; what it shares with a sibling is a row in
//! `contract::HF_ROWS` or an arm of `instruct::create` naming the
//! generation that owns the implementation.

#[cfg(feature = "chat")]
pub mod chat;

/// The declared forward — plain attention with a norm PAIR per block, an
/// alternating sliding window, and softcaps.
#[cfg(feature = "forward")]
pub mod forward;
