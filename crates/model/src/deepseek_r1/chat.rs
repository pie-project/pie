//! DeepSeek's conversation format, re-exported.
//!
//! The words moved to [`crate::shared::deepseek`] the day a second
//! generation bound them: `deepseek_v4`'s row answers the chat question
//! itself now, and a row reaching into a sibling generation for its
//! answer is exactly the edge `tests/sibling_isolation.rs` forbids.
//!
//! This spelling is kept for the DIRECTORY's sake and not for a
//! caller's. It used to say it stays "because it is the one every
//! existing caller uses", and there are no callers: `deepseek_v4` names
//! `shared::deepseek` directly, and this generation has no row, so
//! nothing can reach this path at all. What it marks is that R1's own
//! geometry is still untranscribed — see [`crate::catalog`]'s
//! `NO_ROWS_YET` — and that when a row lands, its template is already
//! written and already tested next door.

pub use crate::shared::deepseek::*;
