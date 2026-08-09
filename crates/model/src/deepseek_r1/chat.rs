//! DeepSeek's conversation format, re-exported.
//!
//! The words moved to [`crate::shared::deepseek`] the day a second
//! generation bound them: `deepseek_v4`'s row answers the chat question
//! itself now, and a row reaching into a sibling generation for its
//! answer is exactly the edge `tests/sibling_isolation.rs` forbids. This
//! spelling stays because it is the one every existing caller uses.

pub use crate::shared::deepseek::*;
