//! pie:instruct — Chat, tool-use, and reasoning capability interfaces
//!
//! These are *exported* by inferlets, not imported. The runtime calls into
//! inferlet-provided implementations of chat, tool-use, and reasoning.

pub mod chat;
pub mod tool_use;
pub mod reasoning;
