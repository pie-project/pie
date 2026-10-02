//! The shared core of the API-compatible inferlets.
//!
//! Every wire format pie serves — OpenAI chat completions, Anthropic messages,
//! and the rest — is the same generation with a different envelope. This crate
//! owns the part that is the same:
//!
//! - [`Request`]: one normalized chat request (messages, tools, constraints,
//!   sampling), which each adapter inferlet parses its wire format into.
//! - [`prompt`]: the request rendered to tokens through the host's chat
//!   template (`inferlet::chat` and `inferlet::tools`).
//! - [`generate`]: the ETA pipeline — chunked prefill, a device-carried decode
//!   loop, and an epilogue sampler with temperature, top-p, top-k, the OpenAI
//!   penalties, and an optional grammar mask.
//! - [`demux`]: the generated token stream classified by the host's chat,
//!   reasoning and tool decoders into [`Event`]s an adapter formats.
//! - [`envelope`] and [`emit`]: the contract with the gateway — how the HTTP
//!   request arrives as this program's input, and how SSE frames and the
//!   final body go back.
//!
//! An adapter is then a translation in both directions and nothing else.

pub mod demux;
pub mod emit;
pub mod envelope;
pub mod error;
pub mod generate;
pub mod holdback;
pub mod prompt;
pub mod request;
pub mod run;
mod speculative;

pub use demux::Event;
pub use envelope::Envelope;
pub use error::ApiError;
pub use generate::Sampling;
pub use request::{Message, Request, ResponseFormat, Role, Thinking, ToolCall, ToolChoice};
pub use run::{Finish, Outcome, run};

/// A short id unique to this process: the instance id's leading characters.
/// Every response id an adapter mints is built on it.
pub fn short_id() -> String {
    inferlet::runtime::instance_id()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(12)
        .collect()
}

/// The id of the `index`-th tool call this process made, so streamed chunks
/// and the final body name the same call the same way.
pub fn call_id(index: usize) -> String {
    format!("call_{}_{index}", short_id())
}
