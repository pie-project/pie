//! Chat-template templating + parsing.
//!
//! Two halves:
//!
//! 1. **Fillers** ([`system`], [`user`], [`assistant`], [`cue`], [`seal`])
//!    produce token sequences for the model's chat template. Higher-level
//!    prompt-buffering facades call them through their `system` /
//!    `user` / `cue` / `seal` methods; for inferlets that build prompts
//!    by hand, these are the public entry points.
//!
//! 2. **Decoder** ([`Decoder`], [`Event`]) parses the model's generated
//!    tokens back into visible text + structural events.
//!
//! Both halves wrap the host's `pie:instruct/chat` interface — chat
//! template knowledge lives in the Pie runtime, not in the SDK.

use crate::Result;
use crate::pie::inferlet::chat::{Decoder as RawDecoder, Event as RawEvent};

// =============================================================================
// Template fillers
// =============================================================================

/// Token sequence for a system-role message.
pub fn system(message: &str) -> Vec<u32> {
    crate::pie::inferlet::chat::system(message)
}

/// Token sequence for a user-role message.
pub fn user(message: &str) -> Vec<u32> {
    crate::pie::inferlet::chat::user(message)
}

/// Token sequence for the first user-role message in a fresh chat.
pub fn first_user(message: &str) -> Vec<u32> {
    crate::pie::inferlet::chat::first_user(message)
}

/// Token sequence for a system message followed by the first user message.
pub fn system_user(system: &str, user: &str) -> Vec<u32> {
    crate::pie::inferlet::chat::system_user(system, user)
}

/// Token sequence for an assistant-role message (history replay).
pub fn assistant(message: &str) -> Vec<u32> {
    crate::pie::inferlet::chat::assistant(message)
}

/// Token sequence for the generation cue (tells the model "your turn").
pub fn cue() -> Vec<u32> {
    crate::pie::inferlet::chat::cue()
}

/// Token sequence that seals the current turn (inserts a stop token).
pub fn seal() -> Vec<u32> {
    crate::pie::inferlet::chat::seal()
}

/// Stop-token IDs for the bound model's chat template — add to a decode
/// loop's stop-token set for explicit termination control.
pub fn stop_tokens() -> Vec<u32> {
    crate::pie::inferlet::chat::stop_tokens()
}

// =============================================================================
// Decoder
// =============================================================================

/// One unit of chat-decoded output.
///
/// Per `feed`, exactly one event fires. `Idle` is the no-op signal —
/// the batch was consumed but didn't cross a semantic boundary worth
/// surfacing (e.g. the chunk landed on a token whose visible text is
/// empty, or inside a region this decoder doesn't report on like a
/// reasoning block).
#[derive(Clone, Debug)]
pub enum Event {
    /// No semantic boundary crossed in this batch.
    Idle,
    /// Streamed text chunk (post-detokenization, post-template-strip).
    /// Always non-empty.
    Delta(String),
    /// End-of-turn reached — the string is the full accumulated text
    /// since the last `reset()`.
    Done(String),
    /// The model emitted a special / control token that the chat
    /// template recognized but didn't lower to visible text. The id is
    /// surfaced raw so the caller can decide what to do.
    ///
    /// Common cases this fires for:
    /// - Tool-call boundary markers (e.g. `<|tool_call|>` in some
    ///   templates) — useful as an early-stop hint when you don't have
    ///   `tools::Decoder` attached.
    /// - Custom control tokens injected by fine-tuned models.
    /// - Format markers (turn boundaries, role separators) the host
    ///   template chose to expose rather than swallow.
    ///
    /// Most callers ignore this branch. If you need fine-grained control
    /// over a specific marker, match on the token id.
    Interrupt(u32),
}

/// Stateful chat decoder. Feed token batches in order; events come back
/// per call. `reset()` returns the decoder to its initial state.
pub struct Decoder {
    inner: RawDecoder,
}

impl Decoder {
    /// Construct a decoder for the bound model's chat template.
    pub fn new() -> Self {
        Self {
            inner: crate::pie::inferlet::chat::create_decoder(),
        }
    }

    /// Feed a token batch and get back the event that fired (one per
    /// call). Returns [`Event::Idle`] when nothing semantically happened
    /// — e.g. the batch landed on a token whose visible text is empty,
    /// or inside a region this decoder doesn't report on.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Event> {
        match self.inner.feed(tokens)? {
            // Empty delta means the batch consumed tokens that produced
            // no visible character — surface as Idle, not Delta("").
            RawEvent::Delta(s) if s.is_empty() => Ok(Event::Idle),
            RawEvent::Delta(s) => Ok(Event::Delta(s)),
            RawEvent::Done(s) => Ok(Event::Done(s)),
            RawEvent::Interrupt(t) => Ok(Event::Interrupt(t)),
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}
