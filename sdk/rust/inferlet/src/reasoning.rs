//! Reasoning / thinking-block decoder.
//!
//! Wraps the host's `pie:instruct/reasoning.decoder`. Emits `Start` when
//! the model enters a thinking block, `Delta` for each chunk of reasoning
//! text, and `End` when the block closes (with the full accumulated
//! reasoning text).
//!
//! Compose with [`chat::Decoder`](crate::chat::Decoder) by feeding
//! the same token batch to both — the reasoning decoder's events are
//! independent of chat's (no implicit suppression).

use crate::Result;
use crate::pie::inferlet::reasoning::{Decoder as RawDecoder, Event as RawEvent};

/// One unit of reasoning-decoded output.
///
/// Per `feed`, exactly one event fires. `Idle` is the no-op signal —
/// the batch landed outside any reasoning block, or inside one but on
/// tokens that produced no visible reasoning text.
#[derive(Clone, Debug)]
pub enum Event {
    /// No reasoning boundary crossed in this batch.
    Idle,
    /// The model entered a reasoning block. No text yet.
    Start,
    /// Streamed chunk of reasoning text (post-detokenization). Always
    /// non-empty.
    Delta(String),
    /// The block closed — the string is the full accumulated reasoning
    /// text from `Start` to `End`.
    End(String),
}

/// Stateful reasoning decoder.
pub struct Decoder {
    inner: RawDecoder,
}

impl Decoder {
    /// Construct a decoder for the bound model's reasoning template.
    pub fn new() -> Self {
        Self {
            inner: crate::pie::inferlet::reasoning::create_decoder(),
        }
    }

    /// Feed a token batch and get back the event that fired (one per
    /// call). Returns [`Event::Idle`] when the batch landed outside any
    /// reasoning block, or inside one with no visible reasoning text.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Event> {
        match self.inner.feed(tokens)? {
            RawEvent::Start => Ok(Event::Start),
            RawEvent::Delta(s) if s.is_empty() => Ok(Event::Idle),
            RawEvent::Delta(s) => Ok(Event::Delta(s)),
            RawEvent::Complete(s) => Ok(Event::End(s)),
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}
