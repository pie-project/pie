//! The readers every format shares: plain text up to a stop token, and a
//! thinking block delimited by markers.

use std::sync::Arc;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::{ChatDecoder, ChatEvent, ReasoningDecoder, ReasoningEvent, ToolDecoder, ToolEvent};

/// Accumulates generated text and closes the turn on any stop token.
///
/// A batch may carry the stop token in the middle of itself. What follows it
/// is the next turn's opening, and it used to be discarded; here the reader
/// closes, resets, and keeps going through the same batch.
pub struct GenericChatDecoder {
    decoder: TokenizerDecoder,
    stop_ids: Vec<u32>,
    text: String,
}

impl GenericChatDecoder {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>, stop_ids: Vec<u32>) -> Self {
        Self {
            decoder: tokenizer.decoder(false),
            stop_ids,
            text: String::new(),
        }
    }
}

impl ChatDecoder for GenericChatDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ChatEvent> {
        let mut events = Vec::new();
        let mut rest = tokens;
        loop {
            let stop = rest.iter().position(|token| self.stop_ids.contains(token));
            let Some(stop) = stop else {
                let delta = self.decoder.feed(rest);
                self.text.push_str(&delta);
                if events.is_empty() || !delta.is_empty() {
                    events.push(ChatEvent::Delta(delta));
                }
                return events;
            };
            let delta = self.decoder.feed(&rest[..stop]);
            self.text.push_str(&delta);
            self.text.push_str(&self.decoder.finish());
            self.decoder.reset();
            events.push(ChatEvent::Done(std::mem::take(&mut self.text)));
            rest = &rest[stop + 1..];
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.text.clear();
    }
}

/// Watches for a thinking block: a marker sequence that opens it, one token
/// that closes it.
///
/// The close is a single token by construction, and the constructor says so.
/// A multi-token close needs a matcher that can roll back a partial match, and
/// the one that stood here rolled back by re-feeding the marker's own ids into
/// the content stream — which is only ever right when every prefix of the
/// marker decodes to itself. No format in the catalog needs it: every one of
/// them closes its thinking with one token from the vocabulary.
pub struct ThinkingDecoder {
    decoder: TokenizerDecoder,
    open: Vec<u32>,
    close: u32,
    inside: bool,
    text: String,
    matched: usize,
}

impl ThinkingDecoder {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>, open: Vec<u32>, close: u32) -> Self {
        assert!(
            !open.is_empty(),
            "a thinking block that opens with nothing is a block that never opens"
        );
        Self {
            decoder: tokenizer.decoder(false),
            open,
            close,
            inside: false,
            text: String::new(),
            matched: 0,
        }
    }
}

impl ReasoningDecoder for ThinkingDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ReasoningEvent> {
        let mut events = Vec::new();
        let mut content: Vec<u32> = Vec::new();
        for &token in tokens {
            if self.inside {
                if token == self.close {
                    let delta = self.decoder.feed(&content);
                    content.clear();
                    self.text.push_str(&delta);
                    self.text.push_str(&self.decoder.finish());
                    self.decoder.reset();
                    self.inside = false;
                    events.push(ReasoningEvent::Complete(std::mem::take(&mut self.text)));
                } else {
                    content.push(token);
                }
            } else if token == self.open[self.matched] {
                self.matched += 1;
                if self.matched == self.open.len() {
                    self.matched = 0;
                    self.inside = true;
                    self.decoder.reset();
                    self.text.clear();
                    events.push(ReasoningEvent::Start);
                }
            } else {
                self.matched = 0;
            }
        }
        let delta = self.decoder.feed(&content);
        self.text.push_str(&delta);
        if events.is_empty() || !delta.is_empty() {
            events.push(ReasoningEvent::Delta(delta));
        }
        events
    }

    fn reset(&mut self) {
        self.inside = false;
        self.decoder.reset();
        self.text.clear();
        self.matched = 0;
    }
}

/// For a format whose model does not think out loud.
pub struct NoopReasoningDecoder;

impl ReasoningDecoder for NoopReasoningDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> Vec<ReasoningEvent> {
        vec![ReasoningEvent::Delta(String::new())]
    }

    fn reset(&mut self) {}
}

/// For a format with no tool grammar to detect.
pub struct NoopToolDecoder;

impl ToolDecoder for NoopToolDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> Vec<ToolEvent> {
        Vec::new()
    }

    fn reset(&mut self) {}
}
