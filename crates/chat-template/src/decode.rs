use std::sync::Arc;
use tokenizer::Specials;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::{ChatDecoder, ChatEvent, ReasoningDecoder, ReasoningEvent, ToolDecoder, ToolEvent};

pub struct GenericChatDecoder {
    decoder: TokenizerDecoder,
    stop_ids: Vec<u32>,
    text: String,
}

impl GenericChatDecoder {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>, stop_ids: Vec<u32>) -> Self {
        Self {
            decoder: tokenizer.decoder(Specials::Keep),
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

pub struct ThinkingDecoder {
    decoder: TokenizerDecoder,
    open: Vec<u32>,
    close: u32,
    starts_inside: bool,
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
            decoder: tokenizer.decoder(Specials::Keep),
            open,
            close,
            starts_inside: false,
            inside: false,
            text: String::new(),
            matched: 0,
        }
    }

    #[must_use]
    pub fn opened(tokenizer: Arc<Tokenizer>, open: Vec<u32>, close: u32) -> Self {
        Self {
            starts_inside: true,
            inside: true,
            ..Self::new(tokenizer, open, close)
        }
    }

    /// Inside a thinking block from the first token when `cue` left one
    /// open (a template that writes `<think>` into its generation cue), and
    /// outside when it opened none or opened and closed one (its no-thinking
    /// cue).
    #[must_use]
    pub fn for_cue(tokenizer: Arc<Tokenizer>, open: Vec<u32>, close: u32, cue: &[u32]) -> Self {
        if opens_block(cue, &open, close) {
            Self::opened(tokenizer, open, close)
        } else {
            Self::new(tokenizer, open, close)
        }
    }
}

fn opens_block(cue: &[u32], open: &[u32], close: u32) -> bool {
    !open.is_empty()
        && cue
            .windows(open.len())
            .rposition(|window| window == open)
            .is_some_and(|at| !cue[at + open.len()..].contains(&close))
}

impl ReasoningDecoder for ThinkingDecoder {
    fn starts_inside(&self) -> bool {
        self.starts_inside
    }

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
        self.inside = self.starts_inside;
        self.decoder.reset();
        self.text.clear();
        self.matched = 0;
    }
}

pub struct NoopReasoningDecoder;

impl ReasoningDecoder for NoopReasoningDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> Vec<ReasoningEvent> {
        vec![ReasoningEvent::Delta(String::new())]
    }

    fn reset(&mut self) {}
}

pub struct NoopToolDecoder;

impl ToolDecoder for NoopToolDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> Vec<ToolEvent> {
        Vec::new()
    }

    fn reset(&mut self) {}
}
