use crate::instruct::{
    ChatDecoder, ChatEvent, ReasoningDecoder, ReasoningEvent, ToolDecoder, ToolEvent,
};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

pub fn keep_tail(text: &str, keep: usize) -> &str {
    if text.len() <= keep {
        return text;
    }
    let mut cut = text.len() - keep;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    &text[cut..]
}

pub struct GenericChatDecoder {
    decoder: TokenizerDecoder,
    stop_ids: Vec<u32>,
    text: String,
}

impl GenericChatDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>, stop_ids: Vec<u32>) -> Self {
        Self {
            decoder: tokenizer.decoder(false),
            stop_ids,
            text: String::new(),
        }
    }
}

impl ChatDecoder for GenericChatDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent {
        let stop = tokens
            .iter()
            .position(|token| self.stop_ids.contains(token));
        let content = &tokens[..stop.unwrap_or(tokens.len())];
        let delta = self.decoder.feed(content);
        self.text.push_str(&delta);

        if stop.is_some() {
            self.text.push_str(&self.decoder.finish());
            self.decoder.reset();
            ChatEvent::Done(std::mem::take(&mut self.text))
        } else {
            ChatEvent::Delta(delta)
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.text.clear();
    }
}

pub struct ThinkingDecoder {
    decoder: TokenizerDecoder,
    start_ids: Vec<u32>,
    end_ids: Vec<u32>,
    inside: bool,
    text: String,
    match_pos: usize,
    starts_inside: bool,
}

impl ThinkingDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>, start_ids: Vec<u32>, end_ids: Vec<u32>) -> Self {
        let starts_inside = start_ids.is_empty();
        Self {
            decoder: tokenizer.decoder(false),
            start_ids,
            end_ids,
            inside: starts_inside,
            text: String::new(),
            match_pos: 0,
            starts_inside,
        }
    }
}

impl ReasoningDecoder for ThinkingDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent {
        if !self.inside {
            for &t in tokens {
                if self.match_pos < self.start_ids.len() && t == self.start_ids[self.match_pos] {
                    self.match_pos += 1;
                    if self.match_pos == self.start_ids.len() {
                        self.inside = true;
                        self.match_pos = 0;
                        self.decoder.reset();
                        self.text.clear();
                        return ReasoningEvent::Start;
                    }
                } else {
                    self.match_pos = 0;
                }
            }
            ReasoningEvent::Delta(String::new())
        } else {
            let mut content = Vec::with_capacity(tokens.len());
            for &t in tokens {
                let mut matched = false;
                if self.match_pos < self.end_ids.len() && t == self.end_ids[self.match_pos] {
                    self.match_pos += 1;
                    matched = true;
                } else if self.match_pos > 0 {
                    content.extend_from_slice(&self.end_ids[..self.match_pos]);
                    self.match_pos = 0;
                    if !self.end_ids.is_empty() && t == self.end_ids[0] {
                        self.match_pos = 1;
                        matched = true;
                    }
                }

                if matched {
                    if self.match_pos == self.end_ids.len() {
                        let delta = self.decoder.feed(&content);
                        self.text.push_str(&delta);
                        self.text.push_str(&self.decoder.finish());
                        self.inside = false;
                        self.match_pos = 0;
                        self.decoder.reset();
                        return ReasoningEvent::Complete(std::mem::take(&mut self.text));
                    }
                } else {
                    content.push(t);
                }
            }
            let delta = self.decoder.feed(&content);
            self.text.push_str(&delta);
            ReasoningEvent::Delta(delta)
        }
    }

    fn reset(&mut self) {
        self.inside = self.starts_inside;
        self.decoder.reset();
        self.text.clear();
        self.match_pos = 0;
    }
}

pub struct NoopReasoningDecoder;

impl ReasoningDecoder for NoopReasoningDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> ReasoningEvent {
        ReasoningEvent::Delta(String::new())
    }
    fn reset(&mut self) {}
}

pub struct NoopToolDecoder;

impl ToolDecoder for NoopToolDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> ToolEvent {
        ToolEvent::Start
    }
    fn reset(&mut self) {}
}
