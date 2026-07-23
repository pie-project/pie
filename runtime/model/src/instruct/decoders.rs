//! Shared decoder implementations for instruct models.
//!
//! Provides reusable `ChatDecoder`, `ReasoningDecoder`, and no-op stubs
//! so individual model files don't duplicate the same logic.

use crate::instruct::{
    ChatDecoder, ChatEvent, ReasoningDecoder, ReasoningEvent, ToolDecoder, ToolEvent,
};
use pie_tokenizer::{Tokenizer, TokenizerDecoder};
use std::sync::Arc;

// ─── GenericChatDecoder ──────────────────────────────────────

/// Chat decoder that accumulates tokens, emits incremental text deltas,
/// and stops on any of the given token IDs.
///
/// Uses the tokenizer's incremental decoder so split UTF-8 and byte-fallback
/// sequences are held until complete without re-decoding prior tokens.
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

// ─── ThinkingDecoder ─────────────────────────────────────────

/// Reasoning decoder that matches start/end token sequences to detect
/// `<think>...</think>` blocks (or equivalent delimiters).
///
/// - If `start_ids` is empty, starts in the Inside state (for models
///   whose `cue()` already includes the think-open tag, e.g. OLMo).
/// - Otherwise starts Outside and transitions to Inside on matching
///   `start_ids`.
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

// ─── No-op Decoders ─────────────────────────────────────────

/// No-op reasoning decoder for models without thinking support.
pub struct NoopReasoningDecoder;

impl ReasoningDecoder for NoopReasoningDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> ReasoningEvent {
        ReasoningEvent::Delta(String::new())
    }
    fn reset(&mut self) {}
}

/// No-op tool decoder for models without tool support.
pub struct NoopToolDecoder;

impl ToolDecoder for NoopToolDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> ToolEvent {
        ToolEvent::Start
    }
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_tokenizer::Tokenizer;
    use std::sync::Arc;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    // ─── GenericChatDecoder ──────────────────────────────────

    #[test]
    fn chat_delta_on_normal_tokens() {
        let tok = make_tok(&["hello", "world"]);
        let mut dec = GenericChatDecoder::new(tok, vec![99]);
        match dec.feed(&[0]) {
            ChatEvent::Delta(s) => assert_eq!(s, "hello"),
            other => panic!("expected Delta, got {:?}", other),
        }
    }

    #[test]
    fn chat_done_on_stop_token() {
        let tok = make_tok(&["hello", "world", "<stop>"]);
        let mut dec = GenericChatDecoder::new(tok, vec![2]);
        dec.feed(&[0]); // "hello"
        dec.feed(&[1]); // "world"
        match dec.feed(&[2]) {
            ChatEvent::Done(s) => assert_eq!(s, "helloworld"),
            other => panic!("expected Done, got {:?}", other),
        }
    }

    #[test]
    fn chat_done_returns_accumulated_then_empty_after_reset() {
        let tok = make_tok(&["hello", "<stop>"]);
        let mut dec = GenericChatDecoder::new(tok, vec![1]);
        dec.feed(&[0]); // accumulate "hello"
        dec.reset();
        match dec.feed(&[1]) {
            ChatEvent::Done(s) => assert_eq!(s, ""), // cleared by reset
            other => panic!("expected Done, got {:?}", other),
        }
    }

    // ─── ThinkingDecoder ─────────────────────────────────────

    #[test]
    fn thinking_outside_ignores_non_start_tokens() {
        let tok = make_tok(&["text", "<think>", "reason", "</think>"]);
        let mut dec = ThinkingDecoder::new(tok, vec![1], vec![3]);
        match dec.feed(&[0]) {
            ReasoningEvent::Delta(s) => assert!(s.is_empty()),
            other => panic!("expected empty Delta while outside, got {:?}", other),
        }
    }

    #[test]
    fn thinking_full_cycle() {
        let tok = make_tok(&["text", "<think>", "reason", "</think>"]);
        let mut dec = ThinkingDecoder::new(tok, vec![1], vec![3]);
        // Enter thinking
        match dec.feed(&[1]) {
            ReasoningEvent::Start => {}
            other => panic!("expected Start, got {:?}", other),
        }
        // Accumulate reasoning
        match dec.feed(&[2]) {
            ReasoningEvent::Delta(s) => assert_eq!(s, "reason"),
            other => panic!("expected Delta, got {:?}", other),
        }
        // Exit thinking
        match dec.feed(&[3]) {
            ReasoningEvent::Complete(s) => assert_eq!(s, "reason"),
            other => panic!("expected Complete, got {:?}", other),
        }
        // Back outside
        match dec.feed(&[0]) {
            ReasoningEvent::Delta(s) => assert!(s.is_empty()),
            other => panic!("expected empty Delta after complete, got {:?}", other),
        }
    }

    #[test]
    fn thinking_multi_token_start_sequence() {
        // start_ids = [1, 2] (two tokens needed to enter thinking)
        let tok = make_tok(&["text", "<", "think>", "reason", "</think>"]);
        let mut dec = ThinkingDecoder::new(tok, vec![1, 2], vec![4]);
        // First start token alone doesn't trigger
        match dec.feed(&[1]) {
            ReasoningEvent::Delta(_) => {}
            other => panic!("expected Delta, got {:?}", other),
        }

        // Second start token completes the sequence
        match dec.feed(&[2]) {
            ReasoningEvent::Start => {}
            other => panic!("expected Start, got {:?}", other),
        }
    }

    #[test]
    fn thinking_multi_token_end_is_not_emitted() {
        let tok = make_tok(&["reason", "</", "think>"]);
        let mut dec = ThinkingDecoder::new(tok, vec![], vec![1, 2]);
        match dec.feed(&[0]) {
            ReasoningEvent::Delta(text) => assert_eq!(text, "reason"),
            other => panic!("expected reasoning delta, got {other:?}"),
        }
        match dec.feed(&[1]) {
            ReasoningEvent::Delta(text) => assert!(text.is_empty()),
            other => panic!("expected held delimiter, got {other:?}"),
        }
        match dec.feed(&[2]) {
            ReasoningEvent::Complete(text) => assert_eq!(text, "reason"),
            other => panic!("expected complete reasoning, got {other:?}"),
        }
    }

    #[test]
    fn thinking_partial_end_mismatch_becomes_content() {
        let tok = make_tok(&["</", "x", "</think>"]);
        let mut dec = ThinkingDecoder::new(tok, vec![], vec![0, 2]);
        match dec.feed(&[0]) {
            ReasoningEvent::Delta(text) => assert!(text.is_empty()),
            other => panic!("expected held delimiter, got {other:?}"),
        }
        match dec.feed(&[1]) {
            ReasoningEvent::Delta(text) => assert_eq!(text, "</x"),
            other => panic!("expected restored content, got {other:?}"),
        }
    }

    #[test]
    fn thinking_starts_inside_when_no_start_ids() {
        let tok = make_tok(&["reason", "</think>"]);
        let mut dec = ThinkingDecoder::new(tok, vec![], vec![1]);
        // Already inside — content produces Delta
        match dec.feed(&[0]) {
            ReasoningEvent::Delta(s) => assert_eq!(s, "reason"),
            other => panic!("expected Delta, got {:?}", other),
        }
        // End token completes
        match dec.feed(&[1]) {
            ReasoningEvent::Complete(s) => assert_eq!(s, "reason"),
            other => panic!("expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn thinking_reset_restores_outside() {
        let tok = make_tok(&["<think>", "text", "</think>"]);
        let mut dec = ThinkingDecoder::new(tok, vec![0], vec![2]);
        dec.feed(&[0]); // enter
        dec.feed(&[1]); // accumulate
        dec.reset();
        // After reset, back outside — content ignored
        match dec.feed(&[1]) {
            ReasoningEvent::Delta(s) => assert!(s.is_empty()),
            other => panic!("expected empty Delta after reset, got {:?}", other),
        }
    }

    #[test]
    fn thinking_reset_restores_inside_when_starts_inside() {
        let tok = make_tok(&["reason", "</think>"]);
        let mut dec = ThinkingDecoder::new(tok, vec![], vec![1]);
        dec.feed(&[0]); // accumulate
        dec.feed(&[1]); // complete
        dec.reset();
        // After reset, back inside
        match dec.feed(&[0]) {
            ReasoningEvent::Delta(s) => assert_eq!(s, "reason"),
            other => panic!("expected Delta, got {:?}", other),
        }
    }

    // ─── No-op Decoders ──────────────────────────────────────

    #[test]
    fn noop_reasoning_always_empty() {
        let mut dec = NoopReasoningDecoder;
        match dec.feed(&[0, 1, 2]) {
            ReasoningEvent::Delta(s) => assert!(s.is_empty()),
            other => panic!("expected empty Delta, got {:?}", other),
        }
    }

    #[test]
    fn noop_tool_always_start() {
        let mut dec = NoopToolDecoder;
        match dec.feed(&[0, 1, 2]) {
            ToolEvent::Start => {}
            other => panic!("expected Start, got {:?}", other),
        }
    }
}
