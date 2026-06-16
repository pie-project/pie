//! Shared decoder implementations for instruct models.
//!
//! Provides reusable `ChatDecoder`, `ReasoningDecoder`, and no-op stubs
//! so individual model files don't duplicate the same logic.

use crate::model::instruct::{
    ChatDecoder, ChatEvent, ReasoningDecoder, ReasoningEvent, ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;
use std::sync::Arc;

// ─── GenericChatDecoder ──────────────────────────────────────

/// Chat decoder that accumulates tokens, emits incremental text deltas,
/// and stops on any of the given token IDs.
///
/// Decoding strategy: SentencePiece-based tokenizers (Phi-3, Llama-1/2,
/// some Mistral variants) encode word-leading whitespace as a `▁`
/// prefix that gets re-stripped by the tokenizer's `Strip(start=1)`
/// rule. Calling `decode([single_token])` on each new token therefore
/// loses every inter-token space — output looks like
/// `Onceuponatime`. Avoid that by accumulating *tokens*, decoding the
/// full sequence each fire, and emitting only the suffix not yet seen.
/// Byte-level BPE tokenizers (Qwen, Llama-3, Gemma) already encode
/// spaces in-token, so this code path is a no-op cost on them but
/// keeps the contract uniform.
pub struct GenericChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    token_buf: Vec<u32>,
    text_emitted: usize,
}

impl GenericChatDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>, stop_ids: Vec<u32>) -> Self {
        Self {
            tokenizer,
            stop_ids,
            token_buf: Vec::new(),
            text_emitted: 0,
        }
    }
}

impl ChatDecoder for GenericChatDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent {
        for &t in tokens {
            if self.stop_ids.contains(&t) {
                let full = self.tokenizer.decode(&self.token_buf, false);
                self.token_buf.clear();
                self.text_emitted = 0;
                return ChatEvent::Done(full);
            }
            self.token_buf.push(t);
        }
        let full = self.tokenizer.decode(&self.token_buf, false);
        // Byte-level BPE tokenizers (Qwen, Llama-3, Gemma) split multi-byte
        // characters across tokens. When only part of a character has
        // arrived, HF's decode emits trailing U+FFFD replacements; the
        // *next* fire will re-decode the same prefix into the real char,
        // shifting byte offsets. Slicing by raw byte length therefore
        // lands inside a multi-byte char and panics. Hold back any
        // trailing replacements (and stop on the last safe char boundary)
        // until later fires complete them.
        let safe_end = safe_emit_end(&full);
        let delta = if safe_end > self.text_emitted {
            full[self.text_emitted..safe_end].to_string()
        } else {
            String::new()
        };
        self.text_emitted = safe_end;
        ChatEvent::Delta(delta)
    }

    fn reset(&mut self) {
        self.token_buf.clear();
        self.text_emitted = 0;
    }
}

/// Byte length of the longest prefix of `s` that is safe to emit now —
/// i.e., does not end in a U+FFFD replacement char. Returns 0 if every
/// char in `s` is U+FFFD.
fn safe_emit_end(s: &str) -> usize {
    for (i, c) in s.char_indices().rev() {
        if c != '\u{FFFD}' {
            return i + c.len_utf8();
        }
    }
    0
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
    tokenizer: Arc<Tokenizer>,
    start_ids: Vec<u32>,
    end_ids: Vec<u32>,
    inside: bool,
    /// Tokens accumulated while Inside (excluding the closing match).
    /// Re-decoded in full each fire so partial multi-byte chars from a
    /// previous fire resolve when their trailing bytes arrive.
    token_buf: Vec<u32>,
    /// Byte offset into `decode(token_buf)` of text already emitted as
    /// Delta. Maintained at a U+FFFD-free boundary.
    text_emitted: usize,
    match_pos: usize,
    starts_inside: bool,
}

impl ThinkingDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>, start_ids: Vec<u32>, end_ids: Vec<u32>) -> Self {
        let starts_inside = start_ids.is_empty();
        Self {
            tokenizer,
            start_ids,
            end_ids,
            inside: starts_inside,
            token_buf: Vec::new(),
            text_emitted: 0,
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
                        self.token_buf.clear();
                        self.text_emitted = 0;
                        return ReasoningEvent::Start;
                    }
                } else {
                    self.match_pos = 0;
                }
            }
            ReasoningEvent::Delta(String::new())
        } else {
            for &t in tokens {
                if self.match_pos < self.end_ids.len() && t == self.end_ids[self.match_pos] {
                    self.match_pos += 1;
                    if self.match_pos == self.end_ids.len() {
                        // Closing match — flush the full accumulated decode
                        // (mirrors GenericChatDecoder::Done; any trailing
                        // U+FFFD that never resolved are surfaced here, the
                        // standard outcome on truncated multi-byte input).
                        let full = self.tokenizer.decode(&self.token_buf, false);
                        self.inside = false;
                        self.match_pos = 0;
                        self.token_buf.clear();
                        self.text_emitted = 0;
                        return ReasoningEvent::Complete(full);
                    }
                } else {
                    self.match_pos = 0;
                }
                // Tokens that don't complete the end match — including the
                // ones that started a partial match and reset — are content.
                self.token_buf.push(t);
            }
            let full = self.tokenizer.decode(&self.token_buf, false);
            let safe_end = safe_emit_end(&full);
            let delta = if safe_end > self.text_emitted {
                full[self.text_emitted..safe_end].to_string()
            } else {
                String::new()
            };
            self.text_emitted = safe_end;
            ReasoningEvent::Delta(delta)
        }
    }

    fn reset(&mut self) {
        self.inside = self.starts_inside;
        self.token_buf.clear();
        self.text_emitted = 0;
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
    use crate::model::tokenizer::Tokenizer;
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

    // ─── safe_emit_end ───────────────────────────────────────

    #[test]
    fn safe_end_no_replacements() {
        assert_eq!(safe_emit_end("hello"), 5);
        assert_eq!(safe_emit_end(""), 0);
    }

    #[test]
    fn safe_end_holds_back_trailing_replacement() {
        // 'abc' (3 bytes) + U+FFFD (3 bytes) = 6 bytes total; safe end = 3
        let s = "abc\u{FFFD}";
        assert_eq!(s.len(), 6);
        assert_eq!(safe_emit_end(s), 3);
    }

    #[test]
    fn safe_end_holds_back_multiple_trailing_replacements() {
        let s = "x\u{FFFD}\u{FFFD}\u{FFFD}";
        assert_eq!(safe_emit_end(s), 1);
    }

    #[test]
    fn safe_end_keeps_internal_replacement() {
        // Replacement in the middle is committed; only trailing ones held.
        let s = "a\u{FFFD}b";
        assert_eq!(safe_emit_end(s), s.len());
    }

    #[test]
    fn safe_end_all_replacements_returns_zero() {
        let s = "\u{FFFD}\u{FFFD}";
        assert_eq!(safe_emit_end(s), 0);
    }

    #[test]
    fn safe_end_with_multibyte_char_at_end() {
        let s = "ab\u{1F9E0}"; // 🧠 — 4 bytes
        assert_eq!(s.len(), 6);
        assert_eq!(safe_emit_end(s), 6);
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
