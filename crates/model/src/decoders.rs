use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::instruct::{ChatDecoder, Reasoning, ReasoningDecoder, ToolCall, ToolDecoder};

pub struct GenericChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    pending: Vec<u32>,
}

impl GenericChatDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>, stop_ids: Vec<u32>) -> Self {
        Self { tokenizer, stop_ids, pending: Vec::new() }
    }
}

impl ChatDecoder for GenericChatDecoder {
    fn push(&mut self, token: u32) -> Option<String> {
        if self.stop_ids.contains(&token) {
            return None;
        }
        self.pending.push(token);
        let text = self.tokenizer.decode(&self.pending, true);
        if text.ends_with('\u{fffd}') {
            None
        } else {
            self.pending.clear();
            Some(text)
        }
    }

    fn finish(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            None
        } else {
            let text = self.tokenizer.decode(&self.pending, true);
            self.pending.clear();
            Some(text)
        }
    }
}

pub struct NoopReasoningDecoder;

impl ReasoningDecoder for NoopReasoningDecoder {
    fn push(&mut self, _token: u32) -> Option<Reasoning> {
        None
    }
}

pub struct NoopToolDecoder;

impl ToolDecoder for NoopToolDecoder {
    fn push(&mut self, _token: u32) -> Option<ToolCall> {
        None
    }
}
