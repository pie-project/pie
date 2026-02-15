//! Gemma 2/3 instruct implementation.
//!
//! Uses <start_of_turn>/<end_of_turn> delimiters.
//! System messages are prepended to the first user message.
//! No thinking or tool-use support.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder, ReasoningEvent,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;

pub struct GemmaInstruct {
    tokenizer: Arc<Tokenizer>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl GemmaInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<end_of_turn>", "<eos>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            user_prefix: encode("<start_of_turn>user\n"),
            model_prefix: encode("<start_of_turn>model\n"),
            turn_suffix: encode("<end_of_turn>\n"),
            bos_token: encode("<bos>"),
            stop_ids,
            tokenizer,
        }
    }
}

impl Instruct for GemmaInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        // Gemma has no native system role — this is stored and prepended
        // to the first user message. For now, wrap as user-like.
        let mut tokens = self.user_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.user_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.model_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        self.model_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new() // No tool support
    }

    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> {
        Vec::new() // No tool support
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GemmaChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.stop_ids.clone(),
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

struct GemmaChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for GemmaChatDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent {
        for &t in tokens {
            if self.stop_ids.contains(&t) {
                let text = std::mem::take(&mut self.accumulated);
                return ChatEvent::Done(text);
            }
        }
        let delta = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&delta);
        ChatEvent::Delta(delta)
    }

    fn reset(&mut self) {
        self.accumulated.clear();
    }
}

/// No-op reasoning decoder for models without thinking support.
pub(crate) struct NoopReasoningDecoder;

impl ReasoningDecoder for NoopReasoningDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> ReasoningEvent {
        ReasoningEvent::Delta(String::new())
    }
    fn reset(&mut self) {}
}

/// No-op tool decoder for models without tool support.
pub(crate) struct NoopToolDecoder;

impl ToolDecoder for NoopToolDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> ToolEvent {
        ToolEvent::Start
    }
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::model::tokenizer::Tokenizer;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    #[test]
    fn has_correct_stop_tokens() {
        let tok = make_tok(&["<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>", "user", "model", "\\n", "Hello"]);
        let inst = GemmaInstruct::new(tok);
        let stop = inst.seal();
        assert!(stop.contains(&1));
        assert!(stop.contains(&2));
    }

    #[test]
    fn assistant_uses_model_prefix() {
        let tok = make_tok(&["<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>", "user", "model", "\\n", "Hello"]);
        let inst = GemmaInstruct::new(tok);
        let tokens = inst.assistant("Hello");
        assert!(!tokens.is_empty());
        assert_eq!(&tokens[..inst.model_prefix.len()], &inst.model_prefix[..]);
    }

    #[test]
    fn equip_is_noop() {
        let tok = make_tok(&["<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>", "user", "model", "\\n", "Hello"]);
        let inst = GemmaInstruct::new(tok);
        assert!(inst.equip(&["tool".to_string()]).is_empty());
    }

    #[test]
    fn answer_is_noop() {
        let tok = make_tok(&["<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>", "user", "model", "\\n", "Hello"]);
        let inst = GemmaInstruct::new(tok);
        assert!(inst.answer("fn1", "42").is_empty());
    }

    #[test]
    fn cue_matches_generation_header() {
        let tok = make_tok(&["<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>", "user", "model", "\\n", "Hello"]);
        let inst = GemmaInstruct::new(tok);
        assert_eq!(inst.cue(), inst.model_prefix);
    }
}
