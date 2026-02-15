//! Mistral 3 instruct implementation.
//!
//! Uses [INST]...[/INST] delimiters.
//! System messages are prepended to the first user message.
//! No thinking or tool-use support.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder,
    ToolDecoder,
};
use crate::model::tokenizer::Tokenizer;
use crate::model::gemma::{NoopReasoningDecoder, NoopToolDecoder};

pub struct MistralInstruct {
    tokenizer: Arc<Tokenizer>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl MistralInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["</s>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            bos_token: encode("<s>"),
            stop_ids,
            tokenizer,
        }
    }
}

impl Instruct for MistralInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        // Mistral prepends system to first user [INST] block
        // Store as-is for now
        self.tokenizer.encode(msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode("[INST] ");
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(self.tokenizer.encode(" [/INST]"));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(msg);
        tokens.extend(&self.stop_ids);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        Vec::new() // Mistral generates immediately after [/INST]
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new()
    }

    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> {
        Vec::new()
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(MistralChatDecoder {
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

struct MistralChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for MistralChatDecoder {
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
        let tok = make_tok(&["<s>", "</s>", "[INST]", " ", "[/INST]", "Hello"]);
        let inst = MistralInstruct::new(tok);
        let stop = inst.seal();
        assert_eq!(stop.len(), 1);
        assert_eq!(stop[0], 1);
    }

    #[test]
    fn cue_is_empty() {
        let tok = make_tok(&["<s>", "</s>", "[INST]", " ", "[/INST]", "Hello"]);
        let inst = MistralInstruct::new(tok);
        assert!(inst.cue().is_empty(), "Mistral generation header should be empty");
    }

    #[test]
    fn equip_is_noop() {
        let tok = make_tok(&["<s>", "</s>", "[INST]", " ", "[/INST]", "Hello"]);
        let inst = MistralInstruct::new(tok);
        assert!(inst.equip(&["tool".to_string()]).is_empty());
    }

    #[test]
    fn formatting_nonempty() {
        let tok = make_tok(&["<s>", "</s>", "[INST]", " ", "[/INST]", "Hello"]);
        let inst = MistralInstruct::new(tok);
        assert!(!inst.system("Hello").is_empty());
        assert!(!inst.user("Hello").is_empty());
        assert!(!inst.assistant("Hello").is_empty());
    }
}
