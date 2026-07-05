//! Phi-3 instruct implementation.
//!
//! Chat shape:
//!   <|system|>\n{system}<|end|>\n
//!   <|user|>\n{user}<|end|>\n
//!   <|assistant|>\n{assistant}<|end|>\n
//!   ...
//!   <|assistant|>\n          ← cue
//!
//! Each role marker and `<|end|>` are single special tokens
//! (e.g. id 32010 / 32007 on Phi-3-mini-4k-instruct).

use crate::instruct::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use pie_tokenizer::Tokenizer;
use std::sync::Arc;

pub struct Phi3Instruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    end_suffix: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Phi3Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<|end|>", "<|endoftext|>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let newline = encode("\n");
        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = encode(role);
            v.extend(&newline);
            v
        };
        let mut end_suffix = encode("<|end|>");
        end_suffix.extend(&newline);

        Self {
            system_prefix: make_prefix("<|system|>"),
            user_prefix: make_prefix("<|user|>"),
            assistant_prefix: make_prefix("<|assistant|>"),
            end_suffix,
            stop_ids,
            tokenizer,
        }
    }
}

impl Instruct for Phi3Instruct {
    fn system(&self, message: &str) -> Vec<u32> {
        let mut v = self.system_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.end_suffix);
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        let mut v = self.user_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.end_suffix);
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.assistant_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.end_suffix);
        v
    }

    fn cue(&self) -> Vec<u32> {
        self.assistant_prefix.clone()
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
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}
