//! OLMo-2 instruct implementation.
//!
//! OLMo-2's published chat template is *not* ChatML (despite the family
//! sometimes being grouped with OLMo-3 / qwen-style templates). It uses
//! plain `<|system|>` / `<|user|>` / `<|assistant|>` role markers with
//! newline message separators, prefixed with `<|endoftext|>` as a BOS
//! and terminated with `<|endoftext|>` as the stop token. There is no
//! `<|end|>` between messages.
//!
//! Chat shape (rendered):
//!     <|endoftext|>
//!     <|system|>\n{system}\n
//!     <|user|>\n{user}\n
//!     <|assistant|>\n{assistant}<|endoftext|>
//!     ...
//!     <|assistant|>\n          ← cue
//!
//! Verified by `tokenizer.apply_chat_template` on
//! `allenai/OLMo-2-1124-7B-Instruct`.

use crate::instruct::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use pie_tokenizer::Tokenizer;
use std::sync::Arc;

pub struct Olmo2Instruct {
    tokenizer: Arc<Tokenizer>,
    bos: Vec<u32>,
    system_prefix: Vec<u32>,    // "<|system|>\n"
    user_prefix: Vec<u32>,      // "<|user|>\n"
    assistant_prefix: Vec<u32>, // "<|assistant|>\n"
    newline: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Olmo2Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = encode(role);
            v.extend(&newline);
            v
        };

        let stop_strs = ["<|endoftext|>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            bos: encode("<|endoftext|>"),
            system_prefix: make_prefix("<|system|>"),
            user_prefix: make_prefix("<|user|>"),
            assistant_prefix: make_prefix("<|assistant|>"),
            newline,
            stop_ids,
            tokenizer,
        }
    }
}

impl Instruct for Olmo2Instruct {
    fn system(&self, message: &str) -> Vec<u32> {
        // OLMo-2's template puts <|endoftext|> at the very start of the
        // conversation. We prepend it on the first message (system) so
        // the framing matches the tokenizer's apply_chat_template output.
        let mut v = self.bos.clone();
        v.extend(&self.system_prefix);
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.newline);
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        let mut v = self.user_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.newline);
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.assistant_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        // Assistant turn closes with the EOS token.
        v.extend(&self.stop_ids);
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
