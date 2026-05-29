//! Kimi K2/K2.5 instruct implementation.

use crate::model::instruct::decoders::{
    GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder,
};
use crate::model::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::model::tokenizer::Tokenizer;
use std::sync::Arc;

pub struct KimiInstruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl KimiInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let role_prefix = |role_token: &str, role_name: &str| {
            let mut tokens = encode(role_token);
            tokens.extend(encode(role_name));
            tokens.extend(encode("<|im_middle|>"));
            tokens
        };
        let stop_ids = ["<|im_end|>", "[EOS]"]
            .iter()
            .filter_map(|token| tokenizer.token_to_id(token))
            .collect();

        let mut generation_header = role_prefix("<|im_assistant|>", "assistant");
        generation_header.extend(encode("<think></think>"));

        Self {
            system_prefix: role_prefix("<|im_system|>", "system"),
            user_prefix: role_prefix("<|im_user|>", "user"),
            assistant_prefix: role_prefix("<|im_assistant|>", "assistant"),
            turn_suffix: encode("<|im_end|>"),
            generation_header,
            stop_ids,
            tokenizer,
        }
    }

    fn role_tokens(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn assistant_body(msg: &str) -> String {
        if msg.contains("<think>") {
            msg.to_string()
        } else {
            format!("<think></think>{msg}")
        }
    }
}

impl Instruct for KimiInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.system_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.assistant_prefix, &Self::assistant_body(msg))
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_header.clone()
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
