use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decoders::{GenericChatDecoder, NoopToolDecoder};
use crate::instruct::{ChatDecoder, Instruct, Reasoning, ReasoningDecoder, ToolDecoder};

pub struct Template {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
    think_open: u32,
    think_close: u32,
}

impl Template {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let role_prefix = |token: &str, role: &str| {
            let mut v = encode(token);
            v.extend(encode(role));
            v.extend(encode("<|im_middle|>"));
            v
        };
        let stop_ids: Vec<u32> = ["<|im_end|>", "[EOS]"]
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();
        let think_open = tokenizer.token_to_id("<think>").unwrap_or(0);
        let think_close = tokenizer.token_to_id("</think>").unwrap_or(0);

        let assistant_prefix = role_prefix("<|im_assistant|>", "assistant");
        let mut generation_header = assistant_prefix.clone();
        generation_header.extend(encode("<think></think>"));

        Self {
            system_prefix: role_prefix("<|im_system|>", "system"),
            user_prefix: role_prefix("<|im_user|>", "user"),
            assistant_prefix,
            turn_suffix: encode("<|im_end|>"),
            generation_header,
            stop_ids,
            think_open,
            think_close,
            tokenizer,
        }
    }

    fn turn(&self, prefix: &[u32], message: &str) -> Vec<u32> {
        let mut v = prefix.to_vec();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn thought_out(message: &str) -> String {
        if message.contains("<think>") {
            message.to_string()
        } else {
            format!("<think></think>{message}")
        }
    }
}

impl Instruct for Template {
    fn system(&self, message: &str) -> Vec<u32> {
        self.turn(&self.system_prefix, message)
    }

    fn first_user(&self, message: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, message)
    }

    fn user(&self, message: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, message)
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut v = self.turn(&self.system_prefix, system);
        v.extend(self.turn(&self.user_prefix, user));
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        self.turn(&self.assistant_prefix, &Self::thought_out(message))
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
        Box::new(ThinkingDecoder {
            tokenizer: self.tokenizer.clone(),
            open: self.think_open,
            close: self.think_close,
            thinking: false,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

struct ThinkingDecoder {
    tokenizer: Arc<Tokenizer>,
    open: u32,
    close: u32,
    thinking: bool,
}

impl ReasoningDecoder for ThinkingDecoder {
    fn push(&mut self, token: u32) -> Option<Reasoning> {
        if token == self.open {
            self.thinking = true;
            return None;
        }
        if token == self.close {
            self.thinking = false;
            return None;
        }
        let text = self.tokenizer.decode(&[token], true);
        if self.thinking {
            Some(Reasoning::Thinking(text))
        } else {
            Some(Reasoning::Answer(text))
        }
    }
}
