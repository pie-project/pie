use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;

pub struct Olmo2Instruct {
    tokenizer: Arc<Tokenizer>,
    bos: Vec<u32>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
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
