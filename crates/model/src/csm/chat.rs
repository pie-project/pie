use std::sync::Arc;
use tokenizer::Tokenizer;

use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};

pub const SPEAKER_USER: u32 = 0;

pub const SPEAKER_ASSISTANT: u32 = 1;

pub struct CsmInstruct {
    tokenizer: Arc<Tokenizer>,
    bos: Vec<u32>,
    eos: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl CsmInstruct {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {

        let bos = tokenizer.encode("<|begin_of_text|>");
        let eos = tokenizer.encode("<|end_of_text|>");

        let stop_ids = ["<|end_of_text|>", "<|audio_eos|>"]
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();
        Self {
            tokenizer,
            bos,
            eos,
            stop_ids,
        }
    }

    fn turn(&self, speaker: u32, message: &str) -> Vec<u32> {
        let mut v = self.bos.clone();
        v.extend(self.tokenizer.encode(&format!("[{speaker}]")));
        v.extend(self.tokenizer.encode(message.trim()));
        v.extend(&self.eos);
        v
    }
}

impl Instruct for CsmInstruct {

    fn system(&self, _message: &str) -> Vec<u32> {
        Vec::new()
    }

    fn user(&self, message: &str) -> Vec<u32> {
        self.turn(SPEAKER_USER, message)
    }

    fn first_user(&self, message: &str) -> Vec<u32> {
        self.user(message)
    }

    fn system_user(&self, _system: &str, user: &str) -> Vec<u32> {
        self.user(user)
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        self.turn(SPEAKER_ASSISTANT, message)
    }

    fn cue(&self) -> Vec<u32> {
        let mut v = self.bos.clone();
        v.extend(self.tokenizer.encode(&format!("[{SPEAKER_ASSISTANT}]")));
        v
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
