//! Turns written as `Role: text` lines: a bos opens the prompt, each turn is
//! its role, a colon and the text, closed by a blank line, and the
//! assistant's is closed by an eos. HunyuanImage 3 is spoken this way.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::{
    ChatDecoder, GenericChatDecoder, Instruct, NoopReasoningDecoder, NoopToolDecoder,
    ReasoningDecoder, ToolDecoder, special, specials,
};

pub struct Lines {
    tokenizer: Arc<Tokenizer>,
    bos: u32,
    eos: u32,
    sep: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Lines {
    #[must_use]
    pub fn new<S: AsRef<str>>(tokenizer: Arc<Tokenizer>, bos: &str, eos: &str, stop: &[S]) -> Self {
        let sep = tokenizer.encode("\n\n");
        Self {
            bos: special(&tokenizer, bos),
            eos: special(&tokenizer, eos),
            sep,
            stop_ids: specials(&tokenizer, stop),
            tokenizer,
        }
    }

    fn turn(&self, role: &str, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(&format!("{role}: {}", msg.trim()));
        tokens.extend(&self.sep);
        tokens
    }
}

impl Instruct for Lines {
    fn prefix(&self) -> Vec<u32> {
        vec![self.bos]
    }

    fn system(&self, msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.tokenizer.encode(msg.trim()));
        tokens.extend(&self.sep);
        tokens
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.turn("User", msg));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn("User", msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(&format!("Assistant: {}", msg.trim()));
        tokens.push(self.eos);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        self.tokenizer.encode("Assistant:")
    }

    fn seal(&self) -> Vec<u32> {
        vec![self.eos]
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
