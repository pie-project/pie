//! Text as it is: every turn is its text's tokens, with no role, cue or
//! seal. A text encoder that conditions a diffusion model is spoken this
//! way. A stop token the vocabulary lacks is simply not a stop.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::{
    ChatDecoder, GenericChatDecoder, Instruct, NoopReasoningDecoder, NoopToolDecoder,
    ReasoningDecoder, ToolDecoder,
};

pub struct Raw {
    tokenizer: Arc<Tokenizer>,
    stop: Vec<u32>,
}

impl Raw {
    #[must_use]
    pub fn new<S: AsRef<str>>(tokenizer: Arc<Tokenizer>, stop: &[S]) -> Self {
        let stop = stop
            .iter()
            .filter_map(|marker| tokenizer.token_to_id(marker.as_ref()))
            .collect();
        Self { tokenizer, stop }
    }

    fn text(&self, msg: &str) -> Vec<u32> {
        self.tokenizer.encode(msg)
    }
}

impl Instruct for Raw {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.text(msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.text(msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.text(msg)
    }

    fn cue(&self) -> Vec<u32> {
        Vec::new()
    }

    fn seal(&self) -> Vec<u32> {
        Vec::new()
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            Arc::clone(&self.tokenizer),
            self.stop.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}
