//! Kimi's turn format. It looks like ChatML from a distance and is not one:
//! the role is announced by its own marker (`<|im_user|>`, `<|im_system|>`,
//! `<|im_assistant|>`) rather than a shared opener plus role word, the
//! header closes with `<|im_middle|>`, the turn ends without a trailing
//! newline, and every assistant turn carries an explicit thinking block.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, special, specials};

pub const STOP_TOKENS: &[&str] = &["<|im_end|>", "[EOS]"];
const THINK_OPEN: &str = "<think>";

pub struct Kimi {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Kimi {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let stop_ids = specials(&tokenizer, STOP_TOKENS);
        let im_middle = special(&tokenizer, "<|im_middle|>");
        let im_end = special(&tokenizer, "<|im_end|>");

        let header = |marker: &str, role: &str| -> Vec<u32> {
            let mut tokens = vec![special(&tokenizer, marker)];
            tokens.extend(tokenizer.encode(role));
            tokens.push(im_middle);
            tokens
        };

        let mut generation_header = header("<|im_assistant|>", "assistant");
        generation_header.extend(tokenizer.encode("<think></think>"));

        Self {
            system_prefix: header("<|im_system|>", "system"),
            user_prefix: header("<|im_user|>", "user"),
            assistant_prefix: header("<|im_assistant|>", "assistant"),
            turn_suffix: vec![im_end],
            generation_header,
            stop_ids,
            tokenizer,
        }
    }

    fn turn(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    /// Every assistant turn opens with a thinking block, so a replayed reply
    /// that has none is given the empty one. Checks the start of the message
    /// only — a reply that merely quotes the marker mid-prose stays bodiless.
    fn assistant_body(msg: &str) -> String {
        if msg.trim_start().starts_with(THINK_OPEN) {
            msg.to_string()
        } else {
            format!("<think></think>{msg}")
        }
    }
}

impl Instruct for Kimi {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.system_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.assistant_prefix, &Self::assistant_body(msg))
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_header.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
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
