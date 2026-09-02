//! GLM-5's turn format: `[gMASK]<sop>` opens the conversation, each role is
//! its own marker (`<|system|>`, `<|user|>`, `<|assistant|>`,
//! `<|observation|>`) with no closer, and an assistant turn carries an
//! explicit `<think>…</think>` block. The generation cue is
//! `<|assistant|><think>`, so the reasoning decoder starts inside the block.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decode::{GenericChatDecoder, NoopToolDecoder, ThinkingDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, special, specials};

pub const STOP_TOKENS: &[&str] = &["<|user|>", "<|observation|>", "<|endoftext|>"];
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
/// The template's default reasoning effort, stated as the first system line.
const EFFORT: &str = "Reasoning Effort: Max";

pub struct Glm {
    tokenizer: Arc<Tokenizer>,
    prefix: Vec<u32>,
    system_marker: u32,
    user_marker: u32,
    assistant_marker: u32,
    think_open: Vec<u32>,
    think_close: u32,
    stop_ids: Vec<u32>,
}

impl Glm {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let stop_ids = specials(&tokenizer, STOP_TOKENS);
        let system_marker = special(&tokenizer, "<|system|>");
        let mut prefix = vec![special(&tokenizer, "[gMASK]"), special(&tokenizer, "<sop>")];
        prefix.push(system_marker);
        prefix.extend(tokenizer.encode(EFFORT));
        let think_open = tokenizer.encode(THINK_OPEN);
        let think_close = tokenizer.encode(THINK_CLOSE);
        assert_eq!(think_close.len(), 1, "`</think>` is one token in GLM's vocabulary");
        Self {
            prefix,
            system_marker,
            user_marker: special(&tokenizer, "<|user|>"),
            assistant_marker: special(&tokenizer, "<|assistant|>"),
            think_open,
            think_close: think_close[0],
            stop_ids,
            tokenizer,
        }
    }

    fn turn(&self, marker: u32, msg: &str) -> Vec<u32> {
        let mut tokens = vec![marker];
        tokens.extend(self.tokenizer.encode(msg));
        tokens
    }

    /// A replayed reply without a thinking block is given the empty one.
    fn assistant_body(msg: &str) -> String {
        if msg.trim_start().starts_with(THINK_OPEN) {
            msg.to_string()
        } else {
            format!("{THINK_OPEN}{THINK_CLOSE}{msg}")
        }
    }
}

impl Instruct for Glm {
    fn system(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.prefix.clone();
        tokens.extend(self.turn(self.system_marker, msg));
        tokens
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.prefix.clone();
        tokens.extend(self.user(msg));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(self.user_marker, msg)
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.user(user));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(self.assistant_marker, &Self::assistant_body(msg))
    }

    fn cue(&self) -> Vec<u32> {
        let mut tokens = vec![self.assistant_marker];
        tokens.extend(&self.think_open);
        tokens
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
        Box::new(ThinkingDecoder::opened(
            self.tokenizer.clone(),
            self.think_open.clone(),
            self.think_close,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}
