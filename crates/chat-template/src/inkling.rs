use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decode::{GenericChatDecoder, NoopToolDecoder, ThinkingDecoder};
use crate::{ChatDecoder, Effort, Instruct, ReasoningDecoder, ToolDecoder, special, specials};

pub const STOP_TOKENS: &[&str] = &["<|content_model_end_sampling|>", "<|endoftext|>"];

pub const MARKERS: &[&str] = &[
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|end_message|>",
];

const THINKING_EFFORT: &str = "0.9";

pub struct Inkling {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    tool_header: u32,
    content_text: u32,
    content_thinking: u32,
    end_message: u32,
    end_sampling: u32,
    stop_ids: Vec<u32>,
    effort: Vec<u32>,
}

impl Inkling {
    pub fn try_new(tokenizer: Arc<Tokenizer>) -> Result<Self, String> {
        let stop_ids = specials(&tokenizer, STOP_TOKENS)?;
        let content_text = special(&tokenizer, "<|content_text|>")?;
        let end_message = special(&tokenizer, "<|end_message|>")?;
        let header = |role: &str| -> Result<Vec<u32>, String> {
            Ok(vec![special(&tokenizer, role)?, content_text])
        };
        let system_prefix = header("<|message_system|>")?;
        let mut effort = system_prefix.clone();
        effort.extend(tokenizer.encode(&format!("Thinking effort level: {THINKING_EFFORT}")));
        effort.push(end_message);
        Ok(Self {
            user_prefix: header("<|message_user|>")?,
            model_prefix: header("<|message_model|>")?,
            tool_header: special(&tokenizer, "<|message_tool|>")?,
            content_thinking: special(&tokenizer, "<|content_thinking|>")?,
            end_sampling: special(&tokenizer, "<|content_model_end_sampling|>")?,
            system_prefix,
            content_text,
            end_message,
            stop_ids,
            effort,
            tokenizer,
        })
    }

    fn message(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.push(self.end_message);
        tokens
    }
}

impl Instruct for Inkling {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.message(&self.system_prefix, msg)
    }

    /// The effort line goes ahead of the conversation's first message that
    /// is not a system message: after an opening system turn, which opens
    /// the conversation in its place ([`Instruct::open`]), or ahead of the
    /// first user turn.
    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.effort.clone();
        tokens.extend(self.user(msg));
        tokens
    }

    /// The system turn, then the effort line ([`Instruct::first_user`]).
    /// A further leading system message follows the effort line here, where
    /// the template writes it ahead.
    fn open(&self, system: Option<&str>, _tools: &[String], _effort: Option<Effort>) -> Vec<u32> {
        let Some(msg) = system else {
            return Vec::new();
        };
        let mut tokens = self.system(msg);
        tokens.extend(&self.effort);
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.message(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.message(&self.model_prefix, msg);
        tokens.push(self.end_sampling);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        vec![self.model_prefix[0]]
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        let mut tokens = vec![self.tool_header];
        tokens.extend(self.tokenizer.encode(name));
        tokens.push(self.content_text);
        tokens.extend(self.tokenizer.encode(value));
        tokens.push(self.end_message);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            vec![self.content_thinking],
            self.end_message,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}
