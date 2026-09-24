use std::sync::Arc;

use tokenizer::Tokenizer;

pub mod atem;
pub mod chatml;
pub mod decode;
pub mod deepseek;
pub mod gemma;
pub mod glm;
pub mod harmony;
pub mod inkling;
pub mod kimi;
pub mod kimi3;

pub use decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder, ThinkingDecoder};

pub struct ToolGrammar {
    pub source: String,
}

#[must_use]
pub fn special(tokenizer: &Tokenizer, marker: &str) -> u32 {
    match tokenizer.token_to_id(marker) {
        Some(id) => id,
        None => panic!(
            "this tokenizer has no `{marker}`; a template cannot mark a turn with a token its vocabulary does not contain"
        ),
    }
}

#[must_use]
pub fn specials(tokenizer: &Tokenizer, markers: &[&str]) -> Vec<u32> {
    markers
        .iter()
        .map(|marker| special(tokenizer, marker))
        .collect()
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Delta(String),

    Interrupt(u32),

    Done(String),
}

#[derive(Debug, Clone)]
pub enum ReasoningEvent {
    Start,

    Delta(String),

    Complete(String),
}

#[derive(Debug, Clone)]
pub enum ToolEvent {
    None,

    Start,

    Call(String, String),
}

pub trait ChatDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ChatEvent>;
    fn reset(&mut self);
}

pub trait ReasoningDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ReasoningEvent>;
    fn reset(&mut self);
}

pub trait ToolDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent>;
    fn reset(&mut self);
}

pub trait Instruct: Send + Sync {
    fn prefix(&self) -> Vec<u32> {
        Vec::new()
    }

    fn system(&self, msg: &str) -> Vec<u32>;

    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.user(msg)
    }

    fn user(&self, msg: &str) -> Vec<u32>;

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.user(user));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32>;

    fn cue(&self) -> Vec<u32>;

    fn seal(&self) -> Vec<u32>;

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new()
    }

    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> {
        Vec::new()
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;

    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;

    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}

pub type Build = fn(Arc<Tokenizer>) -> Arc<dyn Instruct>;
