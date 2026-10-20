use std::sync::Arc;

use tokenizer::Tokenizer;

pub mod atem;
pub mod chatml;
pub mod decode;
pub mod deepseek;
pub mod dots;
pub mod gemma;
pub mod glm;
pub mod harmony;
pub mod hy;
pub mod inkling;
pub mod json;
pub mod kimi;
pub mod kimi3;
pub mod lines;
pub mod mimo;
pub mod minimax;
pub mod raw;
pub mod spec;

pub use decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder, ThinkingDecoder};
pub use spec::{FORMATS, Spec, build};

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
pub fn specials<S: AsRef<str>>(tokenizer: &Tokenizer, markers: &[S]) -> Vec<u32> {
    markers
        .iter()
        .map(|marker| special(tokenizer, marker.as_ref()))
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
    /// The first token is already inside a thinking block, which the cue
    /// opened; no [`ReasoningEvent::Start`] precedes it.
    fn starts_inside(&self) -> bool {
        false
    }
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

    /// The system turn of a request the model answers with thinking, as its
    /// template renders it: the template's reasoning-effort instruction ahead
    /// of `msg`, in a system turn of its own when `msg` is `None`. A template
    /// without one writes [`Instruct::system`], or nothing for `None`.
    fn system_thinking(&self, msg: Option<&str>, _effort: Effort) -> Vec<u32> {
        msg.map(|msg| self.system(msg)).unwrap_or_default()
    }

    /// [`Instruct::cue`] for a turn the model answers without thinking, as
    /// the reference template renders `enable_thinking: false`.
    fn cue_without_thinking(&self) -> Vec<u32> {
        self.cue()
    }

    fn seal(&self) -> Vec<u32>;

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new()
    }

    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> {
        Vec::new()
    }

    /// The system turn a conversation opens with, as the template writes it
    /// from the request's system message, its tools and, for a request the
    /// model answers with thinking, its reasoning effort. The default is
    /// [`Instruct::system_thinking`] (or [`Instruct::system`]), then
    /// [`Instruct::equip`].
    fn open(&self, system: Option<&str>, tools: &[String], effort: Option<Effort>) -> Vec<u32> {
        let mut tokens = match effort {
            Some(effort) => self.system_thinking(system, effort),
            None => system.map(|msg| self.system(msg)).unwrap_or_default(),
        };
        if !tools.is_empty() {
            tokens.extend(self.equip(tools));
        }
        tokens
    }

    /// An earlier assistant turn: `msg`, then the `calls` (name, arguments
    /// JSON) it made in the template's call syntax. `current` is a turn after
    /// the conversation's last user query, which some templates replay with
    /// its reasoning. The default writes each call as a ChatML
    /// `<tool_call>{"name": …, "arguments": …}</tool_call>`.
    fn assistant_calls(&self, msg: &str, calls: &[(String, String)], _current: bool) -> Vec<u32> {
        let mut body = msg.to_string();
        for (name, arguments) in calls {
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(&format!(
                "<tool_call>\n{{\"name\": {}, \"arguments\": {}}}\n</tool_call>",
                json::tojson(&serde_json::Value::String(name.clone())),
                json::retojson(arguments)
            ));
        }
        self.assistant(&body)
    }

    /// Consecutive tool results (name, value), as the template writes them
    /// together. The default is one [`Instruct::answer`] each.
    fn answers(&self, results: &[(String, String)]) -> Vec<u32> {
        results
            .iter()
            .flat_map(|(name, value)| self.answer(name, value))
            .collect()
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;

    /// [`Instruct::reasoning_decoder`] for the turn `cue` began: inside a
    /// thinking block from the first token when the cue left one open.
    fn reasoning_decoder_for_cue(&self, _cue: &[u32]) -> Box<dyn ReasoningDecoder> {
        self.reasoning_decoder()
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;

    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}

/// The reasoning effort a template names for a thinking request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Effort {
    /// The template's default when a request names none.
    #[default]
    Xhigh,
    Medium,
    Low,
}

pub type Build = fn(Arc<Tokenizer>) -> Arc<dyn Instruct>;
