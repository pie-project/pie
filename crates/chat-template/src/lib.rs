//! What a turn of conversation looks like on the way in, and what a stream of
//! tokens means on the way back out.
//!
//! A chat template is two halves of one agreement. The writing half turns a
//! role and a message into the exact tokens the model was trained to see
//! around them; the reading half watches generated tokens go by and says when
//! a reply began, when the reasoning block closed, and when a tool was called.
//! Both halves name the same markers, which is the whole reason they live in
//! one file per format rather than one file per direction.
//!
//! This crate knows nothing about model SKUs. It ships the formats —
//! [`chatml`], [`harmony`], [`gemma`], [`deepseek`], [`kimi`] — and the
//! catalog beside it names which model speaks which.

use std::sync::Arc;

use tokenizer::Tokenizer;

pub mod chatml;
pub mod decode;
pub mod deepseek;
pub mod gemma;
pub mod harmony;
pub mod kimi;

pub use decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder, ThinkingDecoder};

/// The grammar that constrains well-formed tool-call output, in the EBNF the
/// grammar engine compiles.
pub struct ToolGrammar {
    pub source: String,
}

/// The id of a marker the template spells, or a panic naming the marker.
///
/// Every marker in every format here is one token in the vocabulary the model
/// was trained with — that is what makes it a marker rather than a phrase. A
/// tokenizer that cannot spell one is a tokenizer paired with the wrong model,
/// and the failure belongs at the moment the template is built. It used to be
/// a `filter_map` over the stop list, which turned a missing `<|im_end|>` into
/// a model that generates until the token budget runs out.
#[must_use]
pub fn special(tokenizer: &Tokenizer, marker: &str) -> u32 {
    match tokenizer.token_to_id(marker) {
        Some(id) => id,
        None => panic!(
            "this tokenizer has no `{marker}`; a template cannot mark a turn with a token its vocabulary does not contain"
        ),
    }
}

/// [`special`] over a list — the shape a stop list arrives in.
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
    /// Nothing has happened yet. A decoder never yields this — an idle batch
    /// yields no events at all — but a caller that must answer with one event
    /// per feed needs a value for "still nothing", and inventing `Start` for
    /// it is what made `Start` meaningless in the first place.
    None,

    /// A tool-call span opened.
    Start,

    /// A complete call: the function's name, and its arguments as JSON.
    Call(String, String),
}

/// The three readers all take a batch of tokens and answer with everything
/// that batch contained.
///
/// The `Vec` is not decoration. Each of these used to return one event and
/// drop the rest of the batch on the floor: a batch that carried the stop
/// token in the middle lost every token after it, and a batch carrying two
/// tool calls surfaced one. Whatever a batch says, it says here in order.
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

/// One format, both directions.
///
/// `equip` and `answer` default to writing nothing, because a format without
/// a tool grammar has nothing to write there and a stub that says `Vec::new()`
/// in four files says it four times.
pub trait Instruct: Send + Sync {
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

/// The constructor every format exposes to the catalog beside it.
pub type Build = fn(Arc<Tokenizer>) -> Arc<dyn Instruct>;
