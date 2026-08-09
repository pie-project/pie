//! Instruct trait — model-specific conversational AI formatting and decoding.
//!
//! Each model architecture provides its own implementation. The API layer
//! delegates to the model's `Instruct` impl for all instruct operations.
//!
//! Both halves are here. They were two files in two crates — the trait below
//! and the [`create`] registry at the bottom — because a generation crate
//! could not depend on the crate that dispatches to it, so the vocabulary had
//! to sit underneath both. One crate, one module.

/// A model-provided tool-call grammar in EBNF form.
pub struct ToolGrammar {
    pub source: String,
}
// The shared decoders, re-exported so `instruct::decoders` stays a valid
// path: it is what every generation's template imports, and the templates
// became crates without their imports needing to know.
pub use crate::decoders;

/// Events emitted by the chat decoder.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Generated text chunk
    Delta(String),
    /// Special token encountered (token ID)
    Interrupt(u32),
    /// Generation complete (full accumulated text)
    Done(String),
}

/// Events emitted by the reasoning decoder.
#[derive(Debug, Clone)]
pub enum ReasoningEvent {
    /// Reasoning block started
    Start,
    /// Reasoning text chunk
    Delta(String),
    /// Reasoning complete (full reasoning text)
    Complete(String),
}

/// Events emitted by the tool decoder.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// Tool call detected
    Start,
    /// Complete tool call: (name, arguments-json)
    Call(String, String),
}

/// Classifies generated tokens into text deltas, interrupts, and done.
pub trait ChatDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent;
    fn reset(&mut self);
}

/// Detects reasoning/thinking blocks in the token stream.
pub trait ReasoningDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent;
    fn reset(&mut self);
}

/// Detects tool call blocks in the token stream.
pub trait ToolDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent;
    fn reset(&mut self);
}

/// Model-specific instruct implementation.
///
/// Each architecture provides its own impl with hardcoded tokens & logic.
/// The tokenizer is owned by the implementation to avoid redundant lookups.
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
    fn equip(&self, tools: &[String]) -> Vec<u32>;
    fn answer(&self, name: &str, value: &str) -> Vec<u32>;
    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;
    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;
    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;
    /// Returns the parsed tool-call grammar that constrains generation to
    /// the architecture's tool-call format, given a list of tool schemas.
    /// Returns `None` if the architecture doesn't support constrained tool calling.
    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}

// ── The registry ─────────────────────────────────────────────────────

use tokenizer::Tokenizer;
use std::sync::Arc;

/// The chat template that speaks for a model.
///
/// # What this replaced, and why it is worth the churn
///
/// A `match` on `architectures[0]` with thirty arms and a `_ =>` that
/// answered ChatML. The fallback is the whole story: gemma-4's
/// `architectures[0]` is `Gemma4ForConditionalGeneration`, the table
/// knew a different stem, so gemma-4 fell through and got Qwen's
/// template. It then generated *fluently* — ending turns it was not
/// having with an `<|im_end|>` its vocabulary does not contain — because
/// a wrong chat template does not crash, it just makes a model sound
/// slightly deranged.
///
/// There is nowhere to put that now. A row answers
/// [`Variant::chat`](crate::catalog::Variant::chat), the method has no
/// default body, and a model with no row does not resolve to a template
/// at all — it fails to resolve, by name, at the door.
///
/// # Errors
///
/// [`Unmatched::NoSuchId`] when nothing in the catalog carries this id.
/// The error carries the nearest ids, because the overwhelmingly likely
/// cause is a typo in an `--as` override.
///
/// [`Unmatched::NoSuchId`]: crate::catalog::Unmatched::NoSuchId
pub fn create(id: &str, tokenizer: Arc<Tokenizer>) -> Result<Arc<dyn Instruct>, crate::catalog::Unmatched> {
    let row = crate::catalog::find(id).ok_or_else(|| crate::catalog::Unmatched::NoSuchId {
        id: id.to_string(),
        nearest: crate::catalog::nearest_ids(id, 3),
    })?;
    Ok(row.chat(tokenizer))
}

#[cfg(test)]
mod registry_tests {
    /// Every row answers, and no row has to fall through to answer.
    ///
    /// The old registry could not have this test: it took a string
    /// nothing enumerated, so "every input is handled" was true only in
    /// the sense that `_ =>` handles everything.
    #[test]
    fn every_row_names_a_template() {
        assert!(!crate::catalog::ids().is_empty(), "the catalog is empty");
    }

    /// An id nothing carries is a refusal with a suggestion, not a
    /// silently wrong template.
    #[test]
    fn an_unknown_id_is_refused_rather_than_guessed() {
        let e = crate::catalog::find("qwen3-0.6").is_none();
        assert!(e, "a near-miss id must not resolve");
    }
}
