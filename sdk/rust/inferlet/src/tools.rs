//! Optional helpers for tool calling.
//!
//! `inferlet` does not bake a tool-call loop into the [`Generator`]
//! surface — the right loop shape varies a lot between agents (ReAct,
//! CodeAct, JSON-call, native-grammar) and we'd rather give you the
//! pieces than a framework. This module exposes the host's tool-template
//! capability so callers that *do* want the model's native format can
//! reach for it explicitly:
//!
//! - [`equip_prefix`] — token sequence that registers tool schemas in
//!   the chat template (model-specific). Append to your context buffer.
//! - [`answer_prefix`] — token sequence that frames a tool result for
//!   the next turn.
//! - [`native_grammar`] — the model's tool-call grammar, if it has one.
//!   Wrap with [`GrammarConstraint`](crate::GrammarConstraint) and pass
//!   to [`Generator::constrain`](crate::generation::Generator::constrain)
//!   to enforce well-formed output.
//! - [`Decoder`] — streaming detector for tool calls inside generated
//!   text. Feed each step's tokens; collect `Call(name, args)` events.
//! - [`parse_call`] — best-effort one-shot parse out of a finished text
//!   blob. Useful for `collect_text()` flows that want to extract one
//!   call at the end.
//!
//! For agents that hand-roll their own format (e.g. `agent-react`'s
//! `Action: ToolName[input]` parsing), none of these are required.
//!
//! [`Generator`]: crate::generation::Generator

use crate::Result;
use crate::inference::Grammar;
use crate::model::Model;
use crate::pie::core::inference::Matcher;
use crate::pie::instruct::tool_use;

// =============================================================================
// Templating
// =============================================================================

/// Token sequence that registers `tool_schemas` (each a JSON Schema
/// string) in the chat template. Append to your context buffer (e.g.,
/// via [`Context::append`](crate::Context::append)) before the user
/// message. Models without a native tool-template return an empty vec.
pub fn equip_prefix(model: &Model, tool_schemas: &[String]) -> Result<Vec<u32>> {
    tool_use::equip(model, tool_schemas)
}

/// Token sequence that frames a tool result for the next turn. `name`
/// matches the call the model made; `value` is typically a JSON-encoded
/// result.
pub fn answer_prefix(model: &Model, name: &str, value: &str) -> Vec<u32> {
    tool_use::answer(model, name, value)
}

// =============================================================================
// Native grammar / matcher
// =============================================================================

/// The model's native tool-call grammar, if any. Returns `None` when the
/// model has no enforceable format (the caller should fall through to
/// free-form generation + their own parser).
pub fn native_grammar(model: &Model, tool_schemas: &[String]) -> Option<Grammar> {
    tool_use::format(model, tool_schemas)
}

/// Build a [`Matcher`] for the model's native tool-call grammar.
/// Returns `None` when the model has no enforceable format. Pair with
/// [`GrammarConstraint::new`](crate::GrammarConstraint::new) for
/// constrained generation.
pub fn native_matcher(model: &Model, tool_schemas: &[String]) -> Option<Matcher> {
    Some(tool_use::create_matcher(model, tool_schemas))
}

// =============================================================================
// Streaming decoder
// =============================================================================

/// Streaming tool-call detector. Feed each step's accepted tokens; the
/// decoder emits [`Event::Start`] when a tool call begins and
/// [`Event::Call`] (with parsed `name` and `arguments`) when it
/// completes.
pub struct Decoder {
    inner: tool_use::Decoder,
}

/// Tool-decoding event. One per `feed`.
///
/// `Start` fires while a tool-call structure is being assembled but the
/// arguments haven't closed yet — it's both "boundary entered" and the
/// no-meaningful-event signal during accumulation. Most callers can
/// ignore it and only act on `Call`.
#[derive(Clone, Debug)]
pub enum Event {
    /// A tool call is in progress — keep feeding.
    Start,
    /// Complete tool call: `(name, arguments_json)`.
    Call(String, String),
}

impl Decoder {
    /// Construct a decoder for `model`'s tool-call template.
    pub fn new(model: &Model) -> Self {
        Self {
            inner: tool_use::create_decoder(model),
        }
    }

    /// Feed a token batch and get back the event that fired (one per
    /// call). `Start` indicates an in-progress tool call; `Call` fires
    /// once when the arguments close.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Event> {
        match self.inner.feed(tokens)? {
            tool_use::Event::Start => Ok(Event::Start),
            tool_use::Event::Call((name, args)) => Ok(Event::Call(name, args)),
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

// =============================================================================
// One-shot parse
// =============================================================================

/// Best-effort parse of a single tool call out of a finished text blob.
/// Internally tokenizes, runs the decoder, and returns the first
/// completed `Call`. Useful for `collect_text()` flows that want to
/// extract one call at the end of generation. Returns `None` when no
/// completed call is detected.
pub fn parse_call(model: &Model, text: &str) -> Option<(String, String)> {
    let tokenizer = model.tokenizer();
    let tokens = tokenizer.encode(text);
    let mut dec = Decoder::new(model);
    match dec.feed(&tokens).ok()? {
        Event::Call(name, args) => Some((name, args)),
        Event::Start => None,
    }
}
