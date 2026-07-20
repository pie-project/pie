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
// Tool trait — schema metadata for chat-template registration
// =============================================================================

/// Tool metadata used by [`Context::equip`](crate::Context::equip) to splice
/// schemas into the model's chat template. Implement directly for dynamic
/// tools, or derive via the [`#[tool]`](inferlet_macros::tool) macro for
/// static ones.
///
/// `schema` returns the JSON Schema for the args **object only** —
/// `{"type":"object","properties":{...},"required":[...]}` — without the
/// outer `{name, description, parameters}` envelope. The SDK wraps it in
/// whatever shape the host's `equip_prefix` expects.
pub trait Tool {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn schema(&self) -> &'static str;
}

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

/// Token sequence for a system turn that also registers `tool_schemas`.
///
/// Prefer this over `system` + [`equip_prefix`] whenever both are being
/// emitted: templates that nest tool declarations inside the first system
/// turn (Gemma 4) can only be produced this way, and templates that keep them
/// separate produce byte-identical output either way.
pub fn system_equip_prefix(
    model: &Model,
    system: &str,
    tool_schemas: &[String],
) -> Result<Vec<u32>> {
    tool_use::system_equip(model, system, tool_schemas)
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
    ///
    /// This decoder does not know which tools were declared, so it can only
    /// apply the template's lexical rules. Prefer [`Decoder::with_tools`]
    /// whenever the toolset is known: it additionally refuses a call naming a
    /// tool the model was never shown.
    pub fn new(model: &Model) -> Self {
        Self {
            inner: tool_use::create_decoder(model),
        }
    }

    /// Construct a decoder that also enforces declared-tool membership.
    ///
    /// `tool_schemas` is the same set passed to
    /// [`equip_prefix`]/[`system_equip_prefix`]. The decoder reports a call
    /// only when its name is one of those tools, so an undeclared name is
    /// refused even when generation was not constrained by a grammar.
    pub fn with_tools(model: &Model, tool_schemas: &[String]) -> Self {
        Self {
            inner: tool_use::create_decoder_for_tools(model, tool_schemas),
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
