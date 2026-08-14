//! Instruct trait — model-specific conversational AI formatting and decoding.
//!
//! Each model architecture provides its own implementation. The API layer
//! delegates to the model's `Instruct` impl for all instruct operations.
//!
//! The *vocabulary* only. `create()` — the registry that picks an
//! implementation for an `arch_name` — is in `pie-model`, because it names
//! every generation and a generation crate cannot depend on the thing that
//! dispatches to it.

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

/// A tool call an assistant turn made, as the caller replays it.
///
/// Same two fields the decoder reports in [`ToolEvent::Call`]: what came out of
/// a turn is what goes back into the next one's history.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments_json: String,
}

/// One tool result, as the caller replays it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolObservation {
    pub name: String,
    pub value: String,
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
    /// Replay an assistant turn that called tools.
    ///
    /// `reasoning_header` is the caller's answer to a question no single turn
    /// can answer: the Qwen3.5+ template keeps a reasoning header on every
    /// assistant turn AFTER the last real user query
    /// (`loop.index0 > ns.last_query_index`), and only the holder of the whole
    /// message list knows where that boundary is.
    ///
    /// The default drops the calls, which is what every model whose template
    /// has no tool-call surface should do.
    fn assistant_call(&self, msg: &str, _calls: &[ToolCall], _reasoning_header: bool) -> Vec<u32> {
        self.assistant(msg)
    }
    fn cue(&self) -> Vec<u32>;
    /// The generation header for a turn the caller has asked NOT to think.
    ///
    /// A separate method rather than a flag on [`Self::cue`] because for most
    /// templates there is no such header and the honest answer is the same one:
    /// the default returns it.
    fn cue_without_thinking(&self) -> Vec<u32> {
        self.cue()
    }
    fn seal(&self) -> Vec<u32>;
    fn equip(&self, tools: &[String]) -> Vec<u32>;
    /// Declare tools and the caller's system content as ONE opening turn.
    ///
    /// Templates differ on whether the two are separable. The default is the
    /// separable form — two turns — which is what a template that renders the
    /// system message independently of `tools` produces. A template that NESTS
    /// the system content inside the tool declaration overrides this; there the
    /// two calls cannot reproduce the one turn.
    fn equip_into_system(&self, system: &str, tools: &[String]) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.equip(tools));
        tokens
    }
    fn answer(&self, name: &str, value: &str) -> Vec<u32>;
    /// A run of consecutive tool results, as the template turns them into turns.
    ///
    /// The default is one turn each, which is what [`Self::answer`] alone can
    /// express. A template that batches a run into a single turn overrides it.
    fn answer_all(&self, observations: &[ToolObservation]) -> Vec<u32> {
        let mut tokens = Vec::new();
        for observation in observations {
            tokens.extend(self.answer(&observation.name, &observation.value));
        }
        tokens
    }
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
