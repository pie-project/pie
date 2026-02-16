//! Instruct extension trait for `Context`.
//!
//! Thin ergonomic wrappers over the `pie:instruct` WIT interfaces
//! (chat, tool-use, reasoning) so inferlets can call methods directly
//! on a `Context` instead of passing it as an argument to free functions.

use crate::context::Context;
use crate::model::Model;
use crate::Result;

// Re-export the instruct sub-interface types for convenience.
pub use crate::pie::instruct::chat::{
    self as chat,
    Decoder as ChatDecoder,
    Event as ChatEvent,
};
pub use crate::pie::instruct::tool_use::{
    self as tool_use,
    Decoder as ToolDecoder,
    Event as ToolEvent,
};
pub use crate::pie::instruct::reasoning::{
    self as reasoning,
    Decoder as ReasoningDecoder,
    Event as ReasoningEvent,
};

// Re-export the matcher type from core inference.
pub use crate::pie::core::inference::Matcher;

/// Extension trait that lifts `pie:instruct` free functions into methods on [`Context`].
pub trait InstructExt {
    // ── Chat ────────────────────────────────────────────────────────

    /// Fill a system message into the context.
    fn system(&self, message: &str);

    /// Fill a user message into the context.
    fn user(&self, message: &str);

    /// Fill an assistant message (for history replay).
    fn assistant(&self, message: &str);

    /// Cue the model to generate (fills the generation header).
    fn cue(&self);

    /// Seal the current turn (inserts a stop token).
    fn seal(&self);

    /// Returns the stop token IDs for the given model.
    fn stop_tokens(model: &Model) -> Vec<u32>;

    /// Create a chat decoder to classify generated tokens
    /// into delta / interrupt / done events.
    fn create_chat_decoder(model: &Model) -> ChatDecoder;

    // ── Tool Use ────────────────────────────────────────────────────

    /// Register available tools (list of JSON schema strings).
    fn equip_tools(&self, tools: &[String]) -> Result<()>;

    /// Provide a tool result after a tool call.
    fn answer_tool(&self, name: &str, value: &str);

    /// Create a tool-call decoder to detect tool calls in generated tokens.
    fn create_tool_decoder(model: &Model) -> ToolDecoder;

    /// Create a grammar matcher that constrains generation to valid tool calls.
    fn create_tool_matcher(model: &Model, tools: &[String]) -> Matcher;

    // ── Reasoning ───────────────────────────────────────────────────

    /// Create a reasoning decoder to detect reasoning blocks in generated tokens.
    fn create_reasoning_decoder(model: &Model) -> ReasoningDecoder;
}

impl InstructExt for Context {
    // ── Chat ────────────────────────────────────────────────────────

    fn system(&self, message: &str) {
        chat::system(self, message);
    }

    fn user(&self, message: &str) {
        chat::user(self, message);
    }

    fn assistant(&self, message: &str) {
        chat::assistant(self, message);
    }

    fn cue(&self) {
        chat::cue(self);
    }

    fn seal(&self) {
        chat::seal(self);
    }

    fn stop_tokens(model: &Model) -> Vec<u32> {
        chat::stop_tokens(model)
    }

    fn create_chat_decoder(model: &Model) -> ChatDecoder {
        chat::create_decoder(model)
    }

    // ── Tool Use ────────────────────────────────────────────────────

    fn equip_tools(&self, tools: &[String]) -> Result<()> {
        tool_use::equip(self, &tools.to_vec())
    }

    fn answer_tool(&self, name: &str, value: &str) {
        tool_use::answer(self, name, value);
    }

    fn create_tool_decoder(model: &Model) -> ToolDecoder {
        tool_use::create_decoder(model)
    }

    fn create_tool_matcher(model: &Model, tools: &[String]) -> Matcher {
        tool_use::create_matcher(model, &tools.to_vec())
    }

    // ── Reasoning ───────────────────────────────────────────────────

    fn create_reasoning_decoder(model: &Model) -> ReasoningDecoder {
        reasoning::create_decoder(model)
    }
}

// =============================================================================
// Unified Decoder
// =============================================================================

/// Unified event emitted by [`Decoder`].
///
/// Merges chat, reasoning, and tool-use decoder outputs into a single enum
/// so callers only need one `match` arm per token.
pub enum Event {
    /// Raw token with no semantic significance (yet).
    Token,
    /// Generated text chunk (outside of reasoning blocks).
    Text(String),
    /// Reasoning/thinking text chunk.
    Thinking(String),
    /// Reasoning block complete (full accumulated thinking text).
    ThinkingDone(String),
    /// A complete tool call was detected: (name, arguments_json).
    ToolCall(String, String),
    /// Generation complete (full accumulated response text).
    Done(String),
}

/// Tracks whether we are inside a reasoning (thinking) block.
enum DecoderState {
    /// Normal text generation.
    Normal,
    /// Inside a reasoning block — chat deltas are suppressed,
    /// reasoning deltas are forwarded as `Event::Thinking`.
    Reasoning,
}

/// Unified decoder that internally muxes the chat, reasoning, and tool-use
/// WIT decoder resources.
///
/// Created via [`Model::decoder`] or [`Decoder::new`], then configured
/// with builder methods and fed tokens one batch at a time.
///
/// ```ignore
/// let mut decoder = model.decoder()
///     .with_reasoning()
///     .with_tool_use();
///
/// let mut stream = ctx.generate(sampler);
/// while let Some(tokens) = stream.next().await? {
///     match decoder.feed(&tokens)? {
///         Event::Text(s) => { /* ... */ }
///         Event::Done(s) => break,
///         _ => {}
///     }
/// }
/// ```
///
/// Event priority:
/// 1. Reasoning transitions (start / complete) override chat deltas.
/// 2. Tool calls override chat deltas.
/// 3. Chat done is always forwarded.
/// 4. Otherwise, chat deltas are forwarded as `Event::Text`.
pub struct Decoder {
    chat: ChatDecoder,
    reasoning: ReasoningDecoder,
    tools: ToolDecoder,
    use_reasoning: bool,
    use_tools: bool,
    state: DecoderState,
}

impl Decoder {
    /// Create a decoder (chat-only by default).
    ///
    /// Prefer [`Model::decoder`] for ergonomics.
    pub fn new(model: &Model) -> Self {
        Self {
            chat: chat::create_decoder(model),
            reasoning: reasoning::create_decoder(model),
            tools: tool_use::create_decoder(model),
            use_reasoning: false,
            use_tools: false,
            state: DecoderState::Normal,
        }
    }

    /// Enable reasoning (thinking block) decoding.
    pub fn with_reasoning(mut self) -> Self {
        self.use_reasoning = true;
        self
    }

    /// Enable tool-use decoding.
    pub fn with_tool_use(mut self) -> Self {
        self.use_tools = true;
        self
    }

    /// Feed a batch of token IDs and get back a single unified event.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Event> {
        // 1. Reasoning decoder (highest priority for state transitions)
        if self.use_reasoning {
            match self.reasoning.feed(tokens)? {
                ReasoningEvent::Start => {
                    self.state = DecoderState::Reasoning;
                    let _ = self.chat.feed(tokens);
                    if self.use_tools {
                        let _ = self.tools.feed(tokens);
                    }
                    return Ok(Event::Thinking(String::new()));
                }
                ReasoningEvent::Complete(s) => {
                    self.state = DecoderState::Normal;
                    let _ = self.chat.feed(tokens);
                    if self.use_tools {
                        let _ = self.tools.feed(tokens);
                    }
                    return Ok(Event::ThinkingDone(s));
                }
                ReasoningEvent::Delta(s) => {
                    if matches!(self.state, DecoderState::Reasoning) && !s.is_empty() {
                        let _ = self.chat.feed(tokens);
                        if self.use_tools {
                            let _ = self.tools.feed(tokens);
                        }
                        return Ok(Event::Thinking(s));
                    }
                }
            }
        }

        // 2. Tool decoder
        if self.use_tools {
            match self.tools.feed(tokens)? {
                ToolEvent::Call(name_args) => {
                    let _ = self.chat.feed(tokens);
                    return Ok(Event::ToolCall(name_args.0, name_args.1));
                }
                ToolEvent::Start => {}
            }
        }

        // 3. Chat decoder (lowest priority)
        match self.chat.feed(tokens)? {
            ChatEvent::Done(s) => Ok(Event::Done(s)),
            ChatEvent::Delta(s) => {
                if matches!(self.state, DecoderState::Reasoning) {
                    Ok(Event::Token)
                } else {
                    Ok(Event::Text(s))
                }
            }
            ChatEvent::Interrupt(_) => Ok(Event::Token),
        }
    }

    /// Reset all sub-decoders to their initial state.
    pub fn reset(&mut self) {
        self.chat.reset();
        if self.use_reasoning {
            self.reasoning.reset();
        }
        if self.use_tools {
            self.tools.reset();
        }
        self.state = DecoderState::Normal;
    }
}

