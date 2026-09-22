//! The generated token stream classified into what an API returns.
//!
//! The host offers three decoders over the same tokens: `chat` (text and
//! the turn's end), `reasoning` (thinking blocks) and `tools` (complete tool
//! calls). They are independent — each sees every token — so this demux
//! routes each token to the one that owns it: a token inside a thinking
//! block is reasoning, a token inside a tool call is a call, and everything
//! else is content.
//!
//! One seam: the tools decoder reports a call only once it is complete, and
//! its "start" event is not distinguishable from "nothing" at the host
//! boundary. The start of a call is therefore caught on the content side,
//! by holding back the tool-call marker text (`<tool_call>` on every
//! tool-capable template the host has) so it never streams out as content.

use inferlet::chat;
use inferlet::pie::inferlet::{reasoning, tools};

use crate::holdback::HoldBack;
use crate::request::ToolCall;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A chunk of the model's reasoning.
    Reasoning(String),
    /// The reasoning block closed.
    ReasoningEnd,
    /// A chunk of content.
    Text(String),
    /// A complete tool call.
    ToolCall(ToolCall),
}

pub(crate) struct Demux {
    reasoning: reasoning::Decoder,
    tools: tools::Decoder,
    chat: chat::Decoder,
    in_reasoning: bool,
    /// Drop whitespace at the start of a reasoning block: the template opens
    /// it with a newline.
    reasoning_strip_leading: bool,
    in_tool_call: bool,
    content_filter: HoldBack,
    /// Drop whitespace at the start of the next content: the template puts
    /// a blank line after a thinking block or a tool call, and that is
    /// formatting, not an answer.
    strip_leading: bool,
    pub reasoning_text: String,
    pub tool_calls: Vec<ToolCall>,
}

impl Demux {
    pub fn new() -> Self {
        Self {
            reasoning: reasoning::Decoder::new(),
            tools: tools::Decoder::new(),
            chat: chat::Decoder::new(),
            in_reasoning: false,
            reasoning_strip_leading: true,
            in_tool_call: false,
            content_filter: HoldBack::new(vec!["<tool_call>".to_string()]),
            strip_leading: true,
            reasoning_text: String::new(),
            tool_calls: Vec::new(),
        }
    }

    /// Classify one token. Stop tokens must not be fed: the caller ends the
    /// turn on them and then calls [`Demux::finish`].
    pub fn feed(&mut self, token: u32) -> Result<Vec<Event>, String> {
        let mut events = Vec::new();

        match self
            .reasoning
            .feed(&[token])
            .map_err(|e| format!("reasoning decoder: {e:?}"))?
        {
            reasoning::Event::Start => {
                self.in_reasoning = true;
                self.reasoning_strip_leading = true;
                return Ok(events);
            }
            reasoning::Event::Complete(_) => {
                self.in_reasoning = false;
                self.strip_leading = true;
                events.push(Event::ReasoningEnd);
                return Ok(events);
            }
            reasoning::Event::Delta(delta) if self.in_reasoning => {
                let delta = if self.reasoning_strip_leading {
                    delta.trim_start().to_string()
                } else {
                    delta
                };
                if !delta.is_empty() {
                    self.reasoning_strip_leading = false;
                    self.reasoning_text.push_str(&delta);
                    events.push(Event::Reasoning(delta));
                }
                return Ok(events);
            }
            reasoning::Event::Delta(_) => {}
        }

        match self
            .tools
            .feed(&[token])
            .map_err(|e| format!("tools decoder: {e:?}"))?
        {
            tools::Event::Call(call) => {
                self.in_tool_call = false;
                self.strip_leading = true;
                self.content_filter.flush();
                let call = ToolCall {
                    id: crate::call_id(self.tool_calls.len()),
                    name: call.name,
                    arguments: call.arguments_json,
                };
                self.tool_calls.push(call.clone());
                events.push(Event::ToolCall(call));
                return Ok(events);
            }
            tools::Event::Start => {}
        }

        match self
            .chat
            .feed(&[token])
            .map_err(|e| format!("chat decoder: {e:?}"))?
        {
            chat::Event::Delta(delta) => {
                if self.in_tool_call {
                    return Ok(events);
                }
                let released = self.content_filter.push(&delta);
                self.emit_text(&released.text, &mut events);
                if released.hit.is_some() {
                    self.in_tool_call = true;
                }
            }
            chat::Event::Done(_) | chat::Event::Interrupt(_) => {}
        }
        Ok(events)
    }

    /// The stream ended: release held content.
    pub fn finish(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        if !self.in_tool_call {
            let rest = self.content_filter.flush();
            self.emit_text(&rest, &mut events);
        }
        events
    }

    fn emit_text(&mut self, text: &str, events: &mut Vec<Event>) {
        let text = if self.strip_leading {
            let trimmed = text.trim_start();
            if trimmed.is_empty() {
                return;
            }
            self.strip_leading = false;
            trimmed
        } else {
            text
        };
        events.push(Event::Text(text.to_string()));
    }
}
