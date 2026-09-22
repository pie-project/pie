//! Anthropic Messages (`POST /v1/messages`) over the loaded model.
//!
//! A translation: the Messages request becomes a [`compat::Request`], and
//! the [`compat::Event`] stream becomes content blocks — one `message`
//! body, or the named SSE events (`message_start`, `content_block_start`,
//! `content_block_delta`, `content_block_stop`, `message_delta`,
//! `message_stop`) with `stream: true`. Reasoning is a `thinking` block.
//!
//! Supported: `system` (string or text blocks), `messages` with `text`,
//! `tool_use` and `tool_result` blocks, `tools` (custom tools with
//! `input_schema`), `tool_choice` (`auto`/`any`/`tool`/`none`), `thinking`
//! (`enabled`/`disabled`; off when absent, as the API says),
//! `stop_sequences`, `temperature`, `top_p`, `top_k`, `max_tokens`
//! (required), `stream`.
//!
//! Refused with 400: image and document blocks, built-in tool types, and
//! more than one candidate. `anthropic-version` and `metadata` are ignored.

use compat::{
    ApiError, Envelope, Event, Finish, Message, Request, Role, Thinking, ToolCall, ToolChoice, emit,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize, Default)]
#[serde(default)]
struct Body {
    max_tokens: Option<usize>,
    messages: Vec<Msg>,
    system: Option<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    thinking: Option<Value>,
    stop_sequences: Option<Vec<String>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stream: bool,
}

#[derive(Deserialize)]
struct Msg {
    role: String,
    content: Value,
}

fn invalid(message: impl Into<String>) -> ApiError {
    ApiError::new(
        400,
        json!({"type": "error", "error": {"type": "invalid_request_error", "message": message.into()}}),
    )
}

/// Text blocks (or a bare string) joined.
fn text_of(value: &Value, what: &str) -> Result<String, ApiError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut text = String::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        text.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    Some(other) => {
                        return Err(invalid(format!(
                            "{what}: block type `{other}` is not supported here"
                        )));
                    }
                    None => return Err(invalid(format!("{what}: block has no `type`"))),
                }
            }
            Ok(text)
        }
        Value::Null => Ok(String::new()),
        _ => Err(invalid(format!(
            "{what}: must be a string or an array of blocks"
        ))),
    }
}

fn parse(body: Value) -> Result<Request, ApiError> {
    let body: Body =
        serde_json::from_value(body).map_err(|e| invalid(format!("request body: {e}")))?;
    let max_tokens = body
        .max_tokens
        .ok_or_else(|| invalid("`max_tokens` is required"))?;
    if max_tokens == 0 {
        return Err(invalid("`max_tokens` must be at least 1"));
    }
    if body.messages.is_empty() {
        return Err(invalid("`messages` must not be empty"));
    }

    let mut messages = Vec::new();
    if let Some(system) = &body.system {
        let text = text_of(system, "system")?;
        if !text.is_empty() {
            messages.push(Message::text(Role::System, text));
        }
    }

    let mut call_names: Vec<(String, String)> = Vec::new();
    for (i, m) in body.messages.iter().enumerate() {
        let what = format!("messages[{i}]");
        match m.role.as_str() {
            "user" => match &m.content {
                Value::String(s) => messages.push(Message::text(Role::User, s.clone())),
                Value::Array(blocks) => {
                    let mut text = String::new();
                    for block in blocks {
                        match block.get("type").and_then(Value::as_str) {
                            Some("text") => {
                                text.push_str(
                                    block.get("text").and_then(Value::as_str).unwrap_or(""),
                                );
                            }
                            Some("tool_result") => {
                                let id = block
                                    .get("tool_use_id")
                                    .and_then(Value::as_str)
                                    .unwrap_or("");
                                let content = match block.get("content") {
                                    Some(v) => text_of(v, &format!("{what}.tool_result"))?,
                                    None => String::new(),
                                };
                                let mut result = Message::text(Role::Tool, content);
                                result.tool_name = call_names
                                    .iter()
                                    .find(|(call_id, _)| call_id == id)
                                    .map(|(_, name)| name.clone());
                                messages.push(result);
                            }
                            Some(other) => {
                                return Err(invalid(format!(
                                    "{what}: block type `{other}` is not supported; only `text` and `tool_result` are"
                                )));
                            }
                            None => return Err(invalid(format!("{what}: block has no `type`"))),
                        }
                    }
                    if !text.is_empty() {
                        messages.push(Message::text(Role::User, text));
                    }
                }
                _ => {
                    return Err(invalid(format!(
                        "{what}: content must be a string or blocks"
                    )));
                }
            },
            "assistant" => {
                let mut message = Message::text(Role::Assistant, String::new());
                match &m.content {
                    Value::String(s) => message.content = s.clone(),
                    Value::Array(blocks) => {
                        for block in blocks {
                            match block.get("type").and_then(Value::as_str) {
                                Some("text") => message.content.push_str(
                                    block.get("text").and_then(Value::as_str).unwrap_or(""),
                                ),
                                Some("tool_use") => {
                                    let id = block
                                        .get("id")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string();
                                    let name = block
                                        .get("name")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string();
                                    let arguments = block
                                        .get("input")
                                        .cloned()
                                        .unwrap_or(json!({}))
                                        .to_string();
                                    call_names.push((id.clone(), name.clone()));
                                    message.tool_calls.push(ToolCall {
                                        id,
                                        name,
                                        arguments,
                                    });
                                }
                                Some("thinking") | Some("redacted_thinking") => {}
                                Some(other) => {
                                    return Err(invalid(format!(
                                        "{what}: block type `{other}` is not supported"
                                    )));
                                }
                                None => {
                                    return Err(invalid(format!("{what}: block has no `type`")));
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(invalid(format!(
                            "{what}: content must be a string or blocks"
                        )));
                    }
                }
                messages.push(message);
            }
            other => return Err(invalid(format!("{what}: unknown role `{other}`"))),
        }
    }

    let mut req = Request::new(messages);
    req.max_tokens = max_tokens;
    req.stream = body.stream;
    for (i, tool) in body.tools.unwrap_or_default().iter().enumerate() {
        if let Some(kind) = tool.get("type").and_then(Value::as_str)
            && kind != "custom"
        {
            return Err(invalid(format!(
                "tools[{i}]: built-in tool type `{kind}` is not supported"
            )));
        }
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid(format!("tools[{i}]: `name` is required")))?;
        let function = json!({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description").cloned().unwrap_or(Value::Null),
                "parameters": tool.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
            },
        });
        req.tools.push(function.to_string());
    }
    req.tool_choice = match body
        .tool_choice
        .as_ref()
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str)
    {
        None | Some("auto") => ToolChoice::Auto,
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => {
            let name = body
                .tool_choice
                .as_ref()
                .and_then(|c| c.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("tool_choice of type `tool` needs `name`"))?;
            ToolChoice::Named(name.to_string())
        }
        Some(other) => return Err(invalid(format!("unknown tool_choice type `{other}`"))),
    };
    if req.tools.is_empty() && req.forces_tool_call() {
        return Err(invalid("tool_choice forces a call but `tools` is empty"));
    }
    req.thinking = match body
        .thinking
        .as_ref()
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
    {
        Some("enabled") => Thinking::Enabled,
        _ => Thinking::Disabled,
    };
    if let Some(t) = body.temperature {
        req.sampling.temperature = t;
    }
    if let Some(p) = body.top_p {
        req.sampling.top_p = p;
    }
    if let Some(k) = body.top_k {
        req.sampling.top_k = k;
    }
    req.sampling.validate().map_err(invalid)?;
    req.stop = body.stop_sequences.unwrap_or_default();
    Ok(req)
}

fn stop_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop => "end_turn",
        Finish::Length => "max_tokens",
        Finish::ToolCalls => "tool_use",
        Finish::StopSequence => "stop_sequence",
    }
}

fn tool_use_block(index: usize, call: &ToolCall) -> Value {
    let input: Value = serde_json::from_str(&call.arguments).unwrap_or(json!({}));
    let id = format!("toolu_{}_{index}", compat::short_id());
    json!({"type": "tool_use", "id": id, "name": call.name, "input": input})
}

fn usage(input: usize, output: usize) -> Value {
    json!({"input_tokens": input, "output_tokens": output})
}

/// The open content block of a streamed reply.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Thinking,
    Text,
}

struct Stream {
    index: usize,
    open: Option<Block>,
    calls: usize,
}

impl Stream {
    fn close(&mut self) {
        if self.open.take().is_some() {
            emit::send_event(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": self.index}),
            );
            self.index += 1;
        }
    }

    fn ensure(&mut self, block: Block) {
        if self.open == Some(block) {
            return;
        }
        self.close();
        let content_block = match block {
            Block::Thinking => json!({"type": "thinking", "thinking": ""}),
            Block::Text => json!({"type": "text", "text": ""}),
        };
        emit::send_event(
            "content_block_start",
            &json!({"type": "content_block_start", "index": self.index, "content_block": content_block}),
        );
        self.open = Some(block);
    }

    fn delta(&self, delta: Value) {
        emit::send_event(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        );
    }

    fn on(&mut self, event: Event) {
        match event {
            Event::Reasoning(text) => {
                self.ensure(Block::Thinking);
                self.delta(json!({"type": "thinking_delta", "thinking": text}));
            }
            Event::ReasoningEnd => self.close(),
            Event::Text(text) => {
                self.ensure(Block::Text);
                self.delta(json!({"type": "text_delta", "text": text}));
            }
            Event::ToolCall(call) => {
                self.close();
                let block = tool_use_block(self.calls, &call);
                let mut start = block.clone();
                start["input"] = json!({});
                emit::send_event(
                    "content_block_start",
                    &json!({"type": "content_block_start", "index": self.index, "content_block": start}),
                );
                self.delta(json!({"type": "input_json_delta", "partial_json": call.arguments}));
                emit::send_event(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": self.index}),
                );
                self.index += 1;
                self.calls += 1;
            }
        }
    }
}

#[inferlet::main]
async fn main(input: String) -> Result<String, String> {
    let envelope = Envelope::parse(&input).map_err(invalid)?;
    let req = parse(envelope.body)?;
    let id = format!("msg_{}", compat::short_id());
    let model = inferlet::model::name();
    let stream = req.stream;

    if stream {
        // Clients read `input_tokens` off `message_start`, which goes out
        // before generation begins, so the prompt is rendered once here for
        // its count and again inside `run`.
        let prompt_tokens = compat::prompt::build(&req)?.len();
        emit::send_event(
            "message_start",
            &json!({"type": "message_start", "message": {
                "id": id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": usage(prompt_tokens, 0),
            }}),
        );
    }
    let mut state = Stream {
        index: 0,
        open: None,
        calls: 0,
    };
    let outcome = compat::run(&req, |event| {
        if stream {
            state.on(event);
        }
    })
    .await?;

    if stream {
        if state.index == 0 && state.open.is_none() {
            state.ensure(Block::Text);
        }
        state.close();
        emit::send_event(
            "message_delta",
            &json!({"type": "message_delta",
                "delta": {"stop_reason": stop_reason(outcome.finish), "stop_sequence": outcome.stop_sequence},
                "usage": usage(outcome.prompt_tokens, outcome.completion_tokens)}),
        );
        emit::send_event("message_stop", &json!({"type": "message_stop"}));
        return Ok(String::new());
    }

    let mut content = Vec::new();
    let reasoning = outcome.reasoning.trim();
    if !reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": reasoning, "signature": ""}));
    }
    if !outcome.content.is_empty() || outcome.tool_calls.is_empty() {
        content.push(json!({"type": "text", "text": outcome.content}));
    }
    for (i, call) in outcome.tool_calls.iter().enumerate() {
        content.push(tool_use_block(i, call));
    }
    Ok(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(outcome.finish),
        "stop_sequence": outcome.stop_sequence,
        "usage": usage(outcome.prompt_tokens, outcome.completion_tokens),
    })
    .to_string())
}
