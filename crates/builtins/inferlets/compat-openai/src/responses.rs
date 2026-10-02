//! OpenAI Responses (`POST /v1/responses`) over the loaded model.
//!
//! A translation: `input` items become a [`compat::Request`], and the
//! [`compat::Event`] stream becomes output items — a `reasoning` item, a
//! `message` with `output_text`, and `function_call` items — either as one
//! `response` body or as the named streaming events (`response.created`,
//! `response.output_item.added`, `response.output_text.delta`, …,
//! `response.completed`).
//!
//! Supported: `input` as a string or as message / `function_call` /
//! `function_call_output` items, `instructions`, function `tools`,
//! `tool_choice`, `text.format` (`text`/`json_object`/`json_schema`),
//! `reasoning.effort` (`none` switches thinking off), `max_output_tokens`,
//! `temperature`, `top_p`, `top_k`, `stream`.
//!
//! Refused with 400: `previous_response_id` (this server keeps no
//! conversation state), non-text input parts, and built-in tool types.

use compat::{
    ApiError, Envelope, Event, Finish, Message, Request, ResponseFormat, Role, Thinking, ToolCall,
    ToolChoice, emit,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::invalid;

#[derive(Deserialize, Default)]
#[serde(default)]
struct Body {
    input: Value,
    instructions: Option<String>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    text: Option<Value>,
    reasoning: Option<Value>,
    max_output_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stream: bool,
    previous_response_id: Option<String>,
}

fn content_text(content: &Value, what: &str) -> Result<String, ApiError> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("output_text") | Some("text") => {
                        text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                    Some(other) => {
                        return Err(invalid(
                            format!(
                                "{what}: part type `{other}` is not supported; only text parts are"
                            ),
                            Some("input"),
                        ));
                    }
                    None => {
                        return Err(invalid(
                            format!("{what}: part has no `type`"),
                            Some("input"),
                        ));
                    }
                }
            }
            Ok(text)
        }
        Value::Null => Ok(String::new()),
        _ => Err(invalid(
            format!("{what}: content must be a string or parts"),
            Some("input"),
        )),
    }
}

fn parse(body: Value) -> Result<Request, ApiError> {
    let body: Body =
        serde_json::from_value(body).map_err(|e| invalid(format!("request body: {e}"), None))?;
    if body.previous_response_id.is_some() {
        return Err(invalid(
            "`previous_response_id` is not supported: this server keeps no conversation state; send the whole conversation as `input`",
            Some("previous_response_id"),
        ));
    }
    let mut messages = Vec::new();
    if let Some(instructions) = body.instructions.filter(|s| !s.is_empty()) {
        messages.push(Message::text(Role::System, instructions));
    }
    let mut call_names: Vec<(String, String)> = Vec::new();
    match &body.input {
        Value::String(s) => messages.push(Message::text(Role::User, s.clone())),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                let what = format!("input[{i}]");
                match item.get("type").and_then(Value::as_str) {
                    None | Some("message") => {
                        let role = match item.get("role").and_then(Value::as_str) {
                            Some("user") => Role::User,
                            Some("assistant") => Role::Assistant,
                            Some("system") | Some("developer") => Role::System,
                            Some(other) => {
                                return Err(invalid(
                                    format!("{what}: unknown role `{other}`"),
                                    Some("input"),
                                ));
                            }
                            None => {
                                return Err(invalid(
                                    format!("{what}: message has no `role`"),
                                    Some("input"),
                                ));
                            }
                        };
                        let text =
                            content_text(item.get("content").unwrap_or(&Value::Null), &what)?;
                        messages.push(Message::text(role, text));
                    }
                    Some("function_call") => {
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let arguments = match item.get("arguments") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                            None => "{}".to_string(),
                        };
                        call_names.push((call_id.clone(), name.clone()));
                        let call = ToolCall {
                            id: call_id,
                            name,
                            arguments,
                        };
                        match messages.last_mut() {
                            Some(last) if last.role == Role::Assistant => {
                                last.tool_calls.push(call)
                            }
                            _ => {
                                let mut m = Message::text(Role::Assistant, String::new());
                                m.tool_calls.push(call);
                                messages.push(m);
                            }
                        }
                    }
                    Some("function_call_output") => {
                        let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                        let output = match item.get("output") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                            None => String::new(),
                        };
                        let mut m = Message::text(Role::Tool, output);
                        m.tool_name = call_names
                            .iter()
                            .find(|(id, _)| id == call_id)
                            .map(|(_, name)| name.clone());
                        messages.push(m);
                    }
                    Some("reasoning") => {}
                    Some(other) => {
                        return Err(invalid(
                            format!("{what}: item type `{other}` is not supported"),
                            Some("input"),
                        ));
                    }
                }
            }
        }
        Value::Null => return Err(invalid("`input` is required", Some("input"))),
        _ => {
            return Err(invalid(
                "`input` must be a string or an array of items",
                Some("input"),
            ));
        }
    }
    if messages.is_empty() {
        return Err(invalid("`input` must not be empty", Some("input")));
    }

    let mut req = Request::new(messages);
    req.stream = body.stream;
    for (i, tool) in body.tools.unwrap_or_default().iter().enumerate() {
        match tool.get("type").and_then(Value::as_str) {
            Some("function") => {
                let name = tool.get("name").and_then(Value::as_str).ok_or_else(|| {
                    invalid(format!("tools[{i}]: `name` is required"), Some("tools"))
                })?;
                let function = json!({"type": "function", "function": {
                    "name": name,
                    "description": tool.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": tool.get("parameters").cloned().unwrap_or(json!({"type": "object"})),
                }});
                req.tools.push(function.to_string());
            }
            Some(other) => {
                return Err(invalid(
                    format!("tools[{i}]: tool type `{other}` is not supported"),
                    Some("tools"),
                ));
            }
            None => {
                return Err(invalid(
                    format!("tools[{i}]: tool has no `type`"),
                    Some("tools"),
                ));
            }
        }
    }
    req.tool_choice = match body.tool_choice {
        None => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => {
                return Err(invalid(
                    format!("unknown tool_choice `{other}`"),
                    Some("tool_choice"),
                ));
            }
        },
        Some(Value::Object(o)) => {
            let name = o
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("tool_choice object needs `name`", Some("tool_choice")))?;
            ToolChoice::Named(name.to_string())
        }
        Some(_) => {
            return Err(invalid(
                "tool_choice must be a string or an object",
                Some("tool_choice"),
            ));
        }
    };
    if req.tools.is_empty() && req.forces_tool_call() {
        return Err(invalid(
            "tool_choice forces a call but `tools` is empty",
            Some("tool_choice"),
        ));
    }
    let format = body.text.as_ref().and_then(|t| t.get("format"));
    req.response_format = match format.and_then(|f| f.get("type")).and_then(Value::as_str) {
        None | Some("text") => ResponseFormat::Text,
        Some("json_object") => ResponseFormat::JsonObject,
        Some("json_schema") => {
            let schema = format.and_then(|f| f.get("schema")).ok_or_else(|| {
                invalid(
                    "text.format of type json_schema needs `schema`",
                    Some("text"),
                )
            })?;
            ResponseFormat::JsonSchema(schema.to_string())
        }
        Some(other) => {
            return Err(invalid(
                format!("unknown text.format type `{other}`"),
                Some("text"),
            ));
        }
    };
    req.thinking = match body
        .reasoning
        .as_ref()
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str)
    {
        Some("none") => Thinking::Disabled,
        Some(_) => Thinking::Enabled,
        None => Thinking::ModelDefault,
    };
    if let Some(max) = body.max_output_tokens {
        if max == 0 {
            return Err(invalid(
                "max_output_tokens must be at least 1",
                Some("max_output_tokens"),
            ));
        }
        req.max_tokens = max;
    }
    if let Some(t) = body.temperature {
        req.sampling.temperature = t;
    }
    if let Some(p) = body.top_p {
        req.sampling.top_p = p;
    }
    if let Some(k) = body.top_k {
        req.sampling.top_k = k;
    }
    req.sampling.validate().map_err(|e| invalid(e, None))?;
    Ok(req)
}

fn usage(input: usize, output: usize) -> Value {
    json!({
        "input_tokens": input,
        "output_tokens": output,
        "total_tokens": input + output,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    })
}

/// The response object, with whatever `output` and `status` it has now.
struct Shell {
    id: String,
    created_at: u64,
    model: String,
}

impl Shell {
    fn response(
        &self,
        status: &str,
        output: Vec<Value>,
        usage: Option<Value>,
        finish: Option<Finish>,
    ) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "error": null,
            "incomplete_details": match finish {
                Some(Finish::Length) => json!({"reason": "max_output_tokens"}),
                _ => Value::Null,
            },
            "model": self.model,
            "output": output,
            "usage": usage,
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "store": false,
            "metadata": {},
        })
    }
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({"type": "reasoning", "id": id, "summary": [], "content": [{"type": "reasoning_text", "text": text}]})
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({"type": "message", "id": id, "status": status, "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]})
}

fn function_call_item(id: &str, call: &ToolCall, status: &str) -> Value {
    json!({"type": "function_call", "id": id, "call_id": call.id, "name": call.name,
        "arguments": call.arguments, "status": status})
}

/// The open output item of a streamed reply.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Item {
    Reasoning,
    Message,
}

struct Stream {
    seq: u64,
    index: usize,
    open: Option<Item>,
    item_id: String,
    text: String,
    reasoning: String,
    output: Vec<Value>,
}

impl Stream {
    fn event(&mut self, kind: &str, mut data: Value) {
        data["type"] = Value::String(kind.to_string());
        data["sequence_number"] = Value::from(self.seq);
        self.seq += 1;
        emit::send_event(kind, &data);
    }

    fn close(&mut self) {
        let Some(item) = self.open.take() else {
            return;
        };
        let id = self.item_id.clone();
        let done = match item {
            Item::Reasoning => {
                let text = std::mem::take(&mut self.reasoning);
                self.event("response.reasoning_text.done", json!({"item_id": id, "output_index": self.index, "content_index": 0, "text": text}));
                reasoning_item(&id, &text)
            }
            Item::Message => {
                let text = std::mem::take(&mut self.text);
                self.event("response.output_text.delta", json!({"item_id": id, "output_index": self.index, "content_index": 0, "delta": "", "logprobs": []}));
                self.event("response.output_text.done", json!({"item_id": id, "output_index": self.index, "content_index": 0, "text": text, "logprobs": []}));
                let part =
                    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []});
                self.event("response.content_part.done", json!({"item_id": id, "output_index": self.index, "content_index": 0, "part": part}));
                message_item(&id, &text, "completed")
            }
        };
        self.event(
            "response.output_item.done",
            json!({"output_index": self.index, "item": done.clone()}),
        );
        self.output.push(done);
        self.index += 1;
    }

    fn ensure(&mut self, item: Item) {
        if self.open == Some(item) {
            return;
        }
        self.close();
        let id = match item {
            Item::Reasoning => format!("rs_{}_{}", compat::short_id(), self.index),
            Item::Message => format!("msg_{}_{}", compat::short_id(), self.index),
        };
        let added = match item {
            Item::Reasoning => json!({"type": "reasoning", "id": id, "summary": []}),
            Item::Message => message_item(&id, "", "in_progress"),
        };
        self.event(
            "response.output_item.added",
            json!({"output_index": self.index, "item": added}),
        );
        if item == Item::Message {
            let part =
                json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []});
            self.event("response.content_part.added", json!({"item_id": id, "output_index": self.index, "content_index": 0, "part": part}));
        }
        self.item_id = id;
        self.open = Some(item);
    }

    fn on(&mut self, event: Event) {
        match event {
            Event::Reasoning(delta) => {
                self.ensure(Item::Reasoning);
                self.reasoning.push_str(&delta);
                let id = self.item_id.clone();
                self.event("response.reasoning_text.delta", json!({"item_id": id, "output_index": self.index, "content_index": 0, "delta": delta}));
            }
            Event::ReasoningEnd => self.close(),
            Event::Text(delta) => {
                self.ensure(Item::Message);
                self.text.push_str(&delta);
                let id = self.item_id.clone();
                self.event("response.output_text.delta", json!({"item_id": id, "output_index": self.index, "content_index": 0, "delta": delta, "logprobs": []}));
            }
            Event::ToolCall(call) => {
                self.close();
                let id = format!("fc_{}_{}", compat::short_id(), self.index);
                let mut added = function_call_item(&id, &call, "in_progress");
                added["arguments"] = Value::String(String::new());
                self.event(
                    "response.output_item.added",
                    json!({"output_index": self.index, "item": added}),
                );
                self.event(
                    "response.function_call_arguments.delta",
                    json!({"item_id": id, "output_index": self.index, "delta": call.arguments}),
                );
                self.event(
                    "response.function_call_arguments.done",
                    json!({"item_id": id, "output_index": self.index, "arguments": call.arguments}),
                );
                let done = function_call_item(&id, &call, "completed");
                self.event(
                    "response.output_item.done",
                    json!({"output_index": self.index, "item": done.clone()}),
                );
                self.output.push(done);
                self.index += 1;
            }
        }
    }
}

pub async fn serve(envelope: Envelope) -> Result<String, String> {
    let req = parse(envelope.body)?;
    let shell = Shell {
        id: format!("resp_{}", compat::short_id()),
        created_at: envelope.time,
        model: inferlet::model::name(),
    };
    let stream = req.stream;
    let mut state = Stream {
        seq: 0,
        index: 0,
        open: None,
        item_id: String::new(),
        text: String::new(),
        reasoning: String::new(),
        output: Vec::new(),
    };
    if stream {
        let created = shell.response("in_progress", Vec::new(), None, None);
        state.event("response.created", json!({"response": created.clone()}));
        state.event("response.in_progress", json!({"response": created}));
    }
    let outcome = compat::run(&req, |event| {
        if stream {
            state.on(event);
        }
    })
    .await?;

    let status = if outcome.finish == Finish::Length {
        "incomplete"
    } else {
        "completed"
    };
    if stream {
        if state.output.is_empty() && state.open.is_none() {
            state.ensure(Item::Message);
        }
        state.close();
        let output = std::mem::take(&mut state.output);
        let usage = usage(outcome.prompt_tokens, outcome.completion_tokens);
        let response = shell.response(status, output, Some(usage), Some(outcome.finish));
        let kind = if status == "completed" {
            "response.completed"
        } else {
            "response.incomplete"
        };
        state.event(kind, json!({"response": response}));
        return Ok(String::new());
    }

    let mut output = Vec::new();
    let reasoning = outcome.reasoning.trim();
    if !reasoning.is_empty() {
        output.push(reasoning_item(
            &format!("rs_{}", compat::short_id()),
            reasoning,
        ));
    }
    if !outcome.content.is_empty() || outcome.tool_calls.is_empty() {
        output.push(message_item(
            &format!("msg_{}", compat::short_id()),
            &outcome.content,
            "completed",
        ));
    }
    for (i, call) in outcome.tool_calls.iter().enumerate() {
        output.push(function_call_item(
            &format!("fc_{}_{i}", compat::short_id()),
            call,
            "completed",
        ));
    }
    let usage = usage(outcome.prompt_tokens, outcome.completion_tokens);
    Ok(shell
        .response(status, output, Some(usage), Some(outcome.finish))
        .to_string())
}
