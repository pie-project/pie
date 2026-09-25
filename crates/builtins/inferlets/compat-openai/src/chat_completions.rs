//! OpenAI Chat Completions (`POST /v1/chat/completions`) over the loaded
//! model.
//!
//! A translation: the OpenAI request body becomes a [`compat::Request`],
//! and the [`compat::Event`] stream becomes `chat.completion.chunk` frames
//! (with `stream: true`) or one `chat.completion` body. Reasoning goes out
//! as `reasoning_content`, the convention the open inference servers share.
//!
//! Supported: `messages` (system/user/assistant/tool, text parts), `tools`
//! and `tool_choice` (`none`/`auto`/`required`/named), `response_format`
//! (`text`/`json_object`/`json_schema`), `stream` and
//! `stream_options.include_usage`, `temperature`, `top_p`, `top_k`,
//! `max_tokens`/`max_completion_tokens`, `stop`, `seed`,
//! `presence_penalty`, `frequency_penalty`, `repetition_penalty`,
//! `reasoning_effort: "none"` and `chat_template_kwargs.enable_thinking` to
//! switch thinking off or on.
//!
//! Refused with 400: `n > 1`, `logprobs`, and non-text content parts. Every
//! other unknown field is ignored, as OpenAI does.

use compat::{
    ApiError, Envelope, Event, Finish, Message, Request, ResponseFormat, Role, Thinking, ToolCall,
    ToolChoice, emit,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{invalid, stop_sequences, usage};

#[derive(Deserialize, Default)]
#[serde(default)]
struct Body {
    messages: Vec<Msg>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    response_format: Option<Value>,
    stream: bool,
    stream_options: Option<StreamOptions>,
    temperature: Option<f32>,
    speculative: Option<bool>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    stop: Option<Value>,
    seed: Option<i64>,
    n: Option<u32>,
    logprobs: Option<bool>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    reasoning_effort: Option<String>,
    chat_template_kwargs: Option<Value>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize)]
struct Msg {
    role: String,
    content: Option<Value>,
    tool_calls: Option<Vec<WireToolCall>>,
    tool_call_id: Option<String>,
    name: Option<String>,
}

#[derive(Deserialize)]
struct WireToolCall {
    id: Option<String>,
    function: WireFunction,
}

#[derive(Deserialize)]
struct WireFunction {
    name: String,
    arguments: Option<Value>,
}

fn content_text(content: Option<Value>, param: &str) -> Result<String, ApiError> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                    Some(other) => {
                        return Err(invalid(
                            format!(
                                "content part type `{other}` is not supported; only `text` parts are"
                            ),
                            Some(param),
                        ));
                    }
                    None => return Err(invalid("content part has no `type`", Some(param))),
                }
            }
            Ok(text)
        }
        Some(_) => Err(invalid(
            "`content` must be a string or an array of parts",
            Some(param),
        )),
    }
}

/// The normalized request, and whether a streamed reply ends with a usage
/// chunk.
fn parse(body: Value) -> Result<(Request, bool), ApiError> {
    let body: Body =
        serde_json::from_value(body).map_err(|e| invalid(format!("request body: {e}"), None))?;
    if body.messages.is_empty() {
        return Err(invalid("`messages` must not be empty", Some("messages")));
    }
    if body.n.is_some_and(|n| n != 1) {
        return Err(invalid("`n` other than 1 is not supported", Some("n")));
    }
    if body.logprobs == Some(true) {
        return Err(invalid("`logprobs` is not supported", Some("logprobs")));
    }

    // Tool-result messages carry a call id; the template wants the tool's
    // name, which the assistant turn that made the call knows.
    let mut call_names: Vec<(String, String)> = Vec::new();
    let mut messages = Vec::with_capacity(body.messages.len());
    for (i, m) in body.messages.into_iter().enumerate() {
        let param = format!("messages[{i}]");
        let role = match m.role.as_str() {
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => {
                return Err(invalid(format!("unknown role `{other}`"), Some(&param)));
            }
        };
        let content = content_text(m.content, &param)?;
        let mut message = Message::text(role, content);
        if role == Role::Assistant {
            for (j, call) in m.tool_calls.unwrap_or_default().into_iter().enumerate() {
                let id = call.id.unwrap_or_else(|| format!("call_{i}_{j}"));
                let arguments = match call.function.arguments {
                    None | Some(Value::Null) => "{}".to_string(),
                    Some(Value::String(s)) => s,
                    Some(other) => other.to_string(),
                };
                call_names.push((id.clone(), call.function.name.clone()));
                message.tool_calls.push(ToolCall {
                    id,
                    name: call.function.name,
                    arguments,
                });
            }
        }
        if role == Role::Tool {
            let name = m.name.or_else(|| {
                let id = m.tool_call_id.as_deref()?;
                call_names
                    .iter()
                    .find(|(call_id, _)| call_id == id)
                    .map(|(_, name)| name.clone())
            });
            message.tool_name = name;
        }
        messages.push(message);
    }

    let mut req = Request::new(messages);
    req.stream = body.stream;
    req.tools = body
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.to_string())
        .collect();
    req.tool_choice = match body.tool_choice {
        None => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "none" => ToolChoice::None,
            "auto" => ToolChoice::Auto,
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
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid(
                        "tool_choice object needs `function.name`",
                        Some("tool_choice"),
                    )
                })?;
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
    req.response_format = match body.response_format {
        None => ResponseFormat::Text,
        Some(v) => match v.get("type").and_then(Value::as_str) {
            None | Some("text") => ResponseFormat::Text,
            Some("json_object") => ResponseFormat::JsonObject,
            Some("json_schema") => {
                let schema = v
                    .get("json_schema")
                    .and_then(|s| s.get("schema"))
                    .ok_or_else(|| {
                        invalid(
                            "response_format json_schema needs `json_schema.schema`",
                            Some("response_format"),
                        )
                    })?;
                ResponseFormat::JsonSchema(schema.to_string())
            }
            Some(other) => {
                return Err(invalid(
                    format!("unknown response_format type `{other}`"),
                    Some("response_format"),
                ));
            }
        },
    };
    if let Some(t) = body.temperature {
        req.sampling.temperature = t;
    }
    if let Some(enabled) = body.speculative {
        req.sampling.speculative = enabled;
    }
    if let Some(p) = body.top_p {
        req.sampling.top_p = p;
    }
    if let Some(k) = body.top_k {
        req.sampling.top_k = k;
    }
    if let Some(p) = body.presence_penalty {
        req.sampling.presence_penalty = p;
    }
    if let Some(p) = body.frequency_penalty {
        req.sampling.frequency_penalty = p;
    }
    if let Some(p) = body.repetition_penalty {
        req.sampling.repetition_penalty = p;
    }
    if let Some(seed) = body.seed {
        req.sampling.set_seed(seed);
    }
    req.sampling.validate().map_err(|e| invalid(e, None))?;
    if let Some(max) = body.max_completion_tokens.or(body.max_tokens) {
        if max == 0 {
            return Err(invalid("max_tokens must be at least 1", Some("max_tokens")));
        }
        req.max_tokens = max;
    }
    req.stop = stop_sequences(body.stop)?;
    req.thinking = match body.reasoning_effort.as_deref() {
        Some("none") => Thinking::Disabled,
        Some(_) => Thinking::Enabled,
        None => Thinking::ModelDefault,
    };
    if let Some(enable) = body
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool)
    {
        req.thinking = if enable {
            Thinking::Enabled
        } else {
            Thinking::Disabled
        };
    }
    let include_usage = body.stream_options.is_some_and(|o| o.include_usage);
    Ok((req, include_usage))
}

fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::StopSequence => "stop",
        Finish::Length => "length",
        Finish::ToolCalls => "tool_calls",
    }
}

fn tool_call_json(index: usize, call: &ToolCall) -> Value {
    json!({
        "index": index,
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    })
}

struct Ids {
    id: String,
    created: u64,
    model: String,
}

impl Ids {
    fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}],
        })
    }
}

pub async fn serve(envelope: Envelope) -> Result<String, String> {
    let (req, include_usage) = parse(envelope.body)?;
    let ids = Ids {
        id: format!("chatcmpl-{}", compat::short_id()),
        created: envelope.time,
        model: inferlet::model::name(),
    };
    let stream = req.stream;

    if stream {
        emit::send(&ids.chunk(json!({"role": "assistant", "content": ""}), None));
    }
    let mut calls = 0usize;
    let outcome = compat::run(&req, |event| {
        if !stream {
            return;
        }
        match event {
            Event::Reasoning(text) => {
                emit::send(&ids.chunk(json!({"reasoning_content": text}), None));
            }
            Event::ReasoningEnd => {}
            Event::Text(text) => {
                emit::send(&ids.chunk(json!({"content": text}), None));
            }
            Event::ToolCall(call) => {
                emit::send(&ids.chunk(json!({"tool_calls": [tool_call_json(calls, &call)]}), None));
                calls += 1;
            }
        }
    })
    .await?;

    if stream {
        emit::send(&ids.chunk(json!({}), Some(finish_reason(outcome.finish))));
        if include_usage {
            emit::send(&json!({
                "id": ids.id,
                "object": "chat.completion.chunk",
                "created": ids.created,
                "model": ids.model,
                "choices": [],
                "usage": usage(outcome.prompt_tokens, outcome.completion_tokens),
            }));
        }
        emit::send_raw("[DONE]");
        return Ok(String::new());
    }

    let mut message = json!({"role": "assistant"});
    message["content"] = if outcome.content.is_empty() && !outcome.tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(outcome.content)
    };
    let reasoning = outcome.reasoning.trim();
    if !reasoning.is_empty() {
        message["reasoning_content"] = Value::String(reasoning.to_string());
    }
    if !outcome.tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(
            outcome
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, c)| tool_call_json(i, c))
                .collect(),
        );
    }
    Ok(json!({
        "id": ids.id,
        "object": "chat.completion",
        "created": ids.created,
        "model": ids.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(outcome.finish), "logprobs": null}],
        "usage": usage(outcome.prompt_tokens, outcome.completion_tokens),
    })
    .to_string())
}
