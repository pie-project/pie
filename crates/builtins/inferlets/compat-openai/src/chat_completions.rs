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
//! `max_tokens`/`max_completion_tokens`, `ignore_eos`, `stop`, `seed`,
//! `presence_penalty`, `frequency_penalty`, `repetition_penalty`, vLLM's
//! `min_tokens` and `min_p`, `n` (each choice its own generation, flown
//! together), `logprobs` and `top_logprobs` (up to 20, of the raw logits,
//! every sampled token's as vLLM reports them, the stop token's among them),
//! `reasoning_effort` (`none` switches thinking off; `low`, `medium` and
//! `high`/`xhigh` set the effort a template states, Qwen3.8's) and
//! `chat_template_kwargs.enable_thinking` to switch thinking off or on, and
//! an assistant message's `reasoning_content` replayed where the template
//! replays reasoning.
//!
//! Refused with 400: content parts other than text and images. Every other
//! unknown field is ignored, as OpenAI does.

use compat::{
    ApiError, Effort, Envelope, Event, Finish, Message, Request, ResponseFormat, Role, Thinking,
    ToolCall, ToolChoice, emit,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{chat_logprob, choice_request, choices, invalid, stop_sequences, usage};

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
    min_tokens: usize,
    ignore_eos: bool,
    stop: Option<Value>,
    seed: Option<i64>,
    n: Option<u32>,
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
    min_p: Option<f32>,
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
    /// An earlier answer's reasoning, as the open servers return it.
    reasoning_content: Option<String>,
    reasoning: Option<String>,
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

/// The text of `content`, and the pictures its `image_url` parts carry.
fn content_parts(content: Option<Value>, param: &str) -> Result<(String, Vec<Vec<u8>>), ApiError> {
    match content {
        None | Some(Value::Null) => Ok((String::new(), Vec::new())),
        Some(Value::String(s)) => Ok((s, Vec::new())),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            let mut images = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                    Some("image_url") => {
                        let url = part
                            .get("image_url")
                            .and_then(|u| u.get("url").or(Some(u)))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        images.push(
                            compat::image::data_url(url).map_err(|e| invalid(e, Some(param)))?,
                        );
                    }
                    Some(other) => {
                        return Err(invalid(
                            format!(
                                "content part type `{other}` is not supported; only `text` and `image_url` parts are"
                            ),
                            Some(param),
                        ));
                    }
                    None => return Err(invalid("content part has no `type`", Some(param))),
                }
            }
            Ok((text, images))
        }
        Some(_) => Err(invalid(
            "`content` must be a string or an array of parts",
            Some(param),
        )),
    }
}

/// The normalized request, whether a streamed reply ends with a usage
/// chunk, and how many choices it asks for.
fn parse(body: Value) -> Result<(Request, bool, u32), ApiError> {
    let body: Body =
        serde_json::from_value(body).map_err(|e| invalid(format!("request body: {e}"), None))?;
    if body.messages.is_empty() {
        return Err(invalid("`messages` must not be empty", Some("messages")));
    }
    let n = choices(body.n)?;
    let logprobs = match (body.logprobs.unwrap_or(false), body.top_logprobs) {
        (false, Some(k)) if k > 0 => {
            return Err(invalid(
                "`top_logprobs` needs `logprobs: true`",
                Some("top_logprobs"),
            ));
        }
        (false, _) => None,
        (true, k) => Some(k.unwrap_or(0)),
    };

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
        let (content, images) = content_parts(m.content, &param)?;
        if !images.is_empty() && role != Role::User {
            return Err(invalid("only user messages may carry images", Some(&param)));
        }
        let mut message = Message::text(role, content);
        message.images = images;
        if role == Role::Assistant {
            message.reasoning = m.reasoning_content.or(m.reasoning).unwrap_or_default();
        }
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
    if let Some(p) = body.min_p {
        req.sampling.min_p = p;
    }
    req.sampling.logprobs = logprobs;
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
    req.min_tokens = body.min_tokens;
    req.ignore_eos = body.ignore_eos;
    req.stop = stop_sequences(body.stop)?;
    let kwargs = body.chat_template_kwargs.as_ref();
    let effort = body.reasoning_effort.as_deref().or_else(|| {
        kwargs
            .and_then(|k| k.get("reasoning_effort"))
            .and_then(Value::as_str)
    });
    req.thinking = match effort {
        Some("none") => Thinking::Disabled,
        Some(_) => Thinking::Enabled,
        None => Thinking::ModelDefault,
    };
    req.effort = match effort {
        Some("low" | "minimal") => Effort::Low,
        Some("medium") => Effort::Medium,
        _ => Effort::Xhigh,
    };
    if let Some(enable) = kwargs
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
    Ok((req, include_usage, n))
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
    fn chunk(&self, index: u32, delta: Value, finish: Option<&str>, logprobs: Value) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": index, "delta": delta, "finish_reason": finish, "logprobs": logprobs}],
        })
    }
}

/// The `logprobs` a chunk or a choice carries: the entries, or null where
/// the request asked for none.
fn logprobs_field(asked: bool, entries: Vec<Value>) -> Value {
    if asked {
        json!({"content": entries})
    } else {
        Value::Null
    }
}

/// One choice's generation, its chunks streamed under `index`.
async fn choice(req: &Request, index: u32, ids: &Ids) -> Result<compat::Outcome, String> {
    let stream = req.stream;
    let asked = req.sampling.logprobs.is_some();
    // The role rides on the first delta, as the open servers send it: an
    // empty role chunk ahead of the prefill would read as the first token. A
    // turn the gateway continues builds and drops that delta again, with the
    // frames the client already has.
    let mut role = Some("assistant");
    let mut delta = |mut v: Value| {
        if let Some(r) = role.take() {
            v["role"] = json!(r);
        }
        v
    };
    let mut calls = 0usize;
    // Logprobs of sampled tokens not yet sent: they ride on the next chunk.
    let mut pending: Vec<Value> = Vec::new();
    let outcome = compat::run(req, |event| {
        if !stream {
            return;
        }
        let body = match event {
            Event::Logprobs(steps) => {
                pending.extend(steps.iter().map(|(t, lp)| chat_logprob(*t, lp)));
                return;
            }
            Event::Reasoning(text) => delta(json!({"reasoning_content": text})),
            Event::ReasoningEnd => return,
            Event::Text(text) => delta(json!({"content": text})),
            Event::ToolCall(call) => {
                let d = delta(json!({"tool_calls": [tool_call_json(calls, &call)]}));
                calls += 1;
                d
            }
        };
        let logprobs = logprobs_field(asked, std::mem::take(&mut pending));
        emit::send(&ids.chunk(index, body, None, logprobs));
    })
    .await?;
    if stream {
        let logprobs = logprobs_field(asked, pending);
        emit::send(&ids.chunk(
            index,
            delta(json!({})),
            Some(finish_reason(outcome.finish)),
            logprobs,
        ));
    }
    Ok(outcome)
}

pub async fn serve(envelope: Envelope) -> Result<String, String> {
    let (req, include_usage, n) = parse(envelope.body)?;
    let ids = Ids {
        id: format!("chatcmpl-{}", compat::short_id()),
        created: envelope.time,
        model: inferlet::model::name(),
    };
    let stream = req.stream;
    let asked = req.sampling.logprobs.is_some();

    let requests: Vec<Request> = (0..n).map(|i| choice_request(&req, i)).collect();
    let outcomes: Vec<compat::Outcome> = if n == 1 {
        vec![choice(&req, 0, &ids).await?]
    } else {
        compat::join_all(
            requests
                .iter()
                .zip(0u32..)
                .map(|(req, i)| {
                    Box::pin(choice(req, i, &ids))
                        as std::pin::Pin<Box<dyn std::future::Future<Output = _>>>
                })
                .collect(),
        )
        .await
        .into_iter()
        .collect::<Result<_, _>>()?
    };
    let total = usage_of(&outcomes);

    if stream {
        if include_usage {
            emit::send(&json!({
                "id": ids.id,
                "object": "chat.completion.chunk",
                "created": ids.created,
                "model": ids.model,
                "choices": [],
                "usage": total,
            }));
        }
        emit::send_raw("[DONE]");
        return Ok(String::new());
    }

    let choices: Vec<Value> = outcomes
        .into_iter()
        .zip(0u32..)
        .map(|(outcome, index)| {
            let mut message = json!({"role": "assistant"});
            message["content"] = if outcome.content.is_empty() && !outcome.tool_calls.is_empty() {
                Value::Null
            } else {
                Value::String(outcome.content.clone())
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
            let logprobs = logprobs_field(
                asked,
                outcome
                    .logprobs
                    .iter()
                    .map(|(t, lp)| chat_logprob(*t, lp))
                    .collect(),
            );
            json!({"index": index, "message": message, "finish_reason": finish_reason(outcome.finish), "logprobs": logprobs})
        })
        .collect();
    Ok(json!({
        "id": ids.id,
        "object": "chat.completion",
        "created": ids.created,
        "model": ids.model,
        "choices": choices,
        "usage": total,
    })
    .to_string())
}

/// The usage of every choice together: the prompt once, every choice's
/// completion.
fn usage_of(outcomes: &[compat::Outcome]) -> Value {
    let mut total = usage(&outcomes[0]);
    let completion: usize = outcomes.iter().map(|o| o.completion_tokens).sum();
    total["completion_tokens"] = json!(completion);
    total["total_tokens"] = json!(outcomes[0].prompt_tokens + completion);
    total
}
