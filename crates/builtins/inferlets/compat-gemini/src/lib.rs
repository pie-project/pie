//! Gemini `generateContent` and `streamGenerateContent`
//! (`POST /v1beta/models/{model}:{method}`) over the loaded model.
//!
//! A translation: `contents` become a [`compat::Request`], and the
//! [`compat::Event`] stream becomes candidate parts — `text`, thought
//! parts (with `includeThoughts`), and `functionCall` parts — as one
//! response or, for `streamGenerateContent?alt=sse`, one chunk per SSE
//! frame. Without `alt=sse` a stream is the JSON array of its chunks, as
//! the API defines it, returned whole.
//!
//! Supported: `contents` with `text`, `functionCall` and `functionResponse`
//! parts, `systemInstruction`, `tools[].functionDeclarations`,
//! `toolConfig.functionCallingConfig` (`AUTO`/`ANY`/`NONE` and
//! `allowedFunctionNames`), and in `generationConfig`: `temperature`,
//! `topP`, `topK`, `maxOutputTokens`, `stopSequences`, `seed`,
//! `presencePenalty`, `frequencyPenalty`, `responseMimeType` with
//! `responseSchema` or `responseJsonSchema`, and `thinkingConfig`
//! (`thinkingBudget: 0` switches thinking off, `includeThoughts` returns
//! it).
//!
//! Refused with 400: media parts, built-in tools, and `candidateCount > 1`.

use compat::{
    ApiError, Envelope, Event, Finish, Message, Request, ResponseFormat, Role, Thinking, ToolCall,
    ToolChoice, emit,
};
use serde_json::{Value, json};

fn error(code: u16, status: &str, message: impl Into<String>) -> ApiError {
    ApiError::new(
        code,
        json!({"error": {"code": code, "message": message.into(), "status": status}}),
    )
}

fn invalid(message: impl Into<String>) -> ApiError {
    error(400, "INVALID_ARGUMENT", message)
}

/// `key` in camelCase or snake_case.
fn field<'a>(object: &'a Value, camel: &str, snake: &str) -> Option<&'a Value> {
    object.get(camel).or_else(|| object.get(snake))
}

fn parts_text(content: &Value, what: &str) -> Result<String, ApiError> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Object(_) => {
            let mut text = String::new();
            for part in content
                .get("parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                } else {
                    return Err(invalid(format!(
                        "{what}: only text parts are supported here"
                    )));
                }
            }
            Ok(text)
        }
        _ => Err(invalid(format!("{what}: expected content with parts"))),
    }
}

/// Gemini's `responseSchema` spells types in upper case (`OBJECT`); JSON
/// Schema wants them lower.
fn lower_types(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(t)) = map.get_mut("type") {
                *t = t.to_lowercase();
            }
            for v in map.values_mut() {
                lower_types(v);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(lower_types),
        _ => {}
    }
}

/// The normalized request, and whether thoughts are returned
/// (`thinkingConfig.includeThoughts`).
fn parse(body: &Value) -> Result<(Request, bool), ApiError> {
    let Some(object) = body.as_object() else {
        return Err(invalid("request body must be a JSON object"));
    };
    let mut messages = Vec::new();
    if let Some(system) = field(body, "systemInstruction", "system_instruction") {
        let text = parts_text(system, "systemInstruction")?;
        if !text.is_empty() {
            messages.push(Message::text(Role::System, text));
        }
    }
    let contents = object
        .get("contents")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("`contents` is required"))?;
    for (i, content) in contents.iter().enumerate() {
        let what = format!("contents[{i}]");
        let role = match content.get("role").and_then(Value::as_str) {
            None | Some("user") => Role::User,
            Some("model") => Role::Assistant,
            Some(other) => return Err(invalid(format!("{what}: unknown role `{other}`"))),
        };
        let mut text = String::new();
        let mut calls = Vec::new();
        for part in content
            .get("parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                continue;
            }
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                text.push_str(t);
            } else if let Some(call) = field(part, "functionCall", "function_call") {
                calls.push(ToolCall {
                    id: String::new(),
                    name: call
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    arguments: call.get("args").cloned().unwrap_or(json!({})).to_string(),
                });
            } else if let Some(response) = field(part, "functionResponse", "function_response") {
                let mut m = Message::text(
                    Role::Tool,
                    response
                        .get("response")
                        .cloned()
                        .unwrap_or(json!({}))
                        .to_string(),
                );
                m.tool_name = response
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                messages.push(m);
            } else {
                return Err(invalid(format!(
                    "{what}: only text, functionCall and functionResponse parts are supported"
                )));
            }
        }
        if !text.is_empty() || !calls.is_empty() {
            let mut m = Message::text(role, text);
            m.tool_calls = calls;
            messages.push(m);
        }
    }
    if messages.is_empty() {
        return Err(invalid("`contents` must not be empty"));
    }

    let mut req = Request::new(messages);
    for tool in object
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(declarations) = field(tool, "functionDeclarations", "function_declarations")
        else {
            return Err(invalid("only `functionDeclarations` tools are supported"));
        };
        for declaration in declarations.as_array().into_iter().flatten() {
            let name = declaration
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("functionDeclarations[]: `name` is required"))?;
            let mut parameters = field(
                declaration,
                "parametersJsonSchema",
                "parameters_json_schema",
            )
            .cloned()
            .or_else(|| declaration.get("parameters").cloned())
            .unwrap_or(json!({"type": "object"}));
            lower_types(&mut parameters);
            req.tools.push(
                json!({"type": "function", "function": {
                    "name": name,
                    "description": declaration.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": parameters,
                }})
                .to_string(),
            );
        }
    }
    if let Some(config) = field(body, "toolConfig", "tool_config")
        .and_then(|c| field(c, "functionCallingConfig", "function_calling_config"))
    {
        let allowed: Vec<String> = field(config, "allowedFunctionNames", "allowed_function_names")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        req.tool_choice = match config.get("mode").and_then(Value::as_str) {
            None | Some("AUTO") | Some("MODE_UNSPECIFIED") | Some("VALIDATED") => ToolChoice::Auto,
            Some("NONE") => ToolChoice::None,
            Some("ANY") => match allowed.as_slice() {
                [one] => ToolChoice::Named(one.clone()),
                _ => ToolChoice::Required,
            },
            Some(other) => {
                return Err(invalid(format!(
                    "unknown functionCallingConfig mode `{other}`"
                )));
            }
        };
    }
    if req.tools.is_empty() && req.forces_tool_call() {
        return Err(invalid(
            "functionCallingConfig mode ANY needs function declarations",
        ));
    }

    let mut include_thoughts = false;
    if let Some(config) = field(body, "generationConfig", "generation_config") {
        if field(config, "candidateCount", "candidate_count")
            .and_then(Value::as_u64)
            .is_some_and(|n| n != 1)
        {
            return Err(invalid("`candidateCount` other than 1 is not supported"));
        }
        if let Some(t) = config.get("temperature").and_then(Value::as_f64) {
            req.sampling.temperature = t as f32;
        }
        if let Some(p) = field(config, "topP", "top_p").and_then(Value::as_f64) {
            req.sampling.top_p = p as f32;
        }
        if let Some(k) = field(config, "topK", "top_k").and_then(Value::as_u64) {
            req.sampling.top_k = k as u32;
        }
        if let Some(p) =
            field(config, "presencePenalty", "presence_penalty").and_then(Value::as_f64)
        {
            req.sampling.presence_penalty = p as f32;
        }
        if let Some(p) =
            field(config, "frequencyPenalty", "frequency_penalty").and_then(Value::as_f64)
        {
            req.sampling.frequency_penalty = p as f32;
        }
        if let Some(seed) = config.get("seed").and_then(Value::as_i64) {
            req.sampling.set_seed(seed);
        }
        if let Some(max) =
            field(config, "maxOutputTokens", "max_output_tokens").and_then(Value::as_u64)
        {
            if max == 0 {
                return Err(invalid("`maxOutputTokens` must be at least 1"));
            }
            req.max_tokens = max as usize;
        }
        req.stop = field(config, "stopSequences", "stop_sequences")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let mime = field(config, "responseMimeType", "response_mime_type").and_then(Value::as_str);
        if mime == Some("application/json") {
            req.response_format =
                if let Some(schema) = field(config, "responseJsonSchema", "response_json_schema") {
                    ResponseFormat::JsonSchema(schema.to_string())
                } else if let Some(schema) = field(config, "responseSchema", "response_schema") {
                    let mut schema = schema.clone();
                    lower_types(&mut schema);
                    ResponseFormat::JsonSchema(schema.to_string())
                } else {
                    ResponseFormat::JsonObject
                };
        } else if let Some(other) = mime.filter(|m| *m != "text/plain") {
            return Err(invalid(format!(
                "`responseMimeType` `{other}` is not supported"
            )));
        }
        if let Some(thinking) = field(config, "thinkingConfig", "thinking_config") {
            include_thoughts = field(thinking, "includeThoughts", "include_thoughts")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            req.thinking = match field(thinking, "thinkingBudget", "thinking_budget")
                .and_then(Value::as_i64)
            {
                Some(0) => Thinking::Disabled,
                Some(_) => Thinking::Enabled,
                None => Thinking::ModelDefault,
            };
        }
    }
    req.sampling.validate().map_err(invalid)?;
    Ok((req, include_thoughts))
}

fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::StopSequence | Finish::ToolCalls => "STOP",
        Finish::Length => "MAX_TOKENS",
    }
}

fn function_call_part(call: &ToolCall) -> Value {
    let args: Value = serde_json::from_str(&call.arguments).unwrap_or(json!({}));
    json!({"functionCall": {"name": call.name, "args": args}})
}

struct Shell {
    id: String,
    model: String,
}

impl Shell {
    fn chunk(
        &self,
        parts: Vec<Value>,
        finish: Option<Finish>,
        usage: Option<(usize, usize)>,
    ) -> Value {
        let mut out = json!({
            "candidates": [{"content": {"parts": parts, "role": "model"}, "index": 0}],
            "modelVersion": self.model,
            "responseId": self.id,
        });
        if let Some(finish) = finish {
            out["candidates"][0]["finishReason"] = json!(finish_reason(finish));
        }
        if let Some((prompt, completion)) = usage {
            out["usageMetadata"] = json!({
                "promptTokenCount": prompt,
                "candidatesTokenCount": completion,
                "totalTokenCount": prompt + completion,
            });
        }
        out
    }
}

#[inferlet::main]
async fn main(input: String) -> Result<String, String> {
    let envelope = Envelope::parse(&input).map_err(invalid)?;
    // `/v1beta/models/{model}:{method}`; `pie run` gives no path at all.
    let (model, method) = match envelope.path.rsplit_once('/') {
        Some((_, last)) => match last.split_once(':') {
            Some((model, method)) => (model.to_string(), method.to_string()),
            None => (last.to_string(), "generateContent".to_string()),
        },
        None => (String::new(), "generateContent".to_string()),
    };
    let stream = match method.as_str() {
        "generateContent" => false,
        "streamGenerateContent" => true,
        other => {
            return Err(error(
                404,
                "NOT_FOUND",
                format!("method `{other}` is not supported"),
            )
            .into());
        }
    };
    let sse = stream && envelope.query.get("alt").map(String::as_str) == Some("sse");
    let (req, include_thoughts) = parse(&envelope.body)?;
    let shell = Shell {
        id: compat::short_id(),
        model: if model.is_empty() {
            inferlet::model::name()
        } else {
            model
        },
    };

    let outcome = compat::run(&req, |event| {
        if !sse {
            return;
        }
        let part = match event {
            Event::Reasoning(text) if include_thoughts => json!({"text": text, "thought": true}),
            Event::Reasoning(_) | Event::ReasoningEnd => return,
            Event::Text(text) => json!({"text": text}),
            Event::ToolCall(call) => function_call_part(&call),
        };
        emit::send(&shell.chunk(vec![part], None, None));
    })
    .await?;

    let usage = (outcome.prompt_tokens, outcome.completion_tokens);
    if sse {
        emit::send(&shell.chunk(Vec::new(), Some(outcome.finish), Some(usage)));
        return Ok(String::new());
    }
    let mut parts = Vec::new();
    let reasoning = outcome.reasoning.trim();
    if include_thoughts && !reasoning.is_empty() {
        parts.push(json!({"text": reasoning, "thought": true}));
    }
    if !outcome.content.is_empty() || outcome.tool_calls.is_empty() {
        parts.push(json!({"text": outcome.content}));
    }
    for call in &outcome.tool_calls {
        parts.push(function_call_part(call));
    }
    let response = shell.chunk(parts, Some(outcome.finish), Some(usage));
    Ok(if stream {
        json!([response]).to_string()
    } else {
        response.to_string()
    })
}
