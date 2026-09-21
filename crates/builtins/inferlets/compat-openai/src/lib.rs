//! The OpenAI API over the loaded model: one program behind three routes.
//!
//! - `POST /v1/chat/completions` — [`chat_completions`]
//! - `POST /v1/completions` — [`completions`] (the legacy raw-prompt API)
//! - `POST /v1/responses` — [`responses`]
//!
//! The gateway names the route in the launch envelope's `path`; run by
//! hand (`pie run compat-openai`) with no path, the body says which API it
//! is: `messages` is a chat completion, `prompt` a completion, `input` a
//! response.

use compat::{ApiError, Envelope};
use serde_json::{Value, json};

pub mod chat_completions;
pub mod completions;
pub mod responses;

/// A 400 in OpenAI's error shape, blaming `param` when one is to blame.
fn invalid(message: impl Into<String>, param: Option<&str>) -> ApiError {
    ApiError::new(
        400,
        json!({"error": {"message": message.into(), "type": "invalid_request_error", "param": param, "code": null}}),
    )
}

fn usage(prompt: usize, completion: usize) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    })
}

/// The `stop` field: absent, one string, or an array of them.
fn stop_sequences(stop: Option<Value>) -> Result<Vec<String>, ApiError> {
    match stop {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s]),
        Some(Value::Array(items)) => Ok(items
            .into_iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect()),
        Some(_) => Err(invalid(
            "`stop` must be a string or an array of strings",
            Some("stop"),
        )),
    }
}

enum Api {
    ChatCompletions,
    Completions,
    Responses,
}

fn route(envelope: &Envelope) -> Result<Api, ApiError> {
    match envelope.path.trim_end_matches('/') {
        "/v1/chat/completions" => return Ok(Api::ChatCompletions),
        "/v1/completions" => return Ok(Api::Completions),
        "/v1/responses" => return Ok(Api::Responses),
        "" => {}
        other => {
            return Err(ApiError::new(
                404,
                json!({"error": {"message": format!("no OpenAI API at `{other}`"), "type": "invalid_request_error", "param": null, "code": null}}),
            ));
        }
    }
    let body = &envelope.body;
    if body.get("messages").is_some() {
        Ok(Api::ChatCompletions)
    } else if body.get("prompt").is_some() {
        Ok(Api::Completions)
    } else if body.get("input").is_some() {
        Ok(Api::Responses)
    } else {
        Err(invalid(
            "the body has none of `messages`, `prompt` or `input`, so it is not a chat completion, a completion or a response",
            None,
        ))
    }
}

#[inferlet::main]
async fn main(input: String) -> Result<String, String> {
    let envelope = Envelope::parse(&input).map_err(|e| invalid(e, None))?;
    match route(&envelope)? {
        Api::ChatCompletions => chat_completions::serve(envelope).await,
        Api::Completions => completions::serve(envelope).await,
        Api::Responses => responses::serve(envelope).await,
    }
}
