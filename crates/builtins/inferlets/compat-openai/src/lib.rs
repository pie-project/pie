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

fn usage(outcome: &compat::Outcome) -> Value {
    json!({
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
        "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": outcome.cached_tokens},
    })
}

/// A token's text as vLLM reports it beside a logprob: the token decoded on
/// its own, empty where it is part of a character it does not finish.
fn token_text(token: u32) -> String {
    let text = inferlet::model::decode(&[token]).unwrap_or_default();
    if text.contains('\u{fffd}') {
        String::new()
    } else {
        text
    }
}

/// One chat-completions logprob entry: the sampled token and its
/// alternatives, each with its text's UTF-8 bytes. The alternatives are
/// listed as vLLM lists them: the sampled token first, then the likeliest
/// others, as many as were asked.
fn chat_logprob(token: u32, logprob: &compat::Logprob) -> Value {
    let entry = |token: u32, logprob: f32| {
        let text = token_text(token);
        let bytes: Vec<u8> = text.as_bytes().to_vec();
        json!({"token": text, "logprob": logprob.max(-9999.0), "bytes": bytes})
    };
    let mut out = entry(token, logprob.logprob);
    let asked = logprob.top.len();
    out["top_logprobs"] = Value::Array(
        std::iter::once((token, logprob.logprob))
            .chain(logprob.top.iter().copied().filter(|&(t, _)| t != token))
            .take(asked)
            .map(|(token, logprob)| entry(token, logprob))
            .collect(),
    );
    out
}

/// The `n` field: how many choices, at least one.
fn choices(n: Option<u32>) -> Result<u32, ApiError> {
    match n {
        None => Ok(1),
        Some(0) => Err(invalid("`n` must be at least 1", Some("n"))),
        Some(n) if n > 128 => Err(invalid("`n` must be at most 128", Some("n"))),
        Some(n) => Ok(n),
    }
}

/// The request of the `index`-th of `n` choices: its own random stream.
fn choice_request(req: &compat::Request, index: u32) -> compat::Request {
    let mut req = req.clone();
    req.sampling.seed = req
        .sampling
        .seed
        .wrapping_add(index.wrapping_mul(0x9e37_79b9));
    req
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
