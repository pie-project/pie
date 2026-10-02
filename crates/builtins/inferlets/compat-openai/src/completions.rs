//! OpenAI legacy Completions (`POST /v1/completions`): the prompt is
//! continued verbatim, with no chat template.
//!
//! Supported: `prompt` (one string, or a one-element array), `max_tokens`
//! (default 16), `temperature`, `top_p`, `top_k`, `stop`, `seed`, `echo`,
//! the penalties, `stream` and `stream_options.include_usage`.
//!
//! Refused with 400: more than one prompt, token-array prompts, `suffix`,
//! `logprobs`, and `n > 1`. Anything a thinking model emits as a thinking
//! block is returned as text: this API has no other place for it.

use compat::{ApiError, Envelope, Event, Finish, Request, emit};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{invalid, stop_sequences, usage};

#[derive(Deserialize, Default)]
#[serde(default)]
struct Body {
    prompt: Option<Value>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop: Option<Value>,
    seed: Option<i64>,
    n: Option<u32>,
    logprobs: Option<Value>,
    echo: bool,
    suffix: Option<String>,
    stream: bool,
    stream_options: Option<StreamOptions>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct StreamOptions {
    include_usage: bool,
}

struct Parsed {
    req: Request,
    prompt: String,
    echo: bool,
    include_usage: bool,
}

fn parse(body: Value) -> Result<Parsed, ApiError> {
    let body: Body =
        serde_json::from_value(body).map_err(|e| invalid(format!("request body: {e}"), None))?;
    let prompt = match body.prompt {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s,
        Some(Value::Array(items)) => match items.as_slice() {
            [Value::String(s)] => s.clone(),
            [] => String::new(),
            [_, _, ..] => {
                return Err(invalid(
                    "only one prompt per request is supported",
                    Some("prompt"),
                ));
            }
            _ => {
                return Err(invalid(
                    "token-array prompts are not supported; send text",
                    Some("prompt"),
                ));
            }
        },
        Some(_) => return Err(invalid("`prompt` must be a string", Some("prompt"))),
    };
    if body.n.is_some_and(|n| n != 1) {
        return Err(invalid("`n` other than 1 is not supported", Some("n")));
    }
    if body.logprobs.as_ref().is_some_and(|v| !v.is_null()) {
        return Err(invalid("`logprobs` is not supported", Some("logprobs")));
    }
    if body.suffix.is_some() {
        return Err(invalid("`suffix` is not supported", Some("suffix")));
    }
    let mut req = Request::new(Vec::new());
    req.raw_prompt = Some(prompt.clone());
    req.stream = body.stream;
    req.max_tokens = body.max_tokens.unwrap_or(16);
    if req.max_tokens == 0 {
        return Err(invalid("max_tokens must be at least 1", Some("max_tokens")));
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
    req.stop = stop_sequences(body.stop)?;
    Ok(Parsed {
        req,
        prompt,
        echo: body.echo,
        include_usage: body.stream_options.is_some_and(|o| o.include_usage),
    })
}

fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::StopSequence | Finish::ToolCalls => "stop",
        Finish::Length => "length",
    }
}

struct Ids {
    id: String,
    created: u64,
    model: String,
}

impl Ids {
    fn chunk(&self, text: &str, finish: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish}],
        })
    }
}

pub async fn serve(envelope: Envelope) -> Result<String, String> {
    let parsed = parse(envelope.body)?;
    let ids = Ids {
        id: format!("cmpl-{}", compat::short_id()),
        created: envelope.time,
        model: inferlet::model::name(),
    };
    let stream = parsed.req.stream;
    if stream && parsed.echo {
        emit::send(&ids.chunk(&parsed.prompt, None));
    }
    let mut text = String::new();
    let outcome = compat::run(&parsed.req, |event| {
        let piece = match event {
            Event::Text(t) | Event::Reasoning(t) => t,
            Event::ReasoningEnd => return,
            Event::ToolCall(call) => call.arguments,
        };
        if stream {
            emit::send(&ids.chunk(&piece, None));
        } else {
            text.push_str(&piece);
        }
    })
    .await?;

    if stream {
        emit::send(&ids.chunk("", Some(finish_reason(outcome.finish))));
        if parsed.include_usage {
            emit::send(&json!({
                "id": ids.id, "object": "text_completion", "created": ids.created, "model": ids.model,
                "choices": [], "usage": usage(outcome.prompt_tokens, outcome.completion_tokens),
            }));
        }
        emit::send_raw("[DONE]");
        return Ok(String::new());
    }
    let text = if parsed.echo {
        format!("{}{text}", parsed.prompt)
    } else {
        text
    };
    Ok(json!({
        "id": ids.id,
        "object": "text_completion",
        "created": ids.created,
        "model": ids.model,
        "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish_reason(outcome.finish)}],
        "usage": usage(outcome.prompt_tokens, outcome.completion_tokens),
    })
    .to_string())
}
