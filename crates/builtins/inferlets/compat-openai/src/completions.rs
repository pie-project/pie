//! OpenAI legacy Completions (`POST /v1/completions`): the prompt is
//! continued verbatim, with no chat template.
//!
//! Supported: `prompt` (one string, or a one-element array), `max_tokens`
//! (default 16), `temperature`, `top_p`, `top_k`, `stop`, `seed`, `echo`,
//! the penalties, `stream`, `stream_options.include_usage`, `ignore_eos`
//! (vLLM's and SGLang's extension), vLLM's `min_tokens` and `min_p`, `n`,
//! and `logprobs` (an integer, up to 20: the legacy `tokens`,
//! `token_logprobs`, `top_logprobs`, `text_offset` lists).
//!
//! Refused with 400: more than one prompt, token-array prompts and
//! `suffix`. Anything a thinking model emits as a thinking block is returned
//! as text: this API has no other place for it.

use compat::{ApiError, Envelope, Event, Finish, Request, emit};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{choice_request, choices, invalid, stop_sequences, token_text, usage};

#[derive(Deserialize, Default)]
#[serde(default)]
struct Body {
    prompt: Option<Value>,
    max_tokens: Option<usize>,
    min_tokens: usize,
    ignore_eos: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop: Option<Value>,
    seed: Option<i64>,
    n: Option<u32>,
    logprobs: Option<Value>,
    min_p: Option<f32>,
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
    n: u32,
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
    let n = choices(body.n)?;
    let logprobs = match body.logprobs {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .and_then(|k| u32::try_from(k).ok())
                .ok_or_else(|| invalid("`logprobs` must be an integer", Some("logprobs")))?,
        ),
    };
    if body.suffix.is_some() {
        return Err(invalid("`suffix` is not supported", Some("suffix")));
    }
    let mut req = Request::new(Vec::new());
    req.raw_prompt = Some(prompt.clone());
    req.stream = body.stream;
    req.max_tokens = body.max_tokens.unwrap_or(16);
    req.ignore_eos = body.ignore_eos;
    if req.max_tokens == 0 {
        return Err(invalid("max_tokens must be at least 1", Some("max_tokens")));
    }
    req.min_tokens = body.min_tokens;
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
    if let Some(p) = body.min_p {
        req.sampling.min_p = p;
    }
    req.sampling.logprobs = logprobs;
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
        n,
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
    fn chunk(&self, index: u32, text: &str, finish: Option<&str>, logprobs: Value) -> Value {
        json!({
            "id": self.id,
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": index, "text": text, "logprobs": logprobs, "finish_reason": finish}],
        })
    }
}

/// Legacy logprobs over `steps`, the first token's text starting at
/// `offset`: each token, its logprob, its alternatives (the sampled token
/// among them, as vLLM lists it) and where its text starts.
fn legacy_logprobs(steps: &[(u32, compat::Logprob)], offset: usize) -> Value {
    let (mut tokens, mut values, mut tops, mut offsets) = (vec![], vec![], vec![], vec![]);
    let mut at = offset;
    for (token, lp) in steps {
        let text = token_text(*token);
        let mut top = serde_json::Map::new();
        let alternatives = std::iter::once((*token, lp.logprob))
            .chain(lp.top.iter().copied().filter(|&(t, _)| t != *token))
            .take(lp.top.len());
        for (alt, alt_lp) in alternatives {
            top.entry(token_text(alt)).or_insert(json!(alt_lp));
        }
        offsets.push(at);
        at += text.len();
        tokens.push(text);
        values.push(lp.logprob);
        tops.push(Value::Object(top));
    }
    json!({"tokens": tokens, "token_logprobs": values, "top_logprobs": tops, "text_offset": offsets})
}

/// One choice's generation, its chunks streamed under `index`.
async fn choice(
    parsed: &Parsed,
    req: &Request,
    index: u32,
    ids: &Ids,
) -> Result<(compat::Outcome, String), String> {
    let stream = req.stream;
    let asked = req.sampling.logprobs.is_some();
    if stream && parsed.echo {
        emit::send(&ids.chunk(index, &parsed.prompt, None, Value::Null));
    }
    let mut text = String::new();
    let mut offset = if parsed.echo { parsed.prompt.len() } else { 0 };
    let mut pending: Vec<(u32, compat::Logprob)> = Vec::new();
    let outcome = compat::run(req, |event| {
        let piece = match event {
            Event::Text(t) | Event::Reasoning(t) => t,
            Event::ReasoningEnd => return,
            Event::ToolCall(call) => call.arguments,
            Event::Logprobs(steps) => {
                pending.extend(steps);
                return;
            }
        };
        if stream {
            let logprobs = if asked {
                let lp = legacy_logprobs(&pending, offset);
                pending.clear();
                lp
            } else {
                Value::Null
            };
            offset += piece.len();
            emit::send(&ids.chunk(index, &piece, None, logprobs));
        } else {
            text.push_str(&piece);
        }
    })
    .await?;
    if stream {
        let logprobs = if asked {
            legacy_logprobs(&pending, offset)
        } else {
            Value::Null
        };
        emit::send(&ids.chunk(index, "", Some(finish_reason(outcome.finish)), logprobs));
    }
    Ok((outcome, text))
}

pub async fn serve(envelope: Envelope) -> Result<String, String> {
    let parsed = parse(envelope.body)?;
    let ids = Ids {
        id: format!("cmpl-{}", compat::short_id()),
        created: envelope.time,
        model: inferlet::model::name(),
    };
    let stream = parsed.req.stream;
    let asked = parsed.req.sampling.logprobs.is_some();
    let requests: Vec<Request> = (0..parsed.n)
        .map(|i| choice_request(&parsed.req, i))
        .collect();
    let outcomes: Vec<(compat::Outcome, String)> = if parsed.n == 1 {
        vec![choice(&parsed, &parsed.req, 0, &ids).await?]
    } else {
        compat::join_all(
            requests
                .iter()
                .zip(0u32..)
                .map(|(req, i)| {
                    Box::pin(choice(&parsed, req, i, &ids))
                        as std::pin::Pin<Box<dyn std::future::Future<Output = _>>>
                })
                .collect(),
        )
        .await
        .into_iter()
        .collect::<Result<_, _>>()?
    };
    let mut total = usage(&outcomes[0].0);
    let completion: usize = outcomes.iter().map(|(o, _)| o.completion_tokens).sum();
    total["completion_tokens"] = json!(completion);
    total["total_tokens"] = json!(outcomes[0].0.prompt_tokens + completion);

    if stream {
        if parsed.include_usage {
            emit::send(&json!({
                "id": ids.id, "object": "text_completion", "created": ids.created, "model": ids.model,
                "choices": [], "usage": total,
            }));
        }
        emit::send_raw("[DONE]");
        return Ok(String::new());
    }
    let choices: Vec<Value> = outcomes
        .into_iter()
        .zip(0u32..)
        .map(|((outcome, text), index)| {
            let text = if parsed.echo {
                format!("{}{text}", parsed.prompt)
            } else {
                text
            };
            let offset = if parsed.echo { parsed.prompt.len() } else { 0 };
            let logprobs = if asked {
                legacy_logprobs(&outcome.logprobs, offset)
            } else {
                Value::Null
            };
            json!({"index": index, "text": text, "logprobs": logprobs, "finish_reason": finish_reason(outcome.finish)})
        })
        .collect();
    Ok(json!({
        "id": ids.id,
        "object": "text_completion",
        "created": ids.created,
        "model": ids.model,
        "choices": choices,
        "usage": total,
    })
    .to_string())
}
