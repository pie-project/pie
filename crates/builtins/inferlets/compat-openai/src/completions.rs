//! OpenAI legacy Completions (`POST /v1/completions`): the prompt is
//! continued verbatim, with no chat template.
//!
//! Supported: `prompt` (one string, or a one-element array), `max_tokens`
//! (default 16), `temperature`, `top_p`, `top_k`, `stop`, `seed`, `echo`,
//! the penalties, `stream`, `stream_options.include_usage`, `ignore_eos`
//! (vLLM's and SGLang's extension), vLLM's `min_tokens` and `min_p`, `n`,
//! vLLM's `response_format` (as chat completions take it, `structural_tag`
//! among them),
//! and `logprobs` (an integer, up to 20: the legacy `tokens`,
//! `token_logprobs`, `top_logprobs`, `text_offset` lists).
//!
//! With `echo`, `logprobs` covers the prompt too, as OpenAI's did: the
//! echoed text's tokens come first, scored under the tokens before them,
//! the first `null` where nothing precedes it (a model whose raw prompts
//! open on a prefix such as BOS scores even the first). vLLM's
//! `prompt_logprobs` (an integer, unstreamed) puts every prompt token, the
//! prefix among them, in a choice's `prompt_logprobs` in vLLM's shape.
//! Either costs a prefill read out row by row, apart from the prefix cache.
//!
//! A prompt may be token ids (one array, or an array of one array), used as
//! they are.
//!
//! Refused with 400: more than one prompt, `echo` on token ids, `suffix` and
//! a non-empty `logit_bias`. Anything a thinking model emits as a thinking block is returned
//! as text: this API has no other place for it.

use compat::{ApiError, Envelope, Event, Finish, Request, emit};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    alternatives, choices, invalid, no_logit_bias, run_choices, stop_reason, stop_sequences,
    token_label, token_text, usage_of,
};

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
    stop_token_ids: Option<Vec<u32>>,
    include_stop_str_in_output: bool,
    response_format: Option<Value>,
    seed: Option<i64>,
    n: Option<u32>,
    logprobs: Option<Value>,
    logprobs_mode: Option<String>,
    prompt_logprobs: Option<i64>,
    min_p: Option<f32>,
    echo: bool,
    suffix: Option<String>,
    stream: bool,
    stream_options: Option<StreamOptions>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    logit_bias: Option<Value>,
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
    /// How many alternatives the legacy `logprobs` lists carry.
    logprobs: Option<u32>,
    /// How many alternatives vLLM's `prompt_logprobs` carries, where asked.
    prompt_logprobs: Option<u32>,
    as_ids: bool,
}

impl Parsed {
    /// The echoed prompt is scored for the legacy lists.
    fn echo_logprobs(&self) -> Option<u32> {
        self.logprobs.filter(|_| self.echo)
    }
}

fn parse(body: Value) -> Result<Parsed, ApiError> {
    let body: Body =
        serde_json::from_value(body).map_err(|e| invalid(format!("request body: {e}"), None))?;
    let ids = |items: &[Value]| -> Option<Vec<u32>> {
        items
            .iter()
            .map(|v| v.as_u64().and_then(|t| u32::try_from(t).ok()))
            .collect()
    };
    let (prompt, prompt_tokens) = match body.prompt {
        None | Some(Value::Null) => (String::new(), None),
        Some(Value::String(s)) => (s, None),
        Some(Value::Array(items)) => match items.as_slice() {
            [Value::String(s)] => (s.clone(), None),
            [] => (String::new(), None),
            [Value::Array(one)] => match ids(one) {
                Some(tokens) => (String::new(), Some(tokens)),
                None => return Err(invalid("token ids must be integers", Some("prompt"))),
            },
            items if items.iter().all(Value::is_number) => match ids(items) {
                Some(tokens) => (String::new(), Some(tokens)),
                None => return Err(invalid("token ids must be integers", Some("prompt"))),
            },
            _ => {
                return Err(invalid(
                    "only one prompt per request is supported",
                    Some("prompt"),
                ));
            }
        },
        Some(_) => return Err(invalid("`prompt` must be a string", Some("prompt"))),
    };
    if prompt_tokens.is_some() && body.echo {
        return Err(invalid(
            "`echo` with a token-array prompt is not supported",
            Some("echo"),
        ));
    }
    let n = choices(body.n)?;
    let logprobs = match body.logprobs {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .and_then(|k| u32::try_from(k).ok())
                .ok_or_else(|| invalid("`logprobs` must be an integer", Some("logprobs")))?,
        ),
    };
    no_logit_bias(&body.logit_bias)?;
    if body.suffix.is_some() {
        return Err(invalid("`suffix` is not supported", Some("suffix")));
    }
    let mut req = Request::new(Vec::new());
    match prompt_tokens {
        Some(tokens) => req.prompt_tokens = Some(tokens),
        None => req.raw_prompt = Some(prompt.clone()),
    }
    req.stream = body.stream;
    req.return_token_ids = body.return_token_ids;
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
    req.sampling.logprobs_mode = crate::logprobs_mode(body.logprobs_mode)?;
    let echo_logprobs = logprobs.filter(|_| body.echo);
    let prompt_logprobs = crate::prompt_logprobs(body.prompt_logprobs, body.stream)?;
    req.sampling.prompt_logprobs = echo_logprobs.max(prompt_logprobs);
    if let Some(seed) = body.seed {
        req.sampling.set_seed(seed);
    }
    req.sampling.validate().map_err(|e| invalid(e, None))?;
    req.stop = stop_sequences(body.stop)?;
    req.stop_token_ids = body.stop_token_ids.unwrap_or_default();
    req.include_stop_str_in_output = body.include_stop_str_in_output;
    req.response_format = crate::response_format(body.response_format)?;
    Ok(Parsed {
        req,
        prompt,
        echo: body.echo,
        include_usage: body.stream_options.is_some_and(|o| o.include_usage),
        n,
        logprobs,
        prompt_logprobs,
        as_ids: body.return_tokens_as_token_ids,
    })
}

fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::StopSequence | Finish::ToolCalls => "stop",
        Finish::Length => "length",
        Finish::Abort => "abort",
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

/// The legacy logprobs lists, built a run of tokens at a time: each token,
/// its logprob, its [`alternatives`] and the character its text starts at,
/// `null`s for a token nothing scored.
#[derive(Default)]
struct Legacy {
    as_ids: bool,
    tokens: Vec<String>,
    values: Vec<Value>,
    tops: Vec<Value>,
    offsets: Vec<usize>,
}

impl Legacy {
    fn new(as_ids: bool) -> Self {
        Self {
            as_ids,
            ..Self::default()
        }
    }

    /// `steps` with the first token's text starting at character `offset`,
    /// each with at most `top` alternatives.
    fn push<'a>(
        &mut self,
        steps: impl IntoIterator<Item = (u32, Option<&'a compat::Logprob>)>,
        offset: usize,
        top: usize,
    ) -> &mut Self {
        let mut at = offset;
        for (token, lp) in steps {
            self.offsets.push(at);
            at += token_text(token).chars().count();
            self.tokens.push(token_label(token, self.as_ids));
            let Some(lp) = lp else {
                self.values.push(Value::Null);
                self.tops.push(Value::Null);
                continue;
            };
            let lp = compat::Logprob {
                top: lp.top.iter().copied().take(top).collect(),
                ..lp.clone()
            };
            let mut alts = serde_json::Map::new();
            for (alt, alt_lp) in alternatives(token, &lp) {
                alts.entry(token_label(alt, self.as_ids))
                    .or_insert(json!(alt_lp.max(-9999.0)));
            }
            self.values.push(json!(lp.logprob.max(-9999.0)));
            self.tops.push(Value::Object(alts));
        }
        self
    }

    fn sampled(&mut self, steps: &[(u32, compat::Logprob)], offset: usize) -> &mut Self {
        let top = steps.first().map_or(0, |(_, lp)| lp.top.len());
        self.push(steps.iter().map(|(t, lp)| (*t, Some(lp))), offset, top)
    }

    fn json(&mut self) -> Value {
        let this = std::mem::take(self);
        json!({"tokens": this.tokens, "token_logprobs": this.values, "top_logprobs": this.tops, "text_offset": this.offsets})
    }
}

/// The echoed prompt's tokens and their scores: the prompt's, past the
/// model's prefix, which the echoed text does not spell.
fn echoed(prompt: &[(u32, Option<compat::Logprob>)]) -> &[(u32, Option<compat::Logprob>)] {
    let prefix = inferlet::chat::prefix().len();
    prompt.get(prefix..).unwrap_or_default()
}

/// Legacy logprobs over sampled `steps`, the first token's text starting at
/// character `offset`.
fn legacy_logprobs(parsed: &Parsed, steps: &[(u32, compat::Logprob)], offset: usize) -> Value {
    Legacy::new(parsed.as_ids).sampled(steps, offset).json()
}

/// Legacy logprobs over the echoed prompt (where scored), then the sampled
/// `steps` after the echoed text.
fn echoed_legacy_logprobs(
    parsed: &Parsed,
    prompt: &[(u32, Option<compat::Logprob>)],
    steps: &[(u32, compat::Logprob)],
) -> Value {
    let mut legacy = Legacy::new(parsed.as_ids);
    if let Some(top) = parsed.echo_logprobs() {
        let prompt = echoed(prompt).iter().map(|(t, lp)| (*t, lp.as_ref()));
        legacy.push(prompt, 0, top as usize);
    }
    legacy.sampled(steps, text_offset(parsed)).json()
}

/// Where a choice's generated text starts, in characters: after the prompt
/// where it is echoed.
fn text_offset(parsed: &Parsed) -> usize {
    if parsed.echo {
        parsed.prompt.chars().count()
    } else {
        0
    }
}

/// One choice's generation, its chunks streamed under its index.
async fn choice(
    parsed: &Parsed,
    req: Request,
    ids: &Ids,
) -> Result<(compat::Outcome, String), String> {
    let index = req.choice;
    let stream = req.stream;
    let asked = req.sampling.logprobs.is_some();
    // Scored, the echo waits for its scores, which come before any token.
    let echo_scored = parsed.echo_logprobs().is_some();
    if stream && parsed.echo && !echo_scored {
        emit::send(&ids.chunk(index, &parsed.prompt, None, Value::Null));
    }
    let mut text = String::new();
    let mut offset = text_offset(parsed);
    let mut pending: Vec<(u32, compat::Logprob)> = Vec::new();
    let token_ids = req.return_token_ids;
    let mut pending_ids = crate::PendingIds::default();
    let send = |mut chunk: Value, pending_ids: &mut crate::PendingIds| {
        if token_ids {
            pending_ids.attach(&mut chunk, true);
        }
        emit::send(&chunk);
    };
    let outcome = compat::run(&req, |event| {
        let piece = match event {
            Event::Text(t) => t,
            Event::Reasoning(_) | Event::ReasoningEnd | Event::ToolCall(_) => return,
            Event::Logprobs(steps) => {
                pending.extend(steps);
                return;
            }
            Event::PromptLogprobs(prompt) => {
                if stream && echo_scored {
                    let logprobs = echoed_legacy_logprobs(parsed, &prompt, &[]);
                    send(
                        ids.chunk(index, &parsed.prompt, None, logprobs),
                        &mut pending_ids,
                    );
                }
                return;
            }
            Event::PromptTokens(prompt) => {
                pending_ids.prompt = Some(prompt);
                return;
            }
            Event::Tokens(tokens) => {
                pending_ids.tokens.extend(tokens);
                return;
            }
        };
        if stream {
            let logprobs = if asked {
                let lp = legacy_logprobs(parsed, &pending, offset);
                pending.clear();
                lp
            } else {
                Value::Null
            };
            offset += piece.chars().count();
            send(ids.chunk(index, &piece, None, logprobs), &mut pending_ids);
        } else {
            text.push_str(&piece);
        }
    })
    .await?;
    if stream {
        let logprobs = if asked {
            legacy_logprobs(parsed, &pending, offset)
        } else {
            Value::Null
        };
        let mut last = ids.chunk(index, "", Some(finish_reason(outcome.finish)), logprobs);
        last["choices"][0]["stop_reason"] = stop_reason(&outcome);
        send(last, &mut pending_ids);
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
    let outcomes = run_choices(&parsed.req, parsed.n, |req| choice(&parsed, req, &ids)).await?;
    let total = usage_of(outcomes.iter().map(|(o, _)| o));

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
    // Only the first choice scored the prompt ([`crate::choice_request`]).
    let first_scored = outcomes
        .first()
        .map(|(o, _)| o.prompt_logprobs.clone())
        .unwrap_or_default();
    let scored = &first_scored;
    let choices: Vec<Value> = outcomes
        .into_iter()
        .zip(0u32..)
        .map(|((outcome, text), index)| {
            let text = if parsed.echo {
                format!("{}{text}", parsed.prompt)
            } else {
                text
            };
            let logprobs = if asked {
                echoed_legacy_logprobs(&parsed, scored, &outcome.logprobs)
            } else {
                Value::Null
            };
            let mut choice = json!({"index": index, "text": text, "logprobs": logprobs, "finish_reason": finish_reason(outcome.finish), "stop_reason": stop_reason(&outcome)});
            if let Some(top) = parsed.prompt_logprobs {
                choice["prompt_logprobs"] = crate::vllm_prompt_logprobs(scored, top as usize);
            }
            if parsed.req.return_token_ids {
                choice["prompt_token_ids"] = json!(outcome.prompt_token_ids);
                choice["token_ids"] = json!(outcome.token_ids);
            }
            choice
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_echo_with_logprobs_scores_the_prompt_with_as_many_alternatives() {
        let parsed = parse(json!({"prompt": "Hi", "echo": true, "logprobs": 3})).unwrap();
        assert_eq!(parsed.req.sampling.prompt_logprobs, Some(3));
        assert_eq!(parsed.echo_logprobs(), Some(3));
        assert_eq!(parsed.prompt_logprobs, None);
    }

    #[test]
    fn logprobs_without_echo_leave_the_prompt_unscored() {
        for body in [
            json!({"prompt": "Hi", "logprobs": 3}),
            json!({"prompt": "Hi", "echo": true}),
        ] {
            let parsed = parse(body).unwrap();
            assert_eq!(parsed.req.sampling.prompt_logprobs, None);
            assert_eq!(parsed.echo_logprobs(), None);
        }
    }

    #[test]
    fn prompt_logprobs_beside_an_echo_score_with_the_wider_of_the_two() {
        let parsed = parse(json!({
            "prompt": "Hi", "echo": true, "logprobs": 2, "prompt_logprobs": 5,
        }))
        .unwrap();
        assert_eq!(parsed.req.sampling.prompt_logprobs, Some(5));
        assert_eq!(parsed.echo_logprobs(), Some(2));
        assert_eq!(parsed.prompt_logprobs, Some(5));
    }

    #[test]
    fn a_structural_tag_reads_its_structures_and_triggers() {
        let parsed = parse(json!({
            "prompt": "Hi",
            "response_format": {
                "type": "structural_tag",
                "structures": [{"begin": "<function=f>", "schema": {"type": "object"}, "end": "</function>"}],
                "triggers": ["<function="],
            },
        }))
        .unwrap();
        let compat::ResponseFormat::StructuralTag(tag) = parsed.req.response_format else {
            panic!("not a structural tag");
        };
        assert_eq!(tag.triggers, vec!["<function=".to_string()]);
        assert_eq!(tag.structures[0].begin, "<function=f>");
        assert_eq!(tag.structures[0].schema, r#"{"type":"object"}"#);
        assert_eq!(tag.structures[0].end, "</function>");
    }

    #[test]
    fn a_structural_tag_without_structures_or_with_a_lone_trigger_is_refused() {
        for format in [
            json!({"type": "structural_tag", "structures": [], "triggers": ["<f"]}),
            json!({"type": "structural_tag", "structures": [{"begin": "<f>", "schema": {}}]}),
            json!({"type": "structural_tag", "structures": [{"begin": "<f>"}], "triggers": ["<f"]}),
            json!({"type": "structural_tag", "structures": [{"begin": "<f>", "schema": {}}], "triggers": ["<g"]}),
            json!({"type": "structural_tag", "format": {"type": "triggered_tags"}}),
        ] {
            assert!(parse(json!({"prompt": "Hi", "response_format": format})).is_err());
        }
    }

    #[test]
    fn a_logit_bias_is_refused_by_name() {
        let refused = parse(json!({"prompt": "Hi", "logit_bias": {"50256": -100}}))
            .err()
            .expect("a logit_bias is refused");
        assert!(refused.body.to_string().contains("logit_bias is not supported"));
        for bias in [json!(null), json!({})] {
            assert!(parse(json!({"prompt": "Hi", "logit_bias": bias})).is_ok());
        }
    }

    #[test]
    fn prompt_logprobs_streamed_negative_or_past_the_cap_are_refused() {
        for body in [
            json!({"prompt": "Hi", "prompt_logprobs": 1, "stream": true}),
            json!({"prompt": "Hi", "prompt_logprobs": -1}),
            json!({"prompt": "Hi", "prompt_logprobs": 21}),
        ] {
            assert!(parse(body).is_err());
        }
    }
}
