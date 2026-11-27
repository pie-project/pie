//! One request, start to finish: prompt, constraint, generation, and the
//! classified event stream — with stop tokens, stop sequences and the
//! token budget applied.

use std::cell::RefCell;
use std::ops::ControlFlow;

use inferlet::chat;
use inferlet::grammar::{Grammar, Matcher};
use inferlet::pie::inferlet::tools;

use crate::demux::{Demux, Event};
use crate::generate::{Guide, Logprob, generate};
use crate::holdback::HoldBack;
use crate::prompt;
use crate::request::{Request, ResponseFormat, ToolCall};
use crate::trigger::{self, Triggered};

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// The token budget ran out.
    Length,
    /// The turn ended with tool calls.
    ToolCalls,
    StopSequence,
    /// The host stopped it (a policy's terminate): the tokens so far stand,
    /// and [`Outcome::stop_reason`] says why.
    Abort,
}

/// How a fire refused because the host stopped the process reads: this,
/// then the reason. The runtime's `policy::STOPPED`.
pub const STOPPED: &str = "stopped by policy: ";

#[derive(Clone, Debug)]
pub struct Outcome {
    pub finish: Finish,
    /// The stop sequence that ended generation, when one did.
    pub stop_sequence: Option<String>,
    /// Why the host stopped it, when it did ([`Finish::Abort`]).
    pub stop_reason: Option<String>,
    pub stop_token: Option<u32>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    /// Every sampled token's logprobs, in order (the stop token that ended
    /// the turn among them), where the request asked for them.
    pub logprobs: Vec<(u32, Logprob)>,
    /// Every prompt token with its logprob under the tokens before it
    /// (`None` for the first, which nothing conditions), the model's prefix
    /// among them, where the request asked for them.
    pub prompt_logprobs: Vec<(u32, Option<Logprob>)>,
    pub prompt_token_ids: Vec<u32>,
    pub token_ids: Vec<u32>,
}

/// The prompt's tokens beside their scores: the first has none, each after
/// it the score of the row before it.
fn pair_prompt(tokens: &[u32], scored: &[Logprob]) -> Vec<(u32, Option<Logprob>)> {
    tokens
        .iter()
        .zip(std::iter::once(None).chain(scored.iter().cloned().map(Some)))
        .map(|(&token, lp)| (token, lp))
        .collect()
}

/// The constraint a request puts on the output, as a live guide.
fn constraint(req: &Request) -> Result<Option<Box<dyn Guide>>, String> {
    let forced = req.forces_tool_call();
    let formatted = req.response_format != ResponseFormat::Text;
    if forced && formatted {
        return Err(
            "a forced tool call and a response_format cannot both constrain one response".into(),
        );
    }
    if forced {
        let callable = req.callable_tools();
        if callable.is_empty() {
            return Err("tool_choice names a tool that is not in `tools`".into());
        }
        // A model with no tool-call syntax cannot be held to one.
        let grammar = tools::format(&callable)
            .ok_or("tool_choice forces a call, but this model has no tool-call format")?;
        return Ok(Some(Box::new(Matcher::new(&grammar))));
    }
    let grammar = match &req.response_format {
        ResponseFormat::Text => return Ok(None),
        ResponseFormat::StructuralTag(tag) => {
            tag.validate()?;
            let triggered = Triggered::new(tag.clone(), trigger::Host::new());
            return Ok(Some(Box::new(triggered)));
        }
        ResponseFormat::JsonObject => Grammar::json(),
        ResponseFormat::JsonSchema(schema) => Grammar::from_json_schema(schema)
            .map_err(|e| format!("response_format schema: {e:?}"))?,
    };
    Ok(Some(Box::new(Matcher::new(&grammar))))
}

/// Content on its way to the sink passes the stop-sequence filter; what the
/// filter lets through is the content. Every other event goes straight out.
struct Output {
    stops: HoldBack,
    include_stop: bool,
    content: String,
    finish: Option<Finish>,
    stop_sequence: Option<String>,
    stop_token: Option<u32>,
}

impl Output {
    fn new(req: &Request) -> Self {
        Self {
            stops: HoldBack::new(req.stop.clone()),
            include_stop: req.include_stop_str_in_output,
            content: String::new(),
            finish: None,
            stop_sequence: None,
            stop_token: None,
        }
    }

    fn deliver(&mut self, events: Vec<Event>, sink: &mut dyn FnMut(Event)) -> ControlFlow<()> {
        for event in events {
            match event {
                Event::Text(text) => {
                    let released = self.stops.push(&text);
                    self.send(released.text, sink);
                    if let Some(hit) = released.hit {
                        let marker = self.stops.marker(hit).to_string();
                        if self.include_stop {
                            self.send(marker.clone(), sink);
                        }
                        self.finish = Some(Finish::StopSequence);
                        self.stop_sequence = Some(marker);
                        return ControlFlow::Break(());
                    }
                }
                other => sink(other),
            }
        }
        ControlFlow::Continue(())
    }

    fn end(&mut self, held: Vec<Event>, sink: &mut dyn FnMut(Event)) {
        if self.stop_sequence.is_some() {
            return;
        }
        if self.deliver(held, sink).is_continue() {
            let rest = self.stops.flush();
            self.send(rest, sink);
        }
    }

    fn send(&mut self, text: String, sink: &mut dyn FnMut(Event)) {
        if !text.is_empty() {
            self.content.push_str(&text);
            sink(Event::Text(text));
        }
    }
}

/// Run `req`, handing every event to `sink` as it happens.
pub async fn run(req: &Request, sink: impl FnMut(Event)) -> Result<Outcome, String> {
    // The prompt's logprobs and the tokens' events both go out through it.
    let sink = RefCell::new(sink);
    let mut prompt = prompt::build(req)?;
    let prompt_len = prompt.tokens.len();
    // A turn continued here goes on after what the client was sent.
    let continued = crate::continuation::get()
        .map(|c| c.tokens)
        .unwrap_or_default();
    prompt.tokens.extend_from_slice(&continued);
    let mut generated = continued.len() as u32;
    let mut guide = constraint(req)?;
    let stop_tokens = chat::stop_tokens();
    let callable = !req.callable_tools().is_empty();
    let mut demux = Demux::new(
        &chat::cue_thinking(prompt::thinks(req)),
        req.choice,
        callable,
    );
    if req.raw_prompt.is_some() || req.prompt_tokens.is_some() {
        demux.verbatim();
    }
    let mut out = Output::new(req);
    let mut error: Option<String> = None;
    let ids = req.return_token_ids;
    let prompt_token_ids = if ids {
        prompt.tokens[..prompt_len].to_vec()
    } else {
        Vec::new()
    };
    if ids && req.stream {
        (sink.borrow_mut())(Event::PromptTokens(prompt_token_ids.clone()));
    }
    let mut token_ids: Vec<u32> = Vec::new();
    // What the client was sent already passes through the decoders, the
    // stop hold-back and the adapter's stream, whose frames the client has
    // (`crate::continuation`), so they all stand where they stood.
    for &token in &continued {
        if ids {
            token_ids.push(token);
            if req.stream {
                (sink.borrow_mut())(Event::Tokens(vec![token]));
            }
        }
        if let Ok(events) = demux.feed(token) {
            let _ = out.deliver(events, &mut *sink.borrow_mut());
        }
    }

    // A continued turn's own tokens are no part of the prompt it scores.
    let mut prompt_logprobs = Vec::new();
    let mut on_prompt = |scored: &[Logprob]| {
        prompt_logprobs = pair_prompt(&prompt.tokens[..prompt_len], scored);
        if req.stream {
            (sink.borrow_mut())(Event::PromptLogprobs(prompt_logprobs.clone()));
        }
    };

    let mut logprobs: Vec<(u32, Logprob)> = Vec::new();
    let mut on_token = |token: u32, logprob: Option<Logprob>| -> ControlFlow<()> {
        crate::continuation::note(token);
        if ids {
            token_ids.push(token);
            if req.stream {
                (sink.borrow_mut())(Event::Tokens(vec![token]));
            }
        }
        if let Some(logprob) = logprob {
            if req.stream {
                (sink.borrow_mut())(Event::Logprobs(vec![(token, logprob.clone())]));
            }
            logprobs.push((token, logprob));
        }
        generated += 1;
        if stop_tokens.contains(&token) {
            if req.ignore_eos || generated as usize <= req.min_tokens {
                return ControlFlow::Continue(());
            }
            out.finish = Some(Finish::Stop);
            return ControlFlow::Break(());
        }
        let stop_token = req.stop_token_ids.contains(&token) && generated as usize > req.min_tokens;
        if stop_token && !req.include_stop_str_in_output {
            out.finish = Some(Finish::StopSequence);
            out.stop_token = Some(token);
            return ControlFlow::Break(());
        }
        let flow = match demux.feed(token) {
            Ok(events) => out.deliver(events, &mut *sink.borrow_mut()),
            Err(e) => {
                error = Some(e);
                return ControlFlow::Break(());
            }
        };
        if stop_token && flow.is_continue() {
            out.finish = Some(Finish::StopSequence);
            out.stop_token = Some(token);
            return ControlFlow::Break(());
        }
        flow
    };
    // Its passes state their rows final, stop sequences or
    // none: a stop sequence only cuts the output (`Output::deliver`), the
    // set ends with the request, and nothing is taken back.
    let ran = generate(
        &prompt,
        req.sampling,
        req.max_tokens.saturating_sub(continued.len()),
        guide.as_mut().map(|g| g.as_mut() as &mut dyn Guide),
        &mut on_prompt,
        &mut on_token,
    )
    .await;
    // Stopped by the host, it ends here with what it sampled.
    let mut stop_reason = None;
    let sampled = match ran {
        Ok(done) => done.sampled,
        Err(e) => match e.find(STOPPED) {
            Some(at) => {
                stop_reason = Some(e[at + STOPPED.len()..].to_string());
                generated as usize - continued.len()
            }
            None => return Err(e),
        },
    };
    let cached_tokens = inferlet::session::cached_tokens() as usize;
    let mut sink = sink.into_inner();
    if let Some(e) = error {
        return Err(e);
    }

    out.end(demux.finish(), &mut sink);

    let mut finish = match out.finish {
        _ if stop_reason.is_some() => Finish::Abort,
        Some(f) => f,
        None if guide.as_ref().is_some_and(|g| g.is_terminated()) => Finish::Stop,
        None => Finish::Length,
    };
    if finish == Finish::Stop && !demux.tool_calls.is_empty() {
        finish = Finish::ToolCalls;
    }

    // A continued turn's counts are the turn's, not this launch's: what the
    // client was sent before its worker was lost is completion, not prompt.
    let continued = continued.len();
    Ok(Outcome {
        finish,
        stop_sequence: out.stop_sequence,
        stop_token: out.stop_token,
        stop_reason,
        prompt_tokens: prompt.tokens.len() - continued,
        cached_tokens: cached_tokens.min(prompt.tokens.len() - continued),
        completion_tokens: sampled + continued,
        content: out.content,
        reasoning: demux.reasoning_text,
        tool_calls: demux.tool_calls,
        logprobs,
        prompt_logprobs,
        prompt_token_ids,
        token_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_prompt_token_has_no_score_and_each_after_it_the_row_before_its() {
        let score = |logprob| Logprob {
            logprob,
            top: Vec::new(),
            rank: Some(1),
        };
        let paired = pair_prompt(&[10, 11, 12], &[score(-1.0), score(-2.0)]);
        assert_eq!(
            paired,
            vec![(10, None), (11, Some(score(-1.0))), (12, Some(score(-2.0)))]
        );
    }

    fn stream(stop: &str, include: bool, pieces: &[&str]) -> (String, Option<String>) {
        let mut req = Request::new(Vec::new());
        req.stop = vec![stop.to_string()];
        req.include_stop_str_in_output = include;
        let mut out = Output::new(&req);
        let mut sent = String::new();
        let mut sink = |event: Event| {
            if let Event::Text(text) = event {
                sent.push_str(&text);
            }
        };
        for piece in pieces {
            if out
                .deliver(vec![Event::Text(piece.to_string())], &mut sink)
                .is_break()
            {
                break;
            }
        }
        out.end(vec![Event::Text("<after>".into())], &mut sink);
        assert_eq!(sent, out.content, "what streamed is the content");
        (out.content, out.stop_sequence)
    }

    #[test]
    fn a_stop_sequence_ends_the_content_and_is_in_it_only_when_asked() {
        let pieces = ["{\"city\": 1}\n</tool", "_call>\nmore"];
        let stop = Some("</tool_call>".to_string());
        assert_eq!(
            stream("</tool_call>", false, &pieces),
            ("{\"city\": 1}\n".into(), stop.clone())
        );
        assert_eq!(
            stream("</tool_call>", true, &pieces),
            ("{\"city\": 1}\n</tool_call>".into(), stop)
        );
    }

    #[test]
    fn text_held_for_a_stop_sequence_that_never_came_is_content_at_the_end() {
        assert_eq!(
            stream("END", false, &["THE E"]),
            ("THE E<after>".into(), None)
        );
    }
}
