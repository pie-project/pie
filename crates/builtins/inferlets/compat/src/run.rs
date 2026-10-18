//! One request, start to finish: prompt, constraint, generation, and the
//! classified event stream — with stop tokens, stop sequences and the
//! token budget applied.

use std::ops::ControlFlow;

use inferlet::chat;
use inferlet::grammar::{Grammar, Matcher};
use inferlet::pie::inferlet::tools;

use crate::demux::{Demux, Event};
use crate::generate::generate;
use crate::holdback::HoldBack;
use crate::prompt;
use crate::request::{Request, ResponseFormat, ToolCall};

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// The token budget ran out.
    Length,
    /// The turn ended with tool calls.
    ToolCalls,
    /// A stop sequence appeared.
    StopSequence,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub finish: Finish,
    /// The stop sequence that ended generation, when one did.
    pub stop_sequence: Option<String>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Each report takes a lock on the host.
const REPORT_EVERY: u32 = 32;

/// The constraint a request puts on the output, as a live matcher.
fn constraint(req: &Request) -> Result<Option<Matcher>, String> {
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
        return Ok(Some(tools::create_matcher(&callable)));
    }
    let grammar = match &req.response_format {
        ResponseFormat::Text => return Ok(None),
        ResponseFormat::JsonObject => Grammar::json(),
        ResponseFormat::JsonSchema(schema) => Grammar::from_json_schema(schema)
            .map_err(|e| format!("response_format schema: {e:?}"))?,
    };
    Ok(Some(Matcher::new(&grammar)))
}

/// Content on its way to the sink passes the stop-sequence filter; what the
/// filter lets through is the content. Every other event goes straight out.
struct Output {
    stops: HoldBack,
    content: String,
    finish: Option<Finish>,
    stop_hit: Option<usize>,
}

impl Output {
    fn deliver(&mut self, events: Vec<Event>, sink: &mut dyn FnMut(Event)) -> ControlFlow<()> {
        for event in events {
            match event {
                Event::Text(text) => {
                    let released = self.stops.push(&text);
                    self.send(released.text, sink);
                    if let Some(hit) = released.hit {
                        self.finish = Some(Finish::StopSequence);
                        self.stop_hit = Some(hit);
                        return ControlFlow::Break(());
                    }
                }
                other => sink(other),
            }
        }
        ControlFlow::Continue(())
    }

    fn send(&mut self, text: String, sink: &mut dyn FnMut(Event)) {
        if !text.is_empty() {
            self.content.push_str(&text);
            sink(Event::Text(text));
        }
    }
}

/// Run `req`, handing every event to `sink` as it happens.
pub async fn run(req: &Request, mut sink: impl FnMut(Event)) -> Result<Outcome, String> {
    let mut prompt = prompt::build(req)?;
    // The worker's policy sees the prompt while this process still waits.
    let prompt_tokens = prompt.tokens.len() as u32;
    inferlet::session::set_tokens(prompt_tokens, 0);
    // A turn that moved here continues after what the client was sent.
    let resumed = crate::resume::get().map(|r| r.tokens).unwrap_or_default();
    prompt.tokens.extend_from_slice(&resumed);
    let mut generated = resumed.len() as u32;
    let matcher = constraint(req)?;
    let stop_tokens = chat::stop_tokens();
    let mut demux = Demux::new(&chat::cue_thinking(prompt::thinks(req)));
    let mut out = Output {
        stops: HoldBack::new(req.stop.clone()),
        content: String::new(),
        finish: None,
        stop_hit: None,
    };
    let mut error: Option<String> = None;
    // What the client was sent already passes through the decoders and the
    // stop hold-back unseen, so they stand where they stood.
    for &token in &resumed {
        if let Ok(events) = demux.feed(token) {
            let _ = out.deliver(events, &mut |_| {});
        }
    }

    let mut on_token = |token: u32| -> ControlFlow<()> {
        crate::resume::note(token);
        generated += 1;
        if generated == 1 || generated.is_multiple_of(REPORT_EVERY) {
            inferlet::session::set_tokens(prompt_tokens, generated);
        }
        if stop_tokens.contains(&token) {
            if req.ignore_eos || generated as usize <= req.min_tokens {
                return ControlFlow::Continue(());
            }
            out.finish = Some(Finish::Stop);
            return ControlFlow::Break(());
        }
        match demux.feed(token) {
            Ok(events) => out.deliver(events, &mut sink),
            Err(e) => {
                error = Some(e);
                ControlFlow::Break(())
            }
        }
    };
    let (sampled, cached_tokens) = generate(
        &prompt,
        req.sampling,
        req.max_tokens.saturating_sub(resumed.len()),
        matcher.as_ref(),
        &mut on_token,
    )
    .await?;
    inferlet::session::set_tokens(prompt_tokens, generated);
    if let Some(e) = error {
        return Err(e);
    }

    let _ = out.deliver(demux.finish(), &mut sink);
    if out.finish.is_none() {
        let rest = out.stops.flush();
        out.send(rest, &mut sink);
    }

    let mut finish = match out.finish {
        Some(f) => f,
        None if matcher.as_ref().is_some_and(|m| m.is_terminated()) => Finish::Stop,
        None => Finish::Length,
    };
    if finish == Finish::Stop && !demux.tool_calls.is_empty() {
        finish = Finish::ToolCalls;
    }

    // A resumed turn's counts are the turn's, not this launch's: what the
    // client was sent before the move is completion, not prompt.
    let resumed = resumed.len();
    Ok(Outcome {
        finish,
        stop_sequence: out.stop_hit.and_then(|i| req.stop.get(i).cloned()),
        prompt_tokens: prompt.tokens.len() - resumed,
        cached_tokens: cached_tokens.min(prompt.tokens.len() - resumed),
        completion_tokens: sampled + resumed,
        content: out.content,
        reasoning: demux.reasoning_text,
        tool_calls: demux.tool_calls,
    })
}
