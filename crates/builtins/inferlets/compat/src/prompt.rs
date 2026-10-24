//! The request rendered to tokens through the host's chat template.
//!
//! The host fills roles (`inferlet::chat`) and tool turns (`inferlet::tools`)
//! for whichever model is loaded, so this module only decides the order:
//! the opening system turn (the system message, the tools and the reasoning
//! effort, as the template combines them), then the conversation, then the
//! generation cue. A user turn's pictures go just ahead of its text,
//! each as the span its handle spells.

use crate::request::{Effort, Message, Request, Role, Thinking};
use inferlet::pie::inferlet::tools;
use inferlet::{chat, media, model};

pub struct Prompt {
    pub tokens: Vec<u32>,
    /// Each picture with the row its span starts at, in order.
    pub images: Vec<(u32, media::Image)>,
    pub ends: Vec<u32>,
    /// The tokens the opening system turn takes, when the template wrote
    /// one.
    pub system: Option<u32>,
}

pub fn build(req: &Request) -> Result<Prompt, String> {
    let mut tokens = Vec::new();
    let mut images = Vec::new();
    let mut ends = Vec::new();
    let mut system_tokens = None;
    if let Some(raw) = &req.raw_prompt {
        tokens.extend(chat::prefix());
        tokens.extend(model::encode(raw));
        ends.push(tokens.len() as u32);
    } else {
        let messages = &req.messages;
        // The template opens on one system turn: the first system message,
        // the tools (described even under `tool_choice: none`, because the
        // conversation may replay calls to them) and a thinking request's
        // reasoning effort, in the order the template writes them.
        let lead = messages
            .iter()
            .take_while(|m| m.role == Role::System)
            .count();
        let system = messages
            .first()
            .filter(|_| lead > 0)
            .map(|m| m.content.clone());
        tokens.extend(chat::open(system.as_deref(), &req.tools, effort(req)));
        if !tokens.is_empty() {
            ends.push(tokens.len() as u32);
            system_tokens = Some(tokens.len() as u32);
        }
        for m in messages.iter().take(lead).skip(1) {
            tokens.extend(chat::system(&m.content));
            ends.push(tokens.len() as u32);
        }

        // Turns after the last user query are the current tool loop, which
        // some templates replay with their reasoning.
        let last_query = messages.iter().rposition(|m| m.role == Role::User);
        // The opening (a system turn, the tools) already opened the
        // conversation; `first_user` opens it only where nothing has.
        let mut seen_user = !tokens.is_empty();
        let mut at = lead;
        while at < messages.len() {
            let m = &messages[at];
            match m.role {
                Role::System => tokens.extend(chat::system(&m.content)),
                Role::User => {
                    let turn = if seen_user {
                        chat::user(&m.content)
                    } else {
                        chat::first_user(&m.content)
                    };
                    let text_at = text_at(&turn, &m.content);
                    tokens.extend(&turn[..text_at]);
                    for bytes in &m.images {
                        let image =
                            media::Image::from_bytes(bytes).map_err(|e| format!("image: {e:?}"))?;
                        images.push((tokens.len() as u32, image));
                        tokens.extend(images.last().unwrap().1.tokens());
                    }
                    tokens.extend(&turn[text_at..]);
                    seen_user = true;
                }
                Role::Assistant => {
                    let calls: Vec<tools::ToolCall> = m
                        .tool_calls
                        .iter()
                        .map(|c| tools::ToolCall {
                            name: c.name.clone(),
                            arguments_json: c.arguments.clone(),
                        })
                        .collect();
                    let current = last_query.is_some_and(|q| at > q);
                    tokens.extend(tools::assistant_calls(&assistant_body(m), &calls, current));
                }
                Role::Tool => {
                    // The template writes consecutive results in one turn.
                    let run = messages[at..]
                        .iter()
                        .take_while(|m| m.role == Role::Tool)
                        .map(|m| (m.tool_name.clone().unwrap_or_default(), m.content.clone()))
                        .collect::<Vec<_>>();
                    at += run.len() - 1;
                    tokens.extend(tools::answers(&run));
                }
            }
            ends.push(tokens.len() as u32);
            at += 1;
        }

        tokens.extend(chat::cue_thinking(thinks(req)));
    }
    // The prefill needs at least one token.
    if tokens.is_empty() {
        tokens.push(0);
    }
    Ok(Prompt {
        tokens,
        images,
        ends,
        system: system_tokens,
    })
}

/// Whether `req` asks for a thinking answer; [`build`] cues it so, and the
/// demux reads the same cue.
pub(crate) fn thinks(req: &Request) -> bool {
    req.thinking != Thinking::Disabled
}

/// Where `text` begins in its rendered `turn`, so pictures sit inside the
/// turn ahead of it; the turn's start if it cannot be found.
fn text_at(turn: &[u32], text: &str) -> usize {
    let find = |turn: &[u32], body: &[u32]| {
        (!body.is_empty())
            .then(|| turn.windows(body.len()).position(|w| w == body))
            .flatten()
    };
    if let Some(at) = find(turn, &model::encode(text)) {
        return at;
    }
    // An empty text: the turn ends with the closing a probe turn has after its text.
    let probe = model::encode("x");
    let probed = chat::user("x");
    find(&probed, &probe)
        .map(|at| turn.len().saturating_sub(probed.len() - at - probe.len()))
        .unwrap_or(0)
}

/// The reasoning effort the system turn states: none for a request the
/// model answers without thinking.
fn effort(req: &Request) -> Option<chat::ReasoningEffort> {
    thinks(req).then_some(match req.effort {
        Effort::Xhigh => chat::ReasoningEffort::Xhigh,
        Effort::Medium => chat::ReasoningEffort::Medium,
        Effort::Low => chat::ReasoningEffort::Low,
    })
}

/// An assistant turn's text with the reasoning it carried in front, inside
/// the thinking block the host's template splits it back out of.
fn assistant_body(m: &Message) -> String {
    if m.reasoning.trim().is_empty() {
        m.content.clone()
    } else {
        format!("<think>\n{}\n</think>\n\n{}", m.reasoning.trim(), m.content)
    }
}
