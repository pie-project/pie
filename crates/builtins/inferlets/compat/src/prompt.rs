//! The request rendered to tokens through the host's chat template.
//!
//! The host fills roles (`inferlet::chat`) and tool turns (`inferlet::tools`)
//! for whichever model is loaded, so this module only decides the order:
//! leading system messages, then the tool manifest, then the conversation,
//! then the generation cue. A user turn's pictures go just ahead of its text,
//! each as the span its handle spells.

use crate::request::{Message, Request, Role, Thinking};
use inferlet::pie::inferlet::tools;
use inferlet::{chat, media, model};

pub struct Prompt {
    pub tokens: Vec<u32>,
    /// Each picture with the row its span starts at, in order.
    pub images: Vec<(u32, media::Image)>,
}

pub fn build(req: &Request) -> Result<Prompt, String> {
    let mut tokens = Vec::new();
    let mut images = Vec::new();
    if let Some(raw) = &req.raw_prompt {
        tokens.extend(chat::prefix());
        tokens.extend(model::encode(raw));
    } else {
        let messages = &req.messages;
        let lead = messages
            .iter()
            .take_while(|m| m.role == Role::System)
            .count();
        for m in &messages[..lead] {
            tokens.extend(chat::system(&m.content));
        }

        // Tools the model may not call (`tool_choice: none`) are still
        // described, because the conversation may replay calls to them.
        if !req.tools.is_empty() {
            let equipped = tools::equip(&req.tools).map_err(|e| format!("equip tools: {e:?}"))?;
            tokens.extend(equipped);
        }

        let mut seen_user = false;
        for m in &messages[lead..] {
            match m.role {
                Role::System => tokens.extend(chat::system(&m.content)),
                Role::User => {
                    let turn = if seen_user {
                        chat::user(&m.content)
                    } else {
                        chat::first_user(&m.content)
                    };
                    let at = text_at(&turn, &m.content);
                    tokens.extend(&turn[..at]);
                    for bytes in &m.images {
                        let image =
                            media::Image::from_bytes(bytes).map_err(|e| format!("image: {e:?}"))?;
                        images.push((tokens.len() as u32, image));
                        tokens.extend(images.last().unwrap().1.tokens());
                    }
                    tokens.extend(&turn[at..]);
                    seen_user = true;
                }
                Role::Assistant => tokens.extend(chat::assistant(&assistant_body(m))),
                Role::Tool => tokens.extend(tools::answer(
                    m.tool_name.as_deref().unwrap_or(""),
                    &m.content,
                )),
            }
        }

        tokens.extend(chat::cue_thinking(req.thinking != Thinking::Disabled));
    }
    // The prefill needs at least one token.
    if tokens.is_empty() {
        tokens.push(0);
    }
    Ok(Prompt { tokens, images })
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

/// An assistant turn's replayed body: its text, then each tool call it made
/// in the model's own call syntax, so the model sees its earlier calls the
/// way it wrote them.
///
/// The call syntax is the ChatML one (`<tool_call>{…}</tool_call>`), which is
/// what every tool-capable template in the host currently parses. A host
/// filler for "an assistant turn that called tools" would make this exact
/// for every template; until then, this is the one place it is spelled out.
fn assistant_body(m: &Message) -> String {
    let mut body = m.content.clone();
    for call in &m.tool_calls {
        let arguments: serde_json::Value = serde_json::from_str(&call.arguments)
            .unwrap_or_else(|_| serde_json::Value::String(call.arguments.clone()));
        let payload = serde_json::json!({"name": call.name, "arguments": arguments});
        if !body.is_empty() {
            body.push('\n');
        }
        body.push_str("<tool_call>\n");
        body.push_str(&payload.to_string());
        body.push_str("\n</tool_call>");
    }
    body
}
