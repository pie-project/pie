//! The request rendered to tokens through the host's chat template.
//!
//! The host fills roles (`inferlet::chat`) and tool turns (`inferlet::tools`)
//! for whichever model is loaded, so this module only decides the order:
//! leading system messages, then the tool manifest, then the conversation,
//! then the generation cue.

use crate::request::{Message, Request, Role, Thinking};
use inferlet::pie::inferlet::tools;
use inferlet::{chat, model};

pub fn build(req: &Request) -> Result<Vec<u32>, String> {
    let mut tokens = Vec::new();
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
                    tokens.extend(if seen_user {
                        chat::user(&m.content)
                    } else {
                        chat::first_user(&m.content)
                    });
                    seen_user = true;
                }
                Role::Assistant => tokens.extend(chat::assistant(&assistant_body(m))),
                Role::Tool => tokens.extend(tools::answer(
                    m.tool_name.as_deref().unwrap_or(""),
                    &m.content,
                )),
            }
        }

        tokens.extend(chat::cue());
        if req.thinking == Thinking::Disabled {
            tokens.extend(no_thinking_cue());
        }
    }
    // The prefill needs at least one token.
    if tokens.is_empty() {
        tokens.push(0);
    }
    Ok(tokens)
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

/// The tokens that tell a thinking model not to think this turn: an empty
/// thinking block right after the cue, which is what the model's own
/// template emits for `enable_thinking: false`. Only when the vocabulary
/// has the block markers as single tokens; on any other model the request
/// to disable thinking is moot and this is nothing.
fn no_thinking_cue() -> Vec<u32> {
    let open = model::encode("<think>");
    let close = model::encode("</think>");
    if open.len() != 1 || close.len() != 1 {
        return Vec::new();
    }
    let mut tokens = open;
    tokens.extend(model::encode("\n\n"));
    tokens.extend(close);
    tokens.extend(model::encode("\n\n"));
    tokens
}
