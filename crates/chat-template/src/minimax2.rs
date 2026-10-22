//! MiniMax-M2.x's format (`chat_template.jinja` beside the checkpoint): a
//! prompt opens with `]~!b[` and a system turn (the caller's system prompt,
//! or the identity the release states, and the tools); every turn is
//! `]~b]role\n` … `[e~[\n`, the assistant's role `ai`; the cue opens a
//! `<think>` block the model closes; an earlier answer is replayed without
//! its reasoning; calls are `<minimax:tool_call>` blocks of `<invoke>`s, and
//! results come back in a `tool` turn as `<response>`s.

use std::sync::Arc;

use serde_json::Value;
use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::decode::{GenericChatDecoder, ThinkingDecoder};
use crate::minimax::{lift_calls, py_json};
use crate::{
    ChatDecoder, Effort, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, special, specials,
};

pub const STOP_TOKENS: &[&str] = &[EOS];

const BOD: &str = "]~!b[";
const BOS: &str = "]~b]";
const EOS: &str = "[e~[";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<minimax:tool_call>";
const CALL_CLOSE: &str = "</minimax:tool_call>";

pub struct MiniMax2 {
    tokenizer: Arc<Tokenizer>,
    identity: String,
    think_open: u32,
    think_close: u32,
    stop_ids: Vec<u32>,
}

impl MiniMax2 {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>, identity: &str) -> Self {
        let _ = specials(&tokenizer, &[BOD, BOS, EOS, CALL_OPEN, CALL_CLOSE]);
        Self {
            identity: identity.to_string(),
            think_open: special(&tokenizer, THINK_OPEN),
            think_close: special(&tokenizer, THINK_CLOSE),
            stop_ids: specials(&tokenizer, STOP_TOKENS),
            tokenizer,
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }
}

fn turn(role: &str, body: &str) -> String {
    format!("{BOS}{role}\n{body}{EOS}\n")
}

fn tools_section(tools: &[String]) -> String {
    let mut out = String::from(
        "\n\n# Tools\nYou may call one or more tools to assist with the user query.\n\
         Here are the tools available in JSONSchema format:\n\n<tools>\n",
    );
    for tool in tools {
        let function = match serde_json::from_str::<Value>(tool) {
            Ok(Value::Object(mut tool)) => match tool.remove("function") {
                Some(function) => function,
                None => Value::Object(tool),
            },
            Ok(other) => other,
            Err(_) => Value::String(tool.clone()),
        };
        out.push_str("<tool>");
        py_json(&function, &mut out);
        out.push_str("</tool>\n");
    }
    out.push_str(
        "</tools>\n\nWhen making tool calls, use XML format to invoke tools and pass parameters:\n\
         \n<minimax:tool_call>\n<invoke name=\"tool-name-1\">\n\
         <parameter name=\"param-key-1\">param-value-1</parameter>\n\
         <parameter name=\"param-key-2\">param-value-2</parameter>\n...\n</invoke>\n\
         </minimax:tool_call>",
    );
    out
}

/// An earlier assistant turn as the template replays it: its reasoning
/// dropped, its text, then its calls.
fn assistant_body(msg: &str) -> String {
    let (content, calls) = lift_calls(msg);
    let content = match content.rfind(THINK_CLOSE) {
        Some(at) => content[at + THINK_CLOSE.len()..].trim_matches('\n'),
        None => content.as_str(),
    };
    let mut out = content.to_string();
    if !calls.is_empty() {
        out.push_str(&format!("\n{CALL_OPEN}\n"));
        for (name, arguments) in &calls {
            out.push_str(&format!("<invoke name=\"{name}\">\n"));
            if let Value::Object(map) = arguments {
                for (key, value) in map {
                    out.push_str(&format!("<parameter name=\"{key}\">"));
                    match value {
                        Value::String(text) => out.push_str(text),
                        other => py_json(other, &mut out),
                    }
                    out.push_str("</parameter>\n");
                }
            }
            out.push_str("</invoke>\n");
        }
        out.push_str(CALL_CLOSE);
    }
    out
}

impl Instruct for MiniMax2 {
    fn prefix(&self) -> Vec<u32> {
        self.encode(BOD)
    }

    fn system(&self, msg: &str) -> Vec<u32> {
        self.encode(&format!("{BOD}{}", turn("system", msg)))
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.encode(&format!(
            "{BOD}{}{}",
            turn("system", &self.identity),
            turn("user", msg)
        ))
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.encode(&turn("user", msg))
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.encode(&turn("ai", &assistant_body(msg)))
    }

    fn cue(&self) -> Vec<u32> {
        self.encode(&format!("{BOS}ai\n{THINK_OPEN}\n"))
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    /// The opening with the tools in its system turn, as the template
    /// writes them for a conversation that states no system prompt.
    fn equip(&self, tools: &[String]) -> Vec<u32> {
        self.encode(&format!(
            "{BOD}{}",
            turn(
                "system",
                &format!("{}{}", self.identity, tools_section(tools))
            )
        ))
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        self.answers(&[(name.to_string(), value.to_string())])
    }

    /// Consecutive results in one `tool` turn.
    fn answers(&self, results: &[(String, String)]) -> Vec<u32> {
        if results.is_empty() {
            return Vec::new();
        }
        let body: Vec<String> = results
            .iter()
            .map(|(_, value)| format!("<response>{value}</response>"))
            .collect();
        self.encode(&turn("tool", &body.join("\n")))
    }

    /// One opening system turn: the system message (or the identity) and
    /// the tools.
    fn open(&self, system: Option<&str>, tools: &[String], _effort: Option<Effort>) -> Vec<u32> {
        if tools.is_empty() {
            return system.map(|msg| self.system(msg)).unwrap_or_default();
        }
        let first = system.unwrap_or(&self.identity);
        self.encode(&format!(
            "{BOD}{}",
            turn("system", &format!("{first}{}", tools_section(tools)))
        ))
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder::opened(
            self.tokenizer.clone(),
            vec![self.think_open],
            self.think_close,
        ))
    }

    fn reasoning_decoder_for_cue(&self, cue: &[u32]) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder::for_cue(
            self.tokenizer.clone(),
            vec![self.think_open],
            self.think_close,
            cue,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(MiniMax2ToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
        })
    }
}

/// The `<invoke name="…">` elements of a call block, each `<parameter
/// name="…">` value JSON where it parses as JSON and a string where not.
fn invokes(block: &str) -> Vec<(String, String)> {
    let mut calls = Vec::new();
    let mut rest = block;
    while let Some(at) = rest.find("<invoke name=\"") {
        let after = &rest[at + "<invoke name=\"".len()..];
        let Some(quote) = after.find("\">") else {
            break;
        };
        let name = after[..quote].to_string();
        let body_at = &after[quote + 2..];
        let end = body_at.find("</invoke>").unwrap_or(body_at.len());
        let mut body = &body_at[..end];
        let mut arguments = serde_json::Map::new();
        while let Some(p) = body.find("<parameter name=\"") {
            let after = &body[p + "<parameter name=\"".len()..];
            let Some(quote) = after.find("\">") else {
                break;
            };
            let key = after[..quote].to_string();
            let value_at = &after[quote + 2..];
            let Some(close) = value_at.find("</parameter>") else {
                break;
            };
            let raw = &value_at[..close];
            let value = serde_json::from_str::<Value>(raw.trim())
                .unwrap_or_else(|_| Value::String(raw.trim_matches('\n').to_string()));
            arguments.insert(key, value);
            body = &value_at[close + "</parameter>".len()..];
        }
        calls.push((name, Value::Object(arguments).to_string()));
        rest = &body_at[(end + "</invoke>".len()).min(body_at.len())..];
    }
    calls
}

struct MiniMax2ToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
}

impl ToolDecoder for MiniMax2ToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);
        let mut events = Vec::new();
        loop {
            if self.inside {
                let Some(at) = self.accumulated.find(CALL_CLOSE) else {
                    return events;
                };
                let block = self.accumulated[..at].to_string();
                self.accumulated = self.accumulated[at + CALL_CLOSE.len()..].to_string();
                self.inside = false;
                for (name, arguments) in invokes(&block) {
                    events.push(ToolEvent::Call(name, arguments));
                }
            } else {
                let Some(at) = self.accumulated.find(CALL_OPEN) else {
                    return events;
                };
                self.accumulated = self.accumulated[at + CALL_OPEN.len()..].to_string();
                self.inside = true;
                events.push(ToolEvent::Start);
            }
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax2_every_case() {
        an_invoke_reads_back_to_its_arguments();
    }

    fn an_invoke_reads_back_to_its_arguments() {
        let calls = invokes(
            "\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Paris</parameter>\n<parameter name=\"days\">2</parameter>\n</invoke>\n",
        );
        assert_eq!(
            calls,
            vec![(
                "get_weather".to_string(),
                r#"{"city":"Paris","days":2}"#.to_string()
            )]
        );
    }
}
