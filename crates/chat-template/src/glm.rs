use std::sync::Arc;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::decode::{GenericChatDecoder, NoopToolDecoder, ThinkingDecoder};
use crate::{
    ChatDecoder, Effort, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, special, specials,
};

pub const STOP_TOKENS: &[&str] = &["<|user|>", "<|observation|>", "<|endoftext|>"];
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const EFFORT: &str = "Reasoning Effort: Max";
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Which GLM release's template: GLM-5.3's (its effort line opening every
/// prompt), GLM-4.7's (the same turns, no effort line, XML tool calls), or
/// GLM-4.5 / 4.6's (a newline after each role marker, the thinking block
/// the model opens itself).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    Glm5,
    Glm47,
    Glm45,
}

pub struct Glm {
    tokenizer: Arc<Tokenizer>,
    style: Style,
    prefix: Vec<u32>,
    system_marker: u32,
    user_marker: u32,
    assistant_marker: u32,
    observation: u32,
    think_open: Vec<u32>,
    think_close: u32,
    stop_ids: Vec<u32>,
}

impl Glm {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self::with_style(tokenizer, Style::Glm5)
    }

    #[must_use]
    pub fn with_style(tokenizer: Arc<Tokenizer>, style: Style) -> Self {
        let stop_ids = specials(&tokenizer, STOP_TOKENS);
        let system_marker = special(&tokenizer, "<|system|>");
        let mut prefix = vec![special(&tokenizer, "[gMASK]"), special(&tokenizer, "<sop>")];
        if style == Style::Glm5 {
            prefix.push(system_marker);
            prefix.extend(tokenizer.encode(EFFORT));
        }
        let think_open = tokenizer.encode(THINK_OPEN);
        let think_close = tokenizer.encode(THINK_CLOSE);
        assert_eq!(
            think_close.len(),
            1,
            "`</think>` is one token in GLM's vocabulary"
        );
        Self {
            prefix,
            style,
            system_marker,
            user_marker: special(&tokenizer, "<|user|>"),
            assistant_marker: special(&tokenizer, "<|assistant|>"),
            observation: special(&tokenizer, "<|observation|>"),
            think_open,
            think_close: think_close[0],
            stop_ids,
            tokenizer,
        }
    }

    /// What GLM-4.5 writes after a role marker.
    fn gap(&self) -> &'static str {
        if self.style == Style::Glm45 { "\n" } else { "" }
    }

    fn turn(&self, marker: u32, msg: &str) -> Vec<u32> {
        let mut tokens = vec![marker];
        tokens.extend(self.tokenizer.encode(self.gap()));
        tokens.extend(self.tokenizer.encode(msg));
        tokens
    }

    fn assistant_body(msg: &str) -> String {
        if msg.trim_start().starts_with(THINK_OPEN) {
            msg.to_string()
        } else {
            format!("{THINK_OPEN}{THINK_CLOSE}{msg}")
        }
    }

    /// An earlier assistant turn as GLM-4.5 / 4.7 replay it: its reasoning
    /// dropped behind an empty block, its answer stripped, and each call it
    /// made (in ChatML's `<tool_call>{json}</tool_call>`) in GLM's XML.
    fn replay_body(&self, msg: &str) -> String {
        let answer = match msg.rfind(THINK_CLOSE) {
            Some(at) => msg[at + THINK_CLOSE.len()..].trim_start_matches('\n'),
            None => msg,
        };
        let (text, calls) = split_calls(answer);
        let nl = self.gap();
        let mut body = if self.style == Style::Glm45 {
            format!("\n{THINK_OPEN}{THINK_CLOSE}")
        } else {
            THINK_CLOSE.to_string()
        };
        let text = text.trim();
        if !text.is_empty() {
            body.push_str(nl);
            body.push_str(text);
        }
        for (name, arguments) in calls {
            body.push_str(nl);
            body.push_str(TOOL_CALL_OPEN);
            body.push_str(&name);
            if let serde_json::Value::Object(map) = arguments {
                for (key, value) in map {
                    let value = match value {
                        serde_json::Value::String(s) => s,
                        other => crate::llama::pythonic(&other),
                    };
                    body.push_str(&format!(
                        "{nl}<arg_key>{key}</arg_key>{nl}<arg_value>{value}</arg_value>"
                    ));
                }
            }
            body.push_str(nl);
            body.push_str(TOOL_CALL_CLOSE);
        }
        body
    }

    fn tool_system_prompt(&self, tools: &[String]) -> String {
        let mut prompt = String::from(
            "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        );
        for tool in tools {
            let line = serde_json::from_str::<serde_json::Value>(tool)
                .map(|v| crate::llama::pythonic(&v))
                .unwrap_or_else(|_| tool.clone());
            prompt.push_str(&line);
            prompt.push('\n');
        }
        prompt.push_str(
            "</tools>\n\nFor each function call, output the function name and arguments within \
             the following XML format:\n",
        );
        prompt.push_str(if self.style == Style::Glm45 {
            "<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>"
        } else {
            "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>"
        });
        prompt
    }
}

/// `msg`'s text and the calls written in it as ChatML's
/// `<tool_call>{"name", "arguments"}</tool_call>`.
fn split_calls(msg: &str) -> (String, Vec<(String, serde_json::Value)>) {
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut rest = msg;
    while let Some(at) = rest.find(TOOL_CALL_OPEN) {
        text.push_str(&rest[..at]);
        let after = &rest[at + TOOL_CALL_OPEN.len()..];
        let Some(end) = crate::chatml::closer_outside_string(after, TOOL_CALL_CLOSE) else {
            text.push_str(&rest[at..]);
            rest = "";
            break;
        };
        match serde_json::from_str::<serde_json::Value>(after[..end].trim()) {
            Ok(parsed) if parsed["name"].is_string() => calls.push((
                parsed["name"].as_str().unwrap_or_default().to_string(),
                parsed
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
            )),
            _ => text.push_str(&rest[at..at + TOOL_CALL_OPEN.len() + end + TOOL_CALL_CLOSE.len()]),
        }
        rest = &after[end + TOOL_CALL_CLOSE.len()..];
    }
    text.push_str(rest);
    (text, calls)
}

impl Instruct for Glm {
    /// What a raw prompt goes after: `[gMASK]<sop>`, and nothing for
    /// GLM-5.3, whose opening is a system turn (its effort line) a raw
    /// prompt would continue.
    fn prefix(&self) -> Vec<u32> {
        if self.style == Style::Glm5 {
            return Vec::new();
        }
        self.prefix.clone()
    }

    fn system(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.prefix.clone();
        tokens.extend(self.turn(self.system_marker, msg));
        tokens
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.prefix.clone();
        tokens.extend(self.user(msg));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(self.user_marker, msg)
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.user(user));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        if self.style == Style::Glm5 {
            return self.turn(self.assistant_marker, &Self::assistant_body(msg));
        }
        let mut tokens = vec![self.assistant_marker];
        tokens.extend(self.tokenizer.encode(&self.replay_body(msg)));
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        let mut tokens = vec![self.assistant_marker];
        if self.style != Style::Glm45 {
            tokens.extend(&self.think_open);
        }
        tokens
    }

    fn cue_without_thinking(&self) -> Vec<u32> {
        let mut tokens = vec![self.assistant_marker];
        match self.style {
            Style::Glm5 => tokens.extend(&self.think_open),
            Style::Glm47 => tokens.push(self.think_close),
            Style::Glm45 => {
                tokens.extend(self.tokenizer.encode("\n"));
                tokens.extend(&self.think_open);
                tokens.push(self.think_close);
            }
        }
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    /// The opening with the tools in a system turn of its own, as the
    /// template writes it ahead of every other turn.
    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if self.style == Style::Glm5 {
            return Vec::new();
        }
        let mut tokens = self.prefix.clone();
        tokens.push(self.system_marker);
        tokens.extend(self.tokenizer.encode(&self.tool_system_prompt(tools)));
        tokens
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        self.answers(&[(name.to_string(), value.to_string())])
    }

    /// Consecutive results under one observation marker.
    fn answers(&self, results: &[(String, String)]) -> Vec<u32> {
        if self.style == Style::Glm5 || results.is_empty() {
            return Vec::new();
        }
        let nl = self.gap();
        let mut tokens = vec![self.observation];
        for (_, value) in results {
            tokens.extend(self.tokenizer.encode(&format!(
                "{nl}<tool_response>{nl}{value}{nl}</tool_response>"
            )));
        }
        tokens
    }

    /// The opening: the tools' system turn, then the system message's, as
    /// the template writes them.
    fn open(&self, system: Option<&str>, tools: &[String], _effort: Option<Effort>) -> Vec<u32> {
        if self.style == Style::Glm5 || tools.is_empty() {
            return system.map(|msg| self.system(msg)).unwrap_or_default();
        }
        let mut tokens = self.equip(tools);
        if let Some(msg) = system {
            tokens.extend(self.turn(self.system_marker, msg));
        }
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        if self.style == Style::Glm45 {
            return Box::new(ThinkingDecoder::new(
                self.tokenizer.clone(),
                self.think_open.clone(),
                self.think_close,
            ));
        }
        Box::new(ThinkingDecoder::opened(
            self.tokenizer.clone(),
            self.think_open.clone(),
            self.think_close,
        ))
    }

    fn reasoning_decoder_for_cue(&self, cue: &[u32]) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder::for_cue(
            self.tokenizer.clone(),
            self.think_open.clone(),
            self.think_close,
            cue,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        if self.style == Style::Glm5 {
            return Box::new(NoopToolDecoder);
        }
        Box::new(GlmToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
        })
    }
}

/// One call's body, what sits between `<tool_call>` and `</tool_call>`, as
/// its name and its arguments' JSON; none where it does not read.
fn parse_call(call: &str) -> Option<(String, String)> {
    let (name, mut rest) = match call.find("<arg_key>") {
        Some(at) => (call[..at].trim(), &call[at..]),
        None => (call.trim(), ""),
    };
    if name.is_empty() {
        return None;
    }
    let mut arguments = serde_json::Map::new();
    while let Some(open) = rest.find("<arg_key>") {
        let after = &rest[open + "<arg_key>".len()..];
        let close = after.find("</arg_key>")?;
        let key = after[..close].trim().to_string();
        let after = &after[close + "</arg_key>".len()..];
        let vopen = after.find("<arg_value>")?;
        let after = &after[vopen + "<arg_value>".len()..];
        let vclose = after.find("</arg_value>")?;
        let raw = &after[..vclose];
        let value = serde_json::from_str::<serde_json::Value>(raw.trim())
            .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
        arguments.insert(key, value);
        rest = &after[vclose + "</arg_value>".len()..];
    }
    Some((
        name.to_string(),
        serde_json::Value::Object(arguments).to_string(),
    ))
}

/// GLM's calls: `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>
/// …</tool_call>` (GLM-4.5 with newlines between), each value JSON where it
/// parses as JSON and a string where it does not.
struct GlmToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
}

impl ToolDecoder for GlmToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);
        let mut events = Vec::new();
        loop {
            if self.inside {
                let Some(at) = self.accumulated.find(TOOL_CALL_CLOSE) else {
                    return events;
                };
                let call = self.accumulated[..at].to_string();
                self.accumulated = self.accumulated[at + TOOL_CALL_CLOSE.len()..].to_string();
                self.inside = false;
                if let Some((name, arguments)) = parse_call(&call) {
                    events.push(ToolEvent::Call(name, arguments));
                }
            } else {
                let Some(at) = self.accumulated.find(TOOL_CALL_OPEN) else {
                    return events;
                };
                self.accumulated = self.accumulated[at + TOOL_CALL_OPEN.len()..].to_string();
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
    fn glm_every_case() {
        a_call_reads_its_arguments_as_json_where_they_parse();
    }

    fn a_call_reads_its_arguments_as_json_where_they_parse() {
        let (name, arguments) = parse_call(
            "get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>\n<arg_value>2</arg_value>\n",
        )
        .expect("a call");
        assert_eq!(name, "get_weather");
        assert_eq!(arguments, r#"{"city":"Paris","days":2}"#);
    }
}
