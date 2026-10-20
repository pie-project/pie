//! MiMo-V2's template: ChatML turns closed by a bare `<|im_end|>` and
//! written as they are, every assistant turn opening a thinking block, the
//! tools a system turn of their own, and Qwen3-Coder's XML calls.

use std::sync::Arc;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::chatml::tool_names;
use crate::decode::{GenericChatDecoder, ThinkingDecoder};
use crate::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar, special, specials,
};

pub const STOP_TOKENS: &[&str] = &["<|im_end|>", "<|endoftext|>"];
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

pub struct Mimo {
    tokenizer: Arc<Tokenizer>,
    im_start: u32,
    im_end: u32,
    newline: Vec<u32>,
    think_open: u32,
    think_close: u32,
    stop_ids: Vec<u32>,
}

impl Mimo {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            im_start: special(&tokenizer, "<|im_start|>"),
            im_end: special(&tokenizer, "<|im_end|>"),
            newline: tokenizer.encode("\n"),
            think_open: special(&tokenizer, THINK_OPEN),
            think_close: special(&tokenizer, THINK_CLOSE),
            stop_ids: specials(&tokenizer, STOP_TOKENS),
            tokenizer,
        }
    }

    fn header(&self, role: &str) -> Vec<u32> {
        let mut tokens = vec![self.im_start];
        tokens.extend(self.tokenizer.encode(role));
        tokens.extend(&self.newline);
        tokens
    }

    fn turn(&self, role: &str, msg: &str) -> Vec<u32> {
        let mut tokens = self.header(role);
        tokens.extend(self.tokenizer.encode(msg));
        tokens.push(self.im_end);
        tokens
    }

    /// The template opens every assistant turn on its reasoning, an empty
    /// block when the turn kept none.
    fn assistant_body(msg: &str) -> String {
        if msg.starts_with(THINK_OPEN) {
            msg.to_string()
        } else {
            format!("{THINK_OPEN}{THINK_CLOSE}{msg}")
        }
    }

    fn tool_system_prompt(tools: &[String]) -> String {
        let mut prompt = String::from("You are provided with the following tools:\n\n<tools>");
        for tool in tools {
            prompt.push('\n');
            prompt.push_str(tool);
        }
        prompt.push_str("\n</tools>");
        prompt
    }
}

impl Instruct for Mimo {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.turn("system", msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn("user", msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn("assistant", &Self::assistant_body(msg))
    }

    fn cue(&self) -> Vec<u32> {
        self.header("assistant")
    }

    fn cue_without_thinking(&self) -> Vec<u32> {
        let mut tokens = self.header("assistant");
        tokens.extend([self.think_open, self.think_close]);
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        self.system(&Self::tool_system_prompt(tools))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.turn("tool", value)
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            vec![self.think_open],
            self.think_close,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(MimoToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        let alternatives = tool_names(tools)?;
        let source = format!(
            r#"root ::= tool-call+
tool-call ::= "<tool_call><function=" tool-name ">" parameter* "</function></tool_call>"
parameter ::= "<parameter=" parameter-name ">" parameter-value "</parameter>"
parameter-name ::= [A-Za-z0-9_.-]+
parameter-value ::= ([^<] | "<" [^/])*
tool-name ::= {alternatives}"#
        );
        Some(ToolGrammar { source })
    }
}

struct MimoToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
}

/// A parameter's text as JSON: itself when it is JSON (a number, an object),
/// else the string it spells.
fn parameter_value(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap_or_else(|_| serde_json::Value::String(text.to_string()))
}

/// `<function=NAME><parameter=K>V</parameter>…</function>` as the name and
/// its arguments' JSON object.
pub(crate) fn parse_call(span: &str) -> Option<(String, String)> {
    let rest = &span[span.find(FUNCTION_OPEN)? + FUNCTION_OPEN.len()..];
    let close = rest.find('>')?;
    let name = rest[..close].trim().to_string();
    let mut body = &rest[close + 1..];
    if let Some(end) = body.rfind(FUNCTION_CLOSE) {
        body = &body[..end];
    }
    let mut arguments = serde_json::Map::new();
    while let Some(at) = body.find(PARAMETER_OPEN) {
        let rest = &body[at + PARAMETER_OPEN.len()..];
        let close = rest.find('>')?;
        let key = rest[..close].trim().to_string();
        let value = &rest[close + 1..];
        let end = value.find(PARAMETER_CLOSE)?;
        let text = value[..end].strip_prefix('\n').unwrap_or(&value[..end]);
        let text = text.strip_suffix('\n').unwrap_or(text);
        arguments.insert(key, parameter_value(text));
        body = &value[end + PARAMETER_CLOSE.len()..];
    }
    Some((name, serde_json::Value::Object(arguments).to_string()))
}

impl ToolDecoder for MimoToolDecoder {
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
    fn mimo_every_case() {
        a_call_reads_its_parameters_as_json_where_they_are_json();
        a_call_without_parameters_has_empty_arguments();
        a_turn_without_reasoning_opens_an_empty_block();
    }

    fn a_call_reads_its_parameters_as_json_where_they_are_json() {
        let (name, arguments) = parse_call(
            "<function=get_weather><parameter=city>Paris, France</parameter>\
             <parameter=days>3</parameter><parameter=units>{\"t\": \"c\"}</parameter></function>",
        )
        .unwrap();
        assert_eq!(name, "get_weather");
        let arguments: serde_json::Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(
            arguments,
            serde_json::json!({"city": "Paris, France", "days": 3, "units": {"t": "c"}})
        );
    }

    fn a_call_without_parameters_has_empty_arguments() {
        assert_eq!(
            parse_call("<function=now></function>"),
            Some(("now".to_string(), "{}".to_string()))
        );
    }

    fn a_turn_without_reasoning_opens_an_empty_block() {
        assert_eq!(Mimo::assistant_body("four"), "<think></think>four");
        assert_eq!(
            Mimo::assistant_body("<think>x</think>four"),
            "<think>x</think>four"
        );
    }
}
