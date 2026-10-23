//! Llama 3.1 / 3.3's template: header-delimited turns closed by
//! `<|eot_id|>`, a system turn that always opens on the knowledge cutoff and
//! the date the reference template fixes, JSON tool calls
//! (`{"name": …, "parameters": …}`), and tool results in `ipython` turns.

use std::sync::Arc;
use tokenizer::Specials;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::chatml::{JSON_GRAMMAR, tool_names};
use crate::decode::{GenericChatDecoder, NoopReasoningDecoder};
use crate::json::tojson;
use crate::{
    ChatDecoder, Effort, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar, special,
    specials,
};

pub const STOP_TOKENS: &[&str] = &["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"];

/// What the reference template writes ahead of every system turn's text.
const DATES: &str = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n";

/// The reference template's tool preamble (`tools_in_user_message = false`).
const TOOLS: &str = "You have access to the following functions. To call a function, please respond with JSON for a function call.Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n";

pub struct Llama {
    tokenizer: Arc<Tokenizer>,
    bos: u32,
    start_header: u32,
    end_header: u32,
    eot: u32,
    python_tag: Option<u32>,
    stop_ids: Vec<u32>,
}

impl Llama {
    pub fn try_new(tokenizer: Arc<Tokenizer>) -> Result<Self, String> {
        Ok(Self {
            bos: special(&tokenizer, "<|begin_of_text|>")?,
            start_header: special(&tokenizer, "<|start_header_id|>")?,
            end_header: special(&tokenizer, "<|end_header_id|>")?,
            eot: special(&tokenizer, "<|eot_id|>")?,
            python_tag: tokenizer.token_to_id("<|python_tag|>"),
            stop_ids: specials(&tokenizer, STOP_TOKENS)?,
            tokenizer,
        })
    }

    fn header(&self, role: &str) -> Vec<u32> {
        let mut tokens = vec![self.start_header];
        tokens.extend(self.tokenizer.encode(role));
        tokens.push(self.end_header);
        tokens.extend(self.tokenizer.encode("\n\n"));
        tokens
    }

    fn turn(&self, role: &str, body: &str) -> Vec<u32> {
        let mut tokens = self.header(role);
        tokens.extend(self.tokenizer.encode(body));
        tokens.push(self.eot);
        tokens
    }

    /// The opening system turn: the beginning of text, then the turn the
    /// reference writes, its dates ahead of `body`.
    fn opening(&self, environment: bool, body: &str) -> Vec<u32> {
        let lead = if environment {
            "Environment: ipython\n"
        } else {
            ""
        };
        let mut tokens = vec![self.bos];
        tokens.extend(self.turn("system", &format!("{lead}{DATES}{}", body.trim())));
        tokens
    }

    /// An assistant turn's tool calls, replayed in ChatML's
    /// `<tool_call>{"name", "arguments"}</tool_call>` spelling, rewritten as
    /// the reference writes them: `{"name": …, "parameters": …}`.
    fn replay_body(msg: &str) -> String {
        let (text, calls) = crate::chatml::split_calls(msg);
        if calls.is_empty() {
            return text.trim().to_string();
        }
        let calls: Vec<String> = calls
            .iter()
            .map(|call| {
                let parameters = call
                    .get("arguments")
                    .or_else(|| call.get("parameters"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                format!(
                    "{{\"name\": \"{}\", \"parameters\": {}}}",
                    call["name"].as_str().unwrap_or_default(),
                    tojson(&parameters)
                )
            })
            .collect();
        calls.join("\n")
    }
}

impl Instruct for Llama {
    fn prefix(&self) -> Vec<u32> {
        vec![self.bos]
    }

    fn system(&self, msg: &str) -> Vec<u32> {
        self.opening(false, msg)
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.opening(false, "");
        tokens.extend(self.user(msg));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn("user", msg.trim())
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn("assistant", &Self::replay_body(msg))
    }

    fn cue(&self) -> Vec<u32> {
        self.header("assistant")
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    /// The tools open the conversation in a system turn of their own, as the
    /// reference writes it with no system message.
    fn equip(&self, tools: &[String]) -> Vec<u32> {
        self.open(None, tools, None)
    }

    /// One opening system turn: under tools, `Environment: ipython`, the
    /// dates, the tools and then the system message, as the reference writes
    /// them (`tools_in_user_message: false`).
    fn open(&self, system: Option<&str>, tools: &[String], _effort: Option<Effort>) -> Vec<u32> {
        if tools.is_empty() {
            return system.map(|msg| self.system(msg)).unwrap_or_default();
        }
        let mut body = String::from(TOOLS);
        for tool in tools {
            let pretty = serde_json::from_str::<serde_json::Value>(tool)
                .ok()
                .and_then(|v| indented(&v))
                .unwrap_or_else(|| tool.clone());
            body.push_str(&pretty);
            body.push_str("\n\n");
        }
        body.push_str(system.map(str::trim).unwrap_or_default());
        let mut tokens = vec![self.bos];
        tokens.extend(self.turn("system", &format!("Environment: ipython\n{DATES}{body}")));
        tokens
    }

    /// The reference writes a tool's result through `tojson`, a string
    /// result quoted like any other value.
    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.turn(
            "ipython",
            &tojson(&serde_json::Value::String(value.to_string())),
        )
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(LlamaToolDecoder {
            decoder: self.tokenizer.decoder(Specials::Keep),
            python_tag: self.python_tag,
            text: String::new(),
            started: false,
            closed: false,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if tools.is_empty() {
            return None;
        }
        let alternatives = tool_names(tools)?;
        let source = format!(
            r#"root ::= tool-call (";" tool-call)*
tool-call ::= "{{\"name\": \"" tool-name "\", \"parameters\": " json-object "}}"
tool-name ::= {alternatives}
{JSON_GRAMMAR}"#
        );
        Some(ToolGrammar { source })
    }
}

/// `value` as the reference's `tojson(indent=4)` writes it.
/// serde's pretty form indents two spaces a level and writes no raw newline
/// inside a string, so each line's indent doubles.
fn indented(value: &serde_json::Value) -> Option<String> {
    let pretty = serde_json::to_string_pretty(value).ok()?;
    let lines: Vec<String> = pretty
        .lines()
        .map(|line| {
            let body = line.trim_start_matches(' ');
            format!("{}{body}", " ".repeat(2 * (line.len() - body.len())))
        })
        .collect();
    Some(lines.join("\n"))
}

/// A Llama answer that calls a tool is the call itself: a JSON object (after
/// `<|python_tag|>`, where the model writes one) as the turn's first text,
/// several separated by `;`. Text that opens on anything else is no call.
struct LlamaToolDecoder {
    decoder: TokenizerDecoder,
    python_tag: Option<u32>,
    text: String,
    started: bool,
    closed: bool,
}

/// The end of the first whole JSON object `text` opens with, if it holds one.
fn object_end(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let bytes = text.as_bytes();
    let mut at = 0;
    while at < bytes.len() {
        match bytes[at] {
            b'\\' if in_string => at += 1,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(at + 1);
                }
            }
            _ => {}
        }
        at += 1;
    }
    None
}

impl ToolDecoder for LlamaToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        if self.closed {
            return Vec::new();
        }
        let kept: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|t| Some(*t) != self.python_tag)
            .collect();
        self.text.push_str(&self.decoder.feed(&kept));
        let mut events = Vec::new();
        loop {
            let lead = self.text.trim_start_matches([' ', '\n', ';']);
            if lead.is_empty() {
                return events;
            }
            if !lead.starts_with('{') {
                if !self.started {
                    self.closed = true;
                }
                return events;
            }
            if !self.started {
                self.started = true;
                events.push(ToolEvent::Start);
            }
            let Some(end) = object_end(lead) else {
                return events;
            };
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&lead[..end])
                && let Some(name) = parsed["name"].as_str()
            {
                let parameters = parsed
                    .get("parameters")
                    .or_else(|| parsed.get("arguments"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                events.push(ToolEvent::Call(name.to_string(), parameters.to_string()));
            }
            self.text = lead[end..].to_string();
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.text.clear();
        self.started = false;
        self.closed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_every_case() {
        a_replayed_call_is_written_with_its_parameters();
        an_object_ends_outside_its_strings();
    }

    fn a_replayed_call_is_written_with_its_parameters() {
        let body = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        assert_eq!(
            Llama::replay_body(body),
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}"
        );
        assert_eq!(Llama::replay_body(" four \n"), "four");
    }

    fn an_object_ends_outside_its_strings() {
        assert_eq!(object_end("{\"a\": \"}\"} tail"), Some(10));
        assert_eq!(object_end("{\"a\": {"), None);
    }
}
