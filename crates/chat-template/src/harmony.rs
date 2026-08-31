//! Harmony: gpt-oss's channelled turn format.
//!
//! `<|start|>{role}<|message|>{text}<|end|>`, with the assistant's output split
//! across named channels — `analysis` for reasoning, `final` for the reply,
//! `commentary to=functions.{name}` for a tool call.

use std::sync::Arc;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::decode::{GenericChatDecoder, ThinkingDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, special, specials};

pub const STOP_TOKENS: &[&str] = &["<|endoftext|>", "<|return|>", "<|call|>"];

const CALL_HEADER: &str = "<|channel|>commentary to=functions.";
const MESSAGE: &str = "<|message|>";
const CALL_ENDS: &[&str] = &["<|call|>", "<|end|>", "<|return|>"];

pub struct Harmony {
    tokenizer: Arc<Tokenizer>,
    developer_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_final_prefix: Vec<u32>,
    end: u32,
    stop_ids: Vec<u32>,
    analysis_prefix: Vec<u32>,
    generation_prefix: Vec<u32>,
}

impl Harmony {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let stop_ids = specials(&tokenizer, STOP_TOKENS);

        let start = special(&tokenizer, "<|start|>");
        let message = special(&tokenizer, "<|message|>");
        let channel = special(&tokenizer, "<|channel|>");
        let end = special(&tokenizer, "<|end|>");

        let role_prefix = |role: &str| -> Vec<u32> {
            let mut tokens = vec![start];
            tokens.extend(tokenizer.encode(role));
            tokens.push(message);
            tokens
        };

        let channel_prefix = |name: &str| -> Vec<u32> {
            let mut tokens = vec![start];
            tokens.extend(tokenizer.encode("assistant"));
            tokens.push(channel);
            tokens.extend(tokenizer.encode(name));
            tokens.push(message);
            tokens
        };

        let mut analysis_prefix = vec![channel];
        analysis_prefix.extend(tokenizer.encode("analysis"));
        analysis_prefix.push(message);

        let mut generation_prefix = vec![start];
        generation_prefix.extend(tokenizer.encode("assistant"));

        Self {
            developer_prefix: role_prefix("developer"),
            user_prefix: role_prefix("user"),
            assistant_final_prefix: channel_prefix("final"),
            end,
            stop_ids,
            analysis_prefix,
            generation_prefix,
            tokenizer,
        }
    }

    fn wrap(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.push(self.end);
        tokens
    }
}

fn typescript_type(spec: &serde_json::Value) -> String {
    match spec.get("type").and_then(|kind| kind.as_str()) {
        Some("string") => {
            if let Some(choices) = spec.get("enum").and_then(|choices| choices.as_array()) {
                let parts: Vec<String> = choices
                    .iter()
                    .filter_map(|choice| choice.as_str())
                    .map(|choice| format!("\"{choice}\""))
                    .collect();
                parts.join(" | ")
            } else if spec
                .get("nullable")
                .and_then(|nullable| nullable.as_bool())
                .unwrap_or(false)
            {
                "string | null".to_string()
            } else {
                "string".to_string()
            }
        }
        Some("number" | "integer") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some("array") => match spec.get("items") {
            Some(items) => format!("{}[]", typescript_type(items)),
            None => "any[]".to_string(),
        },
        Some("object") => {
            if let Some(properties) = spec.get("properties").and_then(|it| it.as_object()) {
                let required: Vec<&str> = spec
                    .get("required")
                    .and_then(|it| it.as_array())
                    .map(|names| names.iter().filter_map(|name| name.as_str()).collect())
                    .unwrap_or_default();
                let mut lines = Vec::new();
                for (name, property) in properties {
                    let optional = if required.contains(&name.as_str()) {
                        ""
                    } else {
                        "?"
                    };
                    lines.push(format!(
                        "{}{}: {}",
                        name,
                        optional,
                        typescript_type(property)
                    ));
                }
                format!("{{\n{}\n}}", lines.join(",\n"))
            } else {
                "object".to_string()
            }
        }
        _ => {
            if let Some(one_of) = spec.get("oneOf").and_then(|it| it.as_array()) {
                let types: Vec<String> = one_of.iter().map(typescript_type).collect();
                types.join(" | ")
            } else {
                "any".to_string()
            }
        }
    }
}

impl Instruct for Harmony {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.developer_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.assistant_final_prefix, msg)
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }

        let mut prompt = String::from(" # Tools\n\n");
        prompt.push_str(" ## functions\n\n");
        prompt.push_str("namespace functions {\n\n");
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
                let function = parsed.get("function").unwrap_or(&parsed);
                let name = function
                    .get("name")
                    .and_then(|name| name.as_str())
                    .unwrap_or("unknown");
                let description = function
                    .get("description")
                    .and_then(|it| it.as_str())
                    .unwrap_or("");
                prompt.push_str(&format!("// {description}\n"));
                prompt.push_str(&format!("type {name} = "));
                if let Some(properties) = function
                    .get("parameters")
                    .and_then(|it| it.get("properties"))
                    .and_then(|it| it.as_object())
                {
                    let required: Vec<&str> = function
                        .get("parameters")
                        .and_then(|it| it.get("required"))
                        .and_then(|it| it.as_array())
                        .map(|names| names.iter().filter_map(|name| name.as_str()).collect())
                        .unwrap_or_default();
                    prompt.push_str("(_: {\n");
                    for (parameter, spec) in properties {
                        if let Some(description) =
                            spec.get("description").and_then(|it| it.as_str())
                        {
                            prompt.push_str(&format!("// {description}\n"));
                        }
                        let optional = if required.contains(&parameter.as_str()) {
                            ""
                        } else {
                            "?"
                        };
                        let rendered = typescript_type(spec);
                        prompt.push_str(&format!("{parameter}{optional}: {rendered},\n"));
                    }
                    prompt.push_str("}) => any;\n\n");
                } else {
                    prompt.push_str("() => any;\n\n");
                }
            }
        }
        prompt.push_str("} // namespace functions");
        self.wrap(&self.developer_prefix, &prompt)
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        let header =
            format!("<|start|>functions.{name} to=assistant<|channel|>commentary<|message|>");
        let json = serde_json::to_string(value).unwrap_or_else(|_| format!("\"{value}\""));
        let mut tokens = self.tokenizer.encode(&header);
        tokens.extend(self.tokenizer.encode(&json));
        tokens.push(self.end);
        tokens
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
            self.analysis_prefix.clone(),
            self.end,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(HarmonyToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            calling: None,
        })
    }
}

/// Reads the commentary channel's tool calls back out of generated text.
///
/// gpt-oss had a `NoopToolDecoder` here: this same module rendered a tool
/// namespace into the developer turn and then nothing read the calls it
/// produced. The span is the one the header above names —
/// `<|channel|>commentary to=functions.{name}<|message|>{json}<|call|>` — with
/// harmony's optional `<|constrain|>json` between the name and the message.
struct HarmonyToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    calling: Option<String>,
}

fn first_end(text: &str) -> Option<(usize, usize)> {
    CALL_ENDS
        .iter()
        .filter_map(|marker| text.find(marker).map(|at| (at, marker.len())))
        .min()
}

impl ToolDecoder for HarmonyToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        let mut events = Vec::new();
        loop {
            if let Some(name) = self.calling.clone() {
                let Some((at, len)) = first_end(&self.accumulated) else {
                    return events;
                };
                let arguments = self.accumulated[..at].trim().to_string();
                self.accumulated = self.accumulated[at + len..].to_string();
                self.calling = None;
                events.push(ToolEvent::Call(name, arguments));
            } else {
                let Some(at) = self.accumulated.find(CALL_HEADER) else {
                    return events;
                };
                let after = at + CALL_HEADER.len();
                let Some(head) = self.accumulated[after..].find(MESSAGE) else {
                    self.accumulated = self.accumulated[at..].to_string();
                    return events;
                };
                let declared = self.accumulated[after..after + head]
                    .split(|c: char| c.is_whitespace() || c == '<')
                    .next()
                    .unwrap_or_default()
                    .to_string();
                self.accumulated = self.accumulated[after + head + MESSAGE.len()..].to_string();
                if declared.is_empty() {
                    continue;
                }
                self.calling = Some(declared);
                events.push(ToolEvent::Start);
            }
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.calling = None;
    }
}
