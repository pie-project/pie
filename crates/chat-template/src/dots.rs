use std::sync::Arc;
use tokenizer::Specials;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::decode::{GenericChatDecoder, ThinkingDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, special, specials};

/// Dots3's template (`chat_template.jinja` beside the checkpoint): every turn
/// is its role's marker, the text, and the role's end marker, with no
/// newline between turns; a conversation without a system turn opens with
/// the default one. Tool calls are `<dots_function_call>` XML after the
/// answer, and tool results ride a user turn.
pub const STOP_TOKENS: &[&str] = &["<|endoftext|>", "<|endofassistant|>"];

const SYSTEM: &str = "<|system|>";
const END_OF_SYSTEM: &str = "<|endofsystem|>";
const USER: &str = "<|user|>";
const END_OF_USER: &str = "<|endofuser|>";
const ASSISTANT: &str = "<|assistant|>";
const END_OF_ASSISTANT: &str = "<|endofassistant|>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<dots_function_call>";
const CALL_CLOSE: &str = "</dots_function_call>";
const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

pub struct Dots {
    tokenizer: Arc<Tokenizer>,
    system: u32,
    end_of_system: u32,
    user: u32,
    end_of_user: u32,
    assistant: u32,
    end_of_assistant: u32,
    think_open: u32,
    think_close: u32,
    stop_ids: Vec<u32>,
}

impl Dots {
    pub fn try_new(tokenizer: Arc<Tokenizer>) -> Result<Self, String> {
        Ok(Self {
            system: special(&tokenizer, SYSTEM)?,
            end_of_system: special(&tokenizer, END_OF_SYSTEM)?,
            user: special(&tokenizer, USER)?,
            end_of_user: special(&tokenizer, END_OF_USER)?,
            assistant: special(&tokenizer, ASSISTANT)?,
            end_of_assistant: special(&tokenizer, END_OF_ASSISTANT)?,
            think_open: special(&tokenizer, THINK_OPEN)?,
            think_close: special(&tokenizer, THINK_CLOSE)?,
            stop_ids: specials(&tokenizer, STOP_TOKENS)?,
            tokenizer,
        })
    }

    fn turn(&self, open: u32, text: &str, close: u32) -> Vec<u32> {
        let mut tokens = vec![open];
        tokens.extend(self.tokenizer.encode(text));
        tokens.push(close);
        tokens
    }

    /// An assistant turn as the template replays it: its reasoning, if any,
    /// in a `<think>` block on lines of its own, then the answer.
    fn assistant_body(msg: &str) -> String {
        let Some((before, after)) = msg.split_once(THINK_CLOSE) else {
            return msg.to_string();
        };
        let reasoning = before.trim_end_matches('\n');
        let reasoning = reasoning
            .rsplit_once(THINK_OPEN)
            .map_or(reasoning, |(_, r)| r)
            .trim_start_matches('\n')
            .trim();
        let answer = after
            .rsplit_once(THINK_CLOSE)
            .map_or(after, |(_, a)| a)
            .trim_start_matches('\n');
        if reasoning.is_empty() {
            answer.to_string()
        } else {
            format!("{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n{answer}")
        }
    }

    fn tool_system_prompt(tools: &[String]) -> String {
        let mut prompt = format!(
            "{DEFAULT_SYSTEM}\n\n# Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>"
        );
        for tool in tools {
            prompt.push('\n');
            prompt.push_str(tool);
        }
        prompt.push_str(
            "\n</tools>\n\n\
             When making tool calls, use XML format to invoke tools and pass parameters:\n\n\
             <dots_function_call>\n\
             <invoke name=\"tool-name-1\">\n\
             <parameter name=\"param-key-1\">\nparam-value-1\n</parameter>\n\
             <parameter name=\"param-key-2\">\nparam-value-2\n</parameter>\n\
             ...\n\
             </invoke>\n\
             </dots_function_call>",
        );
        prompt
    }
}

impl Instruct for Dots {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.turn(self.system, msg, self.end_of_system)
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.system(DEFAULT_SYSTEM);
        tokens.extend(self.user(msg));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(self.user, msg, self.end_of_user)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(
            self.assistant,
            &Self::assistant_body(msg),
            self.end_of_assistant,
        )
    }

    fn cue(&self) -> Vec<u32> {
        vec![self.assistant]
    }

    /// The template's `enable_thinking: false` also writes `<no_think>` at
    /// the end of each user turn, which a turn already written cannot take;
    /// the empty block is what holds the answer to no reasoning.
    fn cue_without_thinking(&self) -> Vec<u32> {
        let gap = self.tokenizer.encode("\n\n");
        let mut tokens = vec![self.assistant, self.think_open];
        tokens.extend(&gap);
        tokens.push(self.think_close);
        tokens.extend(&gap);
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        self.system(&Self::tool_system_prompt(tools))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.turn(
            self.user,
            &format!("\n<dots_function_response>\n{value}\n</dots_function_response>"),
            self.end_of_user,
        )
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
        Box::new(DotsToolDecoder {
            decoder: self.tokenizer.decoder(Specials::Keep),
            accumulated: String::new(),
            inside: false,
        })
    }
}

struct DotsToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
}

/// The text between `open` and the next `close` in `text`, and the rest.
fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<(&'a str, &'a str)> {
    let (_, rest) = text.split_once(open)?;
    rest.split_once(close)
}

/// A call's `<invoke name="…">` and its `<parameter name="…">` values, the
/// newline the template writes around each value taken off. A value that
/// reads as JSON other than a string is that value; any other is a string.
fn parse_call(body: &str) -> Option<(String, String)> {
    let (name, mut rest) = between(body, "<invoke name=\"", "\">")?;
    let mut args = serde_json::Map::new();
    while let Some((key, after)) = between(rest, "<parameter name=\"", "\">") {
        let (value, after) = after.split_once("</parameter>")?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let parsed = match serde_json::from_str::<serde_json::Value>(value) {
            Ok(v) if !v.is_string() => v,
            _ => serde_json::Value::String(value.to_string()),
        };
        args.insert(key.to_string(), parsed);
        rest = after;
    }
    Some((
        name.to_string(),
        serde_json::Value::Object(args).to_string(),
    ))
}

impl ToolDecoder for DotsToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        let mut events = Vec::new();
        loop {
            if self.inside {
                let Some(at) = self.accumulated.find(CALL_CLOSE) else {
                    return events;
                };
                let body = self.accumulated[..at].to_string();
                self.accumulated = self.accumulated[at + CALL_CLOSE.len()..].to_string();
                self.inside = false;
                if let Some((name, args)) = parse_call(&body) {
                    events.push(ToolEvent::Call(name, args));
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
    fn dots_every_case() {
        a_turn_with_reasoning_replays_it_on_lines_of_its_own();
        a_turn_without_reasoning_replays_verbatim();
        a_call_reads_its_name_and_its_parameters();
    }

    fn a_turn_with_reasoning_replays_it_on_lines_of_its_own() {
        assert_eq!(
            Dots::assistant_body("<think>\nlet me count\n\n</think>\n\nfour"),
            "<think>\nlet me count\n</think>\n\nfour"
        );
        assert_eq!(Dots::assistant_body("<think>\n\n</think>\n\nfour"), "four");
    }

    fn a_turn_without_reasoning_replays_verbatim() {
        assert_eq!(Dots::assistant_body("four"), "four");
    }

    fn a_call_reads_its_name_and_its_parameters() {
        let body = "\n<invoke name=\"weather\">\n<parameter name=\"city\">\nParis\n</parameter>\n\
                    <parameter name=\"days\">\n3\n</parameter>\n</invoke>\n";
        assert_eq!(
            parse_call(body),
            Some((
                "weather".to_string(),
                r#"{"city":"Paris","days":3}"#.to_string()
            ))
        );
    }
}
