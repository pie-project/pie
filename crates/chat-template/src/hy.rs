//! Tencent Hy4's turns (`chat_template.jinja` beside the checkpoint): every
//! turn is `<｜hy_start｜>role<｜hy_middle｜>`…`<｜hy_end｜>`, each marker
//! suffixed `:opensource`. The leading system turn ends with the reasoning
//! mode, and a conversation without one opens with the mode alone. An
//! assistant turn writes its thinking block before its text and its calls
//! after; the cue opens the thinking block, or writes it empty without
//! thinking.

use std::sync::Arc;
use tokenizer::Specials;

use serde_json::Value;
use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::chatml::{closer_outside_string, tool_names};
use crate::decode::{GenericChatDecoder, ThinkingDecoder};
use crate::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar, special, specials,
};

pub const STOP_TOKENS: &[&str] = &[END];

pub const MARKERS: &[&str] = &[
    START,
    MIDDLE,
    REASONING_MODE,
    THINK_OPEN,
    THINK_CLOSE,
    TOOL_CALLS_OPEN,
    TOOL_CALLS_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    ARG_KEY_OPEN,
    ARG_KEY_CLOSE,
    ARG_VALUE_OPEN,
    ARG_VALUE_CLOSE,
    TOOL_RESPONSE_OPEN,
    TOOL_RESPONSE_CLOSE,
];

const START: &str = "<｜hy_start:opensource｜>";
const MIDDLE: &str = "<｜hy_middle:opensource｜>";
const END: &str = "<｜hy_end:opensource｜>";
const REASONING_MODE: &str = "<｜reasoning_mode:opensource｜>";
const THINK_OPEN: &str = "<think:opensource>";
const THINK_CLOSE: &str = "</think:opensource>";
const TOOL_CALLS_OPEN: &str = "<tool_calls:opensource>";
const TOOL_CALLS_CLOSE: &str = "</tool_calls:opensource>";
const TOOL_CALL_OPEN: &str = "<tool_call:opensource>";
const TOOL_CALL_CLOSE: &str = "</tool_call:opensource>";
const ARG_KEY_OPEN: &str = "<arg_key:opensource>";
const ARG_KEY_CLOSE: &str = "</arg_key:opensource>";
const ARG_VALUE_OPEN: &str = "<arg_value:opensource>";
const ARG_VALUE_CLOSE: &str = "</arg_value:opensource>";
const TOOL_RESPONSE_OPEN: &str = "<tool_response:opensource>";
const TOOL_RESPONSE_CLOSE: &str = "</tool_response:opensource>";

/// The call syntax a replayed assistant turn may carry instead of Hy's own.
const CHATML_CALL_OPEN: &str = "<tool_call>";
const CHATML_CALL_CLOSE: &str = "</tool_call>";

pub struct Hy {
    tokenizer: Arc<Tokenizer>,
    thinking: bool,
    start: u32,
    middle: u32,
    end: u32,
    mode: Vec<u32>,
    think_open: u32,
    think_close: u32,
    tool_response_open: u32,
    tool_response_close: u32,
    stop_ids: Vec<u32>,
}

impl Hy {
    /// `thinking`: the reference template's `reasoning_effort: high`; without
    /// it, `no_think`, which every turn's thinking block is written empty in.
    pub fn try_new(tokenizer: Arc<Tokenizer>, thinking: bool) -> Result<Self, String> {
        let effort = if thinking { "high" } else { "no_think" };
        let mut mode = vec![special(&tokenizer, REASONING_MODE)?];
        mode.extend(tokenizer.encode(&format!("reasoning_effort:{effort}")));
        Ok(Self {
            thinking,
            start: special(&tokenizer, START)?,
            middle: special(&tokenizer, MIDDLE)?,
            end: special(&tokenizer, END)?,
            mode,
            think_open: special(&tokenizer, THINK_OPEN)?,
            think_close: special(&tokenizer, THINK_CLOSE)?,
            tool_response_open: special(&tokenizer, TOOL_RESPONSE_OPEN)?,
            tool_response_close: special(&tokenizer, TOOL_RESPONSE_CLOSE)?,
            stop_ids: specials(&tokenizer, STOP_TOKENS)?,
            tokenizer,
        })
    }

    fn header(&self, role: &str) -> Vec<u32> {
        let mut tokens = vec![self.start];
        tokens.extend(self.tokenizer.encode(role));
        tokens.push(self.middle);
        tokens
    }

    fn turn(&self, role: &str, body: &[u32]) -> Vec<u32> {
        let mut tokens = self.header(role);
        tokens.extend(body);
        tokens.push(self.end);
        tokens
    }

    /// The leading system turn: its text, then the reasoning mode.
    fn leading(&self, msg: &str) -> Vec<u32> {
        let mut body = self.tokenizer.encode(msg);
        body.extend(&self.mode);
        self.turn("system", &body)
    }

    /// A supplied thinking block is replayed as written, unless the template
    /// does not think; otherwise the block is written empty.
    fn assistant_body(&self, msg: &str) -> String {
        let msg = calls_written_as_hy(msg).unwrap_or_else(|| msg.to_string());
        if let Some(thought) = msg.strip_prefix(THINK_OPEN)
            && let Some(at) = thought.find(THINK_CLOSE)
        {
            if self.thinking {
                return msg;
            }
            return format!(
                "{THINK_OPEN}{THINK_CLOSE}{}",
                &thought[at + THINK_CLOSE.len()..]
            );
        }
        format!("{THINK_OPEN}{THINK_CLOSE}{msg}")
    }

    fn tool_system_prompt(tools: &[String]) -> String {
        let mut prompt = String::from(
            "# Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>",
        );
        for tool in tools {
            let Some(line) = tool_line(tool) else {
                continue;
            };
            prompt.push('\n');
            prompt.push_str(&line);
        }
        prompt.push_str(&format!(
            "\n</tools>\n\n\
             For function call returns, you should first print {TOOL_CALLS_OPEN}\n\
             For each function call, you should return object like:\n\
             {TOOL_CALL_OPEN}{{function-name}}\
             {ARG_KEY_OPEN}{{arg-key-1}}{ARG_KEY_CLOSE}{ARG_VALUE_OPEN}{{arg-value-1}}{ARG_VALUE_CLOSE}\
             {ARG_KEY_OPEN}{{arg-key-2}}{ARG_KEY_CLOSE}{ARG_VALUE_OPEN}{{arg-value-2}}{ARG_VALUE_CLOSE}\
             ...{TOOL_CALL_CLOSE}\n\
             At the end of function call returns, you should print {TOOL_CALLS_CLOSE}"
        ));
        prompt
    }
}

/// A tool as the template lists it: the function object, without its
/// `defer_loading` and `strict` keys; a deferred tool is not listed.
fn tool_line(tool: &str) -> Option<String> {
    let Ok(parsed) = serde_json::from_str::<Value>(tool) else {
        return Some(tool.to_string());
    };
    let function = parsed.get("function").unwrap_or(&parsed);
    let Some(fields) = function.as_object() else {
        return Some(python_json(function));
    };
    if fields.get("defer_loading").is_some_and(truthy) {
        return None;
    }
    let pairs: Vec<String> = fields
        .iter()
        .filter(|(key, _)| *key != "defer_loading" && *key != "strict")
        .map(|(key, value)| format!("\"{key}\": {}", python_json(value)))
        .collect();
    Some(format!("{{{}}}", pairs.join(", ")))
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// JSON as the template's `tojson` writes it: Python's `json.dumps`, keys
/// in their given order, `", "` and `": "` between items, text unescaped.
fn python_json(value: &Value) -> String {
    match value {
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(python_json).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(fields) => {
            let pairs: Vec<String> = fields
                .iter()
                .map(|(key, value)| {
                    format!("{}: {}", Value::from(key.as_str()), python_json(value))
                })
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
        scalar => scalar.to_string(),
    }
}

/// A call's argument as the template writes it: a string as itself, any
/// other value as its JSON.
fn argument(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python_json(other),
    }
}

/// A replayed turn whose calls are written `<tool_call>{"name": …,
/// "arguments": …}</tool_call>`, rewritten in Hy's call syntax; `None` if
/// the turn has no such calls or one does not parse.
fn calls_written_as_hy(msg: &str) -> Option<String> {
    let at = msg.find(CHATML_CALL_OPEN)?;
    let text = &msg[..at];
    let mut written = text.strip_suffix('\n').unwrap_or(text).to_string();
    written.push_str(TOOL_CALLS_OPEN);
    let mut rest = &msg[at..];
    loop {
        rest = rest.trim_start_matches('\n');
        if rest.is_empty() {
            break;
        }
        let body = rest.strip_prefix(CHATML_CALL_OPEN)?;
        let close = closer_outside_string(body, CHATML_CALL_CLOSE)?;
        let call: Value = serde_json::from_str(body[..close].trim()).ok()?;
        written.push_str(TOOL_CALL_OPEN);
        written.push_str(call.get("name")?.as_str()?);
        for (key, value) in call.get("arguments")?.as_object()? {
            written.push_str(&format!(
                "{ARG_KEY_OPEN}{key}{ARG_KEY_CLOSE}{ARG_VALUE_OPEN}{}{ARG_VALUE_CLOSE}",
                argument(value)
            ));
        }
        written.push_str(TOOL_CALL_CLOSE);
        rest = &body[close + CHATML_CALL_CLOSE.len()..];
    }
    written.push_str(TOOL_CALLS_CLOSE);
    Some(written)
}

impl Instruct for Hy {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.leading(msg)
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.turn("system", &self.mode);
        tokens.extend(self.user(msg));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn("user", &self.tokenizer.encode(msg))
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.user(user));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(
            "assistant",
            &self.tokenizer.encode(&self.assistant_body(msg)),
        )
    }

    fn cue(&self) -> Vec<u32> {
        let mut tokens = self.header("assistant");
        tokens.push(self.think_open);
        if !self.thinking {
            tokens.push(self.think_close);
        }
        tokens
    }

    fn cue_without_thinking(&self) -> Vec<u32> {
        let mut tokens = self.header("assistant");
        tokens.extend([self.think_open, self.think_close]);
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    /// The leading system turn the tools are listed in; a conversation that
    /// is equipped opens with it and goes on with [`Instruct::user`].
    fn equip(&self, tools: &[String]) -> Vec<u32> {
        self.leading(&Self::tool_system_prompt(tools))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        let mut body = vec![self.tool_response_open];
        body.extend(self.tokenizer.encode(value));
        body.push(self.tool_response_close);
        self.turn("tool", &body)
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        if self.thinking {
            return Box::new(ThinkingDecoder::opened(
                self.tokenizer.clone(),
                vec![self.think_open],
                self.think_close,
            ));
        }
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            vec![self.think_open],
            self.think_close,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(HyToolDecoder {
            decoder: self.tokenizer.decoder(Specials::Keep),
            accumulated: String::new(),
            inside: false,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        let alternatives = tool_names(tools)?;
        let arg_value = text_without("arg-value", ARG_VALUE_CLOSE);
        let source = format!(
            r#"root ::= "{TOOL_CALLS_OPEN}" tool-call+ "{TOOL_CALLS_CLOSE}"
tool-call ::= "{TOOL_CALL_OPEN}" tool-name tool-arg* "{TOOL_CALL_CLOSE}"
tool-arg ::= "{ARG_KEY_OPEN}" arg-key "{ARG_KEY_CLOSE}" "{ARG_VALUE_OPEN}" arg-value "{ARG_VALUE_CLOSE}"
tool-name ::= {alternatives}
arg-key ::= [^<>\n]+
{arg_value}"#
        );
        Some(ToolGrammar { source })
    }
}

/// The rule `name`: any text that does not hold `closer`, which opens with
/// the only `<` it holds. A value is free text — code, markup — and ends
/// only where its closer is written.
fn text_without(name: &str, closer: &str) -> String {
    let rest = closer
        .strip_prefix('<')
        .filter(|rest| {
            rest.len() > 1 && rest.is_ascii() && !rest.contains(['<', '"', '\\', ']', '^', '-'])
        })
        .expect("a closer opens with its only `<` and holds no grammar metacharacter");
    let prefixes: Vec<&str> = (1..rest.len()).map(|at| &rest[..at]).collect();
    let misses: Vec<String> = rest
        .char_indices()
        .map(|(at, c)| match at {
            0 => format!("[^{c}<]"),
            _ => format!("\"{}\" [^{c}<]", &rest[..at]),
        })
        .collect();
    let parts: Vec<String> = prefixes.iter().map(|p| format!("\"{p}\"")).collect();
    format!(
        "{name} ::= ([^<] | \"<\" {name}-again* {name}-miss)* (\"<\" {name}-again* {name}-part?)?\n\
         {name}-again ::= {name}-part? \"<\"\n\
         {name}-part ::= {}\n\
         {name}-miss ::= {}\n",
        parts.join(" | "),
        misses.join(" | ")
    )
}

struct HyToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
}

/// A call's span, `name<arg_key>k</arg_key><arg_value>v</arg_value>…`, as
/// its name and its arguments' JSON. A value that reads as JSON is that
/// value, as the template wrote it; any other value is a string.
fn parse_call(span: &str) -> Option<(String, String)> {
    let (name, mut rest) = span.split_once(ARG_KEY_OPEN).unwrap_or((span, ""));
    let mut arguments = serde_json::Map::new();
    while !rest.is_empty() {
        let (key, after) = rest.split_once(ARG_KEY_CLOSE)?;
        let after = after.trim_start().strip_prefix(ARG_VALUE_OPEN)?;
        let (value, after) = after.split_once(ARG_VALUE_CLOSE)?;
        let value = serde_json::from_str::<Value>(value)
            .ok()
            .filter(|parsed| !parsed.is_string())
            .unwrap_or_else(|| Value::from(value));
        arguments.insert(key.trim().to_string(), value);
        rest = match after.split_once(ARG_KEY_OPEN) {
            Some((_, next)) => next,
            None => "",
        };
    }
    Some((
        name.trim().to_string(),
        Value::Object(arguments).to_string(),
    ))
}

impl ToolDecoder for HyToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        let mut events = Vec::new();
        loop {
            if self.inside {
                let Some(at) = self.accumulated.find(TOOL_CALL_CLOSE) else {
                    return events;
                };
                let span = self.accumulated[..at].to_string();
                self.accumulated = self.accumulated[at + TOOL_CALL_CLOSE.len()..].to_string();
                self.inside = false;
                if let Some((name, arguments)) = parse_call(&span) {
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
    fn hy_every_case() {
        a_chatml_call_is_replayed_in_hy_syntax();
        a_call_reads_back_its_arguments();
        a_tool_is_listed_as_its_function();
    }

    fn a_chatml_call_is_replayed_in_hy_syntax() {
        let body = "Checking.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"days\":2,\"at\":[1,{\"x\":\"</tool_call>\"}]}}\n</tool_call>";
        assert_eq!(
            calls_written_as_hy(body).as_deref(),
            Some(
                "Checking.<tool_calls:opensource><tool_call:opensource>get_weather\
                 <arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris</arg_value:opensource>\
                 <arg_key:opensource>days</arg_key:opensource><arg_value:opensource>2</arg_value:opensource>\
                 <arg_key:opensource>at</arg_key:opensource><arg_value:opensource>[1, {\"x\": \"</tool_call>\"}]</arg_value:opensource>\
                 </tool_call:opensource></tool_calls:opensource>"
            )
        );
        assert_eq!(calls_written_as_hy("no calls"), None);
    }

    fn a_call_reads_back_its_arguments() {
        let span = "get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris</arg_value:opensource><arg_key:opensource>days</arg_key:opensource><arg_value:opensource>2</arg_value:opensource>";
        assert_eq!(
            parse_call(span),
            Some((
                "get_weather".to_string(),
                r#"{"city":"Paris","days":2}"#.to_string()
            ))
        );
        assert_eq!(
            parse_call("now"),
            Some(("now".to_string(), "{}".to_string()))
        );
    }

    fn a_tool_is_listed_as_its_function() {
        let tool = r#"{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object","properties":{}}}}"#;
        assert_eq!(
            tool_line(tool).as_deref(),
            Some(r#"{"name": "f", "parameters": {"type": "object", "properties": {}}}"#)
        );
        let deferred = r#"{"name":"g","defer_loading":true}"#;
        assert_eq!(tool_line(deferred), None);
    }
}
