use std::sync::Arc;

use serde_json::{Map, Value};
use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::decode::{GenericChatDecoder, ThinkingDecoder};
use crate::json::tojson;
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, special, specials};

/// MiniMax-M3's format (`chat_template.jinja` beside the checkpoint): a
/// prompt opens with `]~!b[`, then a system turn the template always writes
/// (the model's identity and its thinking instructions) and a developer turn
/// (the caller's system prompt, or a default, and the tools); every turn is
/// `]~b]role\n` … `[e~[\n`, the assistant's role is `ai`; an assistant turn
/// opens with its reasoning in `<mm:think>`…`</mm:think>`, or with a bare
/// `</mm:think>` when it has none; tool calls are XML whose every tag is led by
/// the `]<]minimax[>[` namespace token.
///
/// The template's thinking mode is left unset, as a caller that names none
/// renders it: the instructions say `adaptive` and the cue writes no prefix.
/// [`Instruct::cue_without_thinking`] writes the `disabled` cue's
/// `</mm:think>` under those instructions.
pub const STOP_TOKENS: &[&str] = &[EOS];

pub const MARKERS: &[&str] = &[BOD, BOS, EOS, NS, THINK_OPEN, THINK_CLOSE];

const BOD: &str = "]~!b[";
const BOS: &str = "]~b]";
const EOS: &str = "[e~[";
const NS: &str = "]<]minimax[>[";
const THINK_OPEN: &str = "<mm:think>";
const THINK_CLOSE: &str = "</mm:think>";

const IDENTITY: &str = "Your model version is MiniMax-M3, developed by MiniMax. Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is a global AI foundation model company committed to advancing the frontiers of AI towards AGI.";

const THINKING: &str = "\n\n<thinking_instructions>\n\
You have a thinking capability that allows you to reason step by step before responding. When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags before your response. When thinking is disabled, begin your response directly after the </mm:think> prefix. When thinking is adaptive, decide on your own whether to think for the current turn.\n\
Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing function/tool results.\n\
</thinking_instructions>";

const DEVELOPER: &str = "You are a helpful assistant.";

pub struct MiniMax {
    tokenizer: Arc<Tokenizer>,
    think_open: u32,
    think_close: u32,
    stop_ids: Vec<u32>,
}

impl MiniMax {
    pub fn try_new(tokenizer: Arc<Tokenizer>) -> Result<Self, String> {
        // Every marker is spelled into text, so only its presence is checked.
        let _ = specials(&tokenizer, MARKERS)?;
        Ok(Self {
            think_open: special(&tokenizer, THINK_OPEN)?,
            think_close: special(&tokenizer, THINK_CLOSE)?,
            stop_ids: specials(&tokenizer, STOP_TOKENS)?,
            tokenizer,
        })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }
}

fn turn(role: &str, body: &str) -> String {
    format!("{BOS}{role}\n{body}{EOS}\n")
}

/// The prompt's opening: the system turn, then the developer turn holding
/// `developer` (the default identity when empty).
fn opening(developer: &str) -> String {
    let developer = if developer.is_empty() {
        DEVELOPER
    } else {
        developer
    };
    format!(
        "{BOD}{}{}",
        turn("system", &format!("{IDENTITY}{THINKING}")),
        turn("developer", developer)
    )
}

fn tools_section(tools: &[String]) -> String {
    if tools.is_empty() {
        return String::new();
    }
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
        out.push_str(&tojson(&function));
        out.push_str("</tool>\n");
    }
    out.push_str("</tools>\n\n");
    out.push_str(&format!(
        "To call tools, wrap all invocations in a single {NS}<tool_call>{NS}</tool_call> block. \
         Parameter values containing nested objects or arrays are recursively expanded into XML \
         elements. Example:\n\n{NS}<tool_call>\n\
         {NS}<invoke name=\"tool-name-1\">{NS}<param-1>value-1{NS}</param-1>{NS}<param-2>\
         {NS}<item>{NS}<key-a>val-a{NS}</key-a>{NS}<key-b>val-b{NS}</key-b>{NS}</item>\
         {NS}</param-2>{NS}</invoke>\n\
         {NS}<invoke name=\"tool-name-2\">{NS}<param-1>value-1{NS}</param-1>{NS}</invoke>\n\
         {NS}</tool_call>"
    ));
    out
}

/// The template's `to_xml`: a mapping's non-null fields and a list's items as
/// namespaced elements, a scalar as its text.
fn to_xml(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if !value.is_null() {
                    out.push_str(&format!("{NS}<{key}>"));
                    to_xml(value, out);
                    out.push_str(&format!("{NS}</{key}>"));
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                out.push_str(&format!("{NS}<item>"));
                to_xml(item, out);
                out.push_str(&format!("{NS}</item>"));
            }
        }
        Value::Null => {}
        Value::String(text) => out.push_str(text),
        other => out.push_str(&other.to_string()),
    }
}

fn invoke(name: &str, arguments: &Value, out: &mut String) {
    out.push_str(&format!("{NS}<invoke name=\"{name}\">"));
    let arguments = match arguments {
        Value::String(text) => serde_json::from_str(text).unwrap_or(Value::Null),
        other => other.clone(),
    };
    if let Value::Object(map) = &arguments {
        for (key, value) in map {
            if !value.is_null() {
                out.push_str(&format!("{NS}<{key}>"));
                to_xml(value, out);
                out.push_str(&format!("{NS}</{key}>"));
            }
        }
    }
    out.push_str(&format!("{NS}</invoke>\n"));
}

/// A replayed turn's ChatML calls lifted out of its text
/// ([`crate::chatml::lift_calls`]); a turn already in the namespaced form is
/// kept whole.
fn lift_calls(msg: &str) -> (String, Vec<(String, Value)>) {
    if msg.contains(NS) {
        return (msg.to_string(), Vec::new());
    }
    crate::chatml::lift_calls(msg)
}

/// An assistant turn as the template writes one: its reasoning split off at
/// the last `</mm:think>`, then its text, then its calls.
fn assistant_body(msg: &str) -> String {
    let (content, calls) = lift_calls(msg);
    let (reasoning, content) = match content.find(THINK_CLOSE) {
        Some(first) => {
            let before = content[..first].trim_matches('\n');
            let reasoning = before
                .rsplit(THINK_OPEN)
                .next()
                .unwrap_or_default()
                .trim_matches('\n');
            let last = content.rfind(THINK_CLOSE).unwrap_or(first);
            let after = content[last + THINK_CLOSE.len()..].trim_matches('\n');
            (reasoning.to_string(), after.to_string())
        }
        None => (String::new(), content),
    };
    let mut out = if reasoning.is_empty() {
        THINK_CLOSE.to_string()
    } else {
        format!("{THINK_OPEN}{reasoning}{THINK_CLOSE}")
    };
    out.push_str(&content);
    if !calls.is_empty() {
        out.push_str(&format!("{NS}<tool_call>\n"));
        for (name, arguments) in &calls {
            invoke(name, arguments, &mut out);
        }
        out.push_str(&format!("{NS}</tool_call>"));
    }
    out
}

impl Instruct for MiniMax {
    fn prefix(&self) -> Vec<u32> {
        self.encode(BOD)
    }

    fn system(&self, msg: &str) -> Vec<u32> {
        self.encode(&opening(msg))
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.encode(&format!("{}{}", opening(DEVELOPER), turn("user", msg)))
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.encode(&turn("user", msg))
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        self.encode(&format!("{}{}", opening(system), turn("user", user)))
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.encode(&turn("ai", &assistant_body(msg)))
    }

    fn cue(&self) -> Vec<u32> {
        self.encode(&format!("{BOS}ai\n"))
    }

    fn cue_without_thinking(&self) -> Vec<u32> {
        self.encode(&format!("{BOS}ai\n{THINK_CLOSE}"))
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    /// The opening with the tools in its developer turn, as the template
    /// writes them for a conversation that states no system prompt.
    fn equip(&self, tools: &[String]) -> Vec<u32> {
        self.encode(&opening(&format!("{DEVELOPER}{}", tools_section(tools))))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.encode(&format!("{BOS}tool\n<response>{value}</response>{EOS}\n"))
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
        Box::new(MiniMaxToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
        })
    }
}

struct MiniMaxToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
}

/// `body`'s namespaced elements as `(tag, inner)` pairs, each closed at the
/// matching depth; `None` if it is not a run of elements.
fn elements(body: &str) -> Option<Vec<(String, &str)>> {
    let mut out = Vec::new();
    let mut rest = body.trim_matches('\n');
    while !rest.is_empty() {
        let after = rest.strip_prefix(NS)?.strip_prefix('<')?;
        let tag = &after[..after.find('>')?];
        if tag.starts_with('/') {
            return None;
        }
        let open = format!("{NS}<{tag}>");
        let close = format!("{NS}</{tag}>");
        let inner_at = NS.len() + 1 + tag.len() + 1;
        let mut depth = 1;
        let mut at = inner_at;
        let end = loop {
            let next_close = rest[at..].find(&close)? + at;
            match rest[at..].find(&open).map(|o| o + at) {
                Some(next_open) if next_open < next_close => {
                    depth += 1;
                    at = next_open + open.len();
                }
                _ => {
                    depth -= 1;
                    if depth == 0 {
                        break next_close;
                    }
                    at = next_close + close.len();
                }
            }
        };
        out.push((tag.to_string(), &rest[inner_at..end]));
        rest = rest[end + close.len()..].trim_matches('\n');
    }
    Some(out)
}

/// An argument read back from its XML: elements all named `item` are a list,
/// other elements an object, and text a number or boolean where it reads as
/// one and a string otherwise.
fn from_xml(body: &str) -> Value {
    if body.starts_with(NS)
        && let Some(children) = elements(body)
    {
        if children.iter().all(|(tag, _)| tag == "item") {
            return Value::Array(children.iter().map(|(_, inner)| from_xml(inner)).collect());
        }
        let mut map = Map::new();
        for (tag, inner) in children {
            map.insert(tag, from_xml(inner));
        }
        return Value::Object(map);
    }
    match serde_json::from_str::<Value>(body.trim()) {
        Ok(value @ (Value::Number(_) | Value::Bool(_))) => value,
        _ => Value::String(body.to_string()),
    }
}

fn parse_invoke(span: &str) -> Option<(String, String)> {
    const HEAD: &str = "<invoke name=\"";
    let start = span.find(HEAD)? + HEAD.len();
    let name_len = span[start..].find("\">")?;
    let name = span[start..start + name_len].to_string();
    let body = &span[start + name_len + 2..];
    let arguments = if body.trim().is_empty() {
        Value::Object(Map::new())
    } else {
        match from_xml(body) {
            object @ Value::Object(_) => object,
            _ => return None,
        }
    };
    Some((name, arguments.to_string()))
}

impl ToolDecoder for MiniMaxToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let open = format!("{NS}<tool_call>");
        let close = format!("{NS}</tool_call>");
        let end = format!("{NS}</invoke>");
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        let mut events = Vec::new();
        loop {
            if self.inside {
                let call = self.accumulated.find(&end);
                let closed = self.accumulated.find(&close);
                match (call, closed) {
                    (Some(at), closed) if closed.is_none_or(|c| at < c) => {
                        let span = self.accumulated[..at].to_string();
                        self.accumulated = self.accumulated[at + end.len()..].to_string();
                        if let Some((name, arguments)) = parse_invoke(&span) {
                            events.push(ToolEvent::Call(name, arguments));
                        }
                    }
                    (_, Some(c)) => {
                        self.accumulated = self.accumulated[c + close.len()..].to_string();
                        self.inside = false;
                    }
                    _ => return events,
                }
            } else {
                let Some(at) = self.accumulated.find(&open) else {
                    return events;
                };
                self.accumulated = self.accumulated[at + open.len()..].to_string();
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
    fn minimax_every_case() {
        a_turn_splits_its_reasoning_at_the_last_close();
        a_chatml_call_is_replayed_as_namespaced_xml();
        an_invoke_reads_back_to_its_arguments();
    }

    fn a_turn_splits_its_reasoning_at_the_last_close() {
        assert_eq!(
            assistant_body("<mm:think>\nok\n</mm:think>\n\nIt is sunny."),
            "<mm:think>ok</mm:think>It is sunny."
        );
        assert_eq!(assistant_body("four"), "</mm:think>four");
    }

    fn a_chatml_call_is_replayed_as_namespaced_xml() {
        let body = assistant_body(
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1, \"b\": [\"x\"], \"c\": null}}\n</tool_call>",
        );
        assert_eq!(
            body,
            format!(
                "</mm:think>{NS}<tool_call>\n{NS}<invoke name=\"f\">{NS}<a>1{NS}</a>{NS}<b>{NS}<item>x{NS}</item>{NS}</b>{NS}</invoke>\n{NS}</tool_call>"
            )
        );
    }

    fn an_invoke_reads_back_to_its_arguments() {
        let span = format!(
            "\n{NS}<invoke name=\"get_weather\">{NS}<city>Paris{NS}</city>{NS}<days>2{NS}</days>{NS}<opts>{NS}<units>{NS}<item>c{NS}</item>{NS}<item>f{NS}</item>{NS}</units>{NS}<hourly>true{NS}</hourly>{NS}</opts>"
        );
        let (name, arguments) = parse_invoke(&span).expect("the invoke parses");
        assert_eq!(name, "get_weather");
        let arguments: Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(
            arguments,
            serde_json::json!({"city": "Paris", "days": 2, "opts": {"units": ["c", "f"], "hourly": true}})
        );
    }
}
