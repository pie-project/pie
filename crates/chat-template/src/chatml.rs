use std::borrow::Cow;
use std::sync::Arc;
use tokenizer::Specials;

use tokenizer::{Tokenizer, TokenizerDecoder};

use crate::decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder, ThinkingDecoder};
use crate::json::{retojson, tojson};
use crate::{
    ChatDecoder, Effort, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar, special,
    specials,
};

#[derive(Clone, Debug)]
pub struct ChatML {
    pub thinking: bool,
    pub preserve_thinking: bool,
    /// The template opens a thinking request's system turn with its
    /// reasoning-effort instruction (Qwen3.8).
    pub reasoning_effort: bool,
    pub tools: bool,
    /// The template describes tools and reads calls in Qwen3-Coder's XML
    /// (`<function=NAME><parameter=K>V</parameter></function>`), as Qwen 3.5
    /// and later write them; otherwise in Hermes JSON.
    pub xml_tools: bool,
    pub generation_suffix: String,
    pub stop_tokens: Vec<String>,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

pub struct ChatMLInstruct {
    tokenizer: Arc<Tokenizer>,
    config: ChatML,

    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    quiet_header: Vec<u32>,
    stop_ids: Vec<u32>,

    think_open: Vec<u32>,
    think_close: u32,

    tool_response_prefix: Vec<u32>,
    tool_response_suffix: Vec<u32>,
}

impl ChatMLInstruct {
    pub fn try_new(tokenizer: Arc<Tokenizer>, config: ChatML) -> Result<Self, String> {
        let stop_ids = specials(&tokenizer, &config.stop_tokens)?;

        let im_start = special(&tokenizer, "<|im_start|>")?;
        let im_end = special(&tokenizer, "<|im_end|>")?;
        let newline = tokenizer.encode("\n");

        let header = |role: &str| -> Vec<u32> {
            let mut tokens = vec![im_start];
            tokens.extend(tokenizer.encode(role));
            tokens.extend(&newline);
            tokens
        };

        let mut turn_suffix = vec![im_end];
        turn_suffix.extend(&newline);

        let mut tool_response_prefix = tokenizer.encode("<tool_response>");
        tool_response_prefix.extend(&newline);
        let mut tool_response_suffix = newline.clone();
        tool_response_suffix.extend(tokenizer.encode("</tool_response>"));

        let mut generation_header = header("assistant");
        generation_header.extend(tokenizer.encode(&config.generation_suffix));
        // An empty thinking block, as the reference template writes for
        // `enable_thinking: false`.
        let mut quiet_header = header("assistant");
        if config.thinking {
            let gap = tokenizer.encode("\n\n");
            quiet_header.push(special(&tokenizer, THINK_OPEN)?);
            quiet_header.extend(&gap);
            quiet_header.push(special(&tokenizer, THINK_CLOSE)?);
            quiet_header.extend(&gap);
        }

        Ok(Self {
            system_prefix: header("system"),
            user_prefix: header("user"),
            assistant_prefix: header("assistant"),
            generation_header,
            quiet_header,
            turn_suffix,
            stop_ids,
            think_open: vec![special(&tokenizer, THINK_OPEN)?],
            think_close: special(&tokenizer, THINK_CLOSE)?,
            tool_response_prefix,
            tool_response_suffix,
            tokenizer,
            config,
        })
    }

    fn turn(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        self.verbatim(prefix, msg.trim())
    }

    fn verbatim(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    /// The reasoning-effort instruction a request with `effort` opens with.
    fn instruction(&self, effort: Option<Effort>) -> Option<&'static str> {
        if !(self.config.thinking && self.config.reasoning_effort) {
            return None;
        }
        match effort? {
            Effort::Xhigh => Some(EFFORT_XHIGH),
            Effort::Low => Some(EFFORT_LOW),
            Effort::Medium => None,
        }
    }

    /// The system turn without tools. Qwen3.8 writes no turn for an empty
    /// system message; the templates before it write the empty turn.
    fn plain_system(&self, msg: Option<&str>, effort: Option<Effort>) -> Vec<u32> {
        let content = msg.map(str::trim);
        let skip_empty = self.config.reasoning_effort;
        match (self.instruction(effort), content) {
            (Some(i), Some(c)) if !c.is_empty() => {
                self.turn(&self.system_prefix, &format!("{i}\n\n{c}"))
            }
            (Some(i), _) => self.turn(&self.system_prefix, i),
            (None, Some(c)) if !(skip_empty && c.is_empty()) => self.system(c),
            (None, _) => Vec::new(),
        }
    }

    /// An assistant turn's reasoning and answer, split where its thinking
    /// block closes.
    fn split_reasoning(msg: &str) -> (&str, &str) {
        match msg.rfind(THINK_CLOSE) {
            Some(at) => {
                let head = &msg[..at];
                let reasoning = head
                    .rfind(THINK_OPEN)
                    .map_or(head, |open| &head[open + THINK_OPEN.len()..]);
                (reasoning.trim(), msg[at + THINK_CLOSE.len()..].trim())
            }
            None => ("", msg.trim()),
        }
    }

    fn without_thinking(msg: &str) -> &str {
        match msg.rfind(THINK_CLOSE) {
            Some(at) => msg[at + THINK_CLOSE.len()..].trim_start_matches('\n'),
            None => msg,
        }
    }

    /// A template that preserves thinking (Qwen3.8) writes every earlier
    /// assistant turn as `<think>\n{reasoning}\n</think>\n\n{answer}`, an empty
    /// block where the turn had no reasoning; one that strips it replays the
    /// answer alone.
    fn replay_body<'a>(config: &ChatML, msg: &'a str) -> Cow<'a, str> {
        if !config.thinking {
            return Cow::Borrowed(msg);
        }
        if !config.preserve_thinking {
            return Cow::Borrowed(Self::without_thinking(msg));
        }
        let (reasoning, answer) = Self::split_reasoning(msg);
        Cow::Owned(format!(
            "{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n{answer}"
        ))
    }

    /// The template's tool description, each tool written as `tojson`
    /// writes it.
    fn tools_block(&self, tools: &[String]) -> String {
        let (head, tail) = if self.config.xml_tools {
            (XML_TOOLS_HEAD, XML_TOOLS_TAIL)
        } else {
            (HERMES_TOOLS_HEAD, HERMES_TOOLS_TAIL)
        };
        let mut block = String::from(head);
        for tool in tools {
            block.push('\n');
            block.push_str(&retojson(tool));
        }
        block.push_str(tail);
        block
    }

    /// One call in the template's syntax.
    fn call_text(&self, name: &str, arguments: &str) -> String {
        if !self.config.xml_tools {
            return format!(
                "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {arguments}}}\n</tool_call>"
            );
        }
        let mut text = format!("<tool_call>\n<function={name}>\n");
        if let Ok(serde_json::Value::Object(arguments)) = serde_json::from_str(arguments) {
            for (key, value) in arguments {
                let value = match value {
                    serde_json::Value::String(s) => s,
                    other => tojson(&other),
                };
                text.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
            }
        }
        text.push_str("</function>\n</tool_call>");
        text
    }
}

const HERMES_TOOLS_HEAD: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const HERMES_TOOLS_TAIL: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
/// Qwen 3.5–3.8's `chat_template.jinja`, word for word.
const XML_TOOLS_HEAD: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const XML_TOOLS_TAIL: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

pub fn tool_names(tools: &[String]) -> Option<String> {
    let mut names: Vec<String> = Vec::new();
    for tool in tools {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
            let name = parsed
                .get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| parsed.get("name"))
                .and_then(|name| name.as_str());
            if let Some(name) = name {
                names.push(format!("\"{name}\""));
            }
        }
    }
    if names.is_empty() {
        return None;
    }
    Some(names.join(" | "))
}

/// The JSON value production every tool grammar in this crate ends with.
///
/// `json-char` bars U+0000-U+001F, as RFC 8259 §7 does: a raw control
/// character the grammar let through would be a call `serde_json` refuses.
/// `ws` sits where RFC 8259 §2 allows insignificant whitespace — around the
/// structural characters and nowhere else — so a model that writes `: ` or a
/// newline between tokens keeps a legal token rather than running to budget.
/// A run of it is bounded (an indented newline fits), so a model the grammar
/// holds cannot spend its budget on whitespace either.
pub const JSON_GRAMMAR: &str = r#"json-object ::= "{" ws (json-members ws)? "}"
json-members ::= json-pair (ws "," ws json-pair)*
json-pair ::= json-string ws ":" ws json-value
json-value ::= json-string | json-number | json-object | json-array | "true" | "false" | "null"
json-string ::= "\"" json-chars "\""
json-chars ::= json-char*
json-char ::= [^"\\\x00-\x1f] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-array ::= "[" ws (json-value (ws "," ws json-value)* ws)? "]"
ws ::= [ \t\n\r]{0,20}
"#;

/// Qwen3.8's `chat_template.jinja`, word for word; `medium` writes none.
const EFFORT_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const EFFORT_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

impl Instruct for ChatMLInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.system_prefix, msg)
    }

    fn system_thinking(&self, msg: Option<&str>, effort: Effort) -> Vec<u32> {
        self.plain_system(msg, Some(effort))
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(
            &self.assistant_prefix,
            &Self::replay_body(&self.config, msg),
        )
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_header.clone()
    }

    fn cue_without_thinking(&self) -> Vec<u32> {
        self.quiet_header.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if !self.config.tools || tools.is_empty() {
            return Vec::new();
        }
        self.verbatim(&self.system_prefix, &self.tools_block(tools))
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        self.answers(&[(name.to_string(), value.to_string())])
    }

    /// The template writes consecutive results in one user turn.
    fn answers(&self, results: &[(String, String)]) -> Vec<u32> {
        if !self.config.tools || results.is_empty() {
            return Vec::new();
        }
        let mut tokens = self.user_prefix.clone();
        for (i, (_, value)) in results.iter().enumerate() {
            if i > 0 {
                tokens.extend(self.tokenizer.encode("\n"));
            }
            let value = if self.config.xml_tools {
                value.trim()
            } else {
                value
            };
            tokens.extend(&self.tool_response_prefix);
            tokens.extend(self.tokenizer.encode(value));
            tokens.extend(&self.tool_response_suffix);
        }
        tokens.extend(&self.turn_suffix);
        tokens
    }

    /// The template writes the tools into the system turn, around the
    /// system message: after it (Hermes) or ahead of it, behind the
    /// reasoning-effort instruction (Qwen 3.5 and later).
    fn open(&self, system: Option<&str>, tools: &[String], effort: Option<Effort>) -> Vec<u32> {
        if !self.config.tools || tools.is_empty() {
            return self.plain_system(system, effort);
        }
        let block = self.tools_block(tools);
        let text = if self.config.xml_tools {
            let mut text = String::new();
            if let Some(instruction) = self.instruction(effort) {
                text.push_str(instruction);
                text.push_str("\n\n");
            }
            text.push_str(&block);
            if let Some(content) = system.map(str::trim).filter(|c| !c.is_empty()) {
                text.push_str("\n\n");
                text.push_str(content);
            }
            text
        } else {
            match system {
                Some(content) => format!("{content}\n\n{block}"),
                None => block,
            }
        };
        self.verbatim(&self.system_prefix, &text)
    }

    /// Qwen 3.5 and later replay a turn's reasoning where thinking is
    /// preserved or the turn follows the last user query; Hermes Qwen3 only
    /// for the latter, and only when there is reasoning to replay.
    fn assistant_calls(&self, msg: &str, calls: &[(String, String)], current: bool) -> Vec<u32> {
        let (reasoning, answer) = Self::split_reasoning(msg);
        let keep = self.config.thinking
            && if self.config.xml_tools {
                self.config.preserve_thinking || current
            } else {
                self.config.preserve_thinking || (current && !reasoning.is_empty())
            };
        let mut body = if keep {
            format!("{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n{answer}")
        } else if self.config.thinking {
            answer.to_string()
        } else {
            msg.trim().to_string()
        };
        for (i, (name, arguments)) in calls.iter().enumerate() {
            let gap = match (self.config.xml_tools, i, answer.is_empty()) {
                (true, 0, false) => "\n\n",
                (_, 0, true) => "",
                _ => "\n",
            };
            body.push_str(gap);
            body.push_str(&self.call_text(name, arguments));
        }
        self.verbatim(&self.assistant_prefix, &body)
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        if !self.config.thinking {
            return Box::new(NoopReasoningDecoder);
        }
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.think_open.clone(),
            self.think_close,
        ))
    }

    fn reasoning_decoder_for_cue(&self, cue: &[u32]) -> Box<dyn ReasoningDecoder> {
        if !self.config.thinking {
            return Box::new(NoopReasoningDecoder);
        }
        Box::new(ThinkingDecoder::for_cue(
            self.tokenizer.clone(),
            self.think_open.clone(),
            self.think_close,
            cue,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        if !self.config.tools {
            return Box::new(NoopToolDecoder);
        }
        Box::new(ChatMLToolDecoder {
            decoder: self.tokenizer.decoder(Specials::Keep),
            accumulated: String::new(),
            inside: false,
            xml: self.config.xml_tools,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if !self.config.tools || tools.is_empty() {
            return None;
        }
        let alternatives = tool_names(tools)?;
        if self.config.xml_tools {
            let source = format!(
                r#"root ::= tool-call ("\n" tool-call)*
tool-call ::= "<tool_call>\n<function=" tool-name ">\n" parameter* "</function>\n</tool_call>"
parameter ::= "<parameter=" parameter-name ">\n" parameter-value "\n</parameter>\n"
parameter-name ::= [A-Za-z0-9_.-]+
parameter-value ::= ([^<] | "<" [^/])*
tool-name ::= {alternatives}"#
            );
            return Some(ToolGrammar { source });
        }
        let source = format!(
            r#"root ::= tool-call ("\n" tool-call)*
tool-call ::= "<tool_call>\n" tool-json "\n</tool_call>"
tool-json ::= "{{"  "\"name\": \"" tool-name "\", \"arguments\": " json-object "}}"
tool-name ::= {alternatives}
{JSON_GRAMMAR}"#
        );
        Some(ToolGrammar { source })
    }
}

struct ChatMLToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
    xml: bool,
}

/// The first `closer` that is not inside a JSON string. Measured on the 27B:
/// an argument string held `</tool_call>` verbatim, and a search blind to
/// string context cut the call in half and dropped both halves.
pub(crate) fn closer_outside_string(text: &str, closer: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let closer = closer.as_bytes();
    let mut at = 0;
    let mut in_string = false;
    while at < bytes.len() {
        match bytes[at] {
            b'\\' if in_string => at += 1,
            b'"' => in_string = !in_string,
            _ if !in_string && bytes[at..].starts_with(closer) => return Some(at),
            _ => {}
        }
        at += 1;
    }
    None
}

/// `msg`'s text and the calls written in it as ChatML's
/// `<tool_call>{"name", "arguments"}</tool_call>`, each the call's JSON; a
/// call whose JSON does not read or names nothing stays in the text.
pub(crate) fn split_calls(msg: &str) -> (String, Vec<serde_json::Value>) {
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut rest = msg;
    while let Some(at) = rest.find(TOOL_CALL_OPEN) {
        text.push_str(&rest[..at]);
        let after = &rest[at + TOOL_CALL_OPEN.len()..];
        let Some(end) = closer_outside_string(after, TOOL_CALL_CLOSE) else {
            text.push_str(&rest[at..]);
            rest = "";
            break;
        };
        match serde_json::from_str::<serde_json::Value>(after[..end].trim()) {
            Ok(call) if call["name"].is_string() => calls.push(call),
            _ => text.push_str(&rest[at..at + TOOL_CALL_OPEN.len() + end + TOOL_CALL_CLOSE.len()]),
        }
        rest = &after[end + TOOL_CALL_CLOSE.len()..];
    }
    text.push_str(rest);
    (text, calls)
}

/// [`split_calls`] as each call's name and arguments (an empty object for
/// none), the text's trailing newlines dropped ahead of the calls.
pub(crate) fn lift_calls(msg: &str) -> (String, Vec<(String, serde_json::Value)>) {
    let (mut text, calls) = split_calls(msg);
    let calls: Vec<_> = calls
        .into_iter()
        .map(|mut call| {
            let arguments = call
                .get_mut("arguments")
                .map(serde_json::Value::take)
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
            (
                call["name"].as_str().unwrap_or_default().to_string(),
                arguments,
            )
        })
        .collect();
    if !calls.is_empty() {
        text.truncate(text.trim_end_matches('\n').len());
    }
    (text, calls)
}

impl ToolDecoder for ChatMLToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> Vec<ToolEvent> {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        let mut events = Vec::new();
        loop {
            if self.inside {
                // An XML call's values are bare text, where a quote opens no string.
                let close = if self.xml {
                    self.accumulated.find(TOOL_CALL_CLOSE)
                } else {
                    closer_outside_string(&self.accumulated, TOOL_CALL_CLOSE)
                };
                let Some(at) = close else {
                    return events;
                };
                let call = self.accumulated[..at].trim().to_string();
                self.accumulated = self.accumulated[at + TOOL_CALL_CLOSE.len()..].to_string();
                self.inside = false;
                if self.xml {
                    if let Some((name, arguments)) = crate::mimo::parse_call(&call) {
                        events.push(ToolEvent::Call(name, arguments));
                    }
                } else if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&call)
                    && let Some(name) = parsed["name"].as_str()
                {
                    events.push(ToolEvent::Call(
                        name.to_string(),
                        parsed["arguments"].to_string(),
                    ));
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

    const TURN: &str = "<think>\nlet me count\n</think>\n\nfour";

    fn config(thinking: bool, preserve_thinking: bool) -> ChatML {
        ChatML {
            thinking,
            preserve_thinking,
            reasoning_effort: false,
            tools: true,
            xml_tools: false,
            generation_suffix: String::new(),
            stop_tokens: vec!["<|im_end|>".to_string()],
        }
    }

    #[test]
    fn chatml_every_case() {
        a_stripping_model_replays_the_answer_without_its_reasoning();
        a_preserving_model_replays_the_turn_whole();
        a_non_thinking_model_replays_verbatim();
        a_call_is_lifted_whole_past_a_closer_inside_its_arguments();
    }

    fn a_call_is_lifted_whole_past_a_closer_inside_its_arguments() {
        let (text, calls) = lift_calls(
            "Saving.\n<tool_call>\n{\"name\": \"save\", \"arguments\": {\"text\": \"a </tool_call> b\"}}\n</tool_call>",
        );
        assert_eq!(text, "Saving.");
        assert_eq!(
            calls,
            vec![(
                "save".to_string(),
                serde_json::json!({"text": "a </tool_call> b"})
            )]
        );
    }

    fn a_stripping_model_replays_the_answer_without_its_reasoning() {
        assert_eq!(
            ChatMLInstruct::replay_body(&config(true, false), TURN),
            "four"
        );
    }

    fn a_preserving_model_replays_the_turn_whole() {
        assert_eq!(ChatMLInstruct::replay_body(&config(true, true), TURN), TURN);
        assert_eq!(
            ChatMLInstruct::replay_body(&config(true, true), "four"),
            "<think>\n\n</think>\n\nfour"
        );
    }

    fn a_non_thinking_model_replays_verbatim() {
        assert_eq!(
            ChatMLInstruct::replay_body(&config(false, false), TURN),
            TURN
        );
    }
}
