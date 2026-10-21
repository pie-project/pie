//! Qwen3-Coder's template: ChatML turns written as they are (no thinking
//! block), the tools a system turn of their own opening on the coder's
//! identity and listed as XML, calls as `<tool_call><function=NAME>
//! <parameter=KEY>value</parameter></function></tool_call>`, results as
//! `<tool_response>`s in a user turn.

use std::sync::Arc;

use serde_json::Value;
use tokenizer::Tokenizer;

use crate::decode::{GenericChatDecoder, NoopReasoningDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, special, specials};

pub const STOP_TOKENS: &[&str] = &["<|im_end|>", "<|endoftext|>"];

const IDENTITY: &str =
    "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.";

const FORMAT: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

pub struct Qwen3Coder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
}

impl Qwen3Coder {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let _ = special(&tokenizer, "<|im_start|>");
        Self {
            stop_ids: specials(&tokenizer, STOP_TOKENS),
            tokenizer,
        }
    }

    fn turn(&self, role: &str, body: &str) -> Vec<u32> {
        self.tokenizer
            .encode(&format!("<|im_start|>{role}\n{body}<|im_end|>\n"))
    }
}

/// A value as the template's `string` filter writes it: Python's `str`.
fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        other => crate::llama::pythonic(other),
    }
}

/// `render_extra_keys`: each key not handled, a mapping or list as JSON and
/// anything else as its string.
fn extra_keys(map: &serde_json::Map<String, Value>, handled: &[&str], out: &mut String) {
    for (key, value) in map {
        if handled.contains(&key.as_str()) {
            continue;
        }
        let text = match value {
            Value::Object(_) | Value::Array(_) => crate::llama::pythonic(value),
            other => python_str(other),
        };
        out.push_str(&format!("\n<{key}>{text}</{key}>"));
    }
}

fn tools_section(tools: &[String]) -> String {
    let mut out =
        String::from("\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        let Ok(Value::Object(mut tool)) = serde_json::from_str::<Value>(tool) else {
            continue;
        };
        if let Some(Value::Object(function)) = tool.remove("function") {
            tool = function;
        }
        let name = tool.get("name").map(python_str).unwrap_or_default();
        out.push_str(&format!("\n<function>\n<name>{name}</name>"));
        if let Some(description) = tool.get("description") {
            out.push_str(&format!(
                "\n<description>{}</description>",
                python_str(description).trim()
            ));
        }
        out.push_str("\n<parameters>");
        if let Some(Value::Object(parameters)) = tool.get("parameters") {
            if let Some(Value::Object(properties)) = parameters.get("properties") {
                for (param, fields) in properties {
                    out.push_str(&format!("\n<parameter>\n<name>{param}</name>"));
                    if let Value::Object(fields) = fields {
                        if let Some(kind) = fields.get("type") {
                            out.push_str(&format!("\n<type>{}</type>", python_str(kind)));
                        }
                        if let Some(description) = fields.get("description") {
                            out.push_str(&format!(
                                "\n<description>{}</description>",
                                python_str(description).trim()
                            ));
                        }
                        extra_keys(fields, &["name", "type", "description"], &mut out);
                    }
                    out.push_str("\n</parameter>");
                }
            }
            extra_keys(parameters, &["type", "properties"], &mut out);
        }
        out.push_str("\n</parameters>");
        extra_keys(
            &tool,
            &["type", "name", "description", "parameters"],
            &mut out,
        );
        out.push_str("\n</function>");
    }
    out.push_str("\n</tools>");
    out.push_str(FORMAT);
    out
}

/// An earlier assistant turn: its text, and each call it made (in ChatML's
/// `<tool_call>{"name", "arguments"}</tool_call>`) in the coder's XML.
fn assistant_body(msg: &str) -> String {
    let (text, calls) = crate::minimax::lift_calls(msg);
    if calls.is_empty() {
        return format!("\n{msg}");
    }
    let mut out = String::new();
    let text = text.trim();
    if !text.is_empty() {
        out.push_str(&format!("\n{text}\n"));
    }
    for (name, arguments) in calls {
        out.push_str(&format!("\n<tool_call>\n<function={name}>\n"));
        if let Value::Object(map) = arguments {
            for (key, value) in map {
                let text = match value {
                    Value::Object(_) | Value::Array(_) => crate::llama::pythonic(&value),
                    other => python_str(&other),
                };
                out.push_str(&format!("<parameter={key}>\n{text}\n</parameter>\n"));
            }
        }
        out.push_str("</function>\n</tool_call>");
    }
    out
}

impl Instruct for Qwen3Coder {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.turn("system", msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn("user", msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.tokenizer.encode(&format!(
            "<|im_start|>assistant{}<|im_end|>\n",
            assistant_body(msg)
        ))
    }

    fn cue(&self) -> Vec<u32> {
        self.tokenizer.encode("<|im_start|>assistant\n")
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    /// The tools in a system turn of their own, opening on the coder's
    /// identity, as the template writes them with no system message.
    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        self.turn("system", &format!("{IDENTITY}{}", tools_section(tools)))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.turn(
            "user",
            &format!("<tool_response>\n{value}\n</tool_response>\n"),
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
        crate::mimo::xml_call_decoder(&self.tokenizer)
    }
}
