use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decoders::GenericChatDecoder;
use crate::instruct::{
    ChatDecoder, Instruct, Reasoning, ReasoningDecoder, ToolCall, ToolDecoder,
};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

pub struct Template {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    tool_response_open: Vec<u32>,
    tool_response_close: Vec<u32>,
    stop_ids: Vec<u32>,
    think_open: u32,
    think_close: u32,
}

impl Template {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>", "<|user|>", "<|assistant|>"]
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();
        let think_open = tokenizer.token_to_id("<think>").unwrap_or(0);
        let think_close = tokenizer.token_to_id("</think>").unwrap_or(0);

        let make_prefix = |role: &str| encode(&format!("<|im_start|>{role}\n"));
        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            model_prefix: make_prefix("assistant"),
            turn_suffix: encode("<|im_end|>\n"),
            tool_response_open: encode("<tool_response>\n"),
            tool_response_close: encode("\n</tool_response>"),
            stop_ids,
            think_open,
            think_close,
            tokenizer,
        }
    }

    fn turn(&self, prefix: &[u32], message: &str) -> Vec<u32> {
        let mut v = prefix.to_vec();
        v.extend(self.tokenizer.encode(message.trim()));
        v.extend(&self.turn_suffix);
        v
    }

    fn strip_thinking(message: &str) -> &str {
        match message.rfind("</think>") {
            Some(pos) => message[pos + "</think>".len()..].trim_start_matches('\n'),
            None => message,
        }
    }

    fn tool_prompt(tools: &[String]) -> String {
        let mut prompt = String::from(
            "# Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>",
        );
        for tool in tools {
            prompt.push('\n');
            prompt.push_str(tool);
        }
        prompt.push_str(
            "\n</tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
             </tool_call>",
        );
        prompt
    }
}

impl Instruct for Template {
    fn system(&self, message: &str) -> Vec<u32> {
        self.turn(&self.system_prefix, message)
    }

    fn first_user(&self, message: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, message)
    }

    fn user(&self, message: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, message)
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut v = self.turn(&self.system_prefix, system);
        v.extend(self.turn(&self.user_prefix, user));
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        self.turn(&self.model_prefix, Self::strip_thinking(message))
    }

    fn cue(&self) -> Vec<u32> {
        self.model_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        self.system(&Self::tool_prompt(tools))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        let mut v = self.user_prefix.clone();
        v.extend(&self.tool_response_open);
        v.extend(self.tokenizer.encode(value.trim()));
        v.extend(&self.tool_response_close);
        v.extend(&self.turn_suffix);
        v
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder {
            tokenizer: self.tokenizer.clone(),
            open: self.think_open,
            close: self.think_close,
            thinking: false,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(ToolCallDecoder {
            tokenizer: self.tokenizer.clone(),
            pending: String::new(),
            inside: false,
        })
    }
}

struct ThinkingDecoder {
    tokenizer: Arc<Tokenizer>,
    open: u32,
    close: u32,
    thinking: bool,
}

impl ReasoningDecoder for ThinkingDecoder {
    fn push(&mut self, token: u32) -> Option<Reasoning> {
        if token == self.open {
            self.thinking = true;
            return None;
        }
        if token == self.close {
            self.thinking = false;
            return None;
        }
        let text = self.tokenizer.decode(&[token], true);
        if self.thinking {
            Some(Reasoning::Thinking(text))
        } else {
            Some(Reasoning::Answer(text))
        }
    }
}

struct ToolCallDecoder {
    tokenizer: Arc<Tokenizer>,
    pending: String,
    inside: bool,
}

impl ToolDecoder for ToolCallDecoder {
    fn push(&mut self, token: u32) -> Option<ToolCall> {
        self.pending.push_str(&self.tokenizer.decode(&[token], true));
        loop {
            if !self.inside {
                let open = self.pending.find(TOOL_CALL_OPEN)?;
                self.pending = self.pending[open + TOOL_CALL_OPEN.len()..].to_string();
                self.inside = true;
                continue;
            }
            let close = self.pending.find(TOOL_CALL_CLOSE)?;
            let body = self.pending[..close].trim().to_string();
            self.pending = self.pending[close + TOOL_CALL_CLOSE.len()..].to_string();
            self.inside = false;
            if let Some(name) = quoted_after(&body, "\"name\"") {
                let arguments =
                    braced_after(&body, "\"arguments\"").unwrap_or_else(|| "{}".to_string());
                return Some(ToolCall { name, arguments });
            }
            return None;
        }
    }
}

fn value_after<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let at = body.find(key)? + key.len();
    let rest = body[at..].trim_start();
    Some(rest.strip_prefix(':')?.trim_start())
}

fn quoted_after(body: &str, key: &str) -> Option<String> {
    let rest = value_after(body, key)?.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn braced_after(body: &str, key: &str) -> Option<String> {
    let rest = value_after(body, key)?;
    if !rest.starts_with('{') {
        return None;
    }
    let mut depth = 0usize;
    for (at, ch) in rest.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..=at].to_string());
                }
            }
            _ => {}
        }
    }
    None
}
