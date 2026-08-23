use crate::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, ThinkingDecoder};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
    pub generation_suffix: &'static str,

    pub stop_tokens: &'static [&'static str],
}

pub struct QwenInstruct {
    tokenizer: Arc<Tokenizer>,
    config: ChatMLConfig,

    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,

    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,

    tool_response_prefix_tokens: Vec<u32>,
    tool_response_suffix_tokens: Vec<u32>,
}

impl QwenInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>, config: ChatMLConfig) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_ids: Vec<u32> = config
            .stop_tokens
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let im_start = encode("<|im_start|>");
        let im_end = encode("<|im_end|>");
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = im_start.clone();
            v.extend(encode(role));
            v.extend(&newline);
            v
        };

        let mut turn_suffix = im_end;
        turn_suffix.extend(&newline);

        let think_prefix = encode("<think>");
        let think_suffix = encode("</think>");

        let mut tool_resp_prefix = encode("<tool_response>");
        tool_resp_prefix.extend(&newline);
        let mut tool_resp_suffix = newline.clone();
        tool_resp_suffix.extend(encode("</tool_response>"));

        let mut generation_header = make_prefix("assistant");
        generation_header.extend(encode(config.generation_suffix));

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            assistant_prefix: make_prefix("assistant"),
            generation_header,
            turn_suffix,
            stop_ids,
            think_prefix_ids: think_prefix,
            think_suffix_ids: think_suffix,
            tool_response_prefix_tokens: tool_resp_prefix,
            tool_response_suffix_tokens: tool_resp_suffix,
            tokenizer,
            config,
        }
    }

    fn role_tokens(&self, role: &str, msg: &str) -> Vec<u32> {
        let prefix = match role {
            "system" => &self.system_prefix,
            "user" => &self.user_prefix,
            "assistant" => &self.assistant_prefix,
            _ => &self.user_prefix,
        };
        let mut tokens = prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn strip_thinking(msg: &str) -> &str {
        if let Some(pos) = msg.rfind("</think>") {
            msg[pos + "</think>".len()..].trim_start_matches('\n')
        } else {
            msg
        }
    }

    fn build_tool_system_prompt(tools: &[String]) -> String {
        let mut prompt = String::from(
            " # Tools\n\n\
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

    fn build_tool_call_grammar(tools: &[String]) -> Option<String> {
        let mut names: Vec<String> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
                let name = parsed
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .or_else(|| parsed.get("name"))
                    .and_then(|n| n.as_str());
                if let Some(n) = name {
                    names.push(format!("\"{}\"", n));
                }
            }
        }
        if names.is_empty() {
            return None;
        }

        let name_alt = names.join(" | ");
        let grammar = format!(
            r#"root ::= tool-call ("\n" tool-call)*
tool-call ::= "<tool_call>\n" tool-json "\n</tool_call>"
tool-json ::= "{{"  "\"name\": \"" tool-name "\", \"arguments\": " json-object "}}"
tool-name ::= {name_alt}
json-object ::= "{{" json-members? "}}"
json-members ::= json-pair ("," json-pair)*
json-pair ::= json-string ":" json-value
json-value ::= json-string | json-number | json-object | json-array | "true" | "false" | "null"
json-string ::= "\"" json-chars "\""
json-chars ::= json-char*
json-char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-array ::= "[" (json-value ("," json-value)*)? "]"
"#,
            name_alt = name_alt
        );
        Some(grammar)
    }
}

impl Instruct for QwenInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("system", msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("user", msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let stripped = if self.config.has_thinking {
            Self::strip_thinking(msg)
        } else {
            msg
        };
        self.role_tokens("assistant", stripped)
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_header.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if !self.config.has_tools {
            return Vec::new();
        }
        let prompt = Self::build_tool_system_prompt(tools);
        self.system(&prompt)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        if !self.config.has_tools {
            return Vec::new();
        }

        let mut tokens = self.user_prefix.clone();
        tokens.extend(&self.tool_response_prefix_tokens);
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_response_suffix_tokens);
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        if !self.config.has_thinking {
            return Box::new(NoopReasoningDecoder);
        }
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.think_prefix_ids.clone(),
            self.think_suffix_ids.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(QwenToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
            has_tools: self.config.has_tools,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if !self.config.has_tools || tools.is_empty() {
            return None;
        }
        let source = Self::build_tool_call_grammar(tools)?;
        Some(ToolGrammar { source })
    }
}

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

struct QwenToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
    has_tools: bool,
}

impl ToolDecoder for QwenToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        if !self.has_tools {
            return ToolEvent::Start;
        }
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        loop {
            if !self.inside {
                let Some(pos) = self.accumulated.find(TOOL_CALL_OPEN) else {
                    return ToolEvent::Start;
                };
                self.accumulated = self.accumulated[pos + TOOL_CALL_OPEN.len()..].to_string();
                self.inside = true;
                continue;
            }
            let Some(pos) = self.accumulated.find(TOOL_CALL_CLOSE) else {
                return ToolEvent::Start;
            };
            let call_json = self.accumulated[..pos].trim().to_string();
            self.accumulated = self.accumulated[pos + TOOL_CALL_CLOSE.len()..].to_string();
            self.inside = false;
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&call_json)
                && let Some(name) = v["name"].as_str()
            {
                let args = v["arguments"].to_string();
                return ToolEvent::Call(name.to_string(), args);
            }
            return ToolEvent::Start;
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
    }
}

pub const QWEN_CHATML: ChatMLConfig = ChatMLConfig {
    has_thinking: true,
    has_tools: true,
    generation_suffix: "",
    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
};

pub const NEMOTRON_CHATML: ChatMLConfig = ChatMLConfig {
    has_thinking: true,
    has_tools: false,
    generation_suffix: "<think>\n",
    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
};

pub const GLM_CHATML: ChatMLConfig = ChatMLConfig {
    has_thinking: true,
    has_tools: true,
    generation_suffix: "",
    stop_tokens: &["<|im_end|>", "<|endoftext|>", "<|user|>", "<|assistant|>"],
};

pub const PLAIN_CHATML: ChatMLConfig = ChatMLConfig {
    has_thinking: false,
    has_tools: false,
    generation_suffix: "",
    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
};
