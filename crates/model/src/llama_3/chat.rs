use crate::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::shared::decoders::{GenericChatDecoder, ThinkingDecoder};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

pub struct LlamaInstruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    ipython_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
}

impl LlamaInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<|eot_id|>", "<|end_of_text|>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let start_header = encode("<|start_header_id|>");
        let end_header = encode("<|end_header_id|>");
        let double_nl = encode("\n\n");

        let make_role = |role: &str| -> Vec<u32> {
            let mut v = start_header.clone();
            v.extend(encode(role));
            v.extend(&end_header);
            v.extend(&double_nl);
            v
        };

        let system_prefix = make_role("system");
        let user_prefix = make_role("user");
        let assistant_prefix = make_role("assistant");
        let ipython_prefix = make_role("ipython");

        let turn_suffix = encode("<|eot_id|>");

        Self {
            system_prefix,
            user_prefix,
            assistant_prefix: assistant_prefix.clone(),
            ipython_prefix,
            turn_suffix,
            generation_header: assistant_prefix,
            stop_ids,
            think_prefix_ids: encode("<think>"),
            think_suffix_ids: encode("</think>"),
            tokenizer,
        }
    }

    fn role_tokens(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }
}

impl Instruct for LlamaInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.system_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.assistant_prefix, msg)
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_header.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }

        let mut prompt = String::from("Environment: ipython\n");
        prompt.push_str("Cutting Knowledge Date: December 2023\n");
        prompt.push_str("Today Date: 26 Jul 2024\n\n");
        prompt.push_str("You have access to the following functions. To call a function, please respond with JSON for a function call.");
        prompt.push_str("Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.");
        prompt.push_str("Do not use variables.\n\n");

        for tool in tools {
            prompt.push_str(tool);
            prompt.push_str("\n\n");
        }
        self.system(&prompt)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.role_tokens(&self.ipython_prefix, value)
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if tools.is_empty() {
            return None;
        }

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
            r#"root ::= tool-call
tool-call ::= "{{" ws "\"name\"" ws ":" ws "\"" tool-name "\"" "," ws "\"parameters\"" ws ":" ws json-object "}}"
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
ws ::= [ \t\n]*
"#,
            name_alt = name_alt
        );
        Some(ToolGrammar { source: grammar })
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
            self.think_prefix_ids.clone(),
            self.think_suffix_ids.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(LlamaToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
        })
    }
}

struct LlamaToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
}

impl ToolDecoder for LlamaToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);
        let trimmed = self.accumulated.trim();
        if trimmed.starts_with('{')
            && trimmed.ends_with('}')
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed)
            && let Some(name) = v["name"].as_str()
        {

            let params = v["parameters"].to_string();
            let name = name.to_string();
            self.accumulated.clear();
            return ToolEvent::Call(name, params);
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
    }
}
