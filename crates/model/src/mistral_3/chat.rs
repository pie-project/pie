use crate::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, keep_tail};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

pub struct MistralInstruct {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,

    inst_start: Vec<u32>,
    inst_end: Vec<u32>,
    sys_start: Vec<u32>,
    sys_end: Vec<u32>,
    tools_start: Vec<u32>,
    tools_end: Vec<u32>,
    tool_results_start: Vec<u32>,
    tool_results_end: Vec<u32>,
}

impl MistralInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["</s>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let inst_start = encode("[INST]");
        let inst_end = encode("[/INST]");

        Self {
            stop_ids,
            inst_start,
            inst_end,
            sys_start: encode("[SYSTEM_PROMPT]"),
            sys_end: encode("[/SYSTEM_PROMPT]"),
            tools_start: encode("[AVAILABLE_TOOLS]"),
            tools_end: encode("[/AVAILABLE_TOOLS]"),
            tool_results_start: encode("[TOOL_RESULTS]"),
            tool_results_end: encode("[/TOOL_RESULTS]"),
            tokenizer,
        }
    }
}

impl Instruct for MistralInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.sys_start.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.sys_end);
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.inst_start.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.inst_end);
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(msg);
        tokens.extend(&self.stop_ids);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        Vec::new()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        let json_list = format!("[{}]", tools.join(","));
        let mut tokens = self.tools_start.clone();
        tokens.extend(self.tokenizer.encode(&json_list));
        tokens.extend(&self.tools_end);
        tokens
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        let mut tokens = self.tool_results_start.clone();
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_results_end);
        tokens
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
        Box::new(MistralToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            state: ToolState::Outside,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
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
            r#"root ::= tool-call+
tool-call ::= "[TOOL_CALLS]" tool-name "[ARGS]" json-object
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
        Some(ToolGrammar { source: grammar })
    }
}

#[derive(Debug, PartialEq)]
enum ToolState {
    Outside,
    InsideName,

    InsideArgs(String),
}

struct MistralToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    state: ToolState,
}

impl ToolDecoder for MistralToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        loop {
            match self.state {
                ToolState::Outside => {
                    if let Some(pos) = self.accumulated.find("[TOOL_CALLS]") {
                        self.accumulated =
                            self.accumulated[pos + "[TOOL_CALLS]".len()..].to_string();
                        self.state = ToolState::InsideName;
                        continue;
                    }
                    if self.accumulated.len() > 200 {
                        self.accumulated = keep_tail(&self.accumulated, 50).to_string();
                    }
                    return ToolEvent::Start;
                }
                ToolState::InsideName => {
                    if let Some(pos) = self.accumulated.find("[ARGS]") {
                        let name = self.accumulated[..pos].trim().to_string();
                        self.accumulated = self.accumulated[pos + "[ARGS]".len()..].to_string();
                        self.state = ToolState::InsideArgs(name);
                        continue;
                    }
                    return ToolEvent::Start;
                }
                ToolState::InsideArgs(ref name) => {

                    let mut end_pos = None;

                    if let Some(pos) = self.accumulated.find("[TOOL_CALLS]") {
                        end_pos = Some(pos);
                    } else if let Some(pos) = self.accumulated.find("</s>") {
                        end_pos = Some(pos);
                    }

                    if let Some(pos) = end_pos {
                        let name = name.clone();
                        let args = self.accumulated[..pos].trim().to_string();

                        self.accumulated = self.accumulated[pos..].to_string();
                        self.state = ToolState::Outside;
                        return ToolEvent::Call(name, args);
                    }
                    return ToolEvent::Start;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.state = ToolState::Outside;
    }
}
