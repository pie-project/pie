use crate::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::shared::decoders::{GenericChatDecoder, ThinkingDecoder};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

pub struct R1Instruct {
    tokenizer: Arc<Tokenizer>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    eos_ids: Vec<u32>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,

    tool_call_begin: Vec<u32>,
    tool_call_end: Vec<u32>,
    tool_outputs_begin: Vec<u32>,
    tool_outputs_end: Vec<u32>,
    tool_output_begin: Vec<u32>,
    tool_output_end: Vec<u32>,
}

impl R1Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<｜end▁of▁sentence｜>", "<|EOT|>"];
        let eos_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            user_prefix: encode("<｜User｜>"),
            assistant_prefix: encode("<｜Assistant｜>"),
            eos_ids,
            think_prefix_ids: encode("<think>\n"),
            think_suffix_ids: encode("</think>\n"),
            tool_call_begin: encode("<｜tool▁call▁begin｜>"),
            tool_call_end: encode("<｜tool▁call▁end｜>"),
            tool_outputs_begin: encode("<｜tool▁outputs▁begin｜>"),
            tool_outputs_end: encode("<｜tool▁outputs▁end｜>"),
            tool_output_begin: encode("<｜tool▁output▁begin｜>"),
            tool_output_end: encode("<｜tool▁output▁end｜>"),
            tokenizer,
        }
    }

    fn strip_thinking(msg: &str) -> &str {
        if let Some(pos) = msg.rfind("</think>") {
            &msg[pos + "</think>".len()..]
        } else {
            msg
        }
    }

    fn build_tool_system_prompt(tools: &[String]) -> String {
        let mut prompt = String::from(
            "You are a helpful assistant with tool calling capabilities. \
             When a tool call is needed, you MUST use the following format to issue the call:\n\
             <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNCTION_NAME\n\
             ```json\n\
             {\"param1\": \"value1\", \"param2\": \"value2\"}\n\
             ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n\n\
             Make sure the JSON is valid.\
             ## Tools\n\n\
             ### Function\n\n\
             You have the following functions available:\n\n",
        );
        for tool in tools {
            prompt.push_str("\n```json\n");
            prompt.push_str(tool);
            prompt.push_str("\n```\n");
        }
        prompt
    }
}

impl Instruct for R1Instruct {
    fn system(&self, msg: &str) -> Vec<u32> {

        self.tokenizer.encode(msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {

        let mut tokens = self.user_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {

        let stripped = Self::strip_thinking(msg);
        let mut tokens = self.assistant_prefix.clone();
        tokens.extend(self.tokenizer.encode(stripped));
        tokens.extend(&self.eos_ids[..1]);
        tokens
    }

    fn cue(&self) -> Vec<u32> {

        self.assistant_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.eos_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {

        let prompt = Self::build_tool_system_prompt(tools);
        self.system(&prompt)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {

        let mut tokens = self.tool_outputs_begin.clone();
        tokens.extend(&self.tool_output_begin);
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_output_end);
        tokens.extend(&self.tool_outputs_end);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.eos_ids.clone(),
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
        Box::new(R1ToolDecoder {
            decoder: self.tokenizer.decoder(false),
            tool_call_begin: self.tool_call_begin.clone(),
            tool_call_end: self.tool_call_end.clone(),
            accumulated: String::new(),
            inside: false,
            match_pos: 0,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {

        let mut names: Vec<String> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool)
                && let Some(name) = parsed
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
            {
                names.push(format!("\"{}\"", name));
            }
        }
        if names.is_empty() {

            return None;
        }

        let name_alt = names.join(" | ");
        let grammar = format!(
            r#"root ::= "<｜tool▁calls▁begin｜>" tool-call+ "<｜tool▁calls▁end｜>"
tool-call ::= "<｜tool▁call▁begin｜>" "function" "<｜tool▁sep｜>" tool-name "\n```json\n" json-object "\n```" "<｜tool▁call▁end｜>"
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

const TOOL_SEP: &str = "<｜tool▁sep｜>";

struct R1ToolDecoder {
    decoder: TokenizerDecoder,
    tool_call_begin: Vec<u32>,
    tool_call_end: Vec<u32>,
    accumulated: String,
    inside: bool,
    match_pos: usize,
}

impl ToolDecoder for R1ToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        if !self.inside {

            for &t in tokens {
                if self.match_pos < self.tool_call_begin.len()
                    && t == self.tool_call_begin[self.match_pos]
                {
                    self.match_pos += 1;
                    if self.match_pos == self.tool_call_begin.len() {
                        self.inside = true;
                        self.match_pos = 0;
                        self.accumulated.clear();
                        return ToolEvent::Start;
                    }
                } else {
                    self.match_pos = 0;
                }
            }
        } else {

            for &t in tokens {
                if self.match_pos < self.tool_call_end.len()
                    && t == self.tool_call_end[self.match_pos]
                {
                    self.match_pos += 1;
                    if self.match_pos == self.tool_call_end.len() {
                        self.inside = false;
                        self.match_pos = 0;

                        let content = std::mem::take(&mut self.accumulated);

                        if let Some(sep_pos) = content.find("\n```json\n") {
                            let header = &content[..sep_pos];
                            let json_start = sep_pos + "\n```json\n".len();
                            if let Some(json_end) = content[json_start..].find("\n```") {
                                let args = content[json_start..json_start + json_end].to_string();

                                let name = match header.rsplit_once(TOOL_SEP) {
                                    Some((_, name)) => name,

                                    None => header.rsplit_once('\n').map_or(header, |(_, n)| n),
                                };
                                return ToolEvent::Call(name.trim().to_string(), args);
                            }
                        }
                    }
                } else {
                    self.match_pos = 0;
                }
            }
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
        self.match_pos = 0;
    }
}
