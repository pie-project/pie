//! Mistral 3 instruct implementation.
//!
//! Implements Mistral V3 chat template features:
//! - [INST]...[/INST] for user messages
//! - [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] for system messages
//! - [AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS] for tool definitions
//! - [TOOL_CALLS]name[ARGS]args for tool calls
//! - [TOOL_RESULTS]content[/TOOL_RESULTS] for tool outputs
//!
//! Reference: Mistral V3 Jinja chat template.

use pie_model_common::decoders::{GenericChatDecoder, NoopReasoningDecoder};
use pie_model_common::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use pie_tokenizer::{Tokenizer, TokenizerDecoder};
use std::sync::Arc;

// The implementation below mirrors the published Mistral V3 jinja chat
// template; the verbatim copy that used to sit here as a static was never
// read — the checkpoint's own `chat_template` is the reference.

// =============================================================================
// MistralInstruct
// =============================================================================

pub struct MistralInstruct {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    // Delimiters
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
            current_name: None,
        })
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

// =============================================================================
// Tool Decoder
// =============================================================================

#[derive(Debug, PartialEq)]
enum ToolState {
    Outside,
    InsideName,
    InsideArgs,
}

struct MistralToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    state: ToolState,
    current_name: Option<String>,
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
                        let keep = self.accumulated.len() - 50;
                        self.accumulated = self.accumulated[keep..].to_string();
                    }
                    return ToolEvent::Start;
                }
                ToolState::InsideName => {
                    if let Some(pos) = self.accumulated.find("[ARGS]") {
                        let name = self.accumulated[..pos].trim().to_string();
                        self.current_name = Some(name);
                        self.accumulated = self.accumulated[pos + "[ARGS]".len()..].to_string();
                        self.state = ToolState::InsideArgs;
                        continue;
                    }
                    return ToolEvent::Start; // Wait for ARGS
                }
                ToolState::InsideArgs => {
                    // Check for next marker
                    let mut end_pos = None;

                    if let Some(pos) = self.accumulated.find("[TOOL_CALLS]") {
                        end_pos = Some(pos);
                    } else if let Some(pos) = self.accumulated.find("</s>") {
                        end_pos = Some(pos);
                    }

                    if let Some(pos) = end_pos {
                        let args = self.accumulated[..pos].trim().to_string();
                        // Keep marker for next iteration/state check logic if needed.
                        self.accumulated = self.accumulated[pos..].to_string();

                        if let Some(name) = self.current_name.take() {
                            self.state = ToolState::Outside;
                            return ToolEvent::Call(name, args);
                        }
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
        self.current_name = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_tokenizer::Tokenizer;
    use std::sync::Arc;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn mistral() -> MistralInstruct {
        let tok = make_tok(&[
            "<s>",
            "</s>",
            " ",
            "[INST]",
            "[/INST]",
            "[SYSTEM_PROMPT]",
            "[/SYSTEM_PROMPT]",
            "[AVAILABLE_TOOLS]",
            "[/AVAILABLE_TOOLS]",
            "[TOOL_CALLS]",
            "[ARGS]",
            "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]",
            "Hello",
            "world",
            "func",
            "{",
            "}",
            ":",
            "\"",
            "arg",
            "name",
            "f",
            "Hi",
            "result",
            ",",
            "[",
            "]",
            "INST",
            "SYSTEM_PROMPT",
            "AVAILABLE_TOOLS",
            "TOOL_CALLS",
            "ARGS",
            "TOOL_RESULTS",
            r#"[{"name":"f"}]"#, // Add complex token to handle non-splitting mock tokenizer
        ]);
        MistralInstruct::new(tok)
    }

    #[test]
    fn system_format() {
        let inst = mistral();
        let tokens = inst.system("Hi");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "[SYSTEM_PROMPT]Hi[/SYSTEM_PROMPT]");
    }

    #[test]
    fn user_format() {
        let inst = mistral();
        let tokens = inst.user("Hi");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "[INST]Hi[/INST]");
    }

    #[test]
    fn equip_format() {
        let inst = mistral();
        let tokens = inst.equip(&[r#"{"name":"f"}"#.to_string()]);
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, r#"[AVAILABLE_TOOLS][{"name":"f"}][/AVAILABLE_TOOLS]"#);
    }

    #[test]
    fn answer_format() {
        let inst = mistral();
        let tokens = inst.answer("f", "result");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "[TOOL_RESULTS]result[/TOOL_RESULTS]");
    }

    #[test]
    fn grammar_generation() {
        let inst = mistral();
        let tools = vec![r#"{"function":{"name":"foo"}}"#.to_string()];
        let g = inst.tool_call_grammar(&tools).unwrap();
        assert!(g.source.contains("tool-call ::= \"[TOOL_CALLS]\""));
        assert!(g.source.contains("foo"));
    }

    #[test]
    fn full_conversation() {
        let inst = mistral();
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hi"));
        tokens.extend(inst.user("Hi"));
        tokens.extend(inst.assistant("Hi"));
        tokens.extend(inst.user("Hi"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "[SYSTEM_PROMPT]Hi[/SYSTEM_PROMPT]\
             [INST]Hi[/INST]\
             Hi</s>\
             [INST]Hi[/INST]"
        );
    }

    #[test]
    fn tool_decoder_parses_call() {
        let inst = mistral();
        let mut dec = inst.tool_decoder();
        dec.feed(&[9]); // [TOOL_CALLS]
        dec.feed(&[22]); // "f"
        dec.feed(&[10]); // [ARGS]
        let event = dec.feed(&[16, 17, 1]); // "{}" + "</s>"
        match event {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "f");
                assert_eq!(args, "{}");
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }
}
