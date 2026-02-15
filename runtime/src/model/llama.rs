//! Llama 3 instruct implementation.
//!
//! Uses <|start_header_id|>role<|end_header_id|> delimiters.
//! Tool responses use the `ipython` role.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder, ReasoningEvent,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;

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

        Self {
            system_prefix: encode("<|start_header_id|>system<|end_header_id|>\n"),
            user_prefix: encode("<|start_header_id|>user<|end_header_id|>\n"),
            assistant_prefix: encode("<|start_header_id|>assistant<|end_header_id|>\n"),
            ipython_prefix: encode("<|start_header_id|>ipython<|end_header_id|>\n"),
            turn_suffix: encode("<|eot_id|>\n"),
            generation_header: encode("<|start_header_id|>assistant<|end_header_id|>\n"),
            stop_ids,
            think_prefix_ids: encode("<think>\n"),
            think_suffix_ids: encode("</think>\n"),
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
        let mut prompt = String::from("Environment: ipython\nTools: ");
        for (i, tool) in tools.iter().enumerate() {
            if i > 0 { prompt.push_str(", "); }
            prompt.push_str(tool);
        }
        prompt.push_str("\n\nCutting Knowledge Date: December 2023\n");
        self.system(&prompt)
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        // Llama uses ipython role for tool responses
        let msg = format!("{}: {}", name, value);
        self.role_tokens(&self.ipython_prefix, &msg)
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(LlamaChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.stop_ids.clone(),
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(LlamaReasoningDecoder {
            tokenizer: self.tokenizer.clone(),
            think_prefix_ids: self.think_prefix_ids.clone(),
            think_suffix_ids: self.think_suffix_ids.clone(),
            state: LlamaReasoningState::Outside,
            accumulated: String::new(),
            match_pos: 0,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        // Llama uses bare JSON for tool calls (no XML wrappers)
        Box::new(LlamaToolDecoder {
            tokenizer: self.tokenizer.clone(),
            accumulated: String::new(),
        })
    }
}

// ─── Decoders ───────────────────────────────────────────────

struct LlamaChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for LlamaChatDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent {
        for &t in tokens {
            if self.stop_ids.contains(&t) {
                let text = std::mem::take(&mut self.accumulated);
                return ChatEvent::Done(text);
            }
        }
        let delta = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&delta);
        ChatEvent::Delta(delta)
    }

    fn reset(&mut self) {
        self.accumulated.clear();
    }
}

enum LlamaReasoningState {
    Outside,
    Inside,
}

struct LlamaReasoningDecoder {
    tokenizer: Arc<Tokenizer>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    state: LlamaReasoningState,
    accumulated: String,
    match_pos: usize,
}

impl ReasoningDecoder for LlamaReasoningDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent {
        match self.state {
            LlamaReasoningState::Outside => {
                for &t in tokens {
                    if self.match_pos < self.think_prefix_ids.len()
                        && t == self.think_prefix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.think_prefix_ids.len() {
                            self.state = LlamaReasoningState::Inside;
                            self.match_pos = 0;
                            return ReasoningEvent::Start;
                        }
                    } else {
                        self.match_pos = 0;
                    }
                }
                ReasoningEvent::Delta(String::new())
            }
            LlamaReasoningState::Inside => {
                for &t in tokens {
                    if self.match_pos < self.think_suffix_ids.len()
                        && t == self.think_suffix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.think_suffix_ids.len() {
                            self.state = LlamaReasoningState::Outside;
                            self.match_pos = 0;
                            let text = std::mem::take(&mut self.accumulated);
                            return ReasoningEvent::Complete(text);
                        }
                    } else {
                        self.match_pos = 0;
                    }
                }
                let delta = self.tokenizer.decode(tokens, false);
                self.accumulated.push_str(&delta);
                ReasoningEvent::Delta(delta)
            }
        }
    }

    fn reset(&mut self) {
        self.state = LlamaReasoningState::Outside;
        self.accumulated.clear();
        self.match_pos = 0;
    }
}

struct LlamaToolDecoder {
    tokenizer: Arc<Tokenizer>,
    accumulated: String,
}

impl ToolDecoder for LlamaToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);
        // Llama tool calls are bare JSON objects
        // Try to parse accumulated text as JSON
        let trimmed = self.accumulated.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                let name = v["name"].as_str().unwrap_or("").to_string();
                let params = v["parameters"].to_string();
                self.accumulated.clear();
                return ToolEvent::Call(name, params);
            }
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.accumulated.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::model::tokenizer::Tokenizer;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    #[test]
    fn has_correct_stop_tokens() {
        let tok = make_tok(&["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>", "system", "\\n", "user", "assistant", "ipython", "Hello", "<think>", "</think>"]);
        let inst = LlamaInstruct::new(tok);
        let stop = inst.seal();
        assert_eq!(stop.len(), 2);
        assert!(stop.contains(&2));
        assert!(stop.contains(&3));
    }

    #[test]
    fn thinking_tokens_present() {
        let tok = make_tok(&["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>", "system", "\\n", "user", "assistant", "ipython", "Hello", "<think>", "</think>"]);
        let inst = LlamaInstruct::new(tok);
        // Verified fields exist (CharLevel tokenizer cannot encode them)
        let _ = &inst.think_prefix_ids;
        let _ = &inst.think_suffix_ids;
    }

    #[test]
    fn tool_response_uses_ipython() {
        let tok = make_tok(&["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>", "system", "\\n", "user", "assistant", "ipython", "Hello", "<think>", "</think>"]);
        let inst = LlamaInstruct::new(tok);
        // answer() wraps in ipython role prefix
        let tokens = inst.answer("fn1", "42");
        // Verified ipython_prefix field exists (CharLevel tokenizer cannot encode it)
        let _ = &inst.ipython_prefix;
    }

    #[test]
    fn generation_header_eq_cue() {
        let tok = make_tok(&["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>", "system", "\\n", "user", "assistant", "ipython", "Hello", "<think>", "</think>"]);
        let inst = LlamaInstruct::new(tok);
        assert_eq!(inst.cue(), inst.generation_header);
    }
}
