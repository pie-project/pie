//! ChatML-family instruct implementation.
//!
//! Covers Qwen3, Qwen2.5, OLMo3, and any ChatML-based model.
//! Configurable via `ChatMLConfig` for thinking/tool support.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder, ReasoningEvent,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;

// =============================================================================
// Configuration
// =============================================================================

/// Feature flags for ChatML-family models.
pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
    /// Qwen2.5 wraps tool schemas in <tools></tools> XML tags
    pub wrap_tools_xml: bool,
    /// Stop token strings (vary per sub-architecture)
    pub stop_tokens: &'static [&'static str],
}

// =============================================================================
// QwenInstruct
// =============================================================================

pub struct QwenInstruct {
    tokenizer: Arc<Tokenizer>,
    config: ChatMLConfig,
    // Pre-tokenized delimiters
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
    // Thinking delimiters
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    // Tool delimiters
    tool_response_prefix_tokens: Vec<u32>,
    tool_response_suffix_tokens: Vec<u32>,
}

impl QwenInstruct {
    /// Create with full config.
    pub fn new(tokenizer: Arc<Tokenizer>, config: ChatMLConfig) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_ids: Vec<u32> = config.stop_tokens
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();
        Self {
            system_prefix: encode("<|im_start|>system\n"),
            user_prefix: encode("<|im_start|>user\n"),
            assistant_prefix: encode("<|im_start|>assistant\n"),
            turn_suffix: encode("<|im_end|>\n"),
            generation_header: encode("<|im_start|>assistant\n"),
            stop_ids,
            think_prefix_ids: encode("<think>\n"),
            think_suffix_ids: encode("</think>\n"),
            tool_response_prefix_tokens: encode("<tool_response>\n"),
            tool_response_suffix_tokens: encode("\n</tool_response>"),
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
}

impl Instruct for QwenInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("system", msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("user", msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("assistant", msg)
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
        let mut prompt = String::new();
        if self.config.wrap_tools_xml {
            // Qwen2.5 style: wrap in <tools></tools>
            prompt.push_str("\n\n# Tools\n\nYou may call one or more functions.\n\n<tools>\n");
            for tool in tools {
                prompt.push_str(tool);
                prompt.push('\n');
            }
            prompt.push_str("</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        } else {
            // Qwen3 style: simpler
            prompt.push_str("# Tools\n\nYou are provided with the following tools:\n\n");
            for tool in tools {
                prompt.push_str(tool);
                prompt.push_str("\n\n");
            }
        }
        self.system(&prompt)
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        if !self.config.has_tools {
            return Vec::new();
        }
        // Tool responses go in a user turn with <tool_response> wrapper
        let mut tokens = self.user_prefix.clone();
        tokens.extend(&self.tool_response_prefix_tokens);
        tokens.extend(self.tokenizer.encode(&format!("{}: {}", name, value)));
        tokens.extend(&self.tool_response_suffix_tokens);
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(QwenChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.stop_ids.clone(),
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(QwenReasoningDecoder {
            tokenizer: self.tokenizer.clone(),
            think_prefix_ids: self.think_prefix_ids.clone(),
            think_suffix_ids: self.think_suffix_ids.clone(),
            state: ReasoningState::Outside,
            accumulated: String::new(),
            match_pos: 0,
            has_thinking: self.config.has_thinking,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(QwenToolDecoder {
            tokenizer: self.tokenizer.clone(),
            accumulated: String::new(),
            inside: false,
            has_tools: self.config.has_tools,
        })
    }
}

// =============================================================================
// Chat Decoder
// =============================================================================

struct QwenChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for QwenChatDecoder {
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

// =============================================================================
// Reasoning Decoder
// =============================================================================

enum ReasoningState {
    Outside,
    Inside,
}

struct QwenReasoningDecoder {
    tokenizer: Arc<Tokenizer>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    state: ReasoningState,
    accumulated: String,
    match_pos: usize,
    has_thinking: bool,
}

impl ReasoningDecoder for QwenReasoningDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent {
        if !self.has_thinking {
            return ReasoningEvent::Delta(String::new());
        }
        match self.state {
            ReasoningState::Outside => {
                for &t in tokens {
                    if self.match_pos < self.think_prefix_ids.len()
                        && t == self.think_prefix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.think_prefix_ids.len() {
                            self.state = ReasoningState::Inside;
                            self.match_pos = 0;
                            return ReasoningEvent::Start;
                        }
                    } else {
                        self.match_pos = 0;
                    }
                }
                ReasoningEvent::Delta(String::new())
            }
            ReasoningState::Inside => {
                for &t in tokens {
                    if self.match_pos < self.think_suffix_ids.len()
                        && t == self.think_suffix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.think_suffix_ids.len() {
                            self.state = ReasoningState::Outside;
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
        self.state = ReasoningState::Outside;
        self.accumulated.clear();
        self.match_pos = 0;
    }
}

// =============================================================================
// Tool Decoder
// =============================================================================

struct QwenToolDecoder {
    tokenizer: Arc<Tokenizer>,
    accumulated: String,
    inside: bool,
    has_tools: bool,
}

impl ToolDecoder for QwenToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        if !self.has_tools {
            return ToolEvent::Start;
        }
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);

        if !self.inside {
            if self.accumulated.contains("<tool_call>") {
                self.inside = true;
                if let Some(pos) = self.accumulated.find("<tool_call>") {
                    self.accumulated = self.accumulated[pos + "<tool_call>".len()..].to_string();
                }
                return ToolEvent::Start;
            }
        } else if self.accumulated.contains("</tool_call>") {
            if let Some(pos) = self.accumulated.find("</tool_call>") {
                let call_json = self.accumulated[..pos].trim().to_string();
                self.accumulated = self.accumulated[pos + "</tool_call>".len()..].to_string();
                self.inside = false;
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&call_json) {
                    let name = v["name"].as_str().unwrap_or("").to_string();
                    let args = v["arguments"].to_string();
                    return ToolEvent::Call(name, args);
                }
            }
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.accumulated.clear();
        self.inside = false;
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

    fn qwen3(tok: Arc<Tokenizer>) -> QwenInstruct {
        QwenInstruct::new(tok, ChatMLConfig {
            has_thinking: true, has_tools: true, wrap_tools_xml: false,
            stop_tokens: &["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        })
    }

    fn qwen25(tok: Arc<Tokenizer>) -> QwenInstruct {
        QwenInstruct::new(tok, ChatMLConfig {
            has_thinking: false, has_tools: true, wrap_tools_xml: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        })
    }

    fn olmo3(tok: Arc<Tokenizer>) -> QwenInstruct {
        QwenInstruct::new(tok, ChatMLConfig {
            has_thinking: true, has_tools: false, wrap_tools_xml: false,
            stop_tokens: &["<|im_end|>"],
        })
    }

    #[test]
    fn qwen3_has_3_stop_tokens() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = qwen3(tok);
        assert_eq!(inst.stop_ids.len(), 3);
        assert!(inst.stop_ids.contains(&1));
        assert!(inst.stop_ids.contains(&0));
        assert!(inst.stop_ids.contains(&2));
    }

    #[test]
    fn qwen25_has_2_stop_tokens() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = qwen25(tok);
        assert_eq!(inst.stop_ids.len(), 2);
        assert!(inst.stop_ids.contains(&1));
        assert!(inst.stop_ids.contains(&2));
        assert!(!inst.stop_ids.contains(&0));
    }

    #[test]
    fn olmo3_has_1_stop_token() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = olmo3(tok);
        assert_eq!(inst.stop_ids.len(), 1);
        assert_eq!(inst.stop_ids[0], 1);
    }

    #[test]
    fn qwen3_thinking_enabled() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = qwen3(tok);
        assert!(inst.config.has_thinking);
    }

    #[test]
    fn qwen25_thinking_disabled() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = qwen25(tok);
        assert!(!inst.config.has_thinking);
    }

    #[test]
    fn equip_noop_when_disabled() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = olmo3(tok);
        assert!(inst.equip(&["tool".to_string()]).is_empty());
        assert!(inst.answer("fn1", "42").is_empty());
    }

    #[test]
    fn equip_produces_tokens_when_enabled() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = qwen3(tok);
        assert!(inst.config.has_tools, "tools should be enabled");
    }

    #[test]
    fn answer_produces_tokens_when_enabled() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = qwen3(tok);
        assert!(inst.config.has_tools, "tools should be enabled for answer");
    }

    #[test]
    fn seal_returns_stop_ids() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = qwen3(tok);
        assert_eq!(inst.seal(), inst.stop_ids);
    }

    #[test]
    fn generation_header_matches_cue() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system", "\\n", "user", "assistant", "Hello", " world", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
        let inst = qwen3(tok);
        assert_eq!(inst.cue(), inst.generation_header);
    }
}
