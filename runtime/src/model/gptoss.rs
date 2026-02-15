//! GPT-OSS instruct implementation.
//!
//! Uses channel-based formatting with analysis/final channels.
//! Reasoning uses the `analysis` channel, not XML tags.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder, ReasoningEvent,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;

pub struct GptOssInstruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    developer_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_final_prefix: Vec<u32>,
    assistant_analysis_prefix: Vec<u32>,
    end_token: Vec<u32>,
    stop_ids: Vec<u32>,
    // Channel tokens for decoder
    analysis_prefix_ids: Vec<u32>,
    final_prefix_ids: Vec<u32>,
}

impl GptOssInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<|endoftext|>", "<|return|>", "<|call|>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            system_prefix: encode("<|start|>system<|message|>"),
            developer_prefix: encode("<|start|>developer<|message|>"),
            user_prefix: encode("<|start|>user<|message|>"),
            assistant_final_prefix: encode("<|start|>assistant<|channel|>final<|message|>"),
            assistant_analysis_prefix: encode("<|start|>assistant<|channel|>analysis<|message|>"),
            end_token: encode("<|end|>"),
            stop_ids,
            analysis_prefix_ids: encode("<|channel|>analysis<|message|>"),
            final_prefix_ids: encode("<|channel|>final<|message|>"),
            tokenizer,
        }
    }

    fn wrap(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.end_token);
        tokens
    }
}

impl Instruct for GptOssInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        // GPT-OSS uses developer role for system-like messages
        self.wrap(&self.developer_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.assistant_final_prefix, msg)
    }

    fn cue(&self) -> Vec<u32> {
        // GPT-OSS generation prompt is just the assistant start header
        let mut tokens = Vec::new();
        tokens.extend(self.tokenizer.encode("<|start|>assistant"));
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        // GPT-OSS tool calling uses <|call|> stop token
        // Tool schemas would be in system/developer prompt
        Vec::new()
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        let msg = format!("{}: {}", name, value);
        self.wrap(&self.user_prefix, &msg)
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GptOssChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.stop_ids.clone(),
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        // GPT-OSS reasoning uses channel-based detection
        Box::new(GptOssReasoningDecoder {
            tokenizer: self.tokenizer.clone(),
            analysis_prefix_ids: self.analysis_prefix_ids.clone(),
            final_prefix_ids: self.final_prefix_ids.clone(),
            end_ids: self.end_token.clone(),
            state: GptOssState::Outside,
            accumulated: String::new(),
            match_pos: 0,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(GptOssToolDecoder)
    }
}

// ─── Decoders ───────────────────────────────────────────────

struct GptOssChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for GptOssChatDecoder {
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

enum GptOssState {
    Outside,
    InsideAnalysis,
}

struct GptOssReasoningDecoder {
    tokenizer: Arc<Tokenizer>,
    analysis_prefix_ids: Vec<u32>,
    final_prefix_ids: Vec<u32>,
    end_ids: Vec<u32>,
    state: GptOssState,
    accumulated: String,
    match_pos: usize,
}

impl ReasoningDecoder for GptOssReasoningDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent {
        match self.state {
            GptOssState::Outside => {
                for &t in tokens {
                    if self.match_pos < self.analysis_prefix_ids.len()
                        && t == self.analysis_prefix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.analysis_prefix_ids.len() {
                            self.state = GptOssState::InsideAnalysis;
                            self.match_pos = 0;
                            return ReasoningEvent::Start;
                        }
                    } else {
                        self.match_pos = 0;
                    }
                }
                ReasoningEvent::Delta(String::new())
            }
            GptOssState::InsideAnalysis => {
                // End on <|end|> token
                for &t in tokens {
                    if self.match_pos < self.end_ids.len()
                        && t == self.end_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.end_ids.len() {
                            self.state = GptOssState::Outside;
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
        self.state = GptOssState::Outside;
        self.accumulated.clear();
        self.match_pos = 0;
    }
}

/// GPT-OSS uses <|call|> stop token for tool calling, no in-band tool decoder
struct GptOssToolDecoder;

impl ToolDecoder for GptOssToolDecoder {
    fn feed(&mut self, _tokens: &[u32]) -> ToolEvent {
        ToolEvent::Start
    }

    fn reset(&mut self) {}
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
        let tok = make_tok(&["<|start|>", "<|message|>", "<|channel|>", "<|end|>", "<|endoftext|>", "<|return|>", "<|call|>", "system", "developer", "user", "assistant", "analysis", "final", "\\n", "Hello"]);
        let inst = GptOssInstruct::new(tok);
        let stop = inst.seal();
        assert!(stop.contains(&4));
    }

    #[test]
    fn system_uses_developer_prefix() {
        let tok = make_tok(&["<|start|>", "<|message|>", "<|channel|>", "<|end|>", "<|endoftext|>", "<|return|>", "<|call|>", "system", "developer", "user", "assistant", "analysis", "final", "\\n", "Hello"]);
        let inst = GptOssInstruct::new(tok);
        let sys = inst.system("Hello");
        // system() should use developer_prefix, not system prefix
        assert!(!sys.is_empty());
        assert_eq!(&sys[..inst.developer_prefix.len()], &inst.developer_prefix[..]);
    }

    #[test]
    fn user_starts_with_user_prefix() {
        let tok = make_tok(&["<|start|>", "<|message|>", "<|channel|>", "<|end|>", "<|endoftext|>", "<|return|>", "<|call|>", "system", "developer", "user", "assistant", "analysis", "final", "\\n", "Hello"]);
        let inst = GptOssInstruct::new(tok);
        let usr = inst.user("Hello");
        assert!(!usr.is_empty());
        assert_eq!(&usr[..inst.user_prefix.len()], &inst.user_prefix[..]);
    }
}
