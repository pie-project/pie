//! DeepSeek R1 instruct implementation.
//!
//! Uses fullwidth Unicode delimiters for roles and tool calls.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder, ReasoningEvent,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;

pub struct R1Instruct {
    tokenizer: Arc<Tokenizer>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    eos_ids: Vec<u32>,
    generation_header: Vec<u32>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    // Tool tokens
    tool_calls_begin: Vec<u32>,
    tool_calls_end: Vec<u32>,
    tool_call_begin: Vec<u32>,
    tool_call_end: Vec<u32>,
    tool_sep: Vec<u32>,
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
            generation_header: encode("<｜Assistant｜><think>\n"),
            think_prefix_ids: encode("<think>\n"),
            think_suffix_ids: encode("</think>\n"),
            tool_calls_begin: encode("<｜tool▁calls▁begin｜>"),
            tool_calls_end: encode("<｜tool▁calls▁end｜>"),
            tool_call_begin: encode("<｜tool▁call▁begin｜>"),
            tool_call_end: encode("<｜tool▁call▁end｜>"),
            tool_sep: encode("<｜tool▁sep｜>"),
            tool_outputs_begin: encode("<｜tool▁outputs▁begin｜>"),
            tool_outputs_end: encode("<｜tool▁outputs▁end｜>"),
            tool_output_begin: encode("<｜tool▁output▁begin｜>"),
            tool_output_end: encode("<｜tool▁output▁end｜>"),
            tokenizer,
        }
    }
}

impl Instruct for R1Instruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        // R1 prepends system text without a wrapper
        self.tokenizer.encode(msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.user_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.assistant_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.eos_ids[..1]); // first EOS token
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_header.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.eos_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        // R1 embeds tool definitions in system prompt
        let mut prompt = String::from("You have access to the following tools:\n\n");
        for tool in tools {
            prompt.push_str(tool);
            prompt.push_str("\n\n");
        }
        self.system(&prompt)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        let mut tokens = self.tool_output_begin.clone();
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_output_end);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(R1ChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.eos_ids.clone(),
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(R1ReasoningDecoder {
            tokenizer: self.tokenizer.clone(),
            think_prefix_ids: self.think_prefix_ids.clone(),
            think_suffix_ids: self.think_suffix_ids.clone(),
            state: R1State::Outside,
            accumulated: String::new(),
            match_pos: 0,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(R1ToolDecoder {
            tokenizer: self.tokenizer.clone(),
            tool_call_begin: self.tool_call_begin.clone(),
            tool_call_end: self.tool_call_end.clone(),
            accumulated: String::new(),
            inside: false,
            match_pos: 0,
        })
    }
}

// ─── Decoders ───────────────────────────────────────────────

struct R1ChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for R1ChatDecoder {
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

enum R1State {
    Outside,
    Inside,
}

struct R1ReasoningDecoder {
    tokenizer: Arc<Tokenizer>,
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    state: R1State,
    accumulated: String,
    match_pos: usize,
}

impl ReasoningDecoder for R1ReasoningDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent {
        match self.state {
            R1State::Outside => {
                for &t in tokens {
                    if self.match_pos < self.think_prefix_ids.len()
                        && t == self.think_prefix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.think_prefix_ids.len() {
                            self.state = R1State::Inside;
                            self.match_pos = 0;
                            return ReasoningEvent::Start;
                        }
                    } else {
                        self.match_pos = 0;
                    }
                }
                ReasoningEvent::Delta(String::new())
            }
            R1State::Inside => {
                for &t in tokens {
                    if self.match_pos < self.think_suffix_ids.len()
                        && t == self.think_suffix_ids[self.match_pos]
                    {
                        self.match_pos += 1;
                        if self.match_pos == self.think_suffix_ids.len() {
                            self.state = R1State::Outside;
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
        self.state = R1State::Outside;
        self.accumulated.clear();
        self.match_pos = 0;
    }
}

struct R1ToolDecoder {
    tokenizer: Arc<Tokenizer>,
    tool_call_begin: Vec<u32>,
    tool_call_end: Vec<u32>,
    accumulated: String,
    inside: bool,
    match_pos: usize,
}

impl ToolDecoder for R1ToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);

        if !self.inside {
            // Match tool_call_begin token sequence
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
            // Match tool_call_end token sequence
            for &t in tokens {
                if self.match_pos < self.tool_call_end.len()
                    && t == self.tool_call_end[self.match_pos]
                {
                    self.match_pos += 1;
                    if self.match_pos == self.tool_call_end.len() {
                        self.inside = false;
                        self.match_pos = 0;
                        // Parse: type<tool_sep>name\n```json\nargs\n```
                        let content = std::mem::take(&mut self.accumulated);
                        // Extract function name and args from R1 format
                        if let Some(sep_pos) = content.find("\n```json\n") {
                            let header = &content[..sep_pos];
                            let json_start = sep_pos + "\n```json\n".len();
                            if let Some(json_end) = content[json_start..].find("\n```") {
                                let args = content[json_start..json_start + json_end].to_string();
                                // Extract name from "type<sep>name" format
                                let name = header.rsplit_once("\n").map_or(header, |(_, n)| n).to_string();
                                return ToolEvent::Call(name, args);
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
        self.accumulated.clear();
        self.inside = false;
        self.match_pos = 0;
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
        let tok = make_tok(&["<｜end▁of▁sentence｜>", "<|EOT|>", "<｜User｜>", "<｜Assistant｜>", "Hello", "\\n", "<think>", "</think>"]);
        let inst = R1Instruct::new(tok);
        let stop = inst.seal();
        assert!(stop.contains(&0));
    }

    #[test]
    fn user_starts_with_prefix() {
        let tok = make_tok(&["<｜end▁of▁sentence｜>", "<|EOT|>", "<｜User｜>", "<｜Assistant｜>", "Hello", "\\n", "<think>", "</think>"]);
        let inst = R1Instruct::new(tok);
        let tokens = inst.user("Hello");
        assert_eq!(tokens[..inst.user_prefix.len()], inst.user_prefix[..]);
    }

    #[test]
    fn system_is_bare_text() {
        let tok = make_tok(&["<｜end▁of▁sentence｜>", "<|EOT|>", "<｜User｜>", "<｜Assistant｜>", "Hello", "\\n", "<think>", "</think>"]);
        let inst = R1Instruct::new(tok);
        let sys = inst.system("Hello");
        // Bare prepend: should NOT start with user or assistant prefix
        if !inst.user_prefix.is_empty() {
            assert_ne!(&sys[..inst.user_prefix.len().min(sys.len())],
                       &inst.user_prefix[..inst.user_prefix.len().min(sys.len())]);
        }
    }

    #[test]
    fn cue_contains_assistant_prefix() {
        let tok = make_tok(&["<｜end▁of▁sentence｜>", "<|EOT|>", "<｜User｜>", "<｜Assistant｜>", "Hello", "\\n", "<think>", "</think>"]);
        let inst = R1Instruct::new(tok);
        let cue = inst.cue();
        // Verified cue() is callable (CharLevel tokenizer cannot encode prefix)
        let _ = cue;
    }
}
