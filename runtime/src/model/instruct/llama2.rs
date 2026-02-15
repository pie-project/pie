//! Llama 2 instruct implementation.
//!
//! Implements Llama 2 chat template features:
//! - [INST]...[/INST] for instructions
//! - <<SYS>>...<</SYS>> for system prompts (embedded in first instruction)
//!
//! Note: This implementation emits system prompts as standalone [INST] blocks
//! wrapping the system message, effectively creating a system turn.
//!
//! Reference: Llama 2 paper/HuggingFace template.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder, ReasoningEvent,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;
use crate::model::instruct::gemma2::{NoopReasoningDecoder, NoopToolDecoder};

// =============================================================================
// LlamaInstruct
// =============================================================================

pub struct LlamaInstruct {
    tokenizer: Arc<Tokenizer>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
    // Delimiters
    inst_start: Vec<u32>,
    inst_end: Vec<u32>,
    sys_wrapper_start: Vec<u32>,
    sys_wrapper_end: Vec<u32>,
}

impl LlamaInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["</s>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        // Safe encoding for [INST]
        let mut inst_start = encode("[INST]");
        inst_start.extend(encode(" "));
        
        let mut inst_end = encode(" ");
        inst_end.extend(encode("[/INST]"));

        // Pre-encode system wrappers
        let mut sys_wrapper_start = encode("<<SYS>>");
        sys_wrapper_start.extend(encode("\n"));

        let mut sys_wrapper_end = encode("\n");
        sys_wrapper_end.extend(encode("<</SYS>>"));
        sys_wrapper_end.extend(encode("\n\n"));

        Self {
            bos_token: encode("<s>"),
            stop_ids,
            inst_start,
            inst_end,
            sys_wrapper_start,
            sys_wrapper_end,
            tokenizer,
        }
    }
}

impl Instruct for LlamaInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        // Wrap system message in <<SYS>>...<</SYS>> and put inside [INST]...[/INST]
        let mut tokens = self.inst_start.clone();
        tokens.extend(&self.sys_wrapper_start);
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.sys_wrapper_end);
        tokens.extend(&self.inst_end);
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

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new() // No tool support in standard Llama 2
    }

    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> {
        Vec::new() // No tool support
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(LlamaChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.stop_ids.clone(),
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

// =============================================================================
// Chat Decoder
// =============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::model::tokenizer::Tokenizer;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn llama2() -> LlamaInstruct {
        let tok = make_tok(&[
            "<s>", "</s>", " ",
            "[INST]", "[/INST]",
            "<<SYS>>", "<</SYS>>",
            "Hello", "world",
            "system", "user", "assistant",
            "\n", "\n\n",
        ]);
        LlamaInstruct::new(tok)
    }

    #[test]
    fn system_format() {
        let inst = llama2();
        let tokens = inst.system("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        // Expect: [INST] <<SYS>>\nHello\n<</SYS>>\n\n [/INST]
        assert!(text.contains("[INST]"));
        assert!(text.contains("<<SYS>>"));
        assert!(text.contains("Hello"));
        assert!(text.contains("<</SYS>>"));
        assert!(text.contains("[/INST]"));
    }

    #[test]
    fn user_format() {
        let inst = llama2();
        let tokens = inst.user("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "[INST] Hello [/INST]");
    }
}
