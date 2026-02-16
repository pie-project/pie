//! Llama 2 instruct implementation.
//!
//! Implements Llama 2 chat template features:
//! - [INST]...[/INST] for instructions
//! - <<SYS>>...<</SYS>> for system prompts (embedded in first instruction)
//!
//! `system()` returns the bare <<SYS>> wrapper. The caller should embed
//! this at the start of the first user turn's [INST] content.
//!
//! Reference: Llama 2 paper/HuggingFace template.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder,
    Instruct,
    ReasoningDecoder,
    ToolDecoder,
};
use crate::model::instruct::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::model::tokenizer::Tokenizer;

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
        // Llama 2 reference: <<SYS>> block is embedded inside the first
        // [INST]...[/INST] user turn. Return the inner wrapper; the caller
        // should embed this at the start of the first user turn content.
        let mut tokens = self.sys_wrapper_start.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.sys_wrapper_end);
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
        Box::new(GenericChatDecoder::new(self.tokenizer.clone(), self.stop_ids.clone()))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
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
        // Returns <<SYS>> wrapper without [INST]; caller embeds in first user turn
        assert!(!text.contains("[INST]"));
        assert!(text.contains("<<SYS>>"));
        assert!(text.contains("Hello"));
        assert!(text.contains("<</SYS>>"));
    }

    #[test]
    fn user_format() {
        let inst = llama2();
        let tokens = inst.user("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "[INST] Hello [/INST]");
    }

    #[test]
    fn full_conversation() {
        let inst = llama2();
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<<SYS>>\nHello\n<</SYS>>\n\n\
             [INST] Hello [/INST]\
             Hello</s>\
             [INST] Hello [/INST]"
        );
    }

    #[test]
    fn equip_is_noop() {
        assert!(llama2().equip(&["tool".to_string()]).is_empty());
    }

    #[test]
    fn answer_is_noop() {
        assert!(llama2().answer("fn1", "42").is_empty());
    }
}
