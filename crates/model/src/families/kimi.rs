//! Kimi's `<|im_system|>` / `<|im_middle|>` / `<|im_end|>` conversation
//! format.
//!
//! Kimi K2 wrote it down first, which is why it lived in
//! `kimi_k2/chat.rs` — and the old `instruct::create` pointed
//! `"kimi_k2" | "kimi_k25" | "kimi_k3"` at that one constructor, which
//! is a sibling edge the isolation rule forbids the moment it stops
//! being a table cell and becomes a row's own answer. Two generations,
//! one format, so the words are here and `kimi_k2::chat` re-exports
//! them.
//!
//! It reads like ChatML and is not: the role marker is a triple
//! (`<|im_user|>` `user` `<|im_middle|>`) rather than a single opener,
//! and the assistant turn carries an EMPTY thinking block. A model
//! prompted in plain ChatML still answers, which is exactly why the
//! `_ =>` fallback that produced one was invisible.

use crate::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;

pub struct KimiInstruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl KimiInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let role_prefix = |role_token: &str, role_name: &str| {
            let mut tokens = encode(role_token);
            tokens.extend(encode(role_name));
            tokens.extend(encode("<|im_middle|>"));
            tokens
        };
        let stop_ids = ["<|im_end|>", "[EOS]"]
            .iter()
            .filter_map(|token| tokenizer.token_to_id(token))
            .collect();

        let mut generation_header = role_prefix("<|im_assistant|>", "assistant");
        generation_header.extend(encode("<think></think>"));

        Self {
            system_prefix: role_prefix("<|im_system|>", "system"),
            user_prefix: role_prefix("<|im_user|>", "user"),
            assistant_prefix: role_prefix("<|im_assistant|>", "assistant"),
            turn_suffix: encode("<|im_end|>"),
            generation_header,
            stop_ids,
            tokenizer,
        }
    }

    fn role_tokens(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn assistant_body(msg: &str) -> String {
        if msg.contains("<think>") {
            msg.to_string()
        } else {
            format!("<think></think>{msg}")
        }
    }
}

impl Instruct for KimiInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.system_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.assistant_prefix, &Self::assistant_body(msg))
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_header.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new()
    }

    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> {
        Vec::new()
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
        Box::new(NoopToolDecoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Arc<Tokenizer> {
        let words: Vec<String> = [
            "<|im_system|>",
            "<|im_user|>",
            "<|im_assistant|>",
            "<|im_middle|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "system",
            "user",
            "assistant",
            "Hi",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        Arc::new(Tokenizer::from_vocab(&words))
    }

    /// The role marker is a TRIPLE, which is the whole difference from
    /// the ChatML the `_ =>` arm used to hand this family.
    #[test]
    fn a_turn_is_the_role_triple_and_not_a_chatml_opener() {
        let tok = vocab();
        let k = KimiInstruct::new(tok.clone());
        assert_eq!(
            tok.decode(&k.user("Hi"), false),
            "<|im_user|>user<|im_middle|>Hi<|im_end|>",
        );
        assert_eq!(
            tok.decode(&k.system("Hi"), false),
            "<|im_system|>system<|im_middle|>Hi<|im_end|>",
        );
        assert!(
            !tok.decode(&k.user("Hi"), false).contains("<|im_start|>"),
            "ChatML's opener is a different token and a different protocol",
        );
    }

    /// The cue opens the thinking block and CLOSES it: this family's
    /// assistant turn begins after an empty `<think></think>`, and a
    /// generation header that omitted it would leave the model to open
    /// one itself.
    #[test]
    fn the_cue_carries_an_empty_thinking_block() {
        let tok = vocab();
        let k = KimiInstruct::new(tok.clone());
        assert_eq!(
            tok.decode(&k.cue(), false),
            "<|im_assistant|>assistant<|im_middle|><think></think>",
        );
        // And an assistant turn that already reasons keeps its own.
        assert_eq!(
            tok.decode(&k.assistant("<think>Hi</think>Hi"), false),
            "<|im_assistant|>assistant<|im_middle|><think>Hi</think>Hi<|im_end|>",
        );
        assert_eq!(
            tok.decode(&k.assistant("Hi"), false),
            "<|im_assistant|>assistant<|im_middle|><think></think>Hi<|im_end|>",
        );
    }

    /// The seal is what the vocabulary actually holds — `[EOS]` is not
    /// in every Kimi tokenizer, so the set is filtered rather than
    /// asserted, and this states that filtering.
    #[test]
    fn the_seal_is_the_stop_tokens_the_tokenizer_knows() {
        let tok = vocab();
        let k = KimiInstruct::new(tok.clone());
        assert_eq!(k.seal(), vec![tok.token_to_id("<|im_end|>").expect("in vocab")]);
        assert!(k.equip(&["tool".to_string()]).is_empty(), "no tool protocol here");
        assert!(k.answer("tool", "{}").is_empty());
    }
}
