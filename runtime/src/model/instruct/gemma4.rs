//! Gemma 4 instruct implementation.
//!
//! Diverges from Gemma 2/3: Gemma 4 introduces a single-token turn
//! delimiter pair (`<|turn>` id 105 / `<turn|>` id 106) instead of the
//! multi-piece `<start_of_turn>` / `<end_of_turn>`.
//!
//! Gemma 4 also has a native system/developer block:
//!
//!   <bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n

use crate::model::instruct::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::model::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::model::tokenizer::Tokenizer;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Variant {
    Gemma4,
    Gemma4Text,
}

pub struct Gemma4Instruct {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Gemma4Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self::for_variant(tokenizer, Gemma4Variant::Gemma4)
    }

    pub fn for_variant(tokenizer: Arc<Tokenizer>, _variant: Gemma4Variant) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        // `<turn|>` (closing) + `<eos>` are both terminal — generation
        // stops at either. The runtime's `seal()` returns this list.
        let stop_strs = ["<turn|>", "<eos>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        // Gemma-4's tokenizer treats `<|turn>` and `<turn|>` as single
        // added tokens (ids 105 and 106 on the E2B vocab); `encode`
        // returns a 1-element vector for each. We assemble the
        // role-prefixes by token concatenation, matching how
        // GemmaInstruct does it for `<start_of_turn>`.
        let open_turn = encode("<|turn>");
        let close_turn = encode("<turn|>");
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = open_turn.clone();
            v.extend(encode(role));
            v.extend(&newline);
            v
        };

        let mut turn_suffix = close_turn;
        turn_suffix.extend(&newline);

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            model_prefix: make_prefix("model"),
            turn_suffix,
            bos_token: encode("<bos>"),
            stop_ids,
            tokenizer,
        }
    }

    fn encode_trimmed(&self, message: &str) -> Vec<u32> {
        self.tokenizer.encode(message.trim())
    }
}

impl Instruct for Gemma4Instruct {
    fn system(&self, message: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.system_prefix);
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn first_user(&self, message: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.user_prefix);
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        let mut v = self.user_prefix.clone();
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.system_prefix);
        v.extend(self.encode_trimmed(system));
        v.extend(&self.turn_suffix);
        v.extend(&self.user_prefix);
        v.extend(self.encode_trimmed(user));
        v.extend(&self.turn_suffix);
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.model_prefix.clone();
        v.extend(self.encode_trimmed(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn cue(&self) -> Vec<u32> {
        self.model_prefix.clone()
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

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn gemma4() -> Gemma4Instruct {
        let tok = make_tok(&[
            "<|turn>", "<turn|>", "<eos>", "<bos>", "system", "user", "model", "\n", "Sys",
            "Hello", "Ok",
        ]);
        Gemma4Instruct::new(tok)
    }

    #[test]
    fn system_user_uses_native_system_turn_once() {
        let inst = gemma4();
        let mut tokens = inst.system_user("Sys", "Hello");
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<bos><|turn>system\nSys<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn first_user_starts_with_bos() {
        let inst = gemma4();
        let mut tokens = inst.first_user("Hello");
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<bos><|turn>user\nHello<turn|>\n<|turn>model\n");
    }

    #[test]
    fn later_user_omits_bos() {
        let inst = gemma4();
        let tokens = inst.user("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<|turn>user\nHello<turn|>\n");
    }

    #[test]
    fn assistant_uses_model_role() {
        let inst = gemma4();
        let tokens = inst.assistant("Ok");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<|turn>model\nOk<turn|>\n");
    }

    #[test]
    fn cached_gemma4_matches_hf_benchmark_prompt_ids() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let tokenizer_path = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots")
            .read_dir()
            .ok()
            .and_then(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path().join("tokenizer.json"))
                    .find(|path| path.exists())
            });
        let Some(tokenizer_path) = tokenizer_path else {
            return;
        };
        let Ok(tokenizer) = Tokenizer::from_file(&tokenizer_path) else {
            return;
        };
        let inst = Gemma4Instruct::new(Arc::new(tokenizer));
        let mut tokens = inst.system_user(
            "You are a helpful benchmarking assistant.",
            "Write a short story about a robot. (Request #0)",
        );
        tokens.extend(inst.cue());
        assert_eq!(
            tokens,
            vec![
                2, 105, 9731, 107, 3048, 659, 496, 11045, 141657, 16326, 236761, 106, 107, 105,
                2364, 107, 6974, 496, 2822, 3925, 1003, 496, 16775, 236761, 568, 3932, 997, 236771,
                236768, 106, 107, 105, 4368, 107,
            ]
        );
    }
}
