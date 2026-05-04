//! Gemma 2/3 instruct implementation.
//!
//! Uses <start_of_turn>/<end_of_turn> delimiters.
//! Gemma has no native system role. `system()` returns bare text with a
//! trailing newline; the caller should prepend it to the first user turn.
//! No thinking or tool-use support.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder,
    Instruct,
    ReasoningDecoder,
    ToolDecoder,
};
use crate::model::instruct::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::model::tokenizer::Tokenizer;

pub struct GemmaInstruct {
    tokenizer: Arc<Tokenizer>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl GemmaInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_strs = ["<end_of_turn>", "<eos>"];
        let stop_ids: Vec<u32> = stop_strs
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let start_turn = encode("<start_of_turn>");
        let end_turn = encode("<end_of_turn>");
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = start_turn.clone();
            v.extend(encode(role));
            v.extend(&newline);
            v
        };

        let mut turn_suffix = end_turn;
        turn_suffix.extend(&newline);

        Self {
            user_prefix: make_prefix("user"),
            model_prefix: make_prefix("model"),
            turn_suffix,
            bos_token: encode("<bos>"),
            stop_ids,
            tokenizer,
        }
    }
}

impl Instruct for GemmaInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        // Gemma has no native system role. The reference template prepends
        // system content (plus a newline separator) to the first user
        // message body. Return bare text; the caller should embed this
        // inside the first user turn rather than emitting a standalone turn.
        let mut tokens = self.tokenizer.encode(msg);
        tokens.extend(self.tokenizer.encode("\n"));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.user_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.model_prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        self.model_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new() // No tool support
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

    fn gemma() -> GemmaInstruct {
        let tok = make_tok(&[
            "<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>",
            "user", "model", "\n", "Hello",
        ]);
        GemmaInstruct::new(tok)
    }

    #[test]
    fn has_correct_stop_tokens() {
        let stop = gemma().seal();
        assert!(stop.contains(&1)); // <end_of_turn>
        assert!(stop.contains(&2)); // <eos>
    }

    #[test]
    fn assistant_uses_model_prefix() {
        let inst = gemma();
        let tokens = inst.assistant("Hello");
        assert!(!tokens.is_empty());
        assert_eq!(&tokens[..inst.model_prefix.len()], &inst.model_prefix[..]);
    }

    #[test]
    fn equip_is_noop() {
        assert!(gemma().equip(&["tool".to_string()]).is_empty());
    }

    #[test]
    fn answer_is_noop() {
        assert!(gemma().answer("fn1", "42").is_empty());
    }

    #[test]
    fn cue_matches_generation_header() {
        let inst = gemma();
        assert_eq!(inst.cue(), inst.model_prefix);
    }

    #[test]
    fn full_conversation() {
        let inst = gemma();
        // Gemma: system() returns bare text + \n (caller embeds in first user turn)
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "Hello\n\
             <start_of_turn>user\nHello<end_of_turn>\n\
             <start_of_turn>model\nHello<end_of_turn>\n\
             <start_of_turn>user\nHello<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn system_is_bare_text() {
        let inst = gemma();
        let tokens = inst.system("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "Hello\n");
        // Must NOT contain turn delimiters
        assert!(!text.contains("<start_of_turn>"));
    }
}
