use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma3Variant {
    Gemma3,
    Gemma3Text,
    Gemma3n,
    Gemma3nText,
}

pub struct Gemma3Instruct {
    tokenizer: Arc<Tokenizer>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Gemma3Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self::for_variant(tokenizer, Gemma3Variant::Gemma3)
    }

    pub fn for_variant(tokenizer: Arc<Tokenizer>, _variant: Gemma3Variant) -> Self {
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
            let mut tokens = start_turn.clone();
            tokens.extend(encode(role));
            tokens.extend(&newline);
            tokens
        };

        let mut turn_suffix = end_turn;
        turn_suffix.extend(&newline);

        let user_prefix = make_prefix("user");
        let model_prefix = make_prefix("model");
        let bos_token = encode("<bos>");

        Self {
            tokenizer,
            user_prefix,
            model_prefix,
            turn_suffix,
            bos_token,
            stop_ids,
        }
    }

    fn encode_trimmed(&self, message: &str) -> Vec<u32> {
        self.tokenizer.encode(message.trim())
    }
}

impl Instruct for Gemma3Instruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(msg);
        tokens.extend(self.tokenizer.encode("\n"));
        tokens
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.bos_token.clone();
        tokens.extend(&self.user_prefix);
        tokens.extend(self.encode_trimmed(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.user_prefix.clone();
        tokens.extend(self.encode_trimmed(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.bos_token.clone();
        tokens.extend(&self.user_prefix);
        tokens.extend(self.tokenizer.encode(system));
        tokens.extend(self.tokenizer.encode("\n"));
        tokens.extend(self.encode_trimmed(user));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.model_prefix.clone();
        tokens.extend(self.encode_trimmed(msg));
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
