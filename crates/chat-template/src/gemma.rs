use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, special, specials};

pub const STOP_TOKENS: &[&str] = &["<turn|>", "<eos>"];

pub struct Gemma {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    cue: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos: u32,
    stop_ids: Vec<u32>,
}

impl Gemma {
    #[must_use]
    /// `thinking`: the reference template's cue closes an empty thought
    /// channel when the turn does not think (26B-A4B and 31B; E4B's does not).
    pub fn new(tokenizer: Arc<Tokenizer>, thinking: bool) -> Self {
        let stop_ids = specials(&tokenizer, STOP_TOKENS);

        let open_turn = special(&tokenizer, "<|turn>");
        let close_turn = special(&tokenizer, "<turn|>");
        let newline = tokenizer.encode("\n");

        let header = |role: &str| -> Vec<u32> {
            let mut tokens = vec![open_turn];
            tokens.extend(tokenizer.encode(role));
            tokens.extend(&newline);
            tokens
        };

        let mut turn_suffix = vec![close_turn];
        turn_suffix.extend(&newline);

        let mut cue = header("model");
        if thinking {
            cue.push(special(&tokenizer, "<|channel>"));
            cue.extend(tokenizer.encode("thought"));
            cue.extend(&newline);
            cue.push(special(&tokenizer, "<channel|>"));
        }

        Self {
            system_prefix: header("system"),
            user_prefix: header("user"),
            model_prefix: header("model"),
            cue,
            turn_suffix,
            bos: special(&tokenizer, "<bos>"),
            stop_ids,
            tokenizer,
        }
    }

    fn turn(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg.trim()));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn opening(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.turn(prefix, msg));
        tokens
    }
}

impl Instruct for Gemma {
    fn prefix(&self) -> Vec<u32> {
        vec![self.bos]
    }

    fn system(&self, msg: &str) -> Vec<u32> {
        self.opening(&self.system_prefix, msg)
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.opening(&self.user_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, msg)
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.opening(&self.system_prefix, system);
        tokens.extend(self.user(user));
        tokens
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.model_prefix, msg)
    }

    fn cue(&self) -> Vec<u32> {
        self.cue.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
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
