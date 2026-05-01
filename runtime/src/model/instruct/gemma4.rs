//! Gemma 4 instruct implementation.
//!
//! Diverges from Gemma 2/3: Gemma 4 introduces a single-token turn
//! delimiter pair (`<|turn>` id 105 / `<turn|>` id 106) instead of the
//! multi-piece `<start_of_turn>` / `<end_of_turn>`. The chat template
//! shape is otherwise the same:
//!
//!   <bos><|turn>user\n{user}<turn|>\n<|turn>model\n
//!
//! No native system role; like Gemma 2/3, callers should fold any
//! system prompt into the first user turn.

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder,
    Instruct,
    ReasoningDecoder,
    ToolDecoder,
};
use crate::model::instruct::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::model::tokenizer::Tokenizer;

pub struct Gemma4Instruct {
    tokenizer: Arc<Tokenizer>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Gemma4Instruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
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
            user_prefix: make_prefix("user"),
            model_prefix: make_prefix("model"),
            turn_suffix,
            bos_token: encode("<bos>"),
            stop_ids,
            tokenizer,
        }
    }
}

impl Instruct for Gemma4Instruct {
    fn system(&self, message: &str) -> Vec<u32> {
        // Gemma-4 has no system role. Fall back to bare text so callers
        // can prepend it to the first user turn.
        let mut v = self.tokenizer.encode(message);
        v.extend(self.tokenizer.encode("\n"));
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        let mut v = self.bos_token.clone();
        v.extend(&self.user_prefix);
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.model_prefix.clone();
        v.extend(self.tokenizer.encode(message));
        v.extend(&self.turn_suffix);
        v
    }

    fn cue(&self) -> Vec<u32> {
        self.model_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, _tools: &[String]) -> Vec<u32> { Vec::new() }
    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> { Vec::new() }

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
