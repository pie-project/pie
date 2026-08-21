
use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;


pub struct Template {
    tokenizer: Arc<Tokenizer>,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    model_prefix: Vec<u32>,
    turn_suffix: Vec<u32>,
    bos_token: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl Template {
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

impl Instruct for Template {
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
