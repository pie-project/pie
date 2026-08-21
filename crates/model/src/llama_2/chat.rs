use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};
use crate::shared::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use std::sync::Arc;
use tokenizer::Tokenizer;

pub struct LlamaInstruct {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,

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

        let mut inst_start = encode("[INST]");
        inst_start.extend(encode(" "));

        let mut inst_end = encode(" ");
        inst_end.extend(encode("[/INST]"));

        let mut sys_wrapper_start = encode("<<SYS>>");
        sys_wrapper_start.extend(encode("\n"));

        let mut sys_wrapper_end = encode("\n");
        sys_wrapper_end.extend(encode("<</SYS>>"));
        sys_wrapper_end.extend(encode("\n\n"));

        Self {
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
