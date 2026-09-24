use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, special, specials};

/// Kimi-K3's XTML format (`encoding_k3.py` beside the checkpoint): every
/// message is `<|open|>message role="…"<|sep|>` … `<|close|>message<|sep|>`
/// `<|end_of_msg|>`; an assistant message wraps its text in a `response`
/// element (and, when thinking, a `think` element before it); the cue is an
/// opened assistant message with an opened `response`.
pub const STOP_TOKENS: &[&str] = &["<|end_of_msg|>", "[EOS]", "[EOT]"];

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const END_OF_MSG: &str = "<|end_of_msg|>";

pub struct Kimi3 {
    tokenizer: Arc<Tokenizer>,
    open: u32,
    close: u32,
    sep: u32,
    end_of_msg: u32,
    stop_ids: Vec<u32>,
}

impl Kimi3 {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            open: special(&tokenizer, OPEN),
            close: special(&tokenizer, CLOSE),
            sep: special(&tokenizer, SEP),
            end_of_msg: special(&tokenizer, END_OF_MSG),
            stop_ids: specials(&tokenizer, STOP_TOKENS),
            tokenizer,
        }
    }

    fn open_tag(&self, tag: &str, attrs: &[(&str, &str)]) -> Vec<u32> {
        let mut tokens = vec![self.open];
        tokens.extend(self.tokenizer.encode(tag));
        for (key, value) in attrs {
            let value = value.replace('&', "&amp;").replace('"', "&quot;");
            tokens.extend(self.tokenizer.encode(&format!(" {key}")));
            tokens.extend(self.tokenizer.encode("=\""));
            tokens.extend(self.tokenizer.encode(&value));
            tokens.extend(self.tokenizer.encode("\""));
        }
        tokens.push(self.sep);
        tokens
    }

    fn close_tag(&self, tag: &str) -> Vec<u32> {
        let mut tokens = vec![self.close];
        tokens.extend(self.tokenizer.encode(tag));
        tokens.push(self.sep);
        tokens
    }

    fn message(&self, role: &str, body: Vec<u32>) -> Vec<u32> {
        let mut tokens = self.open_tag("message", &[("role", role)]);
        tokens.extend(body);
        tokens.extend(self.close_tag("message"));
        tokens.push(self.end_of_msg);
        tokens
    }
}

impl Instruct for Kimi3 {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.message("system", self.tokenizer.encode(msg))
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.message("user", self.tokenizer.encode(msg))
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut body = self.open_tag("response", &[]);
        body.extend(self.tokenizer.encode(msg));
        body.extend(self.close_tag("response"));
        self.message("assistant", body)
    }

    fn cue(&self) -> Vec<u32> {
        let mut tokens = self.open_tag("message", &[("role", "assistant")]);
        tokens.extend(self.open_tag("response", &[]));
        tokens
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
