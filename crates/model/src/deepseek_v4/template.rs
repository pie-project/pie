use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decoders::GenericChatDecoder;
use crate::instruct::{ChatDecoder, Instruct, Reasoning, ReasoningDecoder, ToolCall, ToolDecoder};

pub struct Template {
    tokenizer: Arc<Tokenizer>,
    bos: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    stop_ids: Vec<u32>,
    think_open: u32,
    think_close: u32,
    call_open: u32,
    call_close: u32,
    outputs_begin: Vec<u32>,
    outputs_end: Vec<u32>,
    output_begin: Vec<u32>,
    output_end: Vec<u32>,
}

impl Template {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let id = |s: &str| tokenizer.token_to_id(s).unwrap_or(0);
        let stop_ids: Vec<u32> = ["<｜end▁of▁sentence｜>", "<|EOT|>"]
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            bos: encode("<｜begin▁of▁sentence｜>"),
            user_prefix: encode("<｜User｜>"),
            assistant_prefix: encode("<｜Assistant｜>"),
            stop_ids,
            think_open: id("<think>"),
            think_close: id("</think>"),
            call_open: id("<｜tool▁call▁begin｜>"),
            call_close: id("<｜tool▁call▁end｜>"),
            outputs_begin: encode("<｜tool▁outputs▁begin｜>"),
            outputs_end: encode("<｜tool▁outputs▁end｜>"),
            output_begin: encode("<｜tool▁output▁begin｜>"),
            output_end: encode("<｜tool▁output▁end｜>"),
            tokenizer,
        }
    }

    fn turn(&self, prefix: &[u32], message: &str) -> Vec<u32> {
        let mut v = prefix.to_vec();
        v.extend(self.tokenizer.encode(message.trim()));
        v
    }

    fn strip_thinking(message: &str) -> &str {
        match message.rfind("</think>") {
            Some(at) => &message[at + "</think>".len()..],
            None => message,
        }
    }

    fn tool_prompt(tools: &[String]) -> String {
        let mut prompt = String::from(
            "You are a helpful assistant with tool calling capabilities. \
             When a tool call is needed, you MUST use the following format to issue the call:\n\
             <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNCTION_NAME\n\
             ```json\n\
             {\"param\": \"value\"}\n\
             ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n\n\
             Make sure the JSON is valid.\n\n## Tools\n\n\
             You have the following functions available:\n",
        );
        for tool in tools {
            prompt.push_str("\n```json\n");
            prompt.push_str(tool);
            prompt.push_str("\n```\n");
        }
        prompt
    }
}

impl Instruct for Template {
    fn system(&self, message: &str) -> Vec<u32> {
        self.tokenizer.encode(message.trim())
    }

    fn first_user(&self, message: &str) -> Vec<u32> {
        let mut v = self.bos.clone();
        v.extend(self.turn(&self.user_prefix, message));
        v
    }

    fn user(&self, message: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, message)
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut v = self.bos.clone();
        v.extend(self.system(system));
        v.extend(self.turn(&self.user_prefix, user));
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        let mut v = self.turn(&self.assistant_prefix, Self::strip_thinking(message));
        v.extend(&self.stop_ids[..1]);
        v
    }

    fn cue(&self) -> Vec<u32> {
        self.assistant_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        self.system(&Self::tool_prompt(tools))
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        let mut v = self.outputs_begin.clone();
        v.extend(&self.output_begin);
        v.extend(self.tokenizer.encode(value));
        v.extend(&self.output_end);
        v.extend(&self.outputs_end);
        v
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder {
            tokenizer: self.tokenizer.clone(),
            open: self.think_open,
            close: self.think_close,
            thinking: false,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(R1ToolDecoder {
            tokenizer: self.tokenizer.clone(),
            open: self.call_open,
            close: self.call_close,
            pending: Vec::new(),
            inside: false,
        })
    }
}

struct ThinkingDecoder {
    tokenizer: Arc<Tokenizer>,
    open: u32,
    close: u32,
    thinking: bool,
}

impl ReasoningDecoder for ThinkingDecoder {
    fn push(&mut self, token: u32) -> Option<Reasoning> {
        if token == self.open {
            self.thinking = true;
            return None;
        }
        if token == self.close {
            self.thinking = false;
            return None;
        }
        let text = self.tokenizer.decode(&[token], true);
        if self.thinking {
            Some(Reasoning::Thinking(text))
        } else {
            Some(Reasoning::Answer(text))
        }
    }
}

struct R1ToolDecoder {
    tokenizer: Arc<Tokenizer>,
    open: u32,
    close: u32,
    pending: Vec<u32>,
    inside: bool,
}

impl ToolDecoder for R1ToolDecoder {
    fn push(&mut self, token: u32) -> Option<ToolCall> {
        if token == self.open {
            self.inside = true;
            self.pending.clear();
            return None;
        }
        if !self.inside {
            return None;
        }
        if token != self.close {
            self.pending.push(token);
            return None;
        }
        self.inside = false;
        let body = self.tokenizer.decode(&std::mem::take(&mut self.pending), true);
        let (header, rest) = body.split_once("\n```json\n")?;
        let arguments = rest.split("\n```").next()?.to_string();
        let name = match header.rsplit_once("<｜tool▁sep｜>") {
            Some((_, name)) => name,
            None => header,
        };
        Some(ToolCall {
            name: name.trim().to_string(),
            arguments,
        })
    }
}
