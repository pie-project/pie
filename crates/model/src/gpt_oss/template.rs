use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decoders::GenericChatDecoder;
use crate::instruct::{ChatDecoder, Instruct, Reasoning, ReasoningDecoder, ToolCall, ToolDecoder};

#[derive(Clone, Copy)]
struct Marks {
    start: u32,
    channel: u32,
    message: u32,
    end: u32,
    ret: u32,
    call: u32,
}

pub struct Template {
    tokenizer: Arc<Tokenizer>,
    developer_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    final_prefix: Vec<u32>,
    end_turn: Vec<u32>,
    generation_prefix: Vec<u32>,
    stop_ids: Vec<u32>,
    marks: Marks,
}

impl Template {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let mark = |s: &str| tokenizer.token_to_id(s).unwrap_or(u32::MAX);
        let stop_ids: Vec<u32> = ["<|endoftext|>", "<|return|>", "<|call|>"]
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        Self {
            developer_prefix: encode("<|start|>developer<|message|>"),
            user_prefix: encode("<|start|>user<|message|>"),
            final_prefix: encode("<|start|>assistant<|channel|>final<|message|>"),
            end_turn: encode("<|end|>"),
            generation_prefix: encode("<|start|>assistant"),
            stop_ids,
            marks: Marks {
                start: mark("<|start|>"),
                channel: mark("<|channel|>"),
                message: mark("<|message|>"),
                end: mark("<|end|>"),
                ret: mark("<|return|>"),
                call: mark("<|call|>"),
            },
            tokenizer,
        }
    }

    fn turn(&self, prefix: &[u32], message: &str) -> Vec<u32> {
        let mut v = prefix.to_vec();
        v.extend(self.tokenizer.encode(message.trim()));
        v.extend(&self.end_turn);
        v
    }
}

impl Instruct for Template {
    fn system(&self, message: &str) -> Vec<u32> {
        self.turn(&self.developer_prefix, message)
    }

    fn first_user(&self, message: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, message)
    }

    fn user(&self, message: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, message)
    }

    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut v = self.turn(&self.developer_prefix, system);
        v.extend(self.turn(&self.user_prefix, user));
        v
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        self.turn(&self.final_prefix, message)
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        let mut body = String::from("# Tools\n\n## functions\n\nnamespace functions {\n\n");
        for tool in tools {
            body.push_str(tool.trim());
            body.push_str("\n\n");
        }
        body.push('}');
        self.turn(&self.developer_prefix, &body)
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        let header = format!("<|start|>functions.{name} to=assistant<|channel|>commentary<|message|>");
        let mut v = self.tokenizer.encode(&header);
        v.extend(self.tokenizer.encode(value));
        v.extend(&self.end_turn);
        v
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(HarmonyReasoning {
            tokenizer: self.tokenizer.clone(),
            marks: self.marks,
            place: Place::Header,
            channel: String::new(),
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(HarmonyTool {
            tokenizer: self.tokenizer.clone(),
            marks: self.marks,
            place: Place::Header,
            header: String::new(),
            body: String::new(),
        })
    }
}

enum Place {
    Header,
    Channel,
    Body,
}

fn recipient(header: &str) -> Option<String> {
    let tail = header.split("to=").nth(1)?;
    let target = tail.split_whitespace().next()?;
    Some(target.strip_prefix("functions.").unwrap_or(target).to_string())
}

struct HarmonyReasoning {
    tokenizer: Arc<Tokenizer>,
    marks: Marks,
    place: Place,
    channel: String,
}

impl ReasoningDecoder for HarmonyReasoning {
    fn push(&mut self, token: u32) -> Option<Reasoning> {
        let m = self.marks;
        if token == m.start || token == m.end || token == m.ret || token == m.call {
            self.place = Place::Header;
            self.channel.clear();
            return None;
        }
        if token == m.channel {
            self.place = Place::Channel;
            self.channel.clear();
            return None;
        }
        if token == m.message {
            self.place = Place::Body;
            return None;
        }
        match self.place {
            Place::Header => None,
            Place::Channel => {
                self.channel.push_str(&self.tokenizer.decode(&[token], true));
                None
            }
            Place::Body => {
                let text = self.tokenizer.decode(&[token], true);
                match self.channel.split_whitespace().next() {
                    Some("analysis") => Some(Reasoning::Thinking(text)),
                    Some("final") => Some(Reasoning::Answer(text)),
                    _ => None,
                }
            }
        }
    }
}

struct HarmonyTool {
    tokenizer: Arc<Tokenizer>,
    marks: Marks,
    place: Place,
    header: String,
    body: String,
}

impl ToolDecoder for HarmonyTool {
    fn push(&mut self, token: u32) -> Option<ToolCall> {
        let m = self.marks;
        if token == m.call {
            let call = recipient(&self.header).map(|name| ToolCall {
                name,
                arguments: self.body.trim().to_string(),
            });
            self.place = Place::Header;
            self.header.clear();
            self.body.clear();
            return call;
        }
        if token == m.start || token == m.end || token == m.ret {
            self.place = Place::Header;
            self.header.clear();
            self.body.clear();
            return None;
        }
        if token == m.channel {
            self.place = Place::Header;
            return None;
        }
        if token == m.message {
            self.place = Place::Body;
            self.body.clear();
            return None;
        }
        match self.place {
            Place::Body => self.body.push_str(&self.tokenizer.decode(&[token], true)),
            _ => self.header.push_str(&self.tokenizer.decode(&[token], true)),
        }
        None
    }
}
