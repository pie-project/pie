use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent};
use crate::shared::decoders::{GenericChatDecoder, ThinkingDecoder, keep_tail};
use std::sync::Arc;
use tokenizer::{Tokenizer, TokenizerDecoder};

pub struct OlmoInstruct {
    tokenizer: Arc<Tokenizer>,
    im_start: Vec<u32>,
    im_end: Vec<u32>,
    newline: Vec<u32>,

    system_role: Vec<u32>,
    user_role: Vec<u32>,
    assistant_role: Vec<u32>,
    environment_role: Vec<u32>,

    think_start: Vec<u32>,
    think_end: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl OlmoInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);

        let im_start = encode("<|im_start|>");
        let im_end = encode("<|im_end|>");
        let newline = encode("\n");
        let eos_token = encode("<|endoftext|>");

        let mut stop_ids = im_end.clone();
        stop_ids.extend(&eos_token);

        Self {
            im_start,
            im_end,
            newline,
            system_role: encode("system"),
            user_role: encode("user"),
            assistant_role: encode("assistant"),
            environment_role: encode("environment"),
            think_start: encode("<think>"),
            think_end: encode("</think>"),
            stop_ids,
            tokenizer,
        }
    }

    fn wrap(&self, role: &[u32], content: &str) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(role);
        tokens.extend(&self.newline);
        tokens.extend(self.tokenizer.encode(content));
        tokens.extend(&self.im_end);
        tokens.extend(&self.newline);
        tokens
    }
}

impl Instruct for OlmoInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.system_role, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.user_role, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(&self.assistant_role);
        tokens.extend(&self.newline);
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.im_end);
        tokens.extend(&self.newline);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(&self.assistant_role);
        tokens.extend(&self.newline);
        tokens.extend(&self.think_start);
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        let preamble = "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024. ";
        let mut msg = preamble.to_string();
        msg.push_str("<functions>");
        msg.push_str(&tools.join("\n"));
        msg.push_str("</functions>");

        self.system(&msg)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.wrap(&self.environment_role, value)
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {

        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {

        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            vec![],
            self.think_end.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(OlmoToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            state: ToolState::Outside,
        })
    }
}

struct OlmoToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    state: ToolState,
}

#[derive(Debug, PartialEq)]
enum ToolState {
    Outside,
    Inside,
}

impl ToolDecoder for OlmoToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        loop {
            match self.state {
                ToolState::Outside => {
                    if let Some(pos) = self.accumulated.find("<function_calls>") {
                        self.accumulated =
                            self.accumulated[pos + "<function_calls>".len()..].to_string();
                        self.state = ToolState::Inside;
                        continue;
                    }
                    if self.accumulated.len() > 200 {
                        self.accumulated = keep_tail(&self.accumulated, 50).to_string();
                    }
                    return ToolEvent::Start;
                }
                ToolState::Inside => {
                    if let Some(pos) = self.accumulated.find("</function_calls>") {
                        let content = self.accumulated[..pos].trim().to_string();
                        self.accumulated =
                            self.accumulated[pos + "</function_calls>".len()..].to_string();
                        self.state = ToolState::Outside;

                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                            let call = match val.as_array() {
                                Some(arr) => arr.first(),
                                None if val.is_object() => Some(&val),
                                None => None,
                            };

                            if let Some(call) = call
                                && let Some(name) = call["name"].as_str()
                            {
                                let args = call["arguments"].to_string();
                                return ToolEvent::Call(name.to_string(), args);
                            }
                        }

                        return ToolEvent::Start;
                    }
                    return ToolEvent::Start;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.state = ToolState::Outside;
    }
}
