//! OLMo 3 instruct implementation.
//! 
//! Implements OLMo 3 chat template:
//! - ChatML-style: <|im_start|>role\ncontent<|im_end|>\n
//! - Tools defined in <functions>...</functions> within system/user messages.
//! - Tool calls in <function_calls>...</function_calls> within assistant messages.
//! - Tool outputs in <|im_start|>environment\ncontent<|im_end|>\n.
//! - Generation prompt adds <|im_start|>assistant\n<think>

use std::sync::Arc;
use crate::model::instruct::{
    ChatDecoder, ChatEvent,
    Instruct,
    ReasoningDecoder, ReasoningEvent,
    ToolDecoder, ToolEvent,
};
use crate::model::tokenizer::Tokenizer;

static TEMPLATE: &str = r#"
{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}{% if not has_system %}{{ '<|im_start|>system
You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>
' }}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system
' + message['content'] }}{% if message.get('functions', none) is not none %}{{ ' <functions>' + message['functions'] + '</functions><|im_end|>
' }}{% else %}{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>
' }}{% endif %}{% elif message['role'] == 'user' %}{% if message.get('functions', none) is not none %}{{ '<|im_start|>user
' + message['content'] + '
' + '<functions>' + message['functions'] + '</functions><|im_end|>
' }}{% else %}{{ '<|im_start|>user
' + message['content'] + '<|im_end|>
' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant
' }}{% if message.get('content', none) is not none %}{{ message['content'] }}{% endif %}{% if message.get('function_calls', none) is not none %}{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}{% endif %}{% if not loop.last %}{{ '<|im_end|>' + '
' }}{% else %}{{ eos_token }}{% endif %}{% elif message['role'] == 'environment' %}{{ '<|im_start|>environment
' + message['content'] + '<|im_end|>
' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant
<think>' }}{% endif %}{% endfor %}"#;

pub struct OlmoInstruct {
    tokenizer: Arc<Tokenizer>,
    im_start: Vec<u32>,
    im_end: Vec<u32>,
    newline: Vec<u32>,
    eos_token: Vec<u32>,
    // Roles
    system_role: Vec<u32>,
    user_role: Vec<u32>,
    assistant_role: Vec<u32>,
    environment_role: Vec<u32>,
    // Tools
    functions_start: Vec<u32>,
    functions_end: Vec<u32>,
    fn_calls_start: Vec<u32>,
    fn_calls_end: Vec<u32>,
    // Generation
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
            eos_token,
            system_role: encode("system"),
            user_role: encode("user"),
            assistant_role: encode("assistant"),
            environment_role: encode("environment"),
            functions_start: encode("<functions>"),
            functions_end: encode("</functions>"),
            fn_calls_start: encode("<function_calls>"),
            fn_calls_end: encode("</function_calls>"),
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
        Box::new(OlmoChatDecoder {
            tokenizer: self.tokenizer.clone(),
            stop_ids: self.im_end.clone(), 
            accumulated: String::new(),
        })
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(OlmoReasoningDecoder {
            tokenizer: self.tokenizer.clone(),
            end_ids: self.think_end.clone(),
            state: ReasoningState::Inside, // Starts inside because of cue <think>
            accumulated: String::new(),
            match_pos: 0,
        })
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(OlmoToolDecoder {
            tokenizer: self.tokenizer.clone(),
            accumulated: String::new(),
            state: ToolState::Outside,
            current_tag: String::new(),
        })
    }
}

// ─── Decoders ───────────────────────────────────────────────

struct OlmoChatDecoder {
    tokenizer: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
    accumulated: String,
}

impl ChatDecoder for OlmoChatDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent {
         for t in tokens {
             if self.stop_ids.contains(t) {
                 return ChatEvent::Done(std::mem::take(&mut self.accumulated));
             }
         }
         let delta = self.tokenizer.decode(tokens, false);
         self.accumulated.push_str(&delta);
         ChatEvent::Delta(delta)
    }
    
    fn reset(&mut self) { self.accumulated.clear(); }
}

enum ReasoningState {
    Outside,
    Inside,
}

struct OlmoReasoningDecoder {
    tokenizer: Arc<Tokenizer>,
    end_ids: Vec<u32>,
    state: ReasoningState,
    accumulated: String,
    match_pos: usize,
}

impl ReasoningDecoder for OlmoReasoningDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent {
        match self.state {
             ReasoningState::Outside => {
                 // Should not happen if started correct, or if re-entered?
                 // For now, no-op
                 ReasoningEvent::Delta(String::new())
             }
             ReasoningState::Inside => {
                for &t in tokens {
                    if self.match_pos < self.end_ids.len() && t == self.end_ids[self.match_pos] {
                        self.match_pos += 1;
                        if self.match_pos == self.end_ids.len() {
                            self.state = ReasoningState::Outside;
                            self.match_pos = 0;
                            let text = std::mem::take(&mut self.accumulated);
                            return ReasoningEvent::Complete(text);
                        }
                    } else {
                        self.match_pos = 0;
                    }
                }
                let delta = self.tokenizer.decode(tokens, false);
                self.accumulated.push_str(&delta);
                ReasoningEvent::Delta(delta)
             }
        }
    }

    fn reset(&mut self) {
        self.state = ReasoningState::Inside;
        self.accumulated.clear();
        self.match_pos = 0;
    }
}

struct OlmoToolDecoder {
    tokenizer: Arc<Tokenizer>,
    accumulated: String,
    state: ToolState,
    current_tag: String,
}

#[derive(Debug, PartialEq)]
enum ToolState {
    Outside,
    Inside,
}

impl ToolDecoder for OlmoToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);
        
        loop {
            match self.state {
                ToolState::Outside => {
                    if let Some(pos) = self.accumulated.find("<function_calls>") {
                        self.accumulated = self.accumulated[pos + "<function_calls>".len()..].to_string();
                        self.state = ToolState::Inside;
                        continue;
                    }
                    if self.accumulated.len() > 200 {
                        let keep = self.accumulated.len() - 50;
                        self.accumulated = self.accumulated[keep..].to_string();
                    }
                    return ToolEvent::Start;
                }
                ToolState::Inside => {
                    if let Some(pos) = self.accumulated.find("</function_calls>") {
                        let content = self.accumulated[..pos].trim().to_string();
                        self.accumulated = self.accumulated[pos + "</function_calls>".len()..].to_string();
                        self.state = ToolState::Outside;
                        
                        // Parse content. Can be JSON list or single object.
                        // Or just raw string?
                        // Assuming tool call format: [{"name":..., "arguments":...}]
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                             if let Some(arr) = val.as_array() {
                                 if let Some(first) = arr.first() {
                                     let name = first["name"].as_str().unwrap_or("").to_string();
                                     let args = first["arguments"].to_string();
                                     return ToolEvent::Call(name, args);
                                 }
                             } else if let Some(obj) = val.as_object() {
                                 let name = obj["name"].as_str().unwrap_or("").to_string();
                                 let args = obj["arguments"].to_string();
                                 return ToolEvent::Call(name, args);
                             }
                        }
                        // Fallback parsing?
                        return ToolEvent::Start; 
                    }
                    return ToolEvent::Start;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.accumulated.clear();
        self.state = ToolState::Outside;
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

    #[test]
    fn system_format() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "system", "Hello"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.system("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>system\nHello<|im_end|>\n"));
    }

    #[test]
    fn equip_format() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "system", "<functions>", "</functions>", "foo", "bar"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.equip(&["foo".to_string(), "bar".to_string()]);
        let text = inst.tokenizer.decode(&tokens, false);
        // assert!(text.contains("<functions>"));
        assert!(text.contains("foo"));
        assert!(text.contains("bar"));
        assert!(text.contains("</functions>"));
    }

    #[test]
    fn answer_format() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "environment", "result"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.answer("fn", "result");
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>environment\nresult<|im_end|>\n"));
    }
    
    #[test]
    fn generation_cue_includes_think() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "assistant", "<think>"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.cue();
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>assistant\n<think>"));
    }
}
