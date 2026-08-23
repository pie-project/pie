pub trait Instruct: Send + Sync {
    fn system(&self, message: &str) -> Vec<u32>;
    fn first_user(&self, message: &str) -> Vec<u32>;
    fn user(&self, message: &str) -> Vec<u32>;
    fn system_user(&self, system: &str, user: &str) -> Vec<u32>;
    fn assistant(&self, message: &str) -> Vec<u32>;
    fn cue(&self) -> Vec<u32>;
    fn seal(&self) -> Vec<u32>;
    fn equip(&self, tools: &[String]) -> Vec<u32>;
    fn answer(&self, name: &str, value: &str) -> Vec<u32>;
    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;
    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;
    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;
}

pub trait ChatDecoder: Send {
    fn push(&mut self, token: u32) -> Option<String>;
    fn finish(&mut self) -> Option<String>;
}

pub trait ReasoningDecoder: Send {
    fn push(&mut self, token: u32) -> Option<Reasoning>;
}

pub trait ToolDecoder: Send {
    fn push(&mut self, token: u32) -> Option<ToolCall>;
}

pub enum Reasoning {
    Thinking(String),
    Answer(String),
}

pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}
