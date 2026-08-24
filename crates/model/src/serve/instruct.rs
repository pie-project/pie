pub struct ToolGrammar {
    pub source: String,
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Delta(String),

    Interrupt(u32),

    Done(String),
}

#[derive(Debug, Clone)]
pub enum ReasoningEvent {
    Start,

    Delta(String),

    Complete(String),
}

#[derive(Debug, Clone)]
pub enum ToolEvent {
    Start,

    Call(String, String),
}

pub trait ChatDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent;
    fn reset(&mut self);
}

pub trait ReasoningDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent;
    fn reset(&mut self);
}

pub trait ToolDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent;
    fn reset(&mut self);
}

pub trait Instruct: Send + Sync {
    fn system(&self, msg: &str) -> Vec<u32>;
    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.user(msg)
    }
    fn user(&self, msg: &str) -> Vec<u32>;
    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.user(user));
        tokens
    }
    fn assistant(&self, msg: &str) -> Vec<u32>;
    fn cue(&self) -> Vec<u32>;
    fn seal(&self) -> Vec<u32>;
    fn equip(&self, tools: &[String]) -> Vec<u32>;
    fn answer(&self, name: &str, value: &str) -> Vec<u32>;
    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;
    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;
    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;

    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}
