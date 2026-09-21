//! The normalized chat request every adapter parses its wire format into.

use crate::generate::Sampling;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// A tool's result, replayed after the assistant turn that called it.
    Tool,
}

/// One tool call, as the model emitted it or as the client replays it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// The wire-level id: [`crate::call_id`] for a call the model made,
    /// whatever the client sent for a replayed one.
    pub id: String,
    pub name: String,
    /// The arguments as a JSON document.
    pub arguments: String,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    /// Text content. Only text: image and audio parts are refused by the
    /// adapters until the chat template can place them.
    pub content: String,
    /// Tool calls an assistant turn made (empty for every other role).
    pub tool_calls: Vec<ToolCall>,
    /// For a [`Role::Tool`] message: the name of the tool that answered.
    pub tool_name: Option<String>,
}

impl Message {
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_name: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum ToolChoice {
    /// The model decides (the default when tools are given).
    #[default]
    Auto,
    /// Tools are described but the model must not call one.
    None,
    /// The model must call some tool (grammar-forced).
    Required,
    /// The model must call this tool (grammar-forced).
    Named(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum ResponseFormat {
    #[default]
    Text,
    /// Any JSON object (grammar-forced).
    JsonObject,
    /// A JSON document matching this JSON Schema (grammar-forced).
    JsonSchema(String),
}

/// Model reasoning ("thinking") for a request: on, off, or whatever the
/// model does on its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Thinking {
    #[default]
    ModelDefault,
    Enabled,
    Disabled,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub messages: Vec<Message>,
    /// A raw completion prompt: the text is tokenized as-is after the
    /// model's prefix, with no chat template, and `messages` and `tools`
    /// are ignored. The legacy completions API.
    pub raw_prompt: Option<String>,
    /// Tool definitions as JSON documents in the OpenAI function shape
    /// (`{"type":"function","function":{"name":…,"parameters":…}}`), which
    /// is what the chat templates embed verbatim.
    pub tools: Vec<String>,
    pub tool_choice: ToolChoice,
    pub response_format: ResponseFormat,
    pub sampling: Sampling,
    pub max_tokens: usize,
    /// Stop sequences: generation ends when the content ends with one, and
    /// the sequence itself is not returned.
    pub stop: Vec<String>,
    pub thinking: Thinking,
    pub stream: bool,
}

/// The cap when a request names none. OpenAI's default is "up to the
/// context window"; an offline single-model server picks something finite
/// so a runaway generation does not hold the KV pool forever.
const DEFAULT_MAX_TOKENS: usize = 1024;

impl Request {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            raw_prompt: None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            response_format: ResponseFormat::Text,
            sampling: Sampling::default(),
            max_tokens: DEFAULT_MAX_TOKENS,
            stop: Vec::new(),
            thinking: Thinking::ModelDefault,
            stream: false,
        }
    }

    /// The tools the model may call: all of them, one of them, or none.
    pub fn callable_tools(&self) -> Vec<String> {
        match &self.tool_choice {
            ToolChoice::None => Vec::new(),
            ToolChoice::Named(name) => self
                .tools
                .iter()
                .filter(|tool| tool_name(tool).as_deref() == Some(name.as_str()))
                .cloned()
                .collect(),
            ToolChoice::Auto | ToolChoice::Required => self.tools.clone(),
        }
    }

    pub fn forces_tool_call(&self) -> bool {
        matches!(
            self.tool_choice,
            ToolChoice::Required | ToolChoice::Named(_)
        )
    }
}

/// The `function.name` (or top-level `name`) of a tool definition.
fn tool_name(tool: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(tool).ok()?;
    parsed
        .get("function")
        .and_then(|f| f.get("name"))
        .or_else(|| parsed.get("name"))
        .and_then(|n| n.as_str())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callable_tools_follow_tool_choice() {
        let a = r#"{"type":"function","function":{"name":"a"}}"#.to_string();
        let b = r#"{"type":"function","function":{"name":"b"}}"#.to_string();
        let mut req = Request::new(vec![]);
        req.tools = vec![a.clone(), b.clone()];
        assert_eq!(req.callable_tools(), vec![a.clone(), b.clone()]);
        req.tool_choice = ToolChoice::Named("b".into());
        assert_eq!(req.callable_tools(), vec![b.clone()]);
        req.tool_choice = ToolChoice::None;
        assert!(req.callable_tools().is_empty());
        assert_eq!(tool_name(&a).as_deref(), Some("a"));
        assert_eq!(tool_name(r#"{"name":"x"}"#).as_deref(), Some("x"));
        assert_eq!(tool_name("not json"), None);
    }
}
