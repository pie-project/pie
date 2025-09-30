use serde::Serialize;
use serde_json::Value;

// --- Simplified Data Structures ---

/// Represents a single tool call.
#[derive(Serialize, Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Represents a single message in the conversation history.
/// This struct is no longer generic over a lifetime. All string data is owned.
#[derive(Serialize, Clone, Debug)]
struct Message {
    role: String,
    content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

// --- API Implementation ---

#[derive(Debug, Clone)]
pub struct ChatFormatter {
    messages: Vec<Message>,
}

impl ChatFormatter {
    /// Creates a new Conversation instance with a given chat template.
    pub fn new() -> Self {
        ChatFormatter {
            messages: Vec::new(),
        }
    }

    /// Adds a system message to the conversation.
    /// The `content` parameter now accepts any type that implements `ToString`.
    pub fn system<T: ToString>(&mut self, content: T) {
        self.messages.push(Message {
            role: "system".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
        });
    }

    /// Adds a user message to the conversation.
    pub fn user<T: ToString>(&mut self, content: T) {
        self.messages.push(Message {
            role: "user".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
        });
    }

    /// Adds a simple assistant message with only text content.
    pub fn assistant<T: ToString>(&mut self, content: T) {
        // We specify a type for `None` to help the compiler infer the generic `R`
        // in `assistant_response`. `&str` is a good, lightweight choice.
        self.assistant_response(content, None::<&str>, None);
    }

    /// Adds a comprehensive assistant response, optionally including reasoning and tool calls.
    /// Both `content` and `reasoning` are now generic over `ToString`.
    pub fn assistant_response<T: ToString, R: ToString>(
        &mut self,
        content: T,
        reasoning: Option<R>,
        tool_calls: Option<Vec<ToolCall>>,
    ) {
        self.messages.push(Message {
            role: "assistant".to_string(),
            content: content.to_string(),
            reasoning_content: reasoning.map(|s| s.to_string()),
            tool_calls,
        });
    }

    /// Adds a tool response message to the conversation.
    pub fn tool<T: ToString>(&mut self, content: T) {
        self.messages.push(Message {
            role: "tool".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
        });
    }

    pub fn has_messages(&self) -> bool {
        !self.messages.is_empty()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Renders the entire conversation into a single formatted string prompt.
    pub fn render(
        &self,
        template: &str,
        add_generation_prompt: bool,
        begin_of_sequence: bool,
    ) -> String {
        minijinja::render!(
            template,
            messages => self.messages,
            add_generation_prompt => add_generation_prompt,
            begin_of_sequence => begin_of_sequence,
        )
    }
}
