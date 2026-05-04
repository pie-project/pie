//! OpenResponses JSON types based on the specification.
//!
//! See: https://www.openresponses.org/specification

use serde::{Deserialize, Serialize};

// ============================================================================
// Request Types
// ============================================================================

/// The main request body for POST /responses
#[derive(Debug, Deserialize)]
pub struct CreateResponseBody {
    /// Model identifier (we ignore this and use auto model)
    #[serde(default)]
    pub model: String,

    /// Input items (messages, function call outputs, etc.). Per spec, may be
    /// either a string (interpreted as a single user message) or a list of
    /// items.
    pub input: InputBody,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Maximum output tokens
    #[serde(default)]
    pub max_output_tokens: Option<usize>,

    /// Sampling temperature
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Top-p sampling
    #[serde(default)]
    pub top_p: Option<f32>,

    /// System instructions (alternative to system message)
    #[serde(default)]
    pub instructions: Option<String>,

    /// Tools available for the model to call
    #[serde(default)]
    pub tools: Vec<Tool>,
}

/// Tool definition
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum Tool {
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        parameters: Option<serde_json::Value>,
        #[serde(default)]
        strict: Option<bool>,
    },
}

/// Spec-compliant `input` accepts either a bare string (promoted to a single
/// user message) or an array of input items.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputBody {
    Text(String),
    Items(Vec<LooseInputItem>),
}

impl InputBody {
    /// Normalize to a list of input items, promoting a bare string into a
    /// single user message per spec.
    pub fn into_items(self) -> Vec<InputItem> {
        match self {
            InputBody::Items(v) => v.into_iter().map(InputItem::from).collect(),
            InputBody::Text(s) => vec![InputItem::Message(InputMessage {
                id: None,
                role: Role::User,
                content: MessageContent::Text(s),
                status: None,
            })],
        }
    }
}

/// Input item variants. Each item carries a `type` discriminator; callers
/// that omit it on a message item are accepted via [`LooseInputItem`].
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    Message(InputMessage),

    #[serde(rename = "function_call")]
    FunctionCall(InputFunctionCall),

    #[serde(rename = "function_call_output")]
    FunctionCallOutput(InputFunctionCallOutput),

    #[serde(rename = "item_reference")]
    ItemReference { id: String },
}

/// Loose deserializer for input items: tries the tagged form first, falls
/// back to a bare `InputMessage` if `type` was omitted (spec default is
/// `"message"`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum LooseInputItem {
    Tagged(InputItem),
    UntaggedMessage(InputMessage),
}

impl From<LooseInputItem> for InputItem {
    fn from(v: LooseInputItem) -> Self {
        match v {
            LooseInputItem::Tagged(it) => it,
            LooseInputItem::UntaggedMessage(m) => InputItem::Message(m),
        }
    }
}

/// A function call input item (from model output, used in multi-turn)
#[derive(Debug, Deserialize)]
pub struct InputFunctionCall {
    #[serde(default)]
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub status: Option<String>,
}

/// A function call output item (result from client)
#[derive(Debug, Deserialize)]
pub struct InputFunctionCallOutput {
    pub call_id: String,
    pub output: String,
}

/// A message input item
#[derive(Debug, Deserialize)]
pub struct InputMessage {
    #[serde(default)]
    pub id: Option<String>,

    pub role: Role,

    pub content: MessageContent,

    #[serde(default)]
    pub status: Option<String>,
}

/// Message role
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Developer,
}

/// Message content - can be string or array of content parts
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => {
                parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::InputText { text } => Some(text.clone()),
                        ContentPart::OutputText { text, .. } => Some(text.clone()),
                    })
                    .collect::<Vec<_>>()
                    .join("")
            }
        }
    }
}

/// Content part variants
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },

    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Annotation>,
    },
}

/// Annotation on output text (citations, etc.)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Annotation {
    #[serde(rename = "type")]
    pub annotation_type: String,
}

// ============================================================================
// Response Types
// ============================================================================

/// The complete response resource. Per the OpenResponses spec, the
/// discriminator is `object`, not `type`; required scalar fields include
/// `id`, `object`, `created_at`, `completed_at`, `status`, `model`, and
/// `incomplete_details`.
#[derive(Debug, Serialize)]
pub struct ResponseResource {
    pub id: String,

    #[serde(rename = "object")]
    pub object: String,

    pub created_at: i64,

    pub completed_at: Option<i64>,

    pub status: ResponseStatus,

    pub incomplete_details: Option<IncompleteDetails>,

    pub model: String,

    pub output: Vec<OutputItem>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorPayload>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ResponseResource {
    pub fn new(id: String, model: String) -> Self {
        // wstd's `wasi:clocks/wall-clock` round-trip is overkill for a
        // best-effort timestamp; just stamp 0 in WASM where SystemTime is
        // unavailable. OpenAI clients compare timestamps for ordering, not
        // absolute values, so 0 is acceptable for now.
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            id,
            object: "response".to_string(),
            created_at,
            completed_at: None,
            status: ResponseStatus::InProgress,
            incomplete_details: None,
            model,
            output: Vec::new(),
            error: None,
            usage: None,
        }
    }
}

/// Reason a response stopped before completion (max tokens, content filter,
/// etc.). Spec field — required even when null.
#[derive(Debug, Clone, Serialize)]
pub struct IncompleteDetails {
    pub reason: String,
}

/// Response status
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Incomplete,
}

/// Output item variants
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message(OutputMessage),

    #[serde(rename = "function_call")]
    FunctionCall(OutputFunctionCall),
}

/// An output function call
#[derive(Debug, Serialize, Clone)]
pub struct OutputFunctionCall {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: ItemStatus,
}

/// An output message
#[derive(Debug, Serialize, Clone)]
pub struct OutputMessage {
    pub id: String,
    pub role: Role,
    pub status: ItemStatus,
    pub content: Vec<OutputContentPart>,
}

/// Output content part
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum OutputContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Annotation>,
    },
}

/// Item status
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// Token usage statistics
#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// Error payload
#[derive(Debug, Clone, Serialize)]
pub struct ErrorPayload {
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
    pub message: String,
    pub param: Option<String>,
}

// ============================================================================
// Streaming Event Types
// ============================================================================

/// Base streaming event with common fields
#[derive(Debug, Serialize)]
pub struct StreamingEvent<T> {
    #[serde(rename = "type")]
    pub event_type: String,
    pub sequence_number: u32,
    #[serde(flatten)]
    pub data: T,
}

/// response.created event data
#[derive(Debug, Serialize)]
pub struct ResponseCreatedData {
    pub response: ResponseResource,
}

/// response.in_progress event data
#[derive(Debug, Serialize)]
pub struct ResponseInProgressData {
    pub response: ResponseResource,
}

/// response.completed event data
#[derive(Debug, Serialize)]
pub struct ResponseCompletedData {
    pub response: ResponseResource,
}

/// response.output_item.added event data
#[derive(Debug, Serialize)]
pub struct OutputItemAddedData {
    pub output_index: u32,
    pub item: OutputItem,
}

/// response.output_item.done event data
#[derive(Debug, Serialize)]
pub struct OutputItemDoneData {
    pub output_index: u32,
    pub item: OutputItem,
}

/// response.content_part.added event data
#[derive(Debug, Serialize)]
pub struct ContentPartAddedData {
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub part: OutputContentPart,
}

/// response.content_part.done event data
#[derive(Debug, Serialize)]
pub struct ContentPartDoneData {
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub part: OutputContentPart,
}

/// response.output_text.delta event data
#[derive(Debug, Serialize)]
pub struct OutputTextDeltaData {
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub delta: String,
}

/// response.output_text.done event data
#[derive(Debug, Serialize)]
pub struct OutputTextDoneData {
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub text: String,
}
