//! Instruct trait — model-specific conversational AI formatting and decoding.
//!
//! Each model architecture provides its own implementation. The API layer
//! delegates to the model's `Instruct` impl for all instruct operations.

use std::sync::Arc;
use crate::model::tokenizer::Tokenizer;

// Shared decoders
pub mod decoders;

// Model implementations
pub mod gemma2;
pub mod gemma3;
pub mod gptoss;
pub mod llama2;
pub mod llama3;
pub mod mistral3;
pub mod olmo3;
pub mod qwen2;
pub mod qwen3;
pub mod r1;

/// Events emitted by the chat decoder.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Generated text chunk
    Delta(String),
    /// Special token encountered (token ID)
    Interrupt(u32),
    /// Generation complete (full accumulated text)
    Done(String),
}

/// Events emitted by the reasoning decoder.
#[derive(Debug, Clone)]
pub enum ReasoningEvent {
    /// Reasoning block started
    Start,
    /// Reasoning text chunk
    Delta(String),
    /// Reasoning complete (full reasoning text)
    Complete(String),
}

/// Events emitted by the tool decoder.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// Tool call detected
    Start,
    /// Complete tool call: (name, arguments-json)
    Call(String, String),
}

/// Classifies generated tokens into text deltas, interrupts, and done.
pub trait ChatDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent;
    fn reset(&mut self);
}

/// Detects reasoning/thinking blocks in the token stream.
pub trait ReasoningDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent;
    fn reset(&mut self);
}

/// Detects tool call blocks in the token stream.
pub trait ToolDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent;
    fn reset(&mut self);
}

/// Model-specific instruct implementation.
///
/// Each architecture provides its own impl with hardcoded tokens & logic.
/// The tokenizer is owned by the implementation to avoid redundant lookups.
pub trait Instruct: Send + Sync {
    fn system(&self, msg: &str) -> Vec<u32>;
    fn user(&self, msg: &str) -> Vec<u32>;
    fn assistant(&self, msg: &str) -> Vec<u32>;
    fn cue(&self) -> Vec<u32>;
    fn seal(&self) -> Vec<u32>;
    fn equip(&self, tools: &[String]) -> Vec<u32>;
    fn answer(&self, name: &str, value: &str) -> Vec<u32>;
    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;
    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;
    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;

    /// Returns an EBNF grammar string that constrains generation to valid
    /// tool-call format for this architecture, given a list of tool schemas.
    /// Returns `None` if the architecture doesn't support constrained tool calling.
    fn tool_call_grammar(&self, _tools: &[String]) -> Option<String> {
        None
    }
}

/// Create the appropriate instruct implementation for the given architecture.
pub fn create(arch_name: &str, tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    use self::qwen3::{QwenInstruct, ChatMLConfig};

    match arch_name {
        "qwen3" => Arc::new(QwenInstruct::new(tokenizer, ChatMLConfig {
            has_thinking: true,
            has_tools: true,
            stop_tokens: &["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        })),
        "qwen2" => Arc::new(self::qwen2::new(tokenizer)),
        "llama2" => Arc::new(self::llama2::LlamaInstruct::new(tokenizer)),
        "llama3" | "l4ma" => Arc::new(self::llama3::LlamaInstruct::new(tokenizer)),
        "r1" | "deepseek_v3" => Arc::new(self::r1::R1Instruct::new(tokenizer)),
        "gptoss" | "gpt_oss" => Arc::new(self::gptoss::GptOssInstruct::new(tokenizer)),
        "gemma2" | "gemma3" => Arc::new(self::gemma2::GemmaInstruct::new(tokenizer)),
        "mistral3" | "ministral3" => Arc::new(self::mistral3::MistralInstruct::new(tokenizer)),
        "olmo3" => Arc::new(self::olmo3::OlmoInstruct::new(tokenizer)),
        _ => Arc::new(QwenInstruct::new(tokenizer, ChatMLConfig {
            has_thinking: false,
            has_tools: false,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        })),
    }
}
