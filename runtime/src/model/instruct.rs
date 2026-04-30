//! Instruct trait — model-specific conversational AI formatting and decoding.
//!
//! Each model architecture provides its own implementation. The API layer
//! delegates to the model's `Instruct` impl for all instruct operations.

use std::sync::Arc;
use crate::inference::structured::grammar::Grammar;
use crate::model::tokenizer::Tokenizer;

/// A model-provided tool-call grammar: the EBNF source (used as a stable
/// cache key when compiling for a tokenizer) paired with the parsed AST.
pub struct ToolGrammar {
    pub source: String,
    pub grammar: Arc<Grammar>,
}

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

    /// Returns the parsed tool-call grammar that constrains generation to
    /// the architecture's tool-call format, given a list of tool schemas.
    /// Returns `None` if the architecture doesn't support constrained tool calling.
    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}

/// Dispatch on the pie internal arch name resolved by the Python driver.
///
/// HF identity (`architectures[0]` class name, `model_type`, and any
/// disambiguating config fields like `rope_scaling`) never crosses the FFI
/// boundary — Python's `pie_driver.model.resolve()` is the single owner
/// of that translation. See pie-project/pie#328: the regression there
/// came from the vllm/sgl drivers forwarding a raw HF class name and
/// Rust silently hitting the fallback arm below.
pub fn create(arch_name: &str, tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    use self::qwen3::{QwenInstruct, ChatMLConfig};

    match arch_name {
        "qwen3" => Arc::new(QwenInstruct::new(tokenizer, ChatMLConfig {
            has_thinking: true,
            has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        })),
        "qwen2" => Arc::new(self::qwen2::new(tokenizer)),
        "llama2" => Arc::new(self::llama2::LlamaInstruct::new(tokenizer)),
        "llama3" | "l4ma" => Arc::new(self::llama3::LlamaInstruct::new(tokenizer)),
        "r1" | "deepseek_v3" => Arc::new(self::r1::R1Instruct::new(tokenizer)),
        "gptoss" | "gpt_oss" => Arc::new(self::gptoss::GptOssInstruct::new(tokenizer)),
        "gemma2" | "gemma3" => Arc::new(self::gemma2::GemmaInstruct::new(tokenizer)),
        "mistral3" | "ministral3" => Arc::new(self::mistral3::MistralInstruct::new(tokenizer)),
        "olmo3" => Arc::new(self::olmo3::OlmoInstruct::new(tokenizer)),
        // Silent fallback returns Qwen ChatML — wrong template for non-Qwen
        // models. Any unrecognised name here means the Python driver skipped
        // pie_driver.model.resolve() or sent an HF class name; that's the
        // bug, not a missing arm here.
        _ => Arc::new(QwenInstruct::new(tokenizer, ChatMLConfig {
            has_thinking: false,
            has_tools: false,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tokenizer::Tokenizer;

    fn make_tok() -> Arc<Tokenizer> {
        let v: Vec<String> = vec![
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system", "\n", "user", "assistant",
            "<think>", "</think>",
        ].into_iter().map(String::from).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    /// Resolved pie internal name -> reasoning enabled. The Python driver
    /// (pie_driver.model.resolve) is responsible for sending this exact form.
    #[test]
    fn qwen3_pie_name_enables_thinking() {
        let tok = make_tok();
        let inst = create("qwen3", tok.clone());
        let mut dec = inst.reasoning_decoder();
        let think_id = tok.encode("<think>");
        match dec.feed(&think_id) {
            ReasoningEvent::Start => {}
            other => panic!(
                "create(\"qwen3\") gave a no-op reasoning decoder: {other:?}"
            ),
        }
    }

    /// Raw HF class name MUST hit the fallback arm. Pie runtime is not in the
    /// business of recognising HF strings — that translation is the Python
    /// driver's job (#328). If this test starts failing, someone added an
    /// HF alias to `create()`; remove it and fix the driver instead.
    #[test]
    fn hf_class_name_hits_fallback() {
        let tok = make_tok();
        let inst = create("Qwen3ForCausalLM", tok.clone());
        let mut dec = inst.reasoning_decoder();
        let think_id = tok.encode("<think>");
        match dec.feed(&think_id) {
            ReasoningEvent::Delta(s) if s.is_empty() => {}
            other => panic!(
                "create(\"Qwen3ForCausalLM\") was supposed to hit the no-op \
                 fallback (NoopReasoningDecoder always returns Delta(\"\")), \
                 got {other:?}. Did someone alias an HF class name in \
                 create()? See #328 — keep HF resolution in pie_driver.model.resolve()."
            ),
        }
    }
}
