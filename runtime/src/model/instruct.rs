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

/// Create the appropriate instruct implementation for the given architecture.
///
/// Accepts both pie's lowercase internal arch names (e.g. `"qwen3"`, what the
/// `pie_driver` path forwards) and the HuggingFace `architectures[0]` strings
/// (e.g. `"Qwen3ForCausalLM"`, what the `pie_driver_vllm` / `pie_driver_sgl`
/// paths forward — see `pie/src/pie/capabilities.py::DriverCapabilities`).
///
/// **Why both forms:** `pie_driver/loader.py` translates HF `model_type` →
/// pie internal name via `HF_TO_PIE_ARCH` before the handshake, but
/// `pie_driver_vllm/loader.py` and `pie_driver_sgl/loader.py` forward
/// `architectures[0]` verbatim. Without aliasing, any model loaded through
/// the vllm/sgl drivers silently falls through to the `_` arm, which uses
/// `has_thinking: false` and `has_tools: false` — that was the root cause of
/// pie-project/pie#328 (Qwen3 `.with_reasoning()` returning empty `thinking`).
///
/// `LlamaForCausalLM` is intentionally NOT aliased here: it's used by both
/// llama2 and llama3 in HF and disambiguation needs config-level signals
/// (e.g. rope_scaling). The Python driver MUST hand off the disambiguated
/// pie internal name (`"llama2"` or `"llama3"`) for those.
pub fn create(arch_name: &str, tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    use self::qwen3::{QwenInstruct, ChatMLConfig};

    match arch_name {
        "qwen3" | "Qwen3ForCausalLM" => Arc::new(QwenInstruct::new(tokenizer, ChatMLConfig {
            has_thinking: true,
            has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        })),
        "qwen2" | "Qwen2ForCausalLM" | "Qwen2_5ForCausalLM" => Arc::new(self::qwen2::new(tokenizer)),
        "llama2" => Arc::new(self::llama2::LlamaInstruct::new(tokenizer)),
        "llama3" | "l4ma" => Arc::new(self::llama3::LlamaInstruct::new(tokenizer)),
        "r1" | "deepseek_v3" | "DeepseekV3ForCausalLM" => Arc::new(self::r1::R1Instruct::new(tokenizer)),
        "gptoss" | "gpt_oss" | "GptOssForCausalLM" => Arc::new(self::gptoss::GptOssInstruct::new(tokenizer)),
        "gemma2" | "gemma3" | "Gemma2ForCausalLM" | "Gemma3ForCausalLM" | "Gemma3TextForCausalLM"
            => Arc::new(self::gemma2::GemmaInstruct::new(tokenizer)),
        "mistral3" | "ministral3" | "Mistral3ForCausalLM" => Arc::new(self::mistral3::MistralInstruct::new(tokenizer)),
        "olmo3" | "Olmo3ForCausalLM" => Arc::new(self::olmo3::OlmoInstruct::new(tokenizer)),
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

    /// Build a synthetic CharLevel tokenizer that has `<think>` and `</think>`
    /// at known positions. Used to verify `reasoning_decoder()` returns a real
    /// `ThinkingDecoder` (not a `NoopReasoningDecoder`) for a given arch_name.
    fn make_tok() -> Arc<Tokenizer> {
        let v: Vec<String> = vec![
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system", "\n", "user", "assistant",
            "<think>", "</think>",
        ].into_iter().map(String::from).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    /// Feeding the `<think>` token MUST transition the reasoning decoder
    /// into the inside state (`Start` event). If it instead returns
    /// `Delta("")`, the create() call resolved to `NoopReasoningDecoder`,
    /// meaning the arch_name fell through to the `_` arm.
    fn assert_thinking_enabled(arch_name: &str) {
        let tok = make_tok();
        let inst = create(arch_name, tok.clone());
        let think_id = tok.encode("<think>");
        assert!(!think_id.is_empty(), "<think> must encode to non-empty for {arch_name}");
        let mut dec = inst.reasoning_decoder();
        match dec.feed(&think_id) {
            ReasoningEvent::Start => {}
            other => panic!(
                "create({arch_name:?}).reasoning_decoder().feed(<think>) \
                 expected Start, got {other:?} — arch_name fell through to the \
                 has_thinking=false fallback arm (this was pie-project/pie#328)"
            ),
        }
    }

    #[test]
    fn qwen3_internal_name_enables_thinking() {
        assert_thinking_enabled("qwen3");
    }

    #[test]
    fn qwen3_hf_arch_name_enables_thinking() {
        // Regression: pie_driver_vllm / pie_driver_sgl forward `architectures[0]`
        // (the HF class name) verbatim. Pie runtime MUST recognise this form.
        assert_thinking_enabled("Qwen3ForCausalLM");
    }
}
