//! Instruct trait — model-specific conversational AI formatting and decoding.
//!
//! Each model architecture provides its own implementation. The API layer
//! delegates to the model's `Instruct` impl for all instruct operations.
//!
//! Both halves are here. They were two files in two crates — the trait below
//! and the [`create`] registry at the bottom — because a generation crate
//! could not depend on the crate that dispatches to it, so the vocabulary had
//! to sit underneath both. One crate, one module.

/// A model-provided tool-call grammar in EBNF form.
pub struct ToolGrammar {
    pub source: String,
}
// The shared decoders, re-exported so `instruct::decoders` stays a valid
// path: it is what every generation's template imports, and the templates
// became crates without their imports needing to know.
pub use crate::decoders;

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
    /// Returns the parsed tool-call grammar that constrains generation to
    /// the architecture's tool-call format, given a list of tool schemas.
    /// Returns `None` if the architecture doesn't support constrained tool calling.
    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}

// ── The registry ─────────────────────────────────────────────────────

use tokenizer::Tokenizer;
use std::sync::Arc;

/// Create the appropriate instruct implementation for the given architecture.
///
/// This match is the chat aspect's registry: `model_type` in, implementation
/// out, with every N:1 reuse (nemotron_h speaking ChatML, deepseek_v4 speaking
/// R1) stated as its own arm rather than implied by a directory. The rows
/// dispatch on the *model type*; the generation crates only hold the
/// implementations.
pub fn create(arch_name: &str, tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    use crate::families::chatml::{ChatMLConfig, QwenInstruct};

    match arch_name {
        "qwen3" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_moe"
        | "qwen3_vl" | "qwen3_vl_text" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
        "nemotron_h" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: false,
                generation_suffix: "<think>\n",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
        "qwen2" => Arc::new(crate::qwen_2::chat::new(tokenizer)),
        "llama2" => Arc::new(crate::llama_2::chat::LlamaInstruct::new(tokenizer)),
        "llama3" | "l4ma" => Arc::new(crate::llama_3::chat::LlamaInstruct::new(tokenizer)),
        "r1" | "deepseek_v3" | "deepseek_v4" => {
            Arc::new(crate::deepseek_r1::chat::R1Instruct::new(tokenizer))
        }
        "kimi_k2" | "kimi_k25" | "kimi_k3" => {
            Arc::new(crate::kimi_k2::chat::KimiInstruct::new(tokenizer))
        }
        "glm_moe_dsa" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>", "<|user|>", "<|assistant|>"],
            },
        )),
        "gptoss" | "gpt_oss" => Arc::new(crate::gpt_oss::chat::GptOssInstruct::new(tokenizer)),
        "gemma2" => Arc::new(crate::gemma_2::chat::GemmaInstruct::new(tokenizer)),
        "gemma3" => Arc::new(crate::gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            crate::gemma_3::chat::Gemma3Variant::Gemma3,
        )),
        "gemma3_text" => Arc::new(crate::gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            crate::gemma_3::chat::Gemma3Variant::Gemma3Text,
        )),
        "gemma3n" => Arc::new(crate::gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            crate::gemma_3::chat::Gemma3Variant::Gemma3n,
        )),
        "gemma3n_text" => Arc::new(crate::gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            crate::gemma_3::chat::Gemma3Variant::Gemma3nText,
        )),
        "gemma4" => Arc::new(crate::gemma_4::chat::Gemma4Instruct::for_variant(
            tokenizer,
            crate::gemma_4::chat::Gemma4Variant::Gemma4,
        )),
        "gemma4_text" => Arc::new(crate::gemma_4::chat::Gemma4Instruct::for_variant(
            tokenizer,
            crate::gemma_4::chat::Gemma4Variant::Gemma4Text,
        )),
        "mistral3" | "ministral3" => {
            Arc::new(crate::mistral_3::chat::MistralInstruct::new(tokenizer))
        }
        "olmo2" => Arc::new(crate::olmo_2::chat::Olmo2Instruct::new(tokenizer)),
        "olmo3" => Arc::new(crate::olmo_3::chat::OlmoInstruct::new(tokenizer)),
        "phi3" => Arc::new(crate::phi_3::chat::Phi3Instruct::new(tokenizer)),
        _ => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: false,
                has_tools: false,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
    }
}
