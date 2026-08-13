//! The chat aspect's registry: `arch_name` in, an [`Instruct`] out.
//!
//! The *vocabulary* — the `Instruct` trait, the decoders and their events —
//! lives in `pie-model-common`, because every generation implements it and a
//! generation crate cannot depend on the registry that dispatches to it. What
//! is left here is the dispatch itself.

pub use pie_model_common::instruct::*;

use pie_tokenizer::Tokenizer;
use std::sync::Arc;

/// Create the appropriate instruct implementation for the given architecture.
///
/// This match is the chat aspect's registry: `model_type` in, implementation
/// out, with every N:1 reuse (nemotron_h speaking ChatML, deepseek_v4 speaking
/// R1) stated as its own arm rather than implied by a directory. The rows
/// dispatch on the *model type*; the generation crates only hold the
/// implementations.
pub fn create(arch_name: &str, tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    use pie_model_qwen_3::chat::{ChatMLConfig, QwenInstruct, ToolCallFormat};

    match arch_name {
        "qwen3" | "qwen3_moe" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                tool_call_format: ToolCallFormat::Json,
                generation_suffix: "",
                thinking_off_suffix: "",
                trim_content: false,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_6"
        | "qwen3_6_moe" | "qwen3_vl" | "qwen3_vl_text" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                tool_call_format: ToolCallFormat::Qwen35Xml,
                // The template opens the model's turn INSIDE a reasoning block
                // (`<|im_start|>assistant\n<think>\n`), and closes an empty one
                // when the caller turns thinking off. An empty suffix here made
                // every sampled first token come from a position the checkpoint
                // was never trained to generate at.
                generation_suffix: "<think>\n",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                trim_content: true,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
        "nemotron_h" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: false,
                tool_call_format: ToolCallFormat::Json,
                generation_suffix: "<think>\n",
                thinking_off_suffix: "<think>\n",
                trim_content: false,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
        "qwen2" => Arc::new(pie_model_qwen_2::chat::new(tokenizer)),
        "llama2" => Arc::new(pie_model_llama_2::chat::LlamaInstruct::new(tokenizer)),
        "llama3" | "l4ma" => Arc::new(pie_model_llama_3::chat::LlamaInstruct::new(tokenizer)),
        "r1" | "deepseek_v3" | "deepseek_v4" => {
            Arc::new(pie_model_deepseek_r1::chat::R1Instruct::new(tokenizer))
        }
        "kimi_k2" | "kimi_k25" | "kimi_k3" => {
            Arc::new(pie_model_kimi_k2::chat::KimiInstruct::new(tokenizer))
        }
        "glm_moe_dsa" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                tool_call_format: ToolCallFormat::Json,
                generation_suffix: "",
                thinking_off_suffix: "",
                trim_content: false,
                stop_tokens: &["<|im_end|>", "<|endoftext|>", "<|user|>", "<|assistant|>"],
            },
        )),
        "gptoss" | "gpt_oss" => Arc::new(pie_model_gpt_oss::chat::GptOssInstruct::new(tokenizer)),
        "gemma2" => Arc::new(pie_model_gemma_2::chat::GemmaInstruct::new(tokenizer)),
        "gemma3" => Arc::new(pie_model_gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            pie_model_gemma_3::chat::Gemma3Variant::Gemma3,
        )),
        "gemma3_text" => Arc::new(pie_model_gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            pie_model_gemma_3::chat::Gemma3Variant::Gemma3Text,
        )),
        "gemma3n" => Arc::new(pie_model_gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            pie_model_gemma_3::chat::Gemma3Variant::Gemma3n,
        )),
        "gemma3n_text" => Arc::new(pie_model_gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            pie_model_gemma_3::chat::Gemma3Variant::Gemma3nText,
        )),
        "gemma4" => Arc::new(pie_model_gemma_4::chat::Gemma4Instruct::for_variant(
            tokenizer,
            pie_model_gemma_4::chat::Gemma4Variant::Gemma4,
        )),
        "gemma4_text" => Arc::new(pie_model_gemma_4::chat::Gemma4Instruct::for_variant(
            tokenizer,
            pie_model_gemma_4::chat::Gemma4Variant::Gemma4Text,
        )),
        "mistral3" | "ministral3" => {
            Arc::new(pie_model_mistral_3::chat::MistralInstruct::new(tokenizer))
        }
        "olmo2" => Arc::new(pie_model_olmo_2::chat::Olmo2Instruct::new(tokenizer)),
        "olmo3" => Arc::new(pie_model_olmo_3::chat::OlmoInstruct::new(tokenizer)),
        "phi3" => Arc::new(pie_model_phi_3::chat::Phi3Instruct::new(tokenizer)),
        _ => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: false,
                has_tools: false,
                tool_call_format: ToolCallFormat::Json,
                generation_suffix: "",
                thinking_off_suffix: "",
                trim_content: false,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
    }
}
