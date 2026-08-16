//! Qwen2.5 instruct.
//!
//! Identical ChatML structure and tool format to Qwen3, but without
//! thinking/reasoning support; it delegates to the shared `QwenInstruct`.

// `QwenInstruct` is vendor-shared chat: Qwen3 owns it, and Qwen2, Qwen3.5,
// GLM-5 and Nemotron-H bind it. The generation that implements a thing keeps
// it; the others name it.
use crate::shared::chatml::{ChatMLConfig, QwenInstruct};
use std::sync::Arc;
use tokenizer::Tokenizer;

// The checkpoint's own `chat_template` is the reference for what follows.

/// Create a Qwen2.5 instruct implementation: the Qwen3 ChatML base with tools
/// enabled and no thinking support.
pub fn new(tokenizer: Arc<Tokenizer>) -> QwenInstruct {
    QwenInstruct::new(
        tokenizer,
        ChatMLConfig {
            has_thinking: false,
            has_tools: true,
            generation_suffix: "",
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        },
    )
}
