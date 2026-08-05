//! Qwen2.5 instruct implementation.
//!
//! Identical ChatML structure and tool format as Qwen3, but without
//! thinking/reasoning support. Delegates to the shared QwenInstruct
//! implementation in qwen3.rs.
//!
//! Reference: Qwen2.5 Jinja chat template.

// `QwenInstruct` is vendor-shared chat: Qwen3 owns it, and Qwen2, Qwen3.5,
// GLM-5 and Nemotron-H bind it. The generation that implements a thing keeps
// it; the others name it.
use pie_model_qwen_3::chat::{ChatMLConfig, QwenInstruct};
use pie_tokenizer::Tokenizer;
use std::sync::Arc;

// The implementation below mirrors the published Qwen2 jinja chat template;
// the verbatim copy that used to sit here as a static was never read — the
// checkpoint's own `chat_template` is the reference.

/// Create a Qwen2.5 instruct implementation.
///
/// Uses the same ChatML base as Qwen3 but with:
/// - No thinking/reasoning support
/// - Tools enabled (same format as Qwen3)
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
