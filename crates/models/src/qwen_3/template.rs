use std::sync::Arc;

use chat_template::chatml::{ChatML, ChatMLInstruct};
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn chatml(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(ChatMLInstruct::new(
        tokenizer,
        ChatML {
            thinking: true,
            preserve_thinking: false,
            tools: true,
            generation_suffix: "",
            stop_tokens: super::tokenizer::STOP_TOKENS,
        },
    ))
}

/// ChatML with interleaved thinking: a replayed assistant turn keeps its
/// `<think>` block, matching how Qwen3.8 was trained (3.5/3.6 strip it).
#[must_use]
pub fn chatml_interleaved(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(ChatMLInstruct::new(
        tokenizer,
        ChatML {
            thinking: true,
            preserve_thinking: true,
            tools: true,
            generation_suffix: "",
            stop_tokens: super::tokenizer::STOP_TOKENS,
        },
    ))
}
