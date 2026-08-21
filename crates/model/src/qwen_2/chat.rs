use crate::shared::chatml::{ChatMLConfig, QwenInstruct};
use std::sync::Arc;
use tokenizer::Tokenizer;

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
