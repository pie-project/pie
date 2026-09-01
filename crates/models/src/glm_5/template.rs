use std::sync::Arc;

use chat_template::chatml::{ChatML, ChatMLInstruct};
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
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
