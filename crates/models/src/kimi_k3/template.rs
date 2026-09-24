use std::sync::Arc;

use chat_template::kimi::Kimi;
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(Kimi::new(tokenizer))
}

/// Kimi-K3's XTML chat format (the released model).
#[must_use]
pub fn instruct3(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(chat_template::kimi3::Kimi3::new(tokenizer))
}
