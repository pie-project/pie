use std::sync::Arc;

use chat_template::deepseek::DeepSeek;
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn r1(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(DeepSeek::new(tokenizer))
}
