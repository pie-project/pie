use std::sync::Arc;

use chat_template::kimi::Kimi;
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(Kimi::new(tokenizer))
}
