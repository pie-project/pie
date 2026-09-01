use std::sync::Arc;

use chat_template::harmony::Harmony;
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn gpt_oss(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(Harmony::new(tokenizer))
}
