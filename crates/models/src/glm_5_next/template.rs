use std::sync::Arc;

use chat_template::glm::Glm;
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(Glm::new(tokenizer))
}
