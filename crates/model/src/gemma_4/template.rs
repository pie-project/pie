use std::sync::Arc;

use chat_template::gemma::Gemma;
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn gemma4(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(Gemma::new(tokenizer))
}
