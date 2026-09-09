use std::sync::Arc;

use tokenizer::Tokenizer;

type TemplateFn = fn(Arc<Tokenizer>) -> Arc<dyn Instruct>;

pub use chat_template::{
    ChatDecoder, ChatEvent, GenericChatDecoder, Instruct, NoopReasoningDecoder, NoopToolDecoder,
    ReasoningDecoder, ReasoningEvent, ThinkingDecoder, ToolDecoder, ToolEvent, ToolGrammar,
    special, specials,
};

pub type TemplateRow = (&'static str, fn(Arc<Tokenizer>) -> Arc<dyn Instruct>);

#[must_use]
pub fn templates() -> Vec<TemplateRow> {
    crate::skus()
        .map(|sku| (sku.name.as_str(), sku.template))
        .collect()
}

#[must_use]
pub fn template_of(name: &str) -> Option<TemplateFn> {
    crate::sku(name).map(|sku| sku.template)
}
