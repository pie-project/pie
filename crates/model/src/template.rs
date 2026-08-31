use std::sync::Arc;

use tokenizer::Tokenizer;

pub use chat_template::{
    ChatDecoder, ChatEvent, GenericChatDecoder, Instruct, NoopReasoningDecoder, NoopToolDecoder,
    ReasoningDecoder, ReasoningEvent, ThinkingDecoder, ToolDecoder, ToolEvent, ToolGrammar,
    special, specials,
};

pub type TemplateRow = (&'static str, fn(Arc<Tokenizer>) -> Arc<dyn Instruct>);

#[must_use]
pub fn templates() -> Vec<TemplateRow> {
    [
        crate::deepseek_v4::TEMPLATES,
        crate::gemma_4::TEMPLATES,
        crate::glm_5::TEMPLATES,
        crate::gpt_oss::TEMPLATES,
        crate::kimi_k3::TEMPLATES,
        crate::qwen_3::TEMPLATES,
        crate::qwen_4::TEMPLATES,
    ]
    .concat()
}

#[must_use]
pub fn template_of(sku: &str) -> Option<fn(Arc<Tokenizer>) -> Arc<dyn Instruct>> {
    templates()
        .into_iter()
        .find(|(name, _)| *name == sku)
        .map(|(_, make)| make)
}
