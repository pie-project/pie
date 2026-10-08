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
    crate::deployments()
        .map(|d| (d.name.as_str(), d.template))
        .collect()
}

#[must_use]
pub fn template_of(name: &str) -> Option<TemplateFn> {
    crate::catalog::parse(name).map(|(entry, _)| entry.template)
}

/// The template a package names `name`.
#[must_use]
pub fn named(name: &str) -> Option<TemplateFn> {
    match name {
        "gemma_4" => Some(crate::gemma_4::template::gemma4),
        "hunyuan_image_3" => Some(crate::hunyuan_image_3::template::instruct),
        "mini_dit" => Some(crate::mini_dit::template::instruct),
        "gpt_oss" => Some(crate::gpt_oss::template::gpt_oss),
        "inkling" => Some(crate::inkling::template::inkling),
        "glm_5" => Some(crate::glm_5::template::instruct),
        "qwen_3_chatml_interleaved" => Some(crate::qwen_3::template::chatml_interleaved),
        "muse_glimmer" => Some(crate::muse_glimmer::template::muse_glimmer),
        _ => None,
    }
}
