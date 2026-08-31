use std::sync::Arc;

use chat_template::chatml::{ChatML, ChatMLInstruct};
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn chatml(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
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

/// **QWEN3.8'S OWN READING OF THE SAME GRAMMAR** — ChatML with the
/// interleaved-thinking default: a replayed assistant turn keeps its
/// `<think>` block, because that is what this generation was trained to see
/// (its `chat_template.jinja` flips `preserve_thinking`'s default to true and
/// stops stripping `</think>` from history; qwen3.5/3.6 strip).
///
/// **WHAT THIS DELIBERATELY DOES NOT WRITE: `reasoning_effort`.** The 3.8
/// template can inject a per-effort instruction sentence into the system turn
/// (`xhigh`, its default, injects one too). That sentence is CONTENT, not
/// grammar — plain prose in the system message, spellable by any guest that
/// wants effort control — and this writer takes no kwargs, so the ruling is
/// media.wit's own shape: the host answers the grammar the model was trained
/// on, and what to SAY in it belongs to the inferlet.
#[must_use]
pub fn chatml_interleaved(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(ChatMLInstruct::new(
        tokenizer,
        ChatML {
            thinking: true,
            preserve_thinking: true,
            tools: true,
            generation_suffix: "",
            stop_tokens: super::tokenizer::STOP_TOKENS,
        },
    ))
}
