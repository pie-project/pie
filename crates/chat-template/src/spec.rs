//! A template as a model's package states it: a format this crate ships,
//! configured. A package names no Rust; it picks a format and sets what the
//! format leaves open, and [`build`] is the one place a name becomes an
//! [`Instruct`].

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::Instruct;
use crate::chatml::{ChatML, ChatMLInstruct};

/// The formats a package may name, each a module of this crate.
pub const FORMATS: &[&str] = &[
    "chatml", "harmony", "gemma", "deepseek", "glm", "kimi", "kimi3", "inkling", "atem", "lines",
    "raw",
];

/// A format and its settings. A setting a format does not read is refused,
/// so a package states only what its format reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Spec {
    pub format: String,
    /// ChatML: the model reasons in `<think>` blocks.
    pub thinking: bool,
    /// ChatML: a replayed assistant turn keeps its reasoning.
    pub preserve_thinking: bool,
    /// ChatML: the model calls tools.
    pub tools: bool,
    /// ChatML: text after the assistant header that opens every generation.
    pub generation_suffix: String,
    /// ChatML, lines, raw: the tokens a generation stops at.
    pub stop: Vec<String>,
    /// Lines: the token that opens a prompt.
    pub bos: Option<String>,
    /// Lines: the token that closes an assistant turn.
    pub eos: Option<String>,
}

impl Spec {
    /// The format `format` with nothing set.
    #[must_use]
    pub fn of(format: &str) -> Spec {
        Spec {
            format: format.to_string(),
            ..Spec::default()
        }
    }

    /// Which of the settings this spec sets.
    fn set(&self) -> Vec<&'static str> {
        let mut set = Vec::new();
        if self.thinking {
            set.push("thinking");
        }
        if self.preserve_thinking {
            set.push("preserve_thinking");
        }
        if self.tools {
            set.push("tools");
        }
        if !self.generation_suffix.is_empty() {
            set.push("generation_suffix");
        }
        if !self.stop.is_empty() {
            set.push("stop");
        }
        if self.bos.is_some() {
            set.push("bos");
        }
        if self.eos.is_some() {
            set.push("eos");
        }
        set
    }

    /// Refuses a setting the format does not read.
    pub fn check(&self) -> Result<(), String> {
        let reads: &[&str] = match self.format.as_str() {
            "chatml" => &[
                "thinking",
                "preserve_thinking",
                "tools",
                "generation_suffix",
                "stop",
            ],
            "lines" => &["stop", "bos", "eos"],
            "raw" => &["stop"],
            "harmony" | "gemma" | "deepseek" | "glm" | "kimi" | "kimi3" | "inkling" | "atem" => &[],
            other => {
                return Err(format!(
                    "`{other}` names no template format; this build ships {}",
                    FORMATS.join(", ")
                ));
            }
        };
        if let Some(stray) = self.set().into_iter().find(|s| !reads.contains(s)) {
            return Err(format!(
                "the `{}` format reads no `{stray}` setting",
                self.format
            ));
        }
        match self.format.as_str() {
            "chatml" | "raw" if self.stop.is_empty() => Err(format!(
                "the `{}` format needs its stop tokens",
                self.format
            )),
            "lines" if self.stop.is_empty() || self.bos.is_none() || self.eos.is_none() => {
                Err("the `lines` format needs its stop tokens, its bos and its eos".to_string())
            }
            _ => Ok(()),
        }
    }
}

/// The template `spec` states, over `tokenizer`.
pub fn build(spec: &Spec, tokenizer: Arc<Tokenizer>) -> Result<Arc<dyn Instruct>, String> {
    spec.check()?;
    Ok(match spec.format.as_str() {
        "chatml" => Arc::new(ChatMLInstruct::new(
            tokenizer,
            ChatML {
                thinking: spec.thinking,
                preserve_thinking: spec.preserve_thinking,
                tools: spec.tools,
                generation_suffix: spec.generation_suffix.clone(),
                stop_tokens: spec.stop.clone(),
            },
        )),
        "harmony" => Arc::new(crate::harmony::Harmony::new(tokenizer)),
        "gemma" => Arc::new(crate::gemma::Gemma::new(tokenizer)),
        "deepseek" => Arc::new(crate::deepseek::DeepSeek::new(tokenizer)),
        "glm" => Arc::new(crate::glm::Glm::new(tokenizer)),
        "kimi" => Arc::new(crate::kimi::Kimi::new(tokenizer)),
        "kimi3" => Arc::new(crate::kimi3::Kimi3::new(tokenizer)),
        "inkling" => Arc::new(crate::inkling::Inkling::new(tokenizer)),
        "atem" => Arc::new(crate::atem::Atem::new(tokenizer)),
        "lines" => Arc::new(crate::lines::Lines::new(
            tokenizer,
            spec.bos.as_deref().unwrap_or_default(),
            spec.eos.as_deref().unwrap_or_default(),
            &spec.stop,
        )),
        "raw" => Arc::new(crate::raw::Raw::new(tokenizer, &spec.stop)),
        _ => unreachable!("checked above"),
    })
}
