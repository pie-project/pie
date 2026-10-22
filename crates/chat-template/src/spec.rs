//! A template as a package states it: a format this crate ships, with the
//! settings the format reads. [`build`] is the one place a name becomes an
//! [`Instruct`].

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::Instruct;
use crate::chatml::{ChatML, ChatMLInstruct};

pub const FORMATS: &[&str] = &[
    "chatml",
    "harmony",
    "gemma",
    "deepseek",
    "dots",
    "glm",
    "glm47",
    "glm45",
    "kimi",
    "kimi3",
    "inkling",
    "atem",
    "mimo",
    "minimax",
    "minimax2",
    "hy",
    "llama",
    "qwen3_coder",
    "lines",
    "raw",
];

/// A format and its settings; a setting the format does not read is refused.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Spec {
    pub format: String,
    /// ChatML, gemma, hy.
    pub thinking: bool,
    pub preserve_thinking: bool,
    pub reasoning_effort: bool,
    pub tools: bool,
    pub xml_tools: bool,
    pub generation_suffix: String,
    /// MiniMax-M2: the identity its default system turn states.
    pub identity: String,
    /// ChatML, lines, raw.
    pub stop: Vec<String>,
    /// Lines.
    pub bos: Option<String>,
    pub eos: Option<String>,
}

impl Spec {
    #[must_use]
    pub fn of(format: &str) -> Spec {
        Spec {
            format: format.to_string(),
            ..Spec::default()
        }
    }

    fn set(&self) -> Vec<&'static str> {
        let mut set = Vec::new();
        if self.thinking {
            set.push("thinking");
        }
        if self.preserve_thinking {
            set.push("preserve_thinking");
        }
        if self.reasoning_effort {
            set.push("reasoning_effort");
        }
        if self.tools {
            set.push("tools");
        }
        if self.xml_tools {
            set.push("xml_tools");
        }
        if !self.generation_suffix.is_empty() {
            set.push("generation_suffix");
        }
        if !self.identity.is_empty() {
            set.push("identity");
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

    pub fn check(&self) -> Result<(), String> {
        let reads: &[&str] = match self.format.as_str() {
            "chatml" => &[
                "thinking",
                "preserve_thinking",
                "reasoning_effort",
                "tools",
                "xml_tools",
                "generation_suffix",
                "stop",
            ],
            "gemma" | "hy" => &["thinking"],
            "lines" => &["stop", "bos", "eos"],
            "minimax2" => &["identity"],
            "raw" => &["stop"],
            "harmony" | "deepseek" | "dots" | "glm" | "glm47" | "glm45" | "kimi" | "kimi3"
            | "inkling" | "atem" | "mimo" | "minimax" | "llama" | "qwen3_coder" => &[],
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
            "minimax2" if self.identity.is_empty() => {
                Err("the `minimax2` format needs the identity its system turn states".to_string())
            }
            "lines" if self.stop.is_empty() || self.bos.is_none() || self.eos.is_none() => {
                Err("the `lines` format needs its stop tokens, its bos and its eos".to_string())
            }
            _ => Ok(()),
        }
    }
}

pub fn build(spec: &Spec, tokenizer: Arc<Tokenizer>) -> Result<Arc<dyn Instruct>, String> {
    spec.check()?;
    Ok(match spec.format.as_str() {
        "chatml" => Arc::new(ChatMLInstruct::new(
            tokenizer,
            ChatML {
                thinking: spec.thinking,
                preserve_thinking: spec.preserve_thinking,
                reasoning_effort: spec.reasoning_effort,
                tools: spec.tools,
                xml_tools: spec.xml_tools,
                generation_suffix: spec.generation_suffix.clone(),
                stop_tokens: spec.stop.clone(),
            },
        )),
        "harmony" => Arc::new(crate::harmony::Harmony::new(tokenizer)),
        "gemma" => Arc::new(crate::gemma::Gemma::new(tokenizer, spec.thinking)),
        "deepseek" => Arc::new(crate::deepseek::DeepSeek::new(tokenizer)),
        "dots" => Arc::new(crate::dots::Dots::new(tokenizer)),
        "glm" => Arc::new(crate::glm::Glm::new(tokenizer)),
        "glm47" => Arc::new(crate::glm::Glm::with_style(
            tokenizer,
            crate::glm::Style::Glm47,
        )),
        "glm45" => Arc::new(crate::glm::Glm::with_style(
            tokenizer,
            crate::glm::Style::Glm45,
        )),
        "kimi" => Arc::new(crate::kimi::Kimi::new(tokenizer)),
        "kimi3" => Arc::new(crate::kimi3::Kimi3::new(tokenizer)),
        "inkling" => Arc::new(crate::inkling::Inkling::new(tokenizer)),
        "atem" => Arc::new(crate::atem::Atem::new(tokenizer)),
        "mimo" => Arc::new(crate::mimo::Mimo::new(tokenizer)),
        "minimax" => Arc::new(crate::minimax::MiniMax::new(tokenizer)),
        "minimax2" => Arc::new(crate::minimax2::MiniMax2::new(tokenizer, &spec.identity)),
        "hy" => Arc::new(crate::hy::Hy::new(tokenizer, spec.thinking)),
        "llama" => Arc::new(crate::llama::Llama::new(tokenizer)),
        "qwen3_coder" => Arc::new(crate::qwen3_coder::Qwen3Coder::new(tokenizer)),
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
