use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Python,
    JavaScript,
}

impl Language {
    pub const ALL: [Language; 2] = [Language::Python, Language::JavaScript];

    pub fn parse(word: &str) -> Result<Self> {
        match word {
            "python" => Ok(Self::Python),
            "javascript" => Ok(Self::JavaScript),
            other => bail!("unknown language {other:?}: pie knows \"python\" and \"javascript\""),
        }
    }

    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension {
            "py" => Some(Self::Python),
            "js" | "mjs" => Some(Self::JavaScript),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::JavaScript => "javascript",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Python => "py",
            Self::JavaScript => "js",
        }
    }

    pub fn component_file(self) -> String {
        format!("{}.wasm", self.name())
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

pub fn artifact_extension(language: Option<Language>) -> &'static str {
    language.map_or("wasm", Language::extension)
}
