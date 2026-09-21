use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use super::ProgramName;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParameterType {
    String,
    Int,
    Float,
    Bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parameter {
    #[serde(rename = "type")]
    pub param_type: ParameterType,
    #[serde(default)]
    pub optional: bool,
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Package {
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readme: Option<String>,
    /// `"builtin"` marks a program built into the pie binary and served on
    /// its HTTP routes; absent for everything else.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
}

/// `[runtime]`: what the program needs from the host and, for a script,
/// how the host runs it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    /// The host interface version the program was written against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core: Option<String>,
    /// Set when the artifact is source the host runs under that language's
    /// component rather than a component of its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<Language>,
    /// The entry function a script's language component calls; `main` by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
    /// How that function takes the input; `input` by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call: Option<Call>,
}

impl Runtime {
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub package: Package,
    #[serde(default, skip_serializing_if = "Runtime::is_empty")]
    pub runtime: Runtime,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, Parameter>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dependencies: BTreeMap<String, String>,
}

impl Manifest {
    pub fn parse(content: &str) -> Result<Self> {
        toml::from_str(content).map_err(|e| anyhow!("Failed to parse manifest TOML: {}", e))
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| anyhow!("Failed to serialize manifest: {}", e))
    }

    pub fn program_name(&self) -> ProgramName {
        ProgramName {
            name: self.package.name.clone(),
            version: self.package.version.clone(),
        }
    }

    pub fn dependency_names(&self) -> Vec<ProgramName> {
        self.dependencies
            .iter()
            .map(|(name, version)| ProgramName {
                name: name.clone(),
                version: version.clone(),
            })
            .collect()
    }

    /// The language the program's artifact is source in, when it is a
    /// script the host runs under a language component rather than a
    /// component of its own. `None` is a wasm component.
    pub fn language(&self) -> Option<Language> {
        self.runtime.language
    }

    /// The entry function a script's language component calls.
    pub fn entry(&self) -> &str {
        self.runtime.entry.as_deref().unwrap_or("main")
    }

    /// How a script's entry takes the input.
    pub fn call(&self) -> Call {
        self.runtime.call.unwrap_or_default()
    }

    /// The artifact's extension: `wasm` for a component, the language's for
    /// a script.
    pub fn artifact_extension(&self) -> &'static str {
        self.language().map_or("wasm", Language::extension)
    }

    /// The file name a script's tracebacks quote.
    pub fn script_file(&self) -> String {
        format!("{}.{}", self.package.name, self.artifact_extension())
    }
}

/// How a script's entry function takes the input: `input` passes the
/// parsed input as its one argument, `kwargs` spreads the input object as
/// keyword arguments (what a decorated client function declares).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Call {
    #[default]
    Input,
    Kwargs,
}

impl Call {
    pub fn name(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Kwargs => "kwargs",
        }
    }
}

/// A language the host has (or may have) a language component for. The
/// artifact of a program in one of these is its source, stored beside its
/// manifest as `<version>.<extension>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Python,
    JavaScript,
}

impl Language {
    /// Every language the host can run scripts in.
    pub const ALL: [Language; 2] = [Language::Python, Language::JavaScript];

    pub fn parse(word: &str) -> Result<Self> {
        match word {
            "python" => Ok(Self::Python),
            "javascript" => Ok(Self::JavaScript),
            other => bail!(
                "unknown [runtime] language {other:?}: pie knows \"python\" and \"javascript\""
            ),
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

    /// The language component's file name under the languages dir.
    pub fn component_file(self) -> String {
        format!("{}.wasm", self.name())
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
