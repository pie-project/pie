//! Manifest parsing and handling
//!
//! Provides utilities for parsing inferlet manifest TOML files.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use super::ProgramName;

/// Parameter type definition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParameterType {
    String,
    Int,
    Float,
    Bool,
}

/// Parameter definition in the manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parameter {
    #[serde(rename = "type")]
    pub param_type: ParameterType,
    #[serde(default)]
    pub optional: bool,
    pub description: Option<String>,
}

/// Package metadata section.
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
}

/// Program manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Package metadata
    pub package: Package,
    /// Runtime requirements (name -> version)
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub runtime: BTreeMap<String, String>,
    /// Parameter definitions
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, Parameter>,
    /// Dependencies (name -> version)
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dependencies: BTreeMap<String, String>,
}

/// Build manifest URL for a program from registry.
pub fn manifest_url(registry_url: &str, name: &ProgramName) -> String {
    format!(
        "{}/api/v1/inferlets/{}/{}/manifest",
        registry_url.trim_end_matches('/'),
        name.name,
        name.version
    )
}

impl Manifest {
    /// Parse a manifest from TOML content string.
    pub fn parse(content: &str) -> Result<Self> {
        toml::from_str(content)
            .map_err(|e| anyhow!("Failed to parse manifest TOML: {}", e))
    }

    /// Serialize manifest to TOML string.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize manifest: {}", e))
    }

    /// Get program name (name + version).
    pub fn program_name(&self) -> ProgramName {
        ProgramName {
            name: self.package.name.clone(),
            version: self.package.version.clone(),
        }
    }

    /// Get dependencies as ProgramName list.
    pub fn dependency_names(&self) -> Vec<ProgramName> {
        self.dependencies
            .iter()
            .map(|(name, version)| ProgramName {
                name: name.clone(),
                version: version.clone(),
            })
            .collect()
    }

    /// Fetch and parse manifest from a registry URL.
    pub async fn from_url(registry_url: &str, name: &ProgramName) -> Result<Self> {
        let url = manifest_url(registry_url, name);

        let response = reqwest::get(&url).await.map_err(|e| {
            anyhow!("Failed to fetch manifest from {}: {}", url, e)
        })?;

        if !response.status().is_success() {
            bail!(
                "Failed to fetch manifest: {} returned {}",
                url,
                response.status()
            );
        }

        let manifest_content = response.text().await.map_err(|e| {
            anyhow!("Failed to read manifest response: {}", e)
        })?;

        Self::parse(&manifest_content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANONICAL_MANIFEST: &str = r#"
[package]
name = "text-completion"
version = "0.1.0"
description = "Text completion inferlet"
authors = ["Alice <alice@example.com>"]
repository = "https://github.com/pie-project/pie"
readme = "README.md"

[runtime]
core = "0.2.0"
mcp = "0.2.0"

[parameters]
prompt = { type = "string", description = "The user message to complete" }
max_tokens = { type = "int", optional = true, description = "Maximum tokens to generate" }
temperature = { type = "float", optional = true, description = "Sampling temperature" }

[dependencies]
foo = "0.1.0"
bar = "0.2.0"
"#;

    #[test]
    fn test_parse_canonical_manifest() {
        let manifest = Manifest::parse(CANONICAL_MANIFEST).unwrap();
        
        assert_eq!(manifest.package.name, "text-completion");
        assert_eq!(manifest.package.version, "0.1.0");
        assert_eq!(manifest.package.description, Some("Text completion inferlet".to_string()));
        assert_eq!(manifest.package.authors, vec!["Alice <alice@example.com>"]);
        assert_eq!(manifest.package.repository, Some("https://github.com/pie-project/pie".to_string()));
        assert_eq!(manifest.package.readme, Some("README.md".to_string()));
        
        assert_eq!(manifest.runtime.get("core"), Some(&"0.2.0".to_string()));
        assert_eq!(manifest.runtime.get("mcp"), Some(&"0.2.0".to_string()));
        
        assert_eq!(manifest.parameters.len(), 3);
        assert!(manifest.parameters.contains_key("prompt"));
        assert!(manifest.parameters.contains_key("max_tokens"));
        assert!(manifest.parameters.contains_key("temperature"));
        
        let prompt = &manifest.parameters["prompt"];
        assert_eq!(prompt.param_type, ParameterType::String);
        assert!(!prompt.optional);
        
        let max_tokens = &manifest.parameters["max_tokens"];
        assert_eq!(max_tokens.param_type, ParameterType::Int);
        assert!(max_tokens.optional);
        
        assert_eq!(manifest.dependencies.len(), 2);
        assert_eq!(manifest.dependencies.get("foo"), Some(&"0.1.0".to_string()));
        assert_eq!(manifest.dependencies.get("bar"), Some(&"0.2.0".to_string()));
    }

    #[test]
    fn test_parse_minimal_manifest() {
        let minimal = r#"
[package]
name = "minimal"
version = "1.0.0"
"#;
        let manifest = Manifest::parse(minimal).unwrap();
        
        assert_eq!(manifest.package.name, "minimal");
        assert_eq!(manifest.package.version, "1.0.0");
        assert!(manifest.package.description.is_none());
        assert!(manifest.package.authors.is_empty());
        assert!(manifest.runtime.is_empty());
        assert!(manifest.parameters.is_empty());
        assert!(manifest.dependencies.is_empty());
    }

    #[test]
    fn test_roundtrip_serialization() {
        let manifest = Manifest::parse(CANONICAL_MANIFEST).unwrap();
        let serialized = manifest.to_toml().unwrap();
        let reparsed = Manifest::parse(&serialized).unwrap();
        
        // With PartialEq, we can assert full equality
        assert_eq!(manifest, reparsed);
    }

    #[test]
    fn test_program_name() {
        let manifest = Manifest::parse(CANONICAL_MANIFEST).unwrap();
        let name = manifest.program_name();
        
        assert_eq!(name.name, "text-completion");
        assert_eq!(name.version, "0.1.0");
    }

    #[test]
    fn test_dependency_names() {
        let manifest = Manifest::parse(CANONICAL_MANIFEST).unwrap();
        let deps = manifest.dependency_names();
        
        assert_eq!(deps.len(), 2);
        assert!(deps.iter().any(|d| d.name == "foo" && d.version == "0.1.0"));
        assert!(deps.iter().any(|d| d.name == "bar" && d.version == "0.2.0"));
    }
}
