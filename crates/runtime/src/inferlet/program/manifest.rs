//! Parsing and handling for inferlet manifest TOML files.

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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub package: Package,
    /// name -> version
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub runtime: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, Parameter>,
    /// name -> version
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dependencies: BTreeMap<String, String>,
}

pub fn manifest_url(registry_url: &str, name: &ProgramName) -> String {
    format!(
        "{}/api/v1/inferlets/{}/{}/manifest",
        registry_url.trim_end_matches('/'),
        name.name,
        name.version
    )
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

    /// Declared python-runtime version, if this program requires one.
    pub fn python_runtime(&self) -> Option<&str> {
        self.runtime.get("python-runtime").map(String::as_str)
    }

    pub async fn from_url(registry_url: &str, name: &ProgramName) -> Result<Self> {
        let url = manifest_url(registry_url, name);

        let response = reqwest::get(&url)
            .await
            .map_err(|e| anyhow!("Failed to fetch manifest from {}: {}", url, e))?;

        if !response.status().is_success() {
            bail!(
                "Failed to fetch manifest: {} returned {}",
                url,
                response.status()
            );
        }

        let manifest_content = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read manifest response: {}", e))?;

        Self::parse(&manifest_content)
    }
}

