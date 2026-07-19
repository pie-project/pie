//! `pie inferlet info` — inspect registry inferlet metadata.

use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand};
use serde::Deserialize;

use pie_engine::inferlet::program::{Manifest, ProgramName};

use crate::paths;

#[derive(Subcommand, Debug)]
pub enum InferletCmd {
    /// Show manifest metadata and accepted input parameters.
    Info(InfoArgs),
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Inferlet name, with optional version (e.g. `chat-completion`
    /// or `chat-completion@0.1.0`).
    pub inferlet: String,

    /// Config TOML to use for the registry URL. Defaults to
    /// `~/.pie/config.toml`.
    #[arg(short = 'c', long)]
    pub config: Option<PathBuf>,
}

pub async fn run(cmd: InferletCmd) -> Result<()> {
    match cmd {
        InferletCmd::Info(args) => info(args).await,
    }
}

async fn info(args: InfoArgs) -> Result<()> {
    let cfg_path = args.config.unwrap_or_else(paths::default_config_path);
    let cfg = crate::derive::load_worker_config(&cfg_path)?;

    // Runs on the ambient `#[tokio::main]` runtime (no nested runtime).
    let program = resolve_inferlet_id(&args.inferlet, &cfg.server.registry).await?;
    let manifest = Manifest::from_url(&cfg.server.registry, &program).await?;

    print_manifest(&program, &manifest);
    Ok(())
}

#[derive(Deserialize)]
struct RegistryInferlet {
    versions: Vec<RegistryVersion>,
}

#[derive(Deserialize)]
struct RegistryVersion {
    num: String,
}

async fn resolve_inferlet_id(inferlet: &str, registry_url: &str) -> Result<ProgramName> {
    match inferlet.split_once('@') {
        Some((name, "latest")) => {
            validate_bare_inferlet_name(name)?;
            let version = latest_version(name, registry_url).await?;
            Ok(ProgramName {
                name: name.to_string(),
                version,
            })
        }
        Some(_) => ProgramName::parse(inferlet),
        None => {
            validate_bare_inferlet_name(inferlet)?;
            let version = latest_version(inferlet, registry_url).await?;
            Ok(ProgramName {
                name: inferlet.to_string(),
                version,
            })
        }
    }
}

async fn latest_version(name: &str, registry_url: &str) -> Result<String> {
    let url = format!(
        "{}/api/v1/inferlets/{}",
        registry_url.trim_end_matches('/'),
        name
    );
    let resp = reqwest::get(&url)
        .await
        .with_context(|| format!("resolve latest inferlet version from {url}"))?;
    if !resp.status().is_success() {
        bail!(
            "resolve latest inferlet version: {url} returned {}",
            resp.status()
        );
    }
    let body = resp
        .text()
        .await
        .with_context(|| format!("read latest inferlet metadata from {url}"))?;
    latest_version_from_registry_json(&body)
        .with_context(|| format!("resolve latest version for {name:?}"))
}

fn latest_version_from_registry_json(body: &str) -> Result<String> {
    let info: RegistryInferlet =
        serde_json::from_str(body).context("parse registry inferlet metadata")?;
    info.versions
        .into_iter()
        .find(|v| !v.num.is_empty())
        .map(|v| v.num)
        .ok_or_else(|| anyhow!("registry returned no versions"))
}

fn validate_bare_inferlet_name(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        bail!("inferlet name is empty");
    };
    if !first.is_ascii_alphanumeric() {
        bail!("invalid inferlet name {name:?}: must start with an ASCII letter or digit");
    }
    if chars.any(|c| !(c.is_ascii_alphanumeric() || c == '-' || c == '_')) {
        bail!("invalid inferlet name {name:?}: use only ASCII letters, digits, '-' and '_'");
    }
    Ok(())
}

fn print_manifest(program: &ProgramName, manifest: &Manifest) {
    let colorize = std::io::stdout().is_terminal();
    let (bold, dim, cyan, green, reset) = if colorize {
        ("\x1b[1m", "\x1b[2m", "\x1b[36m", "\x1b[32m", "\x1b[0m")
    } else {
        ("", "", "", "", "")
    };

    println!("{bold}{program}{reset}");
    if let Some(description) = &manifest.package.description {
        println!("{description}");
    }
    if let Some(repository) = &manifest.package.repository {
        println!("{dim}{repository}{reset}");
    }

    if manifest.parameters.is_empty() {
        println!("\n{dim}(no parameters){reset}");
        return;
    }

    println!("\n{bold}Parameters{reset}");
    let name_width = manifest
        .parameters
        .keys()
        .map(|name| name.chars().count())
        .max()
        .unwrap_or(4)
        .max("name".len());
    let type_width = manifest
        .parameters
        .values()
        .map(|param| parameter_type_name(&param.param_type).len())
        .max()
        .unwrap_or(4)
        .max("type".len());

    println!(
        "{dim}{:<name_width$}  {:<type_width$}  required  description{reset}",
        "name", "type"
    );
    for (name, param) in &manifest.parameters {
        let required = if param.optional {
            format!("{dim}optional{reset}")
        } else {
            format!("{green}yes{reset}")
        };
        let description = param.description.as_deref().unwrap_or("");
        println!(
            "{cyan}{:<name_width$}{reset}  {:<type_width$}  {:<8}  {}",
            name,
            parameter_type_name(&param.param_type),
            required,
            description
        );
    }
}

fn parameter_type_name(param_type: &pie_engine::inferlet::program::ParameterType) -> &'static str {
    match param_type {
        pie_engine::inferlet::program::ParameterType::String => "string",
        pie_engine::inferlet::program::ParameterType::Int => "int",
        pie_engine::inferlet::program::ParameterType::Float => "float",
        pie_engine::inferlet::program::ParameterType::Bool => "bool",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_version_from_registry_json_uses_first_version() {
        let body = r#"{
            "versions": [
                {"num": "0.2.14"},
                {"num": "0.2.13"}
            ]
        }"#;

        assert_eq!(latest_version_from_registry_json(body).unwrap(), "0.2.14");
    }

    #[test]
    fn bare_inferlet_name_validation_matches_program_names() {
        validate_bare_inferlet_name("text-completion").unwrap();
        validate_bare_inferlet_name("foo_bar-1").unwrap();

        assert!(validate_bare_inferlet_name("").is_err());
        assert!(validate_bare_inferlet_name("-bad").is_err());
        assert!(validate_bare_inferlet_name("bad/name").is_err());
        assert!(validate_bare_inferlet_name("bad.name").is_err());
    }
}
