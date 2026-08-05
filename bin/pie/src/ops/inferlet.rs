//! `pie inferlet` — the programs pie runs, in the registry and on disk.
//!
//! `list`/`download`/`remove` all go through `pie_engine`'s `Repository`
//! rather than touching `$PIE_HOME/programs` directly. The engine loads the
//! same cache at boot, so a CLI with its own idea of the layout would be a
//! CLI that can hide a program from the thing meant to run it.


use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand};
use serde::Deserialize;

use pie_engine::inferlet::program::{Manifest, ProgramName, Repository};

use crate::ui::{self, Align, Mark, Palette, Row, Stream, Table};


#[derive(Subcommand, Debug)]
pub enum InferletCmd {
    /// List the inferlets already downloaded.
    List {
        /// Emit one JSON document instead of the table.
        #[arg(long)]
        json: bool,
    },

    /// Show manifest metadata and accepted input parameters.
    Info(InfoArgs),

    /// Download an inferlet from the registry into the local cache.
    Download(TargetArgs),

    /// Delete a downloaded inferlet. Re-downloadable at any time.
    Remove(TargetArgs),
}

#[derive(Args, Debug)]
pub struct TargetArgs {
    /// Inferlet name, with optional version (e.g. `chat-completion` or
    /// `chat-completion@0.1.0`). Bare names take the newest version.
    pub inferlet: String,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Inferlet name, with optional version (e.g. `chat-completion`
    /// or `chat-completion@0.1.0`).
    pub inferlet: String,
    /// Emit one JSON document instead of the report.
    #[arg(long)]
    pub json: bool,
}

pub async fn run(cmd: InferletCmd, global: &startup::GlobalArgs) -> Result<()> {
    match cmd {
        InferletCmd::List { json } => list(json),
        InferletCmd::Info(args) => info(args, global).await,
        InferletCmd::Download(args) => download(args, global).await,
        InferletCmd::Remove(args) => remove(args).await,
    }
}

/// Where the engine keeps downloaded programs. One expression, so the CLI and
/// `pie cache`'s registry entry cannot point at different directories.
fn programs_dir() -> std::path::PathBuf {
    pie_worker::paths::pie_home().join("programs")
}

/// The newest cached version of a bare inferlet name, if it is already here.
///
/// `pub(crate)` for `pie run`, which asks this before the registry. Downloading
/// is about what the registry has, so `download` and `info` still go straight
/// there; running is about what this machine can run, and a program already on
/// disk needs no network to name. That is not a refinement -- the registry does
/// not serve every inferlet in `pie inferlet list` (the test set is local), so
/// registry-first made `pie run <one of those>` fail with a 404 for a program
/// sitting in the cache.
pub(crate) fn cached_version(name: &str) -> Option<ProgramName> {
    open(String::new())
        .cached()
        .into_iter()
        .map(|(program, _, _)| program)
        .filter(|program| program.name == name)
        .max_by(|a, b| version_order(&a.version).cmp(&version_order(&b.version)))
}

/// A `major.minor.patch` string as something that sorts like a version.
///
/// Not `str::cmp`: versions are three numbers, and comparing them as text puts
/// `0.10.0` BELOW `0.9.0` -- so "the newest one cached" would start returning
/// an older build the first time an inferlet reached its tenth minor. The
/// shape is guaranteed by `ProgramName::parse` (`\d+\.\d+\.\d+`); anything that
/// still fails to split falls back to text, which is no worse than what it
/// replaces.
fn version_order(version: &str) -> (u64, u64, u64, String) {
    let mut parts = version.split('.').map(|p| p.parse::<u64>().ok());
    match (parts.next(), parts.next(), parts.next()) {
        (Some(Some(major)), Some(Some(minor)), Some(Some(patch))) => {
            (major, minor, patch, String::new())
        }
        _ => (0, 0, 0, version.to_string()),
    }
}

/// Open the on-disk cache. The registry URL is only needed for downloads, so
/// listing and removing work with an empty one rather than requiring a config.
fn open(registry_url: String) -> Repository {
    let mut repo = Repository::new(registry_url, programs_dir());
    repo.load_program_cache();
    repo
}

fn list(json: bool) -> Result<()> {
    let repo = open(String::new());
    let cached = repo.cached();
    if json {
        return ui::emit_json(&serde_json::json!(
            cached
                .iter()
                .map(|(name, manifest, size)| serde_json::json!({
                    "name": name.name,
                    "version": name.version,
                    "description": manifest.package.description,
                    "bytes": size,
                }))
                .collect::<Vec<_>>()
        ));
    }

    if cached.is_empty() {
        println!("nothing downloaded yet");
        println!("  inferlets arrive on first use, or with `pie inferlet download <name>`");
        return Ok(());
    }
    let palette = Palette::for_stream(Stream::Stdout);
    let mut table = Table::new([Align::Left, Align::Right, Align::Left], 2);
    for (name, manifest, size) in &cached {
        // Descriptions are author-written and unbounded -- one in the test set
        // runs to a paragraph on mask semantics -- so the table cuts the last
        // column to fit. `pie inferlet info` prints the whole thing.
        let description = manifest
            .package
            .description
            .as_deref()
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        table.push(Row::new(
            Mark::Plain,
            [
                format!("{}@{}", name.name, name.version),
                ui::bytes(*size),
                description,
            ],
        ));
    }
    table.print(&palette);
    Ok(())
}

async fn download(args: TargetArgs, global: &startup::GlobalArgs) -> Result<()> {
    let (cfg_path, _) = startup::cli_config_path(global);
    let cfg = crate::derive::load_worker_config(&cfg_path)?;
    let name = resolve_inferlet_id(&args.inferlet, &cfg.server.registry).await?;
    let mut repo = open(cfg.server.registry.clone());
    if repo.exists(&name) {
        println!("{}@{} already downloaded", name.name, name.version);
        return Ok(());
    }
    // `force_overwrite: false` -- the `exists` check above already answered
    // that, and reporting "already downloaded" beats silently doing nothing.
    repo.add_from_registry(&name, false).await?;
    println!("✓ downloaded {}@{}", name.name, name.version);
    Ok(())
}

async fn remove(args: TargetArgs) -> Result<()> {
    // Resolved against the local cache, never the registry: removing is about
    // what is on this disk, and asking the network which version to delete
    // would make the command fail while offline -- and could delete a version
    // other than the one `list` just showed.
    let mut repo = open(String::new());
    let name = match args.inferlet.split_once('@') {
        Some(_) => ProgramName::parse(&args.inferlet)?,
        None => {
            let matching: Vec<ProgramName> = repo
                .cached()
                .into_iter()
                .map(|(name, _, _)| name)
                .filter(|name| name.name == args.inferlet)
                .collect();
            match matching.len() {
                0 => bail!(
                    "{} is not downloaded; `pie inferlet list` shows what is",
                    args.inferlet
                ),
                1 => matching.into_iter().next().unwrap(),
                _ => {
                    let versions: Vec<String> =
                        matching.iter().map(|n| n.version.clone()).collect();
                    bail!(
                        "{} has {} versions downloaded ({}); name the one to remove",
                        args.inferlet,
                        versions.len(),
                        versions.join(", ")
                    );
                }
            }
        }
    };
    if repo.remove(&name)? {
        println!("removed {}@{}", name.name, name.version);
    } else {
        println!("{}@{} is not downloaded", name.name, name.version);
    }
    Ok(())
}

async fn info(args: InfoArgs, global: &startup::GlobalArgs) -> Result<()> {
    // The global `--config` rather than a local one: this reads the registry
    // URL out of the same config the engine would boot from, so resolving it
    // by a different rule than the engine's could point `info` at one registry
    // while `serve` used another.
    let (cfg_path, _) = startup::cli_config_path(global);
    let cfg = crate::derive::load_worker_config(&cfg_path)?;

    // Runs on the ambient `#[tokio::main]` runtime (no nested runtime).
    let program = resolve_inferlet_id(&args.inferlet, &cfg.server.registry).await?;
    let manifest = Manifest::from_url(&cfg.server.registry, &program).await?;

    if args.json {
        // The manifest as the registry serves it, plus the resolved version --
        // which is the part the caller could not have known, since a bare name
        // means "newest".
        return ui::emit_json(&serde_json::json!({
            "name": program.name,
            "version": program.version,
            "description": manifest.package.description,
            "authors": manifest.package.authors,
            "repository": manifest.package.repository,
            "runtime": manifest.runtime,
            "dependencies": manifest.dependencies,
            "parameters": manifest
                .parameters
                .iter()
                .map(|(name, p)| {
                    serde_json::json!({
                        "name": name,
                        "type": format!("{:?}", p.param_type).to_lowercase(),
                        "optional": p.optional,
                        "description": p.description,
                    })
                })
                .collect::<Vec<_>>(),
        }));
    }
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

/// Turn what a person typed into a `name@version`, asking the registry for the
/// version when they did not pin one.
///
/// `pub(crate)` for `pie run`, which resolves exactly the way `download` and
/// `info` do -- a bare name meaning "latest" in one command and something else
/// in another would be its own bug.
pub(crate) async fn resolve_inferlet_id(inferlet: &str, registry_url: &str) -> Result<ProgramName> {
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
    let palette = Palette::for_stream(Stream::Stdout);
    let (bold, dim, cyan, green, reset) = (
        palette.bold(),
        palette.dim(),
        // Cyan is this one screen's own accent for parameter names; the shared
        // palette carries the roles every command uses, not every colour.
        if palette.enabled() { "\x1b[36m" } else { "" },
        palette.green(),
        palette.reset(),
    );

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

    #[test]
    fn versions_order_as_numbers_rather_than_as_text() {
        // The failure this prevents is delayed and quiet: everything is fine
        // until an inferlet ships 0.10.0, at which point "the newest cached
        // version" starts resolving to 0.9.0 and `pie run <name>` silently
        // runs an older build.
        let mut versions = ["0.9.0", "0.10.0", "0.2.0", "1.0.0", "0.10.1"];
        versions.sort_by_key(|v| version_order(v));
        assert_eq!(versions, ["0.2.0", "0.9.0", "0.10.0", "0.10.1", "1.0.0"]);
        assert!(version_order("0.10.0") > version_order("0.9.0"));

        // Anything that is not three numbers still orders deterministically
        // rather than panicking or comparing as if it were 0.0.0 and equal.
        assert_eq!(version_order("not-a-version").3, "not-a-version");
    }
}
