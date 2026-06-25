//! `pie run <inferlet>` — one-shot inferlet launcher.
//!
//! Mirrors `pie/src/pie_cli/commands/run.py`. Boots an in-process
//! one-shot engine, connects via `pie-client`, launches the inferlet,
//! relays process events, and tears the engine down on completion.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use serde::Deserialize;

use pie_client::client::{Client, ProcessEvent};

use crate::{config, paths, serve};

const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Inferlet name from the registry (e.g. `text-completion@0.1.0`).
    pub inferlet: Option<String>,

    /// Local `.wasm` path. Mutually exclusive with `inferlet`; requires
    /// `--manifest`.
    #[arg(short = 'p', long)]
    pub path: Option<PathBuf>,

    /// Path to the manifest TOML when using `--path`.
    #[arg(short = 'm', long)]
    pub manifest: Option<PathBuf>,

    /// Config TOML to use. Defaults to `~/.pie/config.toml`.
    #[arg(short = 'c', long)]
    pub config: Option<PathBuf>,

    /// Override `[server].port` for the one-shot engine.
    #[arg(long)]
    pub port: Option<u16>,

    /// Inferlet input as a JSON string. Defaults to `{}`. Mutually
    /// exclusive with trailing `-- --key value` args.
    #[arg(long, default_value = "{}")]
    pub input: String,

    /// Include inferlet stdout/stderr as it is produced. Without this
    /// flag, only the final return value is printed.
    #[arg(long = "stdout")]
    pub relay_stdout: bool,

    /// Show engine, driver, server, and inferlet diagnostics.
    #[arg(long)]
    pub debug: bool,

    /// Suppress `pie run` progress chatter on stderr.
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Trailing `--key value` args become the inferlet input JSON.
    /// `--max-tokens 64` → `{"max_tokens": 64}` (kebab → snake, with
    /// int/float/bool inference). Bare `--flag` is `true`. Use `--`
    /// to separate from `pie run` flags.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub extra: Vec<String>,
}

pub fn run(mut args: RunArgs) -> Result<()> {
    normalize_path_args(&mut args)?;
    if args.inferlet.is_none() && args.path.is_none() {
        bail!("specify an inferlet name or --path");
    }
    if args.path.is_some() && args.manifest.is_none() {
        bail!("--manifest is required when --path is used");
    }
    if !args.extra.is_empty() {
        if args.input != "{}" {
            bail!("--input and trailing `-- --key value` args are mutually exclusive");
        }
        args.input = serde_json::to_string(&cli_args_to_json(&args.extra))
            .expect("Map<String, Value> is always serializable");
    }

    let cfg_path = args
        .config
        .clone()
        .unwrap_or_else(paths::default_config_path);
    let mut cfg = config::Config::from_toml_file(&cfg_path)
        .with_context(|| format!("loading TOML config from {cfg_path:?}"))?;
    cfg.server.port = args.port.unwrap_or(0);
    cfg.server.verbose = args.debug;

    crate::py_runtime::ensure_installed_best_effort();
    let runtime = serve::build_runtime(&cfg)?;

    runtime.block_on(async move {
        if args.path.is_none() {
            if let Some(inferlet) = args.inferlet.take() {
                args.inferlet = Some(resolve_inferlet_id(&inferlet, &cfg.server.registry).await?);
            }
        }

        let model_label = cfg
            .models
            .first()
            .map(|m| m.hf_repo.clone())
            .unwrap_or_else(|| "default".to_string());
        let status = RunStatus::start(args.quiet || args.debug, model_label);
        // One-shot `pie run` is always single-node: embed an in-proc controller.
        let control_addr = format!("{}:{}", cfg.server.host, cfg.server.port);
        let coordinator =
            match serve::coordination::connect(&serve::TopologyMode::SingleNode, control_addr) {
                Ok(c) => c,
                Err(e) => {
                    status.fail("Engine failed");
                    return Err(e.context("joining control plane"));
                }
            };
        let engine = match serve::start_engine(cfg, coordinator)
            .await
            .context("starting one-shot engine")
        {
            Ok(engine) => {
                status.finish();
                engine
            }
            Err(e) => {
                status.fail("Engine failed");
                return Err(e);
            }
        };
        let url = engine.url.clone();
        let token = engine.token.clone();
        let result = drive_inferlet(&url, &token, &args).await;

        // Print the error before teardown so failures are visible even
        // when shutdown logging follows immediately.
        if let Err(e) = &result {
            eprintln!("pie run: {e:#}");
        }

        shutdown_engine_bounded(engine, args.quiet || !args.debug);

        result
    })
}

fn normalize_path_args(args: &mut RunArgs) -> Result<()> {
    if args.path.is_none() {
        return Ok(());
    }

    let Some(first_positional) = args.inferlet.take() else {
        return Ok(());
    };

    if first_positional.starts_with('-') || !args.extra.is_empty() {
        args.extra.insert(0, first_positional);
        return Ok(());
    }

    bail!("--path is mutually exclusive with inferlet name {first_positional:?}");
}

async fn drive_inferlet(url: &str, token: &str, args: &RunArgs) -> Result<()> {
    let client = Client::connect(url)
        .await
        .with_context(|| format!("connect {url}"))?;
    client
        .auth_by_token(token)
        .await
        .with_context(|| "auth_by_token")?;

    // Resolve the inferlet name. With --path we install first; without,
    // we accept either bare names or `name@version`.
    let inferlet_id = if let (Some(path), Some(manifest)) = (&args.path, &args.manifest) {
        if !path.exists() {
            bail!("wasm not found at {path:?}");
        }
        if !manifest.exists() {
            bail!("manifest not found at {manifest:?}");
        }
        client
            .add_program(path, manifest, /*force_overwrite=*/ true)
            .await
            .with_context(|| format!("add_program {path:?}"))?;
        // Read package name + version out of the manifest TOML.
        manifest_id(manifest)?
    } else {
        args.inferlet
            .clone()
            .expect("argument-validation guaranteed inferlet is Some")
    };

    let mut process = match client
        .launch_process(
            inferlet_id.clone(),
            args.input.clone(),
            /*capture_outputs=*/ true,
            /*token_budget=*/ None,
        )
        .await
        .with_context(|| format!("launch_process {inferlet_id}"))
    {
        Ok(process) => process,
        Err(e) => {
            return Err(e);
        }
    };

    if args.debug {
        eprintln!("(launched {inferlet_id} as {})", process.id());
    }

    let mut status = RunStatus::spawn(
        args.quiet || args.debug || args.relay_stdout,
        inferlet_id.clone(),
    );

    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    loop {
        match process.recv().await? {
            ProcessEvent::Stdout(s) => {
                if args.relay_stdout {
                    let _ = stdout.write_all(s.as_bytes());
                    let _ = stdout.flush();
                }
            }
            ProcessEvent::Stderr(s) => {
                if args.relay_stdout || args.debug {
                    let _ = stderr.write_all(s.as_bytes());
                    let _ = stderr.flush();
                }
            }
            ProcessEvent::Message(m) => {
                if args.debug {
                    eprintln!("[message] {m}");
                }
            }
            ProcessEvent::File(bytes) => {
                if args.debug {
                    eprintln!("[received file: {} bytes]", bytes.len());
                }
            }
            ProcessEvent::Return(v) => {
                status.stop();
                print_return_value(&v, args.debug || args.relay_stdout)?;
                break;
            }
            ProcessEvent::Error(e) => {
                status.fail("Inferlet failed");
                eprintln!("✗ {e}");
                bail!("inferlet errored: {e}");
            }
        }
    }
    // `run` is a one-shot CLI: once the process returned, teardown is
    // handled by the caller immediately after this function returns.
    let _ = tokio::time::timeout(Duration::from_secs(2), client.close()).await;
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

async fn resolve_inferlet_id(inferlet: &str, registry_url: &str) -> Result<String> {
    let (name, should_resolve) = match inferlet.split_once('@') {
        None => (inferlet, true),
        Some((name, "latest")) => (name, true),
        Some(_) => return Ok(inferlet.to_string()),
    };

    validate_bare_inferlet_name(name)?;
    if !should_resolve {
        return Ok(inferlet.to_string());
    }

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
    let latest = latest_version_from_registry_json(&body)
        .with_context(|| format!("resolve latest version for {name:?}"))?;
    Ok(format!("{name}@{latest}"))
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

/// Read the manifest TOML and return `name@version`. Mirrors the same
/// resolve in `pie_cli/commands/run.py`.
fn manifest_id(manifest: &std::path::Path) -> Result<String> {
    let content =
        std::fs::read_to_string(manifest).map_err(|e| anyhow!("read {manifest:?}: {e}"))?;
    let v: toml::Value =
        toml::from_str(&content).map_err(|e| anyhow!("parse {manifest:?}: {e}"))?;
    let name = v
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .ok_or_else(|| anyhow!("manifest missing [package].name"))?;
    let version = v
        .get("package")
        .and_then(|p| p.get("version"))
        .and_then(|n| n.as_str())
        .ok_or_else(|| anyhow!("manifest missing [package].version"))?;
    Ok(format!("{name}@{version}"))
}

/// Convert trailing CLI tokens like `--key value -k v --flag positional` to a
/// JSON object. Mirrors `_cli_args_to_dict` from the legacy Python CLI.
fn cli_args_to_json(args: &[String]) -> serde_json::Value {
    use serde_json::{Map, Value};
    let mut obj: Map<String, Value> = Map::new();
    let mut positional: Vec<Value> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(rest) = a.strip_prefix("--") {
            let key = rest.replace('-', "_");
            let next_is_value = matches!(args.get(i + 1), Some(n) if !n.starts_with('-'));
            if next_is_value {
                obj.insert(key, parse_cli_value(&args[i + 1]));
                i += 2;
            } else {
                obj.insert(key, Value::Bool(true));
                i += 1;
            }
        } else if a.starts_with('-') && a.chars().count() == 2 {
            let key = a[1..].to_string();
            if let Some(next) = args.get(i + 1) {
                obj.insert(key, parse_cli_value(next));
                i += 2;
            } else {
                i += 1;
            }
        } else {
            positional.push(parse_cli_value(a));
            i += 1;
        }
    }
    if !positional.is_empty() {
        obj.insert("_positional".into(), Value::Array(positional));
    }
    Value::Object(obj)
}

/// Infer an int → float → bool → string for a single CLI value.
fn parse_cli_value(s: &str) -> serde_json::Value {
    if let Ok(n) = s.parse::<i64>() {
        return serde_json::Value::Number(n.into());
    }
    if let Ok(f) = s.parse::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(f) {
            return serde_json::Value::Number(num);
        }
    }
    match s {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        _ => serde_json::Value::String(s.to_string()),
    }
}

fn shutdown_engine_bounded(engine: serve::EngineHandle, quiet: bool) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        engine.shutdown();
        let _ = tx.send(());
    });

    if !quiet {
        eprintln!("cleaning up engine...");
    }
    if rx.recv_timeout(SHUTDOWN_GRACE).is_err() {
        eprintln!("engine shutdown did not finish within {SHUTDOWN_GRACE:?}; exiting");
    }
}

struct RunStatus {
    done: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    enabled: bool,
}

impl RunStatus {
    fn start(disabled: bool, model_label: String) -> Self {
        Self::spawn(disabled, format!("loading model ({model_label})"))
    }

    fn spawn(disabled: bool, label: String) -> Self {
        let enabled = !disabled && io::stderr().is_terminal();
        if !enabled {
            return Self {
                done: Arc::new(AtomicBool::new(true)),
                handle: None,
                enabled,
            };
        }

        let done = Arc::new(AtomicBool::new(false));
        let thread_done = Arc::clone(&done);
        let handle = std::thread::spawn(move || {
            const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut idx = 0usize;
            while !thread_done.load(Ordering::Relaxed) {
                eprint!("\r\x1b[K{} {}", FRAMES[idx % FRAMES.len()], label);
                let _ = io::stderr().flush();
                idx = idx.wrapping_add(1);
                std::thread::sleep(Duration::from_millis(90));
            }
        });
        Self {
            done,
            handle: Some(handle),
            enabled,
        }
    }

    fn finish(mut self) {
        self.stop();
    }

    fn fail(mut self, message: &str) {
        self.stop();
        if self.enabled {
            eprintln!("\x1b[31m✗\x1b[0m {message}");
        }
    }

    fn stop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        if self.enabled {
            eprint!("\r\x1b[K");
            let _ = io::stderr().flush();
        }
    }
}

impl Drop for RunStatus {
    fn drop(&mut self) {
        self.stop();
    }
}

fn print_return_value(value: &str, rich: bool) -> Result<()> {
    if !rich || !io::stdout().is_terminal() {
        println!("{value}");
        return Ok(());
    }

    let display = serde_json::from_str::<serde_json::Value>(value)
        .ok()
        .and_then(|json| serde_json::to_string_pretty(&json).ok())
        .unwrap_or_else(|| value.to_string());
    for line in display.lines() {
        println!("{}", highlight_json_line(line));
    }
    Ok(())
}

fn highlight_json_line(line: &str) -> String {
    const RESET: &str = "\x1b[0m";
    const KEY: &str = "\x1b[1;36m";
    const STRING: &str = "\x1b[32m";
    const NUMBER: &str = "\x1b[33m";
    const KEYWORD: &str = "\x1b[35m";
    const PUNCT: &str = "\x1b[90m";

    let mut out = String::with_capacity(line.len() + 32);
    let mut i = 0usize;
    let bytes = line.as_bytes();
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            '"' => {
                let start = i;
                i += 1;
                let mut escaped = false;
                while i < bytes.len() {
                    let ch = bytes[i] as char;
                    if escaped {
                        escaped = false;
                    } else if ch == '\\' {
                        escaped = true;
                    } else if ch == '"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                let token = &line[start..i.min(line.len())];
                let rest = &line[i.min(line.len())..];
                let is_key = rest.trim_start().starts_with(':');
                out.push_str(if is_key { KEY } else { STRING });
                out.push_str(token);
                out.push_str(RESET);
            }
            '-' | '0'..='9' => {
                let start = i;
                i += 1;
                while i < bytes.len()
                    && matches!(bytes[i] as char, '0'..='9' | '.' | 'e' | 'E' | '+' | '-')
                {
                    i += 1;
                }
                out.push_str(NUMBER);
                out.push_str(&line[start..i]);
                out.push_str(RESET);
            }
            't' | 'f' | 'n' => {
                let rest = &line[i..];
                let keyword = ["true", "false", "null"]
                    .iter()
                    .find(|kw| rest.starts_with(**kw));
                if let Some(keyword) = keyword {
                    i += keyword.len();
                    out.push_str(KEYWORD);
                    out.push_str(keyword);
                    out.push_str(RESET);
                } else {
                    out.push(c);
                    i += 1;
                }
            }
            '{' | '}' | '[' | ']' | ':' | ',' => {
                out.push_str(PUNCT);
                out.push(c);
                out.push_str(RESET);
                i += 1;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_run_args(inferlet: Option<&str>, extra: Vec<&str>) -> RunArgs {
        RunArgs {
            inferlet: inferlet.map(str::to_string),
            path: Some(PathBuf::from("out.wasm")),
            manifest: Some(PathBuf::from("Pie.toml")),
            config: None,
            port: None,
            input: "{}".to_string(),
            relay_stdout: false,
            debug: false,
            quiet: true,
            extra: extra.into_iter().map(str::to_string).collect(),
        }
    }

    #[test]
    fn path_run_keeps_documented_trailing_flags_as_input() {
        let mut args = local_run_args(
            Some("--prompt"),
            vec!["The capital of France is", "--max-tokens", "4"],
        );

        normalize_path_args(&mut args).unwrap();

        assert!(args.inferlet.is_none());
        let input = cli_args_to_json(&args.extra);
        assert_eq!(input["prompt"], "The capital of France is");
        assert_eq!(input["max_tokens"], 4);
    }

    #[test]
    fn path_run_rejects_inferlet_name_without_extra_input() {
        let mut args = local_run_args(Some("text-completion"), vec![]);

        let err = normalize_path_args(&mut args).unwrap_err();

        assert!(
            err.to_string()
                .contains("--path is mutually exclusive with inferlet name")
        );
    }

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
