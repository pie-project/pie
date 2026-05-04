//! `pie run <inferlet>` — one-shot inferlet launcher.
//!
//! Mirrors `pie/src/pie_cli/commands/run.py`. Boots an in-process
//! one-shot engine, connects via `pie-client`, launches the inferlet,
//! relays process events, and tears the engine down on completion.

use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;

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

    /// Inferlet input as a JSON string. Defaults to `{}`.
    #[arg(long, default_value = "{}")]
    pub input: String,

    /// Include inferlet stdout as it is produced. Without this flag,
    /// only the final return value is printed.
    #[arg(long = "stdout")]
    pub relay_stdout: bool,

    /// Suppress `pie run` progress chatter on stderr.
    #[arg(short = 'q', long)]
    pub quiet: bool,
}

pub fn run(args: RunArgs) -> Result<()> {
    if args.inferlet.is_none() && args.path.is_none() {
        bail!("specify an inferlet name or --path");
    }
    if args.path.is_some() && args.manifest.is_none() {
        bail!("--manifest is required when --path is used");
    }

    let cfg_path = args
        .config
        .clone()
        .unwrap_or_else(paths::default_config_path);
    let mut cfg = config::Config::from_toml_file(&cfg_path)
        .with_context(|| format!("loading TOML config from {cfg_path:?}"))?;
    if let Some(p) = args.port {
        cfg.server.port = p;
    }
    // Heavy testing phase: always show startup details, even for old
    // temp configs that explicitly set `verbose = false`.
    cfg.server.verbose = true;

    crate::py_runtime::ensure_installed_best_effort();
    let runtime = serve::build_runtime(&cfg)?;

    runtime.block_on(async move {
        let engine = serve::start_engine(cfg)
            .await
            .context("starting one-shot engine")?;
        let url = engine.url.clone();
        let token = engine.token.clone();
        let result = drive_inferlet(&url, &token, &args).await;

        // Print the error before teardown so failures are visible even
        // when shutdown logging follows immediately.
        if let Err(e) = &result {
            eprintln!("pie run: {e:#}");
        }

        shutdown_engine_bounded(engine, args.quiet);

        result
    })
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

    let mut process = client
        .launch_process(
            inferlet_id.clone(),
            args.input.clone(),
            /*capture_outputs=*/ true,
            /*token_budget=*/ None,
        )
        .await
        .with_context(|| format!("launch_process {inferlet_id}"))?;

    if !args.quiet {
        eprintln!("(launched {inferlet_id} as {})", process.id());
    }

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
                let _ = stderr.write_all(s.as_bytes());
                let _ = stderr.flush();
            }
            ProcessEvent::Message(m) => {
                eprintln!("[message] {m}");
            }
            ProcessEvent::File(bytes) => {
                eprintln!("[received file: {} bytes]", bytes.len());
            }
            ProcessEvent::Return(v) => {
                println!("{v}");
                break;
            }
            ProcessEvent::Error(e) => {
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
