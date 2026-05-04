//! `pie run <inferlet>` — one-shot inferlet launcher.
//!
//! Mirrors `pie/src/pie_cli/commands/run.py`. Boots a child `pie serve`
//! (in a subprocess so we don't have to factor the engine's blocking
//! shutdown wait out of `serve.rs`), waits for its readiness lines,
//! connects via `pie-client`, launches the inferlet, streams events,
//! and tears the child down on completion.

use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;

use pie_client::client::{Client, ProcessEvent};

use crate::{config, paths};

const READY_TIMEOUT: Duration = Duration::from_secs(60);

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

    /// Override `[server].port` for the spawned engine.
    #[arg(long)]
    pub port: Option<u16>,

    /// Inferlet input as a JSON string. Defaults to `{}`.
    #[arg(long, default_value = "{}")]
    pub input: String,
}

pub fn run(args: RunArgs) -> Result<()> {
    if args.inferlet.is_none() && args.path.is_none() {
        bail!("specify an inferlet name or --path");
    }
    if args.path.is_some() && args.manifest.is_none() {
        bail!("--manifest is required when --path is used");
    }

    // Async runtime for pie-client. The subprocess + readiness handshake
    // runs sync below and hands the connected client back into the
    // tokio runtime for the inferlet streaming loop.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run_async(args))
}

async fn run_async(args: RunArgs) -> Result<()> {
    let cfg_path = args.config.clone().unwrap_or_else(paths::default_config_path);
    let mut cfg = config::Config::from_toml_file(&cfg_path)
        .with_context(|| format!("loading TOML config from {cfg_path:?}"))?;
    if let Some(p) = args.port {
        cfg.server.port = p;
    }

    // Write the (possibly overridden) config to a temp file the
    // subprocess can read. Doing this even when no overrides are
    // present keeps a single code path.
    let runtime_cfg_path = std::env::temp_dir()
        .join(format!("pie-run-{}.toml", std::process::id()));
    let serialized = toml::to_string(&cfg).map_err(|e| anyhow!("serialize cfg: {e}"))?;
    std::fs::write(&runtime_cfg_path, serialized)
        .map_err(|e| anyhow!("write {runtime_cfg_path:?}: {e}"))?;

    // Locate ourselves; spawn `pie serve --config <tmp>`. We need our
    // own argv[0] so we re-enter the same binary (and embed the same
    // driver flavor).
    let pie_bin = std::env::current_exe().context("current_exe")?;
    let mut child = Command::new(&pie_bin)
        .arg("serve")
        .arg("--config")
        .arg(&runtime_cfg_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawn `{} serve`", pie_bin.display()))?;

    let stdout = child.stdout.take().expect("piped stdout");
    let pid = child.id();

    // Drain stdout in a worker thread, fishing out the auth token.
    // Pass the rest through to our own stdout so the user sees engine
    // boot output.
    let (tx, rx) = mpsc::channel::<TokenOrEof>();
    thread::spawn(move || drain_stdout_for_token(stdout, tx));

    let token = wait_for_token(rx)
        .with_context(|| "engine did not produce an auth token")?;

    let url = format!("ws://{}:{}", cfg.server.host, cfg.server.port);
    let result = drive_inferlet(&url, &token, &args).await;

    // Tear the child down regardless of result.
    eprintln!("cleaning up engine (pid={pid})…");
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    let _ = child.wait();
    let _ = std::fs::remove_file(&runtime_cfg_path);

    result
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

    eprintln!("(launched {inferlet_id} as {})", process.id());

    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    loop {
        match process.recv().await? {
            ProcessEvent::Stdout(s) => {
                let _ = stdout.write_all(s.as_bytes());
                let _ = stdout.flush();
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
    let _ = client.close().await;
    Ok(())
}

/// Read the manifest TOML and return `name@version`. Mirrors the same
/// resolve in `pie_cli/commands/run.py`.
fn manifest_id(manifest: &std::path::Path) -> Result<String> {
    let content = std::fs::read_to_string(manifest)
        .map_err(|e| anyhow!("read {manifest:?}: {e}"))?;
    let v: toml::Value = toml::from_str(&content)
        .map_err(|e| anyhow!("parse {manifest:?}: {e}"))?;
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

// -----------------------------------------------------------------------------
// Token capture from child stdout
// -----------------------------------------------------------------------------

enum TokenOrEof {
    Token(String),
    Eof,
}

/// Reads child stdout line-by-line. Forwards everything to our own
/// stdout (so the user sees engine progress) and sends the token over
/// `tx` once it appears.
fn drain_stdout_for_token<R: std::io::Read>(
    stdout: R,
    tx: mpsc::Sender<TokenOrEof>,
) {
    let reader = BufReader::new(stdout);
    let is_tty = std::io::stdout().is_terminal();
    for line in reader.lines() {
        let Ok(line) = line else {
            break;
        };
        if let Some(rest) = line.strip_prefix("internal token: ") {
            let _ = tx.send(TokenOrEof::Token(rest.trim().to_string()));
        }
        // Suppress engine chatter when piping to a TTY would garble
        // inferlet output; users running interactively still see it
        // via stderr below.
        if is_tty {
            eprintln!("[engine] {line}");
        }
    }
    let _ = tx.send(TokenOrEof::Eof);
}

fn wait_for_token(rx: mpsc::Receiver<TokenOrEof>) -> Result<String> {
    let deadline = std::time::Instant::now() + READY_TIMEOUT;
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            bail!("timed out waiting {READY_TIMEOUT:?} for engine ready");
        }
        match rx.recv_timeout(deadline - now) {
            Ok(TokenOrEof::Token(t)) => return Ok(t),
            Ok(TokenOrEof::Eof) => bail!("engine exited before producing a token"),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                bail!("timed out waiting {READY_TIMEOUT:?} for engine ready")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("engine stdout reader disconnected")
            }
        }
    }
}
