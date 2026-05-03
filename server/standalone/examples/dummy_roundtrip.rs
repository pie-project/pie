//! End-to-end round-trip test: launch the three marketing-tab inferlets
//! against the live `pie-standalone` + `driver-dummy` stack.
//!
//! Run with:
//!
//! ```bash
//! cargo build -p pie-standalone --no-default-features \
//!     --features driver-dummy --release
//! cd inferlets/marketing-tab1-agent && cargo build --target wasm32-wasip2 --release
//! cd inferlets/marketing-tab2-watermark && cargo build --target wasm32-wasip2 --release
//! cd inferlets/marketing-tab3-lora-spec && cargo build --target wasm32-wasip2 --release
//! cargo run --example dummy_roundtrip -p pie-standalone \
//!     --no-default-features --features driver-dummy --release \
//!     -- /path/to/dir/with/tokenizer.json
//! ```
//!
//! The argument is a HuggingFace snapshot directory containing
//! `tokenizer.json`. The dummy driver doesn't load weights, but the
//! runtime side instantiates a real tokenizer from this path.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};

use pie_client::client::{Client, ProcessEvent};

const PORT: u16 = 8093;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
const PER_TAB_TIMEOUT: Duration = Duration::from_secs(60);

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let snapshot_dir = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!(
            "usage: dummy_roundtrip <snapshot_dir> — pass a HF snapshot dir \
             containing tokenizer.json"
        ))?;
    ensure_dir_with_tokenizer(Path::new(&snapshot_dir))?;

    let repo_root = repo_root()?;
    let pie_bin = repo_root.join("target/release/pie");
    if !pie_bin.is_file() {
        anyhow::bail!(
            "pie binary not found at {pie_bin:?}; build it first with \
             `cargo build -p pie-standalone --no-default-features \
             --features driver-dummy --release`"
        );
    }

    let tabs = [
        Tab {
            name: "marketing-tab1-agent",
            wasm: repo_root.join("inferlets/marketing-tab1-agent/target/wasm32-wasip2/release/marketing_tab1_agent.wasm"),
            manifest: repo_root.join("inferlets/marketing-tab1-agent/Pie.toml"),
            input: r#"{"prompt": "What is the weather in Tokyo today?"}"#,
        },
        Tab {
            name: "marketing-tab2-watermark",
            wasm: repo_root.join("inferlets/marketing-tab2-watermark/target/wasm32-wasip2/release/marketing_tab2_watermark.wasm"),
            manifest: repo_root.join("inferlets/marketing-tab2-watermark/Pie.toml"),
            input: r#"{"prompt": "Write a haiku about Rust.", "max_tokens": 16}"#,
        },
        Tab {
            name: "marketing-tab3-lora-spec",
            wasm: repo_root.join("inferlets/marketing-tab3-lora-spec/target/wasm32-wasip2/release/marketing_tab3_lora_spec.wasm"),
            manifest: repo_root.join("inferlets/marketing-tab3-lora-spec/Pie.toml"),
            input: r#"{"prompt": "Solve: 2+2"}"#,
        },
    ];
    for t in &tabs {
        if !t.wasm.is_file() {
            anyhow::bail!(
                "wasm not found for {}: {:?} — build with \
                 `cd inferlets/{} && cargo build --target wasm32-wasip2 --release`",
                t.name, t.wasm, t.name
            );
        }
    }

    // Write the standalone config — `allow_fs = true` so tab 3 can write
    // to `/scratch`; `allow_network` defaults to true.
    let config_path = std::env::temp_dir().join("pie-dummy-roundtrip.toml");
    std::fs::write(&config_path, write_config(&snapshot_dir))?;

    // Spawn the server. Capture stdout so we can fish out the ws token.
    let mut child = Command::new(&pie_bin)
        .arg("--config")
        .arg(&config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("spawn pie subprocess")?;
    let stdout = child.stdout.take().expect("piped stdout");

    // The child writes:
    //   pie-standalone serving on 127.0.0.1:8093 (...)
    //   internal token: <TOKEN>
    //   press Ctrl-C to shut down
    let (tx, rx) = mpsc::channel::<StartupSignal>();
    let _reader = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let mut serving = false;
        let mut token: Option<String> = None;
        while reader.read_line(&mut line).map_or(false, |n| n > 0) {
            print!("[pie] {line}");
            if line.contains("pie-standalone serving on") {
                serving = true;
            } else if let Some(rest) = line.strip_prefix("internal token: ") {
                token = Some(rest.trim().to_string());
            }
            if serving {
                if let Some(t) = token.clone() {
                    let _ = tx.send(StartupSignal { token: t });
                    break;
                }
            }
            line.clear();
        }
        // Continue draining so the child's pipe doesn't fill up.
        let mut line = String::new();
        while reader.read_line(&mut line).map_or(false, |n| n > 0) {
            print!("[pie] {line}");
            line.clear();
        }
    });

    let startup = rx
        .recv_timeout(STARTUP_TIMEOUT)
        .map_err(|_| anyhow!("standalone did not become ready within {STARTUP_TIMEOUT:?}"))?;
    eprintln!("[runner] standalone is ready, token captured");

    // Cleanup-on-drop guard so we kill the subprocess on panic / early
    // return as well as on the happy path.
    let _kill = KillOnDrop(&mut child);

    let result = run_all_tabs(&startup, &tabs).await;

    // Emit a success / fail summary even on error so the loop output is
    // greppable.
    match &result {
        Ok(()) => eprintln!("[runner] ✓ all tabs completed without panic"),
        Err(e) => eprintln!("[runner] ✗ {e:#}"),
    }
    result
}

struct Tab {
    name: &'static str,
    wasm: PathBuf,
    manifest: PathBuf,
    input: &'static str,
}

struct StartupSignal {
    token: String,
}

struct KillOnDrop<'a>(&'a mut std::process::Child);
impl Drop for KillOnDrop<'_> {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn run_all_tabs(startup: &StartupSignal, tabs: &[Tab]) -> Result<()> {
    let ws_url = format!("ws://127.0.0.1:{PORT}");
    let client = Client::connect(&ws_url)
        .await
        .with_context(|| format!("connect {ws_url}"))?;
    client
        .auth_by_token(&startup.token)
        .await
        .context("auth_by_token")?;
    eprintln!("[runner] authenticated");

    for t in tabs {
        eprintln!("\n[runner] === {} ===", t.name);
        run_one_tab(&client, t).await?;
    }
    Ok(())
}

async fn run_one_tab(client: &Client, t: &Tab) -> Result<()> {
    client
        .add_program(&t.wasm, &t.manifest, /*force_overwrite=*/ true)
        .await
        .with_context(|| format!("{}: add_program", t.name))?;
    eprintln!("[runner] {}: program installed", t.name);

    let inferlet = format!("{}@0.1.0", t.name);
    let mut process = client
        .launch_process(inferlet, t.input.to_string(), /*capture_outputs=*/ true, None)
        .await
        .with_context(|| format!("{}: launch_process", t.name))?;
    eprintln!("[runner] {}: launched (pid={})", t.name, process.id());

    let deadline = std::time::Instant::now() + PER_TAB_TIMEOUT;
    loop {
        if std::time::Instant::now() >= deadline {
            anyhow::bail!("{}: timed out after {:?}", t.name, PER_TAB_TIMEOUT);
        }
        let evt = tokio::time::timeout(Duration::from_secs(2), process.recv()).await;
        let Ok(evt) = evt else { continue };
        match evt {
            Ok(ProcessEvent::Stdout(s)) => eprint!("[{} stdout] {s}", t.name),
            Ok(ProcessEvent::Stderr(s)) => eprint!("[{} stderr] {s}", t.name),
            Ok(ProcessEvent::Message(_)) => {}
            Ok(ProcessEvent::File(_)) => {}
            Ok(ProcessEvent::Return(s)) => {
                eprintln!("[runner] {}: returned ({} bytes)", t.name, s.len());
                return Ok(());
            }
            Ok(ProcessEvent::Error(e)) => {
                anyhow::bail!("{}: inferlet errored: {e}", t.name);
            }
            Err(e) => {
                anyhow::bail!("{}: event recv failed: {e:#}", t.name);
            }
        }
    }
}

fn ensure_dir_with_tokenizer(p: &Path) -> Result<()> {
    if !p.is_dir() {
        anyhow::bail!("snapshot_dir {p:?} is not a directory");
    }
    if !p.join("tokenizer.json").is_file() {
        anyhow::bail!("snapshot_dir {p:?} is missing tokenizer.json");
    }
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    // CARGO_MANIFEST_DIR is `<repo>/server/standalone`; go up two.
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = here
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("could not derive repo root from {here:?}"))?
        .to_path_buf();
    Ok(root)
}

fn write_config(snapshot_dir: &str) -> String {
    format!(
        r#"
[server]
host = "127.0.0.1"
port = {PORT}

[auth]
enabled = false

[runtime]
allow_fs = true
allow_network = true

[[model]]
name = "default"
hf_repo = "{snapshot_dir}"

[model.driver]
type = "dummy"
device = ["cpu"]

[model.driver.options]
kv_page_size = 16
max_num_kv_pages = 256
max_batch_tokens = 4096
max_batch_size = 128
vocab_size = 151936
arch_name = "qwen3"
max_model_len = 4096
"#
    )
}
