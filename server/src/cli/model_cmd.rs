//! `pie model { list | download | remove }` — manage HF cache.
//!
//! Mirrors `pie/src/pie_cli/commands/model.py` against the same
//! `~/.cache/huggingface/hub/` layout. Models pulled via either the
//! Python CLI or the standalone are interchangeable.

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use clap::Subcommand;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};

#[derive(Subcommand, Debug)]
pub enum ModelCmd {
    /// List repo IDs already in the local HF cache.
    List,
    /// Download a model snapshot by HuggingFace repo ID.
    Download { repo_id: String },
    /// Remove a cached model by HuggingFace repo ID. Prompts for
    /// confirmation; `--yes` skips the prompt.
    Remove {
        repo_id: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

pub fn run(cmd: ModelCmd) -> Result<()> {
    match cmd {
        ModelCmd::List => list(),
        ModelCmd::Download { repo_id } => download(repo_id),
        ModelCmd::Remove { repo_id, yes } => remove(repo_id, yes),
    }
}

/// HF cache root for model snapshots: `<HF_HOME or ~/.cache/huggingface>/hub/`.
fn hub_dir() -> std::path::PathBuf {
    hf_hub::resolve_cache_dir()
}

/// Convert `models--org--name` ↔ `org/name`. Mirrors
/// `pie_driver_dev.hf_utils.parse_repo_id_from_dirname`.
fn dirname_to_repo_id(dir: &str) -> Option<String> {
    let stripped = dir.strip_prefix("models--")?;
    let parts: Vec<&str> = stripped.split("--").collect();
    match parts.len() {
        1 => Some(parts[0].to_string()),
        2 => Some(format!("{}/{}", parts[0], parts[1])),
        _ => None,
    }
}

fn repo_id_to_dirname(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

// -----------------------------------------------------------------------------
// Pie-compatibility check
// -----------------------------------------------------------------------------

/// HuggingFace `model_type` → PIE arch name. Kept in sync with
/// `driver/dev/src/pie_driver_dev/model/__init__.py`'s `register(...)` calls and
/// the model_type strings the C++ drivers (`driver/cuda/src/loader/`,
/// `driver/portable/src/`) recognise. Architectures supported by *any*
/// of the standalone-linked drivers belong here.
const HF_TO_PIE_ARCH: &[(&str, &str)] = &[
    ("llama", "llama3"),
    ("qwen2", "qwen2"),
    ("qwen3", "qwen3"),
    ("qwen3_5", "qwen3_5"),
    ("qwen3_moe", "qwen3_moe"),
    ("qwen3_5_moe", "qwen3_5_moe"),
    ("qwen3_5_moe_text", "qwen3_5_moe"),
    ("phi3", "phi3"),
    ("mixtral", "mixtral"),
    ("gemma2", "gemma2"),
    ("gemma3_text", "gemma3"),
    ("gemma4_text", "gemma4"),
    ("gemma4", "gemma4"),
    ("mistral3", "mistral3"),
    ("olmo3", "olmo3"),
    ("gptoss", "gptoss"),
    ("gpt_oss", "gptoss"),
];

/// Read `<repo_dir>/snapshots/<latest>/config.json` and look up its
/// `model_type` against [`HF_TO_PIE_ARCH`]. Returns
/// `(true, arch_name)` when supported, `(false, "unsupported type:
/// <model_type>")` when not, or `(false, "no config")` when the
/// snapshot is missing or unreadable.
fn check_pie_compatibility(repo_dir: &Path) -> (bool, String) {
    let snapshots = repo_dir.join("snapshots");
    let snapshot = match std::fs::read_dir(&snapshots) {
        Ok(it) => it
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .next()
            .map(|e| e.path()),
        Err(_) => None,
    };
    let Some(snap) = snapshot else {
        return (false, "no config".to_string());
    };
    let cfg_path = snap.join("config.json");
    let Ok(text) = std::fs::read_to_string(&cfg_path) else {
        return (false, "no config".to_string());
    };
    let Ok(json): serde_json::Result<serde_json::Value> = serde_json::from_str(&text)
    else {
        return (false, "no config".to_string());
    };
    let model_type = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if model_type.is_empty() {
        return (false, "no config".to_string());
    }
    for (hf, pie) in HF_TO_PIE_ARCH {
        if *hf == model_type {
            return (true, pie.to_string());
        }
    }
    (false, format!("unsupported type: {model_type}"))
}

// -----------------------------------------------------------------------------
// list
// -----------------------------------------------------------------------------

fn list() -> Result<()> {
    let hub = hub_dir();
    if !hub.exists() {
        println!("(no HuggingFace cache at {})", hub.display());
        return Ok(());
    }

    let mut entries: Vec<(String, bool, String)> = std::fs::read_dir(&hub)
        .map_err(|e| anyhow!("read {hub:?}: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let repo_id = dirname_to_repo_id(&name)?;
            let (ok, info) = check_pie_compatibility(&e.path());
            Some((repo_id, ok, info))
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    if entries.is_empty() {
        println!("(no models in cache)");
        println!("\n{}", hub.display());
        return Ok(());
    }

    let colorize = std::io::stdout().is_terminal();
    let (green, dim, reset) = if colorize {
        ("\x1b[32m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    for (repo_id, ok, info) in &entries {
        if *ok {
            println!("  {green}✓{reset} {repo_id} {dim}({info}){reset}");
        } else {
            println!("  {dim}○ {repo_id} ({info}){reset}");
        }
    }
    println!("\n{dim}{}{reset}", hub.display());
    Ok(())
}

// -----------------------------------------------------------------------------
// download
// -----------------------------------------------------------------------------

fn download(repo_id: String) -> Result<()> {
    let (owner, name) = parse_repo_id(&repo_id)?;

    println!("Downloading: {repo_id}");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let label = repo_id.clone();
    let snapshot_path = runtime.block_on(async move {
        let client = hf_hub::HFClient::new()
            .map_err(|e| anyhow!("init HF client: {e}"))?;
        let repo = client.model(owner, name);
        let progress = ProgressBar::new();
        let result = repo
            .snapshot_download()
            .progress(progress.clone())
            .send()
            .await
            .map_err(|e| anyhow!("download {label}: {e}"));
        progress.finish();
        result
    })?;
    println!("✓ Downloaded to {}", snapshot_path.display());

    // Post-download compatibility check. The cache layout puts
    // `snapshots/<commit>/` two levels below the repo dir we want to
    // probe — walk back up to that root.
    let repo_dir = snapshot_path
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf());
    if let Some(repo_dir) = repo_dir {
        let (ok, info) = check_pie_compatibility(&repo_dir);
        let colorize = std::io::stdout().is_terminal();
        let (green, yellow, dim, reset) = if colorize {
            ("\x1b[32m", "\x1b[33m", "\x1b[2m", "\x1b[0m")
        } else {
            ("", "", "", "")
        };
        println!();
        if ok {
            println!("{green}✓{reset} Pie compatible (arch: {info})");
            println!("Add to config.toml:");
            println!("  {dim}hf_repo = \"{repo_id}\"{reset}");
        } else {
            println!("{yellow}!{reset} Not Pie compatible ({info})");
        }
    }
    Ok(())
}

fn parse_repo_id(s: &str) -> Result<(String, String)> {
    let mut parts = s.splitn(2, '/');
    let owner = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        bail!("expected `owner/name`, got {s:?}");
    }
    Ok((owner.to_string(), name.to_string()))
}

/// Inline ANSI progress bar driven by `hf_hub`'s [`ProgressHandler`]
/// interface. Tracks the cumulative byte count emitted via
/// `DownloadEvent::AggregateProgress` (xet batches) and per-file
/// `DownloadEvent::Progress` (legacy LFS), and redraws at most every
/// ~100 ms to keep the terminal readable.
#[derive(Clone)]
struct ProgressBar {
    inner: std::sync::Arc<ProgressBarInner>,
}

struct ProgressBarInner {
    total_files: AtomicU64,
    total_bytes: AtomicU64,
    bytes_done: AtomicU64,
    /// Accumulator for legacy (non-xet) per-file progress, keyed by
    /// filename. xet batches report aggregate bytes directly via
    /// [`DownloadEvent::AggregateProgress`] so the legacy path only
    /// fires for old LFS-pointer files.
    per_file: Mutex<std::collections::HashMap<String, u64>>,
    started: Instant,
    last_draw: Mutex<Instant>,
    finished: AtomicBool,
    /// Skip drawing entirely when stderr isn't a TTY (e.g. piped to a
    /// file or running under CI). The download still completes; we
    /// just don't emit ANSI escapes.
    is_tty: bool,
}

impl ProgressBar {
    fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(ProgressBarInner {
                total_files: AtomicU64::new(0),
                total_bytes: AtomicU64::new(0),
                bytes_done: AtomicU64::new(0),
                per_file: Mutex::new(Default::default()),
                started: Instant::now(),
                last_draw: Mutex::new(Instant::now()),
                finished: AtomicBool::new(false),
                is_tty: std::io::stderr().is_terminal(),
            }),
        }
    }

    fn finish(&self) {
        self.inner.finished.store(true, Ordering::Relaxed);
        if self.inner.is_tty {
            // Replace the bar line with a clean blank so the post-
            // download "✓ Downloaded to …" lands on a fresh row.
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }
    }

    fn draw(&self) {
        if !self.inner.is_tty {
            return;
        }
        let now = Instant::now();
        {
            let mut last = self.inner.last_draw.lock().unwrap();
            if now.duration_since(*last).as_millis() < 100 {
                return;
            }
            *last = now;
        }
        let done = self.inner.bytes_done.load(Ordering::Relaxed);
        let total = self.inner.total_bytes.load(Ordering::Relaxed);
        let elapsed = now.duration_since(self.inner.started).as_secs_f64().max(0.001);
        let rate = done as f64 / elapsed;
        let pct = if total > 0 {
            (done as f64 / total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let bar_width = 30usize;
        let filled = (pct * bar_width as f64).round() as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
        let line = format!(
            "\r\x1b[K  {bar} {pct:>5.1}% {done} / {total} @ {rate}/s",
            pct = pct * 100.0,
            done = format_bytes(done),
            total = format_bytes(total),
            rate = format_bytes(rate as u64),
        );
        eprint!("{line}");
        let _ = std::io::stderr().flush();
    }
}

impl ProgressHandler for ProgressBar {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(ev) = event else {
            return;
        };
        match ev {
            DownloadEvent::Start { total_files, total_bytes } => {
                self.inner
                    .total_files
                    .store(*total_files as u64, Ordering::Relaxed);
                self.inner.total_bytes.store(*total_bytes, Ordering::Relaxed);
            }
            DownloadEvent::Progress { files } => {
                // Per-file deltas: keep a running max per filename and
                // sum into `bytes_done`. xet downloads use
                // `AggregateProgress` for the live byte counter, so
                // legacy LFS files are the main consumers of this arm.
                let mut map = self.inner.per_file.lock().unwrap();
                for fp in files {
                    let prev = map.get(&fp.filename).copied().unwrap_or(0);
                    if fp.bytes_completed > prev {
                        self.inner
                            .bytes_done
                            .fetch_add(fp.bytes_completed - prev, Ordering::Relaxed);
                        map.insert(fp.filename.clone(), fp.bytes_completed);
                    }
                }
                self.draw();
            }
            DownloadEvent::AggregateProgress { bytes_completed, total_bytes, .. } => {
                // xet batch: bytes_completed is monotonic per batch.
                // Treat it as authoritative — overwrite, don't accumulate.
                self.inner.bytes_done.store(*bytes_completed, Ordering::Relaxed);
                if *total_bytes > self.inner.total_bytes.load(Ordering::Relaxed) {
                    self.inner.total_bytes.store(*total_bytes, Ordering::Relaxed);
                }
                self.draw();
            }
            DownloadEvent::Complete => {
                self.draw();
            }
        }
    }
}

fn format_bytes(n: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    if n >= GIB {
        format!("{:.2} GiB", n as f64 / GIB as f64)
    } else if n >= MIB {
        format!("{:.1} MiB", n as f64 / MIB as f64)
    } else if n >= KIB {
        format!("{:.1} KiB", n as f64 / KIB as f64)
    } else {
        format!("{n} B")
    }
}

// -----------------------------------------------------------------------------
// remove
// -----------------------------------------------------------------------------

fn remove(repo_id: String, skip_confirm: bool) -> Result<()> {
    let hub = hub_dir();
    let model_dir = hub.join(repo_id_to_dirname(&repo_id));
    if !model_dir.exists() {
        bail!("model {repo_id:?} not found in cache ({})", model_dir.display());
    }

    // Use hf-hub's scanner so the size we report dedups blobs shared
    // between revisions of the same repo — matches `pie model remove`'s
    // Python behavior (`huggingface_hub.scan_cache_dir`).
    let size = scanned_repo_size(&repo_id).unwrap_or_else(|| dir_size(&model_dir).unwrap_or(0));
    let mb = size as f64 / (1024.0 * 1024.0);

    if !skip_confirm {
        if !std::io::stdin().is_terminal() {
            bail!(
                "remove requires confirmation; rerun with `pie model remove {repo_id} --yes`"
            );
        }
        eprint!("Remove {repo_id} ({mb:.1} MiB)? [y/N] ");
        let _ = std::io::stderr().flush();
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(|e| anyhow!("read stdin: {e}"))?;
        let yes = matches!(answer.trim(), "y" | "Y" | "yes" | "YES");
        if !yes {
            println!("(aborted)");
            return Ok(());
        }
    }

    println!("Removing {repo_id} ({mb:.1} MiB)…");
    // HF v1 stores blobs under each repo's own `blobs/` dir, not in a
    // global pool — so removing the repo dir reclaims every blob it
    // referenced. The Python CLI uses
    // `huggingface_hub.scan_cache_dir.delete_revisions`; that API is
    // moot here because the cross-repo blob sharing it accounts for
    // doesn't exist in the on-disk layout.
    std::fs::remove_dir_all(&model_dir)
        .map_err(|e| anyhow!("remove {model_dir:?}: {e}"))?;
    println!("✓ Removed");
    Ok(())
}

/// Scan the HF cache and return the deduped repo size. Returns `None`
/// if the scanner errors, the repo can't be found, or `tokio` fails to
/// boot — callers fall back to a raw `dir_size` walk.
fn scanned_repo_size(repo_id: &str) -> Option<u64> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;
    runtime.block_on(async move {
        let client = hf_hub::HFClient::new().ok()?;
        let info = client.scan_cache().send().await.ok()?;
        info.repos
            .into_iter()
            .find(|r| r.repo_id == repo_id)
            .map(|r| r.size_on_disk)
    })
}

fn dir_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += dir_size(&entry.path())?;
        } else if metadata.is_file() {
            total += metadata.len();
        }
        // Symlinks (HF cache uses them for snapshot/blob deduplication)
        // are intentionally skipped — counting through them would
        // double-count the blobs they point at.
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirname_round_trips() {
        assert_eq!(
            dirname_to_repo_id("models--Qwen--Qwen3-0.6B").as_deref(),
            Some("Qwen/Qwen3-0.6B"),
        );
        assert_eq!(
            dirname_to_repo_id("models--bert-base-uncased").as_deref(),
            Some("bert-base-uncased"),
        );
        assert_eq!(dirname_to_repo_id("not-a-model"), None);
        assert_eq!(dirname_to_repo_id("models--a--b--c"), None);

        assert_eq!(repo_id_to_dirname("Qwen/Qwen3-0.6B"), "models--Qwen--Qwen3-0.6B");
        assert_eq!(repo_id_to_dirname("bert-base-uncased"), "models--bert-base-uncased");
    }

    #[test]
    fn parses_repo_id() {
        assert_eq!(
            parse_repo_id("Qwen/Qwen3-0.6B").unwrap(),
            ("Qwen".to_string(), "Qwen3-0.6B".to_string()),
        );
        assert!(parse_repo_id("missing-slash").is_err());
        assert!(parse_repo_id("a/b/c").is_err());
    }

    #[test]
    fn compat_check_finds_arch() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = tmp.path().join("snapshots").join("abc123");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"), r#"{"model_type": "qwen3"}"#).unwrap();
        let (ok, info) = check_pie_compatibility(tmp.path());
        assert!(ok);
        assert_eq!(info, "qwen3");
    }

    #[test]
    fn compat_check_unsupported_arch() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = tmp.path().join("snapshots").join("abc");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(
            snap.join("config.json"),
            r#"{"model_type": "totally-fake-arch"}"#,
        )
        .unwrap();
        let (ok, info) = check_pie_compatibility(tmp.path());
        assert!(!ok);
        assert!(info.contains("totally-fake-arch"), "got: {info}");
    }

    #[test]
    fn compat_check_missing_config() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("snapshots").join("abc")).unwrap();
        let (ok, info) = check_pie_compatibility(tmp.path());
        assert!(!ok);
        assert_eq!(info, "no config");
    }

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MiB");
        assert!(format_bytes(2_500_000_000).starts_with("2."));
    }
}
