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

use crate::ops::hf::runtime_snapshot_allow_patterns;

#[derive(Subcommand, Debug)]
pub enum ModelCmd {
    /// List the artifacts pie can serve, and any raw snapshots beside them.
    List,
    /// Fetch a model by HuggingFace repo ID and convert it to a `.zt` artifact.
    Download {
        repo_id: String,
        /// Download the complete HF snapshot, including alternate weight
        /// formats Pie does not use. By default, Pie downloads only runtime
        /// artifacts: config/tokenizer files and model*.safetensors.
        #[arg(long)]
        all: bool,
        /// Fetch only. Leaves the HF snapshot as-is and writes no artifact —
        /// for debugging, or for feeding the files to something other than pie.
        #[arg(long)]
        raw: bool,
        /// Delete the HF snapshot once the artifact is written and verified.
        /// Off by default: keeping it means a re-convert costs no download.
        #[arg(long)]
        clean: bool,
        /// Convert even if an up-to-date artifact is already in the store.
        #[arg(long)]
        force: bool,
        /// Split the artifact into shards of about this size (e.g. `16GiB`).
        #[arg(long, value_name = "SIZE", value_parser = crate::ops::convert::parse_size)]
        max_shard_size: Option<u64>,
    },
    /// Remove a stored artifact by name. Prompts for confirmation;
    /// `--yes` skips the prompt.
    Remove {
        /// The store name, as `pie model list` prints it.
        name: String,
        #[arg(long, short = 'y')]
        yes: bool,
        /// Also delete the HF snapshot the artifact was converted from.
        #[arg(long)]
        staging: bool,
    },
    /// Rewrite a checkpoint as pie's canonical `.zt` artifact.
    Convert(crate::ops::convert::ConvertArgs),
}

pub fn run(cmd: ModelCmd) -> Result<()> {
    match cmd {
        ModelCmd::List => list(),
        ModelCmd::Download {
            repo_id,
            all,
            raw,
            clean,
            force,
            max_shard_size,
        } => download(repo_id, all, raw, clean, force, max_shard_size),
        ModelCmd::Remove { name, yes, staging } => remove(name, yes, staging),
        ModelCmd::Convert(args) => crate::ops::convert::run(args),
    }
}

/// HF cache root for model snapshots: `<HF_HOME or ~/.cache/huggingface>/hub/`.
fn hub_dir() -> std::path::PathBuf {
    hf_hub::resolve_cache_dir()
}

/// Convert `models--org--name` ↔ `org/name`.
fn dirname_to_repo_id(dir: &str) -> Option<String> {
    let stripped = dir.strip_prefix("models--")?;
    let parts: Vec<&str> = stripped.split("--").collect();
    match parts.len() {
        1 => Some(parts[0].to_string()),
        2 => Some(format!("{}/{}", parts[0], parts[1])),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// Pie-compatibility check
// -----------------------------------------------------------------------------

/// HuggingFace `model_type` → PIE arch name. Kept in sync with
/// the model_type strings the C++ drivers (`driver/cuda/src/loader/`,
/// `driver/metal/src/`) recognise. Architectures supported by *any*
/// of the standalone-linked drivers belong here.
const HF_TO_PIE_ARCH: &[(&str, &str)] = &[
    ("llama", "llama3"),
    ("qwen2", "qwen2"),
    ("qwen3", "qwen3"),
    ("qwen3_5", "qwen3_5"),
    ("qwen3_moe", "qwen3_moe"),
    ("qwen3_5_moe", "qwen3_5_moe"),
    ("qwen3_5_moe_text", "qwen3_5_moe"),
    ("qwen3_vl", "qwen3_vl"),
    ("qwen3_vl_text", "qwen3_vl"),
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
    ("nemotron_h", "nemotron_h"),
    ("kimi_k3", "kimi_k3"),
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
            .find(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
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
    let Ok(json): serde_json::Result<serde_json::Value> = serde_json::from_str(&text) else {
        return (false, "no config".to_string());
    };
    let model_type = json
        .get("text_config")
        .or_else(|| json.get("llm_config"))
        .and_then(|v| v.get("model_type"))
        .or_else(|| json.get("model_type"))
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
    let colorize = std::io::stdout().is_terminal();
    let (green, dim, reset) = if colorize {
        ("\x1b[32m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "")
    };

    // What pie can serve. This is the store, and it comes first because it is
    // the answer to "what models do I have" — the HF cache below it is where
    // these came *from*.
    let artifacts = crate::ops::store::entries()?;
    println!("Artifacts ({}):", crate::ops::store::dir().display());
    if artifacts.is_empty() {
        println!("  {dim}(none — `pie model download <org>/<name>`){reset}");
    }
    for entry in &artifacts {
        let shards = match entry.shards() {
            0 => String::new(),
            n => format!(", {n} shards"),
        };
        let from = entry
            .source
            .as_deref()
            .map(|s| format!(" ← {s}"))
            .unwrap_or_default();
        let by = entry
            .written_by
            .as_deref()
            .map(|v| format!(" pie {v}"))
            .unwrap_or_else(|| " provenance missing".to_string());
        println!(
            "  {green}●{reset} {} {dim}({}, {} tensors{shards}){reset}{dim}{from},{by}{reset}",
            entry.name,
            crate::ops::store::format_bytes(entry.bytes),
            entry.tensors,
        );
    }

    // Raw snapshots, marked as what they now are: staging for conversion,
    // and disk that `--clean` or `pie model remove --staging` reclaims.
    let hub = hub_dir();
    let mut staged: Vec<(String, bool, String, u64)> = match std::fs::read_dir(&hub) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let repo_id = dirname_to_repo_id(&name)?;
                let (ok, info) = check_pie_compatibility(&e.path());
                let bytes = crate::ops::store::staging_bytes(&e.path());
                Some((repo_id, ok, info, bytes))
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    staged.sort_by(|a, b| a.0.cmp(&b.0));

    if !staged.is_empty() {
        let total: u64 = staged.iter().map(|(_, _, _, b)| b).sum();
        println!(
            "\nRaw snapshots ({}, {}):",
            hub.display(),
            crate::ops::store::format_bytes(total)
        );
        for (repo_id, ok, info, bytes) in &staged {
            let mark = if *ok { "○" } else { "×" };
            println!(
                "  {dim}{mark} {repo_id} ({}, {info}){reset}",
                crate::ops::store::format_bytes(*bytes)
            );
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// download
// -----------------------------------------------------------------------------

fn download(
    repo_id: String,
    all: bool,
    raw: bool,
    clean: bool,
    force: bool,
    max_shard_size: Option<u64>,
) -> Result<()> {
    let (owner, name) = parse_repo_id(&repo_id)?;

    if all {
        println!("Downloading full snapshot: {repo_id}");
    } else {
        println!("Downloading runtime artifacts: {repo_id}");
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let label = repo_id.clone();
    let snapshot_path = runtime.block_on(async move {
        let client = hf_hub::HFClient::new().map_err(|e| anyhow!("init HF client: {e}"))?;
        let repo = client.model(owner, name);
        let progress = ProgressBar::new();
        let allow_patterns = if all {
            None
        } else {
            Some(runtime_snapshot_allow_patterns())
        };
        let result = repo
            .snapshot_download()
            .maybe_allow_patterns(allow_patterns)
            .progress(progress.clone())
            .send()
            .await
            .map_err(|e| anyhow!("download {label}: {e}"));
        progress.finish();
        result
    })?;
    println!("✓ Downloaded to {}", snapshot_path.display());

    if raw {
        println!("\n(--raw: no artifact written; `pie model convert {repo_id}` when you want one)");
        return Ok(());
    }

    // Download is fetch *and* convert: what the runtime serves is the
    // artifact, so a download that stopped at the snapshot would leave the
    // user one undiscoverable step short of a usable model.
    println!();
    crate::ops::convert::run(crate::ops::convert::ConvertArgs {
        source: repo_id.clone(),
        out: None,
        dry_run: false,
        force,
        delete_source: false,
        max_shard_size,
    })?;

    let staging = crate::ops::store::staging_dir(&repo_id);
    if clean {
        if let Some(dir) = &staging {
            // Only after `convert` returned Ok, which means the artifact is
            // written; the snapshot is reconstructible by re-downloading, so
            // this is the reversible half of the two.
            let bytes = crate::ops::store::staging_bytes(dir);
            std::fs::remove_dir_all(dir)
                .map_err(|err| anyhow!("cannot delete {}: {err}", dir.display()))?;
            println!(
                "✓ Removed the HF snapshot ({} reclaimed)",
                crate::ops::store::format_bytes(bytes)
            );
        }
        return Ok(());
    }

    // The snapshot stays by default, so say what it costs. Keeping it means a
    // re-convert needs no network; not saying so means the user discovers
    // twice the disk usage on their own.
    if let Some(dir) = &staging {
        let bytes = crate::ops::store::staging_bytes(dir);
        if bytes > 0 {
            let colorize = std::io::stdout().is_terminal();
            let (dim, reset) = if colorize {
                ("\x1b[2m", "\x1b[0m")
            } else {
                ("", "")
            };
            println!(
                "{dim}The HF snapshot is kept ({}), so a re-convert needs no download.\n\
                 Add --clean, or `pie model remove {repo_id} --staging`, to reclaim it.{reset}",
                crate::ops::store::format_bytes(bytes)
            );
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
        let elapsed = now
            .duration_since(self.inner.started)
            .as_secs_f64()
            .max(0.001);
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
            DownloadEvent::Start {
                total_files,
                total_bytes,
            } => {
                self.inner
                    .total_files
                    .store(*total_files as u64, Ordering::Relaxed);
                self.inner
                    .total_bytes
                    .store(*total_bytes, Ordering::Relaxed);
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
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                // xet batch: bytes_completed is monotonic per batch.
                // Treat it as authoritative — overwrite, don't accumulate.
                self.inner
                    .bytes_done
                    .store(*bytes_completed, Ordering::Relaxed);
                if *total_bytes > self.inner.total_bytes.load(Ordering::Relaxed) {
                    self.inner
                        .total_bytes
                        .store(*total_bytes, Ordering::Relaxed);
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

fn remove(name: String, skip_confirm: bool, staging: bool) -> Result<()> {
    let entry = crate::ops::store::find(&name)?;
    // A store name and a repo ID are different spellings of the same model
    // (`Qwen/Qwen3-0.6B` ↔ `Qwen--Qwen3-0.6B`), and users will type either.
    let repo_id = name.replace("--", "/");
    let staging_dir =
        crate::ops::store::staging_dir(&repo_id).or_else(|| crate::ops::store::staging_dir(&name));

    if entry.is_none() && !(staging && staging_dir.is_some()) {
        bail!(
            "no artifact named {name:?} in {}",
            crate::ops::store::dir().display()
        );
    }

    let mut what = Vec::new();
    let mut bytes = 0u64;
    if let Some(entry) = &entry {
        let files = entry.files.len();
        what.push(format!(
            "artifact {name} ({}, {files} file(s))",
            crate::ops::store::format_bytes(entry.bytes)
        ));
        bytes += entry.bytes;
    }
    if staging {
        if let Some(dir) = &staging_dir {
            let staged = crate::ops::store::staging_bytes(dir);
            what.push(format!(
                "its HF snapshot ({})",
                crate::ops::store::format_bytes(staged)
            ));
            bytes += staged;
        }
    }

    if !skip_confirm {
        if !std::io::stdin().is_terminal() {
            bail!("remove requires confirmation; rerun with `pie model remove {name} --yes`");
        }
        eprint!(
            "Remove {} ({} total)? [y/N] ",
            what.join(" and "),
            crate::ops::store::format_bytes(bytes)
        );
        let _ = std::io::stderr().flush();
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(|e| anyhow!("read stdin: {e}"))?;
        if !matches!(answer.trim(), "y" | "Y" | "yes" | "YES") {
            println!("(aborted)");
            return Ok(());
        }
    }

    if let Some(entry) = &entry {
        crate::ops::store::remove(entry)?;
    }
    if staging {
        if let Some(dir) = &staging_dir {
            std::fs::remove_dir_all(dir)
                .map_err(|e| anyhow!("cannot delete {}: {e}", dir.display()))?;
        }
    }
    println!(
        "✓ Removed {} ({} reclaimed)",
        what.join(" and "),
        crate::ops::store::format_bytes(bytes)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cache_dirname_reads_back_as_a_repo_id() {
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
