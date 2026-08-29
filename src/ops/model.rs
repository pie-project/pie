//! `pie model { list | info | import | remove }` — the models pie serves.
//!
//! `list` and `info` read the artifact store ([`crate::local::store`]) with the
//! HF snapshot cache beside it, because "what do I have" and "where did it come
//! from" are one question. [`import`] is big enough to own a file: it fetches a
//! checkpoint and normalizes it into a `.zt`.
//!
//! `build` STOOD BESIDE IT — "author the family contract, run the load
//! transforms offline, write the runtime tensors" — and R3 deleted it with the
//! contract it authored. Every transform it ran was
//! `model_legacy::contract::author`'s, for a load path that no longer exists:
//! the engine produces its weights from the checkpoint through the SKU's own
//! import table at load, so there is nothing to precompute and no artifact
//! shape to precompute it into.

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, bail};
use clap::Subcommand;

use crate::local::hf::runtime_snapshot_allow_patterns;
use crate::ui::{Align, Answer, Mark, Palette, Row, Table};

pub mod import;

#[derive(Subcommand, Debug)]
pub enum ModelCmd {
    /// List the artifacts pie can serve, and any raw snapshots beside them.
    List,

    /// Show what pie knows about one stored artifact.
    Info {
        /// The store name, as `pie model list` prints it.
        name: String,
    },
    /// Make a model servable: fetch it if it is remote, convert it to a
    /// `.zt` artifact, and put it in the store.
    Import(import::ImportArgs),
    /// Remove a stored artifact by name. Prompts for confirmation;
    /// `--yes` skips the prompt.
    Remove {
        /// The store name, as `pie model list` prints it.
        name: String,
        /// Skip the confirmation prompt.
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

pub fn run(cmd: ModelCmd) -> Result<Answer> {
    match cmd {
        ModelCmd::List => list(),
        ModelCmd::Info { name } => info(name),
        // `download` and `convert` were both "make this servable", so they are
        // one verb now; `import` fetches when the source is remote.
        ModelCmd::Import(args) => import::run(args),
        ModelCmd::Remove { name, yes } => remove(name, yes),
    }
}

/// HF cache root for model snapshots: `<HF_HOME or ~/.cache/huggingface>/hub/`.
fn hub_dir() -> std::path::PathBuf {
    crate::local::hf::resolve_cache_dir()
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

/// Does the snapshot beside this repo hold a checkpoint some catalog SKU can
/// be built from?
///
/// # What this used to ask, and why it does not any more
///
/// It read `config.json`'s `model_type` and looked it up in
/// `model_legacy::ingest`'s `MODEL_TYPES` table — a hand-kept list of the HF
/// strings the legacy contract had a naming pass for. R3 deleted the table
/// with the contract, and there is nothing to replace it WITH, because the
/// new catalog never asks a config what a model is. It asks the TENSORS: the
/// identify door compiles each SKU's load contract against the checkpoint's
/// own metadata, and that is the same question an engine settles at load.
///
/// So this asks that question instead, one step earlier and against the
/// checkpoint headers rather than a document about them.
///
/// # And it asks it through the door serving asks it through
///
/// It read the snapshot itself — `model::snapshot::Snapshot::at`, a
/// safetensors header parser lifted out of a baker harness — and handed
/// `model::identify` a `&dyn Fn(&str) -> Option<shape>` closure over it. Both
/// are gone (menlo M18/M19), and what replaced them is not a rename: the
/// runtime's [`identify`](runtime::engine::load::identify) does not match tensor
/// NAMES, it compiles each candidate's load contract against the checkpoint's
/// own metadata and answers with the SKU whose params the checkpoint actually
/// holds. A name match cannot tell a 3B Qwen from a 0.8B one — every SKU of a
/// family spells its tensors the same way — and that is measured, in the boot
/// smoke that came back with the wrong row.
///
/// Cost is unchanged: that door opens the container's index and runs the
/// contract's arithmetic, so listing a cache of twenty repos still reads
/// twenty headers and no payload.
///
/// Returns `(true, sku)` for a checkpoint that matches a row, and
/// `(false, why)` for one that does not, carrying the refusal's own account —
/// which now names what each candidate did wrong instead of counting how many
/// matched.
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
        return (false, "no snapshot".to_string());
    };
    // "A snapshot with no weights in it" and "a checkpoint no SKU claims" are
    // different operator actions, and the door below answers both with one
    // refusal. `Snapshot::at` used to draw the line by returning `None`; this
    // draws it by looking, which is the same read the door would do first
    // anyway.
    let weights = std::fs::read_dir(&snap).is_ok_and(|entries| {
        entries.filter_map(|entry| entry.ok()).any(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".safetensors") || name.ends_with(".zt")
        })
    });
    if !weights {
        return (false, "no safetensors".to_string());
    }
    // The plane picks an alignment and a tile budget, neither of which can
    // change whether a tensor's SHAPE is the one a contract declares — which
    // is the only question a listing asks — so the door's own note applies:
    // one platform is as good as any, and this listing is a host-side answer
    // about files, not about a device this machine has.
    match runtime::engine::load::identify(&snap, runtime::engine::load::Platform::Cuda) {
        Ok(sku) => (true, sku.to_string()),
        // One line, because this is a table cell. The full per-candidate
        // account is what `pie model import` prints when the load is
        // actually attempted.
        Err(_) => (false, "no SKU".to_string()),
    }
}

// -----------------------------------------------------------------------------
// list
// -----------------------------------------------------------------------------

/// The artifacts pie can serve, and the raw snapshots beside them.
#[derive(serde::Serialize)]
pub struct ModelList {
    store: std::path::PathBuf,
    artifacts: Vec<Artifact>,
    /// Where `import` reads from. Not what serving reads: the HF cache demotes
    /// to a staging area once an artifact exists.
    snapshots_dir: std::path::PathBuf,
    snapshots: Vec<Snapshot>,
    snapshot_bytes: u64,
}

#[derive(serde::Serialize)]
struct Artifact {
    name: String,
    root: std::path::PathBuf,
    shards: usize,
    bytes: u64,
    tensors: usize,
    written_by: Option<String>,
    source: Option<String>,
    /// Per-target builds derived from this archive. Reported so that the store
    /// size a listing shows is the store size on disk: a runtime artifact is
    /// as large as the archive it came from, and a model with three of them
    /// occupies four times what the archive line alone would suggest.
    runtimes: Vec<RuntimeBuild>,
}

#[derive(serde::Serialize)]
struct RuntimeBuild {
    key: String,
    bytes: u64,
    runtime_quant: Option<String>,
}

#[derive(serde::Serialize)]
struct Snapshot {
    repo_id: String,
    /// Whether any linked engine knows this family.
    servable: bool,
    /// The pie arch name when servable, the reason when not.
    detail: String,
    bytes: u64,
}

impl crate::ui::Report for ModelList {
    fn render(&self, palette: &Palette) {
        // What pie can serve. This is the store, and it comes first because it
        // is the answer to "what models do I have" — the HF cache below it is
        // where these came *from*.
        println!("Artifacts ({}):", self.store.display());
        if self.artifacts.is_empty() {
            println!(
                "  {}",
                palette.dim("(none — `pie model import <org>/<name>`)")
            );
        }
        // Four columns, not three. Folding the tensor count in with the
        // provenance put all three facts in the column `Table` cuts to fit, so
        // an 80-column terminal lost the tensor count *and* the source. Only
        // the last column is ever cut, so only the least load-bearing fact --
        // where it came from, which `pie model info` prints in full -- is at
        // risk.
        let mut table = Table::new([Align::Left, Align::Right, Align::Right, Align::Left], 1);
        for artifact in &self.artifacts {
            let shards = match artifact.shards {
                0 => String::new(),
                n => format!(" +{n}"),
            };
            let from = artifact
                .source
                .as_deref()
                .map(|s| format!("← {s},"))
                .unwrap_or_default();
            let by = artifact
                .written_by
                .as_deref()
                .map(|v| format!("pie {v}"))
                .unwrap_or_else(|| "provenance missing".to_string());
            table.push(Row::new(
                // `●` here and `○`/`×` below were this command's private glyph
                // vocabulary, three spellings of what `Mark` already names.
                Mark::Plain,
                [
                    artifact.name.clone(),
                    crate::ui::bytes(artifact.bytes),
                    format!("{} tensors{shards}", artifact.tensors),
                    format!("{from}{by}"),
                ],
            ));
            // Indented under the archive they came from, because that is the
            // relationship: a build is derived and deleting it costs a
            // rebuild, where deleting the archive costs a re-import.
            for runtime in &artifact.runtimes {
                let quant = runtime
                    .runtime_quant
                    .as_deref()
                    .map(|q| format!(", {q}"))
                    .unwrap_or_default();
                table.push(Row::new(
                    Mark::Plain,
                    [
                        format!("  runtime/{}", runtime.key),
                        crate::ui::bytes(runtime.bytes),
                        String::new(),
                        format!("built{quant}"),
                    ],
                ));
            }
        }
        table.print(palette);

        if self.snapshots.is_empty() {
            return;
        }
        // Raw snapshots, marked as what they now are: staging for conversion,
        // and disk that `pie cache clear snapshots` reclaims.
        println!(
            "\nRaw snapshots ({}, {}):",
            self.snapshots_dir.display(),
            crate::ui::bytes(self.snapshot_bytes)
        );
        let mut table = Table::new([Align::Left, Align::Right, Align::Left], 1);
        for snapshot in &self.snapshots {
            table.push(Row::new(
                // "pie cannot serve this family" is the same answer as every
                // other absence, and gets the same glyph.
                if snapshot.servable {
                    Mark::Plain
                } else {
                    Mark::Absent
                },
                [
                    snapshot.repo_id.clone(),
                    crate::ui::bytes(snapshot.bytes),
                    snapshot.detail.clone(),
                ],
            ));
        }
        table.print(palette);
    }
}

fn list() -> Result<Answer> {
    let artifacts = crate::local::store::entries()?;
    let hub = hub_dir();
    let mut snapshots: Vec<Snapshot> = match std::fs::read_dir(&hub) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let repo_id = dirname_to_repo_id(&name)?;
                let (servable, detail) = check_pie_compatibility(&e.path());
                Some(Snapshot {
                    repo_id,
                    servable,
                    detail,
                    bytes: crate::local::store::staging_bytes(&e.path()),
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    snapshots.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));

    Ok(Answer::report(ModelList {
        store: crate::local::store::dir(),
        artifacts: artifacts
            .into_iter()
            .map(|e| Artifact {
                shards: e.shards(),
                runtimes: e
                    .runtimes
                    .iter()
                    .map(|r| RuntimeBuild {
                        key: r.key.clone(),
                        bytes: r.bytes,
                        runtime_quant: r.runtime_quant.clone(),
                    })
                    .collect(),
                name: e.name,
                root: e.root,
                bytes: e.bytes,
                tensors: e.tensors,
                written_by: e.written_by,
                source: e.source,
            })
            .collect(),
        snapshots_dir: hub,
        snapshot_bytes: snapshots.iter().map(|s| s.bytes).sum(),
        snapshots,
    }))
}

/// `pie model info <name>` — one artifact, in detail.
///
/// About a STORE ENTRY, not a HuggingFace repo. The earlier version of this
/// opened `models--org--name/snapshots/*/config.json` and reported an
/// architecture, which stopped being the right question when the artifact
/// became the thing pie serves: a repo is one way an artifact got here, and
/// `source` below is where that is recorded.
fn info(name: String) -> Result<Answer> {
    let Some(entry) = crate::local::store::find(&name)? else {
        bail!("no artifact {name:?} in the store; `pie model list` shows what is there");
    };
    Ok(Answer::report(ModelInfo {
        shards: entry.shards(),
        runtimes: entry
            .runtimes
            .iter()
            .map(|r| RuntimeBuild {
                key: r.key.clone(),
                bytes: r.bytes,
                runtime_quant: r.runtime_quant.clone(),
            })
            .collect(),
        name: entry.name,
        root: entry.root,
        files: entry.files,
        bytes: entry.bytes,
        tensors: entry.tensors,
        written_by: entry.written_by,
        source: entry.source,
    }))
}

/// One store entry, in detail.
#[derive(serde::Serialize)]
pub struct ModelInfo {
    name: String,
    root: std::path::PathBuf,
    files: Vec<std::path::PathBuf>,
    shards: usize,
    bytes: u64,
    tensors: usize,
    written_by: Option<String>,
    source: Option<String>,
    runtimes: Vec<RuntimeBuild>,
}

impl crate::ui::Report for ModelInfo {
    fn render(&self, palette: &Palette) {
        println!("{}", palette.bold(&self.name));
        let mut table = Table::new([Align::Left, Align::Left], 1);
        let mut row = |k: &str, v: String| table.push(Row::new(Mark::Plain, [k.to_string(), v]));
        row("size", crate::ui::bytes(self.bytes));
        row("tensors", self.tensors.to_string());
        row(
            "files",
            match self.shards {
                0 => "one".to_string(),
                n => format!("root + {n} shards"),
            },
        );
        if let Some(source) = &self.source {
            row("source", source.clone());
        }
        if let Some(written_by) = &self.written_by {
            row("written by", format!("pie {written_by}"));
        }
        row("path", crate::ui::short_path(&self.root));
        // What has been built from it, and for what. `pie model build` is the
        // only thing that writes these, and `pie model rm` takes them with the
        // archive because they are derived from it.
        for runtime in &self.runtimes {
            let quant = runtime
                .runtime_quant
                .as_deref()
                .map(|q| format!(" ({q})"))
                .unwrap_or_default();
            row(
                "runtime",
                format!(
                    "{} — {}{quant}",
                    runtime.key,
                    crate::ui::bytes(runtime.bytes)
                ),
            );
        }
        table.print(palette);
        println!(
            "\n{}",
            palette.dim(format!("[model]\nmodel = \"{}\"", self.name))
        );
    }
}

// -----------------------------------------------------------------------------
// download
// -----------------------------------------------------------------------------

/// Fetch a HuggingFace snapshot into the local cache.
///
/// The runtime-artifact filter is not a flag any more: an import converts what
/// it fetches, and the formats the old `--all` added are ones the conversion
/// drops anyway. They were only useful with the `--raw` that has gone with it
/// -- and "get files from HuggingFace without converting them" is
/// `huggingface-cli`'s job, not a mode of a pie command.
pub(crate) fn fetch_snapshot(repo_id: &str) -> Result<std::path::PathBuf> {
    // Checked here rather than at the hub: `owner/name` is the shape every
    // downstream path assumes, and a 404 is a worse way to learn it.
    parse_repo_id(repo_id)?;
    println!("Fetching {repo_id}");

    // Multi-thread, because the downloader fans out across files: on a
    // current-thread runtime the eight tasks would take turns on one core and
    // the transfer would run at one connection's pace.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let label = repo_id.to_string();
    // Built out here so the result line can report what the transfer cost --
    // the bar erases itself when it finishes.
    let progress = ProgressBar::new();
    let sink = progress.sink();
    let snapshot_path = runtime.block_on(async move {
        crate::local::hf::snapshot_download(&label, &runtime_snapshot_allow_patterns(), sink).await
    });
    progress.finish();
    let snapshot_path = snapshot_path?;
    println!(
        "{} fetched to {}{}",
        crate::ui::Mark::Did.render(&crate::ui::Palette::for_stream(crate::ui::Stream::Stdout)),
        crate::ui::short_path(&snapshot_path),
        progress.summary()
    );
    Ok(snapshot_path)
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

/// Inline ANSI progress bar for a snapshot fetch, driven by the downloader's
/// [`crate::local::hf::Progress`] callbacks. Redraws at most every ~100 ms so
/// a hundred-shard transfer does not spend its time writing escape codes.
#[derive(Clone)]
struct ProgressBar {
    inner: std::sync::Arc<ProgressBarInner>,
}

struct ProgressBarInner {
    total_files: AtomicU64,
    total_bytes: AtomicU64,
    bytes_done: AtomicU64,
    started: Instant,
    last_draw: Mutex<Instant>,
    finished: AtomicBool,
    /// Skip drawing entirely when stderr isn't a TTY (e.g. piped to a
    /// file or running under CI). The download still completes; we
    /// just don't emit ANSI escapes.
    is_tty: bool,
}

impl ProgressBarInner {
    fn draw(&self) {
        if !self.is_tty {
            return;
        }
        let now = Instant::now();
        {
            let mut last = self.last_draw.lock().unwrap();
            if now.duration_since(*last).as_millis() < 100 {
                return;
            }
            *last = now;
        }
        let done = self.bytes_done.load(Ordering::Relaxed);
        let total = self.total_bytes.load(Ordering::Relaxed);
        let elapsed = now.duration_since(self.started).as_secs_f64().max(0.001);
        let rate = done as f64 / elapsed;
        let pct = if total > 0 {
            (done as f64 / total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let bar_width = 30usize;
        let filled = (pct * bar_width as f64).round() as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
        // An ETA only once there is a rate worth extrapolating from. Guessing
        // from the first hundred milliseconds swings by minutes, which teaches
        // a reader to ignore the field.
        let eta = if total > done && rate > 1.0 && elapsed > 2.0 {
            let remaining = std::time::Duration::from_secs_f64((total - done) as f64 / rate);
            format!(" {} left", crate::ui::duration(remaining))
        } else {
            String::new()
        };
        let body = format!(
            "  {bar} {pct:>5.1}% {done} / {total} @ {rate}{eta}",
            pct = pct * 100.0,
            done = crate::ui::bytes(done),
            total = crate::ui::bytes(total),
            rate = crate::ui::rate(rate),
        );
        // Cut to the terminal: a line that wraps puts the cursor on a second
        // screen row, and the `\r` that starts the next redraw returns to the
        // start of THAT row, leaving the first behind as debris.
        eprint!("\r\x1b[K{}", crate::ui::clip(&body, crate::ui::width()));
        let _ = std::io::stderr().flush();
    }
}

impl crate::local::hf::Progress for ProgressBarInner {
    fn start(&self, files: u64, bytes: u64) {
        self.total_files.store(files, Ordering::Relaxed);
        self.total_bytes.store(bytes, Ordering::Relaxed);
        self.draw();
    }

    fn advance(&self, bytes: u64) {
        self.bytes_done.fetch_add(bytes, Ordering::Relaxed);
        self.draw();
    }
}

impl ProgressBar {
    fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(ProgressBarInner {
                total_files: AtomicU64::new(0),
                total_bytes: AtomicU64::new(0),
                bytes_done: AtomicU64::new(0),
                started: Instant::now(),
                last_draw: Mutex::new(Instant::now()),
                finished: AtomicBool::new(false),
                is_tty: std::io::stderr().is_terminal(),
            }),
        }
    }

    /// The handle the downloader reports into. Shares this bar's state, so the
    /// result line can still read the byte count after the fetch returns.
    fn sink(&self) -> std::sync::Arc<dyn crate::local::hf::Progress> {
        self.inner.clone()
    }

    fn finish(&self) {
        self.inner.finished.store(true, Ordering::Relaxed);
        if self.inner.is_tty {
            // Replace the bar line with a clean blank so the result line lands
            // on a fresh row.
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }
    }

    /// What the transfer actually cost, for the result line.
    ///
    /// The bar erases itself when it finishes, so without this the only record
    /// of a twenty-minute fetch was that it had ended.
    fn summary(&self) -> String {
        let moved = self.inner.bytes_done.load(Ordering::Relaxed);
        if moved == 0 {
            return String::new();
        }
        format!(
            " ({} in {})",
            crate::ui::bytes(moved),
            crate::ui::duration(self.inner.started.elapsed())
        )
    }
}

// -----------------------------------------------------------------------------
// remove
// -----------------------------------------------------------------------------

/// Delete one artifact from the store.
///
/// Only the artifact. Reclaiming the HuggingFace snapshot it was converted
/// from used to be a `--staging` flag here; it is `pie cache clear snapshots`,
/// which knows about every snapshot rather than the one beside this artifact,
/// asks before deleting, and reports what it got back. A command that removes
/// a model has no business deciding what else its origin is worth keeping.
fn remove(name: String, skip_confirm: bool) -> Result<Answer> {
    let Some(entry) = crate::local::store::find(&name)? else {
        bail!(
            "no artifact named {name:?} in {}",
            crate::local::store::dir().display()
        );
    };

    // The archive AND the builds derived from it, because that is what the
    // removal takes. Reporting the archive alone understated a three-artifact
    // model as one file, in a confirmation prompt whose whole job is to say
    // what is about to be lost.
    let files = entry.files.len() + entry.runtimes.iter().map(|r| r.files.len()).sum::<usize>();
    let derived = match entry.runtimes.len() {
        0 => String::new(),
        n => format!(", {n} build(s)"),
    };
    let what = format!(
        "artifact {name} ({}, {files} file(s){derived})",
        crate::ui::bytes(entry.total_bytes()),
    );

    if !skip_confirm
        && !crate::ui::confirm(
            &format!("Remove {what}?"),
            &format!("pie model remove {name} --yes"),
        )?
    {
        return Ok(Answer::noop("aborted; nothing was removed"));
    }

    crate::local::store::remove(&entry)?;
    Ok(Answer::did(format!("removed {what}")))
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

    /// A directory with no snapshot, and one with a snapshot but no
    /// safetensors, are different answers.
    ///
    /// The three `compat_check_*` tests that STOOD HERE wrote a
    /// `config.json` naming a `model_type` and asserted the legacy table's
    /// answer. There is no table: what the check asks now is whether a
    /// checkpoint's TENSORS match an import row, so a fixture for the
    /// positive case is a real safetensors header — which is
    /// `model`'s own test surface, not this command's.
    #[test]
    fn compat_check_reports_what_it_could_not_read() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            check_pie_compatibility(tmp.path()),
            (false, "no snapshot".to_string()),
        );
        std::fs::create_dir_all(tmp.path().join("snapshots").join("abc")).unwrap();
        assert_eq!(
            check_pie_compatibility(tmp.path()),
            (false, "no safetensors".to_string()),
        );
    }
}
