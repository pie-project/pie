use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, bail};
use clap::Subcommand;

use crate::local::hf::runtime_snapshot_allow_patterns;
use crate::ui::{Align, Answer, Mark, Palette, Row, Table};

pub mod facts;
pub mod import;

#[derive(Subcommand, Debug)]
pub enum ModelCmd {
    List,

    Info {
        name: String,
    },
    Import(import::ImportArgs),
    Remove {
        name: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

pub fn run(cmd: ModelCmd, global: &bootstrap::GlobalArgs) -> Result<Answer> {
    match cmd {
        ModelCmd::List => list(),
        ModelCmd::Info { name } => info(name),
        ModelCmd::Import(args) => import::run(args, global),
        ModelCmd::Remove { name, yes } => remove(name, yes),
    }
}

fn hub_dir() -> std::path::PathBuf {
    crate::local::hf::resolve_cache_dir()
}

fn dirname_to_repo_id(dir: &str) -> Option<String> {
    let stripped = dir.strip_prefix("models--")?;
    let parts: Vec<&str> = stripped.split("--").collect();
    match parts.len() {
        1 => Some(parts[0].to_string()),
        2 => Some(format!("{}/{}", parts[0], parts[1])),
        _ => None,
    }
}

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
    let pipeline = checkpoint::file::diffusers::is_pipeline(&snap);
    let weights = pipeline
        || std::fs::read_dir(&snap).is_ok_and(|entries| {
            entries.filter_map(|entry| entry.ok()).any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.ends_with(".safetensors") || name.ends_with(".zt")
            })
        });
    if !weights {
        return (false, "no safetensors".to_string());
    }
    let Some(platform) = runtime::engine::load::this_box() else {
        return (false, "no engine".to_string());
    };
    match runtime::engine::load::identify(&snap, platform) {
        Ok(sku) => (true, sku.to_string()),
        Err(_) if pipeline => (false, "(no row)".to_string()),
        Err(_) => (false, "no SKU".to_string()),
    }
}

#[derive(serde::Serialize)]
pub struct ModelList {
    store: std::path::PathBuf,
    artifacts: Vec<Artifact>,
    snapshots_dir: std::path::PathBuf,
    snapshots: Vec<Snapshot>,
    snapshot_bytes: u64,
    dead: Option<DeadWeight>,
}

#[derive(serde::Serialize)]
struct DeadWeight {
    dir: std::path::PathBuf,
    files: usize,
    bytes: u64,
}

#[derive(serde::Serialize)]
struct Artifact {
    name: String,
    address: String,
    sku: Option<String>,
    backend: Option<String>,
    root: std::path::PathBuf,
    shards: usize,
    bytes: u64,
    tensors: usize,
    written_by: Option<String>,
    source: Option<String>,
    runtimes: Vec<RuntimeBuild>,
    generative: Option<facts::GenerativeFacts>,
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
    servable: bool,
    detail: String,
    bytes: u64,
}

impl crate::ui::Report for ModelList {
    fn render(&self, palette: &Palette) {
        println!("Artifacts ({}):", self.store.display());
        if self.artifacts.is_empty() {
            println!(
                "  {}",
                palette.dim("(none — `pie model import <org>/<name>`)")
            );
        }
        let mut table = Table::new(
            [
                Align::Left,
                Align::Right,
                Align::Right,
                Align::Left,
                Align::Left,
            ],
            1,
        );
        for artifact in &self.artifacts {
            let shards = match artifact.shards {
                0 => String::new(),
                n => format!(" +{n}"),
            };
            let landing = match (&artifact.sku, &artifact.backend) {
                (Some(sku), Some(backend)) => format!("{sku} · {backend}"),
                (Some(sku), None) => sku.clone(),
                _ => "unstamped".to_string(),
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
                Mark::Plain,
                [
                    artifact.address.clone(),
                    crate::ui::bytes(artifact.bytes),
                    format!("{} tensors{shards}", artifact.tensors),
                    landing,
                    format!("{from}{by}"),
                ],
            ));
            if let Some(generative) = &artifact.generative {
                table.push(Row::new(
                    Mark::Plain,
                    [
                        "  generative".to_string(),
                        String::new(),
                        if generative.text_to_image() {
                            "text-to-image".to_string()
                        } else {
                            String::new()
                        },
                        generative.summary(),
                        String::new(),
                    ],
                ));
            }
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
        println!(
            "\nRaw snapshots ({}, {}):",
            self.snapshots_dir.display(),
            crate::ui::bytes(self.snapshot_bytes)
        );
        let mut table = Table::new([Align::Left, Align::Right, Align::Left], 1);
        for snapshot in &self.snapshots {
            table.push(Row::new(
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
        if self.snapshots.iter().any(|snapshot| snapshot.servable) {
            println!(
                "  {}",
                palette.dim(
                    "The row shown is the one an import picks by itself; `pie model import \
                     <repo> --sku <NAME>` imports as another (`--sku '?'` lists every row this \
                     build ships)."
                )
            );
        }

        if let Some(dead) = &self.dead {
            println!(
                "\nDead weight ({}, {}):",
                dead.dir.display(),
                crate::ui::bytes(dead.bytes)
            );
            println!(
                "  {} file(s) from before the artifact became the serving file. \
                 A boot reads its planes out of the model's own `.zt` now and \
                 opens none of these — they are safe to delete, and nothing here \
                 deletes them.",
                dead.files,
            );
        }
    }
}

fn dead_weight() -> Option<DeadWeight> {
    let dir = bootstrap::paths::pie_home().join("cache").join("weights");
    let mut files = 0usize;
    let mut bytes = 0u64;
    for entry in std::fs::read_dir(&dir).ok()?.filter_map(|it| it.ok()) {
        let path = entry.path();
        let dead = path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("tiers") || extension.eq_ignore_ascii_case("weights")
        });
        if dead && path.is_file() {
            files += 1;
            bytes += entry.metadata().map_or(0, |meta| meta.len());
        }
    }
    (files > 0).then_some(DeadWeight { dir, files, bytes })
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
                address: e.address().to_string(),
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
                generative: facts::of(e.sku.as_deref()),
                name: e.name,
                sku: e.sku,
                backend: e.backend,
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
        dead: dead_weight(),
    }))
}

fn one(name: &str) -> Result<crate::local::store::Entry> {
    match crate::local::store::find(name)? {
        crate::local::store::Resolved::One(entry) => Ok(*entry),
        crate::local::store::Resolved::Missing => bail!(
            "no artifact {name:?} in {}; `pie model list` shows what is there",
            crate::local::store::dir().display()
        ),
        crate::local::store::Resolved::Ambiguous(candidates) => bail!(
            "{name:?} names {} artifacts of one model, not one: {}. Name the \
             one you mean — that is what the rest of the filename is for, and \
             `pie model list` prints each in full.",
            candidates.len(),
            candidates
                .iter()
                .map(|candidate| format!("`{candidate}`"))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    }
}

fn info(name: String) -> Result<Answer> {
    let entry = one(&name)?;
    Ok(Answer::report(ModelInfo {
        sku: entry.sku.clone(),
        backend: entry.backend.clone(),
        address: entry.address().to_string(),
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
        generative: facts::of(entry.sku.as_deref()),
        name: entry.name,
        root: entry.root,
        files: entry.files,
        bytes: entry.bytes,
        tensors: entry.tensors,
        written_by: entry.written_by,
        source: entry.source,
    }))
}

#[derive(serde::Serialize)]
pub struct ModelInfo {
    name: String,
    address: String,
    sku: Option<String>,
    backend: Option<String>,
    root: std::path::PathBuf,
    files: Vec<std::path::PathBuf>,
    shards: usize,
    bytes: u64,
    tensors: usize,
    written_by: Option<String>,
    source: Option<String>,
    runtimes: Vec<RuntimeBuild>,
    generative: Option<facts::GenerativeFacts>,
}

impl crate::ui::Report for ModelInfo {
    fn render(&self, palette: &Palette) {
        println!("{}", palette.bold(&self.address));
        let mut table = Table::new([Align::Left, Align::Left], 1);
        let mut row = |k: &str, v: String| table.push(Row::new(Mark::Plain, [k.to_string(), v]));
        if self.address != self.name {
            row("model", self.name.clone());
        }
        if let Some(sku) = &self.sku {
            row("sku", sku.clone());
        }
        if let Some(backend) = &self.backend {
            row("backend", backend.clone());
        }
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

        if let Some(generative) = &self.generative {
            println!("\n{}", palette.bold("Generative"));
            let mut table = Table::new([Align::Left, Align::Left], 1);
            let mut row =
                |k: &str, v: String| table.push(Row::new(Mark::Plain, [k.to_string(), v]));
            if let Some(latent) = &generative.latent {
                let (w, h) = latent.pixels_per_row();
                row(
                    "latent",
                    format!("{} (one row is {w}x{h} pixels)", latent.line()),
                );
            }
            if let Some(schedule) = &generative.schedule {
                row("schedule", schedule.line());
                if !schedule.pinned_sigmas.is_empty() {
                    row(
                        "sigmas",
                        schedule
                            .pinned_sigmas
                            .iter()
                            .map(|s| format!("{s:.4}"))
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                }
                row("train steps", schedule.train_steps.to_string());
            }
            row("max latent rows", generative.max_latent_rows.to_string());
            table.print(palette);

            println!("\n{}", palette.bold("Readings"));
            let name_width = generative
                .readings
                .iter()
                .map(|r| r.name.chars().count())
                .max()
                .unwrap_or(0);
            let binds_of = |reading: &facts::Reading| {
                let mut binds = Vec::new();
                if reading.tokens {
                    binds.push("tokens");
                }
                if reading.kv {
                    binds.push("kv");
                }
                if binds.is_empty() {
                    "no tokens, no kv".to_string()
                } else {
                    binds.join(" + ")
                }
            };
            let binds_width = generative
                .readings
                .iter()
                .map(|r| binds_of(r).chars().count())
                .max()
                .unwrap_or(0);
            for reading in &generative.readings {
                println!(
                    "  {}  {}  {:<binds_width$}  -> {} {}",
                    palette.accent(format!("{:<name_width$}", reading.name)),
                    palette.dim(format!("#{}", reading.index)),
                    binds_of(reading),
                    reading.readout,
                    reading.readout_width,
                );
                if !reading.ports.is_empty() {
                    println!(
                        "  {}      {}  {}",
                        " ".repeat(name_width),
                        " ".repeat(binds_width),
                        palette.dim(format!("<- {}", reading.ports.join(", "))),
                    );
                }
            }
            if generative.text_to_image() {
                println!(
                    "  {}",
                    palette.dim(
                        "this row has a text reading and a denoise reading: \
                         `pie run text-to-image -- --prompt \"...\"` drives it."
                    )
                );
            }
        }

        if let Some(sku) = &self.sku {
            println!(
                "  {}",
                palette.dim(format!(
                    "imported as `{sku}`; `pie model import <source> --sku <NAME>` imports \
                     the same source as another row (`--sku '?'` lists every row this build \
                     ships)."
                ))
            );
        }
        println!(
            "\n{}",
            palette.dim(format!("[model]\nmodel = \"{}\"", self.address))
        );
    }
}

pub(crate) fn fetch_snapshot(repo_id: &str) -> Result<std::path::PathBuf> {
    parse_repo_id(repo_id)?;
    println!("Fetching {repo_id}");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let label = repo_id.to_string();
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

    fn sink(&self) -> std::sync::Arc<dyn crate::local::hf::Progress> {
        self.inner.clone()
    }

    fn finish(&self) {
        self.inner.finished.store(true, Ordering::Relaxed);
        if self.inner.is_tty {
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }
    }

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

fn remove(name: String, skip_confirm: bool) -> Result<Answer> {
    let entry = one(&name)?;

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
