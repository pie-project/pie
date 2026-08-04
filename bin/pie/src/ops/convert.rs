//! `pie model optimize` — materialize a model as pie's own checkpoint format.
//!
//! The command's contract with the user: everything it does is work the
//! engine would do at load time anyway, done ahead of time, with results that
//! are bit-identical to a cold load. What runs is *derived*, never chosen by
//! flag — the only knobs are operational (`--dry-run`, `--force`,
//! `--delete-source`).
//!
//! Every checkpoint materializes the same way: it is rewritten as a `.zt`
//! artifact under Pie's own cache. Tensors whose encoding the loader can
//! decode (GGUF's blocked schemes) decode to plain dtypes; everything else is
//! copied byte for byte, keeping its encoding — `.zt` carries quantization
//! schemes parametrically, so the copy is exact. What the format then gives
//! for free is the point of the exercise: every tensor lands on a 64 KiB page
//! of its own (what lets the driver mmap-stream routed experts), carries an
//! XXH3 digest, and records its provenance in the file.
//!
//! Passthrough tensors stream from the source through a bounded buffer, so
//! materializing a checkpoint far larger than memory is fine; only the
//! decoded set is ever resident, and only GGUF checkpoints decode today.
//!
//! Family-aware steps — offline quantization to the configured scheme — slot
//! in behind the same command through the driver's device-free
//! `author_contract` entry (`pie_worker::contract_author`).

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;

use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::write::CheckpointWriter;
use pie_loader::checkpoint::{CheckpointMetadata, RawTensor};
use pie_loader::contract::materialize::materialize_contract;
use pie_loader::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use pie_loader::testkit::host_executor::Progress;
use pie_loader::types::{CheckpointFormat, TensorDecl, Visibility};

#[derive(Args, Debug)]
pub struct OptimizeArgs {
    /// HuggingFace repo ID (must already be downloaded; see `pie model download`).
    pub repo_id: String,
    /// Report what would be done — steps, tensor counts, destination —
    /// without doing it.
    #[arg(long)]
    pub dry_run: bool,
    /// Regenerate even when an up-to-date artifact already exists.
    #[arg(long)]
    pub force: bool,
    /// After the artifact is written and every tensor digest verifies, delete
    /// the source weight files it was computed from. Config and tokenizer
    /// files stay.
    #[arg(long)]
    pub delete_source: bool,
}

pub fn run(args: OptimizeArgs) -> Result<()> {
    let snapshot = resolve_snapshot(&args.repo_id)?;
    let metadata = parse_checkpoint_metadata(&snapshot)
        .map_err(|err| anyhow!("cannot read {}: {err}", snapshot.display()))?;

    if metadata
        .files
        .iter()
        .all(|file| file.format == CheckpointFormat::Zt)
    {
        println!(
            "{}: already pie's own format; nothing to materialize",
            args.repo_id
        );
        return Ok(());
    }

    // The artifact is keyed by what it was computed from, so a re-download
    // or a different revision lands beside the old one rather than over it.
    let digest = source_digest(&metadata);
    let out_dir = optimized_dir(&args.repo_id, digest);
    let out_file = out_dir.join("model.zt");
    if out_file.exists() && !args.force {
        println!(
            "{}: already optimized at {}",
            args.repo_id,
            out_dir.display()
        );
        // An artifact already standing in for the source is exactly the
        // situation the flag describes, so honor it here too.
        if args.delete_source {
            if args.dry_run {
                report_would_delete(&metadata);
                return Ok(());
            }
            return delete_source(&args.repo_id, &metadata, &out_file);
        }
        return Ok(());
    }

    let materialization =
        materialize_contract(&metadata).map_err(|err| anyhow!("cannot optimize: {err}"))?;
    println!(
        "materialize: decode {} blocked tensor(s) to plain dtypes, copy {} through",
        materialization.decoded.len(),
        materialization.passthrough.len()
    );
    if args.dry_run {
        println!("dry run: would write {}", out_file.display());
        if args.delete_source {
            report_would_delete(&metadata);
        }
        return Ok(());
    }
    if args.force {
        std::fs::remove_dir_all(&out_dir).ok();
    }

    // The passthrough set, resolved to source addresses up front — the copy
    // total is part of the progress denominator from the first frame.
    let mut passthrough: Vec<&RawTensor> = Vec::with_capacity(materialization.passthrough.len());
    for name in &materialization.passthrough {
        passthrough.push(
            metadata.tensor_by_name(name).ok_or_else(|| {
                anyhow!("'{name}' is in the materialization but not the checkpoint")
            })?,
        );
    }
    let copy_bytes: u64 = passthrough.iter().map(|raw| raw.span_bytes).sum();

    let started = std::time::Instant::now();
    let mut bar = ProgressLine::new();

    // Decode phase: only the blocked tensors go through the plan executor,
    // so only they are ever resident.
    let mut decode_read_bytes = 0u64;
    let decoded = if materialization.contract.tensors.is_empty() {
        None
    } else {
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            max_tile_bytes: 64 << 20,
            ..StorageTarget::default()
        };
        let plan = pie_loader::plan::compile(&metadata, &materialization.contract, target)
            .map_err(|err| anyhow!("cannot compile the decode: {err}"))?;
        let storage = pie_loader::testkit::host_executor::execute_plan_with_progress(
            &plan,
            &snapshot,
            &mut |progress| {
                decode_read_bytes = progress.total_read_bytes;
                bar.render(&Progress {
                    read_bytes: progress.read_bytes,
                    total_read_bytes: progress.total_read_bytes + copy_bytes,
                    finalized: progress.finalized,
                });
            },
        )
        .map_err(|err| anyhow!("decoding failed: {err}"))?;
        Some((plan, storage))
    };

    // Write phase: one pass over the union in ascending name order (canonical
    // form), decoded tensors from executor storage, passthrough streamed from
    // the source files through one bounded buffer.
    enum From<'a> {
        Decoded(&'a TensorDecl),
        Copy(&'a RawTensor),
    }
    let mut entries: Vec<(&str, From<'_>)> = Vec::new();
    if let Some((plan, _)) = &decoded {
        for decl in &plan.tensors {
            entries.push((&decl.name, From::Decoded(decl)));
        }
    }
    for raw in &passthrough {
        entries.push((&raw.name, From::Copy(raw)));
    }
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut provenance = BTreeMap::new();
    provenance.insert("pie_optimize".to_string(), "materialize".to_string());
    provenance.insert("pie_optimize_source".to_string(), format!("{digest:016x}"));
    provenance.insert(
        "pie_optimize_compiler".to_string(),
        pie_loader::plan::compiler_version().to_string(),
    );
    let mut writer = CheckpointWriter::create(&out_file, &provenance)
        .map_err(|err| anyhow!("cannot write the optimized checkpoint: {err}"))?;

    let mut sources: std::collections::HashMap<u32, std::fs::File> =
        std::collections::HashMap::new();
    let mut buffer = vec![0u8; 16 << 20];
    let mut copied = 0u64;
    let mut written_bytes = 0u64;
    for (name, entry) in &entries {
        match entry {
            From::Decoded(decl) => {
                let (_, storage) = decoded.as_ref().expect("decoded entries imply storage");
                let bytes = storage
                    .tensors
                    .get(*name)
                    .ok_or_else(|| anyhow!("the plan declared '{name}' but produced nothing"))?;
                writer
                    .add_tensor(decl, bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                written_bytes += bytes.len() as u64;
            }
            From::Copy(raw) => {
                let file = metadata
                    .files
                    .iter()
                    .find(|file| file.id == raw.file_id)
                    .ok_or_else(|| anyhow!("'{name}' points at a file the checkpoint lacks"))?;
                let handle = match sources.entry(raw.file_id.0) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                        std::fs::File::open(&file.path)
                            .with_context(|| format!("cannot open {}", file.path))?,
                    ),
                };
                let decl = TensorDecl {
                    id: raw.id,
                    name: raw.name.clone(),
                    shape: raw.shape.clone(),
                    encoding: raw.encoding.clone(),
                    alignment: 1,
                    visibility: Visibility::default(),
                };
                writer
                    .begin_tensor(&decl, raw.span_bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                handle
                    .seek(SeekFrom::Start(raw.file_offset))
                    .with_context(|| format!("cannot seek in {}", file.path))?;
                let mut remaining = raw.span_bytes;
                while remaining > 0 {
                    let take = remaining.min(buffer.len() as u64) as usize;
                    handle
                        .read_exact(&mut buffer[..take])
                        .with_context(|| format!("cannot read '{name}' from {}", file.path))?;
                    writer
                        .write(&buffer[..take])
                        .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                    remaining -= take as u64;
                    copied += take as u64;
                    bar.render(&Progress {
                        read_bytes: decode_read_bytes + copied,
                        total_read_bytes: decode_read_bytes + copy_bytes,
                        finalized: Some(name),
                    });
                }
                writer
                    .end_tensor()
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                written_bytes += raw.span_bytes;
            }
        }
    }
    writer
        .finish()
        .map_err(|err| anyhow!("cannot write the optimized checkpoint: {err}"))?;
    bar.finish();

    println!(
        "{}: {} tensors, {} MB in {:.1?} → {}",
        args.repo_id,
        entries.len(),
        written_bytes / (1 << 20),
        started.elapsed(),
        out_file.display()
    );
    if args.delete_source {
        return delete_source(&args.repo_id, &metadata, &out_file);
    }
    Ok(())
}

fn report_would_delete(metadata: &CheckpointMetadata) {
    let bytes: u64 = metadata.files.iter().map(|file| file.size_bytes).sum();
    println!(
        "dry run: would then delete {} source weight file(s), freeing {} MB",
        metadata.files.len(),
        bytes / (1 << 20)
    );
}

/// Deletes the source weight files, after proving the artifact whole.
///
/// The order is the safety argument: every tensor digest in the artifact is
/// verified *first*, so the bytes being deleted are bytes the artifact
/// provably carries. Config and tokenizer files are untouched — only the
/// checkpoint files the metadata names go, each with the blob its cache
/// symlink points at, plus the shard index that would otherwise keep naming
/// files that no longer exist.
fn delete_source(repo_id: &str, metadata: &CheckpointMetadata, artifact: &Path) -> Result<()> {
    let verified = pie_loader::checkpoint::zt::verify_checkpoint(artifact).map_err(|err| {
        anyhow!(
            "refusing to delete the source: {} does not verify: {err}",
            artifact.display()
        )
    })?;

    let mut removed = 0usize;
    let mut freed = 0u64;
    for file in &metadata.files {
        remove_cache_file(Path::new(&file.path))?;
        removed += 1;
        freed += file.size_bytes;
    }
    if let Some(dir) = metadata
        .files
        .first()
        .and_then(|file| Path::new(&file.path).parent())
    {
        let index = dir.join("model.safetensors.index.json");
        if index.exists() {
            remove_cache_file(&index)?;
        }
    }
    println!(
        "{repo_id}: artifact verified ({verified} tensors), deleted {removed} source file(s), freed {} MB",
        freed / (1 << 20)
    );
    Ok(())
}

/// Removes one file from an HF cache: the snapshot entry is usually a symlink
/// into `blobs/`, and the bytes live at the target, so both go.
fn remove_cache_file(path: &Path) -> Result<()> {
    let target = std::fs::symlink_metadata(path)
        .with_context(|| format!("cannot stat {}", path.display()))?
        .file_type()
        .is_symlink()
        .then(|| std::fs::canonicalize(path).ok())
        .flatten();
    std::fs::remove_file(path).with_context(|| format!("cannot delete {}", path.display()))?;
    if let Some(target) = target {
        std::fs::remove_file(&target)
            .with_context(|| format!("cannot delete {}", target.display()))?;
    }
    Ok(())
}

/// A single-line, byte-weighted progress bar over the whole materialization —
/// decode reads and passthrough copies count toward one denominator.
///
/// Renders to stderr only when stderr is a terminal, throttled so the redraw
/// never becomes the work. The name shown is the last tensor published.
struct ProgressLine {
    terminal: bool,
    last_draw: std::time::Instant,
    current: String,
    drew: bool,
}

impl ProgressLine {
    fn new() -> Self {
        use std::io::IsTerminal;
        Self {
            terminal: std::io::stderr().is_terminal(),
            last_draw: std::time::Instant::now(),
            current: String::new(),
            drew: false,
        }
    }

    fn render(&mut self, progress: &Progress<'_>) {
        if !self.terminal {
            return;
        }
        if let Some(name) = progress.finalized {
            self.current = name.to_string();
        }
        let done = progress.read_bytes >= progress.total_read_bytes;
        if !done && self.last_draw.elapsed() < std::time::Duration::from_millis(100) {
            return;
        }
        self.last_draw = std::time::Instant::now();
        self.drew = true;
        let percent = if progress.total_read_bytes == 0 {
            100
        } else {
            (progress.read_bytes * 100 / progress.total_read_bytes).min(100)
        };
        let filled = (percent / 5) as usize;
        let gb = |bytes: u64| bytes as f64 / 1e9;
        eprint!(
            "\r  [{}{}] {percent:3}%  {:.2}/{:.2} GB  {:<48.48}",
            "#".repeat(filled),
            "-".repeat(20 - filled),
            gb(progress.read_bytes),
            gb(progress.total_read_bytes),
            self.current
        );
    }

    fn finish(&mut self) {
        if self.drew {
            eprintln!();
        }
    }
}

/// The snapshot directory of a downloaded repo: `models--org--name/snapshots/`
/// holds one directory per revision; like the rest of `pie model`, the first
/// one present is the one in use.
fn resolve_snapshot(repo_id: &str) -> Result<PathBuf> {
    let repo_dir =
        hf_hub::resolve_cache_dir().join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let entries = std::fs::read_dir(&snapshots)
        .map_err(|_| anyhow!("{repo_id} is not downloaded; run `pie model download {repo_id}`"))?;
    let snapshot = entries
        .filter_map(|entry| entry.ok())
        .find(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|entry| entry.path());
    match snapshot {
        Some(path) => Ok(path),
        None => bail!("{repo_id} has no snapshot under {}", snapshots.display()),
    }
}

/// `$PIE_HOME/optimized/models--org--name/<digest>/`.
fn optimized_dir(repo_id: &str, digest: u64) -> PathBuf {
    startup::paths::pie_home()
        .join("optimized")
        .join(format!("models--{}", repo_id.replace('/', "--")))
        .join(format!("{digest:016x}"))
}

/// Identity of the source checkpoint, from the same facts the loader's own
/// artifact key uses: file names and sizes, never machine paths.
fn source_digest(metadata: &CheckpointMetadata) -> u64 {
    let source = metadata
        .files
        .iter()
        .map(|file| {
            let name = Path::new(&file.path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&file.path);
            format!("{name}:{}", file.size_bytes)
        })
        .collect::<Vec<_>>()
        .join(",");
    pie_loader::cache_key::fnv1a(source.as_bytes())
}
