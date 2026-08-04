//! `pie model optimize` — precompute a model's load-time work.
//!
//! The command's contract with the user: everything it does is work the
//! engine would do at load time anyway, done ahead of time, with results that
//! are bit-identical to a cold load. What runs is *derived*, never chosen by
//! flag — the only knobs are operational (`--dry-run`, `--force`).
//!
//! Today the one derivable step is format normalization: a GGUF checkpoint's
//! blocked tensors decode to plain dtypes and the result is written as a `.zt`
//! checkpoint under Pie's own cache — pie's own artifact, so it is written in
//! the format that carries page alignment and a per-tensor digest rather than
//! the one other tools read. Family-aware steps — offline quantization to the
//! configured scheme, page alignment for expert streaming — slot in behind
//! the same command once the driver can author their contracts without
//! booting a device (`author_contract` reads only the checkpoint table and
//! parsed config, so that is plumbing, not design).

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use clap::Args;

use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::write::WriteTensor;
use pie_loader::checkpoint::write::write_zt;
use pie_loader::contract::normalize::normalize_contract;
use pie_loader::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};

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
}

pub fn run(args: OptimizeArgs) -> Result<()> {
    let snapshot = resolve_snapshot(&args.repo_id)?;
    let metadata = parse_checkpoint_metadata(&snapshot)
        .map_err(|err| anyhow!("cannot read {}: {err}", snapshot.display()))?;

    let Some(normalization) =
        normalize_contract(&metadata).map_err(|err| anyhow!("cannot optimize: {err}"))?
    else {
        println!(
            "{}: nothing to precompute — the checkpoint is already plain dtypes",
            args.repo_id
        );
        return Ok(());
    };

    // The artifact is keyed by what it was computed from, so a re-download
    // or a different revision lands beside the old one rather than over it.
    let digest = source_digest(&metadata);
    let out_dir = optimized_dir(&args.repo_id, digest);
    // `.zt`: the artifact is pie's own, so it is written in the format that
    // carries alignment and a digest rather than the one other tools read.
    let out_file = out_dir.join("model.zt");
    if out_file.exists() && !args.force {
        println!(
            "{}: already optimized at {}",
            args.repo_id,
            out_dir.display()
        );
        return Ok(());
    }

    println!(
        "normalize: decode {} blocked tensor(s) to plain dtypes, pass {} through",
        normalization.decoded.len(),
        normalization.passthrough.len()
    );
    if args.dry_run {
        println!("dry run: would write {}", out_file.display());
        return Ok(());
    }

    let target = StorageTarget {
        tile_map_mask: CONVERT_TILE_MAP_MASK,
        max_tile_bytes: 64 << 20,
        ..StorageTarget::default()
    };
    let plan = pie_loader::plan::compile(&metadata, &normalization.contract, target)
        .map_err(|err| anyhow!("cannot compile the normalization: {err}"))?;
    let started = std::time::Instant::now();
    let mut bar = ProgressLine::new();
    let storage = pie_loader::testkit::host_executor::execute_plan_with_progress(
        &plan,
        &snapshot,
        &mut |progress| bar.render(&progress),
    )
    .map_err(|err| anyhow!("normalization failed: {err}"))?;
    bar.finish();

    let mut tensors = Vec::new();
    for decl in &plan.tensors {
        let bytes = storage
            .tensors
            .get(&decl.name)
            .ok_or_else(|| anyhow!("plan declared '{}' but produced nothing", decl.name))?;
        tensors.push(WriteTensor { decl, bytes });
    }
    let mut provenance = std::collections::BTreeMap::new();
    provenance.insert("pie_optimize".to_string(), "normalize".to_string());
    provenance.insert("pie_optimize_source".to_string(), format!("{digest:016x}"));
    provenance.insert(
        "pie_optimize_compiler".to_string(),
        pie_loader::plan::compiler_version().to_string(),
    );
    if args.force {
        std::fs::remove_dir_all(&out_dir).ok();
    }
    write_zt(&out_file, &provenance, &tensors)
        .map_err(|err| anyhow!("cannot write the optimized checkpoint: {err}"))?;

    let bytes: u64 = tensors.iter().map(|tensor| tensor.bytes.len() as u64).sum();
    println!(
        "{}: {} tensors, {} MB in {:.1?} → {}",
        args.repo_id,
        tensors.len(),
        bytes / (1 << 20),
        started.elapsed(),
        out_file.display()
    );
    Ok(())
}

/// A single-line, byte-weighted progress bar over the executing plan.
///
/// Renders to stderr only when stderr is a terminal, throttled so the redraw
/// never becomes the work. The name shown is the last tensor the plan
/// published, which is the executor's own notion of "where it is".
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

    fn render(&mut self, progress: &pie_loader::testkit::host_executor::Progress<'_>) {
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
        // Same bar glyphs and the same binary units as `pie model download`.
        // This drew `#`/`-` in decimal GB while that drew blocks in GiB, so
        // the two progress displays in one CLI disagreed about both the shape
        // of a bar and the size of a gigabyte.
        let body = format!(
            "  {}{} {percent:3}%  {}/{}  {}",
            "█".repeat(filled),
            "░".repeat(20 - filled),
            crate::ui::bytes(progress.read_bytes),
            crate::ui::bytes(progress.total_read_bytes),
            self.current,
        );
        eprint!("\r\x1b[K{}", crate::ui::clip(&body, crate::ui::width()));
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
fn source_digest(metadata: &pie_loader::checkpoint::CheckpointMetadata) -> u64 {
    let source = metadata
        .files
        .iter()
        .map(|file| {
            let name = std::path::Path::new(&file.path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&file.path);
            format!("{name}:{}", file.size_bytes)
        })
        .collect::<Vec<_>>()
        .join(",");
    pie_loader::cache_key::fnv1a(source.as_bytes())
}
