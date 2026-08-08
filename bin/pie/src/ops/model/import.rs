//! `pie model import` — rewrite any checkpoint as pie's canonical artifact,
//! fetching it first when the source is a repo ID that is not here yet.
//!
//! The conversion engine. Format diversity is handled here, once, and never
//! again: what the runtime serves is always one `.zt` file, so the question
//! "which files make up this model, and in what format" is answered at import
//! rather than at every serve boot.
//!
//! The command's contract with the user: everything it does is work the engine
//! would do at load time anyway, done ahead of time, with results that are
//! bit-identical to a cold load. What runs is *derived*, never chosen by flag —
//! the knobs are operational (`--dry-run`, `--force`, `--delete-source`) or
//! about placement (`--out`), never about what the artifact means.
//!
//! Every checkpoint converts the same way. Tensors whose encoding the loader
//! can decode (GGUF's blocked schemes) decode to plain dtypes; everything else
//! is copied byte for byte, keeping its encoding — `.zt` carries quantization
//! schemes parametrically, so the copy is exact. What the format then gives for
//! free is the point of the exercise: every tensor lands on a 64 KiB page of
//! its own (what lets the driver mmap-stream routed experts), carries an XXH3
//! digest, and records its provenance in the file.
//!
//! Passthrough tensors stream from the source through a bounded buffer, so
//! converting a checkpoint far larger than memory is fine; only the decoded set
//! is ever resident, and only GGUF checkpoints decode today.
//!
//! The family-aware step landed as its own command: `pie model build`
//! authors the serve contract through `pie_model::contract` — no FFI, no
//! driver — and materializes it offline. This command stays the
//! family-blind half of the pair.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;

use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::write::CheckpointWriter;
use pie_loader::checkpoint::{CheckpointMetadata, RawTensor};
use pie_loader::contract::materialize::materialize_contract;
use pie_loader::executor::host::Progress;
use pie_loader::executor::sink::TensorSink;
use pie_loader::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use pie_loader::types::{CheckpointFormat, TensorDecl, Visibility};

// The artifact's on-disk names come from whoever owns them: the loader owns
// the metadata namespace and the provenance attributes, `pie.model/1` owns the
// object its descriptor lives in. A literal here would be a second definition
// of something a reader elsewhere has to match exactly, and a mismatch does
// not fail — the read just finds nothing.
use pie_loader::checkpoint::meta::{SOURCE_KEY, VERSION_KEY, meta_name};
use pie_model_config::DESCRIPTOR_OBJECT;

/// Parses a human-written byte size: `16GiB`, `5GB`, `512MiB`, `1000000`.
///
/// Both conventions are accepted and they mean different things — `GB` is
/// 10^9, `GiB` is 2^30 — because a user who writes one and gets the other has
/// been lied to about the size of their files.
pub fn parse_size(text: &str) -> Result<u64, String> {
    let text = text.trim();
    let split = text
        .find(|c: char| !c.is_ascii_digit() && c != '_')
        .unwrap_or(text.len());
    let (digits, unit) = text.split_at(split);
    let value: u64 = digits
        .replace('_', "")
        .parse()
        .map_err(|_| format!("{text:?} does not start with a number"))?;
    let scale: u64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" => 1_000,
        "m" | "mb" => 1_000_000,
        "g" | "gb" => 1_000_000_000,
        "t" | "tb" => 1_000_000_000_000,
        "kib" => 1 << 10,
        "mib" => 1 << 20,
        "gib" => 1 << 30,
        "tib" => 1u64 << 40,
        other => return Err(format!("unknown size unit {other:?}")),
    };
    value
        .checked_mul(scale)
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("{text:?} is not a usable size"))
}

/// The pie that is running, as recorded in what it writes.
///
/// The same string `pie --version` prints.
pub(crate) fn pie_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[derive(Args, Debug)]
pub struct ImportArgs {
    /// What to import: a HuggingFace repo ID, a snapshot directory, or a
    /// single `.safetensors`/`.gguf`/`.zt` file. A repo ID that is not in the
    /// local cache is fetched first.
    pub source: String,
    /// Write the artifact here instead of the model store. A path ending in
    /// `.zt` is the artifact; a directory receives `<name>.zt`.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Report what would be done — steps, tensor counts, destination —
    /// without doing it.
    #[arg(long)]
    pub dry_run: bool,
    /// Regenerate even when an up-to-date artifact already exists.
    #[arg(long)]
    pub force: bool,
    /// Split the artifact into shards of about this size (e.g. `16GiB`,
    /// `5GB`). Absent means one file, which is the default and the
    /// recommendation — see the note on `--help`.
    #[arg(long, value_name = "SIZE", value_parser = parse_size)]
    pub max_shard_size: Option<u64>,
    /// Delete source weight files only after the artifact tensors copied from
    /// them verify. With --max-shard-size, completed output shards verify and
    /// release covered source files during the import; the final open shard is
    /// released after the root verifies. Config and tokenizer files stay.
    ///
    /// Only for a source that is already on disk. Reclaiming the HuggingFace
    /// snapshots an import downloaded is `pie cache clear snapshots`, which
    /// knows about all of them rather than the one just fetched, asks before
    /// deleting, and says how much it got back.
    #[arg(long)]
    pub delete_source: bool,
}

pub fn run(args: ImportArgs) -> Result<crate::ui::Answer> {
    let source = resolve_source(&args.source)?;
    let metadata = parse_checkpoint_metadata(&source.path)
        .map_err(|err| anyhow!("cannot read {}: {err}", source.path.display()))?;

    let out_file = match &args.out {
        Some(out) => artifact_path(out, &source.name),
        None => store_path(&source.name),
    };

    // A checkpoint that is already an artifact is the one thing left alone —
    // converting `.zt` to `.zt` would rewrite bytes to reproduce them.
    if metadata
        .files
        .iter()
        .all(|file| file.format == CheckpointFormat::Zt)
        && !args.force
    {
        if args.delete_source {
            println!("  (--delete-source has nothing to do: the source is the artifact)");
        }
        return Ok(crate::ui::Answer::noop(format!(
            "{} is already pie's own format; nothing to convert",
            source.name
        )));
    }

    // Up to date means: written by this pie, from this source.
    if out_file.exists() && !args.force {
        if let Some(reason) = staleness(&out_file, pie_version(), &source.origin) {
            println!(
                "{}: rebuilding {} ({reason})",
                source.name,
                out_file.display()
            );
        } else {
            // Said in every branch, because it is the answer to what the user
            // asked. Under `--delete-source` it used to be printed here and
            // then the delete reported underneath it; folding the two into one
            // return dropped it, so the flag turned "nothing was rebuilt" into
            // silence at exactly the moment source files are being deleted.
            let up_to_date = format!(
                "{} is up to date at {}",
                source.name,
                crate::ui::short_path(&out_file)
            );
            // An artifact already standing in for the source is exactly the
            // situation the flag describes, so honor it here too.
            if args.delete_source {
                println!("{up_to_date}");
                if args.dry_run {
                    report_would_delete(&metadata);
                    return Ok(crate::ui::Answer::noop("dry run: nothing was deleted"));
                }
                delete_source(&source.name, &metadata, &out_file)?;
                return Ok(crate::ui::Answer::did("deleted the source files"));
            }
            return Ok(crate::ui::Answer::noop(up_to_date));
        }
    }

    let materialization =
        materialize_contract(&metadata).map_err(|err| anyhow!("cannot convert: {err}"))?;
    println!(
        "convert: decode {} blocked tensor(s) to plain dtypes, copy {} through",
        materialization.decoded.len(),
        materialization.passthrough.len()
    );

    // Metadata compiles here, before any bytes are written: an artifact whose
    // weights are perfect but whose tokenizer would not compile cannot serve,
    // and finding that out after copying 800 GB helps nobody.
    let tokenizer = compile_tokenizer(&source)?;
    match &tokenizer {
        Some(canonical) => println!(
            "convert: tokenizer compiled to {} ({} KiB)",
            pie_tokenizer::canonical::VERSION,
            canonical.byte_size() / 1024
        ),
        None => println!(
            "convert: no tokenizer beside the weights — the artifact will carry none, \
             and serving it needs one from elsewhere"
        ),
    }
    let descriptor = compile_descriptor(&source)?;
    match &descriptor {
        Some(_) => println!(
            "convert: model config normalized to {}",
            pie_model_config::VERSION
        ),
        None => println!("convert: no config.json beside the weights"),
    }

    if let Some(max) = args.max_shard_size {
        println!(
            "import: sharding at about {} per file; the root is {}",
            crate::ui::bytes(max),
            out_file.display()
        );
    }

    if args.dry_run {
        if args.delete_source {
            report_would_delete(&metadata);
        }
        return Ok(crate::ui::Answer::noop(format!(
            "dry run: would write {}",
            crate::ui::short_path(&out_file)
        )));
    }

    // The passthrough set, resolved to source addresses up front — the copy
    // total is part of the progress denominator from the first frame, and
    // resolving the file here leaves the copy loop nothing to look up.
    let mut passthrough: Vec<(&RawTensor, &str)> =
        Vec::with_capacity(materialization.passthrough.len());
    for name in &materialization.passthrough {
        let raw = metadata
            .tensor_by_name(name)
            .ok_or_else(|| anyhow!("'{name}' is in the materialization but not the checkpoint"))?;
        let file = metadata
            .files
            .iter()
            .find(|file| file.id == raw.file_id)
            .ok_or_else(|| anyhow!("'{name}' points at a file the checkpoint lacks"))?;
        passthrough.push((raw, file.path.as_str()));
    }
    let copy_bytes: u64 = passthrough.iter().map(|(raw, _)| raw.span_bytes).sum();

    // pie's own objects, named into the reserved namespace so the write can
    // merge them with the weights in one ascending pass.
    let mut meta: Vec<(String, Vec<u8>)> = Vec::new();
    if let Some(canonical) = &tokenizer {
        for (path, bytes) in canonical.objects() {
            meta.push((meta_name(path), bytes.to_vec()));
        }
    }
    if let Some(descriptor) = &descriptor {
        meta.push((meta_name(DESCRIPTOR_OBJECT), descriptor.clone()));
    }

    let started = std::time::Instant::now();
    let mut bar = ProgressLine::new();

    // Decode phase: the blocked tensors stream through the plan executor
    // into a disk spool, one at a time. Peak memory is one tensor's working
    // set, not the decoded set — which for an F16 checkpoint is the whole
    // model, the case that made the old collect-everything executor a
    // 2x-model-size boot. The spool holds the decoded bytes so the ascending
    // merge below can still interleave them with the passthrough copies.
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
        let mut spool = Spool::create(&out_file)?;
        pie_loader::executor::host::execute_plan_into(
            &plan,
            &source.base(),
            &mut spool,
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
        Some((plan, spool))
    };

    let provenance = BTreeMap::from([
        (VERSION_KEY.to_string(), pie_version().to_string()),
        (SOURCE_KEY.to_string(), source.origin.clone()),
    ]);
    let mut writer = match args.max_shard_size {
        Some(max) => CheckpointWriter::create_sharded(&out_file, &provenance, max),
        None => CheckpointWriter::create(&out_file, &provenance),
    }
    .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;
    let mut decoded = decoded;
    let (written_bytes, deletion) = write_artifact(
        &mut writer,
        decoded.as_mut(),
        &passthrough,
        &meta,
        &mut bar,
        decode_read_bytes,
        copy_bytes,
        &source.name,
        &metadata,
        args.delete_source,
    )?;
    // Closing belongs to whoever opened it: `finish` consumes the writer.
    writer
        .finish()
        .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;
    if let Some((_, spool)) = decoded {
        spool.remove();
    }

    // `ui::bytes` and `ui::duration`, not `/ (1 << 20)` and `{:.1?}`: this line
    // reported megabytes while every other line pie prints reports GiB, and a
    // Debug-formatted Duration ("94.31234s") is not a rendering anyone chose.
    let did = format!(
        "imported {} — {} in {} → {}",
        source.name,
        crate::ui::bytes(written_bytes),
        crate::ui::duration(started.elapsed()),
        crate::ui::short_path(&out_file)
    );
    if let Some(deletion) = deletion {
        deletion.finish(&out_file)?;
    }
    Ok(crate::ui::Answer::did(did))
}

fn report_would_delete(metadata: &CheckpointMetadata) {
    let bytes: u64 = metadata.files.iter().map(|file| file.size_bytes).sum();
    println!(
        "dry run: would then delete {} source weight file(s), freeing {} MB",
        metadata.files.len(),
        bytes / (1 << 20)
    );
}

struct SourceFileProgress {
    path: PathBuf,
    size_bytes: u64,
    remaining_tensors: usize,
    last_shard: Option<u32>,
    removed: bool,
}

/// Destructive import state, advanced only by published, reread output shards.
///
/// A tensor reaching `CheckpointWriter::end_tensor` is not enough: its digest
/// still lives only in the open writer's manifest. A source file becomes
/// reclaimable after all its tensors were written *and* the last output shard
/// carrying one of them has been finished, fsynced, published and verified.
struct IncrementalDeletion {
    repo_id: String,
    files: BTreeMap<u32, SourceFileProgress>,
    verified_shards: usize,
    removed_files: usize,
    freed_bytes: u64,
}

impl IncrementalDeletion {
    fn new(
        repo_id: &str,
        metadata: &CheckpointMetadata,
        remaining: &std::collections::HashMap<u32, usize>,
    ) -> Self {
        let files = metadata
            .files
            .iter()
            .map(|file| {
                (
                    file.id.0,
                    SourceFileProgress {
                        path: PathBuf::from(&file.path),
                        size_bytes: file.size_bytes,
                        remaining_tensors: remaining.get(&file.id.0).copied().unwrap_or(0),
                        last_shard: None,
                        removed: false,
                    },
                )
            })
            .collect();
        Self {
            repo_id: repo_id.to_string(),
            files,
            verified_shards: 0,
            removed_files: 0,
            freed_bytes: 0,
        }
    }

    fn tensor_written(&mut self, raw: &RawTensor, shard: Option<u32>) -> Result<()> {
        let file = self.files.get_mut(&raw.file_id.0).ok_or_else(|| {
            anyhow!(
                "cannot track source file {} for tensor '{}'",
                raw.file_id.0,
                raw.name
            )
        })?;
        file.remaining_tensors = file.remaining_tensors.checked_sub(1).ok_or_else(|| {
            anyhow!(
                "source tensor count underflow for '{}' in {}",
                raw.name,
                file.path.display()
            )
        })?;
        if let Some(shard) = shard {
            file.last_shard = Some(file.last_shard.map_or(shard, |last| last.max(shard)));
        }
        Ok(())
    }

    fn verify_published(
        &mut self,
        writer: &CheckpointWriter,
        sources: &mut std::collections::HashMap<u32, std::fs::File>,
    ) -> Result<()> {
        let published: Vec<PathBuf> = writer
            .published_shards()
            .skip(self.verified_shards)
            .map(Path::to_path_buf)
            .collect();
        for path in published {
            let verified = pie_loader::checkpoint::zt::verify_checkpoint(&path).map_err(|err| {
                anyhow!(
                    "refusing to delete a source: published shard {} does not verify: {err}",
                    path.display()
                )
            })?;
            self.verified_shards += 1;
            println!(
                "import: verified output shard {} ({} tensors)",
                path.display(),
                verified
            );
        }

        let eligible: Vec<u32> = self
            .files
            .iter()
            .filter(|(_, file)| {
                !file.removed
                    && file.remaining_tensors == 0
                    && file
                        .last_shard
                        .is_some_and(|shard| shard as usize <= self.verified_shards)
            })
            .map(|(&id, _)| id)
            .collect();
        for id in eligible {
            // Do not leave an open handle behind after the directory entry is
            // removed. Unix permits it; other supported hosts need not.
            sources.remove(&id);
            self.remove_file(id)?;
        }
        Ok(())
    }

    fn remove_file(&mut self, id: u32) -> Result<()> {
        let file = self
            .files
            .get_mut(&id)
            .expect("eligible source file came from this map");
        remove_cache_file(&file.path)?;
        file.removed = true;
        self.removed_files += 1;
        self.freed_bytes += file.size_bytes;
        println!(
            "{}: deleted verified source shard {}, freed {}",
            self.repo_id,
            file.path.display(),
            crate::ui::bytes(file.size_bytes)
        );
        Ok(())
    }

    fn finish(mut self, artifact: &Path) -> Result<()> {
        let verified = pie_loader::checkpoint::zt::verify_checkpoint(artifact).map_err(|err| {
            anyhow!(
                "refusing to delete the remaining source: {} does not verify: {err}",
                artifact.display()
            )
        })?;
        let pending: Vec<u32> = self
            .files
            .iter()
            .filter(|(_, file)| !file.removed)
            .map(|(&id, _)| id)
            .collect();
        for id in pending {
            self.remove_file(id)?;
        }
        if let Some(dir) = self
            .files
            .values()
            .next()
            .and_then(|file| file.path.parent())
        {
            let index = dir.join("model.safetensors.index.json");
            if index.exists() {
                remove_cache_file(&index)?;
            }
        }
        println!(
            "{}: artifact verified ({} tensors), deleted {} source file(s), freed {}",
            self.repo_id,
            verified,
            self.removed_files,
            crate::ui::bytes(self.freed_bytes)
        );
        Ok(())
    }
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
        "{repo_id}: artifact verified ({verified} tensors), deleted {removed} source file(s), freed {}",
        crate::ui::bytes(freed)
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

/// The decoded tensors, spooled to disk beside the artifact.
///
/// The executor streams tensors out in schedule order; the artifact writer
/// needs them back in ascending-name order, interleaved with the passthrough
/// copies. The spool is the buffer between the two orders, and it is a file
/// rather than a map so the buffer costs disk instead of memory — the
/// decoded set is the whole model for an F16 checkpoint.
pub(crate) struct Spool {
    path: PathBuf,
    file: std::fs::File,
    index: BTreeMap<String, (u64, u64)>,
    offset: u64,
}

impl Spool {
    pub(crate) fn create(out_file: &Path) -> Result<Self> {
        // Beside the artifact, so it lands on the same filesystem the bytes
        // are headed for anyway.
        let path = out_file.with_extension("spool.tmp");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create {}", parent.display()))?;
        }
        let file = std::fs::File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("cannot create spool {}", path.display()))?;
        Ok(Self {
            path,
            file,
            index: BTreeMap::new(),
            offset: 0,
        })
    }

    pub(crate) fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        let (offset, len) = *self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("the plan declared '{name}' but produced nothing"))?;
        let mut bytes = vec![0u8; len as usize];
        use std::io::{Read, Seek, SeekFrom};
        self.file
            .seek(SeekFrom::Start(offset))
            .and_then(|_| self.file.read_exact(&mut bytes))
            .with_context(|| format!("cannot read '{name}' back from the spool"))?;
        Ok(bytes)
    }

    pub(crate) fn remove(self) {
        drop(self.file);
        std::fs::remove_file(&self.path).ok();
    }
}

impl TensorSink for Spool {
    fn publish(
        &mut self,
        name: &str,
        bytes: &[u8],
    ) -> std::result::Result<(), pie_loader::error::Error> {
        use std::io::Write;
        self.file.write_all(bytes).map_err(|err| {
            pie_loader::error::Error::Checkpoint(format!(
                "cannot spool '{name}' to {}: {err}",
                self.path.display()
            ))
        })?;
        self.index
            .insert(name.to_string(), (self.offset, bytes.len() as u64));
        self.offset += bytes.len() as u64;
        Ok(())
    }
}

/// A single-line, byte-weighted progress bar over the whole materialization —
/// decode reads and passthrough copies count toward one denominator.
///
/// Renders to stderr only when stderr is a terminal, throttled so the redraw
/// never becomes the work. The name shown is the last tensor published.
/// The import progress bar: [`crate::ui::Bar`], fed from the loader's own
/// `Progress`.
///
/// The adapter is here rather than in `ui` so the presentation module stays
/// free of `pie_loader` -- what it needs to draw a bar is two numbers and a
/// label, and `Progress` is where those two numbers happen to live today.
pub(crate) struct ProgressLine {
    bar: crate::ui::Bar,
    current: String,
}

impl ProgressLine {
    pub(crate) fn new() -> Self {
        Self {
            bar: crate::ui::Bar::new(),
            current: String::new(),
        }
    }

    pub(crate) fn render(&mut self, progress: &Progress<'_>) {
        if let Some(name) = progress.finalized {
            self.current = name.to_string();
        }
        self.bar
            .draw(progress.read_bytes, progress.total_read_bytes, &self.current);
    }

    pub(crate) fn finish(&mut self) {
        self.bar.finish();
    }
}

/// What `convert` was pointed at, once the pointing is resolved.
pub(crate) struct Source {
    /// The path the loader reads — a snapshot directory or a single file.
    pub(crate) path: PathBuf,
    /// The artifact's name in the store, without the `.zt` suffix.
    pub(crate) name: String,
    /// Where the bytes came from, recorded in the artifact's provenance.
    pub(crate) origin: String,
}

impl Source {
    /// The directory relative paths in the plan resolve against.
    ///
    /// A plan carries its own file table and the executor uses it; this is only
    /// the base for entries that are relative, which is why a single-file
    /// source resolves against the file's directory rather than the file.
    pub(crate) fn base(&self) -> PathBuf {
        if self.path.is_file() {
            self.path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
        } else {
            self.path.clone()
        }
    }
}

/// Resolves the `<source>` argument to something the loader can read.
///
/// Three forms, decided by the filesystem rather than by syntax: an existing
/// path is used as given (a snapshot directory, or a single checkpoint file),
/// and anything else is taken for a HuggingFace repo ID and looked up in the
/// local cache. Deciding on existence rather than on shape is what lets a repo
/// ID and a relative directory share a spelling — `qwen/qwen3-0.6b` is a repo
/// ID unless there is a directory of that name, in which case the user plainly
/// meant the directory.
pub(crate) fn resolve_source(source: &str) -> Result<Source> {
    let path = Path::new(source);
    if path.exists() {
        let name = if path.is_file() {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("model")
                .to_string()
        } else {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model")
                .to_string()
        };
        let origin = std::fs::canonicalize(path)
            .unwrap_or_else(|_| path.to_path_buf())
            .display()
            .to_string();
        return Ok(Source {
            path: path.to_path_buf(),
            name: store_name(&name),
            origin,
        });
    }

    let snapshot = resolve_snapshot(source)?;
    Ok(Source {
        path: snapshot,
        name: store_name(source),
        origin: source.to_string(),
    })
}

/// The snapshot directory of a downloaded repo: `models--org--name/snapshots/`
/// holds one directory per revision; like the rest of `pie model`, the first
/// one present is the one in use.
fn resolve_snapshot(repo_id: &str) -> Result<PathBuf> {
    let repo_dir =
        hf_hub::resolve_cache_dir().join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    if !snapshots.exists() {
        // Fetched here rather than by a separate `download` command. Whether a
        // source needs the network is a property of that source, not a
        // different operation -- and a `download` that stopped at the snapshot
        // left the user one undiscoverable step short of a servable model,
        // which is why it converted too. Two commands doing fetch-and-convert
        // and convert is one command with an argument.
        crate::ops::model::fetch_snapshot(repo_id)?;
    }
    let entries = std::fs::read_dir(&snapshots).map_err(|_| {
        anyhow!("{repo_id} is neither a path nor a model any registry has")
    })?;
    let snapshot = entries
        .filter_map(|entry| entry.ok())
        .find(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|entry| entry.path());
    match snapshot {
        Some(path) => Ok(path),
        None => bail!("{repo_id} has no snapshot under {}", snapshots.display()),
    }
}

/// A repo ID as one filesystem name: `qwen/qwen3-0.6b` → `qwen--qwen3-0.6b`.
///
/// The store is a flat directory, so the separator has to survive as something
/// legal in a filename. `--` rather than a single `-` because model names
/// contain single hyphens freely and the mapping has to stay reversible.
fn store_name(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

/// `$PIE_HOME/models/<name>.zt` — one model, one file, one flat directory.
pub(crate) fn store_path(name: &str) -> PathBuf {
    startup::paths::pie_home()
        .join("models")
        .join(format!("{name}.zt"))
}

/// Where `--out` puts the artifact: a `.zt` path names the file, anything else
/// is a directory to put `<name>.zt` in.
pub(crate) fn artifact_path(out: &Path, name: &str) -> PathBuf {
    if out
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zt"))
    {
        out.to_path_buf()
    } else {
        out.join(format!("{name}.zt"))
    }
}

/// The tokenizer file beside the weights, if there is one.
///
/// The convention the worker already uses (`translate.rs`): `tokenizer.json`,
/// else `tiktoken.model`. A single checkpoint file has no snapshot to look in.
fn tokenizer_path(source: &Source) -> Option<PathBuf> {
    if source.path.is_file() {
        return None;
    }
    let json = source.path.join("tokenizer.json");
    if json.exists() {
        return Some(json);
    }
    let tiktoken = source.path.join("tiktoken.model");
    tiktoken.exists().then_some(tiktoken)
}

/// Compiles the source's `config.json` into the `pie.model/1` descriptor.
///
/// `Ok(None)` when there is no `config.json` — a lone `.gguf` carries its
/// metadata in its own header, and a directory without one is a weights-only
/// checkpoint. An *unreadable or unrecognized* one is an error, for the same
/// reason a broken tokenizer is: the artifact would not serve, and the failure
/// belongs at import where it can be read once.
pub(crate) fn compile_descriptor(source: &Source) -> Result<Option<Vec<u8>>> {
    if source.path.is_file() {
        return Ok(None);
    }
    let path = source.path.join("config.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let root: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| anyhow!("cannot parse {}: {err}", path.display()))?;
    let descriptor = pie_model_config::descriptor(&root, &path.display().to_string())
        .map_err(|err| anyhow!("cannot normalize {}: {err:#}", path.display()))?;
    Ok(Some(serde_json::to_vec(&descriptor)?))
}

/// Compiles the source's tokenizer into its canonical form, if it has one.
///
/// Discovery follows the convention the worker already uses (`translate.rs`):
/// `tokenizer.json`, else `tiktoken.model`, beside the weights. A source that
/// is a single checkpoint file has no snapshot to look in and so has no
/// tokenizer — that is `Ok(None)`, not an error, because converting a lone
/// `.gguf` for its weights is a legitimate thing to do.
///
/// A tokenizer that is *present but does not compile* is an error, and this is
/// where the plan's "rejection moves to import" is actually paid for. pie's
/// tokenizer accepts a small number of modern pipelines and refuses the rest
/// (SentencePiece checkpoints with no `pre_tokenizer`, non-isolated regex
/// splits); today that refusal surfaces at serve boot, after a model has been
/// downloaded and loaded. Failing here means it surfaces once, at import, with
/// the reason — and never produces an artifact that cannot serve.
pub(crate) fn compile_tokenizer(
    source: &Source,
) -> Result<Option<pie_tokenizer::canonical::CanonicalTokenizer>> {
    let Some(path) = tokenizer_path(source) else {
        return Ok(None);
    };

    let tokenizer = pie_tokenizer::Tokenizer::from_file(&path).map_err(|err| {
        anyhow!(
            "cannot compile {}: {err:#}\n\
             pie compiles every tokenizer into one of a small number of modern \
             pipelines, and this one is outside that set. The artifact is not \
             written, because one without a working tokenizer cannot serve.",
            path.display()
        )
    })?;
    tokenizer
        .to_canonical()
        .map(Some)
        .map_err(|err| anyhow!("cannot serialize {}: {err:#}", path.display()))
}

/// Why an existing artifact needs rebuilding, or `None` if it is current.
///
/// An artifact is a function of the pie that wrote it and of what it was
/// written from, so those are the two things compared. The version standing in
/// for the whole converter is deliberate: it changes whenever the plan
/// compiler, either metadata schema, or the layout rules do, and nobody has to
/// remember to add a key when a new one lands.
///
/// It is a release version, not a build identity, so it does *not* move while
/// the converter is being worked on. `--force` is the tool for that.
fn staleness(artifact: &Path, version: &str, source: &str) -> Option<String> {
    let attributes = match pie_loader::checkpoint::zt::read_attributes(artifact) {
        Ok(attributes) => attributes,
        Err(err) => return Some(format!("cannot read its provenance: {err}")),
    };
    match attributes.get(VERSION_KEY) {
        None => return Some("it records no pie version".to_string()),
        Some(recorded) if recorded != version => {
            return Some(format!("pie changed: {recorded} → {version}"));
        }
        Some(_) => {}
    }
    match attributes.get(SOURCE_KEY) {
        None => Some("it records no source".to_string()),
        Some(recorded) if recorded != source => {
            Some(format!("the source changed: {recorded} → {source}"))
        }
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_loader::checkpoint::write::{WriteTensor, write_zt};
    use pie_loader::types::{DType, Encoding, TensorId};

    #[test]
    fn a_repo_id_becomes_one_flat_store_name() {
        assert_eq!(store_name("Qwen/Qwen3-0.6B"), "Qwen--Qwen3-0.6B");
        // A name that already contains hyphens survives, which is why the
        // separator is doubled: `--` cannot be confused for one of them.
        assert_eq!(
            store_name("meta-llama/Llama-3.1-8B"),
            "meta-llama--Llama-3.1-8B"
        );
        // A bare name has no separator to translate.
        assert_eq!(store_name("mymodel"), "mymodel");
    }

    #[test]
    fn sizes_distinguish_the_two_conventions() {
        // `GB` and `GiB` are different numbers, and a user who writes one and
        // is given the other has been told the wrong thing about their files.
        assert_eq!(parse_size("5GB").unwrap(), 5_000_000_000);
        assert_eq!(parse_size("5GiB").unwrap(), 5 << 30);
        assert_eq!(parse_size("16gib").unwrap(), 16 << 30);
        assert_eq!(parse_size("512MiB").unwrap(), 512 << 20);
        assert_eq!(parse_size("1_000_000").unwrap(), 1_000_000);
        assert_eq!(parse_size("2048").unwrap(), 2048);
        assert_eq!(parse_size(" 4 GiB ").unwrap(), 4 << 30);

        assert!(parse_size("").is_err());
        assert!(parse_size("GiB").is_err());
        assert!(parse_size("5 furlongs").is_err());
        // Zero would put every tensor in a file of its own.
        assert!(parse_size("0").is_err());
    }

    #[test]
    fn out_names_a_file_or_a_directory_to_put_one_in() {
        assert_eq!(
            artifact_path(Path::new("/data/custom.zt"), "qwen"),
            PathBuf::from("/data/custom.zt")
        );
        assert_eq!(
            artifact_path(Path::new("/data/models"), "qwen"),
            PathBuf::from("/data/models/qwen.zt")
        );
        // The extension decides, not the case it is written in.
        assert_eq!(
            artifact_path(Path::new("/data/custom.ZT"), "qwen"),
            PathBuf::from("/data/custom.ZT")
        );
    }

    /// An artifact is a function of the pie that wrote it and of what it was
    /// written from, so the up-to-date check compares exactly those. Provenance
    /// it does not carry at all means it predates these keys, which is itself a
    /// reason to rebuild.
    #[test]
    fn staleness_answers_for_the_pie_and_the_source() {
        let dir = tempfile::tempdir().unwrap();
        let decl = TensorDecl {
            id: TensorId(0),
            name: "w".to_string(),
            shape: vec![4],
            encoding: Encoding::Raw(DType::U8),
            alignment: 1,
            visibility: Visibility::default(),
        };
        let write = |path: &Path, provenance: &BTreeMap<String, String>| {
            write_zt(
                path,
                provenance,
                &[WriteTensor {
                    decl: &decl,
                    bytes: &[1u8, 2, 3, 4],
                }],
            )
            .unwrap();
        };

        let path = dir.path().join("model.zt");
        let mut provenance = BTreeMap::new();
        provenance.insert(VERSION_KEY.to_string(), "0.1.0".to_string());
        provenance.insert(SOURCE_KEY.to_string(), "qwen/qwen3-0.6b".to_string());
        write(&path, &provenance);

        assert_eq!(staleness(&path, "0.1.0", "qwen/qwen3-0.6b"), None);
        assert!(
            staleness(&path, "0.2.0", "qwen/qwen3-0.6b")
                .unwrap()
                .contains("pie changed")
        );
        assert!(
            staleness(&path, "0.1.0", "qwen/qwen3-4b")
                .unwrap()
                .contains("source changed")
        );

        // An artifact from before these keys existed records neither, and
        // "no provenance" is a reason to rebuild rather than to trust it.
        let bare = dir.path().join("bare.zt");
        write(&bare, &BTreeMap::new());
        assert!(
            staleness(&bare, "0.1.0", "qwen/qwen3-0.6b")
                .unwrap()
                .contains("no pie version")
        );

        // A path that is not an artifact at all fails loudly rather than
        // reporting "current".
        let junk = dir.path().join("junk.zt");
        std::fs::write(&junk, b"not a zt file").unwrap();
        assert!(staleness(&junk, "0.1.0", "qwen/qwen3-0.6b").is_some());
    }
}

/// One pass over decoded tensors, passthrough tensors and metadata, in
/// ascending name order. Returns the bytes written.
///
/// Ascending order across the *union* is what canonical `.zt` form asks for,
/// and the writer trusts its caller for it. Metadata is interleaved rather
/// than written as a block: `__meta__/` begins with `_` (0x5F), which sorts
/// after digits and capitals but before lowercase, so it lands in the middle
/// of a typical weight namespace.
///
/// Decoded tensors come from executor storage, passthrough is streamed from
/// its source file through one bounded buffer, metadata comes from memory.
/// The two byte counts are the progress denominator: what the decode already
/// read, and what this pass is about to copy.
fn write_artifact(
    writer: &mut CheckpointWriter,
    decoded: Option<&mut (pie_loader::plan::LoadPlan, Spool)>,
    passthrough: &[(&RawTensor, &str)],
    meta: &[(String, Vec<u8>)],
    progress: &mut ProgressLine,
    decode_read_bytes: u64,
    copy_bytes: u64,
    repo_id: &str,
    metadata: &CheckpointMetadata,
    delete_source: bool,
) -> Result<(u64, Option<IncrementalDeletion>)> {
    enum From<'a> {
        Decoded(&'a TensorDecl, &'a RawTensor),
        /// The tensor and the file its bytes are in.
        Copy(&'a RawTensor, &'a str),
        Meta(&'a [u8]),
    }
    let mut entries: Vec<(&str, From<'_>)> = Vec::new();
    let (decoded_plan, spool) = match decoded {
        Some((plan, spool)) => (Some(&*plan), Some(spool)),
        None => (None, None),
    };
    let mut spool = spool;
    if let Some(plan) = decoded_plan {
        for decl in &plan.tensors {
            let raw = metadata.tensor_by_name(&decl.name).ok_or_else(|| {
                anyhow!(
                    "decoded tensor '{}' is not present in the source checkpoint",
                    decl.name
                )
            })?;
            entries.push((&decl.name, From::Decoded(decl, raw)));
        }
    }
    for (raw, path) in passthrough {
        entries.push((&raw.name, From::Copy(raw, path)));
    }
    for (name, bytes) in meta {
        entries.push((name.as_str(), From::Meta(bytes)));
    }
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut remaining = std::collections::HashMap::<u32, usize>::new();
    for (_, entry) in &entries {
        let raw = match entry {
            From::Decoded(_, raw) | From::Copy(raw, _) => Some(*raw),
            From::Meta(_) => None,
        };
        if let Some(raw) = raw {
            *remaining.entry(raw.file_id.0).or_default() += 1;
        }
    }
    let mut deletion =
        delete_source.then(|| IncrementalDeletion::new(repo_id, metadata, &remaining));

    let mut sources: std::collections::HashMap<u32, std::fs::File> =
        std::collections::HashMap::new();
    let mut buffer = vec![0u8; 16 << 20];
    let mut copied = 0u64;
    let mut written_bytes = 0u64;
    for (name, entry) in &entries {
        match entry {
            From::Decoded(decl, raw) => {
                let spool = spool.as_mut().expect("decoded entries imply a spool");
                let bytes = spool.read(name)?;
                writer
                    .begin_tensor(decl, bytes.len() as u64)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                let shard = writer.current_shard_index();
                if let Some(deletion) = &mut deletion {
                    deletion.verify_published(writer, &mut sources)?;
                }
                writer
                    .write(&bytes)
                    .and_then(|_| writer.end_tensor())
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                if let Some(deletion) = &mut deletion {
                    deletion.tensor_written(raw, shard)?;
                }
                written_bytes += bytes.len() as u64;
            }
            From::Meta(bytes) => {
                let path = name
                    .strip_prefix(pie_loader::checkpoint::meta::META_PREFIX)
                    .expect("metadata entries carry the namespace prefix");
                writer
                    .add_meta(path, bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                written_bytes += bytes.len() as u64;
            }
            From::Copy(raw, path) => {
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
                let shard = writer.current_shard_index();
                if let Some(deletion) = &mut deletion {
                    deletion.verify_published(writer, &mut sources)?;
                }
                let handle = match sources.entry(raw.file_id.0) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                        std::fs::File::open(path).with_context(|| format!("cannot open {path}"))?,
                    ),
                };
                handle
                    .seek(SeekFrom::Start(raw.file_offset))
                    .with_context(|| format!("cannot seek in {path}"))?;
                let mut remaining = raw.span_bytes;
                while remaining > 0 {
                    let take = remaining.min(buffer.len() as u64) as usize;
                    handle
                        .read_exact(&mut buffer[..take])
                        .with_context(|| format!("cannot read '{name}' from {path}"))?;
                    writer
                        .write(&buffer[..take])
                        .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                    remaining -= take as u64;
                    copied += take as u64;
                    progress.render(&Progress {
                        read_bytes: decode_read_bytes + copied,
                        total_read_bytes: decode_read_bytes + copy_bytes,
                        finalized: Some(name),
                    });
                }
                writer
                    .end_tensor()
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                if let Some(deletion) = &mut deletion {
                    deletion.tensor_written(raw, shard)?;
                }
                written_bytes += raw.span_bytes;
            }
        }
    }
    progress.finish();
    Ok((written_bytes, deletion))
}
