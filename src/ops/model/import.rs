//! `pie model import`: rewrite a checkpoint as the serving artifact of the
//! SKU that claims it.
//!
//! The SKU's import contract names every plane the engine binds and states,
//! per plane, how it is produced from the source's tensors. This command runs
//! that contract once and writes each plane under the contract's own name, so
//! a boot reads the artifact by name and transforms nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;

use checkpoint::contract::{BiasBy, Expr, ModelContract, ScaleFactor};
use checkpoint::executor::Progress;
use checkpoint::executor::sink::TensorSink;
use checkpoint::file::meta::{SOURCE_ENCODING_KEY, SOURCE_KEY, VERSION_KEY, meta_name};
use checkpoint::file::read::{parse_attributes, parse_metadata};
use checkpoint::file::write::Writer;
use checkpoint::file::{Attributes, Metadata, RawTensor};
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, LoadPlan, StorageInstr, StorageTarget};
use checkpoint::types::{CheckpointFormat, Encoding, FileId, TensorDecl, TensorId, Visibility};
use checkpoint::verify::ContractView;
use runtime::engine::load::Platform;
use worker::weights::CONFIG_OBJECT;

/// Where an `--aux` overlay's tensors land: the overlay's own names behind
/// this prefix, which a model text binds (`qwen_3::model::Recipe::Eagle`).
pub const AUX_PREFIX: &str = "aux.";

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
    /// A draft head to overlay onto this checkpoint: a repo ID, a snapshot
    /// directory, or a single weight file, whose tensors join the artifact
    /// under the `aux.` prefix.
    #[arg(long, value_name = "SOURCE")]
    pub aux: Option<String>,
    /// Overlay the published draft head of this name for this checkpoint:
    /// `--aux` and `--sku` filled in from the catalog's table of published
    /// heads (`models::drafter::PUBLISHED`), so `pie model import
    /// mlx-community/Qwen3.8-27B-4bit --drafter dflash2` is the whole recipe.
    /// A name or a target the table does not know is refused with what it
    /// does know.
    #[arg(long, value_name = "NAME", conflicts_with = "aux")]
    pub drafter: Option<String>,
    /// Import as this catalog row, rather than as the first row whose
    /// contract fits the source. Several rows can only be reached this way —
    /// a family's text row reads every snapshot its vision row does and is
    /// asked first, and a checkpoint carrying a draft head identifies as the
    /// drafting row whether or not one is wanted. The row names are the ones
    /// `pie model list` prints beside each snapshot and `pie model info`
    /// beside each artifact; a name this build does not ship is refused with
    /// the whole catalog, so `--sku '?'` lists them. The named row must read
    /// this checkpoint: nothing here falls back to another.
    #[arg(long, value_name = "NAME")]
    pub sku: Option<String>,
    /// Write the artifact here instead of the model store. A path ending in
    /// `.zt` is the artifact; a directory receives `<name>.zt`.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Report what would be done without doing it.
    #[arg(long)]
    pub dry_run: bool,
    /// Regenerate even when an up-to-date artifact already exists.
    #[arg(long)]
    pub force: bool,
    /// After the artifact is written and every tensor digest verifies, delete
    /// the source weight files it was computed from. Config and tokenizer
    /// files stay.
    #[arg(long, conflicts_with = "consume_source")]
    pub delete_source: bool,
    /// Release each source weight file's bytes as they are read, so the import
    /// needs room for one copy rather than two. Default for a snapshot this
    /// run downloaded.
    #[arg(long, conflicts_with = "keep_source")]
    pub consume_source: bool,
    /// Leave a freshly downloaded snapshot intact.
    #[arg(long)]
    pub keep_source: bool,
}

/// Deletes the weight files a consuming import has been releasing, whether
/// the import finished or gave up: a released file reads back as zeros.
struct Consumed<'a> {
    metadata: &'a Metadata,
}

impl Drop for Consumed<'_> {
    fn drop(&mut self) {
        let _ = remove_source_files(self.metadata);
    }
}

/// The marker a consuming import leaves beside its source from the first
/// released byte until it succeeds. A source that still carries one was
/// half-eaten by a run that died: its weight files keep their sizes and read
/// as zeros where they were released, so a retry — or a re-download, which
/// sees nothing missing — would convert holes into an artifact that loads
/// and answers noise.
const CONSUMING_MARKER: &str = ".pie-consuming";

fn consuming_marker(source: &Path) -> PathBuf {
    if source.is_dir() {
        source.join(CONSUMING_MARKER)
    } else {
        source.parent().map_or_else(
            || PathBuf::from(CONSUMING_MARKER),
            |dir| dir.join(CONSUMING_MARKER),
        )
    }
}

pub fn run(mut args: ImportArgs, global: &bootstrap::GlobalArgs) -> Result<crate::ui::Answer> {
    // `--drafter <name>` is `--aux <head> --sku <row>` looked up for the
    // source, and it is resolved before anything is fetched: a name the table
    // lacks must not cost a download to find out about.
    if let Some(name) = args.drafter.take() {
        let Some(published) = models::drafter::published(&args.source, &name) else {
            let known: Vec<String> = models::drafter::published_for(&args.source)
                .map(|p| format!("`{}` ({})", p.drafter, p.head))
                .collect();
            bail!(
                "--drafter {name}: no published head of that name for {} in this build; {}",
                args.source,
                if known.is_empty() {
                    "this build knows no head for that target — pass the head with `--aux` and \
                     the row with `--sku`".to_string()
                } else {
                    format!("it knows {}", known.join(", "))
                }
            );
        };
        args.aux = Some(published.head.to_string());
        if args.sku.is_none() {
            args.sku = Some(published.sku.to_string());
        }
    }
    // The name is checked against the catalog FIRST, before the source is
    // resolved: a misspelled row must not cost a fourteen-gigabyte download
    // to find out about, and `--sku '?'` is how an operator asks what the rows
    // are called.
    if let Some(name) = args.sku.as_deref() {
        runtime::engine::load::row_named(name).map_err(|why| anyhow!("--sku {name}: {why:#}"))?;
    }
    let mut source = resolve_source(&args.source)?;
    if consuming_marker(&source.path).is_file() {
        bail!(
            "{}: a consuming import of this source died partway (`{CONSUMING_MARKER}` is \
             beside it), so its weight files are holes where they were read and would \
             convert to zeros. Delete the snapshot and fetch it again — a re-download alone \
             does not notice, since the files keep their sizes.",
            crate::ui::short_path(&source.path)
        );
    }
    let mut metadata = parse_metadata(&source.path)
        .map_err(|err| anyhow!("cannot read {}: {err}", source.path.display()))?;
    // An overlay onto an artifact is written into that artifact, and its
    // staging goes beside it — not into a store directory named for a file.
    let onto_artifact = args.aux.is_some()
        && metadata
            .files
            .iter()
            .all(|file| file.format == CheckpointFormat::Zt);
    let out_file = match &args.out {
        Some(out) => artifact_path(out, &source.name),
        None if onto_artifact => source.path.clone(),
        None => store_path(&source.name),
    };
    // The overlay joins `origin`, so an artifact built from the base alone is
    // stale against one built with a head.
    let overlay = match &args.aux {
        Some(spec) => {
            let head = resolve_source(spec)?;
            source.origin = format!("{} +aux {}", source.origin, head.origin);
            Some(stage_overlay(&head, &out_file)?)
        }
        None => None,
    };
    let source = source;

    // **AN OVERLAY ONTO AN ARTIFACT IS IN PLACE.** The base is already pie's
    // own format and may be a ninety-gigabyte file whose source snapshot is
    // gone; only the head's planes are written, appended to the artifact,
    // which is then restamped for the row that reads both.
    if let Some(overlay) = overlay.as_ref().filter(|_| {
        metadata
            .files
            .iter()
            .all(|file| file.format == CheckpointFormat::Zt)
    }) {
        return overlay_onto_artifact(&args, &source, metadata, overlay);
    }

    let clobbered: Vec<&str> = metadata
        .files
        .iter()
        .map(|file| file.path.as_str())
        .filter(|path| same_file(Path::new(path), &out_file))
        .collect();
    if !clobbered.is_empty() {
        bail!(
            "{} is both the source and the destination; an import that overwrote \
             its own input would destroy the weights it was reading. Pass \
             `--out <path>` to write it somewhere else.",
            crate::ui::short_path(&out_file),
        );
    }
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
    // The contract is read for the box this command runs on: a repack is an
    // arrangement of bytes only that box's kernels read.
    let platform = engine_or_refuse()?;
    if out_file.exists() && !args.force {
        if let Some(reason) = staleness(&out_file, platform, &source.origin, args.sku.as_deref()) {
            println!(
                "{}: rebuilding {} ({reason})",
                source.name,
                out_file.display()
            );
        } else {
            let up_to_date = format!(
                "{} is up to date at {}",
                source.name,
                crate::ui::short_path(&out_file)
            );
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

    let mut opened = runtime::engine::load::open_source(&source.path)?;
    if let Some(overlay) = &overlay {
        let head = ztensor::Source::open(&overlay.path)
            .with_context(|| format!("open {}", overlay.path.display()))?;
        opened = ztensor::Source::merge(vec![opened, head])
            .with_context(|| "merge the overlay into the checkpoint's name space")?;
        merge_metadata(&mut metadata, overlay.metadata.clone());
    }
    let metadata = metadata;
    let (sku, contract) = choose_row(
        args.sku.as_deref(),
        &opened,
        &metadata,
        platform,
        &source.path,
    )?;
    drop(opened);
    refuse_a_decode_of_packed_codes(sku, &contract, &metadata)?;
    let attributes = gguf_attributes(&source, &metadata);

    // Compiled once against the whole contract: every shape is checked and
    // every plane's declaration is resolved before a byte moves.
    let landing = checkpoint::plan::compile(&metadata, &contract, decode_target())
        .map_err(|err| anyhow!("{sku}: the contract does not fit this checkpoint: {err}"))?;
    let split = split_contract(&contract, &landing, &metadata)?;
    let read: BTreeSet<&str> = contract
        .tensors
        .iter()
        .flat_map(|tensor| tensor.expr.sources())
        .collect();
    let unread = metadata
        .weights()
        .filter(|tensor| !read.contains(tensor.name.as_str()))
        .count();
    println!(
        "convert: {sku} lands {} plane(s): {} copied through, {} transformed here; \
         {unread} source tensor(s) no plane reads are left out",
        split.copies.len()
            + split
                .decode
                .tensors
                .iter()
                .filter(|t| t.visibility.is_public())
                .count(),
        split.copies.len(),
        split
            .decode
            .tensors
            .iter()
            .filter(|t| t.visibility.is_public())
            .count(),
    );

    // Metadata compiles before any bytes are written: an artifact whose
    // tokenizer would not compile cannot serve.
    let tokenizer = compile_tokenizer(&source, &metadata)?;
    match &tokenizer {
        Some(canonical) => println!(
            "convert: tokenizer compiled to {} ({} KiB)",
            tokenizer::canonical::VERSION,
            canonical.byte_size() / 1024
        ),
        None => println!(
            "convert: no tokenizer beside the weights — the artifact will carry none, \
             and serving it needs one from elsewhere"
        ),
    }
    let config = match carry_config(&source)? {
        Some(bytes) => {
            println!(
                "convert: carrying the checkpoint's config.json ({} bytes) as {CONFIG_OBJECT}",
                bytes.len()
            );
            Some(bytes)
        }
        None => match attributes.as_ref().filter(|a| !a.is_empty()) {
            Some(attributes) => {
                let bytes = attributes.to_json().into_bytes();
                println!(
                    "convert: no config.json beside the weights; carrying the GGUF's own \
                     key-value block ({} bytes) as {CONFIG_OBJECT} instead",
                    bytes.len()
                );
                Some(bytes)
            }
            None => {
                println!("convert: no config.json beside the weights");
                None
            }
        },
    };

    let consume =
        !args.keep_source && !args.delete_source && (args.consume_source || source.fetched);
    let copy_bytes: u64 = split.copies.iter().map(|copy| copy.raw.span_bytes).sum();

    let mut meta: Vec<(String, Vec<u8>)> = Vec::new();
    if let Some(canonical) = &tokenizer {
        for (path, bytes) in canonical.objects() {
            meta.push((meta_name(path), bytes.to_vec()));
        }
    }
    if let Some(config) = &config {
        meta.push((meta_name(CONFIG_OBJECT), config.clone()));
    }

    let started = std::time::Instant::now();
    let mut bar = ProgressLine::new();

    let plan = if split.decode.tensors.is_empty() {
        None
    } else {
        Some(compile_decode(&metadata, &split.decode)?)
    };
    let decode_bytes: u64 = split
        .decode
        .tensors
        .iter()
        .flat_map(|tensor| tensor.expr.sources())
        .collect::<BTreeSet<&str>>()
        .into_iter()
        .filter_map(|name| metadata.tensor_by_name(name))
        .map(|raw| raw.span_bytes)
        .sum();

    let stamp = checkpoint::serving::Stamp::of(&backend_word(platform), sku);
    let trace = runtime::engine::load::trace(sku, platform)?;
    let ranked = runtime::engine::load::sequence(&trace);
    // Off the LANDING, not the decode plan: a quantized plane copied through
    // untouched (a 4-bit `embed`, every U4 projection the lanes read as-is)
    // has its scales and biases attached in the landing alone, and the writer
    // groups every object's planes with its codes, copied or decoded.
    let groups = groups_of(&landing, &trace)?;
    let entries = merge_order(
        plan.as_ref(),
        &split.copies,
        &meta,
        ranked.as_deref(),
        &groups,
    );
    // The decode streams straight into the merge when it produces planes in
    // the order the merge asks for them; otherwise it runs first into a spool.
    let ordered = plan.as_ref().is_none_or(|plan| {
        let asked: Vec<&str> = entries
            .iter()
            .filter(|(_, from)| matches!(from, From::Decoded(_)))
            .map(|(name, _)| *name)
            .collect();
        asked == publish_order(plan)
    });

    // One ledger for every read site, so neither the decode nor the copies
    // release a range the other still wants.
    let base = source.base();
    let ledger = consume.then(|| {
        let mut ledger = match &plan {
            Some(plan) => SourceLedger::of(plan, &base),
            None => SourceLedger::default(),
        };
        for copy in &split.copies {
            ledger.also_read(
                Path::new(copy.path),
                copy.raw.file_offset,
                copy.raw.span_bytes,
            );
        }
        ledger.sort();
        ledger
    });
    if args.dry_run {
        if args.delete_source {
            report_would_delete(&metadata);
        }
        if consume {
            println!(
                "dry run: would then consume {} source weight file(s), freeing {}",
                metadata.files.len(),
                crate::ui::bytes(source_bytes(&metadata))
            );
        }
        println!(
            "dry run: would decode {} and copy {} through",
            crate::ui::bytes(decode_bytes),
            crate::ui::bytes(copy_bytes)
        );
        println!(
            "{}",
            peak_sentence(
                source_bytes(&metadata),
                decode_bytes,
                copy_bytes,
                plan.is_some() && !ordered,
                consume,
            )
        );
        // The name the artifact would be given, not the staging name it is
        // written under: the row is in that filename, and the row is what a
        // dry run behind `--sku` is being asked to confirm.
        return Ok(crate::ui::Answer::noop(format!(
            "dry run: would write {} as `{sku}`",
            crate::ui::short_path(&specialized_path(
                &out_file,
                &source.name,
                &stamp,
                args.out.as_deref()
            ))
        )));
    }

    let mut spool = match &plan {
        Some(plan) if !ordered => {
            let mut spool = Spool::create(&out_file)?;
            let mut execution = checkpoint::executor::Execution::new(plan, &base)
                .streaming()
                .sink(&mut spool);
            if let Some(ledger) = &ledger {
                execution = execution.consuming(ledger);
            }
            execution
                .progress(&mut |progress| {
                    bar.render(&Progress {
                        read_bytes: progress.read_bytes,
                        total_read_bytes: decode_bytes + copy_bytes,
                        finalized: progress.finalized,
                    });
                })
                .run()
                .map_err(|err| anyhow!("decoding failed: {err}"))?;
            Some(spool)
        }
        _ => None,
    };

    let provenance = BTreeMap::from([
        (VERSION_KEY.to_string(), pie_version().to_string()),
        (SOURCE_KEY.to_string(), source.origin.clone()),
        (SOURCE_ENCODING_KEY.to_string(), source_encoding(&metadata)),
    ]);
    let mut writer = Writer::create_serving(&out_file, &provenance, stamp.clone())
        .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;
    for group in &groups {
        writer
            .group(
                group.object.clone(),
                group.planes.iter().cloned(),
                group.tiled,
            )
            .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;
    }
    if consume {
        println!(
            "import: consuming the source as it is read; {} will not survive this run",
            if source.fetched {
                "the snapshot just downloaded"
            } else {
                "the source files"
            }
        );
        let marker = consuming_marker(&source.path);
        std::fs::write(&marker, out_file.to_string_lossy().as_bytes())
            .with_context(|| format!("cannot mark {} as being consumed", marker.display()))?;
    }
    let consumed = consume.then(|| Consumed {
        metadata: &metadata,
    });
    let written_bytes = write_artifact(
        &mut writer,
        &entries,
        plan.as_ref(),
        &base,
        spool.as_mut(),
        &mut bar,
        decode_bytes,
        copy_bytes,
        ledger.as_ref(),
    )?;
    writer
        .finish()
        .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;
    if let Some(spool) = spool {
        spool.remove();
    }
    drop(overlay);
    let out_file = name_the_specialization(out_file, &source.name, &stamp, args.out.as_deref())?;
    if let Err(why) = runtime::engine::load::verify_artifact(&out_file, platform) {
        let removed = std::fs::remove_file(&out_file).is_ok();
        bail!(
            "{}: this import wrote an artifact no boot on this box can load ({why:#}).{}",
            crate::ui::short_path(&out_file),
            if removed {
                " It has been removed."
            } else {
                " It could not be removed; delete it by hand."
            },
        );
    }

    let did = format!(
        "imported {} — {} in {} → {}",
        source.name,
        crate::ui::bytes(written_bytes),
        crate::ui::duration(started.elapsed()),
        crate::ui::short_path(&out_file)
    );
    if args.delete_source {
        delete_source(&source.name, &metadata, &out_file)?;
    }
    if let Some(consumed) = consumed {
        drop(consumed);
        let _ = std::fs::remove_file(consuming_marker(&source.path));
        println!(
            "{}: consumed {} source file(s), {}",
            source.name,
            metadata.files.len(),
            crate::ui::bytes(source_bytes(&metadata))
        );
    }
    let _ = global;
    Ok(crate::ui::Answer::did(did))
}

/// An `--aux` overlay, staged as one `.zt` whose tensors carry
/// [`AUX_PREFIX`], so the base and the head read as one name space. Removed
/// on drop.
/// `pie model import <artifact.zt> --aux <head>`: land a draft head's planes
/// onto an artifact the store already holds, in place.
///
/// The row is identified against the artifact's own names plus the overlay
/// (`Model::import_from_own_with_aux`), so the contract reads every trunk
/// plane from where it already is and every head plane through `aux.`; only
/// the latter are converted and written. They are appended to the artifact
/// through [`Writer::append_serving`], which restamps the file for the
/// drafting row on finish, and the file is renamed for its new SKU. The
/// append is the one non-atomic write in this command: the file's length is
/// held and the file cut back to it on any failure, which restores the
/// artifact byte for byte (its footer sits inside that length).
fn overlay_onto_artifact(
    args: &ImportArgs,
    source: &Source,
    mut metadata: Metadata,
    overlay: &Overlay,
) -> Result<crate::ui::Answer> {
    if args.out.is_some() {
        bail!(
            "an overlay onto an artifact is written in place ({}); `--out` would mean a \
             second copy of the artifact, which this command does not make",
            crate::ui::short_path(&source.path)
        );
    }
    if args.delete_source || args.consume_source {
        bail!("an overlay onto an artifact keeps the artifact; nothing is deleted or consumed");
    }
    let platform = engine_or_refuse()?;
    let base_file = source.path.clone();
    let before = checkpoint::file::serve::stamp_of(&base_file)
        .map_err(|err| anyhow!("cannot read {}: {err}", base_file.display()))?
        .ok_or_else(|| {
            anyhow!(
                "{} carries no serving stamp; an overlay lands on an artifact `pie model \
                 import` wrote",
                crate::ui::short_path(&base_file)
            )
        })?;

    let mut opened = runtime::engine::load::open_source(&base_file)?;
    let head = ztensor::Source::open(&overlay.path)
        .with_context(|| format!("open {}", overlay.path.display()))?;
    opened = ztensor::Source::merge(vec![opened, head])
        .with_context(|| "merge the overlay into the artifact's name space")?;
    merge_metadata(&mut metadata, overlay.metadata.clone());
    let metadata = metadata;
    let (sku, contract) = choose_row(
        args.sku.as_deref(),
        &opened,
        &metadata,
        platform,
        &base_file,
    )?;
    drop(opened);
    if sku == before.sku {
        bail!(
            "{} already serves `{sku}`; the overlay lands on a row that reads the head, and \
             identification chose the row it already was",
            crate::ui::short_path(&base_file)
        );
    }
    refuse_a_decode_of_packed_codes(sku, &contract, &metadata)?;

    // Only the head is written: every plane whose every source is an overlay
    // tensor. A trunk plane the contract reads is where it already is.
    let is_head = |expr: &Expr| {
        let sources = expr.sources();
        !sources.is_empty() && sources.iter().all(|name| name.starts_with(AUX_PREFIX))
    };
    let landing = checkpoint::plan::compile(&metadata, &contract, decode_target())
        .map_err(|err| anyhow!("{sku}: the contract does not fit this artifact: {err}"))?;
    let split = split_contract(&contract, &landing, &metadata)?;
    let copies: Vec<Copy<'_>> = split
        .copies
        .into_iter()
        .filter(|copy| copy.raw.name.starts_with(AUX_PREFIX))
        .collect();
    let decode = ModelContract {
        alignment: split.decode.alignment,
        tensors: split
            .decode
            .tensors
            .iter()
            .filter(|tensor| is_head(&tensor.expr))
            .cloned()
            .collect(),
        groups: Vec::new(),
    };
    let head_planes = copies.len()
        + decode
            .tensors
            .iter()
            .filter(|t| t.visibility.is_public())
            .count();
    let trunk_planes = contract
        .tensors
        .iter()
        .filter(|tensor| !is_head(&tensor.expr))
        .count();
    println!(
        "overlay: {sku} lands {head_planes} head plane(s) onto {} ({} copied through, {} \
         transformed here); {trunk_planes} trunk plane(s) stay where they are",
        crate::ui::short_path(&base_file),
        copies.len(),
        decode
            .tensors
            .iter()
            .filter(|t| t.visibility.is_public())
            .count(),
    );
    if head_planes == 0 {
        bail!("the overlay contributes no plane the row `{sku}` reads");
    }

    let plan = if decode.tensors.is_empty() {
        None
    } else {
        Some(compile_decode(&metadata, &decode)?)
    };
    let decode_bytes: u64 = decode
        .tensors
        .iter()
        .flat_map(|tensor| tensor.expr.sources())
        .collect::<BTreeSet<&str>>()
        .into_iter()
        .filter_map(|name| metadata.tensor_by_name(name))
        .map(|raw| raw.span_bytes)
        .sum();
    let copy_bytes: u64 = copies.iter().map(|copy| copy.raw.span_bytes).sum();
    let stamp = checkpoint::serving::Stamp::of(&backend_word(platform), sku);
    let trace = runtime::engine::load::trace(sku, platform)?;
    let ranked = runtime::engine::load::sequence(&trace);
    // The landing's attachments, as on the import path: a copied-through
    // quantized head plane is grouped with its companions too.
    let groups = groups_of(&landing, &trace)?;
    let entries = merge_order(plan.as_ref(), &copies, &[], ranked.as_deref(), &groups);
    // The decode produces in its own order; the artifact wants the boot's.
    // Where the two differ the decode goes through a spool beside the
    // artifact first (the head is a few gigabytes), as `run` does.
    let ordered = plan.as_ref().is_none_or(|plan| {
        let asked: Vec<&str> = entries
            .iter()
            .filter(|(_, from)| matches!(from, From::Decoded(_)))
            .map(|(name, _)| *name)
            .collect();
        asked == publish_order(plan)
    });
    if args.dry_run {
        return Ok(crate::ui::Answer::noop(format!(
            "dry run: would append {} decoded and {} copied through to {} and restamp it `{sku}`",
            crate::ui::bytes(decode_bytes),
            crate::ui::bytes(copy_bytes),
            crate::ui::short_path(&base_file)
        )));
    }

    let started = std::time::Instant::now();
    let mut bar = ProgressLine::new();
    let base = source.base();
    let mut spool = match &plan {
        Some(plan) if !ordered => {
            let mut spool = Spool::create(&base_file)?;
            checkpoint::executor::Execution::new(plan, &base)
                .streaming()
                .sink(&mut spool)
                .progress(&mut |progress| {
                    bar.render(&Progress {
                        read_bytes: progress.read_bytes,
                        total_read_bytes: decode_bytes + copy_bytes,
                        finalized: progress.finalized,
                    });
                })
                .run()
                .map_err(|err| anyhow!("decoding the head failed: {err}"))?;
            Some(spool)
        }
        _ => None,
    };
    let held = std::fs::metadata(&base_file)
        .with_context(|| format!("stat {}", base_file.display()))?
        .len();
    let provenance = BTreeMap::from([
        (VERSION_KEY.to_string(), pie_version().to_string()),
        (SOURCE_KEY.to_string(), source.origin.clone()),
    ]);
    let restore = |why: anyhow::Error| -> anyhow::Error {
        match std::fs::OpenOptions::new()
            .write(true)
            .open(&base_file)
            .and_then(|file| file.set_len(held))
        {
            Ok(()) => anyhow!(
                "{why:#}. {} was cut back to its {} bytes and is as it was.",
                crate::ui::short_path(&base_file),
                held
            ),
            Err(cut) => anyhow!(
                "{why:#}. {} could NOT be cut back to its {} bytes ({cut}); truncate it \
                 by hand to restore the artifact.",
                crate::ui::short_path(&base_file),
                held
            ),
        }
    };
    let outcome = (|| -> Result<u64> {
        let mut writer = Writer::append_serving(&base_file, &provenance, stamp.clone())
            .map_err(|err| anyhow!("cannot append to the artifact: {err}"))?;
        for group in &groups {
            writer
                .group(
                    group.object.clone(),
                    group.planes.iter().cloned(),
                    group.tiled,
                )
                .map_err(|err| anyhow!("cannot append to the artifact: {err}"))?;
        }
        let written = write_artifact(
            &mut writer,
            &entries,
            plan.as_ref(),
            &base,
            spool.as_mut(),
            &mut bar,
            decode_bytes,
            copy_bytes,
            None,
        )?;
        writer
            .finish()
            .map_err(|err| anyhow!("cannot finish the artifact: {err}"))?;
        Ok(written)
    })();
    if let Some(spool) = spool {
        spool.remove();
    }
    let written_bytes = outcome.map_err(restore)?;

    // The store names an artifact for its SKU; the file follows its stamp.
    let renamed = match base_file
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| checkpoint::serving::Name::parse(name).ok())
    {
        Some(name) => {
            let renamed = base_file
                .parent()
                .unwrap_or(Path::new("."))
                .join(checkpoint::serving::Name::of(&stamp, &name.slug).render());
            std::fs::rename(&base_file, &renamed)
                .with_context(|| format!("cannot rename the artifact to {}", renamed.display()))?;
            renamed
        }
        None => base_file.clone(),
    };
    if let Err(why) = runtime::engine::load::verify_artifact(&renamed, platform) {
        bail!(
            "{}: the overlaid artifact does not load as `{sku}` ({why:#}); cut it back to \
             {held} bytes and rename it for `{}` to restore the artifact",
            crate::ui::short_path(&renamed),
            before.sku
        );
    }
    Ok(crate::ui::Answer::did(format!(
        "overlaid {} — {} in {} → {}",
        overlay.path.display(),
        crate::ui::bytes(written_bytes),
        crate::ui::duration(started.elapsed()),
        crate::ui::short_path(&renamed)
    )))
}

struct Overlay {
    path: PathBuf,
    metadata: Metadata,
}

impl Drop for Overlay {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn stage_overlay(head: &Source, out_file: &Path) -> Result<Overlay> {
    let metadata = parse_metadata(&head.path)
        .map_err(|err| anyhow!("cannot read {}: {err}", head.path.display()))?;
    if let Some(parent) = out_file.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let path = out_file.with_extension("aux.zt");
    let mut writer = Writer::create(&path, &BTreeMap::new())
        .map_err(|err| anyhow!("cannot stage the overlay at {}: {err}", path.display()))?;
    // An object the head's file splits into planes goes back as one object,
    // grouped as that file grouped it, under the aux names.
    let groups = checkpoint::file::read::parse_groups(&head.path)
        .map_err(|err| anyhow!("cannot read {}: {err}", head.path.display()))?;
    for (object, planes) in groups {
        let planes = planes.into_iter().map(|name| format!("{AUX_PREFIX}{name}"));
        writer
            .group(format!("{AUX_PREFIX}{object}"), planes, false)
            .map_err(|err| anyhow!("cannot stage '{object}': {err}"))?;
    }
    let mut tensors: Vec<&RawTensor> = metadata.weights().collect();
    tensors.sort_by_cached_key(|raw| writer.order_key(&format!("{AUX_PREFIX}{}", raw.name)));
    let mut copied = 0u64;
    for (id, raw) in tensors.iter().enumerate() {
        let file = metadata
            .files
            .iter()
            .find(|file| file.id == raw.file_id)
            .ok_or_else(|| anyhow!("'{}' points at a file the overlay lacks", raw.name))?;
        let mut bytes = vec![0u8; usize::try_from(raw.span_bytes)?];
        std::fs::File::open(&file.path)
            .and_then(|f| f.read_exact_at(&mut bytes, raw.file_offset))
            .with_context(|| format!("cannot read '{}' from {}", raw.name, file.path))?;
        let decl = TensorDecl {
            id: TensorId(u32::try_from(id)?),
            name: format!("{AUX_PREFIX}{}", raw.name),
            shape: raw.shape.clone(),
            encoding: raw.encoding.clone(),
            alignment: 1,
            visibility: Visibility::default(),
        };
        writer
            .add_tensor(&decl, &bytes)
            .map_err(|err| anyhow!("cannot stage '{}': {err}", decl.name))?;
        copied += raw.span_bytes;
    }
    writer
        .finish()
        .map_err(|err| anyhow!("cannot stage the overlay: {err}"))?;
    let staged = checkpoint::file::zt::parse(&path)
        .map_err(|err| anyhow!("cannot read back {}: {err}", path.display()))?;
    println!(
        "convert: overlaying {} tensor(s) ({}) from {} under `{AUX_PREFIX}`",
        tensors.len(),
        crate::ui::bytes(copied),
        crate::ui::short_path(&head.path),
    );
    Ok(Overlay {
        path,
        metadata: staged,
    })
}

fn merge_metadata(base: &mut Metadata, extra: Metadata) {
    let files = u32::try_from(base.files.len()).expect("a file count inside u32");
    let tensors = u32::try_from(base.tensors.len()).expect("a tensor count inside u32");
    for mut file in extra.files {
        file.id = FileId(file.id.0 + files);
        base.files.push(file);
    }
    for mut tensor in extra.tensors {
        tensor.id = TensorId(tensor.id.0 + tensors);
        tensor.file_id = FileId(tensor.file_id.0 + files);
        base.tensors.push(tensor);
    }
}

/// One plane the artifact takes byte for byte from the source.
struct Copy<'a> {
    decl: &'a TensorDecl,
    raw: &'a RawTensor,
    path: &'a str,
}

/// The contract's planes, sorted by how they are produced: a bare
/// [`Expr::Src`] is a copy the lanes read; everything else is the decode.
struct Split<'a> {
    copies: Vec<Copy<'a>>,
    decode: ModelContract,
}

fn split_contract<'a>(
    contract: &'a ModelContract,
    landing: &'a LoadPlan,
    metadata: &'a Metadata,
) -> Result<Split<'a>> {
    let produced_for: BTreeSet<&str> = contract
        .tensors
        .iter()
        .flat_map(|tensor| tensor.expr.outputs())
        .collect();
    let mut copies = Vec::new();
    let mut decode = Vec::new();
    for tensor in &contract.tensors {
        let leg = match &tensor.expr {
            Expr::Src(leg)
                if tensor.visibility.is_public()
                    && !produced_for.contains(tensor.name.as_str()) =>
            {
                leg.as_str()
            }
            _ => {
                decode.push(tensor.clone());
                continue;
            }
        };
        let decl = landing
            .tensors
            .iter()
            .find(|decl| decl.name == tensor.name)
            .ok_or_else(|| anyhow!("the plan declares no '{}'", tensor.name))?;
        let raw = metadata
            .tensor_by_name(leg)
            .ok_or_else(|| anyhow!("'{leg}' is in the contract but not the checkpoint"))?;
        let path = metadata
            .files
            .iter()
            .find(|file| file.id == raw.file_id)
            .map(|file| file.path.as_str())
            .ok_or_else(|| anyhow!("'{leg}' points at a file the checkpoint lacks"))?;
        copies.push(Copy { decl, raw, path });
    }
    Ok(Split {
        copies,
        decode: ModelContract {
            alignment: contract.alignment,
            tensors: decode,
            groups: contract.groups.clone(),
        },
    })
}

/// A packed plane is served from its codes, so a chain that lands its values
/// would write bf16 where the artifact must hold codes.
fn refuse_a_decode_of_packed_codes(
    sku: &str,
    contract: &ModelContract,
    metadata: &Metadata,
) -> Result<()> {
    for tensor in &contract.tensors {
        // An internal plane never lands in the artifact: a re-encode takes
        // its stored codes through bf16 on the way to the codes it publishes.
        if !tensor.visibility.is_public() {
            continue;
        }
        if decodes_a_packed_plane(&tensor.expr, metadata, contract) {
            bail!(
                "{sku}: `{}` decodes a plane the checkpoint stores packed, and the artifact \
                 keeps a packed plane as stored",
                tensor.name
            );
        }
    }
    Ok(())
}

/// What the decode is compiled FOR: this host, writing file bytes.
///
/// `BackendKind::Unknown` (the default) because nothing here is staged for a
/// device, and `CONVERT_TILE_MAP_MASK` because the transforms this executor
/// implements are the convert set rather than a kernel table's.
///
/// Named rather than inlined so the test below can compile the same contract
/// the other way against the same target — the two plans are only comparable
/// if the target is one thing.
fn decode_target() -> StorageTarget {
    StorageTarget {
        tile_map_mask: CONVERT_TILE_MAP_MASK,
        max_tile_bytes: 64 << 20,
        ..StorageTarget::default()
    }
}

/// The decode, compiled for streaming execution and checked against its
/// contract before a byte is read.
fn compile_decode(
    metadata: &Metadata,
    contract: &checkpoint::contract::ModelContract,
) -> Result<checkpoint::plan::LoadPlan> {
    let plan = checkpoint::plan::compile_streaming(metadata, contract, decode_target())
        .map_err(|err| anyhow!("cannot compile the decode: {err}"))?;
    if let Err(violations) =
        checkpoint::verify::verify_plan(&plan, Some(&ContractView::of(contract)))
    {
        let listed: Vec<String> = violations.iter().map(ToString::to_string).collect();
        bail!(
            "the compiled decode does not honour its contract:\n  {}",
            listed.join("\n  ")
        );
    }
    Ok(plan)
}

/// What a GGUF source says about itself, or `None` for a source that is not
/// one.
///
/// Read once and lent to both readers. Two of them want it — the rename wants
/// the architecture, and the config wants the whole block — and the file is
/// the same file. The cost is a header parse rather than a scan, but a second
/// one still buys nothing.
fn gguf_attributes(source: &Source, metadata: &Metadata) -> Option<Attributes> {
    if !metadata
        .files
        .iter()
        .any(|file| file.format == CheckpointFormat::Gguf)
    {
        return None;
    }
    parse_attributes(&source.path).ok()
}

fn report_would_delete(metadata: &Metadata) {
    let bytes = source_bytes(metadata);
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
fn delete_source(repo_id: &str, metadata: &Metadata, artifact: &Path) -> Result<()> {
    let verified = checkpoint::file::zt::verify(artifact).map_err(|err| {
        anyhow!(
            "refusing to delete the source: {} does not verify: {err}",
            artifact.display()
        )
    })?;

    remove_source_files(metadata)?;
    println!(
        "{repo_id}: artifact verified ({verified} tensors), deleted {} source file(s), freed {}",
        metadata.files.len(),
        crate::ui::bytes(source_bytes(metadata))
    );
    Ok(())
}

/// The checkpoint files the metadata names, each with the blob its cache
/// symlink points at, plus the shard index that would otherwise keep naming
/// files that no longer exist. Config and tokenizer files are untouched.
fn remove_source_files(metadata: &Metadata) -> Result<()> {
    for file in &metadata.files {
        remove_cache_file(Path::new(&file.path))?;
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
    Ok(())
}

/// What deleting the source weight files gives back.
fn source_bytes(metadata: &Metadata) -> u64 {
    metadata.files.iter().map(|file| file.size_bytes).sum()
}

/// **THE MOST DISK THIS IMPORT WILL HOLD AT ONCE**, which is the question
/// `--dry-run` exists to answer and the one number an operator with a full pool
/// actually needs.
///
/// Up to three files, and only two of them ever stand at full size together.
/// The artifact is what the command is for; the SOURCE stands beside it; and a
/// checkpoint whose decode schedule is not ascending gets a SPOOL as well — the
/// decoded set buffered to disk because the writer needs ascending names and
/// the plan does not produce them in that order. The spool does not add a third
/// copy to the peak: `Spool::read` releases each tensor as it hands it over, so
/// the spool falls exactly as fast as the artifact rises, whatever flags were
/// passed. What it adds is TIME at the peak, and a second full pass over the
/// bytes.
///
/// So the whole of the arithmetic is whether the source is a constant.
/// Without `--consume-source` it is, and the peak is both copies at once.
/// With it the source is a staircase falling as its ranges are read for the
/// last time (see `checkpoint::consume`), stepping down as fast as the spool or
/// the artifact steps up — and the peak is whichever single copy is largest.
///
/// Measured on `mlx-community/DeepSeek-V4-Flash-2bit-DQ` — 89.9 GiB of source,
/// 89.7 GiB decoded, 234 MiB copied, schedule not ascending:
///
/// * without the flag, ~180 GiB — which is why an import started with 45 GiB
///   free filled the disk in forty-five seconds;
/// * with it, ~90 GiB, on a 133 GiB pool.
///
/// The estimate is deliberately not exact. Inward block rounding leaves up to
/// one block per released range allocated, the artifact carries a manifest the
/// weights do not, and `ui::bytes` rounds to a tenth of a GiB anyway. "About"
/// is the honest word and the sentence below uses it.
fn peak_disk(source: u64, decode: u64, copy: u64, consume: bool) -> u64 {
    let artifact = decode.saturating_add(copy);
    if consume {
        // One copy at a time, whichever is bigger. The source is bigger than
        // the artifact for a narrowing conversion and smaller for a widening
        // one, and both happen.
        source.max(artifact)
    } else {
        source.saturating_add(artifact)
    }
}

/// [`peak_disk`] as the line `--dry-run` prints, which is what the operator
/// reads. Separate from the number so a test can assert both.
///
/// `spooled` changes no arithmetic and is still said, because an operator
/// watching `df` through a spooled import sees the pool sit at the peak for
/// twice as long and should know why before deciding something is wrong.
fn peak_sentence(source: u64, decode: u64, copy: u64, spooled: bool, consume: bool) -> String {
    let peak = peak_disk(source, decode, copy, consume);
    let both = peak_disk(source, decode, copy, false);
    let one = peak_disk(source, decode, copy, true);
    let spool = if spooled {
        format!(
            " This checkpoint's schedule is NOT ascending, so the decoded set ({}) is \
             spooled to disk first and the pool sits at its peak for two passes rather \
             than one; the spool itself adds nothing, because it is released as the \
             artifact takes it.",
            crate::ui::bytes(decode),
        )
    } else {
        String::new()
    };
    if consume {
        format!(
            "dry run: `--consume-source` releases each source range as it is read, so the \
             source shrinks while the artifact grows — peak use is about {}, where holding \
             both at once would be {}.{spool}",
            crate::ui::bytes(peak),
            crate::ui::bytes(both),
        )
    } else {
        format!(
            "dry run: peak use is the source ({}) plus the artifact as it grows, about {} \
             in all. `--consume-source` would bring that down to about {}.{spool}",
            crate::ui::bytes(source),
            crate::ui::bytes(both),
            crate::ui::bytes(one),
        )
    }
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

use checkpoint::consume::{SourceLedger, release};

/// The decoded tensors, spooled to disk beside the artifact.
///
/// The executor streams tensors out in schedule order; the artifact writer
/// needs them back in ascending-name order, interleaved with the passthrough
/// copies. The spool is the buffer between the two orders, and it is a file
/// rather than a map so the buffer costs disk instead of memory — the
/// decoded set is the whole model for an F16 checkpoint.
///
/// Every checkpoint seen so far schedules in ascending name order already, so
/// this is the fallback for the ones that will not, and the cost of taking it
/// is measurable: spooling writes the decoded set twice, which is the
/// difference between 13.0 s and 8.9 s on a 12.6 GiB F16 checkpoint.
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

    /// Reads `name` back, and frees it.
    ///
    /// Callers ask in ascending name order — that is what canonical form wants
    /// and what both merges do — so this drops each name from the index as it
    /// hands it over. A second ask for the same tensor, or an ask that goes
    /// backwards, then fails to find it rather than reading bytes the
    /// filesystem has already taken back.
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

        // The spool is written once, read once ascending, and deleted, so a
        // tensor is dead the moment it is handed over. Releasing here is what
        // makes the spool shrink as the artifact grows rather than both
        // standing at full size when the merge ends. `pie model build` is
        // where that matters most -- nothing passes through, so every tensor
        // it writes is spooled first -- and peak filesystem use for
        // `gpt-oss-20b` goes from 40.0 GiB to 27.7 GiB.
        //
        // Dropping the name is what keeps that safe: released bytes read back
        // as zeros, so asking twice has to fail rather than quietly succeed.
        self.index.remove(name);
        release(&self.file, offset, len);
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
    ) -> std::result::Result<(), checkpoint::error::Error> {
        use std::io::Write;
        self.file.write_all(bytes).map_err(|err| {
            checkpoint::error::Error::Checkpoint(format!(
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
/// free of `checkpoint` -- what it needs to draw a bar is two numbers and a
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
        self.bar.draw(
            progress.read_bytes,
            progress.total_read_bytes,
            &self.current,
        );
    }

    pub(crate) fn finish(&mut self) {
        self.bar.finish();
    }
}

/// The distinct encodings a source checkpoint stored its weights in.
///
/// Sorted and comma-separated, so it reads as one fact and compares as one
/// string: `raw:bf16`, `quant:q4_0`, `quant:q4_k,quant:q6_k`. Each part says
/// which KIND it is, for the reason below. Mixed is the normal case for a GGUF —
/// llama.cpp keeps the attention output and the embeddings at a wider scheme
/// than the bulk — and the whole set is kept rather than a "dominant" one,
/// because which tensors were coarse is exactly what a later requantization
/// would compound.
///
/// Metadata objects are excluded: a tokenizer vocabulary is `u8` and saying so
/// would make every artifact claim a `u8` source.
///
/// # Why each part says which kind it is
///
/// The only question anyone asks of this string is "was any of it already
/// quantized" — `pie model build` asks it to warn about rounding twice. That is
/// decided HERE, where the `Encoding` is in hand and the answer is simply which
/// match arm ran.
///
/// It was written without the prefix first, and the reader then carried a list
/// of the raw spellings and treated everything else as quantized. That list is
/// a copy of `DType`'s variants, maintained by hand and by eye: add or rename
/// one and it is silently classified as *quantized*, and every import of a
/// plain checkpoint starts advising the operator about a second rounding that
/// is not happening. Nothing would have failed, which is the problem.
///
/// The scheme *name* is still `Debug`-derived, and that is fine — it is a
/// label, read by operators and never branched on. What must not depend on a
/// `Debug` impl is the classification, and now it does not.
fn source_encoding(metadata: &checkpoint::file::Metadata) -> String {
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for tensor in metadata.weights() {
        seen.insert(match &tensor.encoding {
            // FP8 is stored as a *dtype*, not as a `Quant` scheme -- the
            // checkpoint ships 8-bit floats with a sibling `_scale_inv` -- so
            // reading the variant alone would file an already-rounded
            // checkpoint under `raw:` and tell every later reader it was
            // never quantized. It was rounded once, which is the only thing
            // this string is asked. `build::quantized_source` holds no table
            // by design; this is where the `Encoding` is in hand, so this is
            // where the judgement goes.
            checkpoint::types::Encoding::Raw(dtype)
                if checkpoint::types::is_block_scaled(*dtype) =>
            {
                format!("quant:{}", format!("{dtype:?}").to_lowercase())
            }
            checkpoint::types::Encoding::Raw(dtype) => {
                format!("raw:{}", format!("{dtype:?}").to_lowercase())
            }
            // The variant name lowercased, minus the `Gguf` family prefix:
            // `GgufQ4_0` is the scheme llama.cpp and every model card call
            // `Q4_0`, and this string is read by operators, not by Rust.
            checkpoint::types::Encoding::Quant(spec) => {
                let name = format!("{:?}", spec.scheme);
                format!(
                    "quant:{}",
                    name.strip_prefix("Gguf").unwrap_or(&name).to_lowercase()
                )
            }
        });
    }
    seen.into_iter().collect::<Vec<_>>().join(",")
}

/// What `convert` was pointed at, once the pointing is resolved.
pub(crate) struct Source {
    /// The path the loader reads — a snapshot directory or a single file.
    pub(crate) path: PathBuf,
    /// The artifact's name in the store, without the `.zt` suffix.
    pub(crate) name: String,
    /// Where the bytes came from, recorded in the artifact's provenance.
    pub(crate) origin: String,
    /// Whether this run is what put the bytes on disk.
    ///
    /// The question `--consume-source` turns on: a snapshot this run just
    /// downloaded is one the user never had before and can get again, so
    /// spending it on the artifact takes nothing away. A path that was already
    /// there is the user's, and is left alone unless they say otherwise.
    pub(crate) fetched: bool,
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
            // A store archive is named for its directory, not its file. Every
            // one of them is `archive.zt`, so the stem alone would call every
            // model in the store `archive` — and `pie model build` would write
            // its output under `models/archive/runtime/`, one shared directory
            // for every model on the machine.
            store_archive_name(path).unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("model")
                    .to_string()
            })
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
            fetched: false,
        });
    }

    let (snapshot, fetched) = resolve_snapshot(source)?;
    Ok(Source {
        path: snapshot,
        name: store_name(source),
        origin: source.to_string(),
        fetched,
    })
}

/// The snapshot directory of a downloaded repo: `models--org--name/snapshots/`
/// holds one directory per revision; like the rest of `pie model`, the first
/// one present is the one in use.
///
/// The flag says whether this call is what fetched it, which decides whether
/// the import may consume the snapshot on its way through. Absence of the
/// `snapshots` directory is the test, so a repo that was already in the cache
/// -- however it got there -- counts as the user's.
fn resolve_snapshot(repo_id: &str) -> Result<(PathBuf, bool)> {
    let repo_dir = crate::local::hf::resolve_cache_dir()
        .join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let fetched = !snapshots.exists();
    if fetched {
        // Fetched here rather than by a separate `download` command. Whether a
        // source needs the network is a property of that source, not a
        // different operation -- and a `download` that stopped at the snapshot
        // left the user one undiscoverable step short of a servable model,
        // which is why it converted too. Two commands doing fetch-and-convert
        // and convert is one command with an argument.
        crate::ops::model::fetch_snapshot(repo_id)?;
    }
    let entries = std::fs::read_dir(&snapshots)
        .map_err(|_| anyhow!("{repo_id} is neither a path nor a model any registry has"))?;
    let snapshot = entries
        .filter_map(|entry| entry.ok())
        .find(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|entry| entry.path());
    match snapshot {
        Some(path) => Ok((path, fetched)),
        None => bail!("{repo_id} has no snapshot under {}", snapshots.display()),
    }
}

/// A repo ID as one filesystem name: `qwen/qwen3-0.6b` → `qwen--qwen3-0.6b`.
///
/// The store gives each model a directory, so the separator has to survive as
/// something legal in a single path component. `--` rather than a single `-`
/// because model names contain single hyphens freely and the mapping has to
/// stay reversible.
fn store_name(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

/// Whether two paths name the same file on disk.
///
/// Canonicalized, so a symlink or a `..` cannot spell one file two ways. A
/// path that does not exist canonicalizes to itself, which is the right answer
/// here: a destination that is not there yet cannot be the source.
fn same_file(a: &Path, b: &Path) -> bool {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    canon(a) == canon(b)
}

/// The model name a path names, when it is a store archive.///
/// `.../models/<name>/archive.zt` → `<name>`. Decided by the filename and the
/// parent, not by whether the path is under `$PIE_HOME`: a store copied
/// somewhere else is still a store, and the directory is still what says which
/// model this is.
fn store_archive_name(path: &Path) -> Option<String> {
    if path.file_name()? != crate::local::store::ARCHIVE_FILE {
        return None;
    }
    Some(path.parent()?.file_name()?.to_str()?.to_string())
}

/// `$PIE_HOME/models/<name>/archive.zt` — the general-form artifact.
///
/// One layer of the store, not the whole of it: builds for particular targets
/// land under `<name>/runtime/` and are derived from this. See
/// [`crate::local::store`].
pub(crate) fn store_path(name: &str) -> PathBuf {
    crate::local::store::archive_path(name)
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
/// The convention the worker already uses (`crates/worker/src/translate.rs`): `tokenizer.json`,
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

fn backend_word(platform: Platform) -> String {
    format!("{platform:?}").to_lowercase()
}

/// What [`name_the_specialization`] would name this artifact — the same
/// judgement without the rename, so `--dry-run` can print the filename the
/// import would produce (the row is IN that filename) rather than the
/// staging name it writes under.
fn specialized_path(
    written: &Path,
    slug: &str,
    stamp: &checkpoint::serving::Stamp,
    out: Option<&Path>,
) -> PathBuf {
    if out.is_some_and(|out| {
        out.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zt"))
    }) {
        return written.to_path_buf();
    }
    match written.parent() {
        Some(directory) => directory.join(checkpoint::serving::Name::of(stamp, slug).render()),
        None => written.to_path_buf(),
    }
}

/// `<slug>.<sku>.<backend>.zt`, so one model at two recipes or for two
/// shells can share a directory. An `--out` that names a file keeps the
/// operator's name.
fn name_the_specialization(
    written: PathBuf,
    slug: &str,
    stamp: &checkpoint::serving::Stamp,
    out: Option<&Path>,
) -> Result<PathBuf> {
    let renamed = specialized_path(&written, slug, stamp, out);
    if renamed == written {
        return Ok(written);
    }
    std::fs::rename(&written, &renamed)
        .map_err(|why| anyhow!("cannot name the artifact {}: {why}", renamed.display()))?;
    Ok(renamed)
}

/// **WHICH ROW THIS IMPORT CONVERTS FOR**: the one `--sku` named, or the
/// first whose contract fits.
///
/// Without the flag this is what it has always been — the identification
/// order, first fits wins. With it the name decides, and a name that does not
/// fit is a refusal rather than a fallback: the row is stamped into the
/// artifact and printed in its filename, so an import that silently converted
/// for a neighbouring row would hand back a file the operator did not ask
/// for and would have no reason to re-read.
fn choose_row(
    named: Option<&str>,
    opened: &ztensor::Source,
    metadata: &Metadata,
    platform: Platform,
    checkpoint: &Path,
) -> Result<(&'static str, ModelContract)> {
    let Some(name) = named else {
        return runtime::engine::load::conversion_contract(opened, metadata, platform)
            .ok_or_else(|| refuse_a_source_no_sku_in_this_build_claims(checkpoint));
    };
    runtime::engine::load::conversion_contract_named(opened, metadata, platform, name).map_err(
        |why| {
            anyhow!(
                "--sku {name}: {why:#}\n\
                 This import converts for the row named and for no other; drop `--sku` to \
                 convert {} for the first row whose contract fits it.",
                crate::ui::short_path(checkpoint),
            )
        },
    )
}

fn refuse_a_source_no_sku_in_this_build_claims(checkpoint: &Path) -> anyhow::Error {
    anyhow!(
        "{}: no SKU this build ships claims this checkpoint, so nothing here can say \
         what its planes are. The import performs a SKU's whole landing, so the artifact \
         would hold the source's own tensors under the source's own names — a file that \
         converts, verifies and opens, and that no boot on any box with this catalog can \
         load. `pie model list` prints what a checkpoint identifies as.",
        crate::ui::short_path(checkpoint),
    )
}

/// The setup this binary converts for: the engine it linked.
fn engine_or_refuse() -> Result<Platform> {
    runtime::engine::load::this_box().ok_or_else(|| {
        anyhow!(
            "this pie binary carries no engine, so it cannot say which setup to convert \
             for — and an artifact IS a setup: the backend is stamped in it and printed \
             in its name, and a boot refuses when it disagrees. Rebuild with --features \
             set to cuda or metal."
        )
    })
}

/// Does this chain land, anywhere in it, a value the checkpoint holds packed?
/// Asked of the operand's encoding, which the node does not carry: a
/// `Cast { to: Raw }` over bf16 is a narrowing and over mxfp4 a decode.
/// Untypeable answers as a decode.
fn decodes_a_packed_plane(expr: &Expr, checkpoint: &Metadata, contract: &ModelContract) -> bool {
    let mut decodes = false;
    expr.visit(&mut |node| {
        let operand = match node {
            Expr::Cast {
                src,
                to: Encoding::Raw(_),
            } => src,
            Expr::Scale {
                src,
                factor: ScaleFactor::PerBlock { .. },
            } => src,
            Expr::Bias {
                src,
                by: BiasBy::PerBlock { .. },
            } => src,
            _ => return,
        };
        if !matches!(
            yields(operand, checkpoint, contract),
            Some(Encoding::Raw(_))
        ) {
            decodes = true;
        }
    });
    decodes
}

/// The representation an expression's value is in, or `None` where the
/// chain does not say.
fn yields<'a>(
    expr: &'a Expr,
    checkpoint: &'a Metadata,
    contract: &'a ModelContract,
) -> Option<&'a Encoding> {
    match expr {
        Expr::Src(name) => checkpoint.tensor_by_name(name).map(|held| &held.encoding),
        // A name declared earlier in the same contract says its own
        // encoding. Reading it is what keeps a two-step chain — apply a
        // transform at the stored width, then adapt — from looking like a
        // decode of packed codes when the plane was never packed.
        Expr::Out(name) => contract
            .tensors
            .iter()
            .find(|declared| declared.name == *name)
            .map(|declared| &declared.encoding),
        Expr::Fill { ty, .. } => Some(&ty.encoding),
        Expr::Cast { to, .. } => Some(to),
        Expr::Transmute { to, .. } | Expr::Repack { to, .. } => Some(&to.encoding),
        Expr::Slice { src, .. }
        | Expr::Stride { src, .. }
        | Expr::Gather { src, .. }
        | Expr::Shard { src, .. }
        | Expr::Select { src, .. }
        | Expr::Scale { src, .. }
        | Expr::Unary { src, .. }
        | Expr::Bias { src, .. } => yields(src, checkpoint, contract),
        Expr::Concat { parts, .. } => parts
            .first()
            .and_then(|leg| yields(leg, checkpoint, contract)),
        Expr::SrcIndexed(_) => None,
    }
}

/// Carries the source's `config.json` into the artifact, verbatim.
pub(crate) fn carry_config(source: &Source) -> Result<Option<Vec<u8>>> {
    if source.path.is_file() {
        return Ok(None);
    }
    let path = source.path.join("config.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_slice::<serde_json::Value>(&raw)
        .map_err(|err| anyhow!("cannot parse {}: {err}", path.display()))?;
    Ok(Some(raw))
}

/// Compiles the source's tokenizer into its canonical form, if it has one.
///
/// Discovery follows the convention the worker already uses
/// (`crates/worker/src/translate.rs`):
/// `tokenizer.json`, else `tiktoken.model`, beside the weights — and failing
/// both, the checkpoint's own tables, which is where a GGUF keeps its
/// tokenizer. `Ok(None)` means every one of those was absent.
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
    metadata: &Metadata,
) -> Result<Option<tokenizer::canonical::CanonicalTokenizer>> {
    let Some(path) = tokenizer_path(source) else {
        return gguf_tokenizer(source, metadata);
    };

    let tokenizer = tokenizer::Tokenizer::from_file(&path).map_err(|err| {
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

/// The tokenizer a GGUF carries inside itself.
fn gguf_tokenizer(
    source: &Source,
    metadata: &Metadata,
) -> Result<Option<tokenizer::canonical::CanonicalTokenizer>> {
    if !metadata
        .files
        .iter()
        .any(|file| file.format == CheckpointFormat::Gguf)
    {
        return Ok(None);
    }
    let tables = checkpoint::file::read::parse_tokenizer(&source.path)?;
    if tables.is_empty() {
        return Ok(None);
    }
    let compiled = tokenizer::loader::gguf::from_tables(&tokenizer::loader::gguf::Tables {
        model: &tables.model,
        pre: tables.pre.as_deref(),
        tokens: &tables.tokens,
        token_types: &tables.token_types,
        merges: &tables.merges,
    })
    .and_then(|tokenizer| tokenizer.to_canonical());
    match compiled {
        Ok(canonical) => Ok(Some(canonical)),
        Err(why) => {
            println!("convert: WARNING - this GGUF's own tokenizer does not compile: {why:#}");
            Ok(None)
        }
    }
}

/// Why an existing artifact needs rebuilding, or `None` if it is current:
/// the same stamp a load checks it against (backend, SKU, layout revision),
/// and the source it was written from.
///
/// `asked` is `--sku`, and an artifact serving another row is stale however
/// current it is for its own: the operator named a row, and "up to date" for
/// a different one would leave them with the file they were trying to
/// replace.
fn staleness(
    artifact: &Path,
    platform: Platform,
    source: &str,
    asked: Option<&str>,
) -> Option<String> {
    let stamp = match checkpoint::file::serve::stamp_of(artifact) {
        Ok(Some(stamp)) => stamp,
        Ok(None) => return Some("it carries no serving stamp".to_string()),
        Err(err) => return Some(format!("its serving stamp does not read back: {err}")),
    };
    if let Some(asked) = asked.filter(|asked| *asked != stamp.sku) {
        return Some(format!(
            "it serves `{}` and `--sku {asked}` was asked for",
            stamp.sku
        ));
    }
    if runtime::engine::load::trace(&stamp.sku, platform).is_err() {
        return Some(format!("this build ships no SKU named `{}`", stamp.sku));
    }
    let wanted = checkpoint::serving::Stamp::of(&backend_word(platform), &stamp.sku);
    if let Err(mismatch) = stamp.check(&wanted) {
        return Some(mismatch.to_string());
    }
    let attributes = match checkpoint::file::zt::read_attributes(artifact) {
        Ok(attributes) => attributes,
        Err(err) => return Some(format!("cannot read its provenance: {err}")),
    };
    match attributes.get(SOURCE_KEY) {
        None => Some("it records no source".to_string()),
        Some(recorded) if recorded != source => {
            Some(format!("the source changed: {recorded} → {source}"))
        }
        Some(_) => None,
    }
}

/// Where one entry of the artifact's merge gets its bytes.
enum From<'a> {
    Decoded(&'a TensorDecl),
    Copy(&'a Copy<'a>),
    Meta(&'a [u8]),
}

/// One quantized weight's planes as the artifact stores them: one object,
/// its codes then its scales then its biases, read off the plan's own
/// attachments (never a name suffix).
struct Group {
    object: String,
    planes: Vec<String>,
    tiled: bool,
}

fn groups_of(plan: &LoadPlan, trace: &runtime::engine::load::Trace) -> Result<Vec<Group>> {
    // Declared tensors sit at their own index; a companion an encode
    // GENERATES (a raw source landed as a quantized bank) is numbered past
    // them and found by id.
    let name_of = |id: TensorId| {
        plan.tensors
            .get(id.0 as usize)
            .filter(|decl| decl.id == id)
            .or_else(|| plan.tensors.iter().find(|decl| decl.id == id))
            .map(|decl| decl.name.clone())
            .ok_or_else(|| {
                anyhow!(
                    "the plan attaches tensor {} and declares no such tensor",
                    id.0
                )
            })
    };
    let mut groups = Vec::with_capacity(plan.attachments.len());
    for attachment in &plan.attachments {
        let object = name_of(attachment.tensor)?;
        let mut planes = vec![object.clone(), name_of(attachment.scale_tensor)?];
        if let Some(zero) = attachment.zero_point_tensor {
            planes.push(name_of(zero)?);
        }
        let tiled = trace.params.iter().any(|param| {
            param.name == object && param.dtype == checkpoint::types::DType::U4g64tiled
        });
        groups.push(Group {
            object,
            planes,
            tiled,
        });
    }
    Ok(groups)
}

/// The order the executor hands planes to the sink: the plan's SCHEDULE, not
/// its declarations.
///
/// [`Walk`] publishes at each `Finalize`, and a plan's `Finalize` order is not
/// its `tensors` order: a quantized bank's companions are finalized where the
/// encode makes them, so a bank lands as `w.scales`, `w.biases`, then `w`,
/// while the declarations read `w`, `w.scales`, `w.biases`. Reading the
/// declarations instead is what let a decode that does not produce in the
/// artifact's order be handed to [`Decoded::Streamed`], which then refuses at
/// the first bank.
///
/// [`Walk`]: checkpoint::executor
fn publish_order(plan: &LoadPlan) -> Vec<&str> {
    plan.schedule
        .iter()
        // Ids are dense by construction; the guard means a plan where they
        // are not falls to the spool rather than to a wrong order.
        .filter_map(|id| match plan.instrs.get(id.0 as usize) {
            Some(StorageInstr::Finalize { id: at, name, .. }) if at == id => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

/// The order the artifact's objects are written in: the shell's ranking when
/// there is one (hottest planes first, one forward walk), names otherwise.
/// A group's planes follow its codes in canonical order whatever the
/// ranking says, since they are one object.
fn merge_order<'a>(
    plan: Option<&'a LoadPlan>,
    copies: &'a [Copy<'a>],
    meta: &'a [(String, Vec<u8>)],
    ranked: Option<&[String]>,
    groups: &[Group],
) -> Vec<(&'a str, From<'a>)> {
    let mut entries: Vec<(&'a str, From<'a>)> = Vec::new();
    if let Some(plan) = plan {
        for decl in plan
            .tensors
            .iter()
            .filter(|decl| decl.visibility.is_public())
        {
            entries.push((&decl.name, From::Decoded(decl)));
        }
    }
    for copy in copies {
        entries.push((&copy.decl.name, From::Copy(copy)));
    }
    for (name, bytes) in meta {
        entries.push((name.as_str(), From::Meta(bytes)));
    }
    let rank: BTreeMap<&str, usize> = ranked
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(at, name)| (name.as_str(), at))
        .collect();
    let member: BTreeMap<&str, (&str, usize)> = groups
        .iter()
        .flat_map(|group| {
            group
                .planes
                .iter()
                .enumerate()
                .map(|(at, plane)| (plane.as_str(), (group.object.as_str(), at)))
        })
        .collect();
    entries.sort_by_key(|(name, _)| {
        let (head, at) = member.get(name).copied().unwrap_or((name, 0));
        (rank.get(head).copied().unwrap_or(usize::MAX), head, at)
    });
    entries
}

/// One pass over `entries`. The writer is sequential; the copies are read by
/// a few lanes ahead of it, and the decode either streams into the merge or
/// was spooled first.
#[allow(clippy::too_many_arguments)]
fn write_artifact<'a>(
    writer: &mut Writer,
    entries: &[(&'a str, From<'a>)],
    plan: Option<&'a LoadPlan>,
    base: &Path,
    spool: Option<&'a mut Spool>,
    progress: &mut ProgressLine,
    decode_bytes: u64,
    copy_bytes: u64,
    consume: Option<&SourceLedger>,
) -> Result<u64> {
    let chunks = plan_chunks(entries);
    let lane_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
        .min(chunks.len().max(1));
    let decode_read = std::sync::atomic::AtomicU64::new(match spool {
        Some(_) => decode_bytes,
        None => 0,
    });
    std::thread::scope(|scope| {
        let mut decoded = match (plan, spool) {
            (Some(plan), None) => {
                let (to, from) = std::sync::mpsc::sync_channel(1);
                let decode_read = &decode_read;
                scope.spawn(move || {
                    decode_into(plan, base, &to, consume, &|read, _total| {
                        decode_read.store(read, std::sync::atomic::Ordering::Relaxed);
                    })
                });
                let asked = entries
                    .iter()
                    .filter(|(_, from)| matches!(from, From::Decoded(_)))
                    .map(|(name, _)| (*name).to_string())
                    .collect();
                Decoded::Streamed { from, asked }
            }
            (_, Some(spool)) => Decoded::Spooled(spool),
            (None, None) => Decoded::Nothing,
        };
        let mut lanes = Vec::with_capacity(lane_count);
        for lane in 0..lane_count {
            let (filled, take_filled) = std::sync::mpsc::sync_channel(DEPTH);
            let (recycle, take_recycled) = std::sync::mpsc::sync_channel(DEPTH);
            for _ in 0..DEPTH {
                let _ = recycle.send(vec![0u8; CHUNK as usize]);
            }
            let chunks = &chunks;
            scope.spawn(move || {
                read_lane(chunks, lane, lane_count, consume, &filled, &take_recycled)
            });
            lanes.push((take_filled, recycle));
        }
        let outcome = merge_entries(
            writer,
            entries,
            &chunks,
            &mut decoded,
            &lanes,
            progress,
            &decode_read,
            decode_bytes,
            copy_bytes,
        );
        // Closing the merge's ends wakes every parked reader.
        drop(lanes);
        drop(decoded);
        outcome
    })
}

/// How much one read moves.
const CHUNK: u64 = 16 << 20;

/// Reads each lane keeps outstanding ahead of the writer.
const DEPTH: usize = 4;

type Lane = (
    std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>,
    std::sync::mpsc::SyncSender<Vec<u8>>,
);

#[allow(clippy::too_many_arguments)]
fn merge_entries(
    writer: &mut Writer,
    entries: &[(&str, From<'_>)],
    chunks: &[Chunk<'_>],
    decoded: &mut Decoded<'_>,
    lanes: &[Lane],
    progress: &mut ProgressLine,
    decode_read: &std::sync::atomic::AtomicU64,
    decode_bytes: u64,
    copy_bytes: u64,
) -> Result<u64> {
    let mut copied = 0u64;
    let mut written_bytes = 0u64;
    let mut next = 0usize;
    for (name, entry) in entries {
        match entry {
            From::Decoded(decl) => {
                let bytes = decoded.take(name)?;
                writer
                    .add_tensor(decl, &bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                written_bytes += bytes.len() as u64;
                progress.render(&Progress {
                    read_bytes: decode_read.load(std::sync::atomic::Ordering::Relaxed) + copied,
                    total_read_bytes: decode_bytes + copy_bytes,
                    finalized: Some(name),
                });
            }
            From::Meta(bytes) => {
                let path = name
                    .strip_prefix(checkpoint::file::meta::META_PREFIX)
                    .expect("metadata entries carry the namespace prefix");
                writer
                    .add_meta(path, bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                written_bytes += bytes.len() as u64;
            }
            From::Copy(copy) => {
                writer
                    .begin_tensor(copy.decl, copy.raw.span_bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                let mut remaining = copy.raw.span_bytes;
                while remaining > 0 {
                    let (filled, recycle) = &lanes[next % lanes.len()];
                    let buffer = filled
                        .recv()
                        .map_err(|_| anyhow!("a reader stopped before '{name}'"))?
                        .with_context(|| format!("cannot read '{name}' from {}", copy.path))?;
                    let take = chunks[next].len;
                    next += 1;
                    writer
                        .write(&buffer[..take])
                        .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                    let _ = recycle.send(buffer);
                    remaining -= take as u64;
                    copied += take as u64;
                    progress.render(&Progress {
                        read_bytes: decode_read.load(std::sync::atomic::Ordering::Relaxed) + copied,
                        total_read_bytes: decode_bytes + copy_bytes,
                        finalized: Some(name),
                    });
                }
                writer
                    .end_tensor()
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                written_bytes += copy.raw.span_bytes;
            }
        }
    }
    progress.finish();
    Ok(written_bytes)
}

/// One contiguous read; never spans two tensors.
struct Chunk<'a> {
    path: &'a str,
    offset: u64,
    len: usize,
}

fn plan_chunks<'a>(entries: &[(&str, From<'a>)]) -> Vec<Chunk<'a>> {
    let mut chunks = Vec::new();
    for (_, entry) in entries {
        let From::Copy(copy) = entry else {
            continue;
        };
        let mut offset = copy.raw.file_offset;
        let mut remaining = copy.raw.span_bytes;
        while remaining > 0 {
            let len = remaining.min(CHUNK);
            chunks.push(Chunk {
                path: copy.path,
                offset,
                len: len as usize,
            });
            offset += len;
            remaining -= len;
        }
    }
    chunks
}

/// Read every `lane_count`-th chunk starting at `lane`, recycling one bounded
/// set of buffers with the writer.
///
/// Returning early is not an error path of its own: both channels are closed
/// by the writer when it is done or has given up, and either `recv` failing or
/// `send` failing means exactly that.
fn read_lane(
    chunks: &[Chunk<'_>],
    lane: usize,
    lane_count: usize,
    consume: Option<&SourceLedger>,
    filled: &std::sync::mpsc::SyncSender<std::io::Result<Vec<u8>>>,
    recycled: &std::sync::mpsc::Receiver<Vec<u8>>,
) {
    let mut open: std::collections::HashMap<&str, std::fs::File> = std::collections::HashMap::new();
    for chunk in chunks.iter().skip(lane).step_by(lane_count) {
        let Ok(mut buffer) = recycled.recv() else {
            return;
        };
        let read = read_chunk(&mut open, chunk, &mut buffer, consume);
        if filled.send(read.map(|()| buffer)).is_err() {
            return;
        }
    }
}

/// Fill `buffer` from the chunk's file. Under `consume` the range is
/// released once the ledger says nothing else reads it.
fn read_chunk<'a>(
    open: &mut std::collections::HashMap<&'a str, std::fs::File>,
    chunk: &Chunk<'a>,
    buffer: &mut [u8],
    consume: Option<&SourceLedger>,
) -> std::io::Result<()> {
    let file = match open.entry(chunk.path) {
        std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
        std::collections::hash_map::Entry::Vacant(slot) => slot.insert(
            std::fs::OpenOptions::new()
                .read(true)
                .write(consume.is_some())
                .open(chunk.path)?,
        ),
    };
    file.read_exact_at(&mut buffer[..chunk.len], chunk.offset)?;
    if let Some(ledger) = consume
        && ledger.last_read(Path::new(chunk.path), chunk.offset, chunk.len as u64)
    {
        release(file, chunk.offset, chunk.len as u64);
    }
    Ok(())
}

/// Where the merge gets a decoded tensor's bytes.
enum Decoded<'a> {
    /// The executor is running now, producing planes in schedule order;
    /// `asked` is the set the merge will request, in that same order.
    Streamed {
        from: std::sync::mpsc::Receiver<std::result::Result<(String, Vec<u8>), String>>,
        asked: BTreeSet<String>,
    },
    Spooled(&'a mut Spool),
    Nothing,
}

impl Decoded<'_> {
    fn take(&mut self, name: &str) -> Result<Vec<u8>> {
        match self {
            Self::Streamed { from, asked } => loop {
                let (produced, bytes) = from
                    .recv()
                    .map_err(|_| anyhow!("the decode stopped before '{name}'"))?
                    .map_err(|err| anyhow!("decoding failed: {err}"))?;
                if produced == name {
                    break Ok(bytes);
                }
                if asked.contains(&produced) {
                    break Err(anyhow!(
                        "the decode produced '{produced}' where the artifact wants '{name}', \
                         so its schedule is not the artifact's order"
                    ));
                }
            },
            Self::Spooled(spool) => spool.read(name),
            Self::Nothing => bail!("'{name}' decodes, but no decode ran"),
        }
    }
}

/// Run the decode, handing each tensor straight to the merge; errors travel
/// down the channel.
fn decode_into(
    plan: &checkpoint::plan::LoadPlan,
    base: &Path,
    to: &std::sync::mpsc::SyncSender<std::result::Result<(String, Vec<u8>), String>>,
    consume: Option<&SourceLedger>,
    watch: &(dyn Fn(u64, u64) + Sync),
) {
    let mut sink = Handoff { to };
    let mut execution = checkpoint::executor::Execution::new(plan, base)
        .streaming()
        .sink(&mut sink);
    if let Some(ledger) = consume {
        execution = execution.consuming(ledger);
    }
    let outcome = execution
        .progress(&mut |progress| watch(progress.read_bytes, progress.total_read_bytes))
        .run();
    if let Err(err) = outcome {
        let _ = to.send(Err(err.to_string()));
    }
}

/// The sink that makes the decode a producer instead of a phase.
struct Handoff<'a> {
    to: &'a std::sync::mpsc::SyncSender<std::result::Result<(String, Vec<u8>), String>>,
}

impl TensorSink for Handoff<'_> {
    fn publish(
        &mut self,
        name: &str,
        bytes: &[u8],
    ) -> std::result::Result<(), checkpoint::error::Error> {
        self.to
            .send(Ok((name.to_string(), bytes.to_vec())))
            .map_err(|_| {
                checkpoint::error::Error::Checkpoint(format!(
                    "the artifact writer stopped before '{name}'"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **THE STREAMING DECODE IS GATED ON THE SCHEDULE, NOT THE DECLARATIONS.**
    ///
    /// A plan finalizes a quantized bank's companions where the encode makes
    /// them, so the sink sees `w.scales`, `w.biases`, `w` while the plan's
    /// `tensors` read `w`, `w.scales`, `w.biases` — the order the artifact
    /// wants. Reading the declarations made the two look equal, and the merge
    /// took the streaming path only to refuse at the first bank. What the
    /// merge has to compare against is what the executor publishes.
    #[test]
    fn a_banks_planes_publish_in_schedule_order_not_declaration_order() {
        use checkpoint::types::{BufferId, DType, InstrId};

        let decl = |at: u32, name: &str| TensorDecl {
            id: TensorId(at),
            name: name.to_string(),
            shape: vec![4, 4],
            encoding: Encoding::Raw(DType::Bf16),
            alignment: 64,
            visibility: Visibility::Public,
        };
        let finalize = |at: u32, name: &str| StorageInstr::Finalize {
            id: InstrId(at),
            tensor: BufferId(at),
            name: name.to_string(),
        };
        // The declarations are the artifact's order; the schedule is the
        // encode's.
        let plan = LoadPlan {
            target: decode_target(),
            passes: Vec::new(),
            files: Vec::new(),
            sources: Vec::new(),
            tensors: vec![decl(0, "w"), decl(1, "w.scales"), decl(2, "w.biases")],
            buffers: Vec::new(),
            instrs: vec![
                finalize(0, "w.scales"),
                finalize(1, "w.biases"),
                finalize(2, "w"),
            ],
            schedule: vec![InstrId(0), InstrId(1), InstrId(2)],
            memory: checkpoint::plan::MemoryPlan::default(),
            attachments: Vec::new(),
            groups: Vec::new(),
        };

        assert_eq!(publish_order(&plan), ["w.scales", "w.biases", "w"]);
        let declared: Vec<&str> = plan.tensors.iter().map(|d| d.name.as_str()).collect();
        assert_ne!(
            publish_order(&plan),
            declared,
            "the declarations are exactly what this gate must NOT be read off"
        );
    }

    /// **THE ARITHMETIC THE DRY RUN EXISTS TO STATE**, on the checkpoint it was
    /// measured against: `mlx-community/DeepSeek-V4-Flash-2bit-DQ`, 89.9 GiB
    /// over nineteen shards, 99.7% of which DECODES, with a schedule that is
    /// not ascending.
    ///
    /// The before is the number that stopped a lane: ~180 GiB of peak for a
    /// 90 GiB model on a 133 GiB pool, because the source stood at full size
    /// through a pass that wrote an 89.7 GiB spool and then through the pass
    /// that turned the spool into the artifact. The after is one checkpoint,
    /// because the decode now releases the source as it reads it and
    /// `Spool::read` already released the spool as the artifact took it — the
    /// staircases trade one for one.
    ///
    /// Asserted as the SENTENCE and not only the number: the sentence is what
    /// the operator reads before deciding to start a multi-hour import, and a
    /// peak that is right in a helper and wrong in the line printed is the
    /// failure this whole dry run was added to prevent.
    #[test]
    fn the_dry_run_states_one_checkpoint_of_peak_when_the_decode_releases() {
        // The nineteen shards as they sit in the cache, and the split this
        // build's own dry run reports for them.
        let source = 96_520_617_030u64;
        let decode = 96_282_000_000u64;
        let copy = 245_366_784u64;

        // WITHOUT the flag: the source stands at full size beside the artifact.
        let before = peak_disk(source, decode, copy, false);
        assert_eq!(before, source + decode + copy);
        assert_eq!(crate::ui::bytes(before), "180GiB");

        // WITH it: one checkpoint. The artifact is the larger copy here by a
        // few hundred megabytes — a re-quantization keeps the weights' width
        // and adds a tokenizer, a config and a manifest — so the peak follows
        // the artifact and not the source, which is the point of taking the
        // max rather than naming one of them.
        let after = peak_disk(source, decode, copy, true);
        assert_eq!(after, decode + copy);
        assert!(
            after > source,
            "the artifact is the larger copy on this one"
        );
        assert_eq!(crate::ui::bytes(after), "89.9GiB");
        assert!(
            before > 2 * after - (1 << 30),
            "the flag has to be worth about half: {before} before, {after} after"
        );

        let said = peak_sentence(source, decode, copy, true, true);
        assert!(
            said.contains("peak use is about 89.9GiB, where holding both at once would be 180GiB."),
            "the dry run has to print the peak it computed, and printed: {said}"
        );
        assert!(
            said.contains("`--consume-source` releases each source range as it is read"),
            "and has to say what makes the difference, so an operator can tell \
             whether it applies to them: {said}"
        );
        assert!(
            said.contains("schedule is NOT ascending")
                && said.contains("the spool itself adds nothing"),
            "a spooled import sits at its peak for two passes and the line says so: {said}"
        );

        // And the same checkpoint without the flag still says what it will
        // cost AND what the flag would save, because that is the whole use of
        // running this before a 90 GiB import.
        let unconsumed = peak_sentence(source, decode, copy, true, false);
        assert!(
            unconsumed.contains("about 180GiB in all")
                && unconsumed.contains("down to about 89.9GiB"),
            "{unconsumed}"
        );
    }

    /// A store archive is named for its directory, not for its file.
    ///
    /// Every archive in the store is called `archive.zt`, so the file-stem
    /// rule would have called every model on the machine `archive` — and
    /// `pie model build` would then have written every build into one shared
    /// `models/archive/runtime/` directory.
    #[test]
    fn a_store_archive_takes_its_name_from_its_directory() {
        assert_eq!(
            store_archive_name(Path::new("/home/u/.pie/models/Qwen--Qwen3-0.6B/archive.zt"))
                .as_deref(),
            Some("Qwen--Qwen3-0.6B")
        );
        // Anything else keeps the ordinary stem rule.
        assert_eq!(store_archive_name(Path::new("/data/qwen.zt")), None);
        assert_eq!(store_archive_name(Path::new("archive.gguf")), None);
        // A bare `archive.zt` has no directory to be named for.
        assert_eq!(store_archive_name(Path::new("archive.zt")), None);
    }

    /// The destination is compared to the source by identity, not by spelling.
    ///
    /// Re-importing a store archive resolves the destination to the source
    /// itself, and `--force` skips the "already pie's own format" return that
    /// otherwise covers it. The writer would then publish over the file the
    /// executor is reading, and `--delete-source` would finish by deleting the
    /// result — so the two are compared canonically, where a symlink or a `..`
    /// cannot spell one file two ways.
    #[test]
    fn a_destination_that_is_the_source_is_recognized_through_a_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("archive.zt");
        std::fs::write(&real, b"weights").unwrap();
        let link = dir.path().join("alias.zt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        assert!(same_file(&real, &real));
        assert!(same_file(&link, &real), "a symlink is not a second file");
        assert!(same_file(&dir.path().join("./archive.zt"), &real));
        assert!(!same_file(&dir.path().join("other.zt"), &real));
        // A destination that does not exist yet cannot be the source.
        assert!(!same_file(&dir.path().join("nowhere.zt"), &real));
    }

    /// **THE ROW IS THE OPERATOR'S TO NAME.** `--sku` parses as an optional
    /// name and is absent by default, which is what keeps an import with no
    /// flag exactly the import it was: `choose_row(None, ..)` is the
    /// identification order, first fits wins.
    #[test]
    fn the_row_override_parses_as_a_name_and_is_absent_by_default() {
        use clap::Parser;

        #[derive(Parser)]
        struct Just {
            #[command(flatten)]
            args: ImportArgs,
        }

        let plain = Just::parse_from(["pie", "google/gemma-4-E4B-it"]).args;
        assert_eq!(plain.sku, None, "no flag is no override");

        let named = Just::parse_from([
            "pie",
            "google/gemma-4-E4B-it",
            "--sku",
            "gemma4-e4b-vision-bf16-kv-bf16",
        ])
        .args;
        assert_eq!(named.sku.as_deref(), Some("gemma4-e4b-vision-bf16-kv-bf16"));

        // The name is a value, not a flag: `--sku` with nothing after it is a
        // usage error rather than a silent `None`.
        assert!(Just::try_parse_from(["pie", "google/gemma-4-E4B-it", "--sku"]).is_err());
    }

    /// A name this build does not ship is refused BY NAME, with the catalog,
    /// and refused early — the check runs before the source is resolved, so a
    /// misspelling costs nothing.
    #[test]
    fn an_unknown_row_name_is_refused_with_the_catalog() {
        let why = runtime::engine::load::row_named("gemma4-vision")
            .expect_err("no row carries that name")
            .to_string();
        assert!(why.contains("gemma4-vision"), "{why}");
        assert!(
            why.contains("gemma4-e4b-vision-bf16-kv-bf16"),
            "the refusal lists the rows this build ships, which is how an \
             operator finds the one they meant: {why}"
        );
        // And `--sku '?'` is that same refusal, which is why the flag's help
        // offers it as the way to see the names.
        let listed = runtime::engine::load::row_named("?")
            .expect_err("`?` is not a row")
            .to_string();
        assert!(
            listed.contains("gemma4-e4b-vision-bf16-kv-bf16"),
            "{listed}"
        );
    }

    /// A row that does not read the checkpoint refuses with the contract's
    /// own account and never falls back to the row that would have fitted.
    #[test]
    fn a_row_that_does_not_read_the_checkpoint_refuses_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stranger.zt");
        let mut writer = ztensor::Writer::create(&path).unwrap();
        writer
            .add(
                "a.tensor.no.model.in.this.catalog.reads",
                vec![1u64],
                ztensor::Leaf::U8,
                &[0u8],
            )
            .unwrap();
        writer.finish().unwrap();

        let source = ztensor::Source::open(&path).unwrap();
        let metadata = runtime::engine::load::checkpoint_metadata(&path).unwrap();
        let asked = "gemma4-e4b-vision-bf16-kv-bf16";

        // Without the flag: nothing claims it, and the import says so.
        assert!(
            choose_row(None, &source, &metadata, Platform::Vulkan, &path).is_err(),
            "a checkpoint of one stranger is claimed by no row"
        );

        let why = format!(
            "{:#}",
            choose_row(Some(asked), &source, &metadata, Platform::Vulkan, &path)
                .expect_err("the named row does not read this checkpoint")
        );
        assert!(
            why.contains(asked) && why.contains("--sku"),
            "the refusal names the row that was asked for, and the flag that \
             asked for it: {why}"
        );
        assert!(
            why.contains("for no other"),
            "and says plainly that nothing fell back: {why}"
        );
    }

    /// The chosen row reaches the artifact's NAME, not just its stamp: the
    /// dry run prints this path, and a store holding one snapshot at two rows
    /// keeps them apart by it.
    #[test]
    fn the_chosen_row_is_in_the_filename_the_dry_run_reports() {
        let stamp = checkpoint::serving::Stamp::of("vulkan", "gemma4-e4b-vision-bf16-kv-bf16");
        let written = Path::new("/home/u/.pie/models/google--gemma-4-E4B-it/archive.zt");
        assert_eq!(
            specialized_path(written, "google--gemma-4-E4B-it", &stamp, None),
            Path::new(
                // The slug is lowercased by `serving::slugify`; the row and
                // the backend are the two fields that matter here.
                "/home/u/.pie/models/google--gemma-4-E4B-it/\
                 google--gemma-4-e4b-it.gemma4-e4b-vision-bf16-kv-bf16.vulkan.zt"
            ),
        );
        // The text row of the same snapshot is a different file, which is the
        // whole point of naming the specialization.
        let text = checkpoint::serving::Stamp::of("vulkan", "gemma4-e4b-bf16-kv-bf16");
        assert_ne!(
            specialized_path(written, "google--gemma-4-E4B-it", &stamp, None),
            specialized_path(written, "google--gemma-4-E4B-it", &text, None),
        );
        // An `--out` that names a `.zt` is the operator's own name and keeps it.
        let out = Path::new("/tmp/mine.zt");
        assert_eq!(specialized_path(out, "whatever", &stamp, Some(out)), out);
    }
}
