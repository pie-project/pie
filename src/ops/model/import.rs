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
//! FAMILY-BLIND, and now entirely: it does not know or ask what model this
//! is. It had a family-aware half — an ingest pass that renamed a GGUF's
//! tensors into the vocabulary a legacy load contract would bind, and a
//! `pie model build` beside it that authored that contract — and R3 deleted
//! both with the contract. A driver produces its weights from the checkpoint
//! through the SKU's own import table at load, so a naming pass here would be
//! an earlier, worse answer to a question the load already asks.

use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;

use model_loader::checkpoint::read::{parse_checkpoint_attributes, parse_checkpoint_metadata};
use model_loader::checkpoint::write::CheckpointWriter;
use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
use model_loader::contract::materialize::{Materialization, materialize_contract};
use model_loader::executor::Progress;
use model_loader::executor::sink::TensorSink;
use model_loader::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use model_loader::types::{CheckpointFormat, Encoding, TensorDecl, Visibility};
use model_loader::verify::ContractView;

// The artifact's on-disk names come from whoever owns them: the loader owns
// the metadata namespace and the provenance attributes,
// `model::serve::encoding` owns the object the checkpoint's own config lands
// in. A literal here would be a
// second definition of something a reader elsewhere has to match exactly, and
// a mismatch does not fail — the read just finds nothing.
use model::serve::encoding::CONFIG_OBJECT;
use model_loader::checkpoint::Attributes;
use model_loader::checkpoint::meta::{SOURCE_ENCODING_KEY, SOURCE_KEY, VERSION_KEY, meta_name};

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
    /// After the artifact is written and every tensor digest verifies, delete
    /// the source weight files it was computed from. Config and tokenizer
    /// files stay.
    ///
    /// Only for a source that is already on disk. Reclaiming the HuggingFace
    /// snapshots an import downloaded is `pie cache clear snapshots`, which
    /// knows about all of them rather than the one just fetched, asks before
    /// deleting, and says how much it got back.
    #[arg(long, conflicts_with = "consume_source")]
    pub delete_source: bool,
    /// Release each source weight file's bytes as they are read, so the source
    /// shrinks while the artifact grows and the import needs room for one copy
    /// rather than two. The files are deleted when it ends.
    ///
    /// This is `--delete-source` without its safety property: the bytes go
    /// before the artifact's digests verify, so an import that fails partway
    /// leaves neither a usable source nor a usable artifact and the source has
    /// to be fetched again. That is why it is already the default for a
    /// snapshot this run downloaded — those are re-fetchable by definition —
    /// and has to be asked for when the source was already on disk.
    #[arg(long, conflicts_with = "keep_source")]
    pub consume_source: bool,
    /// Leave a freshly downloaded snapshot intact, which an import otherwise
    /// consumes. For a HuggingFace cache shared with other tools.
    #[arg(long)]
    pub keep_source: bool,
}

/// Deletes the weight files a consuming import has been releasing as it read.
///
/// A guard rather than a call at the end, because the files have to go whether
/// the import succeeded or gave up: a released file keeps its length and reads
/// back as zeros where the holes are, and both pie's downloader and
/// HuggingFace's take a full-length file for a complete one. Leaving one
/// behind would be worse than leaving nothing — the next reader would get
/// zeros instead of an error.
struct Consumed<'a> {
    metadata: &'a CheckpointMetadata,
}

impl Drop for Consumed<'_> {
    fn drop(&mut self) {
        let _ = remove_source_files(self.metadata);
    }
}

pub fn run(args: ImportArgs) -> Result<crate::ui::Answer> {
    let source = resolve_source(&args.source)?;
    let metadata = parse_checkpoint_metadata(&source.path)
        .map_err(|err| anyhow!("cannot read {}: {err}", source.path.display()))?;

    let out_file = match &args.out {
        Some(out) => artifact_path(out, &source.name),
        None => store_path(&source.name),
    };

    // AN IMPORT MAY NOT WRITE OVER WHAT IT IS READING.
    //
    // A store archive is `<name>/archive.zt` and is named for its directory,
    // so re-importing one resolves the destination to the source itself. The
    // `.zt`-in, `.zt`-out early return below catches the ordinary case, but
    // `--force` exists precisely to skip it, and the writer would then publish
    // over the file the executor is streaming out of. `--delete-source` would
    // finish by deleting the result.
    //
    // Compared against the checkpoint's own file list rather than against
    // `source.path`, so a shard is caught as well as a root.
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

    let mut materialization =
        materialize_contract(&metadata).map_err(|err| anyhow!("cannot convert: {err}"))?;
    // The GGUF's own key-value block, when there is one. Read for
    // `carry_config` below, which carries it verbatim as the artifact's
    // `model/config` — the file's own words about itself, in the file's own
    // spelling.
    let attributes = gguf_attributes(&source, &metadata);

    // THE INGEST PASS STOOD HERE, and R3 deleted it with the load contract
    // it was writing names for.
    //
    // It renamed a GGUF's `blk.0.attn_q` into pie's vocabulary, regrouped a
    // rope pair's rows, unfolded llama.cpp's `+1` norm constant and cut
    // expert tensors out of the stacks it had joined — every one of those a
    // `model_legacy::ingest::Ingest` row, keyed to the family the legacy
    // catalog identified, so that the legacy contract could later bind the
    // artifact by name.
    //
    // There is no such binding any more. A driver produces its weights from
    // the CHECKPOINT through the SKU's own import table
    // (`model::import_of`), which reads the source's own spelling and states
    // every rewrite it performs. So a naming pass here would be a second,
    // earlier answer to a question the load already answers — and the wrong
    // one for a GGUF, whose rewrites are the import table's to declare.
    //
    // What this command is, now that it is only that: the family-BLIND half.
    // It fetches, decodes what the loader can decode, copies the rest byte
    // for byte onto its own 64 KiB pages with an XXH3 digest each, and
    // records provenance. Nothing in it asks what model this is.
    // Before the counts are printed, so a `--dry-run` reports what a real run
    // would write. See `declares_tied_head`.
    if declares_tied_head(&source) {
        let heads = tied_head_sources(&metadata);
        for name in drop_tied_head(&mut materialization, &heads) {
            println!(
                "convert: dropping `{name}` — this checkpoint declares \
                 `tie_word_embeddings`, so its head IS the embedding and \
                 carrying it would put a duplicate of the embedding in the \
                 artifact"
            );
        }
    }
    // Three counts and not two, because the middle one used to be folded into
    // the first and stopped being true.
    //
    // `Materialization::decoded` once held both kinds of rewrite a conversion
    // performs: unpacking a self-contained block, and narrowing an F16 or F32
    // tensor to the BF16 every kernel reads. Calling the whole set "blocked"
    // was loose but harmless while blocks dominated it. Since an archive keeps
    // its source packing, the set holds narrowings ONLY -- so the old line
    // reported "decode 65 blocked tensor(s)" for a Q3_K_M import that decoded
    // no block at all, and the blocks it did keep were invisible inside a
    // "copy through" count that also covers plain BF16.
    //
    // The counts are read by an operator deciding whether the import did what
    // they meant, so they are named for what they are. `packed` is counted off
    // the source rather than tracked through the materialization, because that
    // is where the fact is: a tensor is kept packed exactly when its scheme
    // carries its scales inside it.
    let packed = metadata
        .weights()
        .filter(|tensor| {
            matches!(&tensor.encoding, Encoding::Quant(spec) if spec.scheme.is_self_contained())
        })
        .count();
    println!(
        "convert: narrow {} tensor(s) to bf16, keep {} packed as stored, copy {} through",
        materialization.decoded.len(),
        packed,
        materialization.passthrough.len().saturating_sub(packed)
    );
    // Metadata compiles here, before any bytes are written: an artifact whose
    // weights are perfect but whose tokenizer would not compile cannot serve,
    // and finding that out after copying 800 GB helps nobody.
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
        // A GGUF has no config.json and does not need one: it states the same
        // things in its own key-value block, so that is what gets carried.
        // Not a fabricated HuggingFace document -- the keys stay `qwen2.*`
        // and `general.*`, which is what the file actually said.
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

    if let Some(max) = args.max_shard_size {
        println!(
            "import: sharding at about {} per file; the root is {}",
            crate::ui::bytes(max),
            out_file.display()
        );
    }

    // Asking for `--delete-source` is asking for its order -- verify, then
    // delete -- so it turns the default off rather than doubling up with it.
    let consume =
        !args.keep_source && !args.delete_source && (args.consume_source || source.fetched);
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
        return Ok(crate::ui::Answer::noop(format!(
            "dry run: would write {}",
            crate::ui::short_path(&out_file)
        )));
    }

    // The passthrough set, resolved to source addresses up front — the copy
    // total is part of the progress denominator from the first frame, and
    // resolving the file here leaves the copy loop nothing to look up.
    let mut passthrough: Vec<(&RawTensor, &str, &str)> =
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
        // A tensor keeps the name it came in under: import is the
        // family-blind half and renames nothing since R3 cut the ingest
        // pass, so the merge's entry list holds a borrow of the source's
        // own spelling and costs no copy per tensor.
        passthrough.push((raw, file.path.as_str(), raw.name.as_str()));
    }
    let copy_bytes: u64 = passthrough.iter().map(|(raw, _, _)| raw.span_bytes).sum();

    // pie's own objects, named into the reserved namespace so the write can
    // merge them with the weights in one ascending pass.
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

    // The decode, compiled but not yet run.
    //
    // Which of the two ways it runs is decided just below, and both end at the
    // same place: `merge_entries` asks for a decoded tensor by name and gets
    // its bytes. What differs is whether those bytes went to disk first.
    let plan = if materialization.contract.tensors.is_empty() {
        None
    } else {
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            max_tile_bytes: 64 << 20,
            ..StorageTarget::default()
        };
        let plan = model_loader::plan::compile(&metadata, &materialization.contract, target)
            .map_err(|err| anyhow!("cannot compile the decode: {err}"))?;
        // A SECOND OPINION, before a byte is read.
        //
        // `verify` is not a second compiler: it takes the plan as it stands and
        // asks what can be answered from the plan plus the filesystem — is the
        // schedule a permutation of the instructions, is every public
        // declaration finalized exactly once, does every read land inside the
        // file it names at the size that file actually is, and does the result
        // deliver the contract this was compiled from. `compile` cannot catch
        // those itself, because the same wrong belief would have produced both
        // halves.
        //
        // It ran only in this crate's own golden tests, over sixteen compiled
        // plans, while the command a user actually runs skipped it. There is no
        // reason for that asymmetry: an import reads gigabytes and writes an
        // artifact other machines will serve, so it is the caller with the most
        // to gain from finding out here rather than at load.
        if let Err(violations) = model_loader::verify::verify_plan(
            &plan,
            Some(&ContractView::of(&materialization.contract)),
        ) {
            let listed: Vec<String> = violations.iter().map(ToString::to_string).collect();
            bail!(
                "the compiled decode does not honour its contract:\n  {}",
                listed.join("\n  ")
            );
        }
        Some(plan)
    };

    // The decode's read total, taken from the checkpoint rather than from the
    // executor, so the progress denominator is whole from the first frame
    // instead of arriving with it.
    let decode_bytes: u64 = materialization
        .decoded
        .iter()
        .filter_map(|name| metadata.tensor_by_name(name))
        .map(|raw| raw.span_bytes)
        .sum();

    // A schedule that already runs in the artifact's order needs no buffer
    // between the two: its tensors are written as they are produced, and the
    // decode overlaps the write instead of preceding it.
    //
    // Measured on this tree, the spool is what an F16 checkpoint pays most
    // for. `huggyllama/llama-7b` reads 12.6 GiB and wrote 25.1 GiB to convert
    // it -- the decoded set lands on disk once as the spool and once as the
    // artifact -- and the decode ran to completion before the first artifact
    // byte was written. Every checkpoint tried here (safetensors F16 and F32,
    // a five-shard MLX 4-bit mix, GGUF Q4_0) schedules in ascending name
    // order, so this is the path they take.
    //
    // The spool stays for the ones that do not. `weights()` is manifest order
    // and nothing downstream promises to sort it, so a schedule that arrives
    // out of order is a checkpoint pie has not seen rather than a bug, and it
    // gets the buffer it needs.
    let ordered = plan.as_ref().is_none_or(|plan| {
        plan.tensors
            .windows(2)
            .all(|pair| pair[0].name <= pair[1].name)
    });
    let mut spool = match &plan {
        Some(plan) if !ordered => {
            let mut spool = Spool::create(&out_file)?;
            model_loader::executor::Execution::new(plan, &source.base())
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
                .map_err(|err| anyhow!("decoding failed: {err}"))?;
            Some(spool)
        }
        _ => None,
    };

    let provenance = BTreeMap::from([
        (VERSION_KEY.to_string(), pie_version().to_string()),
        (SOURCE_KEY.to_string(), source.origin.clone()),
        // HOW THE SOURCE STORED THESE NUMBERS, BEFORE THIS COMMAND DID.
        //
        // The one fact the artifact cannot state about itself: everything
        // below keeps a self-contained block packed, so the tensors do say
        // what they are — but only to a reader that walks all of them, and a
        // build should not have to open the model to learn whether a second
        // rounding is on the table. See `SOURCE_ENCODING_KEY`.
        (SOURCE_ENCODING_KEY.to_string(), source_encoding(&metadata)),
    ]);
    let mut writer = match args.max_shard_size {
        Some(max) => CheckpointWriter::create_sharded(&out_file, &provenance, max),
        None => CheckpointWriter::create(&out_file, &provenance),
    }
    .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;
    if consume {
        println!(
            "import: consuming the source as it is read; {} will not survive this run",
            if source.fetched {
                "the snapshot just downloaded"
            } else {
                "the source files"
            }
        );
    }
    let consumed = consume.then(|| Consumed {
        metadata: &metadata,
    });
    let written_bytes = write_artifact(
        &mut writer,
        plan.as_ref(),
        &source.base(),
        spool.as_mut(),
        &passthrough,
        &meta,
        &mut bar,
        decode_bytes,
        copy_bytes,
        consume,
    )?;
    // Closing belongs to whoever opened it: `finish` consumes the writer.
    writer
        .finish()
        .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;
    if let Some(spool) = spool {
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
    if args.delete_source {
        delete_source(&source.name, &metadata, &out_file)?;
    }
    if let Some(consumed) = consumed {
        drop(consumed);
        println!(
            "{}: consumed {} source file(s), {}",
            source.name,
            metadata.files.len(),
            crate::ui::bytes(source_bytes(&metadata))
        );
    }
    Ok(crate::ui::Answer::did(did))
}

/// Say whether the artifact this run is about to write is one this build can
/// serve, and do no more than say it.
///
/// A report and not a refusal. Conversion is the family-blind half of the
/// pair, and rewriting a checkpoint this build has no row for is a thing to
/// want: the artifact is a container, and a row for it may land in a later
/// build. What was not defensible was the SILENCE — `import` would finish,
/// print a size and exit 0, and the first word about the model being unknown
/// came from `build` after the bytes were on disk.
///
/// Held against the artifact this run will write, not the checkpoint it is
/// reading, because the two disagree in exactly the way that matters here. A
/// tie is spelled in a catalog row as the ABSENCE of `lm_head`, so a source
/// that ships the materialized head fails a row its own conversion satisfies
/// — see `declares_tied_head`. The projection is `dropped` removed and
/// nothing else: materializing renames no tensor, and `Observed` already
/// reports a packed tensor at its unpacked extents.
///
/// Costs no I/O. `Manifest::check` matches on names and extents, both of
/// which are in the metadata that has already been parsed.
/// What a GGUF source says about itself, or `None` for a source that is not
/// one.
///
/// Read once and lent to both readers. Two of them want it — the rename wants
/// the architecture, and the config wants the whole block — and the file is
/// the same file. The cost is a header parse rather than a scan, but a second
/// one still buys nothing.
fn gguf_attributes(source: &Source, metadata: &CheckpointMetadata) -> Option<Attributes> {
    if !metadata
        .files
        .iter()
        .any(|file| file.format == CheckpointFormat::Gguf)
    {
        return None;
    }
    parse_checkpoint_attributes(&source.path).ok()
}

// `ingest_map`, `apply_ingest`, `unpermute` and `report_servability` STOOD
// HERE — the family-aware half of this command. The first three applied
// `model_legacy::ingest`'s per-family naming pass to the materialization; the
// fourth held the projected artifact against the legacy catalog and printed
// which row it would identify as. R3 deleted the catalog and the contract
// that read those names, so all four are gone: see the note in `run`.

fn report_would_delete(metadata: &CheckpointMetadata) {
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
fn delete_source(repo_id: &str, metadata: &CheckpointMetadata, artifact: &Path) -> Result<()> {
    let verified = model_loader::checkpoint::zt::verify_checkpoint(artifact).map_err(|err| {
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
fn remove_source_files(metadata: &CheckpointMetadata) -> Result<()> {
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
fn source_bytes(metadata: &CheckpointMetadata) -> u64 {
    metadata.files.iter().map(|file| file.size_bytes).sum()
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

/// Gives a range of a file back to the filesystem, without changing the file's
/// length or the bytes outside the range.
///
/// Two callers, one situation: bytes that have just been read for the last
/// time, in a file that is going to be deleted. Releasing as the read passes
/// over them means the old copy shrinks while the new one grows, instead of
/// both standing at full size until the end — which is the difference between
/// needing room for two copies of a checkpoint and needing room for one.
///
/// Best-effort by design, and unchecked for that reason. A filesystem that
/// cannot do this fails the call and the file stays whole, which costs space
/// and nothing else: the bytes were read either way, and the caller deletes
/// the file either way.
fn release(file: &std::fs::File, offset: u64, len: u64) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: `fallocate` only changes the allocation of the range it is
        // given on a file descriptor the caller owns; it writes no memory.
        // Within the range, whole blocks are deallocated and partial ones are
        // zeroed, so no byte outside `offset..offset + len` is touched --
        // which is what lets a caller release a range whose neighbours have
        // not been read yet. `KEEP_SIZE` is redundant with `PUNCH_HOLE`, which
        // never moves the end of the file, and is passed because the manual
        // requires the pair.
        unsafe {
            libc::fallocate(
                file.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                offset as libc::off_t,
                len as libc::off_t,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset, len);
    }
}

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
    ) -> std::result::Result<(), model_loader::error::Error> {
        use std::io::Write;
        self.file.write_all(bytes).map_err(|err| {
            model_loader::error::Error::Checkpoint(format!(
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
/// free of `model_loader` -- what it needs to draw a bar is two numbers and a
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
fn source_encoding(metadata: &model_loader::checkpoint::CheckpointMetadata) -> String {
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
            model_loader::types::Encoding::Raw(dtype) if dtype.is_block_scaled() => {
                format!("quant:{}", format!("{dtype:?}").to_lowercase())
            }
            model_loader::types::Encoding::Raw(dtype) => {
                format!("raw:{}", format!("{dtype:?}").to_lowercase())
            }
            // The variant name lowercased, minus the `Gguf` family prefix:
            // `GgufQ4_0` is the scheme llama.cpp and every model card call
            // `Q4_0`, and this string is read by operators, not by Rust.
            model_loader::types::Encoding::Quant(spec) => {
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

/// Carries the source's `config.json` into the artifact, verbatim.
///
/// # Why this stopped normalizing
///
/// It used to compile the config into a `pie.model/1` descriptor: 136 fields
/// of normalized geometry, which the driver then re-parsed to learn what
/// model it had. That was the *identity* crossing as a document, and it is
/// what the catalog refactor removed — identity is now a manifest match
/// against the tensors, and the tensors are already in the artifact.
///
/// What is left for a config to say is the part the tensors cannot: the
/// declared quantization, because a group size is not an extent of anything.
/// [`model::serve::encoding::Encoding`] reads exactly that, from the
/// checkpoint's own words, so the honest thing to carry is the checkpoint's
/// own words.
///
/// It is also why this can no longer fail on content. A config this command
/// does not understand is not this command's problem — nothing here reads it,
/// and `Encoding` refuses what it cannot parse at the point that needs it.
/// Only unreadable bytes or invalid JSON are errors, and JSON is checked so
/// that an artifact never carries an object no reader can open.
///
/// `Ok(None)` when there is no `config.json` — a lone `.gguf` carries its
/// metadata in its own header, and a directory without one is a weights-only
/// checkpoint.
/// Whether the source declares its output head TIED to the embedding.
///
/// # Why the importer cares
///
/// A tie means the model has no separate head: the forward reads the embedding
/// table transposed. HuggingFace nonetheless ships a materialized
/// `lm_head.weight` beside it in every stock Qwen3 export — byte for byte the
/// same tensor as `model.embed_tokens.weight` — and `catalog::identify` spells
/// a tie as the ABSENCE of that name (the catalog), so the
/// artifact is refused by the one row that describes it:
///
/// ```text
/// matches no catalog row: qwen3-0.6b: unexpected lm_head
/// ```
///
/// with every other name and every extent agreeing. Carrying the duplicate
/// through therefore costs the artifact its identity, and the weight it buys
/// is one nothing reads.
///
/// The config is the authority and the byte comparison is not needed: if the
/// checkpoint declares the head tied then the forward uses the embedding,
/// whatever those bytes happen to hold. A checkpoint that meant them to differ
/// would be one that did not declare the tie.
// `declared_model_type` STOOD HERE - `config.json`'s `model_type`, the
// HuggingFace half of `general.architecture`, read so the ingest pass could
// pick a family's naming table. The pass is gone and nothing else here asks
// what model this is: import is the family-blind half.

fn declares_tied_head(source: &Source) -> bool {
    if source.path.is_file() {
        return false;
    }
    let Ok(raw) = std::fs::read(source.path.join("config.json")) else {
        return false;
    };
    serde_json::from_slice::<serde_json::Value>(&raw)
        .ok()
        .and_then(|v| {
            v.get("tie_word_embeddings")
                .or_else(|| v.get("text_config")?.get("tie_word_embeddings"))
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false)
}

/// The names a tied checkpoint materializes for a head it does not have.
const TIED_HEAD_NAMES: [&str; 2] = ["lm_head.weight", "lm_head.bias"];

/// Removes a materialized tied head from every set that would write it.
///
/// Returns what was dropped, which is what the caller reports. Both sets are
/// swept and the contract with them: a head stored as F16 lands in `decoded`
/// with a `TensorContract` behind it, and one stored as BF16 lands in
/// `passthrough`, so removing it from only the set this checkpoint happened to
/// use would work until the next checkpoint chose the other width.
fn drop_tied_head(materialization: &mut Materialization, heads: &[String]) -> Vec<String> {
    let mut dropped = Vec::new();
    let mut take = |set: &mut Vec<String>| {
        set.retain(|name| {
            let keep = !heads.contains(name);
            if !keep {
                dropped.push(name.clone());
            }
            keep
        });
    };
    take(&mut materialization.decoded);
    take(&mut materialization.passthrough);
    // By the ARTIFACT name, which is what a contract declares. Any rename
    // has already been applied to these, so this list is `TIED_HEAD_NAMES`
    // in both vocabularies at once -- and it is the same list as `heads` for
    // every checkpoint that needed no rename.
    materialization
        .contract
        .tensors
        .retain(|t| !TIED_HEAD_NAMES.contains(&t.name.as_str()));
    dropped
}

/// The source's own names for the tensors that would be written as a head.
///
/// One vocabulary now. It took two while the ingest pass renamed a GGUF's
/// `output.weight` on the way in and this list had to be derived from that
/// rename; with the pass gone, a tensor keeps the name it arrived under, so
/// this is `TIED_HEAD_NAMES` filtered by what the file holds. A GGUF's head
/// is therefore no longer found by this — which is correct rather than a
/// regression: `declares_tied_head` reads `tie_word_embeddings` out of a
/// `config.json` a GGUF does not have, so the drop was never reachable from
/// the format's own statement in the first place.
fn tied_head_sources(metadata: &CheckpointMetadata) -> Vec<String> {
    metadata
        .tensors
        .iter()
        .map(|tensor| &tensor.name)
        .filter(|name| TIED_HEAD_NAMES.contains(&name.as_str()))
        .cloned()
        .collect()
}

// `head_is_a_materialized_tie` and `will_publish` STOOD HERE. The first
// asked the legacy catalog whether the row this checkpoint identifies as
// spells its head as a tied copy, so that a GGUF stating no tie (the format
// has no key for one) still had its duplicate head dropped; the second
// projected the artifact's names for it to ask about. Both die with the
// catalog. What is left is `declares_tied_head`, which reads
// `tie_word_embeddings` out of the checkpoint's own `config.json` — the file
// saying so about itself, which is the only source that was ever a fact.

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
    metadata: &CheckpointMetadata,
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

/// The tokenizer a GGUF carries inside itself, when there is no file beside
/// the weights.
///
/// A GGUF is a whole snapshot in one file, tokenizer included, so "a lone
/// `.gguf` has no tokenizer" was never true — it was true of the DIRECTORY.
///
/// # How faithful this is
///
/// Measured, on `Qwen2.5-0.5B-Instruct-Q4_0.gguf` against the same model's
/// `tokenizer.json` compiled through the other path: `vocab_bytes`,
/// `vocab_offsets`, `merge_table` and `byte_fallback` are equal BYTE FOR BYTE
/// — 976,263, 606,664, 2,422,192 and 1,024 bytes — the pipelines are equal,
/// and seven varied strings encode to the same ids.
///
/// One field differs, and it is llama.cpp's disagreement rather than this
/// reader's: six tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`,
/// `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>`) are `special` here and not
/// in the JSON, because llama.cpp's converter promotes anything spelled
/// `<|…|>` to a control token over the model's own answer. `special` reaches
/// only `decode(skip_special)`, so this changes what a decoder hides and
/// nothing about what the model is fed. GGUF's answer is kept, because a
/// GGUF's tokenizer is the one the file was built with.
///
/// # Errors
///
/// None from here. A GGUF whose tokenizer this build cannot read is reported
/// and carries none, where a `tokenizer.json` that will not compile is
/// refused — and the difference is not inconsistency but whose choice it was.
/// A file beside the weights is one someone put there, so failing to use it
/// is a surprise worth stopping for. A GGUF's tables are simply INSIDE it:
/// refusing them would make `Llama-2-7B-GGUF` un-importable even for its
/// weights, and pie can read that model's tokenizer perfectly well from its
/// HuggingFace form — the gap is this reader's coverage, not pie's. Import is
/// already the command that converts what it cannot serve and says so.
fn gguf_tokenizer(
    source: &Source,
    metadata: &CheckpointMetadata,
) -> Result<Option<tokenizer::canonical::CanonicalTokenizer>> {
    if !metadata
        .files
        .iter()
        .any(|file| file.format == CheckpointFormat::Gguf)
    {
        return Ok(None);
    }
    let tables = model_loader::checkpoint::read::parse_checkpoint_tokenizer(&source.path)?;
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
    let attributes = match model_loader::checkpoint::zt::read_attributes(artifact) {
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

/// Where one entry of the artifact's ascending merge gets its bytes.
enum From<'a> {
    Decoded(&'a TensorDecl),
    /// The tensor and the file its bytes are in.
    Copy(&'a RawTensor, &'a str),
    Meta(&'a [u8]),
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
/// Decoded tensors come from executor storage, passthrough is read by the
/// lanes below, metadata comes from memory.
/// The two byte counts are the progress denominator: what the decode reads,
/// and what this pass copies.
///
/// `spool` present means the decode already ran into it; absent with a `plan`
/// means it runs here, on a thread of its own, straight into the merge.
#[allow(clippy::too_many_arguments)]
fn write_artifact<'a>(
    writer: &mut CheckpointWriter,
    plan: Option<&'a model_loader::plan::LoadPlan>,
    base: &Path,
    spool: Option<&'a mut Spool>,
    passthrough: &[(&'a RawTensor, &'a str, &'a str)],
    meta: &'a [(String, Vec<u8>)],
    progress: &mut ProgressLine,
    decode_bytes: u64,
    copy_bytes: u64,
    consume: bool,
) -> Result<u64> {
    let mut entries: Vec<(&'a str, From<'a>)> = Vec::new();
    if let Some(plan) = plan {
        // The artifact holds what the contract made public. An internal
        // tensor is scaffolding one public tensor is built out of -- the
        // unfolded Gemma norm a cast then narrows -- and writing it would put
        // a name in the artifact that no catalog row can account for.
        for decl in plan
            .tensors
            .iter()
            .filter(|decl| decl.visibility.is_public())
        {
            entries.push((&decl.name, From::Decoded(decl)));
        }
    }
    for (raw, path, output) in passthrough {
        entries.push((output, From::Copy(raw, path)));
    }
    for (name, bytes) in meta {
        entries.push((name.as_str(), From::Meta(bytes)));
    }
    entries.sort_by(|a, b| a.0.cmp(b.0));

    // The copies, cut into reads and dealt round-robin to a few threads.
    //
    // The writer is and must stay sequential: `ztensor` appends, and a
    // tensor's digest accumulates over its bytes in order. The READS have no
    // such constraint, and they were where the wall clock went. A read and the
    // write that follows it are never in flight at once in a single thread, so
    // the device idles for whichever half is not running -- measured on this
    // tree, an import moved ~2.4 GiB/s of combined traffic where the same
    // files sustain ~6.6 GiB/s with several reads outstanding against one
    // writer.
    //
    // Dealing chunk `k` to lane `k % lanes` is what keeps this cheap. Each
    // lane produces its own chunks in order, so taking them back in that same
    // order reassembles the global order for free: there is no reordering
    // buffer here, only channels, and the writer still sees one ordered
    // stream.
    let chunks = plan_chunks(&entries);
    let lane_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
        .min(chunks.len().max(1));
    // When the decode streams it is simply another producer feeding this same
    // merge, running here instead of before it: its reads overlap the
    // artifact's writes, and its output goes to the artifact directly rather
    // than to disk and back.
    let decode_read = std::sync::atomic::AtomicU64::new(match spool {
        // The spooled decode already ran, and already counted its reads.
        Some(_) => decode_bytes,
        None => 0,
    });
    std::thread::scope(|scope| {
        let mut decoded = match (plan, spool) {
            (Some(plan), None) => {
                // One tensor in the channel and one being handed over bounds
                // this at two tensors of memory, which is the promise the
                // spool was keeping: peak is a working set, not the model.
                let (to, from) = std::sync::mpsc::sync_channel(1);
                let decode_read = &decode_read;
                scope.spawn(move || {
                    decode_into(plan, base, &to, &|read, _total| {
                        decode_read.store(read, std::sync::atomic::Ordering::Relaxed);
                    })
                });
                Decoded::Streamed(from)
            }
            (_, Some(spool)) => Decoded::Spooled(spool),
            (None, None) => Decoded::Nothing,
        };
        let mut lanes = Vec::with_capacity(lane_count);
        for lane in 0..lane_count {
            // `DEPTH` full and `DEPTH` empty buffers per lane bounds the
            // memory in flight at `lane_count * DEPTH * CHUNK`, ~512 MiB here.
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
            &entries,
            &chunks,
            &mut decoded,
            &lanes,
            progress,
            &decode_read,
            decode_bytes,
            copy_bytes,
        );
        // Dropping the producers is the shutdown protocol. Whatever happened
        // above -- success, a short read, a failed write -- every reader is
        // parked on one of these channels, and closing the end the merge holds
        // wakes it with the error that tells it to return. Without this the
        // scope never joins.
        drop(lanes);
        drop(decoded);
        outcome
    })
}

/// How much one read moves. Large enough that per-read overhead is noise
/// against an NVMe transfer, small enough that the buffers in flight are
/// bounded in the hundreds of MiB rather than by the largest expert bank.
const CHUNK: u64 = 16 << 20;

/// Reads each lane keeps outstanding ahead of the writer.
const DEPTH: usize = 4;

/// One lane's half of the pipe: where its bytes arrive, and where the writer
/// hands the buffer back once it has been written.
type Lane = (
    std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>,
    std::sync::mpsc::SyncSender<Vec<u8>>,
);

/// The ascending merge itself: one ordered pass over `entries`.
#[allow(clippy::too_many_arguments)]
fn merge_entries(
    writer: &mut CheckpointWriter,
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
                // A checkpoint that decodes end to end has no copies, so this
                // is the only place its progress can come from.
                progress.render(&Progress {
                    read_bytes: decode_read.load(std::sync::atomic::Ordering::Relaxed) + copied,
                    total_read_bytes: decode_bytes + copy_bytes,
                    finalized: Some(name),
                });
            }
            From::Meta(bytes) => {
                let path = name
                    .strip_prefix(model_loader::checkpoint::meta::META_PREFIX)
                    .expect("metadata entries carry the namespace prefix");
                writer
                    .add_meta(path, bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                written_bytes += bytes.len() as u64;
            }
            From::Copy(raw, path) => {
                let decl = TensorDecl {
                    id: raw.id,
                    // The ENTRY's name, which is what this pass is ordered
                    // by. `raw.name` is the source's, and an ingest pass
                    // that renames makes the two differ -- with the sort
                    // reading one and the write the other, which the
                    // canonical writer catches as an out-of-order insert
                    // only when the rename happens to cross a neighbour.
                    name: (*name).to_string(),
                    shape: raw.shape.clone(),
                    encoding: raw.encoding.clone(),
                    alignment: 1,
                    visibility: Visibility::default(),
                };
                writer
                    .begin_tensor(&decl, raw.span_bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                // This tensor's chunks are the next ones in the global
                // sequence, because `plan_chunks` cut them in `entries` order.
                let mut remaining = raw.span_bytes;
                while remaining > 0 {
                    let (filled, recycle) = &lanes[next % lanes.len()];
                    let buffer = filled
                        .recv()
                        .map_err(|_| anyhow!("a reader stopped before '{name}'"))?
                        .with_context(|| format!("cannot read '{name}' from {path}"))?;
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
                written_bytes += raw.span_bytes;
            }
        }
    }
    progress.finish();
    Ok(written_bytes)
}

/// One contiguous read. A chunk never spans two tensors, so its position in
/// the sequence is all the writer needs to know about it.
struct Chunk<'a> {
    path: &'a str,
    offset: u64,
    len: usize,
}

/// The passthrough copies as reads, in the order the writer will append them.
fn plan_chunks<'a>(entries: &[(&str, From<'a>)]) -> Vec<Chunk<'a>> {
    let mut chunks = Vec::new();
    for (_, entry) in entries {
        let From::Copy(raw, path) = entry else {
            continue;
        };
        let mut offset = raw.file_offset;
        let mut remaining = raw.span_bytes;
        while remaining > 0 {
            let len = remaining.min(CHUNK);
            chunks.push(Chunk {
                path,
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
    consume: bool,
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

/// Fill `buffer` from the chunk's file, opening it on first use.
///
/// `read_exact_at` carries its own offset, so lanes sharing a file would not
/// need separate handles for correctness -- they get them anyway because a
/// handle per lane is one less thing to synchronise.
///
/// Under `consume`, the chunk's range is released as soon as it is in the
/// buffer. Chunks partition the passthrough spans and each is read exactly
/// once, by exactly one lane, so a chunk's bytes are dead the moment the read
/// returns — and `release` leaves everything outside the range alone, so lanes
/// working on neighbouring chunks of the same file do not interfere. The
/// handle has to be writable for that, which is the only reason `consume`
/// reaches this far down.
///
/// Only passthrough reads are released. What a decode reads it reads through
/// the loader's own handles, so a conversion-heavy import gets less back while
/// it runs — it gets the rest at the end, when the files are deleted. On the
/// checkpoints where the peak actually hurts this costs little: above 100B
/// parameters almost nothing decodes.
fn read_chunk<'a>(
    open: &mut std::collections::HashMap<&'a str, std::fs::File>,
    chunk: &Chunk<'a>,
    buffer: &mut [u8],
    consume: bool,
) -> std::io::Result<()> {
    let file = match open.entry(chunk.path) {
        std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
        std::collections::hash_map::Entry::Vacant(slot) => slot.insert(
            std::fs::OpenOptions::new()
                .read(true)
                .write(consume)
                .open(chunk.path)?,
        ),
    };
    file.read_exact_at(&mut buffer[..chunk.len], chunk.offset)?;
    if consume {
        release(file, chunk.offset, chunk.len as u64);
    }
    Ok(())
}

/// Where the merge gets a decoded tensor's bytes.
///
/// The two ways differ only in whether the bytes went to disk on the way, and
/// the merge is written not to know which it got.
enum Decoded<'a> {
    /// The executor is running right now, publishing tensors in the order the
    /// merge asks for them.
    Streamed(std::sync::mpsc::Receiver<std::result::Result<(String, Vec<u8>), String>>),
    /// The executor already ran, into a file the merge seeks around in.
    Spooled(&'a mut Spool),
    /// Nothing decodes, so nothing asks.
    Nothing,
}

impl Decoded<'_> {
    /// The bytes for `name`, which the caller asks for in ascending order.
    ///
    /// A decode publishes more than the artifact keeps: `Visibility::Internal`
    /// tensors are published so the arithmetic can name them and then left out
    /// of the bind table, and a checkpoint can declare a tensor this build
    /// drops. Both are names the merge will never ask for, so a stream that is
    /// ascending is consumed by skipping past whatever sorts below the request
    /// rather than by requiring the two sequences to match tensor for tensor.
    ///
    /// What that leaves is the check worth making: a name sorting *above* the
    /// request cannot be skipped past, because the request would then never be
    /// answered. That is the schedule not being ascending after all, and it is
    /// reported rather than assembled.
    fn take(&mut self, name: &str) -> Result<Vec<u8>> {
        match self {
            Self::Streamed(from) => loop {
                let (produced, bytes) = from
                    .recv()
                    .map_err(|_| anyhow!("the decode stopped before '{name}'"))?
                    .map_err(|err| anyhow!("decoding failed: {err}"))?;
                match produced.as_str().cmp(name) {
                    std::cmp::Ordering::Less => continue,
                    std::cmp::Ordering::Equal => break Ok(bytes),
                    std::cmp::Ordering::Greater => {
                        break Err(anyhow!(
                            "the decode produced '{produced}' where the artifact wants '{name}', \
                             so its schedule is not ascending"
                        ));
                    }
                }
            },
            Self::Spooled(spool) => spool.read(name),
            Self::Nothing => bail!("'{name}' decodes, but no decode ran"),
        }
    }
}

/// Run the decode, handing each tensor straight to the merge.
///
/// Errors travel down the channel rather than out of the return type: the
/// merge is the only thing waiting, and a failure it can attribute to a tensor
/// name is worth more than one this thread keeps to itself.
/// `watch` is handed the executor's `(read_bytes, total_read_bytes)`, since
/// this thread cannot touch the caller's progress line.
fn decode_into(
    plan: &model_loader::plan::LoadPlan,
    base: &Path,
    to: &std::sync::mpsc::SyncSender<std::result::Result<(String, Vec<u8>), String>>,
    watch: &(dyn Fn(u64, u64) + Sync),
) {
    let mut sink = Handoff { to };
    let outcome = model_loader::executor::Execution::new(plan, base)
        .streaming()
        .sink(&mut sink)
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
    ) -> std::result::Result<(), model_loader::error::Error> {
        self.to
            .send(Ok((name.to_string(), bytes.to_vec())))
            .map_err(|_| {
                model_loader::error::Error::Checkpoint(format!(
                    "the artifact writer stopped before '{name}'"
                ))
            })
    }
}

#[cfg(test)]
mod tests {

    // FIVE TESTS STOOD HERE — `published()` built a legacy catalog row's
    // manifest as an `Observed` checkpoint, and four assertions held
    // `head_is_a_materialized_tie` against it. They die with the function and
    // with the catalog it asked. What the check did is recorded where it
    // stood, above `carry_config`.

    use super::*;
    use model_loader::checkpoint::write::{WriteTensor, write_zt};
    use model_loader::types::{DType, Encoding, TensorId};

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

    /// A source is asked about its tie by reading its config, not its tensors.
    #[test]
    fn a_tie_is_read_off_the_config_that_declares_it() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let at = |json: &str| {
            std::fs::write(dir.path().join("config.json"), json).expect("write");
            declares_tied_head(&Source {
                path: dir.path().to_path_buf(),
                name: "x".into(),
                origin: "x".into(),
                fetched: false,
            })
        };
        assert!(at(r#"{"tie_word_embeddings": true}"#));
        assert!(!at(r#"{"tie_word_embeddings": false}"#));
        // Silence is not a tie: an untied checkpoint whose head was dropped is
        // a model with no output layer.
        assert!(!at(r#"{"hidden_size": 1024}"#));
        // Multimodal configs nest the text model's flags, and the tie is the
        // TEXT model's.
        assert!(at(r#"{"text_config": {"tie_word_embeddings": true}}"#));
        // Unparseable is not a tie either. Nothing is dropped on a guess.
        assert!(!at("{not json"));
        // A single file has no snapshot to read, so it declares nothing.
        assert!(!declares_tied_head(&Source {
            path: dir.path().join("model.safetensors"),
            name: "x".into(),
            origin: "x".into(),
            fetched: false,
        }));
    }

    // `an_unstacked_mixture_leaves_the_contract_out_of_the_artifacts_order`
    // STOOD HERE — `apply_ingest`'s only test, and it went with the ingest
    // pass. The finding it recorded is worth keeping in words: the cut
    // appended experts in NUMERIC order while everything downstream compares
    // names as STRINGS, and the two part company at ten, which is why `run`
    // still spools an out-of-order set. Nothing in this command reorders a
    // tensor any more, so the case cannot arise here.

    /// The head is dropped from every set that would write it, at either width.
    ///
    /// Both sets are swept because the width decides which one the head lands
    /// in: a BF16 head passes through and an F16 head is decoded to BF16 with a
    /// `TensorContract` behind it. Sweeping only the set a Qwen3 export happens
    /// to use would work until a checkpoint chose the other one.
    #[test]
    fn a_tied_head_is_dropped_from_every_set_that_would_write_it() {
        use model_loader::contract::{Expr, ModelContract, TensorContract};

        let head = |name: &str| {
            TensorContract::new(
                name,
                Expr::src(name).cast(Encoding::Raw(DType::BF16)),
                vec![151936, 1024],
                Encoding::Raw(DType::BF16),
            )
        };
        let mut m = Materialization {
            contract: ModelContract {
                alignment: 1,
                tensors: vec![head("lm_head.weight"), head("model.norm.weight")],
                groups: Vec::new(),
            },
            decoded: vec!["lm_head.weight".into(), "model.norm.weight".into()],
            passthrough: vec!["lm_head.bias".into(), "model.embed_tokens.weight".into()],
            meta: vec!["pie.meta/x".into()],
        };
        // The source's names for the head. Equal to the artifact's here,
        // which is every checkpoint that needs no rename.
        let heads = ["lm_head.weight".to_string(), "lm_head.bias".to_string()];
        let mut dropped = drop_tied_head(&mut m, &heads);
        dropped.sort();
        assert_eq!(dropped, ["lm_head.bias", "lm_head.weight"]);
        assert_eq!(m.decoded, ["model.norm.weight"]);
        assert_eq!(m.passthrough, ["model.embed_tokens.weight"]);
        assert_eq!(
            m.contract
                .tensors
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["model.norm.weight"],
            "a dropped tensor must not be left with a contract that names it"
        );
        // The embedding is what the tie points AT, so dropping the head must
        // never take it -- an artifact without it has no model at all.
        assert!(
            m.passthrough
                .contains(&"model.embed_tokens.weight".to_string())
        );
        assert_eq!(m.meta, ["pie.meta/x"], "metadata is not a weight");
        // Idempotent: a second sweep finds nothing, so a re-import of an
        // artifact this already cleaned reports no drops.
        assert!(drop_tied_head(&mut m, &heads).is_empty());
    }
}
