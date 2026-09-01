//! `pie model import` — rewrite any checkpoint as pie's canonical artifact,
//! fetching it first when the source is a repo ID that is not here yet.
//!
//! The conversion engine. Format diversity is handled here, once, and never
//! again: what the runtime serves is always one `.zt` file, so the question
//! "which files make up this model, and in what format" is answered at import
//! rather than at every serve boot.
//!
//! The command's contract with the user: everything it does is work the runtime
//! would do at load time anyway, done ahead of time, with results that are
//! bit-identical to a cold load. What runs is *derived*, never chosen by flag —
//! the knobs are operational (`--dry-run`, `--force`, `--delete-source`) or
//! about placement (`--out`), never about what the artifact means.
//!
//! Every checkpoint converts the same way, and **SERVE-AS-STORED is the whole
//! of the rule**: a tensor keeps the encoding it arrived in, and the one
//! rewrite this command performs is narrowing an F16 or F32 tensor to the BF16
//! every device kernel reads. GGUF's blocked schemes — every `q*_k`, `q4_0`,
//! `q8_0`, `mxfp4`, and the IQ lattices too — are copied byte for byte with
//! their scheme intact, because `.zt` names a layout profile
//! (`gguf.q4_k/1`) rather than a dtype tag, so the copy is exact and the
//! artifact can be read back as the same quantized tensor. What the format
//! then gives for free is the point of the exercise: every tensor lands on a
//! 64 KiB page of its own (what lets the engine mmap-stream routed experts),
//! carries an XXH3 digest, and records its provenance in the file.
//!
//! **THERE IS NO `--stored` FLAG, AND THERE IS NOTHING FOR ONE TO DO.** Keeping
//! a block as stored was once opt-in-shaped work — `contract::materialize`
//! decoded every self-contained scheme on the way in, because nothing
//! downstream could read a block — and the flip to keeping them landed with
//! the decode moving to the point that needs the tensor unpacked. So the
//! forward door the QNF campaign asks for is already the default and is
//! unconditional; a flag would name a choice that is not being made. What §J5
//! still needs is on the SERVING side: an engine binding that routes a
//! k-quant-schemed plane at a kernel instead of decoding it at load. The
//! artifact this command writes is ready for it, and the `qnf` attribute below
//! is the name that binding will key on.
//!
//! **EVERY TENSOR IS STAMPED WITH WHAT ITS BYTES MEAN**, when the bridge can
//! say: `checkpoint::file::write` puts the QNF spelling of the tensor's
//! encoding (`QuantSpec::term`, `dtype::Fmt`'s mangle) in a `qnf`
//! attribute beside the layout profile — `g32_u4_g8_u6_f16_n_b_g8_u6_f16_n`
//! for a Q4_K block, `bf16` for a narrowed norm. A scheme the bridge refuses
//! (the IQ lattices, whose points are compiled into llama.cpp rather than
//! stored) gets no attribute rather than a guess, and the summary counts them.
//!
//! Passthrough tensors stream from the source through a bounded buffer, so
//! converting a checkpoint far larger than memory is fine; only the narrowed
//! set is ever resident.
//!
//! **THAT LAST CLAUSE IS TRUE AGAIN, AND FOR A WHILE IT WAS NOT.** The decode
//! has always run through `Execution::streaming`, the residency that owns each
//! buffer and frees it at its last use — but the plan it ran was compiled
//! through the pass pipeline's arena half, whose `hoist-bulk-arena-writes`
//! pulls every `Allocate` to the head of the schedule so a device arena can be
//! filled in one sweep. With every allocation ahead of every publish there is
//! no last use to free at, and the whole narrowed set was resident after all.
//! `compile_decode` compiles through `plan::compile_streaming`, which leaves
//! the two arena passes out, and the sentence above went back to describing
//! what the command does.
//!
//! FAMILY-BLIND, and now entirely: it does not know or ask what model this
//! is. It had a family-aware half — an ingest pass that renamed a GGUF's
//! tensors into the vocabulary a legacy load contract would bind, and a
//! `pie model build` beside it that authored that contract — and R3 deleted
//! both with the contract. An engine produces its weights from the checkpoint
//! through the SKU's own import table at load, so a naming pass here would be
//! an earlier, worse answer to a question the load already asks.

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;

use checkpoint::file::read::{parse_attributes, parse_metadata};
use checkpoint::file::write::Writer;
use checkpoint::file::{Metadata, RawTensor};
use checkpoint::contract::materialize::{Materialization, materialize_contract};
use checkpoint::contract::{BiasBy, Expr, ScaleFactor, TensorContract};
use checkpoint::executor::Progress;
use checkpoint::executor::sink::TensorSink;
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use checkpoint::types::{CheckpointFormat, DType, Encoding, TensorDecl, Visibility};
use checkpoint::verify::ContractView;

use checkpoint::file::Attributes;
use checkpoint::file::meta::{SOURCE_ENCODING_KEY, SOURCE_KEY, VERSION_KEY, meta_name};
// The artifact's on-disk names come from whoever owns them: the loader owns
// the metadata namespace and the provenance attributes, and the object the
// checkpoint's own config lands in belongs to the party that reads it back.
// That was `models::serve::encoding`, beside a parser for the document; M18
// deleted the module and the parser did not come with it — the loader reads a
// checkpoint's quantization off its STORED encodings now — so the name lives
// with its remaining readers, in `worker::weights`. A literal here would be a
// second definition of something a reader elsewhere has to match exactly, and
// a mismatch does not fail — the read just finds nothing.
use worker::weights::CONFIG_OBJECT;

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

/// **WHERE AN OVERLAY'S TENSORS LAND** (campaign M-4).
///
/// One string, here, because two parties have to agree on it and neither can
/// derive it: this command writes it and a model text reads it
/// (`models::qwen_3::model::Recipe::Eagle`). It is a `.`-terminated prefix so
/// that the joined name is the overlay's own spelling with a namespace in
/// front of it — `fc.weight` becomes `aux.fc.weight` — which keeps the head's
/// internal names readable in `pie model info` and keeps the two sets
/// provably disjoint.
pub const AUX_PREFIX: &str = "aux.";

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
    /// **An AUX DRAFT HEAD to overlay onto this checkpoint** (campaign M-4):
    /// a repo ID, a snapshot directory, or a single weight file, whose tensors
    /// are copied into the same artifact with every name prefixed `aux.`.
    ///
    /// The prefix is the whole of the contract and it is FAMILY-BLIND, like
    /// everything else here: this command does not know or ask what a draft
    /// head is. What makes those bytes one is a model text naming them —
    /// `qwen_3`'s `Recipe::Eagle` binds `aux.fc.weight` and `aux.layers.0.*`
    /// — exactly as the base checkpoint's own names are the text's to name.
    #[arg(long, value_name = "SOURCE")]
    pub aux: Option<String>,
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
    /// Skip the weight-tier prepare that normally follows a successful import.
    ///
    /// An import ends by running the cold half of a load once, on this box's
    /// device and at this box's budgets, so that the artifact the engine reads
    /// on a warm boot is already on disk (§M wave M-1). It costs the load's
    /// own time — minutes on a large MoE — and tens of gigabytes under
    /// `[model] weight_cache_dir`.
    ///
    /// **THE CASE THAT WANTS IT: a box with a device that will not serve THIS
    /// model.** Converting a shelf of checkpoints for other machines pays that
    /// time and that disk once per model, and what it would write is a file
    /// those machines cannot read — a tier artifact's key is a function of the
    /// RECIPE, so a prepare here answers for this box's backend, tensor
    /// parallelism and precision and for no other box's.
    ///
    /// **AND "THE BUDGETS ARE NOT DECIDED YET" IS NOT ONE, NOT SINCE M-2.**
    /// This paragraph named it while the file was a function of the rungs, and
    /// the wave that made one artifact serve any budget pair took the reason
    /// away with them: an operator who prepares at 4 GiB and later serves at 8
    /// keeps the file they wrote (§M.3, and `tier_key` no longer mixes a
    /// budget in). There is nothing here to wait for.
    ///
    /// **AND KNOW WHAT YOU ARE SKIPPING** (§M wave M-3). This used to buy back
    /// some minutes at the cost of a slow first request: the boot could still
    /// build the artifact itself. It cannot. A SKU whose weights stream — any
    /// model the device budget does not hold outright — will refuse to serve
    /// until a prepare has run, so on the box that serves, this flag defers
    /// the work rather than declining it. `--prepare-only` is how it is picked
    /// back up.
    ///
    /// Accepted in every build. A binary with no engine feature, and a box
    /// with no serving config, never prepares anything and this flag has
    /// nothing to turn off — which is why a machine that ONLY converts has
    /// never needed to pass it.
    #[arg(long, conflicts_with = "prepare_only")]
    pub no_prepare: bool,
    /// **PREPARE AN ARTIFACT THAT IS ALREADY IMPORTED, AND DO NOTHING ELSE**
    /// (§M wave M-3).
    ///
    /// `SOURCE` names something already in the store — a name as
    /// `pie model list` prints it, or a path to a `.zt` or a snapshot
    /// directory — and this command skips every conversion step and runs only
    /// the last one: the cold load, on this box's device and at this box's
    /// budgets, that writes the serving artifact the engine boots from.
    ///
    /// **THIS IS THE REBUILD DOOR.** A serving boot never writes one and never
    /// deletes one, so an artifact that has rotted, that was written by an
    /// older build's format, or that a changed model text has orphaned under
    /// an old key is fixed HERE and nowhere else. Every refusal the engine
    /// prints names this exact command with this exact argument.
    ///
    /// An artifact that is already good is not rewritten: the run opens it,
    /// cuts it and verifies every image it reads, which costs a warm boot and
    /// leaves the file alone. So it is also the integrity check.
    ///
    /// Unlike the prepare that follows an ordinary import, a failure here is
    /// the command's failure — there is no other product to weigh it against.
    #[arg(long)]
    pub prepare_only: bool,
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
    metadata: &'a Metadata,
}

impl Drop for Consumed<'_> {
    fn drop(&mut self) {
        let _ = remove_source_files(self.metadata);
    }
}

pub fn run(args: ImportArgs, global: &bootstrap::GlobalArgs) -> Result<crate::ui::Answer> {
    // **THE REBUILD DOOR, AND IT IS THE FIRST LINE OF THE COMMAND** (§M wave
    // M-3). `--prepare-only` names something that is already imported, so
    // every step below it — fetching, decoding, writing, digesting — is work
    // about a file that exists. It returns before any of it. See
    // `reprepare`.
    if args.prepare_only {
        return reprepare(&args, global);
    }
    let mut source = resolve_source(&args.source)?;
    let metadata = parse_metadata(&source.path)
        .map_err(|err| anyhow!("cannot read {}: {err}", source.path.display()))?;

    // **THE OVERLAY, RESOLVED BESIDE THE BASE** (campaign M-4). A second
    // checkpoint whose tensors land in the same artifact under `AUX_PREFIX`,
    // so that one `.zt` carries a model and the draft head trained for it and
    // the loader opens ONE file. Read here, before anything is written,
    // because an unreadable head must refuse before the base is converted.
    //
    // **AND IT JOINS `origin`**, which is what `staleness` compares a stored
    // artifact against: an artifact built from a base alone and one built from
    // the same base plus a head are different artifacts, and a re-import that
    // called the second up to date would serve a model whose draft head
    // silently went away.
    let aux = match &args.aux {
        Some(spec) => {
            let overlay = resolve_source(spec)?;
            let meta = parse_metadata(&overlay.path)
                .map_err(|err| anyhow!("cannot read {}: {err}", overlay.path.display()))?;
            source.origin = format!("{} +aux {}", source.origin, overlay.origin);
            Some((overlay, meta))
        }
        None => None,
    };
    let source = source;

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
    // There is no such binding any more. An engine produces its weights from
    // the CHECKPOINT through the SKU's own import table
    // (`models::import_of`), which reads the source's own spelling and states
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
    // **THE ONE CONVERSION THAT IS NOT A NARROWING** (§M-3; §J3 closes here).
    //
    // Serve-as-stored says the stored form IS the served form, and one family
    // still asked a load to break that: kimi declares `Mxfp4` expert banks
    // over a checkpoint that ships them BF16, so every boot re-quantized a
    // hundred gigabytes before it could answer. The rule did not change and
    // the site did — this command is the transform point, so the encode runs
    // HERE, once, and the artifact holds the codes.
    //
    // **AND A REPACK RIDES THE SAME ARM** (§J4b), for the same reason one
    // step milder: a relabelling is not a conversion — the served row is the
    // stored row — but it is paid once per weight, and a load that took one
    // would rearrange a hundred gigabytes into m16n8k16 fragment order at
    // every boot. `CONVERT_TILE_MAP_MASK` is the only mask that admits an
    // `Expr::Repack`, so this command is where one runs, and the artifact
    // holds the plane in the order the kernel reads.
    //
    // **AND THE LANDING RIDES IT TOO** (§M-4a). The two arms above are the
    // ones a serving load REFUSES; the rest of what a contract states — the
    // q/k/v fusion, the band a bank is split into, the rank a stored word is
    // lifted to, the fold gemma takes back out — a load performs, and pays
    // for in fragmented reads and a copy at every boot. §M-4a's ruling is
    // that those belong here as well, so the artifact's tensors ARE the
    // planes the engine binds and the landing becomes a read. See
    // `states_an_import_transform` for the set and for the one node that is
    // never taken.
    //
    // **AND IT IS STILL FAMILY-BLIND**, which is the whole reason it is four
    // lines. Nothing below knows what kimi is: it asks the catalog which SKU
    // this source would convert as, reads that SKU's OWN contract, and takes
    // the tensors whose expression already states a transform. The Expr chain
    // is the family's sentence, not this command's; what this command
    // supplies is the place to run it. A source no SKU claims, or one whose
    // SKU states nothing but copies, adds nothing and pays a header parse.
    //
    // **AND "NO SKU CLAIMS THIS SOURCE" IS A REFUSAL NOW, NOT A SHRUG** (§M-4g).
    // The `None` this used to swallow is the whole of the paragraph above
    // failing to apply, and since §M-4a that is not a smaller import — it is a
    // different product. See `refuse_a_source_no_sku_in_this_build_claims`,
    // and see `--no-prepare` for the one path that legitimately writes one.
    let (encoded, sku) = promote_import_transforms(
        &mut materialization,
        &metadata,
        &source.path,
        !args.no_prepare,
    )?;

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
    // WHICH OF THE STORED ONES THE TREE CAN NAME (campaign J5).
    //
    // Keeping a block's bytes needs no decoder -- only a `.zt` profile to keep
    // them under, and there is one for every GGUF scheme this loader parses.
    // Being SERVED from those bytes is the other question, and its first half
    // is whether the QNF bridge can say what the arithmetic is: a scheme it
    // refuses has no spelling for a kernel table to key on, so no serving path
    // will arrive for it however faithfully the bytes were copied.
    //
    // Counted and named rather than silent, because the operator asking "did
    // the import do what I meant" for an IQ-mixed release is asking exactly
    // this. It is not a refusal: the tensor is still kept as stored, and a
    // host decode reads it at load as it always did.
    //
    // Every quantized weight here is a stored one — the only tensors this
    // command rewrites are F16 and F32 — so `weights()` filtered on the
    // encoding is exactly the set to ask about, at no extra pass and with the
    // planar schemes (an `Int8Asymmetric` whose offset nothing in this tree
    // fixes) included beside the blocks.
    let mut unnamed: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in metadata.weights() {
        if let Encoding::Quant(spec) = &tensor.encoding
            && spec.term().is_none()
        {
            *unnamed.entry(format!("{:?}", spec.scheme)).or_default() += 1;
        }
    }
    if !unnamed.is_empty() {
        let listed: Vec<String> = unnamed
            .iter()
            .map(|(scheme, count)| format!("{scheme} ×{count}"))
            .collect();
        println!(
            "convert: {} tensor(s) kept as stored carry no QNF spelling ({}); their \
             bytes are exact, but nothing keyed on QNF can name their arithmetic yet",
            unnamed.values().sum::<usize>(),
            listed.join(", "),
        );
    }
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

    // **THE OVERLAY'S TENSORS, PREFIXED** (campaign M-4). The writer already
    // separates a tensor's SOURCE address from the name it lands under — the
    // entry list is `(raw, file, output)` and has been since the merge was
    // written — so an overlay is a second set of copies with a different
    // `output`, and not a second write path.
    //
    // **COPIES ONLY, AND A NARROWING IS REFUSED BY NAME.** The base's own
    // narrowing runs through a `LoadPlan` and a decode stream; a second plan
    // over a second checkpoint would be a second decoder in a function whose
    // whole shape is one. A head is eleven small tensors and the element it is
    // read in is the element the trace declares, so the honest answer is to
    // say so at the door rather than to grow the machine.
    let mut mat_passthrough: Vec<String> = Vec::new();
    let aux_names: Vec<String> = match &aux {
        Some((overlay, meta)) => {
            let mat = materialize_contract(meta)
                .map_err(|err| anyhow!("cannot convert {}: {err}", overlay.path.display()))?;
            if !mat.decoded.is_empty() {
                bail!(
                    "the aux checkpoint {} holds {} tensor(s) this import would have to \
                     narrow on the way in ({}), and an overlay is copied byte for byte. \
                     Convert it to the element the model text declares first.",
                    crate::ui::short_path(&overlay.path),
                    mat.decoded.len(),
                    mat.decoded.join(", "),
                );
            }
            mat_passthrough = mat.passthrough.clone();
            mat_passthrough
                .iter()
                .map(|name| format!("{AUX_PREFIX}{name}"))
                .collect()
        }
        None => Vec::new(),
    };
    if let Some((overlay, meta)) = &aux {
        // A collision here is a base that already publishes the namespace the
        // overlay is being given, which would put two producers under one name
        // — the thing every other check in this file exists to forbid.
        if let Some(clash) = metadata
            .weights()
            .find(|tensor| tensor.name.starts_with(AUX_PREFIX))
        {
            bail!(
                "{} already publishes `{}`, so an overlay under `{AUX_PREFIX}` would \
                 give one name two producers",
                source.name,
                clash.name,
            );
        }
        // Zipped against the SAME list the names were built from, and not
        // against a second walk of the metadata. Two orderings of one set look
        // identical in the artifact's name list and pair every tensor with
        // somebody else's bytes; what catches it is a shape check four stages
        // later, in a refusal that names neither this command nor the overlay.
        for (name, output) in mat_passthrough.iter().zip(aux_names.iter()) {
            let raw = meta
                .tensor_by_name(name)
                .ok_or_else(|| anyhow!("'{name}' is in the overlay's list but not its files"))?;
            let file = meta
                .files
                .iter()
                .find(|file| file.id == raw.file_id)
                .ok_or_else(|| anyhow!("'{name}' points at a file the overlay lacks"))?;
            passthrough.push((raw, file.path.as_str(), output.as_str()));
        }
        println!(
            "convert: overlaying {} tensor(s) from {} under `{AUX_PREFIX}`",
            aux_names.len(),
            crate::ui::short_path(&overlay.path),
        );
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
        Some(compile_decode(&metadata, &materialization.contract)?)
    };

    // The decode's read total, taken from the checkpoint rather than from the
    // executor, so the progress denominator is whole from the first frame
    // instead of arriving with it.
    let decode_bytes: u64 = materialization
        .decoded
        .iter()
        .chain(encoded.iter())
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
            checkpoint::executor::Execution::new(plan, &source.base())
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
    // **AND THE ARTIFACT SAYS WHAT IT IS FOR** (§M-4b-3). A source some SKU in
    // this build claims produces a SERVING artifact: the same one `.zt`, with
    // one more file attribute stating the recipe it was converted for and a
    // block table per plane, folded from the payload as it streams past. A
    // source no SKU claims is an ordinary checkpoint and gets no stamp — that
    // is `--no-prepare`'s legitimate path, converting a shelf for machines
    // whose build ships the row, and `refuse_a_source_no_sku_in_this_build_claims`
    // is what makes it the only one.
    let stamp = sku
        .map(|sku| serving_stamp(sku, &source.origin))
        .transpose()?;
    // Kept, because the writer consumes it and the FILENAME states the same
    // five facts the stamp does — the owner's rule is that both say the
    // specialization, so both read one value.
    let named = stamp.clone();
    // **THE ORDER THE PLANES WILL LIE IN**, when a shell is linked to state
    // one. Derived from the trace and nothing else — see
    // `runtime::engine::load::sequence` — so it costs no plan compile and no
    // second pass over the payload. `None` writes an unranked artifact, which
    // every build reads correctly and only a streaming boot pays for.
    let ranked = sku
        .and_then(|sku| runtime::engine::load::trace(sku, runtime::engine::load::this_box()).ok())
        .and_then(|trace| runtime::engine::load::sequence(&trace));
    let mut writer = match (args.max_shard_size, stamp) {
        (Some(max), None) => Writer::create_sharded(&out_file, &provenance, max),
        (None, None) => Writer::create(&out_file, &provenance),
        (None, Some(stamp)) => Writer::create_serving(&out_file, &provenance, stamp),
        (Some(_), Some(_)) => {
            // Refused rather than silently written without a stamp. A sharded
            // serving artifact is a real product and its answer is already
            // written down — `Writer::finish_sharded` puts the key on the
            // root, which is the only manifest there is — but nothing
            // constructs one yet, and an artifact that quietly came out
            // unservable is the failure §M-4g exists to stop.
            return Err(anyhow!(
                "`--max-shard-size` and a servable artifact cannot be had together yet: \
                 this checkpoint is claimed by `{}`, so the artifact would carry a \
                 serving stamp, and the sharded writer does not take one. Drop the \
                 flag to write one file.",
                sku.unwrap_or_default(),
            ));
        }
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
        ranked.as_deref(),
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
    let out_file =
        name_the_specialization(out_file, &source.name, named.as_ref(), args.out.as_deref())?;

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
    // **AND THEN THE COLD LOAD, ONCE, HERE** (§M wave M-1). Everything above
    // is format work that needs no device; this is the one step that does, and
    // it runs last because it is the only one that can be skipped without
    // costing the artifact. See `prepare`.
    if !args.no_prepare {
        prepare(global, &out_file);
    }
    Ok(crate::ui::Answer::did(did))
}

/// **RUN THE COLD HALF OF A LOAD SO THE FIRST SERVE DOES NOT HAVE TO** (§M
/// wave M-1: `.wiki/alto/zt-as-serving-artifact.md`).
///
/// The serving artifact the engine reads on a warm boot is written by
/// whichever load materializes the weights first. Before §M that was the first
/// SERVE, which is why the first serve after an import paid the full cold load
/// — measured at 290-440 s on the 4-bit flash SKU against 21.5 s warm. Nothing
/// about that work is a serving decision: it is the same transforms, from the
/// same checkpoint. So it happens here, where the operator is already waiting,
/// and the deployment's first real request meets a warm boot.
///
/// **AND IT IS PREPARED ONCE, FOR ANY BUDGET** (§M.3, wave M-2). The budget
/// this reads out of the box's config decides which planes THIS call happens
/// to put on which tier; it does not decide what the file holds. The artifact
/// carries one image per plane in a budget-free ranking and a boot cuts it, so
/// an operator who later raises `device_weight_budget` does not go back to
/// paying the cold path on their next request.
///
/// **A FAILURE HERE IS NOT AN IMPORT FAILURE, AND THAT IS THE WHOLE POLICY.**
/// The artifact is written and verified by the time this runs; the tier file
/// is an ACCELERATOR, and a boot that does not find one runs the cold path it
/// has always run. So every refusal is printed and swallowed, and
/// `pie model import` exits zero — the alternative is a command that fails
/// after producing exactly what it was asked for.
///
/// **AND IT IS SILENT WHEN IT HAS NOTHING TO SAY.** With no engine feature
/// compiled in, this function is empty. With no serving config on the box
/// there is no device, no budget and no cache directory to prepare against,
/// and an import on a conversion-only machine should read exactly as it read
/// before this existed.
///
/// **AND CONVERSION-ONLY IS STILL A REAL BOX, AFTER §M-3 AS BEFORE IT.** A
/// build with no engine feature converts checkpoints and cannot serve them —
/// that was true when the cold path existed too, since it is the DEVICE this
/// half needs and not the artifact. Killing the cold serving path takes
/// nothing away from such a box: it produces `.zt` files for other machines to
/// serve, exits zero, and says nothing about weights it was never going to load.
///
/// **AND IT IS NO LONGER A PREREQUISITE** (§M-4d). This said the box that DOES
/// serve "must run this, or `--prepare-only` later, before a streamed SKU will
/// boot", and that stopped being true the moment a boot learned to read its
/// planes out of the artifact itself: `experts::Spill::Serving` is asked before
/// either older road, and a serving artifact answers for every plane of the
/// trace. A streamed SKU boots off what this command has already written.
///
/// What is left is worth keeping and is a different thing: this is the one
/// check that THIS BOX can serve the artifact — a device load, which no format
/// test can stand in for. It writes nothing now (`weights::resident` skips the
/// tier write for a deployment served out of its own `.zt`), so what it costs
/// is one cold load and what it buys is finding out here rather than at the
/// first serve.
#[cfg(feature = "_engine-cuda")]
fn prepare(global: &bootstrap::GlobalArgs, artifact: &Path) {
    let Some(cfg) = serving_config(global) else {
        return;
    };
    println!(
        "prepare: one cold load against the engine {} configures, to check this box \
         can serve what was just written (--no-prepare skips it, and \
         `pie model import --prepare-only` runs it later)",
        crate::ui::short_path(&bootstrap::cli_config_path(global).0)
    );
    let started = std::time::Instant::now();
    match worker::embedded_engine::prepare_weight_artifact(&cfg, artifact) {
        Ok(dir) => println!(
            "prepare: this box serves the artifact ({} in {})",
            crate::ui::short_path(&dir),
            crate::ui::duration(started.elapsed())
        ),
        // **STILL SWALLOWED HERE, AND ONLY HERE.** The import's own product is
        // written and verified by the time this runs, and a command that
        // failed after producing exactly what it was asked for is a worse
        // answer than a printed line. What the line may no longer say is "the
        // first boot will be a cold one" — there is no cold boot — so it names
        // the command that finishes the job instead.
        // **AND THE SENTENCE NO LONGER THREATENS A BOOT THAT WILL HAPPEN
        // ANYWAY.** It said a streamed SKU would not boot until a prepare
        // succeeded; since §M-4d the artifact IS the serving file and it will.
        // What a failure here means is narrower and worth saying plainly: the
        // file is fine and THIS BOX could not serve it, which is a fact about
        // the box — a device, a budget, a build without the row — and the
        // operator's next move is to read the reason, not to re-run a step.
        Err(why) => println!(
            "prepare: this box did not serve {} ({why:#}); the artifact itself is \
             written and verified — this is a fact about this machine, not about \
             the file",
            artifact.display(),
        ),
    }
}

/// The no-engine build's half: an import that reaches no device prepares
/// nothing, and says nothing about it.
#[cfg(not(feature = "_engine-cuda"))]
fn prepare(_global: &bootstrap::GlobalArgs, _artifact: &Path) {}

/// **THE SERVING CONFIG, OR `None` FOR A BOX THAT HAS NONE.**
///
/// Split out of [`prepare`] so that [`reprepare`] reads the same file the same
/// way — the tier artifact's KEY is a function of the residency plan the two
/// weight budgets decide, and two readings of one config would name two files.
/// A parse failure is said out loud and answered `None`, because an operator
/// who broke their config should hear it from the command that tried to use
/// it.
#[cfg(feature = "_engine-cuda")]
fn serving_config(global: &bootstrap::GlobalArgs) -> Option<worker::Config> {
    let (config_path, _) = bootstrap::cli_config_path(global);
    // No config is the conversion-only box: nothing to prepare against, and
    // nothing worth saying about it.
    let text = std::fs::read_to_string(&config_path).ok()?;
    match worker::Config::parse(&text) {
        Ok(cfg) => Some(cfg),
        Err(why) => {
            println!("prepare: skipped — {} does not parse: {why}", config_path.display());
            None
        }
    }
}

/// **`pie model import --prepare-only <artifact-or-name>` — THE ONE COMMAND
/// THAT REBUILDS A SERVING ARTIFACT** (§M wave M-3).
///
/// # Why this shape, and not the two that were considered
///
/// The wave had to put the verify-then-replace behind an explicit command
/// before it could kill the cold boot, and there were three candidate shapes.
///
/// **"Just re-run `pie model import <original source>`"** was rejected. It
/// works, and it is what an operator would try — but it pays the whole
/// conversion to reach the one step that was wanted, which on a large MoE is
/// tens of minutes of decode and a second copy of the source on disk, and it
/// needs the ORIGINAL source, which §M.6's own premise says may be gone. The
/// thing that rotted is the tiers file; the `.zt` beside it is intact and is a
/// perfectly good checkpoint to land from. Making the operator re-derive it is
/// asking for a receipt they were never told to keep.
///
/// **A separate verb — `pie model prepare`** — was rejected for now, though
/// §M.6 flags it as where the naming may end up. It would be a second
/// top-level command whose entire body is this one's, and the argument it
/// takes is the same argument, and the config it reads is the same config. One
/// flag on the verb that already owns "make this model servable" is a smaller
/// surface than a second verb that owns half of it.
///
/// **A flag on import it is**, and the deciding property is the message: every
/// engine-side refusal prints `tier::rebuild`'s line, and that line has to be
/// something an operator can paste. It names the CHECKPOINT PATH the refusing
/// load was pointed at — `boot.checkpoint`, which is `[model] model` resolved
/// — and this command's argument is resolved by
/// [`worker::weights::resolve`], the same door `[model] model` goes through.
/// So the pasted line resolves to the same file the engine refused about, by
/// construction rather than by an operator's care.
///
/// # What it does
///
/// It resolves the argument, reads the box's serving config, and calls
/// [`worker::embedded_engine::prepare_weight_artifact`] — the same call an
/// ordinary import ends with, against the same config, so the key is the same
/// key. `Weights::resident` under `Intent::Prepare` then does the rest: an
/// artifact that opens, cuts and verifies is left alone and the run is a warm
/// boot's worth of checking, and anything else is said out loud and REPLACED
/// by `tier::store`'s verify-then-replace, which is the authority this wave
/// moved out of the boot.
///
/// # And a failure here IS a failure
///
/// [`prepare`] swallows its refusals because the import it follows has already
/// produced the artifact it was asked for. This command has no such product:
/// preparing is the whole of what it was asked to do, so a box with no serving
/// config, an engine-less build, an unresolvable name and a refused landing
/// all exit non-zero with their own sentence.
///
/// # Errors
///
/// A `SOURCE` that names nothing on this machine, a box with no serving config
/// or one that will not parse, a build with no engine feature, or whatever the
/// bake and the landing refused.
#[cfg(feature = "_engine-cuda")]
fn reprepare(args: &ImportArgs, global: &bootstrap::GlobalArgs) -> Result<crate::ui::Answer> {
    // `{why:#}` and not `with_context`: the CLI renders one line, so a context
    // that pushed the resolver's own sentence — the one naming the store and
    // `pie model list` — out of it would be a worse message than no context.
    let model = worker::weights::resolve(&args.source)
        .map_err(|why| anyhow!("--prepare-only {}: {why:#}", args.source))?;
    let artifact = model.path().to_path_buf();
    let Some(cfg) = serving_config(global) else {
        bail!(
            "--prepare-only writes the serving artifact this box's engine boots from, \
             and this box states no serving config to read a device, a budget and a \
             weight cache directory out of. Run it where the model will be served."
        );
    };
    println!(
        "prepare: {} — one cold load against the engine {} configures. An artifact \
         that is already good is verified and left alone; a rotted, stale or absent \
         one is written again.",
        crate::ui::short_path(&artifact),
        crate::ui::short_path(&bootstrap::cli_config_path(global).0),
    );
    let started = std::time::Instant::now();
    let dir = worker::embedded_engine::prepare_weight_artifact(&cfg, &artifact)
        .map_err(|why| anyhow!("preparing {}: {why:#}", artifact.display()))?;
    Ok(crate::ui::Answer::did(format!(
        "prepared {} — weight tiers under {} in {}",
        crate::ui::short_path(&artifact),
        crate::ui::short_path(&dir),
        crate::ui::duration(started.elapsed()),
    )))
}

/// The no-engine build's half: there is no device to land on and no artifact
/// format to write, so the flag refuses rather than exiting zero on a job it
/// did not do.
#[cfg(not(feature = "_engine-cuda"))]
fn reprepare(
    _args: &ImportArgs,
    _global: &bootstrap::GlobalArgs,
) -> Result<crate::ui::Answer> {
    bail!(
        "this build has no engine compiled in, so it cannot write a serving artifact. \
         Conversion still works — `pie model import <source>` without --prepare-only — \
         and the box that serves the model is where the prepare belongs."
    )
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

/// The decode, compiled and checked, before a byte is read.
///
/// **`compile_streaming`, BECAUSE THAT IS HOW IT RUNS.** Both places that
/// execute this plan — the spool in `run` and `decode_into` — run it through
/// `Execution::streaming`, the residency that owns each buffer and frees it at
/// its last use. The ordinary pipeline ends its rewrites with
/// `hoist-bulk-arena-writes`, which pulls every `Allocate` to the head of the
/// schedule so a device arena can be filled in one sweep; it does that whether
/// or not there is a bulk write to hoist, so an Unknown-backend plan with none
/// got the reordering and nothing else. Under it there is no last use to free
/// at, and an import of a checkpoint far larger than memory held the whole
/// narrowed set at once — the one thing the module doc says it does not do.
/// `plan::compile_streaming` is the same contract, the same target and the
/// same bytes, minus the two passes that exist to serve an arena this
/// execution never allocates.
///
/// **A SECOND OPINION.** `verify` is not a second compiler: it takes the plan
/// as it stands and asks what can be answered from the plan plus the
/// filesystem — is the schedule a permutation of the instructions, is every
/// public declaration finalized exactly once, does every read land inside the
/// file it names at the size that file actually is, and does the result
/// deliver the contract this was compiled from. `compile` cannot catch those
/// itself, because the same wrong belief would have produced both halves.
///
/// It ran only in the checkpoint crate's own golden tests, over sixteen
/// compiled plans, while the command a user actually runs skipped it. There is
/// no reason for that asymmetry: an import reads gigabytes and writes an
/// artifact other machines will serve, so it is the caller with the most to
/// gain from finding out here rather than at load.
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

// `ingest_map`, `apply_ingest`, `unpermute` and `report_servability` STOOD
// HERE — the family-aware half of this command. The first three applied
// `model_legacy::ingest`'s per-family naming pass to the materialization; the
// fourth held the projected artifact against the legacy catalog and printed
// which row it would identify as. R3 deleted the catalog and the contract
// that read those names, so all four are gone: see the note in `run`.

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

/// Move the SKU's encode chains into the rewritten set, and answer with the
/// source legs they consume.
///
/// The one place this command reads a model family, and it reads it as an
/// EXPRESSION and never as a name: a tensor whose chain holds a
/// [`Expr::Cast`] into a quantized encoding is a tensor the checkpoint ships
/// raw and the SKU serves packed, which is a conversion, which is this
/// command's job. The legs that chain names stop being byte copies — they are
/// what it reads — so they leave the passthrough set and their bytes join the
/// decode's read total.
///
/// The plan publishes the scales plane beside the codes on its own
/// (`plan::build`'s encode outputs), so nothing here has to know that a
/// packed bank is two tensors.
///
/// Answers an empty list for every checkpoint whose SKU states no encode,
/// which is every SKU but one and is the resting state this whole file was
/// written for.
fn promote_import_transforms(
    materialization: &mut Materialization,
    metadata: &Metadata,
    checkpoint: &Path,
    will_prepare: bool,
) -> Result<(Vec<String>, Option<&'static str>)> {
    // **CONVERTED FOR THIS BOX** (§M, §J4c). The chains this promotes are
    // paid once and written into the artifact, and one of them — a repack —
    // is an ARRANGEMENT of a bank's bytes rather than a change to them, legal
    // only for the shell whose kernels read that order. So the contract is
    // read for the setup this command is converting for, which is the box it
    // is running on: the same box the prepare below answers for.
    let platform = runtime::engine::load::this_box();
    let Some((sku, contract)) = runtime::engine::load::conversion_contract(checkpoint, platform)
    else {
        refuse_a_source_no_sku_in_this_build_claims(checkpoint, will_prepare)?;
        return Ok((Vec::new(), None));
    };
    let mut promoted: Vec<TensorContract> = Vec::new();
    let mut kept: Vec<TensorContract> = Vec::new();
    for tensor in contract.tensors {
        if states_an_import_transform(&tensor.expr, metadata) {
            promoted.push(tensor);
        } else {
            kept.push(tensor);
        }
    }
    close_over_the_declared_pairings(&mut promoted, &mut kept);
    keep_the_width_a_surviving_read_declares(sku, materialization, &kept);
    if promoted.is_empty() {
        return Ok((Vec::new(), Some(sku)));
    }
    let mut legs: BTreeSet<String> = BTreeSet::new();
    for tensor in &promoted {
        refuse_a_chain_this_command_cannot_close(sku, tensor)?;
        for leg in tensor.expr.sources() {
            legs.insert(leg.to_string());
        }
    }
    let consumed: Vec<String> = legs.into_iter().collect();
    refuse_a_leg_a_surviving_read_still_needs(sku, &kept, &consumed)?;
    // THREE SETS, AND THE THIRD IS THE ONE THAT WRITES.
    //
    // `passthrough` and `decoded` say which SOURCE names the artifact will
    // hold, and dropping a consumed leg from them is what makes a promotion
    // mean "read here instead of at every boot". But neither set puts a byte
    // on disk: `contract.tensors` does, through `compile_decode` and
    // `write_artifact`'s public decls — and `materialize_contract` has
    // already put an entry there, UNDER THE SOURCE'S OWN NAME, for every F16
    // or F32 tensor it narrows to BF16. A promotion that consumed such a leg
    // left that entry standing, so the artifact held the same values twice:
    // once under the source's name, narrowed, and once under the contract's,
    // transformed. Only the second is a plane the SKU asks for.
    //
    // The predicate is the LEG'S name, not the promoted one — what is dropped
    // is the tensor whose source name is now produced under a contract name,
    // which is exactly what `consumed` lists. And it runs BEFORE the promoted
    // tensors are appended, which is what keeps the identity case honest: a
    // chain whose output name IS its source's — the raw-to-raw cast M-4a
    // widens to — would otherwise retain itself straight back out.
    materialization
        .passthrough
        .retain(|name| !consumed.contains(name));
    materialization
        .decoded
        .retain(|name| !consumed.contains(name));
    materialization
        .contract
        .tensors
        .retain(|tensor| !consumed.contains(&tensor.name));
    refuse_a_name_two_parties_claim(sku, materialization, &promoted)?;
    let banks = promoted.len();
    materialization.contract.tensors.extend(promoted);
    println!(
        "convert: {sku} states {banks} tensor(s) whose form or layout the checkpoint does not \
         hold, so {} source tensor(s) are transformed here instead of at every boot",
        consumed.len(),
    );
    // **THE SKU TRAVELS BACK**, because this is where it was decided. The
    // conversion contract is chosen here — for this box, from this checkpoint
    // — and `Stamp::sku` has to be the SAME answer: an artifact whose stamp
    // named one contract while its planes were compiled under another would
    // be checked against the wrong deployment and pass.
    Ok((consumed, Some(sku)))
}

/// The stamp this box would write for `sku`.
///
/// Everything that is a POLICY — the layout revision, the block size, the
/// digest algorithm, the zeroed adapters — comes from
/// [`serving::Stamp::of`](checkpoint::serving::Stamp::of) and is not spelled
/// here, because a boot compares field by field and a constant spelled twice
/// is a field that can disagree with itself. What this function supplies is
/// the five facts that are about THIS conversion:
///
/// * `backend` is `this_box()`, the SAME call [`promote_import_transforms`]
///   makes to pick the conversion contract. The artifact must say the recipe
///   its planes were compiled under, and two calls that could disagree would
///   be two answers to one question.
/// * `tp_size` is 1 because an import states the WHOLE checkpoint, which
///   `Builder::whole_checkpoint` refuses to do otherwise, by name.
/// * `sku` is the row `conversion_contract` chose, travelled back from where
///   it was decided.
/// * `precision` is a stated catalog fact and deliberately not read off the
///   SKU name — the two DQ rows are named for their two-bit experts and are
///   mostly four-bit. It ERRORS rather than defaulting: `Stamp::check`
///   compares this field, so a placeholder would either refuse a good
///   artifact or pass a bad one.
/// * `model_id` is what the operator named, the same string `pie_source`
///   records, so the two can never disagree. It is BELIEVED and never
///   compared, which is why `serving::LAYOUT_REVISION` sits beside it:
///   `file/meta.rs`'s first ruling is that a believed identity is safe only
///   when paired with a revision.
fn serving_stamp(sku: &str, origin: &str) -> Result<checkpoint::serving::Stamp> {
    let platform = runtime::engine::load::this_box();
    Ok(checkpoint::serving::Stamp::of(
        &format!("{platform:?}").to_lowercase(),
        1,
        sku,
        runtime::engine::load::precision(sku)?,
        (!origin.is_empty()).then(|| origin.to_string()),
    ))
}

/// **THE ARTIFACT'S NAME STATES ITS SPECIALIZATION**, which is the other half
/// of the owner's §M-4 ruling — *"생성된 .zt 파일의 메타데이터와 파일명에는 이
/// specialization에 대해서 기술하도록"* — the metadata half being the stamp.
///
/// The reason is coexistence: one model at two quantizations, or converted for
/// two shells, is two artifacts that must be able to sit in one directory. A
/// single `archive.zt` per store entry cannot hold two, so the second import
/// would silently replace the first and an operator would discover it by
/// serving the wrong one.
///
/// [`serving::Name`] renders `<slug>.<sku>.<backend>-tp<n>.<precision>.zt`.
/// Four of the five are the stamp's own fields, read from it rather than
/// recomputed, so the name and the metadata cannot disagree about the
/// specialization.
///
/// The SLUG is the store's name for the model and NOT `Stamp::model_id`,
/// which is what the operator typed. Those are the same string for a repo-id
/// import and very different for a directory one — `model_id` is then an
/// absolute path, and slugging it produced
/// `root---cache--huggingface--hub--models--…--snapshots--da28692b…` on the
/// first run of this. The store name is what every other pie surface already
/// calls the model, so it is what a filename should say, and
/// [`Name::of`](checkpoint::serving::Name::of) takes it as the one argument
/// the stamp cannot supply.
///
/// **AND THE NAME IS BUILT BY `Name::of`, NOT FIELD BY FIELD.** It slugs every
/// field, which matters for exactly one of them: eight catalog SKUs hold a dot
/// (`qwen35-d0.8b-mlxu4-kv-bf16`), and `.` is the separator `Name::parse`
/// splits on. A name assembled by hand out of the stamp's raw fields renders
/// something `Name::parse` cannot read back — I built one that way first and
/// the round trip is what said so. The filename therefore carries
/// `qwen35-d0-8b-…` where the stamp carries `qwen35-d0.8b-…`: the name is for
/// a human at a directory listing, and the stamp is what a boot compares.
///
/// # It renames rather than choosing the name up front
///
/// The name is a function of the SKU, and the sku is not known until
/// `promote_import_transforms` has picked a conversion contract — which is
/// well after `out_file` has to exist, because the same-file check, the
/// staleness comparison and the destination's own directory all read it. So
/// the file is written where it always was and moved once at the end, which
/// costs one `rename` inside one directory and leaves every check above
/// reading the path it was written to expect.
///
/// # What it does NOT rename
///
/// **An `--out` that names a file.** An operator who wrote
/// `--out /srv/models/mine.zt` named it, and a command that renamed their
/// file underneath them would be answering a question they did not ask. The
/// stamp still states the specialization; the filename is theirs.
///
/// **An artifact with no stamp.** A source no SKU claims has no
/// specialization to state — no sku, no precision — and `Name` has nothing to
/// render. It keeps the store's own name, which is what `--no-prepare`'s
/// shelf-conversion path has always produced.
fn name_the_specialization(
    written: PathBuf,
    slug: &str,
    stamp: Option<&checkpoint::serving::Stamp>,
    out: Option<&Path>,
) -> Result<PathBuf> {
    let Some(stamp) = stamp else {
        return Ok(written);
    };
    if out.is_some_and(|out| {
        out.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zt"))
    }) {
        return Ok(written);
    }
    let Some(directory) = written.parent() else {
        return Ok(written);
    };
    let Some(name) = checkpoint::serving::Name::of(stamp, Some(slug)) else {
        return Ok(written);
    };
    let renamed = directory.join(name.render());
    if renamed == written {
        return Ok(written);
    }
    std::fs::rename(&written, &renamed).map_err(|why| {
        anyhow!(
            "cannot name the artifact {}: {why}",
            renamed.display()
        )
    })?;
    Ok(renamed)
}

/// **REFUSE A SOURCE NO SKU IN THIS BUILD CLAIMS** (§M-4g).
///
/// `conversion_contract` answers `None` when no catalog row's import contract
/// both builds over this checkpoint and compiles against
/// [`CONVERT_TILE_MAP_MASK`], and the caller used to read that as "nothing to
/// promote" and carry on. Before §M-4a that reading was right: promotion was
/// `Cast{Quant}` and `Repack` only, a source whose SKU stated neither added
/// nothing, and one no SKU claimed at all was the same case one step further
/// out. §M-4a took the reading away. The import runs the SKU's OWN contract to
/// produce the planes a serving load binds, so a source no SKU claims cannot
/// produce a servable artifact — and the command still exited zero having
/// written one, with the operator's only evidence a missing line of output.
///
/// **AND `CONVERT_TILE_MAP_MASK` IS THE WIDER DOOR, WHICH IS WHY THIS IS THE
/// PLACE TO ASK.** It is `HOST | ENCODE | DECODE | REPACK` and every device
/// mask is a subset, so a contract that will not compile here will not compile
/// for a device either: `None` from the conversion door is not "not yet" but
/// "not in this build", and `runtime::engine::load::identify` would refuse the
/// artifact for the same reason a serve boot will.
///
/// # Why it is conditional, and on what
///
/// **THE RUN THAT WILL PREPARE IS THE RUN THAT OWES THE ANSWER.** An ordinary
/// import ends by running the cold half of a load on this box's device so the
/// first serve is a warm one — and [`prepare`]'s standing policy is to PRINT
/// its refusals and swallow them, deliberately, because the artifact is
/// written and verified by then and a tier file is an accelerator. So
/// `prepare` is structurally unable to be the party that says "nothing will
/// ever serve this": it is the wrong side of the write. The question therefore
/// gets asked here, before a byte is copied, by the run that was going to
/// prepare.
///
/// **AND `--no-prepare` IS THE PATH THAT LEGITIMATELY CONVERTS ONE.** Its own
/// doc names the case: *a box with a device that will not serve THIS model —
/// converting a shelf of checkpoints for other machines.* A build's catalog is
/// not every build's catalog, and a `.zt` written for a machine whose pie
/// ships the row is a real product, so that path keeps converting and says
/// nothing. `tests/model_artifact.rs`'s spool fixture is exactly this shape —
/// two layers of 64 hidden over a 128-token vocabulary, `no_prepare: true`,
/// and an artifact it parses rather than serves — and it stays green by that
/// property and not by luck.
///
/// **AND THE DEFERRAL STAYS HONEST.** `--no-prepare` also covers "prepare
/// later on this box", and later is `pie model import --prepare-only`, where
/// [`reprepare`] propagates the landing's failure instead of swallowing it.
/// So every path that ends in serving asks this question exactly once, and
/// asks it where a refusal is the command's own answer.
///
/// # What it does not change
///
/// `--out` picks a destination and not a product — a `[model] model` key
/// takes a path as readily as a store name — so it is not the discriminator
/// and is not consulted. The `.zt`-to-`.zt` no-op and `--prepare-only` both
/// return from [`run`] above this line and never reach it. `--dry-run` does
/// reach it, and refuses: a dry run reports what a real run would do, a real
/// run would refuse, and the tokenizer compile three paragraphs down has
/// bailed on a dry run since it was written.
fn refuse_a_source_no_sku_in_this_build_claims(
    checkpoint: &Path,
    will_prepare: bool,
) -> Result<()> {
    if !will_prepare {
        return Ok(());
    }
    bail!(
        "{}: no SKU this build ships claims this checkpoint, so nothing here can \
         say what its planes are. Since the import performs a SKU's whole landing, \
         the artifact would hold the source's own tensors under the source's own \
         names — a file that converts, verifies and opens, and that no serve boot \
         and no `--prepare-only` on any box with this catalog can load. \
         `pie model list` prints what a checkpoint identifies as. If this box is \
         converting for a machine whose build DOES ship the row, say so with \
         `--no-prepare` and the conversion runs.",
        crate::ui::short_path(checkpoint),
    )
}

/// Refuse a promoted chain that reads ANOTHER ENTRY of the same contract.
///
/// The leg walk above is [`Expr::sources`], which sees [`Expr::Src`] and
/// nothing else. A chain may equally read [`Expr::Out`] — an earlier entry of
/// the contract it belongs to, the DAG edge `sources` deliberately does not
/// follow — and such a leg is not a source tensor, so no retain can move it
/// and no promotion carries it. What arrives in `materialization.contract` is
/// then a lone consumer naming a producer that set does not contain, which
/// `infer` refuses when `compile_decode` resolves the expression. The import
/// does fail — but on an unresolved output, in a sentence that names neither
/// this command nor the promotion that caused it.
///
/// **Refused rather than repaired, and that is the honest answer here.** The
/// repair is to pull the producer into the promoted set with its consumer,
/// which means promoting an entry this gate never admitted, deciding what its
/// name in the artifact is, and ordering the two — machinery for a shape
/// nothing in this tree can exercise. The only import verbs that state an
/// `Expr::Out` are `Builder::read_dequant` and `read_dequant_concat`, both
/// with zero callers.
///
/// **§M-4a WIDENED THE GATE PAST THOSE VERBS' NODES AND THIS STAYED
/// UNREACHABLE**, which is worth saying because the old sentence here
/// predicted the opposite: it named a per-block `Scale` or `Bias` as what
/// would make one reachable, and both are now admitted. What keeps them out
/// is the OTHER rule — `dequant_planes` states `Scale`/`Bias` over codes, so
/// [`decodes_a_packed_plane`] refuses the chain before this is asked. The two
/// refusals overlap on purpose: one is about a leg this command cannot carry,
/// the other about bytes it must not write, and a chain can want either.
fn refuse_a_chain_this_command_cannot_close(sku: &str, tensor: &TensorContract) -> Result<()> {
    let Some(producer) = tensor.expr.outputs().first().copied() else {
        return Ok(());
    };
    bail!(
        "{sku}: `{name}` states a transform this command runs, but its chain reads \
         `{producer}` — another tensor of the same contract — through `Expr::Out`. \
         Promoting `{name}` would write it into the artifact and leave `{producer}` \
         behind, and this command has no way to carry a producer across with its \
         consumer. Give `{name}` a chain that reads the checkpoint's own tensors, or \
         teach the promoter to take a producer too.",
        name = tensor.name,
    )
}

/// Refuse a promoted name the artifact already holds under another claim.
///
/// A promoted tensor enters the artifact under the CONTRACT'S name; every
/// other tensor enters under the SOURCE'S. Those are two namespaces, and
/// while the gate admitted only forms a checkpoint does not ship they could
/// not meet — a name the SKU invents for a repacked or encoded plane is not a
/// name the source also has. M-4a widens the gate to transforms whose input
/// and output names can coincide, and the artifact is one flat name table:
/// two entries under one name is `write_artifact`'s ascending merge writing a
/// name twice, with no party to say which bytes are the tensor.
///
/// Checked against `passthrough` and against the surviving contract entries,
/// which between them are every weight the write will emit — `decoded` is the
/// second set's shadow, listing the same names `materialize_contract` gave
/// entries to, so checking it as well would only find the same collision
/// under a less exact name. The check runs AFTER the retains, because a name
/// the promotion has just consumed is not a rival claim: it is the same
/// tensor, arriving under the name the SKU serves it as.
///
/// It cannot fire on any checkpoint this tree imports today. That is the
/// reason to write it now rather than a reason not to: the check is three
/// comparisons, and the widening that makes it reachable should arrive to
/// find it already standing.
fn refuse_a_name_two_parties_claim(
    sku: &str,
    materialization: &Materialization,
    promoted: &[TensorContract],
) -> Result<()> {
    for tensor in promoted {
        let held = if materialization.passthrough.contains(&tensor.name) {
            "copied through from the source byte for byte"
        } else if materialization
            .contract
            .tensors
            .iter()
            .any(|held| held.name == tensor.name)
        {
            "narrowed to bf16 from the source's own F16 or F32"
        } else {
            continue;
        };
        bail!(
            "{sku}: `{name}` is the name this contract gives a plane it transforms at \
             import, and the artifact already holds a tensor called `{name}` — {held}. \
             One name, two claimants, and an artifact entry written twice; the \
             transformed plane needs a name of its own.",
            name = tensor.name,
        );
    }
    Ok(())
}

/// Spares from the narrowing every plane a SURVIVING READ declares at a raw
/// dtype that is not BF16 — the width the model text asked for, kept.
///
/// [`materialize_contract`] narrows every `Raw(F16)` and `Raw(F32)` object to
/// BF16, and it decides that from the SOURCE ENCODING ALONE — it lives in the
/// loader, it names no family convention, and that is the property that lets
/// it live there at all. Its justification, though, is a claim about the
/// CONSUMER: *every device kernel pie ships reads BF16*. That claim was true
/// when it was written and this tree now falsifies it in its own source —
/// `kernels_cuda::attn::ssm` opens with
/// `debug_assert_eq!(a_log.dtype, Dtype::F32, "reads an f32 decay bank")`, and
/// four model texts declare thirteen weights `Dtype::F32` for that reason.
///
/// So the narrowing was overriding the contract, silently, and in two ways
/// that are worth separating because only one of them is arithmetic:
///
/// * **It moved a width the text had settled.** Qwen3.5-0.8B stores exactly
///   36 F32 objects — 18 `linear_attn.A_log` and 18 `linear_attn.norm.weight`
///   — which are exactly the two weights `qwen_3` declares `Dtype::F32`.
///   Source width and declared width AGREED, so the read was an identity and
///   nothing was cast at load. Narrowing them manufactured a disagreement that
///   did not exist, and the load grew a widening `Cast(Src, F32)` per plane:
///   36 planes that could be bound where they lie, landing through a residue
///   path at every boot forever, to undo a conversion nobody asked for.
/// * **And on one of the two it dropped bits.** Measured over all 18 layers:
///   `A_log` is bf16-exact in 18 of 18 planes — an F32 container holding BF16
///   values, so narrowing it is a lossless re-spelling — while
///   `linear_attn.norm.weight` is bf16-exact in 0 of 18, worst relative error
///   3.9e-3. That is the module doc's own invariant broken in words it chose
///   itself: *it drops exactly the bits the runtime drops at load, so the
///   artifact serves what a cold load would have served.* A cold load serves
///   this norm at F32, because the text and the kernel both say F32.
///
/// The fix belongs HERE and not in the loader, for the reason the loader's
/// blindness is a feature: this is the first point that holds both the
/// checkpoint and the SKU's contract, so it is the first point that can ask
/// what the plane is FOR. `kept` is the right set to ask — a `promoted` entry
/// writes its own plane at its own declared encoding and its legs are dropped
/// from `decoded` below regardless.
///
/// Spared planes move to `passthrough`, which copies them byte for byte at the
/// source's width, and their narrowing entry is dropped from the contract that
/// writes — the same three-set discipline the retains below keep, for the same
/// reason: a plane left in both sets is an artifact holding it twice.
///
/// A leg two entries read at two different widths is spared by the wider
/// claim, which is the answer that costs a cast rather than bits.
fn keep_the_width_a_surviving_read_declares(
    sku: &str,
    materialization: &mut Materialization,
    kept: &[TensorContract],
) {
    let mut spared: BTreeSet<String> = BTreeSet::new();
    for tensor in kept {
        let Encoding::Raw(dtype) = &tensor.encoding else {
            continue;
        };
        if *dtype == DType::Bf16 {
            continue;
        }
        for leg in tensor.expr.sources() {
            if materialization.decoded.iter().any(|name| name == leg) {
                spared.insert(leg.to_string());
            }
        }
    }
    if spared.is_empty() {
        return;
    }
    materialization.decoded.retain(|name| !spared.contains(name));
    materialization
        .contract
        .tensors
        .retain(|tensor| !spared.contains(&tensor.name));
    materialization.passthrough.extend(spared.iter().cloned());
    materialization.passthrough.sort();
    println!(
        "convert: {sku} declares {} plane(s) at a raw dtype of its own, so they are copied at \
         the source's width instead of narrowed to bf16",
        spared.len(),
    );
}

/// Does this chain state a transform this command should run ONCE, so that a
/// serving load can read the plane instead of building it?
///
/// **THE SET IS NO LONGER TWO NODES, AND THE REASON IT WIDENED IS NOT COST
/// ALONE** (§M-4a). `Cast { to: Quant }` and `Repack` were here because a
/// device mask REFUSES them — `CUDA_TILE_MAP_MASK` is `CAST|SCALE|DECODE|BIAS`
/// and `CONVERT_TILE_MAP_MASK` is the only one carrying `ENCODE` or `REPACK` —
/// so a load that met one could not proceed at all, and running it here is
/// what makes the refusal survivable. Every other node in the vocabulary is
/// AFFINE ([`Expr::is_affine`]) and never FAILS at load; it costs fragmented
/// I/O and a copy, at every boot, forever. §M-4a's ruling is that the artifact
/// should hold the LANDED plane and not the checkpoint's spelling of it, and
/// that makes the whole landing this command's work:
///
/// * the placement family — [`Expr::Slice`], [`Expr::Stride`],
///   [`Expr::Gather`], [`Expr::Concat`] — which is where a q/k/v fusion, a
///   gate/up interleave and a GGUF's fused-bank split live. A fusion read at
///   load is three spans copied into one buffer; written here it is one
///   contiguous image the engine binds where it lies;
/// * [`Expr::Transmute`], the rename that moves nothing — a stack of expert
///   slabs gaining the axis it is stacked along, MLX's four-bit codes lifted
///   out of the `u32` words they were packed into. It costs no arithmetic and
///   it still costs a name: written here, the plane the SKU asks for is the
///   plane the file holds;
/// * [`Expr::Scale`] and [`Expr::Bias`] over a plane the file stores RAW —
///   gemma's `+1` norm fold and mlx_lm's, taken back out. Arithmetic somebody
///   asked for, a function of the weight alone, and the alternative is an
///   engine rewriting every norm in the stack at every boot. Over a plane the
///   file stores PACKED the same two nodes are a decode, which is the rule
///   below;
/// * [`Expr::Cast`] between two RAW dtypes, which is the F16/F32 narrowing
///   `materialize_contract` performs under the SOURCE's name. Promoting it
///   moves the same bytes under the CONTRACT's name, which is why the caller
///   retains the consumed leg out of `contract.tensors` as well as out of the
///   two name sets — see the three retains below.
///
/// **AND THE ONE HARD RULE: A DECODE IS NEVER PROMOTED.** A
/// [`Expr::Cast`] out of a quantized encoding lands codes as values, and so
/// does a per-block [`Expr::Scale`] or [`Expr::Bias`] — MLX's affine
/// dequantization is exactly that pair. Promoting one would write bf16 where
/// the artifact must hold codes, which is serve-as-stored undone: the whole
/// point of keeping a packed bank packed is that the bytes on disk are the
/// bytes the kernel reads. The old predicate said this by ENCODING A
/// DIRECTION in its pattern — `Cast { to: Encoding::Quant(_) }` admits the
/// encode and cannot match the decode — and a widened `matches!` loses that
/// by accident, because a decode and a narrowing are the same node with a
/// different operand. So the arm is named, explicit and asked FIRST, and it
/// reads the operand's encoding off the checkpoint's own header rather than
/// off the node: see [`decodes_a_packed_plane`].
///
/// **AND [`Expr::Shard`] IS NOT PROMOTED IN THIS COMMIT**, nor is any chain
/// that carries one. Rank belongs to the format wave (§M-4b/M-4e): a rank's
/// band is an INDEX ENTRY's business, keyed `(plane, rank)`, and there is no
/// entry to put it in until the container takes the placement table. At
/// `tp = 1` — the only degree this tree can build, asserted by
/// `Builder::whole_checkpoint` and filtered for by `conversion_contract` —
/// a shard IS the identity, so refusing to promote one costs exactly nothing
/// and leaves the wave that must decide the layout free to decide it.
/// [`Expr::Select`] and [`Expr::SrcIndexed`] are refused for the same shape
/// of reason one level over: their meaning depends on a GROUP INSTANCE, and
/// promoting one would bake instance zero.
fn states_an_import_transform(expr: &Expr, checkpoint: &Metadata) -> bool {
    if decodes_a_packed_plane(expr, checkpoint) {
        return false;
    }
    let mut admitted = false;
    let mut deferred = false;
    expr.visit(&mut |node| match node {
        Expr::Src(_) | Expr::Out(_) | Expr::Fill { .. } => {}
        Expr::Shard { .. } | Expr::Select { .. } | Expr::SrcIndexed(_) => deferred = true,
        Expr::Slice { .. }
        | Expr::Stride { .. }
        | Expr::Gather { .. }
        | Expr::Concat { .. }
        | Expr::Transmute { .. }
        | Expr::Repack { .. }
        | Expr::Cast { .. }
        | Expr::Scale { .. }
        | Expr::Bias { .. } => admitted = true,
    });
    admitted && !deferred
}

/// Does this chain DECODE — land, anywhere in it, a value the checkpoint holds
/// packed into a form that is not packed any more?
///
/// The question [`states_an_import_transform`] asks first, and it is asked of
/// the whole chain at every depth rather than of its root: a promotion moves
/// the ENTIRE expression into the import, so one decode buried under two
/// concats and a slice is a decode this command would run.
///
/// Three nodes can do it, and all three are the same node in two directions:
/// a [`Expr::Cast`] whose `to` is raw, and a per-block [`Expr::Scale`] or
/// [`Expr::Bias`] — `code · scale + zero`, MLX's affine decode, which
/// `Builder::read_dequant` states and whose own doc calls it a decode in
/// plain words. Each is a decode exactly when its OPERAND is quantized, which
/// no field of the node says: `Cast { to: Raw(Bf16) }` over a bf16-stored
/// tensor is the narrowing every import already performs, and the same node
/// over an mxfp4-stored one is the thing that must never happen here.
///
/// **SO THE OPERAND IS TYPED, AND UNTYPEABLE MEANS NO.** [`yields`] reads the
/// encoding off the nodes that state one and off the checkpoint's header at
/// the leaves, and answers `None` where the chain does not say — an
/// [`Expr::Out`] leg, a name this checkpoint does not hold. Both answers that
/// are not "raw" refuse the promotion, because the cost of being wrong is
/// asymmetric: a promotion not taken leaves a transform at load, where it
/// already lives and where it works; a decode taken writes an artifact that
/// holds values where its own contract says codes, and nothing downstream is
/// in a position to notice.
fn decodes_a_packed_plane(expr: &Expr, checkpoint: &Metadata) -> bool {
    let mut decodes = false;
    expr.visit(&mut |node| {
        let operand = match node {
            Expr::Cast { src, to: Encoding::Raw(_) } => src,
            Expr::Scale { src, factor: ScaleFactor::PerBlock { .. } } => src,
            Expr::Bias { src, by: BiasBy::PerBlock { .. } } => src,
            _ => return,
        };
        if !matches!(yields(operand, checkpoint), Some(Encoding::Raw(_))) {
            decodes = true;
        }
    });
    decodes
}

/// What representation an expression's value is IN — the fact
/// [`decodes_a_packed_plane`] needs and no single node carries.
///
/// Four nodes state their own output type and are read off; the placement
/// family and the two value operators preserve their operand's and are
/// recursed through; a [`Expr::Src`] is answered by the checkpoint's own
/// header, which is the only party that knows. `None` is not "raw": it is
/// "this expression does not say", and the one caller treats it as the
/// refusing answer.
///
/// A per-block [`Expr::Scale`] over packed codes genuinely yields a raw value
/// and this reports its operand's encoding instead. Deliberate, and it only
/// ever makes the caller MORE careful: such a chain has already been named a
/// decode by the node itself, so the imprecision cannot admit one.
fn yields<'a>(expr: &'a Expr, checkpoint: &'a Metadata) -> Option<&'a Encoding> {
    match expr {
        Expr::Src(name) => checkpoint.tensor_by_name(name).map(|held| &held.encoding),
        Expr::Fill { ty, .. } => Some(&ty.encoding),
        Expr::Cast { to, .. } => Some(to),
        Expr::Transmute { to, .. } | Expr::Repack { to, .. } => Some(&to.encoding),
        Expr::Slice { src, .. }
        | Expr::Stride { src, .. }
        | Expr::Gather { src, .. }
        | Expr::Shard { src, .. }
        | Expr::Select { src, .. }
        | Expr::Scale { src, .. }
        | Expr::Bias { src, .. } => yields(src, checkpoint),
        Expr::Concat { parts, .. } => parts.first().and_then(|leg| yields(leg, checkpoint)),
        Expr::Out(_) | Expr::SrcIndexed(_) => None,
    }
}

/// Move into `promoted` every entry that a promoted one is PAIRED WITH, until
/// nothing moves.
///
/// **A PACKED PLANE AND ITS FACTORS ARE ONE THING, AND THE ARTIFACT HAS TO
/// HOLD THEM UNDER ONE STEM.** `Builder::read_own` is the verb a serving load
/// reaches when the file already holds a plane under the contract's name, and
/// it claims the companions BY NAME — `claim` pairs a quantized weight with
/// `<w>.scales` and `<w>.biases`, and `interned` declares them there. So an
/// artifact holding `layer.0.q_proj` beside
/// `model.layers.0.self_attn.q_proj.scales` is an artifact no load can read:
/// the first arm fires on the codes, and the factors it then asks for are
/// under a name the promotion left behind.
///
/// It is reachable without any of this being exotic. An MLX affine projection
/// states its codes as a [`Expr::Transmute`] — always, because the file packs
/// them into `u32` words — and its factors as a plain rename whenever the
/// checkpoint already ships them bf16. One node on one entry and none on the
/// other, and the group would split down the middle.
///
/// **PAIRED, AND NOT SUFFIX-MATCHED.** [`Scales::of`](checkpoint::contract::Scales)
/// and `TensorContract::zero_points` are the contract's own statement of which
/// weight a factors plane belongs to, written for exactly the reason this
/// reads them rather than comparing names: "a suffix match is how a plane gets
/// paired with a weight it never belonged to". Both directions travel — a
/// promoted weight pulls its factors across, a promoted factors plane pulls
/// its weight — and to a fixed point, because pulling the scales in is what
/// makes the biases beside them reachable.
fn close_over_the_declared_pairings(
    promoted: &mut Vec<TensorContract>,
    kept: &mut Vec<TensorContract>,
) {
    loop {
        let mut named: BTreeSet<String> = BTreeSet::new();
        for tensor in promoted.iter() {
            named.insert(tensor.name.clone());
            if let Some(scales) = &tensor.scales {
                named.insert(scales.of.clone());
            }
            if let Some(of) = &tensor.zero_points {
                named.insert(of.clone());
            }
        }
        let mut moved = false;
        let mut still: Vec<TensorContract> = Vec::with_capacity(kept.len());
        for tensor in std::mem::take(kept) {
            let joined = named.contains(&tensor.name)
                || tensor
                    .scales
                    .as_ref()
                    .is_some_and(|scales| named.contains(&scales.of))
                || tensor
                    .zero_points
                    .as_ref()
                    .is_some_and(|of| named.contains(of));
            if joined {
                promoted.push(tensor);
                moved = true;
            } else {
                still.push(tensor);
            }
        }
        *kept = still;
        if !moved {
            return;
        }
    }
}

/// Refuse a promotion that eats a source tensor another read still names.
///
/// A promoted chain's legs leave the artifact: that is what makes the
/// promotion mean "read the answer here" rather than "hold both". The entries
/// this command did NOT promote are still read from the source's own
/// spelling at every load, so a leg one of them names must survive — and
/// nothing about the gate above keeps two entries from reaching for the same
/// tensor with only one of them stating a transform.
///
/// **REFUSED RATHER THAN PARTIALLY PROMOTED**, and the alternative is what
/// makes the case: dropping the promotion whose leg is contested would make
/// the artifact's contents depend on the order this command happened to walk
/// the contract in, and would do it silently. A refusal names the two entries
/// and the tensor they disagree over, and points at the repair — promote both
/// or neither, which is the contract author's call and not this command's.
///
/// It cannot fire on any checkpoint this tree imports today: every family's
/// shared legs are shared by entries that state the same node, so they promote
/// together. That is the reason to write it now rather than a reason not to.
fn refuse_a_leg_a_surviving_read_still_needs(
    sku: &str,
    kept: &[TensorContract],
    consumed: &[String],
) -> Result<()> {
    for tensor in kept {
        for leg in tensor.expr.sources() {
            if consumed.iter().any(|eaten| eaten == leg) {
                bail!(
                    "{sku}: `{name}` is read from the checkpoint's `{leg}` at every load, and \
                     another tensor of the same contract states a transform over `{leg}` that \
                     this command would run here — which takes `{leg}` out of the artifact and \
                     leaves `{name}` naming a tensor that is no longer there. Promote both or \
                     neither: give `{name}` a chain that states its own transform, or take the \
                     transform off the tensor that shares its leg.",
                    name = tensor.name,
                );
            }
        }
    }
    Ok(())
}

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
fn tied_head_sources(metadata: &Metadata) -> Vec<String> {
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

/// Carries the source's `config.json` into the artifact, verbatim.
///
/// # Why this stopped normalizing
///
/// It used to compile the config into a `pie.model/1` descriptor: 136 fields
/// of normalized geometry, which the engine then re-parsed to learn what
/// model it had. That was the *identity* crossing as a document, and it is
/// what the catalog refactor removed — identity is now a manifest match
/// against the tensors, and the tensors are already in the artifact.
///
/// What is left for a config to say is the part the tensors cannot: the
/// declared quantization, because a group size is not an extent of anything.
/// `models::serve::encoding::Encoding` read exactly that, from the checkpoint's
/// own words, and M18 deleted it along with the module — a checkpoint's
/// quantization comes off its STORED tensor encodings now, which is a stronger
/// answer to the same question. The honest thing to carry is still the
/// checkpoint's own words.
///
/// It is also why this can no longer fail on content. A config this command
/// does not understand is not this command's problem — nothing here reads it,
/// and the loader refuses an encoding it cannot parse at the point that needs
/// it.
/// Only unreadable bytes or invalid JSON are errors, and JSON is checked so
/// that an artifact never carries an object no reader can open.
///
/// `Ok(None)` when there is no `config.json` — a lone `.gguf` carries its
/// metadata in its own header, and a directory without one is a weights-only
/// checkpoint.
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
    let attributes = match checkpoint::file::zt::read_attributes(artifact) {
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
    writer: &mut Writer,
    plan: Option<&'a checkpoint::plan::LoadPlan>,
    base: &Path,
    spool: Option<&'a mut Spool>,
    passthrough: &[(&'a RawTensor, &'a str, &'a str)],
    meta: &'a [(String, Vec<u8>)],
    ranked: Option<&[String]>,
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
    // **THE ORDER IS THE RANKING'S WHEN THERE IS ONE, AND NAMES OTHERWISE.**
    //
    // A serving artifact's planes should lie in the order a streaming boot
    // reads them — hottest first, one forward walk — and that order is the
    // shell's ranking, which `runtime::engine::load::sequence` derives from
    // the trace. Name order was never a choice here: `Writer::create` opens
    // canonical and canonical form REFUSES non-ascending insertion, so this
    // sort was the writer's rule wearing a caller's clothes.
    // `Writer::create_serving` is non-canonical for exactly this.
    //
    // A plane the ranking does not name keeps its place among the names,
    // AFTER everything ranked: `__meta__/` objects are not served and never
    // appear in a sequence, and a weight the ranking missed is one this shell
    // could not size — it belongs in the file and not in the hot run.
    match ranked {
        Some(order) => {
            let rank: BTreeMap<&str, usize> =
                order.iter().enumerate().map(|(at, name)| (name.as_str(), at)).collect();
            entries.sort_by_key(|(name, _)| {
                (rank.get(name).copied().unwrap_or(usize::MAX), *name)
            });
        }
        None => entries.sort_by(|a, b| a.0.cmp(b.0)),
    }

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
                    .strip_prefix(checkpoint::file::meta::META_PREFIX)
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
    plan: &checkpoint::plan::LoadPlan,
    base: &Path,
    to: &std::sync::mpsc::SyncSender<std::result::Result<(String, Vec<u8>), String>>,
    watch: &(dyn Fn(u64, u64) + Sync),
) {
    let mut sink = Handoff { to };
    let outcome = checkpoint::executor::Execution::new(plan, base)
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

    // FIVE TESTS STOOD HERE — `published()` built a legacy catalog row's
    // manifest as an `Observed` checkpoint, and four assertions held
    // `head_is_a_materialized_tie` against it. They die with the function and
    // with the catalog it asked. What the check did is recorded where it
    // stood, above `carry_config`.

    use super::*;
    use checkpoint::file::write::{WriteTensor, write_zt};
    use checkpoint::types::{DType, Encoding, TensorId};

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

    /// **THE WHOLE LANDING IS PROMOTED, AND A DECODE NEVER IS** (§M-4a).
    ///
    /// The selector is the whole of the family knowledge here, and §M-4a
    /// widened it from two nodes to the landing: the placement family, the
    /// rename, the two value operators at a uniform factor and the raw
    /// narrowing all join the encode and the repack, so that the artifact's
    /// tensors are the planes the engine binds.
    ///
    /// **THE NEGATIVES ARE THE POINT OF THE TEST.** The old predicate could
    /// not express a decode — `Cast { to: Encoding::Quant(_) }` matches one
    /// direction and one direction only — and the widened one can, because a
    /// decode and a narrowing are the SAME NODE with a different operand. So
    /// the decode arm is asserted at three depths and under two spellings
    /// (`Cast`, and MLX's per-block `Scale`), and once with the operand's
    /// encoding known only from the checkpoint's header rather than from any
    /// node of the chain — which is the case a structural predicate cannot
    /// see and would have promoted.
    ///
    /// And `Shard` is asserted absent: rank is §M-4b/M-4e's, the index entry
    /// is where it goes, and at `tp = 1` there is nothing to promote anyway.
    #[test]
    fn the_landing_is_promoted_and_a_decode_never_is() {
        use checkpoint::contract::TensorType;
        use checkpoint::types::{FileId, QuantScheme, QuantSpec, TensorId};

        let mxfp4 = Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        });

        // A checkpoint holding three raw tensors and one packed one. `w`,
        // `a` and `b` are stored bf16; `packed` ships the codes.
        let held = |id: u32, name: &str, encoding: Encoding| RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 64,
            shape: vec![4, 32],
            encoding,
        };
        let checkpoint = Metadata {
            files: Vec::new(),
            tensors: vec![
                held(0, "w", Encoding::Raw(DType::Bf16)),
                held(1, "a", Encoding::Raw(DType::Bf16)),
                held(2, "b", Encoding::Raw(DType::Bf16)),
                held(3, "packed", mxfp4.clone()),
            ],
        };
        let promoted = |expr: &Expr| states_an_import_transform(expr, &checkpoint);

        // A plain copy states nothing at all, and a rename is a copy.
        assert!(!promoted(&Expr::src("w")));

        // THE DECODE, REFUSED FOUR WAYS.
        //
        // Stated by the chain: a transmute names the packed type and the cast
        // takes the value out of it.
        assert!(!promoted(
            &Expr::src("w")
                .transmute(TensorType {
                    shape: vec![4, 32],
                    encoding: mxfp4.clone(),
                })
                .cast(Encoding::Raw(DType::Bf16))
        ));
        // Stated by the FILE and by nothing else — the case that has no
        // structural tell: this is the same node as the narrowing below, over
        // a tensor the checkpoint happens to ship packed.
        assert!(!promoted(
            &Expr::src("packed").cast(Encoding::Raw(DType::Bf16))
        ));
        // At depth, under a chain that would otherwise be promoted twice
        // over: a promotion moves the WHOLE expression, so a decode buried
        // under a concat and a slice is a decode this command would run.
        assert!(!promoted(&Expr::concat(
            0,
            vec![
                Expr::src("a"),
                Expr::src("packed")
                    .cast(Encoding::Raw(DType::Bf16))
                    .slice(0, 0, 2),
            ]
        )));
        // And in MLX's spelling of it, which is not a `Cast` at all: a
        // per-block scale over packed codes is `code · scale`, the first half
        // of an affine decode.
        assert!(!promoted(
            &Expr::src("packed").scale_per_block(Expr::src("a"))
        ));

        // THE NARROWING, WHICH IS THE SAME NODE AND IS PROMOTED. It moves the
        // bytes `materialize_contract` would have written under the SOURCE's
        // name under the CONTRACT's instead; the third retain in
        // `promote_import_transforms` is what stops that being two copies.
        assert!(promoted(&Expr::src("w").cast(Encoding::Raw(DType::Bf16))));

        // THE ENCODE AND THE REPACK — the two a serving load refuses outright,
        // found under the chain a family actually writes and not only at the
        // root.
        assert!(promoted(&Expr::src("w").cast(mxfp4.clone())));
        assert!(promoted(
            &Expr::concat(0, vec![Expr::src("a"), Expr::src("b")]).cast(mxfp4.clone())
        ));
        assert!(promoted(&Expr::src("w").repack(
            checkpoint::types::RepackLayout::TiledAffineU4Weight,
            TensorType {
                shape: vec![16, 64],
                encoding: Encoding::Raw(DType::Bf16),
            }
        )));

        // THE LANDING §M-4a ADDS: the fusion, the band, the rename, the fold.
        assert!(promoted(&Expr::concat(
            0,
            vec![Expr::src("a"), Expr::src("b")]
        )));
        assert!(promoted(&Expr::src("w").slice(0, 0, 2)));
        assert!(promoted(&Expr::src("w").stride(0, 0, 2, 2)));
        assert!(promoted(&Expr::src("w").transmute(TensorType {
            shape: vec![2, 64],
            encoding: Encoding::Raw(DType::Bf16),
        })));
        assert!(promoted(&Expr::src("w").bias(1.0)));
        assert!(promoted(&Expr::src("w").scale(2.0)));

        // AND RANK IS NOT PROMOTED, NOR IS ANY CHAIN CARRYING IT. `Shard` is
        // the identity at `tp = 1` and an index entry's business above it, so
        // refusing costs nothing and leaves §M-4b free to decide the layout.
        assert!(!promoted(&Expr::src("w").shard(0)));
        assert!(!promoted(&Expr::concat(
            0,
            vec![Expr::src("a").shard(0), Expr::src("b").shard(0)]
        )));
    }

    /// A promoted packed plane takes its FACTORS with it, and a promoted
    /// factors plane takes its weight.
    ///
    /// `Builder::read_own` claims a quantized weight's companions by name —
    /// `<w>.scales` and `<w>.biases` — so a group split across the two
    /// namespaces is a group no load can read: the codes under the contract's
    /// name, the factors under the source's. The pairing travels through
    /// `Scales::of` and `zero_points`, which is the contract's own statement
    /// of it, and to a fixed point: promoting the weight pulls the scales,
    /// and the scales' presence is what makes the biases beside them
    /// reachable in the same sweep.
    #[test]
    fn a_promoted_plane_takes_its_declared_factors_with_it() {
        use checkpoint::contract::Scales;
        use checkpoint::types::{QuantGranularity, ScaleForm};

        let plain = |name: &str, expr: Expr| {
            TensorContract::new(name, expr, vec![4, 32], Encoding::Raw(DType::Bf16))
        };
        // The codes state a transform; neither companion does.
        let mut promoted = vec![plain(
            "layer.0.q_proj",
            Expr::concat(0, vec![Expr::src("q.weight"), Expr::src("g.weight")]),
        )];
        let mut kept = vec![
            TensorContract {
                scales: Some(Scales {
                    of: "layer.0.q_proj".to_string(),
                    granularity: QuantGranularity::PerGroup,
                    group_size: 64,
                    channel_axis: 1,
                    form: ScaleForm::Bf16AffineFactors,
                }),
                ..plain("layer.0.q_proj.scales", Expr::src("q.scales"))
            },
            TensorContract {
                zero_points: Some("layer.0.q_proj".to_string()),
                ..plain("layer.0.q_proj.biases", Expr::src("q.biases"))
            },
            plain("layer.0.o_proj", Expr::src("o.weight")),
        ];

        close_over_the_declared_pairings(&mut promoted, &mut kept);

        let names: Vec<&str> = promoted.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "layer.0.q_proj",
                "layer.0.q_proj.scales",
                "layer.0.q_proj.biases"
            ],
            "the codes carried both factors planes across"
        );
        let left: Vec<&str> = kept.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            left,
            ["layer.0.o_proj"],
            "and a plane that pairs with nothing promoted stays where it was"
        );
    }

    /// A plane the SKU reads at its own raw dtype is COPIED AT THE SOURCE'S
    /// WIDTH, and one read at bf16 is still narrowed.
    ///
    /// Both arms, because the whole content of the rule is that it
    /// DISCRIMINATES. `materialize_contract` narrows from the source encoding
    /// alone and is right to — it names no family convention, which is what
    /// lets it live in the loader — so a rule that spared every F32 plane
    /// would undo the narrowing wholesale and hand every mlx affine factor
    /// back to the engine to rewrite at boot. The bf16 arm is what says it
    /// does not.
    ///
    /// The mlx factors are safe for a second and stronger reason than this
    /// test: `checkpoint_dsl::factors` builds a scales or biases entry with
    /// `want = encoding(Dtype::Bf16)` and no branch, so a companion plane
    /// CANNOT declare a width for this rule to spare, whatever the file holds.
    /// The bf16 arm below is the guard for that being true by accident.
    #[test]
    fn a_plane_the_text_declares_wide_is_copied_at_its_width_and_a_bf16_read_is_not() {
        use checkpoint::contract::{Expr, ModelContract, TensorContract};

        // What the loader leaves: both legs narrowed, keyed by source name.
        let narrowing = |name: &str| {
            TensorContract::new(
                name,
                Expr::src(name).cast(Encoding::Raw(DType::Bf16)),
                vec![16],
                Encoding::Raw(DType::Bf16),
            )
        };
        let decay = "model.layers.0.linear_attn.A_log";
        let gate = "model.layers.0.mlp.gate.weight";
        let mut m = Materialization {
            contract: ModelContract {
                alignment: 1,
                tensors: vec![narrowing(decay), narrowing(gate)],
                groups: Vec::new(),
            },
            decoded: vec![decay.into(), gate.into()],
            passthrough: Vec::new(),
            meta: Vec::new(),
        };

        // What the text says: the decay bank at f32 -- `qwen_3`'s own
        // declaration, and the width `kernels_cuda::attn::ssm` debug-asserts
        // -- and the gate at the bf16 every other kernel reads.
        let read = |name: &str, from: &str, dtype: DType| {
            TensorContract::new(
                name,
                Expr::src(from),
                vec![16],
                Encoding::Raw(dtype),
            )
        };
        let kept = [
            read("layer.0.a_log", decay, DType::F32),
            read("layer.0.mlp.gate", gate, DType::Bf16),
        ];
        keep_the_width_a_surviving_read_declares("qwen35-d0.8b-bf16-kv-bf16", &mut m, &kept);

        assert_eq!(m.decoded, vec![gate.to_string()], "only the bf16 read still narrows");
        assert_eq!(m.passthrough, vec![decay.to_string()], "the f32 read is copied as stored");
        let written: Vec<&str> = m.contract.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(written, vec![gate], "the spared plane's narrowing entry is dropped");
    }

    /// A promoted plane whose name the artifact already holds is refused, and
    /// the refusal names the name and both claimants.
    ///
    /// The two namespaces the artifact flattens: a promoted tensor lands under
    /// the CONTRACT'S name, everything else under the SOURCE'S. Both arms are
    /// asserted because the surviving claim can come from either set — a
    /// passthrough copy or a `materialize_contract` narrowing — and a check
    /// that swept only one would pass while the other wrote a name twice.
    #[test]
    fn a_promoted_name_that_collides_with_a_passthrough_is_refused_by_name() {
        use checkpoint::contract::{Expr, ModelContract, TensorContract};

        let bank = |name: &str| {
            TensorContract::new(
                name,
                Expr::src("blk.0.ffn_gate_exps.weight").cast(Encoding::Raw(DType::Bf16)),
                vec![64, 64],
                Encoding::Raw(DType::Bf16),
            )
        };
        let narrowed = |name: &str| {
            TensorContract::new(
                name,
                Expr::src(name).cast(Encoding::Raw(DType::Bf16)),
                vec![64, 64],
                Encoding::Raw(DType::Bf16),
            )
        };
        let materialization = |contract: Vec<TensorContract>, passthrough: Vec<String>| {
            Materialization {
                contract: ModelContract {
                    alignment: 1,
                    tensors: contract,
                    groups: Vec::new(),
                },
                decoded: Vec::new(),
                passthrough,
                meta: Vec::new(),
            }
        };

        // A copy already claims the name.
        let m = materialization(Vec::new(), vec!["blk.0.ffn_gate_exps.weight".into()]);
        let promoted = [bank("blk.0.ffn_gate_exps.weight")];
        let refusal = refuse_a_name_two_parties_claim("kimi-k3", &m, &promoted)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("blk.0.ffn_gate_exps.weight"), "{refusal}");
        assert!(refusal.contains("byte for byte"), "{refusal}");

        // A narrowing already claims it.
        let m = materialization(vec![narrowed("blk.0.ffn_gate_exps.weight")], Vec::new());
        let promoted = [bank("blk.0.ffn_gate_exps.weight")];
        let refusal = refuse_a_name_two_parties_claim("kimi-k3", &m, &promoted)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("blk.0.ffn_gate_exps.weight"), "{refusal}");
        assert!(refusal.contains("F16 or F32"), "{refusal}");

        // And a name no other party holds passes -- the resting state, which
        // is every checkpoint this tree imports today.
        let m = materialization(
            vec![narrowed("model.norm.weight")],
            vec!["model.embed_tokens.weight".into()],
        );
        assert!(
            refuse_a_name_two_parties_claim("kimi-k3", &m, &[bank("blk.0.ffn_gate_exps.weight")])
                .is_ok()
        );
    }

    /// A promoted chain that reads another contract entry is refused, and the
    /// refusal names both the consumer and the producer it cannot carry.
    ///
    /// `Expr::sources` sees `Expr::Src` and stops; the DAG edge is
    /// `Expr::Out`, and a promotion that took the consumer alone would hand
    /// `compile_decode` an expression naming a tensor its contract does not
    /// declare. Asserted against a chain that also states a real gate node,
    /// because the refusal is only reached for tensors the gate admits.
    #[test]
    fn a_promoted_chain_that_reads_another_contract_entry_is_refused_by_name() {
        use checkpoint::contract::{Expr, TensorContract};

        let reads_a_sibling = TensorContract::new(
            "blk.0.attn_q.weight",
            Expr::out("blk.0.attn_q.scaled").repack(
                checkpoint::types::RepackLayout::TiledAffineU4Weight,
                checkpoint::contract::TensorType {
                    shape: vec![16, 64],
                    encoding: Encoding::Raw(DType::Bf16),
                },
            ),
            vec![16, 64],
            Encoding::Raw(DType::Bf16),
        );
        // A repack is admitted by the gate whatever the checkpoint holds —
        // it names a kernel and reads no encoding — so an empty table is the
        // honest fixture here, and it is what a chain rooted at `Expr::Out`
        // would find anyway.
        let nothing = Metadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        assert!(states_an_import_transform(&reads_a_sibling.expr, &nothing));
        let refusal = refuse_a_chain_this_command_cannot_close("qwen3", &reads_a_sibling)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("blk.0.attn_q.weight"), "{refusal}");
        assert!(refusal.contains("blk.0.attn_q.scaled"), "{refusal}");

        // The same chain rooted at the checkpoint's own tensor is what every
        // live contract states, and it passes.
        let reads_the_source = TensorContract::new(
            "blk.0.attn_q.weight",
            Expr::src("model.layers.0.self_attn.q_proj.weight").repack(
                checkpoint::types::RepackLayout::TiledAffineU4Weight,
                checkpoint::contract::TensorType {
                    shape: vec![16, 64],
                    encoding: Encoding::Raw(DType::Bf16),
                },
            ),
            vec![16, 64],
            Encoding::Raw(DType::Bf16),
        );
        assert!(refuse_a_chain_this_command_cannot_close("qwen3", &reads_the_source).is_ok());
    }

    /// The head is dropped from every set that would write it, at either width.
    ///
    /// Both sets are swept because the width decides which one the head lands
    /// in: a BF16 head passes through and an F16 head is decoded to BF16 with a
    /// `TensorContract` behind it. Sweeping only the set a Qwen3 export happens
    /// to use would work until a checkpoint chose the other one.
    #[test]
    fn a_tied_head_is_dropped_from_every_set_that_would_write_it() {
        use checkpoint::contract::{Expr, ModelContract, TensorContract};

        let head = |name: &str| {
            TensorContract::new(
                name,
                Expr::src(name).cast(Encoding::Raw(DType::Bf16)),
                vec![151936, 1024],
                Encoding::Raw(DType::Bf16),
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

    /// **THE DECODE FREES AS IT GOES, AND THE SCHEDULE IS WHERE THAT IS
    /// DECIDED.**
    ///
    /// The honest gate is peak RSS, which no unit test can hold still. What it
    /// rests on is a property of the plan, and that can be pinned exactly: a
    /// buffer is freed at its LAST USE, so a schedule that publishes its first
    /// tensor before it allocates its last buffer has somewhere to free, and
    /// one that allocates everything first does not.
    ///
    /// Both plans below are the same contract against the same
    /// [`decode_target`] — an Unknown backend, which carries no
    /// `BulkExtentWrite` for the coalescer to make and none for the hoist to
    /// move. The hoist reorders it anyway, because it fronts every `Allocate`
    /// whether or not it found a bulk write to put behind them, and reports
    /// zero rewrites while doing it. That is why the import's own streaming
    /// execution held the whole narrowed set: not a refused instruction, just
    /// an order.
    #[test]
    fn the_decode_publishes_a_tensor_before_it_allocates_its_last_buffer() {
        use checkpoint::plan::{LoadPlan, StorageInstr};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.zt");
        let names = ["a", "b", "c", "d"];
        let decls: Vec<TensorDecl> = names
            .iter()
            .enumerate()
            .map(|(i, name)| TensorDecl {
                id: TensorId(i as u32),
                name: (*name).to_string(),
                shape: vec![64],
                encoding: Encoding::Raw(DType::F32),
                alignment: 64,
                visibility: Visibility::default(),
            })
            .collect();
        let payloads: Vec<Vec<u8>> = (0..names.len())
            .map(|seed| {
                (0..64u32)
                    .flat_map(|i| (i as f32 + seed as f32).to_le_bytes())
                    .collect()
            })
            .collect();
        let tensors: Vec<WriteTensor> = decls
            .iter()
            .zip(payloads.iter())
            .map(|(decl, bytes)| WriteTensor { decl, bytes })
            .collect();
        write_zt(&path, &BTreeMap::new(), &tensors).unwrap();

        let metadata = parse_metadata(&path).unwrap();
        let materialization = materialize_contract(&metadata).unwrap();
        // Every one of them narrows, so the decode is the whole file and the
        // schedule below is not a two-instruction degenerate case.
        assert_eq!(materialization.decoded, names);

        let positions = |plan: &LoadPlan, want: fn(&StorageInstr) -> bool| -> Vec<usize> {
            plan.schedule
                .iter()
                .enumerate()
                .filter(|(_, id)| want(plan.instr(**id).unwrap()))
                .map(|(at, _)| at)
                .collect()
        };
        let allocates = |instr: &StorageInstr| matches!(instr, StorageInstr::Allocate { .. });
        let finalizes = |instr: &StorageInstr| matches!(instr, StorageInstr::Finalize { .. });

        let arena =
            checkpoint::plan::compile(&metadata, &materialization.contract, decode_target())
                .unwrap();
        assert!(
            positions(&arena, allocates).iter().max().unwrap()
                < positions(&arena, finalizes).iter().min().unwrap(),
            "the ordinary pipeline fronts every Allocate, which is the arrangement \
             under which nothing is ever at its last use"
        );

        let streaming = compile_decode(&metadata, &materialization.contract).unwrap();
        assert!(
            positions(&streaming, finalizes).iter().min().unwrap()
                < positions(&streaming, allocates).iter().max().unwrap(),
            "and the plan this command compiles publishes its first tensor before it \
             allocates its last buffer"
        );
        assert!(
            !streaming.passes.iter().any(|pass| {
                pass.pass == "hoist-bulk-arena-writes"
                    || pass.pass == "coalesce-persistent-arena-writes"
            }),
            "it says so in the passes it ran: {:?}",
            streaming.passes.iter().map(|pass| &pass.pass).collect::<Vec<_>>()
        );
        // The SCHEDULE is all that differs. Same tensors, same declarations,
        // same instruction set -- a plan that decoded something else would be
        // a different artifact rather than a smaller footprint.
        assert_eq!(arena.passes.len(), streaming.passes.len() + 2);
        assert_eq!(arena.tensors, streaming.tensors);
        assert_eq!(arena.instrs.len(), streaming.instrs.len());
    }
}
