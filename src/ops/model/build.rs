//! `pie model build` — precompute a serve boot, offline.
//!
//! The command `author_abi.cpp` was built to serve, finally landed the way
//! the migration made possible: no FFI at all. The same family author a
//! driver boot runs (`model::contract::author`) writes the serve
//! contract here, the loader compiles it, the streaming host executor
//! materializes it, and the result is a `.zt` whose tensors are the *runtime*
//! tensors — fused QKV banks, stacked experts, requantized weights — under
//! the names the bind path reads. The expensive transforms happen once,
//! offline; loading the optimized artifact afterwards is extent writes.
//!
//! `pie model import` stays the family-blind sibling: it normalizes
//! encodings and touches nothing else. This command is the family-aware
//! step behind the same store, and the split is the one the convert header
//! promised ("family-aware steps slot in behind the same command").
//!
//! What is out of scope, stated rather than implied:
//!
//! * **Tensor parallelism.** A tp>1 materialization is one artifact *per
//!   rank*; the store has no vocabulary for that yet, so `tp_size` is 1.
//! * **Native MXFP4 (Marlin) layouts.** `Repack` is a device-kernel layout
//!   with no host implementation; the host executor refuses the plan. The
//!   routed-decode lowering materializes fine.
//! * **Streamed expert groups.** A group is a paging decision; materializing
//!   one eagerly would build exactly the residency it exists to avoid.
//!
//! Which driver the artifact is for is `--backend`, and it is not cosmetic:
//! CUDA binds fused q/k/v banks under HuggingFace names, while Metal and
//! Vulkan bind in-place projections under MLX names, and an artifact
//! materialized for one family is not what the other's bind path reads. It
//! defaults to `cuda`, which is what the policy silently was before the flag
//! existed.
//!
//! `--backend metal` needed the family schemas to accept their own output
//! first. A serve boot re-authors from *checkpoint* names, so an artifact
//! whose tensors are the runtime tensors is fed back through the rename that
//! produced them; before `mlx::already_lowered` that refused with "Metal llama
//! schema has no declared mapping or skip for 'final_norm.weight'".

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use clap::Args;

use model::catalog::{Override, Unmatched};
use model::encoding::{CONFIG_OBJECT, Encoding};
use model::shared::policy::{Mxfp4MoeRequest, Naming, Policy, Projections, RuntimeQuant};
use model_loader::checkpoint::meta::{
    BACKEND_KEY, CACHE_KEY_KEY, COMPONENT_KEY, CONTRACT_KEY, CONTRACT_REVISION, MODEL_ID_KEY,
    MOE_KEY, RUNTIME_QUANT_KEY, SOURCE_ENCODING_KEY, SOURCE_KEY, SOURCE_STAT_KEY, TP_SIZE_KEY,
    VERSION_KEY, meta_name,
};
use model_loader::checkpoint::read::parse_checkpoint_metadata;
use model_loader::checkpoint::write::CheckpointWriter;
use model_loader::executor::Progress;
use model_loader::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use model_loader::types::Visibility;

use super::import::{
    ProgressLine, Source, Spool, artifact_path, carry_config, compile_tokenizer, pie_version,
    resolve_source,
};

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// What to optimize: a HuggingFace repo ID in the local cache, a snapshot
    /// directory, or a `.zt` artifact.
    pub source: String,
    /// Load-time requantization to bake in: `fp8`, `int8` or `mxfp4` for
    /// `--backend cuda`, `int4` for `--backend metal`, `--backend vulkan` or
    /// `--backend wgpu`. Absent means none — the optimization is then the
    /// layout work alone (fused banks, expert stacks, dequantized schemes).
    ///
    /// Whatever is named here is the transform a serve boot would otherwise
    /// run over every weight, done once and written down instead.
    #[arg(long)]
    pub quant: Option<String>,
    /// The serving device has native FP8 GEMMs. `--quant fp8` on a device
    /// without them is dropped at serve time, so it is dropped here too —
    /// stated as a flag because an offline run cannot probe the device it is
    /// optimizing for.
    #[arg(long)]
    pub fp8_native: bool,
    /// MXFP4 MoE lowering: `routed` (decode on the routed path, the
    /// everywhere-fallback) or `bf16` (eager dequantized stacks). `native`
    /// needs the device's Marlin repack and cannot be materialized offline.
    #[arg(long)]
    pub moe: Option<String>,
    /// Which driver will serve the artifact: `cuda`, `metal`, `vulkan` or
    /// `wgpu`.
    ///
    /// Not cosmetic and not inferable: the drivers read different tensors.
    /// CUDA binds fused q/k/v banks under HuggingFace names; Metal, Vulkan and
    /// wgpu bind in-place projections under MLX names, and an artifact
    /// materialized for one family is not what the other's bind path reads.
    /// Stated as a flag for the reason `--fp8-native` is: an offline run cannot
    /// probe the device it is optimizing for.
    #[arg(long, default_value = "cuda")]
    pub backend: String,
    /// Write the artifact here instead of the store. A path ending in `.zt`
    /// is the artifact; a directory receives `<name>-optimized.zt`.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Report what would be done without doing it.
    #[arg(long)]
    pub dry_run: bool,
    /// Author as this catalog row instead of the one the tensors match.
    ///
    /// The escape hatch for a closed set, and it is deliberately not a way
    /// around the check: the named row's manifest is still held against the
    /// checkpoint, so this can only ever RESOLVE an ambiguity, never
    /// introduce one. Two release names for one geometry is the case it
    /// exists for — nothing in the tensors distinguishes them, so nothing
    /// but a person can.
    #[arg(long = "as", value_name = "ID")]
    pub as_id: Option<String>,
}

/// pie's own objects, lifted out of an artifact being re-optimized.
///
/// An artifact is the one source that arrives with its metadata already
/// compiled, and `carry_config` / `compile_tokenizer` both answer `None` for
/// it — correctly, since there is no `config.json` or `tokenizer.json` beside
/// it. Writing the output without them would silently produce an artifact that
/// cannot serve, so they are carried across verbatim instead.
struct CarriedObjects {
    config_bytes: Vec<u8>,
    tokenizer: Vec<(String, Vec<u8>)>,
    /// What `--quant` the source artifact was already built with, if any.
    baked_quant: Option<String>,
    /// How the checkpoint the archive was imported from stored its numbers.
    ///
    /// See `SOURCE_ENCODING_KEY`. Carried forward so a built artifact can say
    /// what its weights have been through, and read here so a requantization
    /// can say so out loud.
    source_encoding: Option<String>,
}

fn read_carried_objects(path: &Path) -> Result<CarriedObjects> {
    let attributes = model_loader::checkpoint::zt::parse_attributes(path).ok();
    let checkpoint = parse_checkpoint_metadata(path)
        .map_err(|err| anyhow!("cannot read {}: {err}", path.display()))?;
    let config_bytes = model_loader::checkpoint::read::read_meta(&checkpoint, CONFIG_OBJECT)?
        .ok_or_else(|| {
            anyhow!(
                "{} carries no {CONFIG_OBJECT}; it is a checkpoint file rather than a \
                 pie artifact, or an artifact from before the config was carried \
                 verbatim — re-import it",
                path.display()
            )
        })?;
    let mut tokenizer = Vec::with_capacity(tokenizer::canonical::OBJECTS.len());
    for name in tokenizer::canonical::OBJECTS {
        let bytes =
            model_loader::checkpoint::read::read_meta(&checkpoint, name)?.ok_or_else(|| {
                anyhow!(
                    "{} carries a model config but not {name}; an artifact with half \
                     its metadata cannot serve, and this command does not compile the rest",
                    path.display()
                )
            })?;
        tokenizer.push((name.to_string(), bytes));
    }
    Ok(CarriedObjects {
        config_bytes,
        tokenizer,
        baked_quant: attributes
            .as_ref()
            .and_then(|attrs| attrs.text(RUNTIME_QUANT_KEY).map(str::to_string)),
        source_encoding: attributes
            .as_ref()
            .and_then(|attrs| attrs.text(SOURCE_ENCODING_KEY).map(str::to_string)),
    })
}

/// What `pie model build` reads, preferring the store archive.
///
/// [`resolve_source`] answers `import`'s question — where do fresh weights
/// come from — and for a bare name that is the HuggingFace snapshot. `build`'s
/// question is different: it re-lays a model pie already holds, and pie holds
/// it as the archive. Falling through to the snapshot meant `pie model build
/// <name>` re-derived from HuggingFace everything the import had already
/// decided, and could fail identification on a snapshot whose archive builds
/// perfectly well.
///
/// A path is still taken as given. This only changes what a bare name means,
/// and only when the store has one.
fn resolve_build_source(spelled: &str) -> Result<Source> {
    if !Path::new(spelled).exists() {
        for name in [spelled.to_string(), spelled.replace('/', "--")] {
            let archive = crate::local::store::archive_path(&name);
            if archive.is_file() {
                return resolve_source(&archive.to_string_lossy());
            }
        }
    }
    resolve_source(spelled)
}

pub fn run(args: BuildArgs) -> Result<crate::ui::Answer> {
    let source = resolve_build_source(&args.source)?; // Two kinds of source, and the difference is only where the config comes
    // from: an artifact carries the checkpoint's own, and a snapshot has it
    // on disk. Both reach the author the same way from here.
    let carried = if source.path.is_file() {
        Some(read_carried_objects(&source.path)?)
    } else {
        None
    };
    let config_bytes = match &carried {
        Some(objects) => objects.config_bytes.clone(),
        None => carry_config(&source)?.ok_or_else(|| {
            // Identity would survive this — the manifest is matched against
            // the tensors, which are right there. The DECLARED QUANTIZATION
            // would not: a group size is not an extent of anything, so an
            // AWQ checkpoint with no config is indistinguishable from a bf16
            // one here and would be authored as bf16. Refusing is the only
            // answer that is not a guess.
            anyhow!(
                "build needs a snapshot with a config.json: the tensors say what \
                 model this is, but only the config states how its numbers are \
                 quantized"
            )
        })?,
    };
    let config = String::from_utf8(config_bytes.clone())
        .map_err(|err| anyhow!("the model config is not UTF-8: {err}"))?;
    let encoding = Encoding::from_config_json(&config)
        .map_err(|err| anyhow!("cannot read the model config's quantization: {err}"))?;

    let requested = QuantRequest {
        spelled: args.quant.as_deref(),
        fp8_native: args.fp8_native,
        backend: &args.backend,
        baked_quant: carried.as_ref().and_then(|c| c.baked_quant.as_deref()),
        source_label: &source.path.display().to_string(),
        source_encoding: carried.as_ref().and_then(|c| c.source_encoding.as_deref()),
    };
    let resolved = resolve_quant(&requested)?;
    for notice in &resolved.notices {
        println!("optimize: {notice}");
    }
    let runtime_quant = resolved.quant;
    let moe_request = match args.moe.as_deref() {
        None => Mxfp4MoeRequest::Auto,
        Some("routed") => Mxfp4MoeRequest::RoutedDecode,
        Some("bf16") => Mxfp4MoeRequest::EagerBf16,
        Some("native") => bail!(
            "--moe native is a device-kernel layout (Marlin repack) and cannot be \
             materialized offline; optimize with `routed` or `bf16` and let the \
             serve boot repack"
        ),
        Some(other) => bail!("--moe {other:?} is not `routed` or `bf16`"),
    };
    let policy = build_policy(&args.backend, runtime_quant, moe_request)?;
    // The host is the executing device, so the target states exactly what
    // the host executor implements — and tp stays whole: a sharded
    // materialization is one artifact per rank, which the store cannot say
    // yet.
    let target = StorageTarget {
        tile_map_mask: CONVERT_TILE_MAP_MASK,
        max_tile_bytes: 64 << 20,
        preferred_alignment: 256,
        ..StorageTarget::default()
    };

    let metadata = parse_checkpoint_metadata(&source.path)
        .map_err(|err| anyhow!("cannot read {}: {err}", source.path.display()))?;

    // WHAT MODEL THIS IS, ASKED OF THE TENSORS.
    //
    // It used to be asked of the config: `model_type` picked an author out of
    // a table, and a config that misdescribed its own geometry was believed
    // until an assertion several frames later contradicted it, if one did.
    // The row is now matched against the tensor names and extents that are
    // about to be read, so identification and validation are one operation
    // and a checkpoint is a known model or it is not.
    let chosen = args
        .as_id
        .as_ref()
        .map_or(Override::None, |id| Override::Id(id.clone()));
    let row = model::catalog::identify(&metadata, &chosen).map_err(|why| {
        let hint = match &why {
            Unmatched::Ambiguous { .. } => {
                "\n  two rows are one geometry under two names; say which with --as <id>"
            }
            _ => "",
        };
        anyhow!("cannot identify {}: {why}{hint}", source.path.display())
    })?;

    let contract = model::contract::author(row, &encoding, &metadata, &target, &policy)
        .map_err(|err| anyhow!("cannot author '{}': {err}", row.id()))?;
    if !contract.groups.is_empty() {
        bail!(
            "the '{}' contract declares streamed expert groups, which are a paging \
             decision; materializing them eagerly would build the residency they avoid",
            row.id()
        );
    }
    let public = contract
        .tensors
        .iter()
        .filter(|tensor| tensor.visibility == Visibility::Public)
        .count();
    println!(
        "optimize: {} declares {} tensors ({} bound) for {}, quant={:?}, moe={:?}",
        row.id(),
        contract.tensors.len(),
        public,
        args.backend,
        policy.runtime_quant,
        policy.moe_request,
    );

    // The plan before the path, because the path is derived from it. Compiling
    // is metadata work — no weight byte is read — so a dry run pays for it and
    // gets to report the key it would have written under.
    let plan = model_loader::plan::compile(&metadata, &contract, target)
        .map_err(|err| anyhow!("cannot compile: {err}"))?;

    // WHAT THIS BUILD IS FOR, AS ITS NAME.
    //
    // The output used to be `<name>-optimized.zt` beside the archive. That
    // spelled the relationship between the two as a name suffix, allowed only
    // one build to exist at a time, and — because `--backend`, `--quant` and
    // the MoE lowering all land in the tensors and none of them in the name —
    // let a CUDA build be served to Metal with nothing on disk to contradict
    // it. The cache key is over the whole compiled plan, so every one of those
    // facts moves the file.
    let key = model_loader::cache_key::artifact_cache_key(
        &plan,
        &model_loader::cache_key::ArtifactInputs {
            snapshot_dir: &source.base(),
            runtime_quant: &format!("{runtime_quant:?}").to_lowercase(),
            // `pie model build` materializes the whole model; a component is a
            // slice of one, which only the serving path asks for.
            component: model_loader::cache_key::ArtifactInputs::WHOLE_MODEL,
        },
    );
    let out_file = match &args.out {
        Some(out) => artifact_path(out, &format!("{}-{key}", source.name)),
        None => {
            // A RUNTIME IS DERIVED FROM AN ARCHIVE, SO THERE HAS TO BE ONE.
            //
            // The store is `<name>/archive.zt` plus `<name>/runtime/<key>.zt`,
            // and everything that reads it enters through the archive: a model
            // directory holding only builds is invisible to `pie model list`,
            // `info` and `remove`, which makes a multi-gigabyte artifact
            // unreclaimable through the CLI and unaccounted for in the store
            // size. Building straight from a snapshot used to be fine because
            // the output was a flat sibling; now it would create exactly that
            // state. Refused rather than tolerated, because "a runtime hangs
            // off an archive" is the layout, and a reader that must handle its
            // violation is a reader that cannot rely on it.
            let archive = crate::local::store::archive_path(&source.name);
            if !archive.is_file() {
                bail!(
                    "nothing to derive from: the store has no {}. \
                     `pie model import {}` writes one — that artifact is servable \
                     on its own, and this command lays it out for one backend. \
                     To build somewhere else, pass `--out <path>`.",
                    crate::ui::short_path(&archive),
                    args.source,
                );
            }
            crate::local::store::runtime_path(&source.name, &key)
        }
    };
    if args.dry_run {
        return Ok(crate::ui::Answer::noop(format!(
            "dry run: would write {}",
            crate::ui::short_path(&out_file)
        )));
    }

    // Metadata first, weights streamed after — the same shape convert has.
    // The config was read above for its quantization; the artifact carries
    // those same bytes rather than a second reading of the same file.
    let tokenizer = compile_tokenizer(&source, &metadata)?;

    let mut bar = ProgressLine::new();
    let mut spool = Spool::create(&out_file)?;
    model_loader::executor::Execution::new(&plan, &source.base())
        .streaming()
        .sink(&mut spool)
        .progress(&mut |progress| {
            bar.render(&Progress {
                read_bytes: progress.read_bytes,
                total_read_bytes: progress.total_read_bytes,
                finalized: progress.finalized,
            });
        })
        .run()
        .map_err(|err| anyhow!("materializing failed: {err}"))?;

    let mut provenance = BTreeMap::from([
        (VERSION_KEY.to_string(), pie_version().to_string()),
        (
            SOURCE_KEY.to_string(),
            format!("optimize:{}", source.origin),
        ),
        // What this build is for, said out loud. The path says it too, and the
        // path is what the store matches on; these are what survive the file
        // being copied somewhere the store cannot see.
        //
        // They are also what the SERVE path matches on, which is the reason
        // there are six of them and not two. A serve cannot reproduce the key
        // above: it names the plan *this* command compiled, against the host
        // converter's `StorageTarget`, and a serve compiles for its device.
        // Worse, the plan a serve would compile over the archive may not
        // compile at all — a transform outside the backend's tile-map mask is
        // exactly the work `build` exists to move offline. So a runtime is
        // chosen by matching the request it states against the request being
        // made, and every fact that decides the bytes has to be said here or
        // the match is unsound.
        (BACKEND_KEY.to_string(), args.backend.clone()),
        (CACHE_KEY_KEY.to_string(), key.clone()),
        (
            MOE_KEY.to_string(),
            format!("{moe_request:?}").to_lowercase(),
        ),
        // `pie model build` materializes the model entire; a component is a
        // slice, which only the serving path asks for.
        (COMPONENT_KEY.to_string(), "full".to_string()),
        // One artifact, unsharded. See `target` above.
        (TP_SIZE_KEY.to_string(), "1".to_string()),
        (
            SOURCE_STAT_KEY.to_string(),
            model_loader::cache_key::snapshot_stat(&source.base()),
        ),
        (CONTRACT_KEY.to_string(), CONTRACT_REVISION.to_string()),
        // WHICH MODEL THIS IS, DECIDED ONCE, HERE.
        //
        // `identify` above answered it against the *archive*, whose tensors
        // are spelled the way the manifests are. What this command writes is
        // spelled the way the bind path reads, which is a different vocabulary
        // and not one any manifest describes — so a boot that re-derived the
        // row from these tensors would be asking a settled question of
        // evidence that can no longer answer it, and would get "no such model"
        // for a model pie itself just identified.
        //
        // See `MODEL_ID_KEY`: this is believed rather than re-checked, and it
        // is only sound beside `CONTRACT_KEY`, which says the tensors were
        // laid out by the contract that goes with this row.
        (MODEL_ID_KEY.to_string(), row.id().to_string()),
    ]);
    // Carried, not re-derived: this command reads an archive whose weights are
    // already normalized, so the only party that ever knew what the original
    // checkpoint stored is the import that read it. Dropping it here would
    // make a built artifact the one place the chain goes dark.
    if let Some(was) = carried.as_ref().and_then(|c| c.source_encoding.clone()) {
        provenance.insert(SOURCE_ENCODING_KEY.to_string(), was);
    }
    // Only when one was baked in. Absent means "no runtime quantization",
    // which is what every imported artifact and every unquantized build is,
    // so there is nothing to write for them and nothing to read back.
    if runtime_quant != RuntimeQuant::None {
        provenance.insert(
            RUNTIME_QUANT_KEY.to_string(),
            format!("{runtime_quant:?}").to_lowercase(),
        );
    }
    let mut writer = CheckpointWriter::create(&out_file, &provenance)
        .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;

    // One ascending pass over metadata objects and runtime tensors together,
    // which is what canonical form asks for.
    enum Entry<'a> {
        Meta(&'a [u8]),
        Tensor(&'a model_loader::types::TensorDecl),
    }
    let mut meta: Vec<(String, Vec<u8>)> = Vec::new();
    match &carried {
        // Already compiled once, at import. Recompiling is not available here
        // (there is nothing to recompile from) and would not be wanted anyway:
        // this command re-lays weights, so the metadata is carried, not
        // re-derived.
        Some(objects) => {
            for (path, bytes) in &objects.tokenizer {
                meta.push((meta_name(path), bytes.clone()));
            }
            meta.push((meta_name(CONFIG_OBJECT), objects.config_bytes.clone()));
        }
        None => {
            if let Some(canonical) = &tokenizer {
                for (path, bytes) in canonical.objects() {
                    meta.push((meta_name(path), bytes.to_vec()));
                }
            }
            // Not conditional: the run could not have authored anything
            // without this document, so by here it exists.
            meta.push((meta_name(CONFIG_OBJECT), config_bytes.clone()));
        }
    }
    let mut entries: Vec<(&str, Entry<'_>)> = meta
        .iter()
        .map(|(name, bytes)| (name.as_str(), Entry::Meta(bytes)))
        .collect();
    for decl in &plan.tensors {
        if decl.visibility == Visibility::Public {
            entries.push((decl.name.as_str(), Entry::Tensor(decl)));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut written = 0u64;
    for (name, entry) in &entries {
        match entry {
            Entry::Meta(bytes) => {
                let path = name
                    .strip_prefix(model_loader::checkpoint::meta::META_PREFIX)
                    .expect("metadata entries carry the namespace prefix");
                writer
                    .add_meta(path, bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
            }
            Entry::Tensor(decl) => {
                let bytes = spool.read(name)?;
                writer
                    .add_tensor(decl, &bytes)
                    .map_err(|err| anyhow!("cannot write '{name}': {err}"))?;
                written += bytes.len() as u64;
            }
        }
    }
    writer
        .finish()
        .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;
    spool.remove();
    bar.finish();

    Ok(crate::ui::Answer::did(format!(
        "built {} — {} of runtime tensors → {}",
        source.name,
        crate::ui::bytes(written),
        crate::ui::short_path(&out_file)
    )))
}

/// What the named backend's bind path reads.
///
/// The pair moves together -- there is no driver that wants MLX names over
/// fused banks -- so one flag sets both and no combination can be spelled
/// that no bind path reads.
///
/// Vulkan reads exactly what Metal reads, and wgpu reads exactly what Vulkan
/// reads. The three drivers share no code, but they share a bind path shape:
/// MLX names over in-place projections, matvecs over MLX affine int4. One arm
/// rather than three policies that happened to be equal, which would be three
/// things to keep equal.
///
/// For wgpu the equality is not a resemblance: the two shells read ONE name
/// table, `driver/src/names.rs`, which each re-exports under its own path.
/// It used to be two byte-identical copies, and the test below used to diff
/// them; it now checks that neither shell has grown one back.
fn bind_policy(backend: &str) -> Result<(Projections, Naming)> {
    match backend {
        "cuda" => Ok((Projections::Fused, Naming::Hf)),
        "metal" | "vulkan" | "wgpu" => Ok((Projections::InPlace, Naming::Mlx)),
        other => bail!("--backend {other:?} is not `cuda`, `metal`, `vulkan` or `wgpu`"),
    }
}

/// The policy a build authors with, which is the bind path's — except for one
/// flag, deliberately.
///
/// # What is persisted is not what is bound
///
/// [`bind_policy`] says CUDA binds [`Projections::Fused`], and it is right:
/// the CUDA bind path reads a fused q/k/v bank. But fusion is not a fact about
/// which tensors EXIST. `Builder::publish_fused` defines the bank and then
/// re-publishes each projection as `Expr::out(bank).slice(..)` — a non-owning
/// view — precisely so that a bind path reading q/k/v individually still finds
/// them under their own names. On a device those views are pointers into the
/// bank, which is why that function's own doc can say "this is not a
/// persistent duplicate-memory budget".
///
/// A file has no pointers. Every `Public` tensor this command writes gets its
/// own bytes, and a view is `Public` because the bind path reads it by name —
/// so persisting a fusion persists the bank AND the projections it aliases.
/// Measured on Qwen3-0.6B: 366 tensors against the archive's 310, and the 56
/// extra banks total 587,202,560 bytes — exactly the file-size delta, and
/// exactly the delta in resident VRAM (560 MiB), because the boot then uploads
/// both. The artifact was 1.79 GB where the archive was 1.19 GB, and loaded no
/// faster.
///
/// So a build authors the fusion away and lets the load path make it, from the
/// projections, the way it does over the archive. What is given up is a
/// concatenation, which is memory-bandwidth trivial and measured at no
/// difference (457/433 ms archive against 431/444 ms prebuilt); what is bought
/// is that `build` stops making the model bigger. Verified after: 310 tensors,
/// 1.19 GB, 1392 MiB resident — the archive's numbers exactly.
///
/// **The rule this is an instance of: a build must not materialize a tensor
/// that another materialized tensor is a view of.** The transforms `build`
/// exists for — dequantization, expert stacking, requantization — are not
/// views and are unaffected, which is measurable: gpt-oss-20b builds to the
/// same 459 tensors and 12.8 GiB either way.
fn build_policy(
    backend: &str,
    runtime_quant: RuntimeQuant,
    moe_request: Mxfp4MoeRequest,
) -> Result<Policy> {
    let (_bound_projections, naming) = bind_policy(backend)?;
    Ok(Policy {
        projections: Projections::InPlace,
        naming,
        runtime_quant,
        moe_request,
        ..Policy::default()
    })
}

/// The quantized part of a source encoding summary, when there is one.
///
/// [`SOURCE_ENCODING_KEY`] is the sorted distinct set, each member tagged with
/// the kind the importer knew it to be — `raw:bf16`, `quant:q4_0`,
/// `quant:q4_k,quant:q6_k`. So "was any of it quantized" is answered by
/// reading, and this function needs no table and holds no opinion.
///
/// It used to have both: it carried the raw dtype spellings and treated
/// anything else as quantized. That inverted the risk rather than removing it
/// — closed against a new *scheme*, wide open against a new or renamed
/// `DType`, which would have read as quantized and advised the operator
/// against a second rounding of a checkpoint that was never rounded once.
/// Nothing would have failed. The classification belongs where the `Encoding`
/// is in hand, so it moved to `import::source_encoding`, and this reads what
/// that decided.
///
/// An untagged summary is one an older pie wrote. It yields `None` — no notice
/// — rather than a guess: the notice is advice, and advice inferred from a
/// format that did not carry the fact is worse than silence. Re-import and it
/// is tagged.
fn quantized_source(summary: &str) -> Option<String> {
    let quantized: Vec<&str> = summary
        .split(',')
        .filter_map(|part| part.trim().strip_prefix("quant:"))
        .filter(|name| !name.is_empty())
        .collect();
    (!quantized.is_empty()).then(|| quantized.join("/"))
}

/// Everything `pie model build` is told about a requantization.
///
/// Grouped into one value because they are one question. See
/// [`resolve_quant`].
pub struct QuantRequest<'a> {
    /// `--quant` as the operator spelled it, for the message that quotes it.
    pub spelled: Option<&'a str>,
    pub fp8_native: bool,
    pub backend: &'a str,
    /// `--quant` a previous build already baked into this source, if any.
    pub baked_quant: Option<&'a str>,
    /// How to name the source in a message. The operator may have several
    /// artifacts of one model, so the refusal has to say WHICH one is baked.
    pub source_label: &'a str,
    /// [`SOURCE_ENCODING_KEY`] off the source artifact: how the checkpoint it
    /// was imported from stored its numbers.
    pub source_encoding: Option<&'a str>,
}

/// What a build should do about `--quant`, decided once.
#[derive(Debug)]
pub struct ResolvedQuant {
    /// The scheme authoring will actually apply. `None` when the request was
    /// dropped, which is a decision and not a failure.
    pub quant: RuntimeQuant,
    /// Lines to print, without the `optimize: ` prefix.
    pub notices: Vec<String>,
}

/// Decides what `--quant` means for this source, on this backend, once.
///
/// # Why this is one function and was four
///
/// "May these weights be requantized into this scheme?" used to be answered
/// in four places, each from a different source of truth and at a different
/// time, which is why the answers did not agree:
///
/// - `quant_fits` — backend x scheme, a table, before authoring. Correct.
/// - the second-rounding notice — `pie_source_encoding`, before authoring,
///   advisory only.
/// - `Builder::runtime_quant_scheme` — the model's `config.json`, DURING
///   authoring. Structurally unable to fire for an imported archive, which
///   never carries a `quantization_config`; its own comment said so.
/// - `Builder::push_runtime_quant` — a per-tensor dtype whitelist, during
///   authoring, one tensor at a time. It is what actually stopped
///   `--quant fp8 --fp8-native` against a q4_0 archive, and it said
///   "must be BF16/FP16/FP32/F8E4M3" — a true sentence about dtypes that
///   names neither the cause nor the fix.
///
/// The three facts needed to answer properly — what was asked, what the
/// backend reads, how the source stored its numbers — are all in hand HERE,
/// at the CLI boundary, and only two of them are ever in hand inside the
/// builder. So the decision belongs here, and the builder should receive a
/// policy that is already true and merely execute it.
///
/// That the source's encoding is available at all is recent: until an archive
/// kept the packing its source shipped, `pie model import` decoded every block
/// to BF16 and the fact was destroyed on the way in.
fn resolve_quant(req: &QuantRequest<'_>) -> Result<ResolvedQuant> {
    let quant = RuntimeQuant::resolve(req.spelled.unwrap_or(""), req.fp8_native)
        .map_err(|err| anyhow!("--quant: {err}"))?;
    let mut notices = Vec::new();
    // Quantizing weights that are already quantized rounds them twice, and
    // nothing about the weights themselves says which is which -- an FP8
    // tensor a checkpoint shipped and one this command wrote look identical.
    // The source artifact is asked instead, because it is the only party that
    // knows. Re-laying without `--quant` stays available: it is the layout
    // work alone, and it touches no numbers.
    if let Some(baked) = req.baked_quant
        && quant != RuntimeQuant::None
    {
        bail!(
            "{} was already built with --quant {baked}; quantizing it again \
             would round the same weights a second time. Build from the \
             imported artifact or the snapshot instead, or drop --quant to \
             re-lay this one unchanged",
            req.source_label
        );
    }
    // THE SAME ROUNDING, ONE LAYER FURTHER BACK.
    //
    // The refusal above catches a build of a build. This catches the case it
    // cannot see: a build of an IMPORT of an already-quantized checkpoint.
    // `q4_0 -> bf16 -> fp8` rounds the same weights twice, and the second
    // rounding is not recoverable.
    //
    // **Whether that is refused depends on whether the operator has another
    // move, and only this function knows.** On CUDA they do: the blocks decode
    // to BF16 at build and every kernel binds them, so dropping `--quant`
    // serves the model at the precision it was shipped at. Refusing there
    // costs nothing and prevents a silent quality loss.
    //
    // On Metal, Vulkan and wgpu they do not. Measured: `--backend vulkan`
    // without `--quant int4` refuses at authoring -- "the Metal driver binds
    // every projection through its affine-U4 path" -- so int4 is not one way
    // to serve a GGUF there, it is the only way. Refusing would not protect
    // the numbers; it would take the model away. So it is said, not refused,
    // and the operator's real choice -- a different source repo -- is named.
    if quant != RuntimeQuant::None
        && let Some(was) = req.source_encoding
        && let Some(quantized) = quantized_source(was)
    {
        let spelling = format!("{quant:?}").to_lowercase();
        if requantization_is_the_only_way_to_serve(req.backend) {
            notices.push(format!(
                "note -- these weights were {quantized} in the checkpoint they \
                 were imported from, so --quant {spelling} rounds them a second \
                 time. This backend binds every projection through its affine-U4 \
                 path, so there is no build without it; a BF16 or already-4-bit \
                 source would round once"
            ));
        } else {
            bail!(
                "these weights were {quantized} in the checkpoint they were \
                 imported from, so --quant {spelling} would round them a second \
                 time. {} binds them as they are -- drop --quant to serve them \
                 at the precision they were shipped at, or build from a BF16 \
                 source to round once",
                req.backend
            );
        }
    }
    if req.spelled == Some("fp8") && quant == RuntimeQuant::None {
        notices.push("--quant fp8 without --fp8-native is dropped, as serve would drop it".into());
    }
    // Which driver serves this decides which tensors exist, so it is resolved
    // before anything is authored. A requantization is only real if that
    // driver's kernels read what it produces, so the two flags are checked
    // against each other rather than each alone: refusing here is the
    // difference between an artifact that cannot be bound and one that is
    // quietly the wrong numbers.
    quant_fits(req.backend, quant, req.spelled)?;
    Ok(ResolvedQuant { quant, notices })
}

/// Whether this backend can only serve through a requantization.
///
/// Metal, Vulkan and wgpu bind every projection through the MLX affine-U4
/// path, so `--quant int4` is not one way to serve a model there but the only
/// one -- measured: `--backend vulkan` without it refuses at authoring, naming
/// the missing `.scales`. CUDA binds BF16 directly, so `--quant` is optional
/// there and dropping it is a real answer an operator can be given.
///
/// This is the same fact [`quant_fits`] holds, asked the other way round, and
/// it stays two functions on purpose: one refuses a scheme the backend cannot
/// read, the other decides whether an operator has an alternative. Folding
/// them together would put "what can this bind" and "what should we tell them"
/// in one table.
fn requantization_is_the_only_way_to_serve(backend: &str) -> bool {
    matches!(backend, "metal" | "vulkan" | "wgpu")
}

/// Refuses a requantization the serving backend's kernels do not read.
fn quant_fits(backend: &str, quant: RuntimeQuant, spelled: Option<&str>) -> Result<()> {
    match (backend, quant) {
        (
            "metal" | "vulkan" | "wgpu",
            RuntimeQuant::Fp8 | RuntimeQuant::Int8 | RuntimeQuant::Mxfp4,
        ) => {
            bail!(
                "--quant {} is CUDA's; this backend's matvecs read MLX affine, so \
                 `--quant int4` is the requantization it can serve",
                spelled.unwrap_or("")
            )
        }
        ("cuda", RuntimeQuant::Int4) => bail!(
            "--quant int4 is MLX affine, which no CUDA kernel reads; it is \
             what `--backend metal`, `--backend vulkan` and `--backend wgpu` \
             requantize to"
        ),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req<'a>(spelled: Option<&'a str>, backend: &'a str) -> QuantRequest<'a> {
        QuantRequest {
            spelled,
            fp8_native: false,
            backend,
            baked_quant: None,
            source_encoding: None,
            source_label: "artifact.zt",
        }
    }

    /// What `--quant` means is one decision, so it has one place and these
    /// tests are that place's whole surface. They run without a GPU, without
    /// a checkpoint and without authoring anything, which is the point: the
    /// answers used to be discoverable only by building a model on a machine
    /// with the right driver, and two of the four answers disagreed.
    #[test]
    fn a_quant_request_is_resolved_from_the_three_facts_that_decide_it() {
        // Backend x scheme, which was always right.
        assert!(resolve_quant(&req(Some("int4"), "cuda")).is_err());
        assert!(resolve_quant(&req(Some("int4"), "vulkan")).is_ok());
        assert!(resolve_quant(&req(Some("int8"), "vulkan")).is_err());

        // A spelling nothing reads is a refusal, not a silent drop.
        assert!(resolve_quant(&req(Some("int2"), "cuda")).is_err());

        // fp8 without --fp8-native resolves to None and SAYS so. Serve would
        // drop it anyway, so building it in would be a lie about the file.
        let dropped = resolve_quant(&req(Some("fp8"), "cuda")).unwrap();
        assert_eq!(dropped.quant, RuntimeQuant::None);
        assert!(dropped.notices.iter().any(|n| n.contains("is dropped")));

        // The same flag with the flag it needs keeps the scheme and says
        // nothing, there being nothing to say.
        let native = QuantRequest {
            fp8_native: true,
            ..req(Some("fp8"), "cuda")
        };
        let kept = resolve_quant(&native).unwrap();
        assert_eq!(kept.quant, RuntimeQuant::Fp8);
        assert!(kept.notices.is_empty());
    }

    /// An FP8 checkpoint asked for FP8 is a second rounding like any other.
    ///
    /// It reads as a special case and is not one, but it took a detour to get
    /// here. This was decided in `Builder::runtime_quant_scheme`, which
    /// answered `Ok(None)` -- authoring an unrounded artifact for an operator
    /// who had asked for a rounded one, and saying nothing. That guard is
    /// gone, so this refusal is now the only thing standing between GLM-5.1's
    /// FP8 experts and a silent re-encode.
    ///
    /// It only fires because `import::source_encoding` files an FP8 dtype
    /// under `quant:`. FP8 is stored as a dtype rather than a `Quant` scheme,
    /// so the obvious reading of the `Encoding` variant calls an
    /// already-rounded checkpoint raw. Those two lines are one change.
    #[test]
    fn an_fp8_source_is_refused_a_second_fp8_rounding() {
        let already = QuantRequest {
            source_encoding: Some("quant:f8e4m3,raw:bf16"),
            fp8_native: true,
            ..req(Some("fp8"), "cuda")
        };
        let err = resolve_quant(&already).expect_err("cuda serves fp8 as it ships");
        let msg = err.to_string();
        assert!(msg.contains("f8e4m3"), "does not name the source: {msg}");

        // The negative control, and the reason this is a refusal on CUDA and
        // not everywhere: a BF16 source has been rounded zero times, so there
        // is nothing to refuse.
        let fresh = QuantRequest {
            source_encoding: Some("raw:bf16"),
            fp8_native: true,
            ..req(Some("fp8"), "cuda")
        };
        let ok = resolve_quant(&fresh).expect("a bf16 source may be rounded once");
        assert_eq!(ok.quant, RuntimeQuant::Fp8);
    }

    /// The source's own encoding is the third fact, and the newest: before an
    /// archive kept the packing it was shipped with, import decoded every
    /// block to BF16 and there was nothing here to read.
    ///
    /// What it decides is not "is this allowed" but "does the operator have
    /// another move", which is why the same source gets two answers.
    #[test]
    fn a_source_that_was_already_quantized_is_told_about_and_not_refused() {
        let from_gguf = QuantRequest {
            source_encoding: Some("quant:q4_0,quant:q8_0"),
            ..req(Some("int4"), "vulkan")
        };
        let resolved = resolve_quant(&from_gguf).unwrap();
        // NOT refused, and that is deliberate: Vulkan, Metal and wgpu bind
        // every projection through the affine-U4 path, so refusing here does
        // not protect the numbers -- it takes GGUF models away from those
        // backends entirely.
        assert_eq!(resolved.quant, RuntimeQuant::Int4);
        let notice = resolved.notices.join(" ");
        assert!(notice.contains("q4_0/q8_0"), "{notice}");
        assert!(notice.contains("second time"), "{notice}");

        // A source that was never quantized draws no notice. This is the
        // negative control: without it the assertion above passes for a
        // function that warns unconditionally.
        let from_bf16 = QuantRequest {
            source_encoding: Some("raw:bf16"),
            ..req(Some("int4"), "vulkan")
        };
        assert!(resolve_quant(&from_bf16).unwrap().notices.is_empty());

        // An untagged summary is one an older pie wrote, and it yields
        // silence rather than a guess.
        let legacy = QuantRequest {
            source_encoding: Some("q4_0"),
            ..req(Some("int4"), "vulkan")
        };
        assert!(resolve_quant(&legacy).unwrap().notices.is_empty());

        // And without `--quant` there is no second rounding to warn about,
        // however the source was stored.
        let relay = QuantRequest {
            source_encoding: Some("quant:q4_0"),
            ..req(None, "vulkan")
        };
        assert!(resolve_quant(&relay).unwrap().notices.is_empty());
    }

    /// The same source and the same second rounding, on a backend that can
    /// serve without it -- and there it is refused.
    ///
    /// This is the case that sent the user asking. It was already refused
    /// before this, but by `Builder::push_runtime_quant`'s dtype whitelist,
    /// one tensor at a time, after the model was open, saying only
    /// "runtime_quant source '...down_proj.weight' must be
    /// BF16/FP16/FP32/F8E4M3" -- which names a dtype, not the fact that the
    /// weights were already 4-bit, and offers no way forward.
    #[test]
    fn requantizing_a_quantized_source_is_refused_where_the_model_serves_without_it() {
        let to_fp8 = QuantRequest {
            fp8_native: true,
            source_encoding: Some("quant:q4_0,quant:q8_0"),
            ..req(Some("fp8"), "cuda")
        };
        let err = resolve_quant(&to_fp8).unwrap_err().to_string();
        assert!(err.contains("q4_0/q8_0"), "{err}");
        assert!(err.contains("drop --quant"), "{err}");

        // The negative control the refusal above needs: a source that was
        // never quantized requantizes on the same backend without complaint.
        // Without this the assertion passes for a function that refuses every
        // cuda requantization.
        let from_bf16 = QuantRequest {
            fp8_native: true,
            source_encoding: Some("raw:bf16"),
            ..req(Some("fp8"), "cuda")
        };
        let ok = resolve_quant(&from_bf16).unwrap();
        assert_eq!(ok.quant, RuntimeQuant::Fp8);
        assert!(ok.notices.is_empty());
    }

    /// A build of a build is refused; the message names the artifact because
    /// an operator may hold several of one model.
    #[test]
    fn requantizing_an_artifact_that_was_already_quantized_is_refused() {
        let again = QuantRequest {
            baked_quant: Some("int4"),
            ..req(Some("int4"), "vulkan")
        };
        let err = resolve_quant(&again).unwrap_err().to_string();
        assert!(err.contains("artifact.zt"), "{err}");
        assert!(err.contains("already built with --quant int4"), "{err}");

        // Re-laying the same artifact without `--quant` stays available: it
        // is layout work and touches no numbers.
        let relay = QuantRequest {
            baked_quant: Some("int4"),
            ..req(None, "vulkan")
        };
        assert_eq!(resolve_quant(&relay).unwrap().quant, RuntimeQuant::None);
    }

    /// The flag decides which tensors exist, so every accepted spelling has
    /// to name a bind path that reads them -- and Vulkan's is Metal's.
    #[test]
    fn a_vulkan_artifact_is_authored_the_way_a_vulkan_bind_reads_it() {
        assert_eq!(
            bind_policy("vulkan").unwrap(),
            (Projections::InPlace, Naming::Mlx)
        );
        assert_eq!(
            bind_policy("vulkan").unwrap(),
            bind_policy("metal").unwrap()
        );
        assert_eq!(
            bind_policy("cuda").unwrap(),
            (Projections::Fused, Naming::Hf)
        );
        let refused = bind_policy("rocm").unwrap_err().to_string();
        assert!(refused.contains("vulkan"), "got: {refused}");
    }

    /// A BUILD DOES NOT PERSIST A FUSION, ON ANY BACKEND.
    ///
    /// The one place the artifact deliberately disagrees with the bind path,
    /// so it is pinned with its own control: `bind_policy` must still say CUDA
    /// binds fused, because that is true and is what the load path does. What
    /// must not come back is writing that fusion down — it duplicates every
    /// projection it aliases, 587,202,560 bytes on Qwen3-0.6B, on disk and
    /// again in VRAM. See [`build_policy`].
    #[test]
    fn a_build_does_not_persist_a_fusion() {
        for backend in ["cuda", "metal", "vulkan", "wgpu"] {
            let policy = build_policy(backend, RuntimeQuant::None, Mxfp4MoeRequest::Auto).unwrap();
            assert_eq!(
                policy.projections,
                Projections::InPlace,
                "a {backend} build would persist a fused bank beside the \
                 projections it is a view of",
            );
            assert_eq!(
                policy.naming,
                bind_policy(backend).unwrap().1,
                "naming IS a fact about which tensors exist, so a build must \
                 keep honouring the bind path for it",
            );
        }
        assert_eq!(
            bind_policy("cuda").unwrap().0,
            Projections::Fused,
            "the control: CUDA still BINDS a fused bank, and a build that \
             stopped saying so would be describing a different load path",
        );
    }

    /// WHAT COUNTS AS "ALREADY QUANTIZED", AND WHY IT IS SPELLED IN REVERSE.
    ///
    /// The predicate behind the double-rounding notice. Listing the RAW dtypes
    /// and treating everything else as quantized is what keeps it closed: the
    /// set of quantization schemes grows (five IQ lattice decoders landed in
    /// one commit), the set of raw dtypes does not, so a scheme added tomorrow
    /// is caught without anyone remembering to come back here.
    #[test]
    fn a_source_that_was_quantized_is_recognised_however_it_was_spelled() {
        assert_eq!(quantized_source("raw:bf16"), None);
        assert_eq!(quantized_source("raw:bf16,raw:f32"), None);
        assert_eq!(quantized_source(""), None);
        assert_eq!(quantized_source("quant:q4_0"), Some("q4_0".to_string()));
        // The normal GGUF case: llama.cpp keeps some tensors at a wider
        // scheme, and which ones were coarse is what a second rounding
        // compounds — so the whole set is reported, not a dominant one.
        assert_eq!(
            quantized_source("raw:bf16,quant:q4_k,quant:q6_k"),
            Some("q4_k/q6_k".to_string()),
        );
        // A scheme this test has never heard of still counts.
        assert_eq!(
            quantized_source("quant:iq2_xxs"),
            Some("iq2_xxs".to_string())
        );
        // And so does a DTYPE it has never heard of, in the other direction.
        // This is the case the list this replaced got wrong: it knew thirteen
        // raw spellings and called the fourteenth a quantization.
        assert_eq!(quantized_source("raw:some_future_float"), None);
        // An archive an older pie wrote carries no tags. Silence, not a guess.
        assert_eq!(quantized_source("bf16,q4_k"), None);
    }

    /// A requantization the serving kernels cannot read is refused, and the
    /// one they can is not. Vulkan's matvecs are the affine-U4 path, so it
    /// takes `int4` and refuses CUDA's three.
    #[test]
    fn the_quant_a_vulkan_bind_cannot_read_is_refused() {
        assert!(quant_fits("vulkan", RuntimeQuant::Int4, Some("int4")).is_ok());
        assert!(quant_fits("vulkan", RuntimeQuant::None, None).is_ok());
        for (quant, spelled) in [
            (RuntimeQuant::Fp8, "fp8"),
            (RuntimeQuant::Int8, "int8"),
            (RuntimeQuant::Mxfp4, "mxfp4"),
        ] {
            let refused = quant_fits("vulkan", quant, Some(spelled))
                .unwrap_err()
                .to_string();
            assert!(refused.contains(spelled), "got: {refused}");
        }
        assert!(quant_fits("cuda", RuntimeQuant::Int4, Some("int4")).is_err());
        assert!(quant_fits("cuda", RuntimeQuant::Fp8, Some("fp8")).is_ok());
    }

    /// wgpu takes the same artifact as Vulkan, and the reason is checkable.
    ///
    /// # Why this reads two other crates instead of trusting the arm
    ///
    /// `bind_policy` puts `wgpu` in Metal's arm because `driver-wgpu` binds
    /// the tensors `driver-vulkan` binds. That is a claim about ANOTHER crate,
    /// and an arm that merely says so would keep saying so after the two
    /// parted -- an artifact authored for the wrong names does not fail to
    /// build, it fails at `Shell::open` with a missing weight, one command
    /// later and in a different crate's words.
    ///
    /// This used to diff driver-wgpu/src/names.rs against
    /// driver-vulkan/src/names.rs, which were byte-identical copies -- both
    /// gone, so neither is backticked. They are now one file,
    /// `driver/src/names.rs`, which both shells re-export -- so the equality
    /// this arm rests on stopped being a thing to check and became a thing
    /// that holds by construction.
    ///
    /// What is left to check is that it STAYS one file. A shell that grows a
    /// name table of its own has silently reacquired the right to spell
    /// weights differently, and the first symptom would again be a missing
    /// weight at `Shell::open`.
    #[test]
    fn wgpu_is_authored_the_way_vulkan_is_because_it_binds_the_same_names() {
        assert_eq!(
            bind_policy("wgpu").unwrap(),
            bind_policy("vulkan").unwrap(),
            "the flag must author what the wgpu bind path reads"
        );
        assert!(quant_fits("wgpu", RuntimeQuant::Int4, Some("int4")).is_ok());
        assert!(quant_fits("wgpu", RuntimeQuant::Fp8, Some("fp8")).is_err());

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let shared = std::fs::read_to_string(root.join("crates/driver/src/names.rs"))
            .expect("the one name table both shells read");
        // A control: the equality below is about a real table rather than
        // about two files that both failed to be there.
        assert!(
            shared.contains("lm_head"),
            "the table read is the real one, not an empty file"
        );
        for shell in ["driver-wgpu", "driver-vulkan"] {
            assert!(
                !root.join(format!("crates/{shell}/src/names.rs")).exists(),
                "{shell} grew a name table of its own, so one `--backend` arm \
                 can no longer author for both"
            );
            let lib = std::fs::read_to_string(root.join(format!("crates/{shell}/src/lib.rs")))
                .expect("a shell has a lib.rs");
            assert!(
                lib.contains("pub use driver::names;"),
                "{shell} no longer re-exports the shared table"
            );
        }
    }

    /// The refusal names every spelling it would have taken.
    ///
    /// A user who typed `--backend webgpu` learns that `wgpu` exists; one who
    /// learns only that `webgpu` is wrong tries `--backend gpu` next.
    #[test]
    fn an_unknown_backend_is_refused_by_listing_the_known_ones() {
        let refused = bind_policy("webgpu").unwrap_err().to_string();
        for known in ["cuda", "metal", "vulkan", "wgpu"] {
            assert!(
                refused.contains(known),
                "the refusal omits `{known}`: {refused}"
            );
        }
    }
}
