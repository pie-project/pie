//! `pie model build` — precompute a serve boot, offline.
//!
//! The command `author_abi.cpp` was built to serve, finally landed the way
//! the migration made possible: no FFI at all. The same family author a
//! driver boot runs (`pie_model::contract::author`) writes the serve
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
//! CUDA binds fused q/k/v banks under HuggingFace names, Metal binds in-place
//! projections under MLX names, and an artifact materialized for one is not
//! what the other's bind path reads. It defaults to `cuda`, which is what the
//! policy silently was before the flag existed.
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

use pie_loader::checkpoint::meta::{SOURCE_KEY, VERSION_KEY, meta_name};
use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::write::CheckpointWriter;
use pie_loader::executor::host::Progress;
use pie_loader::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use pie_loader::types::Visibility;
use pie_model_common::facts::ModelFacts;
use pie_model_common::policy::{Mxfp4MoeRequest, Naming, Policy, Projections, RuntimeQuant};
use pie_model_config::DESCRIPTOR_OBJECT;

use super::import::{
    ProgressLine, Spool, artifact_path, compile_descriptor, compile_tokenizer, pie_version,
    resolve_source, store_path,
};

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// What to optimize: a HuggingFace repo ID in the local cache, a snapshot
    /// directory, or a `.zt` artifact.
    pub source: String,
    /// Load-time requantization to bake in: `fp8`, `int8` or `mxfp4` for
    /// `--backend cuda`, `int4` for `--backend metal`. Absent means none — the
    /// optimization is then the layout work alone (fused banks, expert stacks,
    /// dequantized schemes).
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
    /// Which driver will serve the artifact: `cuda` or `metal`.
    ///
    /// Not cosmetic and not inferable: the two drivers read different tensors.
    /// CUDA binds fused q/k/v banks under HuggingFace names, Metal binds
    /// in-place projections under MLX names, and an artifact materialized for
    /// one is not the artifact the other's bind path reads. Stated as a flag
    /// for the reason `--fp8-native` is: an offline run cannot probe the
    /// device it is optimizing for.
    #[arg(long, default_value = "cuda")]
    pub backend: String,
    /// Write the artifact here instead of the store. A path ending in `.zt`
    /// is the artifact; a directory receives `<name>-optimized.zt`.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Report what would be done without doing it.
    #[arg(long)]
    pub dry_run: bool,
}

/// pie's own objects, lifted out of an artifact being re-optimized.
///
/// An artifact is the one source that arrives with its metadata already
/// compiled, and `compile_descriptor` / `compile_tokenizer` both answer `None`
/// for it — correctly, since there is no `config.json` or `tokenizer.json` to
/// compile. Writing the output without them would silently produce an artifact
/// that cannot serve, so they are carried across verbatim instead.
struct CarriedObjects {
    descriptor: serde_json::Value,
    descriptor_bytes: Vec<u8>,
    tokenizer: Vec<(String, Vec<u8>)>,
}

fn read_carried_objects(path: &Path) -> Result<CarriedObjects> {
    let checkpoint = parse_checkpoint_metadata(path)
        .map_err(|err| anyhow!("cannot read {}: {err}", path.display()))?;
    let descriptor_bytes =
        pie_loader::checkpoint::read::read_meta(&checkpoint, DESCRIPTOR_OBJECT)?.ok_or_else(
            || {
                anyhow!(
                    "{} carries no {DESCRIPTOR_OBJECT}; it is a checkpoint file rather \
                     than a pie artifact, and optimize needs the normalized config",
                    path.display()
                )
            },
        )?;
    let descriptor: serde_json::Value = serde_json::from_slice(&descriptor_bytes)
        .map_err(|err| anyhow!("cannot parse {}'s model descriptor: {err}", path.display()))?;
    let mut tokenizer = Vec::with_capacity(pie_tokenizer::canonical::OBJECTS.len());
    for name in pie_tokenizer::canonical::OBJECTS {
        let bytes = pie_loader::checkpoint::read::read_meta(&checkpoint, name)?.ok_or_else(
            || {
                anyhow!(
                    "{} carries a model descriptor but not {name}; an artifact with half \
                     its metadata cannot serve, and this command does not compile the rest",
                    path.display()
                )
            },
        )?;
        tokenizer.push((name.to_string(), bytes));
    }
    Ok(CarriedObjects {
        descriptor,
        descriptor_bytes,
        tokenizer,
    })
}

pub fn run(args: BuildArgs) -> Result<crate::ui::Answer> {
    let source = resolve_source(&args.source)?;
    // Two kinds of source, and the difference is only where the descriptor
    // comes from: an artifact carries it compiled, and a snapshot is
    // normalized into one here by the same `pie-model-config` that wrote the
    // artifact's. Both then reach the author through one projection.
    //
    // There were two readers here — one per source — and each defaulted and
    // probed on its own. `ModelFacts::from_descriptor` is the single one now,
    // and it lives beside the authors that read the facts rather than in this
    // command.
    let carried = if source.path.is_file() {
        Some(read_carried_objects(&source.path)?)
    } else {
        None
    };
    let descriptor = match &carried {
        Some(objects) => objects.descriptor_bytes.clone(),
        None => compile_descriptor(&source)?.ok_or_else(|| {
            anyhow!("build needs a snapshot with a config.json to normalize")
        })?,
    };
    let facts = ModelFacts::from_descriptor(&descriptor)
        .map_err(|err| anyhow!("cannot read the compiled model descriptor: {err}"))?;

    let runtime_quant = RuntimeQuant::resolve(args.quant.as_deref().unwrap_or(""), args.fp8_native)
        .map_err(|err| anyhow!("--quant: {err}"))?;
    if args.quant.as_deref() == Some("fp8") && runtime_quant == RuntimeQuant::None {
        println!("optimize: --quant fp8 without --fp8-native is dropped, as serve would drop it");
    }
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
    // Which driver serves this decides which tensors exist, so it is resolved
    // before anything is authored. The pair moves together -- there is no
    // driver that wants MLX names over fused banks -- so one flag sets both
    // and no combination can be spelled that no bind path reads.
    let (projections, naming) = match args.backend.as_str() {
        "cuda" => (Projections::Fused, Naming::Hf),
        "metal" => (Projections::InPlace, Naming::Mlx),
        other => bail!("--backend {other:?} is not `cuda` or `metal`"),
    };
    // A requantization is only real if the serving driver's kernels read what
    // it produces, so the two flags are checked against each other rather than
    // each alone. Refusing here is the difference between an artifact that
    // cannot be bound and one that is quietly the wrong numbers.
    match (args.backend.as_str(), runtime_quant) {
        ("metal", RuntimeQuant::Fp8 | RuntimeQuant::Int8 | RuntimeQuant::Mxfp4) => bail!(
            "--quant {} is CUDA's; Metal's matvecs read MLX affine, so `--quant int4` \
             is the requantization this backend can serve",
            args.quant.as_deref().unwrap_or("")
        ),
        ("cuda", RuntimeQuant::Int4) => bail!(
            "--quant int4 is MLX affine, which no CUDA kernel reads; it is \
             `--backend metal`'s requantization"
        ),
        _ => {}
    }
    let policy = Policy {
        projections,
        naming,
        runtime_quant,
        moe_request,
        ..Policy::default()
    };
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
    let contract = pie_model::contract::author(&facts, &metadata, &target, &policy)
        .map_err(|err| anyhow!("cannot author '{}': {err}", facts.model_type))?
        .ok_or_else(|| anyhow!("no contract author for model_type '{}'", facts.model_type))?;
    if !contract.groups.is_empty() {
        bail!(
            "the '{}' contract declares streamed expert groups, which are a paging \
             decision; materializing them eagerly would build the residency they avoid",
            facts.model_type
        );
    }
    let public = contract
        .tensors
        .iter()
        .filter(|tensor| tensor.visibility == Visibility::Public)
        .count();
    println!(
        "optimize: {} declares {} tensors ({} bound) for {}, quant={:?}, moe={:?}",
        facts.model_type,
        contract.tensors.len(),
        public,
        args.backend,
        policy.runtime_quant,
        policy.moe_request,
    );

    // `-optimized` on both branches. It was on the store branch only, so
    // `--out <a directory>` wrote `<name>.zt` -- which is exactly the name
    // `pie model import` gives the plain converted artifact. Pointed at the
    // store, `--out` therefore replaced a portable checkpoint with one whose
    // tensors are this build's runtime layout, under a name that still said
    // otherwise.
    let optimized = format!("{}-optimized", source.name);
    let out_file = match &args.out {
        Some(out) => artifact_path(out, &optimized),
        None => store_path(&optimized),
    };
    if args.dry_run {
        return Ok(crate::ui::Answer::noop(format!(
            "dry run: would write {}",
            crate::ui::short_path(&out_file)
        )));
    }

    let plan = pie_loader::plan::compile(&metadata, &contract, target)
        .map_err(|err| anyhow!("cannot compile: {err}"))?;

    // Metadata first, weights streamed after — the same shape convert has.
    // The descriptor was compiled above, because the facts were read from it;
    // the artifact carries that same document rather than a second
    // normalization of the same file.
    let tokenizer = compile_tokenizer(&source)?;

    let mut bar = ProgressLine::new();
    let mut spool = Spool::create(&out_file)?;
    pie_loader::executor::host::execute_plan_into(
        &plan,
        &source.base(),
        &mut spool,
        &mut |progress| {
            bar.render(&Progress {
                read_bytes: progress.read_bytes,
                total_read_bytes: progress.total_read_bytes,
                finalized: progress.finalized,
            });
        },
    )
    .map_err(|err| anyhow!("materializing failed: {err}"))?;

    let provenance = BTreeMap::from([
        (VERSION_KEY.to_string(), pie_version().to_string()),
        (
            SOURCE_KEY.to_string(),
            format!("optimize:{}", source.origin),
        ),
    ]);
    let mut writer = CheckpointWriter::create(&out_file, &provenance)
        .map_err(|err| anyhow!("cannot write the artifact: {err}"))?;

    // One ascending pass over metadata objects and runtime tensors together,
    // which is what canonical form asks for.
    enum Entry<'a> {
        Meta(&'a [u8]),
        Tensor(&'a pie_loader::types::TensorDecl),
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
            meta.push((
                meta_name(DESCRIPTOR_OBJECT),
                objects.descriptor_bytes.clone(),
            ));
        }
        None => {
            if let Some(canonical) = &tokenizer {
                for (path, bytes) in canonical.objects() {
                    meta.push((meta_name(path), bytes.to_vec()));
                }
            }
            // Not conditional: the run could not have authored anything
            // without this document, so by here it exists.
            meta.push((meta_name(DESCRIPTOR_OBJECT), descriptor.clone()));
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
                    .strip_prefix(pie_loader::checkpoint::meta::META_PREFIX)
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
