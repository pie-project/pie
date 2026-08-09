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
use model_loader::checkpoint::meta::{SOURCE_KEY, VERSION_KEY, meta_name};
use model_loader::checkpoint::read::parse_checkpoint_metadata;
use model_loader::checkpoint::write::CheckpointWriter;
use model_loader::executor::Progress;
use model_loader::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use model_loader::types::Visibility;

use super::import::{
    ProgressLine, Spool, artifact_path, carry_config, compile_tokenizer, pie_version,
    resolve_source, store_path,
};

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// What to optimize: a HuggingFace repo ID in the local cache, a snapshot
    /// directory, or a `.zt` artifact.
    pub source: String,
    /// Load-time requantization to bake in: `fp8`, `int8` or `mxfp4` for
    /// `--backend cuda`, `int4` for `--backend metal` or `--backend vulkan`. Absent means none — the
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
    /// Which driver will serve the artifact: `cuda`, `metal` or `vulkan`.
    ///
    /// Not cosmetic and not inferable: the drivers read different tensors.
    /// CUDA binds fused q/k/v banks under HuggingFace names; Metal and Vulkan
    /// bind in-place projections under MLX names, and an artifact materialized
    /// for one family is not what the other's bind path reads. Stated as a flag
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
}

fn read_carried_objects(path: &Path) -> Result<CarriedObjects> {
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
    })
}

pub fn run(args: BuildArgs) -> Result<crate::ui::Answer> {
    let source = resolve_source(&args.source)?;
    // Two kinds of source, and the difference is only where the config comes
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
    let (projections, naming) = bind_policy(&args.backend)?;
    // A requantization is only real if the serving driver's kernels read what
    // it produces, so the two flags are checked against each other rather than
    // each alone. Refusing here is the difference between an artifact that
    // cannot be bound and one that is quietly the wrong numbers.
    quant_fits(&args.backend, runtime_quant, args.quant.as_deref())?;
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

    let plan = model_loader::plan::compile(&metadata, &contract, target)
        .map_err(|err| anyhow!("cannot compile: {err}"))?;

    // Metadata first, weights streamed after — the same shape convert has.
    // The config was read above for its quantization; the artifact carries
    // those same bytes rather than a second reading of the same file.
    let tokenizer = compile_tokenizer(&source)?;

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
/// Vulkan reads exactly what Metal reads. The two drivers share no code, but
/// they share a bind path shape: MLX names over in-place projections, matvecs
/// over MLX affine int4. One arm rather than a second policy that happened to
/// be equal, which would be a second thing to keep equal.
fn bind_policy(backend: &str) -> Result<(Projections, Naming)> {
    match backend {
        "cuda" => Ok((Projections::Fused, Naming::Hf)),
        "metal" | "vulkan" => Ok((Projections::InPlace, Naming::Mlx)),
        other => bail!("--backend {other:?} is not `cuda`, `metal` or `vulkan`"),
    }
}

/// Refuses a requantization the serving backend's kernels do not read.
fn quant_fits(backend: &str, quant: RuntimeQuant, spelled: Option<&str>) -> Result<()> {
    match (backend, quant) {
        ("metal" | "vulkan", RuntimeQuant::Fp8 | RuntimeQuant::Int8 | RuntimeQuant::Mxfp4) => {
            bail!(
                "--quant {} is CUDA's; this backend's matvecs read MLX affine, so \
                 `--quant int4` is the requantization it can serve",
                spelled.unwrap_or("")
            )
        }
        ("cuda", RuntimeQuant::Int4) => bail!(
            "--quant int4 is MLX affine, which no CUDA kernel reads; it is \
             what `--backend metal` and `--backend vulkan` requantize to"
        ),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
