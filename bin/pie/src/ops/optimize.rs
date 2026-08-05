//! `pie model optimize` — precompute a serve boot, offline.
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
//! `pie model convert` stays the family-blind sibling: it normalizes
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

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;

use pie_loader::checkpoint::meta::{SOURCE_KEY, VERSION_KEY, meta_name};
use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::write::CheckpointWriter;
use pie_loader::executor::host::Progress;
use pie_loader::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use pie_loader::types::Visibility;
use pie_model::common::facts::ModelFacts;
use pie_model::common::policy::{Mxfp4MoeRequest, Policy, RuntimeQuant};
use pie_model_config::DESCRIPTOR_OBJECT;

use super::convert::{
    ProgressLine, Spool, artifact_path, compile_descriptor, compile_tokenizer, pie_version,
    resolve_source, store_path,
};

#[derive(Args, Debug)]
pub struct OptimizeArgs {
    /// What to optimize: a HuggingFace repo ID in the local cache, a snapshot
    /// directory, or a `.zt` artifact.
    pub source: String,
    /// Load-time requantization to bake in: `fp8`, `int8`, or `mxfp4`.
    /// Absent means none — the optimization is then the layout work alone
    /// (fused banks, expert stacks, dequantized schemes).
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
    /// Write the artifact here instead of the store. A path ending in `.zt`
    /// is the artifact; a directory receives `<name>-optimized.zt`.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Report what would be done without doing it.
    #[arg(long)]
    pub dry_run: bool,
}

/// The facts the author needs, read off `config.json` the way the CUDA
/// driver's parser reads them — including the `text_config` nesting the
/// multimodal releases use.
///
/// This read is temporary scaffolding with a named successor: once serving
/// is artifact-only, the compiled descriptor carries these facts and the
/// request keeps only policy (`plan/model-in-rust.md` §11).
fn read_facts(config: &serde_json::Value) -> Result<ModelFacts> {
    let text = config.get("text_config");
    let get = |key: &str| {
        config
            .get(key)
            .or_else(|| text.and_then(|text| text.get(key)))
    };
    let get_u32 = |key: &str| {
        get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
    };
    let model_type = get("model_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("config.json declares no model_type"))?
        .to_string();
    let quant_method = config
        .get("quantization_config")
        .and_then(|quant| quant.get("quant_method"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let num_hidden_layers = get_u32("num_hidden_layers")
        .or_else(|| get_u32("num_layers"))
        .or_else(|| get_u32("n_layer"))
        .ok_or_else(|| anyhow!("config.json declares no num_hidden_layers"))?;
    // The families that need each of these declare them; absence means the
    // author will not consult them.
    let num_experts = get_u32("num_experts")
        .or_else(|| get_u32("n_routed_experts"))
        .or_else(|| get_u32("num_local_experts"))
        .unwrap_or(0);
    let head_dim = get_u32("head_dim").unwrap_or(0);
    let mamba_groups = get_u32("mamba_n_groups").unwrap_or(0);
    let tied_embeddings = get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let quantization = config.get("quantization");
    let quant_field = |key: &str| {
        quantization
            .and_then(|quant| quant.get(key))
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0)
    };
    Ok(ModelFacts {
        model_type,
        quant_method,
        num_hidden_layers,
        num_experts,
        head_dim,
        mamba_groups,
        tied_embeddings,
        mlx_quant_bits: quant_field("bits"),
        mlx_quant_group_size: quant_field("group_size"),
        num_kv_shared_layers: get_u32("num_kv_shared_layers").unwrap_or(0),
    })
}

pub fn run(args: OptimizeArgs) -> Result<()> {
    let source = resolve_source(&args.source)?;
    if source.path.is_file() {
        bail!(
            "optimize needs a snapshot directory (it reads config.json for the \
             model's facts); {} is a single file",
            source.path.display()
        );
    }
    let config_path = source.path.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("cannot read {}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| anyhow!("cannot parse {}: {err}", config_path.display()))?;
    let facts = read_facts(&config)?;

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
    let policy = Policy {
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
        "optimize: {} declares {} tensors ({} bound), quant={:?}, moe={:?}",
        facts.model_type,
        contract.tensors.len(),
        public,
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
        println!("dry run: would write {}", out_file.display());
        return Ok(());
    }

    let plan = pie_loader::plan::compile(&metadata, &contract, target)
        .map_err(|err| anyhow!("cannot compile: {err}"))?;

    // Metadata first, weights streamed after — the same shape convert has.
    let tokenizer = compile_tokenizer(&source)?;
    let descriptor = compile_descriptor(&source)?;

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
    if let Some(canonical) = &tokenizer {
        for (path, bytes) in canonical.objects() {
            meta.push((meta_name(path), bytes.to_vec()));
        }
    }
    if let Some(descriptor) = &descriptor {
        meta.push((meta_name(DESCRIPTOR_OBJECT), descriptor.clone()));
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

    println!(
        "{}: {} MB of runtime tensors → {}",
        source.name,
        written / (1 << 20),
        out_file.display()
    );
    Ok(())
}
