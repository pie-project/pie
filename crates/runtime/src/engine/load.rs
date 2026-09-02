//! The load door: the runtime traces, and the runtime states the contract.
//!
//! The runtime links `model` because a lane's fact word is the model's own
//! `Classify::of`, so the supergraph is already in this address space.
//!
//! [`LoadRequest`] carries `{ plan, checkpoint, budgets, ordinal }` and not a
//! `ModelContract`, keeping this crate's dependency floor at `model-ir`,
//! `eta-ir`, `serde` and `thiserror`. A shell resolves one via [`contract_for`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
// `checkpoint` the crate vs `Checkpoint` the engine enum below are different
// things; `zt` is imported by module since its parse door is just `parse`.
use checkpoint::file::Metadata;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::serve::stamp_of;
use checkpoint::file::zt;
use checkpoint::contract::ModelContract;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use model_ir::Trace;

/// The platform every door here takes, handed out beside them: a caller that can
/// reach [`identify`] should not have to name `engine` to say which
/// backend it is asking about.
pub use model_ir::Platform;

/// The traced supergraph for `sku`, on `platform`.
/// Errors if this build's catalog has no such SKU; near misses are named.
pub fn trace(sku: &str, platform: Platform) -> Result<Trace> {
    let sku = models::sku(sku).ok_or_else(|| anyhow!("{}", no_such_sku(sku)))?;
    Ok((sku.trace)(platform))
}

/// How `sku` sorts a request into the fact word its lanes carry, reached
/// through the same door as the trace. Runs on every fire.
/// Errors if this build's catalog has no such SKU; near misses are named.
pub fn classify(sku: &str) -> Result<models::ClassifyFn> {
    models::sku(sku)
        .map(|sku| sku.classify)
        .ok_or_else(|| anyhow!("{}", no_such_sku(sku)))
}

/// The one refusal both doors above raise: an unknown SKU, with the ones this build ships.
fn no_such_sku(sku: &str) -> String {
    format!(
        "this build ships no SKU named {sku:?}; it ships {}",
        models::skus()
            .map(|sku| sku.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Opens a checkpoint — a snapshot directory or one container file — as the
/// object model the contract algebra speaks. A sharded checkpoint opens as
/// the union of all its shards, not just the first.
pub fn open_source(checkpoint: &Path) -> Result<ztensor::Source> {
    let containers = containers(checkpoint)?;
    // A lone container stays out of `Source::merge`, which would drop its own manifest.
    if let [container] = containers.as_slice() {
        return ztensor_compat::index(container)
            .or_else(|_| ztensor::Source::open(container))
            .with_context(|| format!("open {container:?} as a tensor container"));
    }
    ztensor_compat::index_all(&containers)
        .or_else(|_| ztensor::Source::open_all(&containers))
        .with_context(|| {
            format!(
                "open the {} containers under {checkpoint:?} as one tensor name space",
                containers.len()
            )
        })
}

/// Which files make up this checkpoint, sorted so a shard set opens the same
/// way twice (`index_all` numbers stores by position, not `read_dir` order).
/// Errors if the directory can't be read, or holds no container at all.
fn containers(checkpoint: &Path) -> Result<Vec<PathBuf>> {
    if !checkpoint.is_dir() {
        return Ok(vec![checkpoint.to_path_buf()]);
    }
    // `model.zt` is one container, not a set, matching `checkpoint::file::read::discover`.
    let root = checkpoint.join("model.zt");
    if root.is_file() {
        return Ok(vec![root]);
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(checkpoint)
        .with_context(|| format!("read the checkpoint directory {checkpoint:?}"))?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    if found.is_empty() {
        return Err(anyhow!(
            "{checkpoint:?} holds no `.safetensors` and no `.zt` container"
        ));
    }
    Ok(found)
}

/// Which SKU a checkpoint is: each catalog import is tried and the answer is
/// checked against the checkpoint's actual shapes, not just matched by name
/// (a name match alone would fit e.g. a 0.8B checkpoint to the 3B row).
/// Errors if no SKU's contract both builds and fits; each refusal is named.
pub fn identify(checkpoint: &Path, platform: Platform) -> Result<&'static str> {
    if stamp_of(checkpoint)?.is_some() {
        return verify_artifact(checkpoint, platform);
    }
    let source = open_source(checkpoint)?;
    let metadata = checkpoint_metadata(checkpoint)?;
    // tp=1 takes every `Shard::Cut` segment whole; alignment and tile budget
    // don't affect the declared shape, which is the only thing checked here.
    let target = checkpoint::plan::StorageTarget::for_backend(backend_of(platform), 0, 1);

    let mut misses: Vec<String> = Vec::new();
    for (sku, read) in models::fits(&source, platform) {
        // Read for the platform being identified: placement is per-platform.
        let contract = match read {
            Ok(contract) => contract,
            Err(why) => {
                misses.push(format!("{}: {why}", sku.name));
                continue;
            }
        };
        match checkpoint::plan::compile(&metadata, &contract, target.clone()) {
            Ok(_) => return Ok(&sku.name),
            Err(why) => misses.push(format!("{}: {why}", sku.name)),
        }
    }
    Err(anyhow!(
        "{checkpoint:?} matches no SKU this build ships:\n  {}",
        misses.join("\n  ")
    ))
}

/// The setup this binary serves, for the commands that convert rather than
/// serve (a converter has no shell to ask, so it names the box it runs on).
/// CUDA wins when both shells are linked; `None` when neither engine links.
#[must_use]
pub fn this_box() -> Option<Platform> {
    #[cfg(feature = "cuda")]
    {
        return Some(Platform::Cuda);
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal", target_vendor = "apple"))]
    {
        return Some(Platform::Metal);
    }
    #[allow(unreachable_code)]
    None
}

/// The SKU a serving artifact was imported for, verified against the file's declared shapes.
pub fn verify_artifact(artifact: &Path, platform: Platform) -> Result<&'static str> {
    let stamp = stamp_of(artifact)?
        .ok_or_else(|| anyhow!("{artifact:?} carries no serving stamp"))?;
    let sku = models::sku(&stamp.sku).ok_or_else(|| {
        anyhow!("{artifact:?} was imported for `{}`; {}", stamp.sku, no_such_sku(&stamp.sku))
    })?;
    let trace = (sku.trace)(platform);
    let source = open_source(artifact)?;
    let metadata = checkpoint_metadata(artifact)?;
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, sku.recipe.tp, platform)
        .map_err(|why| anyhow!("{artifact:?} does not hold every plane of `{}`: {why}", sku.name))?;
    let target = checkpoint::plan::StorageTarget::for_backend(backend_of(platform), 0, 1);
    checkpoint::plan::compile(&metadata, &contract, target)
        .map_err(|why| anyhow!("{artifact:?} does not land as `{}`: {why}", sku.name))?;
    Ok(&sku.name)
}

fn backend_of(platform: Platform) -> checkpoint::types::BackendKind {
    match platform {
        Platform::Metal => checkpoint::types::BackendKind::Metal,
        Platform::Vulkan | Platform::Wgpu => checkpoint::types::BackendKind::Vulkan,
        Platform::Cuda => checkpoint::types::BackendKind::Cuda,
    }
}

/// The SKU whose import contract this checkpoint satisfies, checked against
/// the conversion target rather than a device one (unlike [`identify`]).
/// `None`, not an error: an unidentifiable source is not a failed import.
#[must_use]
pub fn conversion_contract(
    source: &ztensor::Source,
    metadata: &Metadata,
    platform: Platform,
) -> Option<(&'static str, ModelContract)> {
    let target = checkpoint::plan::StorageTarget {
        tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
        ..checkpoint::plan::StorageTarget::default()
    };
    // `PIE_IMPORT_TRACE=1` says why each row that did not fit did not.
    let trace = std::env::var_os("PIE_IMPORT_TRACE").is_some_and(|v| v != "0");
    for (sku, read) in models::fits(source, platform) {
        let contract = match read {
            Ok(contract) => contract,
            Err(why) => {
                if trace {
                    eprintln!("identify: {} does not read this source: {why}", sku.name);
                }
                continue;
            }
        };
        match checkpoint::plan::compile(metadata, &contract, target.clone()) {
            Ok(_) => return Some((&sku.name, contract)),
            Err(why) => {
                if trace {
                    eprintln!("identify: {} reads it but does not compile: {why}", sku.name);
                }
            }
        }
    }
    None
}

/// The checkpoint's own tensor table — a snapshot directory or one container.
pub fn checkpoint_metadata(checkpoint: &Path) -> Result<Metadata> {
    if checkpoint.is_dir() {
        parse_metadata(checkpoint).map_err(|error| anyhow!("reading {checkpoint:?}: {error}"))
    } else {
        zt::parse(checkpoint).map_err(|error| anyhow!("reading {checkpoint:?}: {error}"))
    }
}

/// The function pointer a shell takes at open. See the module header.
/// Errors as a `String`, not `anyhow::Error`, so the pointer stays crate-agnostic.
pub fn contract_for(trace: &Trace, checkpoint: &Path) -> std::result::Result<ModelContract, String> {
    let source = open_source(checkpoint).map_err(|error| format!("{error:#}"))?;
    let stamped = stamp_of(checkpoint).map_err(|error| format!("{error:#}"))?.is_some();
    let sku = models::sku(&trace.name).ok_or_else(|| {
        format!(
            "this build ships no SKU named {:?}, so a checkpoint's tensors cannot be \
             mapped onto its params",
            trace.name
        )
    })?;
    if stamped {
        return checkpoint_dsl::own_contract(&source, &trace.params, sku.recipe.tp, trace.platform)
            .map_err(|error| format!("{checkpoint:?} does not hold every plane of {:?}: {error}", trace.name));
    }
    sku.contract(&source, trace.platform).map_err(|error| {
        format!(
            "the import contract for {:?} does not fit {checkpoint:?}: {error}",
            trace.name
        )
    })
}

/// Everything a load states, for a checkpoint this build can identify.
/// `frames_in_flight` is `[runtime] frame_dispatch_depth`, crossed once here.
/// Errors on a checkpoint no SKU claims, or an unshipped SKU trace.
pub fn request(
    checkpoint: &Path,
    platform: Platform,
    budgets: Budgets,
    residency: Residency,
    ordinal: i32,
    frames_in_flight: u8,
) -> Result<LoadRequest> {
    request_of(None, checkpoint, platform, budgets, residency, ordinal, frames_in_flight)
}

/// The one load door, with or without the operator's named row.
/// `sku: None` identifies the checkpoint ([`identify`]); a name takes that
/// row's trace directly, refused later if the checkpoint doesn't hold it.
pub fn request_of(
    sku: Option<&str>,
    checkpoint: &Path,
    platform: Platform,
    budgets: Budgets,
    residency: Residency,
    ordinal: i32,
    frames_in_flight: u8,
) -> Result<LoadRequest> {
    let sku = match sku {
        Some(named) => {
            tracing::info!(sku = named, ?checkpoint, "serving the sku the config named");
            named.to_string()
        }
        None => {
            let found = identify(checkpoint, platform)?;
            tracing::info!(sku = found, ?checkpoint, "identified");
            found.to_string()
        }
    };
    Ok(LoadRequest {
        trace: trace(&sku, platform)?,
        checkpoint: Checkpoint::Path(checkpoint.to_path_buf()),
        budgets,
        residency,
        ordinal,
        frames_in_flight,
    })
}

/// The order a serving artifact's planes should lie in, hottest first, so a
/// streaming boot walks the file forward once instead of seeking.
/// `None` when no shell is linked to compute a ranking.
#[cfg(feature = "cuda")]
#[must_use]
pub fn sequence(trace: &Trace) -> Option<Vec<String>> {
    let at: std::collections::BTreeMap<&str, usize> = trace
        .params
        .iter()
        .enumerate()
        .map(|(at, param)| (param.name.as_str(), at))
        .collect();
    let mut pairings = engine_cuda::experts::Attachments::new();
    for (codes, param) in trace.params.iter().enumerate() {
        let mut companions = Vec::new();
        if let Some(&scales) = at.get(models::scales_name(&param.name).as_str()) {
            companions.push(scales);
        }
        if let Some(&biases) = at.get(models::biases_name(&param.name).as_str()) {
            companions.push(biases);
        }
        if !companions.is_empty() {
            pairings.insert(codes, companions);
        }
    }
    let ranking = engine_cuda::experts::Ranking::of(trace, &pairings).ok()?;
    Some(
        ranking
            .images()
            .into_iter()
            .filter_map(|(param, ..)| trace.params.get(param as usize))
            .map(|param| param.name.clone())
            .collect(),
    )
}

/// [`sequence`], for a build with no shell linked: unranked is still correct.
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn sequence(_trace: &Trace) -> Option<Vec<String>> {
    None
}

