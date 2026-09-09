use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use checkpoint::contract::ModelContract;
use checkpoint::file::Metadata;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::serve::stamp_of;
use checkpoint::file::zt;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};

pub use model_ir::{Platform, Trace};

pub fn trace(sku: &str, platform: Platform) -> Result<Trace> {
    let sku = models::sku(sku).ok_or_else(|| anyhow!("{}", no_such_sku(sku)))?;
    Ok((sku.trace)(platform))
}

pub fn classify(sku: &str) -> Result<models::ClassifyFn> {
    models::sku(sku)
        .map(|sku| sku.classify)
        .ok_or_else(|| anyhow!("{}", no_such_sku(sku)))
}

fn no_such_sku(sku: &str) -> String {
    format!(
        "this build ships no SKU named {sku:?}; it ships:\n  {}",
        models::skus()
            .map(|sku| sku.name.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    )
}

pub fn open_source(checkpoint: &Path) -> Result<ztensor::Source> {
    if checkpoint::file::diffusers::is_pipeline(checkpoint) {
        return checkpoint::file::diffusers::open(checkpoint)
            .with_context(|| format!("open the diffusers pipeline {checkpoint:?}"));
    }
    let containers = containers(checkpoint)?;
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

fn containers(checkpoint: &Path) -> Result<Vec<PathBuf>> {
    if !checkpoint.is_dir() {
        return Ok(vec![checkpoint.to_path_buf()]);
    }
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

pub fn identify(checkpoint: &Path, platform: Platform) -> Result<&'static str> {
    if stamp_of(checkpoint)?.is_some() {
        return verify_artifact(checkpoint, platform);
    }
    let source = open_source(checkpoint)?;
    let metadata = checkpoint_metadata(checkpoint)?;
    let target = checkpoint::plan::StorageTarget::for_backend(backend_of(platform), 0, 1);

    let mut misses: Vec<String> = Vec::new();
    for (sku, read) in models::fits(&source, platform) {
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
    #[cfg(all(
        not(feature = "cuda"),
        not(all(feature = "metal", target_vendor = "apple")),
        feature = "vulkan"
    ))]
    {
        return Some(Platform::Vulkan);
    }
    #[cfg(all(
        not(feature = "cuda"),
        not(all(feature = "metal", target_vendor = "apple")),
        not(feature = "vulkan"),
        feature = "wgpu"
    ))]
    {
        return Some(Platform::Wgpu);
    }
    #[allow(unreachable_code)]
    None
}

pub fn verify_artifact(artifact: &Path, platform: Platform) -> Result<&'static str> {
    let stamp =
        stamp_of(artifact)?.ok_or_else(|| anyhow!("{artifact:?} carries no serving stamp"))?;
    let sku = models::sku(&stamp.sku).ok_or_else(|| {
        anyhow!(
            "{artifact:?} was imported for `{}`; {}",
            stamp.sku,
            no_such_sku(&stamp.sku)
        )
    })?;
    let trace = (sku.trace)(platform);
    let source = open_source(artifact)?;
    let metadata = checkpoint_metadata(artifact)?;
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, sku.recipe.tp, platform)
        .map_err(|why| {
            anyhow!(
                "{artifact:?} does not hold every plane of `{}`: {why}",
                sku.name
            )
        })?;
    let target = checkpoint::plan::StorageTarget::for_backend(backend_of(platform), 0, 1);
    checkpoint::plan::compile(&metadata, &contract, target)
        .map_err(|why| anyhow!("{artifact:?} does not land as `{}`: {why}", sku.name))?;
    Ok(&sku.name)
}

fn backend_of(platform: Platform) -> checkpoint::types::BackendKind {
    match platform {
        Platform::Metal => checkpoint::types::BackendKind::Metal,
        Platform::Vulkan => checkpoint::types::BackendKind::Vulkan,
        Platform::Wgpu => checkpoint::types::BackendKind::Wgpu,
        Platform::Cuda => checkpoint::types::BackendKind::Cuda,
    }
}

#[must_use]
pub fn conversion_contract(
    source: &ztensor::Source,
    metadata: &Metadata,
    platform: Platform,
) -> Option<(&'static str, ModelContract)> {
    let target = convert_target();
    let trace = std::env::var_os("PIE_IMPORT_TRACE").is_some_and(|v| v != "0");
    let pinned = std::env::var("PIE_IMPORT_SKU").ok();
    for (sku, read) in models::fits(source, platform) {
        if pinned.as_deref().is_some_and(|name| name != sku.name) {
            continue;
        }
        let contract = match read {
            Ok(contract) => contract,
            Err(why) => {
                if trace {
                    eprintln!("identify: {} does not read this source: {why}", sku.name);
                }
                continue;
            }
        };
        if pinned.is_none()
            && let Some(plane) = models::requantizes(&contract)
        {
            if trace {
                eprintln!(
                    "identify: {} reads it only by re-quantizing `{plane}`; not by identification",
                    sku.name
                );
            }
            continue;
        }
        match checkpoint::plan::compile(metadata, &contract, target.clone()) {
            Ok(_) => return Some((&sku.name, contract)),
            Err(why) => {
                if trace {
                    eprintln!(
                        "identify: {} reads it but does not compile: {why}",
                        sku.name
                    );
                }
            }
        }
    }
    None
}

fn convert_target() -> checkpoint::plan::StorageTarget {
    checkpoint::plan::StorageTarget {
        tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
        ..checkpoint::plan::StorageTarget::default()
    }
}

pub fn row_named(name: &str) -> Result<&'static str> {
    Ok(&catalog_row(name)?.name)
}

fn catalog_row(name: &str) -> Result<&'static models::Sku> {
    models::sku(name).ok_or_else(|| anyhow!("{}", no_such_sku(name)))
}

pub fn conversion_contract_named(
    source: &ztensor::Source,
    metadata: &Metadata,
    platform: Platform,
    name: &str,
) -> Result<(&'static str, ModelContract)> {
    let sku = catalog_row(name)?;
    let contract = sku
        .contract(source, platform)
        .map_err(|why| anyhow!("`{}` does not read this checkpoint: {why}", sku.name))?;
    checkpoint::plan::compile(metadata, &contract, convert_target()).map_err(|why| {
        anyhow!(
            "`{}` reads this checkpoint but does not land on it: {why}",
            sku.name
        )
    })?;
    Ok((&sku.name, contract))
}

pub fn checkpoint_metadata(checkpoint: &Path) -> Result<Metadata> {
    if checkpoint.is_dir() {
        parse_metadata(checkpoint).map_err(|error| anyhow!("reading {checkpoint:?}: {error}"))
    } else {
        zt::parse(checkpoint).map_err(|error| anyhow!("reading {checkpoint:?}: {error}"))
    }
}

pub fn contract_for(
    trace: &Trace,
    checkpoint: &Path,
) -> std::result::Result<ModelContract, String> {
    let source = open_source(checkpoint).map_err(|error| format!("{error:#}"))?;
    let stamped = stamp_of(checkpoint)
        .map_err(|error| format!("{error:#}"))?
        .is_some();
    let sku = models::sku(&trace.name).ok_or_else(|| {
        format!(
            "this build ships no SKU named {:?}, so a checkpoint's tensors cannot be \
             mapped onto its params",
            trace.name
        )
    })?;
    if stamped {
        return checkpoint_dsl::own_contract(&source, &trace.params, sku.recipe.tp, trace.platform)
            .map_err(|error| {
                format!(
                    "{checkpoint:?} does not hold every plane of {:?}: {error}",
                    trace.name
                )
            });
    }
    sku.contract(&source, trace.platform).map_err(|error| {
        format!(
            "the import contract for {:?} does not fit {checkpoint:?}: {error}",
            trace.name
        )
    })
}

pub fn request(
    checkpoint: &Path,
    platform: Platform,
    budgets: Budgets,
    residency: Residency,
    ordinal: i32,
    frames_in_flight: u8,
) -> Result<LoadRequest> {
    request_of(
        None,
        checkpoint,
        platform,
        budgets,
        residency,
        ordinal,
        frames_in_flight,
    )
}

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

#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn sequence(_trace: &Trace) -> Option<Vec<String>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_checkpoint_no_row_reads(path: &Path) {
        let mut writer = ztensor::Writer::create(path).expect("create the checkpoint");
        writer
            .add(
                "a.tensor.no.model.in.this.catalog.reads",
                vec![1u64],
                ztensor::Leaf::U8,
                &[0u8],
            )
            .expect("write the stranger");
        writer.finish().expect("finish the checkpoint");
    }

    #[test]
    fn load_every_case() {
        an_unknown_row_name_is_refused_with_the_catalog();
        a_row_that_does_not_read_the_checkpoint_refuses_by_name();
        the_unnamed_door_still_answers_nothing_for_a_checkpoint_no_row_reads();
        the_rows_first_fits_wins_hides_are_reachable_by_name();
    }

    fn an_unknown_row_name_is_refused_with_the_catalog() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = dir.path().join("stranger.zt");
        a_checkpoint_no_row_reads(&path);
        let source = ztensor::Source::open(&path).expect("open the checkpoint");
        let metadata = checkpoint_metadata(&path).expect("read the checkpoint");

        let why = conversion_contract_named(&source, &metadata, Platform::Cuda, "gemma4-vision")
            .expect_err("a name no row carries cannot resolve to a row")
            .to_string();
        assert!(
            why.contains("gemma4-vision"),
            "the refusal names what was asked for: {why}"
        );
        let ships = models::skus()
            .next()
            .expect("a catalog of at least one row");
        assert!(
            why.contains(ships.name.as_str()),
            "and lists the rows this build ships: {why}"
        );
    }

    fn a_row_that_does_not_read_the_checkpoint_refuses_by_name() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = dir.path().join("stranger.zt");
        a_checkpoint_no_row_reads(&path);
        let source = ztensor::Source::open(&path).expect("open the checkpoint");
        let metadata = checkpoint_metadata(&path).expect("read the checkpoint");

        let row = models::skus()
            .find(|sku| sku.recipe.tp == 1)
            .expect("a one-rank row");
        let why = conversion_contract_named(&source, &metadata, Platform::Cuda, &row.name)
            .expect_err("no row reads a checkpoint of one stranger")
            .to_string();
        assert!(
            why.contains(row.name.as_str()),
            "the refusal names the row that was asked for: {why}"
        );
        assert!(
            why.contains("does not read this checkpoint") || why.contains("does not land on it"),
            "and says which half of the reading failed: {why}"
        );
    }

    fn the_unnamed_door_still_answers_nothing_for_a_checkpoint_no_row_reads() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = dir.path().join("stranger.zt");
        a_checkpoint_no_row_reads(&path);
        let source = ztensor::Source::open(&path).expect("open the checkpoint");
        let metadata = checkpoint_metadata(&path).expect("read the checkpoint");

        assert!(conversion_contract(&source, &metadata, Platform::Cuda).is_none());
    }

    fn the_rows_first_fits_wins_hides_are_reachable_by_name() {
        for name in [
            "gemma4-e4b-vision-bf16-kv-bf16",
            "glm53-flash-u8g64-u2g64-kv-bf16",
        ] {
            assert!(
                models::sku(name).is_some(),
                "`{name}` is the row this override exists to reach, and the \
                 catalog no longer ships it under that name"
            );
        }
    }
}
