use std::path::Path;

use anyhow::{Context, Result, anyhow};
use checkpoint::contract::ModelContract;
use checkpoint::file::Metadata;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::serve::stamp_of;
use checkpoint::file::zt;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};

pub use poem_ir::{Platform, Trace};

use crate::catalog::{Deployment, catalog};

/// The deployment `name` spells, of a model this build's packages hold.
pub fn deployment(name: &str) -> Result<Deployment> {
    catalog()
        .parse(name)
        .ok_or_else(|| anyhow!("{}", no_such_deployment(name)))
}

/// The trace each rank of the deployment `name` runs.
pub fn trace(name: &str, platform: Platform) -> Result<Trace> {
    deployment(name)?
        .try_trace(platform)
        .map_err(|why| anyhow!("`{name}` does not serve on {platform:?}: {why}"))
}

fn no_such_deployment(name: &str) -> String {
    format!(
        "{name:?} names no deployment of a model this build ships; it lists:\n  {}",
        catalog()
            .deployments()
            .map(|deployment| deployment.name.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    )
}

/// What a deployment config changes about the deployment its checkpoint
/// states: a checkpoint fixes the model, the precision its weights are stored
/// at and the parts and drafter it carries; a config may leave parts off,
/// serve without the drafter, store the kv at another dtype, and split across
/// ranks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Overrides {
    /// The precision the config expects; a checkpoint stored at another is
    /// refused rather than re-quantized.
    pub precision: Option<Vec<poem_ir::Dtype>>,
    pub kv: Option<poem_ir::Dtype>,
    pub off: Vec<String>,
    /// `Some(None)` serves without the checkpoint's drafter.
    pub drafter: Option<Option<String>>,
    pub tp: Option<u32>,
}

/// The deployment `base` becomes under `overrides`, checked on `platform`.
pub fn compose(base: &str, overrides: &Overrides, platform: Platform) -> Result<String> {
    let base = deployment(base)?;
    let deploy = composed(&base.name, base.deploy.clone(), overrides)?;
    let composed = Deployment::of(base.package.clone(), &base.model, deploy);
    composed
        .check(platform)
        .map_err(|why| anyhow!("`{}` does not serve on {platform:?}: {why}", composed.name))?;
    Ok(composed.name)
}

/// The deployment `deploy`, which `base` names, becomes under `overrides`.
fn composed(
    base: &str,
    mut deploy: poem::star::Deploy,
    overrides: &Overrides,
) -> Result<poem::star::Deploy> {
    if let Some(precision) = &overrides.precision
        && *precision != deploy.weights
    {
        return Err(anyhow!(
            "`{base}` stores its weights at {:?} and the config asks for {precision:?}; import \
             the checkpoint at that precision to serve it",
            deploy.weights
        ));
    }
    if let Some(kv) = overrides.kv {
        deploy.kv = kv;
    }
    for part in &overrides.off {
        if !deploy.parts.iter().any(|have| have == part) {
            return Err(anyhow!(
                "the config leaves {part} off and `{base}` carries none"
            ));
        }
        deploy.parts.retain(|have| have != part);
    }
    match &overrides.drafter {
        None => {}
        Some(None) => deploy.drafter = None,
        Some(Some(drafter)) if deploy.drafter.as_deref() == Some(drafter.as_str()) => {}
        Some(Some(drafter)) => {
            return Err(anyhow!(
                "the config drafts with {drafter} and `{base}` carries {}; import the \
                 checkpoint with that drafter to serve it",
                deploy.drafter.as_deref().unwrap_or("no drafter")
            ));
        }
    }
    if let Some(tp) = overrides.tp {
        deploy.tp = tp;
    }
    Ok(deploy)
}

/// The package an artifact carries, if it carries one.
pub fn package_of(path: &Path) -> Result<Option<poem::star::Package>> {
    if path.is_dir() {
        return Ok(None);
    }
    let Ok(Some(manifest)) = ztensor::read::manifest_of(path) else {
        return Ok(None);
    };
    let Some(ztensor::format::cbor::Value::Map(attributes)) = manifest.attributes.as_ref() else {
        return Ok(None);
    };
    let texts = attributes
        .iter()
        .filter_map(|(key, value)| Some((key.as_text()?, value.as_text()?)));
    poem::star::Package::from_attributes(texts)
        .with_context(|| format!("read the package {path:?} carries"))
}

/// The package the artifact at `path` carries. An artifact is served by its
/// own package and nothing else: one that carries none is not an artifact
/// this build serves.
pub fn carried(path: &Path) -> Result<poem::star::Package> {
    package_of(path)?.ok_or_else(|| {
        anyhow!("{path:?} carries no model package; import the checkpoint again with this build")
    })
}

/// What an artifact serves under `overrides`: the deployment's name, its rank
/// count and the trace each rank runs, all of it from the package the
/// artifact carries, none of it from this build's catalog.
pub fn packaged(
    artifact: &Path,
    overrides: &Overrides,
    platform: Platform,
) -> Result<(String, u32, Trace)> {
    let package = carried(artifact)?;
    let stamp =
        stamp_of(artifact)?.ok_or_else(|| anyhow!("{artifact:?} carries no serving stamp"))?;
    let (model, base) = package.manifest().parse(&stamp.deployment).ok_or_else(|| {
        anyhow!(
            "{artifact:?} was imported as `{}`, which its package `{}` names no deployment of",
            stamp.deployment,
            package.name()
        )
    })?;
    let deploy = composed(&stamp.deployment, base, overrides)?;
    model.admits(&deploy)?;
    let name = model.name(&deploy);
    let trace = package
        .trace(&model.id, &deploy, &name, platform)
        .map_err(|why| anyhow!("`{name}` does not serve on {platform:?}: {why:#}"))?;
    let trace = poem_compiler::shard::shard(trace, deploy.tp)
        .map_err(|why| anyhow!("`{name}` does not split across {} ranks: {why}", deploy.tp))?;
    Ok((name, deploy.tp, trace))
}

/// How the deployment `name` stamped on the artifact at `artifact` is
/// served, as the package it carries spells it.
#[must_use]
pub fn deploy_of(artifact: &Path, name: &str) -> Option<poem::star::Deploy> {
    let package = package_of(artifact).ok()??;
    let (_, deploy) = package.manifest().parse(name)?;
    Some(deploy)
}

/// The attributes an artifact of the deployment `name` carries its package
/// in, if a package holds its model.
pub fn package_attributes(name: &str) -> Result<std::collections::BTreeMap<String, String>> {
    Ok(deployment(name)?.package.attributes())
}

pub fn open_source(checkpoint: &Path) -> Result<ztensor::Source> {
    checkpoint::file::snapshot::open(checkpoint)
        .with_context(|| format!("open the checkpoint {checkpoint:?}"))
}

pub fn identify(checkpoint: &Path, platform: Platform) -> Result<String> {
    if stamp_of(checkpoint)?.is_some() {
        return verify_artifact(checkpoint, platform);
    }
    let source = open_source(checkpoint)?;
    let metadata = checkpoint_metadata(checkpoint)?;
    let target = checkpoint::plan::StorageTarget::for_backend(backend_of(platform), 0, 1);

    let mut misses: Vec<String> = Vec::new();
    for (deployment, read) in catalog().fits(&source, platform) {
        let contract = match read {
            Ok(contract) => contract,
            Err(why) => {
                misses.push(format!("{}: {why}", deployment.name));
                continue;
            }
        };
        match checkpoint::plan::compile(&metadata, &contract, target.clone()) {
            Ok(_) => return Ok(deployment.name.clone()),
            Err(why) => misses.push(format!("{}: {why}", deployment.name)),
        }
    }
    Err(anyhow!(
        "{checkpoint:?} matches no deployment this build ships:\n  {}",
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
    #[cfg(all(
        not(feature = "cuda"),
        not(all(feature = "metal", target_vendor = "apple")),
        not(feature = "vulkan"),
        not(feature = "wgpu"),
        feature = "xla"
    ))]
    {
        return Some(Platform::Xla);
    }
    #[allow(unreachable_code)]
    None
}

pub fn verify_artifact(artifact: &Path, platform: Platform) -> Result<String> {
    let (name, tp, trace) = packaged(artifact, &Overrides::default(), platform)?;
    let source = open_source(artifact)?;
    let metadata = checkpoint_metadata(artifact)?;
    let contract = poem::import::own_contract(&source, &trace.params, tp, platform)
        .map_err(|why| anyhow!("{artifact:?} does not hold every plane of `{name}`: {why}"))?;
    let target = checkpoint::plan::StorageTarget::for_backend(backend_of(platform), 0, 1);
    checkpoint::plan::compile(&metadata, &contract, target)
        .map_err(|why| anyhow!("{artifact:?} does not land as `{name}`: {why}"))?;
    Ok(name)
}

fn backend_of(platform: Platform) -> checkpoint::types::BackendKind {
    match platform {
        Platform::Metal => checkpoint::types::BackendKind::Metal,
        Platform::Vulkan => checkpoint::types::BackendKind::Vulkan,
        Platform::Wgpu => checkpoint::types::BackendKind::Wgpu,
        Platform::Xla => checkpoint::types::BackendKind::Xla,
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
    let pinned = std::env::var("PIE_IMPORT_DEPLOYMENT").ok();
    for deployment in catalog().candidates() {
        let read = deployment.contract(source, platform);
        if pinned
            .as_deref()
            .is_some_and(|name| name != deployment.name)
        {
            continue;
        }
        let contract = match read {
            Ok(contract) => contract,
            Err(why) => {
                if trace {
                    eprintln!(
                        "identify: {} does not read this source: {why}",
                        deployment.name
                    );
                }
                continue;
            }
        };
        if pinned.is_none()
            && let Some(plane) = poem_compiler::catalog::requantizes(&contract)
        {
            if trace {
                eprintln!(
                    "identify: {} reads it only by re-quantizing `{plane}`; not by identification",
                    deployment.name
                );
            }
            continue;
        }
        match checkpoint::plan::compile(metadata, &contract, target.clone()) {
            Ok(_) => return Some((&deployment.name, contract)),
            Err(why) => {
                if trace {
                    eprintln!(
                        "identify: {} reads it but does not compile: {why}",
                        deployment.name
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

pub fn conversion_contract_named(
    source: &ztensor::Source,
    metadata: &Metadata,
    platform: Platform,
    name: &str,
) -> Result<(String, ModelContract)> {
    let deployment = deployment(name)?;
    let contract = deployment
        .contract(source, platform)
        .map_err(|why| anyhow!("`{}` does not read this checkpoint: {why}", deployment.name))?;
    checkpoint::plan::compile(metadata, &contract, convert_target()).map_err(|why| {
        anyhow!(
            "`{}` reads this checkpoint but does not land on it: {why}",
            deployment.name
        )
    })?;
    Ok((deployment.name, contract))
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
    let said = |error: &dyn std::fmt::Display| format!("{error:#}");
    let source = open_source(checkpoint).map_err(|e| said(&e))?;
    if stamp_of(checkpoint).map_err(|e| said(&e))?.is_some() {
        let package = carried(checkpoint).map_err(|e| said(&e))?;
        let (_, deploy) = package.manifest().parse(&trace.name).ok_or_else(|| {
            format!(
                "{:?} names no deployment of the package `{}` {checkpoint:?} carries",
                trace.name,
                package.name()
            )
        })?;
        let tp = deploy.tp;
        return poem::import::own_contract(&source, &trace.params, tp, trace.platform).map_err(
            |error| {
                format!(
                    "{checkpoint:?} does not hold every plane of {:?}: {error}",
                    trace.name
                )
            },
        );
    }
    let deployment = catalog().parse(&trace.name).ok_or_else(|| {
        format!(
            "{:?} names no deployment of a model this build ships, so a checkpoint's tensors \
             cannot be mapped onto its params",
            trace.name
        )
    })?;
    if deployment.deploy.tp > 1 {
        return Err(format!(
            "{:?} is a {}-rank deployment and {checkpoint:?} is an unconverted checkpoint: the \
             ranks band their shares out of a stamped artifact, so convert it first (`pie model \
             import`) and serve the artifact",
            trace.name, deployment.deploy.tp
        ));
    }
    deployment
        .contract(&source, trace.platform)
        .map_err(|error| {
            format!(
                "the import contract for {:?} does not fit {checkpoint:?}: {error}",
                trace.name
            )
        })
}

/// The load of the deployment `checkpoint` states, under the config's
/// `overrides`.
pub fn request(
    overrides: &Overrides,
    checkpoint: &Path,
    platform: Platform,
    budgets: Budgets,
    residency: Residency,
    ordinal: i32,
    frames_in_flight: u8,
) -> Result<LoadRequest> {
    // An artifact is served by the package it carries; a raw checkpoint by
    // the deployment of this build's catalog that reads it.
    let trace = if stamp_of(checkpoint)?.is_some() {
        let (name, _, trace) = packaged(checkpoint, overrides, platform)?;
        tracing::info!(deployment = name, ?checkpoint, "serving its package");
        trace
    } else {
        let base = identify(checkpoint, platform)?;
        let name = compose(&base, overrides, platform)?;
        tracing::info!(deployment = name, identified = base, ?checkpoint, "serving");
        self::trace(&name, platform)?
    };
    Ok(LoadRequest {
        trace,
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
        if let Some(&scales) = at.get(poem::scales_name(&param.name).as_str()) {
            companions.push(scales);
        }
        if let Some(&biases) = at.get(poem::biases_name(&param.name).as_str()) {
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
        let ships = catalog()
            .deployments()
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

        let row = catalog().deployments().next().expect("a one-rank row");
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
                catalog().deployment(name).is_some(),
                "`{name}` is the row this override exists to reach, and the \
                 catalog no longer ships it under that name"
            );
        }
    }
}
