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

/// The platform every door here takes and the trace it answers with, handed
/// out beside them: a caller that can reach [`identify`] or [`trace`] should
/// not have to name `engine` to say which backend it is asking about.
pub use model_ir::{Platform, Trace};

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

/// The one refusal these doors raise: an unknown SKU, with the ones this
/// build ships.
///
/// One name per line rather than one comma-joined paragraph: this is also
/// what `pie model import --sku <a name that is not a row>` prints, and the
/// catalog is sixty-odd rows long — a list an operator has to READ, to pick
/// the row they meant.
fn no_such_sku(sku: &str) -> String {
    format!(
        "this build ships no SKU named {sku:?}; it ships:\n  {}",
        models::skus()
            .map(|sku| sku.name.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
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
/// `None` when no engine links.
///
/// **THE ORDER IS THE POLICY**, and it is vendor-specific first: CUDA wins
/// over everything, Metal over the two portable shells on a Mac, then Vulkan,
/// then wgpu. Each is a shell that runs on the same card the one above it
/// does — wgpu reaches its device THROUGH Vulkan on a Linux box — so a build
/// carrying several would convert for the thinnest of them if this order were
/// the other way round.
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
        Platform::Vulkan => checkpoint::types::BackendKind::Vulkan,
        // Its OWN kind, no longer folded onto Vulkan's: a `.wgpu.zt` is its
        // own artifact, so the plan a wgpu box compiles is the plan its
        // stamp names.
        Platform::Wgpu => checkpoint::types::BackendKind::Wgpu,
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
    let target = convert_target();
    // `PIE_IMPORT_TRACE=1` says why each row that did not fit did not.
    let trace = std::env::var_os("PIE_IMPORT_TRACE").is_some_and(|v| v != "0");
    // `PIE_IMPORT_SKU=<name>` converts for that row alone — how a parity
    // miniature (a row identification never picks, since every real row
    // fits first) becomes an artifact. `pie model import --sku <NAME>` is the
    // same choice made out loud ([`conversion_contract_named`]): it refuses
    // by name where this one silently falls through to "no row claims it".
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

/// What an import compiles against: file bytes on this host, not a device.
/// Named once so [`conversion_contract`] and [`conversion_contract_named`]
/// cannot ask the same checkpoint two different questions.
fn convert_target() -> checkpoint::plan::StorageTarget {
    checkpoint::plan::StorageTarget {
        tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
        ..checkpoint::plan::StorageTarget::default()
    }
}

/// The catalog row `name` names, or the refusal that names it and lists what
/// this build ships.
///
/// The name half of [`conversion_contract_named`], reachable on its own so a
/// command can refuse a misspelling BEFORE it fetches ninety gigabytes of
/// checkpoint to hold it against.
pub fn row_named(name: &str) -> Result<&'static str> {
    Ok(&catalog_row(name)?.name)
}

/// [`row_named`], as the catalog row itself.
fn catalog_row(name: &str) -> Result<&'static models::Sku> {
    models::sku(name).ok_or_else(|| anyhow!("{}", no_such_sku(name)))
}

/// The contract of ONE NAMED ROW against this checkpoint — `pie model import
/// --sku <NAME>`.
///
/// [`conversion_contract`] answers with the first row that fits, and first
/// fits wins is why several rows cannot be reached at all: the text row of a
/// family reads every snapshot its vision row does and is asked first, and a
/// checkpoint that carries an MTP head identifies as the drafting row whether
/// or not the operator wants one. This door takes the row from the operator
/// instead.
///
/// **IT NEVER FALLS BACK.** A name this build does not ship is refused with
/// the catalog, and a row that does not read this checkpoint is refused with
/// the contract's own account of what it wanted — an import that quietly
/// converted for a different row than the one asked for would put a SKU in
/// the stamp and in the filename that the operator did not choose.
pub fn conversion_contract_named(
    source: &ztensor::Source,
    metadata: &Metadata,
    platform: Platform,
    name: &str,
) -> Result<(&'static str, ModelContract)> {
    let sku = catalog_row(name)?;
    // The row's own reading, at the row's own width. A sharded row (`-tp2`)
    // refuses here on its own terms: its import table will not read a WHOLE
    // checkpoint.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint holding one tensor no row in the catalog reads, so every
    /// row refuses it — and the refusal is the row's own.
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

    /// `--sku <a name this build does not ship>` refuses BY NAME and prints
    /// the catalog, because a misspelling is the common case and the operator
    /// cannot otherwise find out what the rows are called.
    #[test]
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

    /// A row that does not read this checkpoint refuses with the CONTRACT'S
    /// own account, and never falls back to the row that would have fitted.
    #[test]
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

    /// The default door is unchanged: a checkpoint no row reads is still
    /// `None` rather than an error, which is what an import turns into its
    /// own refusal.
    #[test]
    fn the_unnamed_door_still_answers_nothing_for_a_checkpoint_no_row_reads() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = dir.path().join("stranger.zt");
        a_checkpoint_no_row_reads(&path);
        let source = ztensor::Source::open(&path).expect("open the checkpoint");
        let metadata = checkpoint_metadata(&path).expect("read the checkpoint");

        assert!(conversion_contract(&source, &metadata, Platform::Cuda).is_none());
    }

    /// Every row an operator can name is a row the catalog can hand back, so
    /// the vision and MTP rows the identification order hides are reachable
    /// through the name and not only through the order.
    #[test]
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
