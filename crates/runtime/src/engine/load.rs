//! The load door: **the runtime traces, and the runtime states the contract**.
//!
//! # Decision 18, from the runtime's side
//!
//! ```text
//!  runtime                                   |  shell
//!  -------                                   |  -----
//!  catalog() -> (sku, tp, Trace, Classify)   |
//!  trace(Platform::Cuda) -> model_ir::Trace --|--> compile(trace, budgets, profile)
//!  import_of(sku)(&Source) -> Contract     --|--> Weights::resident(plan, contract, path)
//!  Classify::of(request) -> Lane::word     --|--> compose -> walk -> replay
//! ```
//!
//! The runtime links `model` anyway — a lane's fact word is the model's own
//! `Classify::of`, computed per lane on the fire path — so the supergraph is
//! already in this address space and handing it across costs a `serde` round
//! trip a remote engine was going to need regardless. `model_compiler::CompiledModel`
//! never crosses: the region template, the class table and the arena carve are
//! answers about a DEVICE.
//!
//! # Why the contract travels beside the request and not inside it
//!
//! [`LoadRequest`] states `{ plan, checkpoint, budgets, ordinal }` and nothing
//! about how a checkpoint's tensors become this plan's params. That is
//! deliberate and it is the contract's own note (`engine::load`): the
//! crate's dependency floor is `model-ir`, `eta-ir`, `serde` and
//! `thiserror`, and a `ModelContract` field would put the whole checkpoint
//! plane in the dependency graph of everyone who reads a `KvHandle`.
//!
//! But a shell needs one, and it must not grow an arm per model family to
//! rediscover it. So the resolution is a function pointer the SHELL takes at
//! open and this module supplies: [`contract_for`]. One pointer, installed by
//! the party that already links the catalog.
//!
//! # What this replaced
//!
//! `ModelLoadDesc { snapshot_dir, runtime_quant: String, mxfp4_moe, component }`
//! and a `Vec` of them, one per rank. Every field of it said something the
//! plan now says outright — the contract's own header walks them one by one —
//! and the `Vec` collapsed because a rank is not a load: `Shard::Cut` is in
//! the plan, and which rank a shell is, is the shell's.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
// `checkpoint` the CRATE, and `Checkpoint` the engine enum below, are two
// different things a sentence in this file can be about: the first is the
// loader, the second is "where the weights are" in a `LoadRequest`. The
// crate's reader module used to be called `checkpoint` too, which made a
// third; it is `file` now, so a path that starts with `checkpoint` means the
// crate and nothing else. `zt` comes in by module because its door is spelled
// `parse`, and a bare `parse` beside `parse_metadata` would not say which of
// the two doors it opened.
use checkpoint::file::Metadata;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::zt;
use checkpoint::contract::ModelContract;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use model_ir::Trace;

/// The platform every door here takes, handed out beside them: a caller that can
/// reach [`identify`] should not have to name `engine` to say which
/// backend it is asking about.
pub use model_ir::Platform;

/// The catalog row a load names: its SKU, the tensor-parallel width it was
/// traced for, the trace itself, and how it sorts a request into the fact
/// word a lane carries.
pub type Row = (&'static str, u32, fn(Platform) -> Trace, ::model::ClassifyFn);

/// Every SKU this build ships.
///
/// A pass-through of `model::catalog`, here so that the rest of the runtime
/// reads the catalog through the load door rather than reaching into the
/// model crate at eleven sites.
#[must_use]
pub fn catalog() -> Vec<Row> {
    ::model::catalog()
}

/// The traced supergraph for `sku`, on `platform`.
///
/// # Errors
///
/// When this build's catalog has no such SKU — with the near misses named,
/// because "unknown model" and "you meant the -bf16 row" are different
/// operator actions.
pub fn trace(sku: &str, platform: Platform) -> Result<Trace> {
    let trace = ::model::trace_of(sku).ok_or_else(|| anyhow!("{}", no_such_sku(sku)))?;
    Ok(trace(platform))
}

/// How `sku` sorts a request into the fact word its lanes carry — the fourth
/// column, reached through the same door as the trace.
///
/// **THIS IS THE HALF OF DECISION 18 THAT RUNS EVERY FIRE.** A lane's word is
/// what `engine::fire::compose` turns into a class and therefore into the row
/// window every guarded node runs over; a fire that submits word 0 for every
/// lane runs its decode rows through the prefill arm. The runtime could not
/// state it while the catalog shipped three columns — a plan's `Guard::Fact`
/// numbers its bits and no reader outside a family's module can name them —
/// and this is the pointer that closed the hole (`palo B-word`).
///
/// # Errors
///
/// When this build's catalog has no such SKU, with the near misses named.
pub fn classify(sku: &str) -> Result<::model::ClassifyFn> {
    ::model::classify_of(sku).ok_or_else(|| anyhow!("{}", no_such_sku(sku)))
}

/// The one refusal both doors above raise, written once: an unknown SKU, and
/// the ones this build does ship.
fn no_such_sku(sku: &str) -> String {
    format!(
        "this build ships no SKU named {sku:?}; it ships {}",
        ::model::catalog()
            .iter()
            .map(|(name, ..)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Open a checkpoint — a snapshot directory or one container file — as the
/// object model the contract algebra speaks.
///
/// A stock hugging-face snapshot is a FOREIGN projection
/// (`ztensor_compat::index`), not a canonical `.zt`; `ztensor::Source::open`
/// is the other door and refuses one. Both are tried, in that order, because
/// the deployments this runtime serves have both.
///
/// # ALL OF THE SHARDS, NOT THE FIRST ONE
///
/// `index` and `Source::open` each take ONE file, which is the whole of a
/// single-container SKU and a silent lie about a sharded one: a fifteen-shard
/// 27B publishes its embedding in `model-00001-of-00015.safetensors` and its
/// final norm in the last, so an import contract built over the first shard
/// alone refuses for a missing tensor that is on disk two files away — a
/// sentence naming a param, when the fault is that this door never looked.
/// So a directory of containers goes through `index_all`/`open_all`, which
/// merge the set into one name space and refuse a name that is in two files.
///
/// A LONE CONTAINER STILL GOES THROUGH THE SINGLE-FILE DOOR, and not as a
/// one-element set: `Source::merge` rebuilds the catalog and drops what a root
/// says about itself, so the manifest a `.zt` root carries — its shard table,
/// which `verify_shards` is the only reader of — would be traded for a merged
/// projection that claims nothing. Every SKU in this build's catalog is one
/// container, so that is the path essentially every load takes.
fn open_source(checkpoint: &Path) -> Result<ztensor::Source> {
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

/// Which files on disk make up this checkpoint, in the order that fixes the
/// merged name space.
///
/// Sorted, because a shard set has to open the same way twice: `index_all`
/// numbers its stores by position, and a `read_dir` order is the filesystem's
/// and not the checkpoint's.
///
/// A directory holding `model.zt` is ONE container and not a set, which is
/// `checkpoint::file::read::discover`'s own rule and is followed here
/// so that the source this module opens and the metadata
/// [`checkpoint_metadata`] parses never disagree about what the checkpoint is:
/// a `.zt` root resolves its own data shards positionally, and handing those
/// shards to `index_all` beside the root would offer the same tensor twice and
/// be refused for a collision the checkpoint does not have.
///
/// # Errors
///
/// A directory that cannot be read, or one that holds no container at all.
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

/// Which SKU a checkpoint is, by asking every import in the catalog **and
/// then checking the answer against the checkpoint's shapes**.
///
/// # Why the name match is not the answer
///
/// `model::identify` returns the first SKU whose import contract builds, and
/// an import contract is a NAME mapping: it says that this checkpoint
/// publishes `model.layers.7.self_attn.q_proj.weight` under the plan's
/// `layer.7.q_proj`, and every SKU of a family spells its tensors the same
/// way. So a 0.8B Qwen checkpoint matches the 3B row's import — measured,
/// with this very smoke, which came back `qwen35-d3b-bf16-kv-bf16` for a
/// 0.8B snapshot.
///
/// That is not a silent failure — the loader refuses at the first param
/// (`'embed': declares shape [151936, 2048] but its expression yields
/// [248320, 1024]`) — but it is the wrong refusal in the wrong place: an
/// operator is told a checkpoint does not fit a model they did not name.
/// So the loop below does that check itself, once per candidate, and the
/// answer is the SKU whose params the checkpoint actually holds.
///
/// `checkpoint::plan::compile` is what does the checking, and it is the
/// same call `engine_cuda::weights::Weights::resident` makes: this is the
/// load's own arithmetic run for its verdict rather than its bytes.
///
/// # Errors
///
/// When no SKU's contract both builds and fits — with each candidate's
/// refusal, which is the only diagnosis an operator can act on.
pub fn identify(checkpoint: &Path, platform: Platform) -> Result<&'static str> {
    let source = open_source(checkpoint)?;
    let metadata = checkpoint_metadata(checkpoint)?;
    // tp=1: `Shard::Cut` segments still describe the whole tensor, and a rank
    // of one takes all of them. The backend chooses an alignment and a tile
    // budget, neither of which can change whether a tensor's SHAPE is the
    // one the contract declares — which is the only question here — so the
    // one the plan will be traced for is as good as any.
    let backend = match platform {
        Platform::Metal => checkpoint::types::BackendKind::Metal,
        Platform::Vulkan | Platform::Wgpu => checkpoint::types::BackendKind::Vulkan,
        Platform::Cuda => checkpoint::types::BackendKind::Cuda,
    };
    let target = checkpoint::plan::StorageTarget::for_backend(backend, 0, 1);

    let mut misses: Vec<String> = Vec::new();
    for (sku, import) in ::model::imports() {
        // A row traced for more than one rank describes a shard, not a
        // checkpoint; `model::identify` skips those for the same reason.
        if !::model::catalog()
            .iter()
            .any(|(name, tp, ..)| *name == sku && *tp == 1)
        {
            continue;
        }
        let contract = match import(&source) {
            Ok(contract) => contract,
            Err(why) => {
                misses.push(format!("{sku}: {why}"));
                continue;
            }
        };
        match checkpoint::plan::compile(&metadata, &contract, target.clone()) {
            Ok(_) => return Ok(sku),
            Err(why) => misses.push(format!("{sku}: {why}")),
        }
    }
    Err(anyhow!(
        "{checkpoint:?} matches no SKU this build ships:\n  {}",
        misses.join("\n  ")
    ))
}

/// The checkpoint's own tensor table — a snapshot directory or one container.
///
/// The same two doors `Weights::resident` opens, for the same reason: a
/// directory is discovered the way `pie model import` discovers one, a file is
/// read directly.
fn checkpoint_metadata(checkpoint: &Path) -> Result<Metadata> {
    if checkpoint.is_dir() {
        parse_metadata(checkpoint).map_err(|error| anyhow!("reading {checkpoint:?}: {error}"))
    } else {
        zt::parse(checkpoint).map_err(|error| anyhow!("reading {checkpoint:?}: {error}"))
    }
}

/// **THE FUNCTION POINTER A SHELL TAKES AT OPEN.** See the module header.
///
/// The plan names itself, and the catalog's import table is keyed by the same
/// name, so the shell hands back the plan it was loaded with and gets the
/// contract that publishes this checkpoint's tensors under that plan's param
/// names.
///
/// # Errors
///
/// A `String`, not an `anyhow::Error`, because the shell's own refusal type
/// carries a sentence and the pointer must not put an error crate in an
/// engine's signature.
pub fn contract_for(trace: &Trace, checkpoint: &Path) -> std::result::Result<ModelContract, String> {
    let import = ::model::import_of(&trace.name).ok_or_else(|| {
        format!(
            "this build ships no import contract for {:?}, so a checkpoint's \
             tensors cannot be mapped onto its params",
            trace.name
        )
    })?;
    let source = open_source(checkpoint).map_err(|error| format!("{error:#}"))?;
    import(&source).map_err(|error| {
        format!(
            "the import contract for {:?} does not fit {checkpoint:?}: {error}",
            trace.name
        )
    })
}

/// Everything a load states, for a checkpoint this build can identify.
///
/// `frames_in_flight` is the deployment's `[runtime] frame_dispatch_depth`,
/// crossing the boundary once (article 8): an engine that settles
/// asynchronously carves its staging ring and its settlement event pool from
/// it, and one that does not ignores it.
///
/// `residency` is the weight policy — two budgets, uncapped by default (alto
/// design §7). It is stated HERE, beside the budgets, because it is the same
/// kind of thing: a ceiling the deployment sets and the shell bakes against,
/// not a per-fire decision.
///
/// # Errors
///
/// A checkpoint no SKU claims, or a SKU whose trace this build does not ship.
pub fn request(
    checkpoint: &Path,
    platform: Platform,
    budgets: Budgets,
    residency: Residency,
    ordinal: i32,
    frames_in_flight: u8,
) -> Result<LoadRequest> {
    let sku = identify(checkpoint, platform)?;
    Ok(LoadRequest {
        trace: trace(sku, platform)?,
        checkpoint: Checkpoint::Path(checkpoint.to_path_buf()),
        budgets,
        residency,
        ordinal,
        frames_in_flight,
    })
}

/// Everything a load states, for a checkpoint whose SKU the caller already
/// knows.
///
/// `frames_in_flight` as [`request`].
///
/// # Errors
///
/// A SKU whose trace this build does not ship.
pub fn request_for(
    sku: &str,
    checkpoint: &Path,
    platform: Platform,
    budgets: Budgets,
    residency: Residency,
    ordinal: i32,
    frames_in_flight: u8,
) -> Result<LoadRequest> {
    Ok(LoadRequest {
        trace: trace(sku, platform)?,
        checkpoint: Checkpoint::Path(checkpoint.to_path_buf()),
        budgets,
        residency,
        ordinal,
        frames_in_flight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-tensor safetensors container, written by hand.
    ///
    /// The format is `[header_len: u64 LE][JSON header][data]`, and the
    /// projection is strict about the tail — the tensor ranges must tile the
    /// data section exactly and end at EOF — so the payload is sized from the
    /// header and nothing is padded after it. Written rather than fetched
    /// because the shard question is about NAMES, and a name space needs no
    /// weights: this fixture is what lets the gate below run on a machine with
    /// no GPU and no 27B snapshot in its cache.
    fn shard(dir: &Path, file: &str, tensor: &str) -> PathBuf {
        let header = format!(
            "{{{tensor:?}:{{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}}}"
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let path = dir.join(file);
        std::fs::write(&path, bytes).expect("the fixture container writes");
        path
    }

    /// **THE REGRESSION THIS MODULE EXISTS TO HOLD** (build log 25): a sharded
    /// checkpoint is every shard, and `open_source` took the first one.
    ///
    /// The second shard here is the final norm, which is exactly where a
    /// fifteen-shard 27B keeps it — under the old door the contract algebra
    /// was told the tensor does not exist, and refused naming a param instead
    /// of naming the file that was never opened.
    #[test]
    fn a_sharded_checkpoint_opens_as_the_union_of_its_shards() {
        let snapshot = tempfile::tempdir().expect("a temp dir");
        shard(
            snapshot.path(),
            "model-00001-of-00002.safetensors",
            "model.embed_tokens.weight",
        );
        shard(
            snapshot.path(),
            "model-00002-of-00002.safetensors",
            "model.norm.weight",
        );

        let source = open_source(snapshot.path()).expect("both shards open as one name space");
        let names: Vec<&str> = source.names().collect();
        assert_eq!(
            names,
            vec!["model.embed_tokens.weight", "model.norm.weight"],
            "the source is not the union of the shards, so an import contract \
             would refuse for a tensor that is on disk"
        );
    }

    /// The single-container SKU — every row this build's catalog ships — reads
    /// the way it always did, one file through the one-file door.
    #[test]
    fn a_lone_container_is_still_read_whole() {
        let snapshot = tempfile::tempdir().expect("a temp dir");
        let only = shard(snapshot.path(), "model.safetensors", "model.norm.weight");

        for checkpoint in [snapshot.path(), only.as_path()] {
            let source = open_source(checkpoint).expect("the container opens");
            let names: Vec<&str> = source.names().collect();
            assert_eq!(
                names,
                vec!["model.norm.weight"],
                "a one-file checkpoint answers differently through {checkpoint:?}"
            );
        }
    }

    /// A directory with nothing to open still says so in the checkpoint's own
    /// terms: the shard path must not turn "no container here" into a merge
    /// refusal about an empty set.
    #[test]
    fn a_directory_holding_no_container_is_refused_by_name() {
        let snapshot = tempfile::tempdir().expect("a temp dir");
        std::fs::write(snapshot.path().join("config.json"), b"{}").expect("the fixture writes");

        let refused = open_source(snapshot.path()).expect_err("there is nothing to open");
        let said = format!("{refused:#}");
        assert!(
            said.contains("holds no `.safetensors` and no `.zt` container"),
            "the refusal does not name what is missing: {said}"
        );
    }
}
