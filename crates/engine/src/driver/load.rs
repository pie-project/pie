//! The load door: **the engine traces, and the engine states the contract**.
//!
//! # Decision 18, from the engine's side
//!
//! ```text
//!  engine                                    |  shell
//!  ------                                    |  -----
//!  catalog() -> (sku, tp, Trace, Classify)   |
//!  trace(Platform::Cuda) -> model_ir::Plan --|--> compile(plan, budgets, profile)
//!  import_of(sku)(&Source) -> Contract     --|--> Weights::resident(plan, contract, path)
//!  Classify::of(request) -> Lane::word     --|--> compose -> walk -> replay
//! ```
//!
//! The engine links `model` anyway — a lane's fact word is the model's own
//! `Classify::of`, computed per lane on the fire path — so the supergraph is
//! already in this address space and handing it across costs a `serde` round
//! trip a remote driver was going to need regardless. `model_compiler::Baked`
//! never crosses: the region template, the class table and the arena carve are
//! answers about a DEVICE.
//!
//! # Why the contract travels beside the request and not inside it
//!
//! [`LoadRequest`] states `{ plan, checkpoint, budgets, ordinal }` and nothing
//! about how a checkpoint's tensors become this plan's params. That is
//! deliberate and it is the contract's own note (`driver-api::load`): the
//! crate's dependency floor is `model-ir`, `tensor-ir`, `serde` and
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
use driver_api::load::{Budgets, Checkpoint, LoadRequest};
use driver_api::model_ir::Plan;
use model_loader::contract::ModelContract;

/// The platform every door here takes, handed out beside them: a caller that can
/// reach [`identify`] should not have to name `driver-api` to say which
/// backend it is asking about.
pub use driver_api::model_ir::Platform;

/// The catalog row a load names: its SKU, the tensor-parallel width it was
/// traced for, the trace itself, and how it sorts a request into the fact
/// word a lane carries.
pub type Row = (&'static str, u32, fn(Platform) -> Plan, ::model::ClassifyFn);

/// Every SKU this build ships.
///
/// A pass-through of `model::catalog`, here so that the rest of the engine
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
pub fn trace(sku: &str, platform: Platform) -> Result<Plan> {
    let trace = ::model::trace_of(sku).ok_or_else(|| anyhow!("{}", no_such_sku(sku)))?;
    Ok(trace(platform))
}

/// How `sku` sorts a request into the fact word its lanes carry — the fourth
/// column, reached through the same door as the trace.
///
/// **THIS IS THE HALF OF DECISION 18 THAT RUNS EVERY FIRE.** A lane's word is
/// what `driver::fire::compose` turns into a class and therefore into the row
/// window every guarded node runs over; a fire that submits word 0 for every
/// lane runs its decode rows through the prefill arm. The engine could not
/// state it while the catalog shipped three columns — a plan's `Cond::Fact`
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
/// the deployments this engine serves have both.
fn open_source(checkpoint: &Path) -> Result<ztensor::Source> {
    let container = if checkpoint.is_dir() {
        let mut found: Vec<PathBuf> = std::fs::read_dir(checkpoint)
            .with_context(|| format!("read the checkpoint directory {checkpoint:?}"))?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
            })
            .collect();
        found.sort();
        found.into_iter().next().ok_or_else(|| {
            anyhow!("{checkpoint:?} holds no `.safetensors` and no `.zt` container")
        })?
    } else {
        checkpoint.to_path_buf()
    };
    ztensor_compat::index(&container)
        .or_else(|_| ztensor::Source::open(&container))
        .with_context(|| format!("open {container:?} as a tensor container"))
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
/// `model_loader::plan::compile` is what does the checking, and it is the
/// same call `driver_cuda::weights::Weights::resident` makes: this is the
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
        Platform::Metal => model_loader::types::BackendKind::Metal,
        Platform::Vulkan | Platform::Wgpu => model_loader::types::BackendKind::Vulkan,
        Platform::Cuda => model_loader::types::BackendKind::Cuda,
    };
    let target = model_loader::plan::StorageTarget::for_backend(backend, 0, 1);

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
        match model_loader::plan::compile(&metadata, &contract, target.clone()) {
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
fn checkpoint_metadata(checkpoint: &Path) -> Result<model_loader::checkpoint::CheckpointMetadata> {
    if checkpoint.is_dir() {
        model_loader::checkpoint::read::parse_checkpoint_metadata(checkpoint)
            .map_err(|error| anyhow!("reading {checkpoint:?}: {error}"))
    } else {
        model_loader::checkpoint::zt::parse_checkpoint(checkpoint)
            .map_err(|error| anyhow!("reading {checkpoint:?}: {error}"))
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
/// carries a sentence and the pointer must not put an error crate in a
/// driver's signature.
pub fn contract_for(plan: &Plan, checkpoint: &Path) -> std::result::Result<ModelContract, String> {
    let import = ::model::import_of(&plan.name).ok_or_else(|| {
        format!(
            "this build ships no import contract for {:?}, so a checkpoint's \
             tensors cannot be mapped onto its params",
            plan.name
        )
    })?;
    let source = open_source(checkpoint).map_err(|error| format!("{error:#}"))?;
    import(&source).map_err(|error| {
        format!(
            "the import contract for {:?} does not fit {checkpoint:?}: {error}",
            plan.name
        )
    })
}

/// Everything a load states, for a checkpoint this build can identify.
///
/// # Errors
///
/// A checkpoint no SKU claims, or a SKU whose trace this build does not ship.
pub fn request(
    checkpoint: &Path,
    platform: Platform,
    budgets: Budgets,
    ordinal: i32,
) -> Result<LoadRequest> {
    let sku = identify(checkpoint, platform)?;
    Ok(LoadRequest {
        plan: trace(sku, platform)?,
        checkpoint: Checkpoint::Path(checkpoint.to_path_buf()),
        budgets,
        ordinal,
    })
}

/// Everything a load states, for a checkpoint whose SKU the caller already
/// knows.
///
/// # Errors
///
/// A SKU whose trace this build does not ship.
pub fn request_for(
    sku: &str,
    checkpoint: &Path,
    platform: Platform,
    budgets: Budgets,
    ordinal: i32,
) -> Result<LoadRequest> {
    Ok(LoadRequest {
        plan: trace(sku, platform)?,
        checkpoint: Checkpoint::Path(checkpoint.to_path_buf()),
        budgets,
        ordinal,
    })
}
