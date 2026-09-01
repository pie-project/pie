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
pub type Row = (&'static str, u32, fn(Platform) -> Trace, models::ClassifyFn);

/// Every SKU this build ships.
///
/// A pass-through of `models::catalog`, here so that the rest of the runtime
/// reads the catalog through the load door rather than reaching into the
/// model crate at eleven sites.
#[must_use]
pub fn catalog() -> Vec<Row> {
    models::catalog()
}

/// The traced supergraph for `sku`, on `platform`.
///
/// # Errors
///
/// When this build's catalog has no such SKU — with the near misses named,
/// because "unknown model" and "you meant the -bf16 row" are different
/// operator actions.
pub fn trace(sku: &str, platform: Platform) -> Result<Trace> {
    let trace = models::trace_of(sku).ok_or_else(|| anyhow!("{}", no_such_sku(sku)))?;
    Ok(trace(platform))
}

/// **The served numeric form a catalog row states**, for the one caller that
/// has to write it down: `pie model import`, which puts it in the artifact's
/// `pie.serving/1` stamp and in its filename.
///
/// Beside [`catalog`], [`trace`] and [`classify`] because it is the same kind
/// of thing — a fact the catalog states about a row — and because `pie` does
/// not depend on `models` directly, it reaches it through this door.
///
/// # Errors
///
/// A row this build's catalog does not ship. Not a default: `precision` is a
/// field `serving::Stamp::check` COMPARES, so a stamp carrying a placeholder
/// would be checked against a deployment and either refuse a good artifact or
/// pass a bad one. `models::precision_disagreements` is what makes this
/// unreachable for a row that is in the catalog.
pub fn precision(sku: &str) -> Result<&'static str> {
    models::precision_of(sku).ok_or_else(|| {
        anyhow::anyhow!(
            "the catalog claims `{sku}` and states no precision for it; the artifact's \
             stamp and its filename both carry that field"
        )
    })
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
pub fn classify(sku: &str) -> Result<models::ClassifyFn> {
    models::classify_of(sku).ok_or_else(|| anyhow!("{}", no_such_sku(sku)))
}

/// The one refusal both doors above raise, written once: an unknown SKU, and
/// the ones this build does ship.
fn no_such_sku(sku: &str) -> String {
    format!(
        "this build ships no SKU named {sku:?}; it ships {}",
        models::catalog()
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
/// `models::identify` returns the first SKU whose import contract builds, and
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
    for (sku, tp, import) in models::imports() {
        // A row traced for more than one rank describes a shard, not a
        // checkpoint; `models::identify` skips those for the same reason.
        if !models::catalog()
            .iter()
            .any(|(name, tp, ..)| *name == sku && *tp == 1)
        {
            continue;
        }
        // **READ FOR THE SETUP THIS IS IDENTIFYING FOR** (§J4c). A family's
        // text may state a `Dtype` PLACEMENT — an arrangement of a bank's
        // bytes some platforms' kernels read and others cannot — and
        // `model_dsl::place` resolves one against the platform the
        // declaration is read under. The trace gets that through `catalog!`;
        // an import contract takes a checkpoint and a world size and gets it
        // here, from the same `platform` the target above is built for. Read
        // under no setup, a contract would state the text's own arrangement
        // and then be checked against a plan traced for this one — two
        // readings of one text, free to disagree about a plane neither can
        // see.
        let contract = match models::placing_for(platform, || import(&source, tp)) {
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

/// **THE SETUP THIS BINARY SERVES**, for the commands that convert rather
/// than serve.
///
/// A serving boot never asks: `worker::embedded_engine` opens a backend and
/// hands the `Platform` down beside it, so a shell's own platform is a fact
/// about the shell it opened. `pie model import` has no shell — it reads a
/// checkpoint and writes an artifact — and it still has to name a setup,
/// because an artifact IS one (§M, and [`conversion_contract`]'s note). The
/// honest answer for a converter is the box it is converting on, which is
/// what its prepare already answers for: the same device, the same budgets,
/// the same tier key.
///
/// **THE FEATURE IS THE FACT AND THE TARGET IS NOT.** `engine-metal` is
/// target-gated as well as feature-gated — a build with the flag and no Apple
/// target links a shell whose device half does not exist — so the flag alone
/// is what says a binary MEANT Metal, and pairing it with the target is what
/// says the device is there. A build with both shells linked is a real build
/// and CUDA wins it: a box with an NVIDIA card is converting for the card.
///
/// A build with NO engine linked converts for `Cuda`. That is the shape of
/// every artifact this tree has ever written and the one an unqualified
/// conversion keeps writing, so a converter-only machine's output does not
/// change under this wave.
#[must_use]
pub fn this_box() -> Platform {
    #[cfg(feature = "_engine-cuda")]
    {
        Platform::Cuda
    }
    #[cfg(all(not(feature = "_engine-cuda"), feature = "engine-metal", target_vendor = "apple"))]
    {
        Platform::Metal
    }
    #[cfg(all(
        not(feature = "_engine-cuda"),
        not(all(feature = "engine-metal", target_vendor = "apple"))
    ))]
    {
        Platform::Cuda
    }
}

/// **WHAT THIS SOURCE WOULD BE CONVERTED AS** — the SKU whose import contract
/// this checkpoint satisfies, asked at the CONVERSION target rather than at a
/// device one.
///
/// [`identify`] above answers the SERVING question and compiles against a
/// backend, so a contract that states a conversion — a bank a family declares
/// quantized over a checkpoint that ships it raw — is refused there, and
/// refused correctly: no device target carries an encode
/// ([`CUDA_TILE_MAP_MASK`](checkpoint::plan::CUDA_TILE_MAP_MASK)). That is
/// exactly the source `pie model import` has to be able to name, so it asks
/// this instead, against the mask conversion compiles under.
///
/// `None` and not an error: an unidentifiable source is not a failed import.
/// Every checkpoint this command has ever converted converts the same way
/// whether or not a SKU claims it, and the contract is read for one reason
/// only — to learn which tensors this build would have to encode.
///
/// **AND IT TAKES A `platform`, BECAUSE AN ARTIFACT IS SETUP-SPECIFIC** (§M,
/// §J4c). "Every checkpoint converts the same way" is true of the ENCODES
/// above and was never true of a PLACEMENT: a repack is an arrangement of a
/// bank's bytes into the fragment order one shell's kernels read, and writing
/// one into an artifact a different shell will boot is how a `.zt` comes to
/// load in 0.1s and answer nonsense. §M already said what makes a tier key —
/// "a function of the RECIPE — backend, tensor parallelism, precision" — and
/// this is that sentence reaching the planes themselves. The command passes
/// the setup it is converting FOR, which for `pie model import` is the box it
/// is running on, the same box whose device and budgets its prepare answers
/// for.
#[must_use]
pub fn conversion_contract(
    checkpoint: &Path,
    platform: Platform,
) -> Option<(&'static str, ModelContract)> {
    let source = open_source(checkpoint).ok()?;
    let metadata = checkpoint_metadata(checkpoint).ok()?;
    let target = checkpoint::plan::StorageTarget {
        tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
        ..checkpoint::plan::StorageTarget::default()
    };
    for (sku, tp, import) in models::imports() {
        // A row traced for more than one rank describes a shard, not a
        // checkpoint — [`identify`]'s rule, for its reason.
        if !models::catalog()
            .iter()
            .any(|(name, tp, ..)| *name == sku && *tp == 1)
        {
            continue;
        }
        let Ok(contract) = models::placing_for(platform, || import(&source, tp)) else {
            continue;
        };
        if checkpoint::plan::compile(&metadata, &contract, target.clone()).is_ok() {
            return Some((sku, contract));
        }
    }
    None
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
    let import = models::import_of(&trace.name).ok_or_else(|| {
        format!(
            "this build ships no import contract for {:?}, so a checkpoint's \
             tensors cannot be mapped onto its params",
            trace.name
        )
    })?;
    let source = open_source(checkpoint).map_err(|error| format!("{error:#}"))?;
    // **THE PLAN'S OWN PLATFORM, WHICH IS THE WHOLE POINT OF ASKING HERE**
    // (§J4c). This contract exists to publish a checkpoint's tensors under
    // THIS plan's param names, so the setup it is read for is the setup the
    // plan was traced for and cannot be anything else. `Trace::platform` is
    // that word, already on the value in hand — see `identify` above for what
    // a reading with no setup would cost.
    models::placing_for(trace.platform, || import(&source)).map_err(|error| {
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
    request_of(None, checkpoint, platform, budgets, residency, ordinal, frames_in_flight)
}

/// **THE ONE DOOR, WITH THE OPERATOR'S ROW OR WITHOUT IT.**
///
/// `sku` is `[model] sku`: `None` identifies one ([`identify`]), and a name
/// takes that row's trace without asking the checkpoint a second question.
///
/// **NAMING A ROW IS NOT A SECOND KIND OF LOAD**, which is why this is one
/// function and not two — `request_for` stood beside `request` with no caller
/// in the tree, and two doors onto the same load is what let the second one
/// rot. A checkpoint that fits two rows is the whole reason the key exists: a
/// vision artifact holds a text trunk AND a tower, so it fits its family's
/// text row and its own, and identification takes the first — deliberately the
/// cheap one, because a two-unit load stands the fold down. An operator who
/// wants the tower says so.
///
/// A named row that does NOT fit is refused where every mismatched load is:
/// `checkpoint::plan::compile`, inside the weight residency, naming the param
/// and both shapes. That is later than [`identify`]'s own verdict and it is
/// the honest place for it — the operator named a row, so "this checkpoint
/// does not hold it" is an answer about the row they named.
///
/// # Errors
///
/// A checkpoint no SKU claims (when identifying), or a SKU whose trace this
/// build does not ship.
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
    // **THE TWO FACTS A SHELL CANNOT LOOK UP** (§M-4c). A shell must not know
    // a model family — both engines' `models` edges are DEV for that reason —
    // so the degree and the served numeric form are stated here, by the one
    // party that sees the catalog. The other three facts a serving artifact is
    // checked against are already in the trace (`name` is the sku,
    // `platform.backend()` is the backend), which is why they are not
    // restated: a request carrying its own copy of a fact the trace already
    // holds is a fact that can disagree with itself.
    let tp_size = tp_of(&sku);
    let precision = precision(&sku)?.to_string();
    Ok(LoadRequest {
        trace: trace(&sku, platform)?,
        checkpoint: Checkpoint::Path(checkpoint.to_path_buf()),
        budgets,
        residency,
        ordinal,
        frames_in_flight,
        tp_size,
        precision,
    })
}

/// **THE ORDER A SERVING ARTIFACT'S PLANES SHOULD LIE IN** — the shell's own
/// ranking, as plane names, for `pie model import` to write them in.
///
/// `emit` writes the objects in the order it is handed and
/// `checkpoint::serving::sequence` reads that order back off the offsets. What
/// makes it worth getting right is the one thing a payload run is for: a
/// streaming boot walking the file FORWARD ONCE, hottest planes first, rather
/// than seeking through a name-ordered file. An artifact in any other order is
/// still correct — `emit`'s doc says it "reads perfectly and merely reads
/// unranked" — so this is locality and never meaning.
///
/// # It is HERE and not in the shell, and that is the dependency direction
///
/// The ranking is `engine_cuda::experts::Ranking`, and it needs to know which
/// planes travel together. A boot learns that from a COMPILED LOAD PLAN, and
/// `pie model import` has none: the plan would be compiled against the
/// artifact it is about to write. What states the pairing without a plan is
/// [`models::scales_name`] and `biases_name` — the two functions that MINT
/// a companion's name — asked of a trace whose params those same calls
/// declared. **That is not a suffix match**, which this tree refuses by name;
/// it is the minting function run forwards.
///
/// And `model_dsl` is a DEV dependency of both shells on purpose — a shell
/// must not know a model family — so the derivation cannot live in one. It
/// lives at the seam that already sees both, beside [`precision`] and
/// [`this_box`].
///
/// `None` when no shell is linked. A build that cannot rank writes an
/// unranked artifact, which is the same graceful answer `--no-prepare`'s
/// shelf conversion has always given.
#[cfg(feature = "_engine-cuda")]
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

/// [`sequence`], for a build with no shell linked: no ranking, and an
/// unranked artifact is a correct one.
#[cfg(not(feature = "_engine-cuda"))]
#[must_use]
pub fn sequence(_trace: &Trace) -> Option<Vec<String>> {
    None
}

/// The tensor-parallel degree the catalog states for `sku`.
///
/// One for a row this build does not ship, which is unreachable from
/// [`request_of`] — the sku there either came from [`identify`], which only
/// returns rows that are in the catalog, or from `[model] sku`, whose trace
/// lookup one line below refuses a name the catalog does not hold. It is a
/// default rather than an error only because there is no reachable caller to
/// give an error to.
fn tp_of(sku: &str) -> u64 {
    models::catalog()
        .iter()
        .find(|(row, _, _, _)| *row == sku)
        .map_or(1, |(_, tp, _, _)| u64::from(*tp))
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
