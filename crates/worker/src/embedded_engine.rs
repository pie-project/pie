//! Engine-backend bootstrap helpers for pie-worker.
//!
//! This module exposes:
//!   * [`EngineCapabilities`] — typed engine capability payloads.
//!   * `device_boot` — maps the operator's options onto the cuda shell's own
//!     [`DeviceBoot`], which crosses the seam as a struct. The per-launch
//!     bootstrap TOML it replaced is gone with the file round-trip.
//!   * [`create_engine_backend`] — build a runtime-owned [`::runtime::engine::EngineBox`]
//!     plus its caps before `::runtime::bootstrap`.

use std::path::Path;
// `PathBuf` appears only in cuda-gated signatures (`prepare_weight_artifact`,
// `device_boot`), so the import is gated as they are.
#[cfg(feature = "_engine-cuda")]
use std::path::PathBuf;

// `Context` is NOT cuda-gated, though it was: `register_operator_adapters`
// reads its plane files and registers them on every build, so a gate that
// named only the cuda and apple-metal seams left `cargo test -p worker` (which
// builds the lib with `cfg(test)` false) unable to resolve `with_context`.
use anyhow::{Context, Result, anyhow};

#[cfg(feature = "_engine-cuda")]
use ::runtime::engine::backend::{DeviceBoot, Graphs, Knobs, ordinal_of};

#[cfg(any(feature = "_engine-cuda", test))]
use crate::config::CudaNativeEngineOptions;
#[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
use crate::config::MetalEngineOptions;
use crate::engine_ffi::Flavor;

// THE TWO LINK ANCHORS ARE GONE WITH THE C++ THEY SERVED.
//
// `PIE_LOADER_ENTRY_ANCHOR` and `PIE_FORWARD_ENTRY_ANCHOR` existed because a
// linker never pulls an rlib member in on behalf of a C++ reference: the only
// callers of `pie_loader_compile_model` and `pie_forward_trace_llama_like`
// were the C++ engines, which link after Rust, so without a reference from
// reachable Rust the entry points were simply absent at final link.
//
// Both engines are Rust now. `checkpoint` and `model` are called directly,
// through their own types, and there is nothing on the far side of an FFI
// boundary to keep alive.

// THE NCCL UNIQUE-ID MINT STOOD HERE — an `extern "C"` pair
// (`ncclGetUniqueId`, `ncclGetErrorString`), the only NCCL symbols in the
// workspace, minting the id a TP group rendezvoused on. Its one consumer was
// the boot TOML's `[distributed]` table, which no engine read; both left with
// the file, and `build.rs` lost its `-lnccl` with them.

/// Per-flavor engine options, passed to native-engine creation helpers so the
/// caller doesn't have to discriminate on `EngineKind` in two places.
///
/// `Clone` exists so `serve.rs` can rebuild a per-group variant
/// (different `device`) from a model-level template without
/// re-deserializing TOML.
#[derive(Clone)]
pub enum EngineOptions {
    #[cfg(feature = "_engine-cuda")]
    CudaNative(CudaNativeEngineOptions),
    #[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
    Metal(MetalEngineOptions),
    // `Metal`, `Vulkan` and `Wgpu` STOOD HERE, with their engines, until R3
    // took all three out of the workspace. They return at P5; see
    // `engine_ffi::retired_msg`.
}

impl EngineOptions {
    /// Which compiled flavor this options bundle targets.
    ///
    /// With no `engine-*` feature this enum has NO variants, so there is no
    /// value to be called on and the match is empty. That is stated with a
    /// wildcard rather than left to inference, because an empty match on a
    /// `&self` of an uninhabited type is not something the compiler will
    /// accept as exhaustive through a reference.
    pub fn flavor(&self) -> Flavor {
        match self {
            #[cfg(feature = "_engine-cuda")]
            EngineOptions::CudaNative(_) => Flavor::Cuda,
            #[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
            EngineOptions::Metal(_) => Flavor::Metal,
            #[cfg(not(any(
                feature = "_engine-cuda",
                all(feature = "engine-metal", target_vendor = "apple")
            )))]
            _ => unreachable!("`EngineOptions` has no variants in this build"),
        }
    }
}

// `TpLaunch` STOOD HERE. Its only real consumers were the boot TOML's
// `[distributed]` table — written, read by no engine — and the launch state
// dir's rank suffix; both are gone, and `open::cuda_group` still refuses a
// multi-rank launch by name until `palo B-tp` builds one.

/// **WHERE THIS DEPLOYMENT'S MATERIALIZED WEIGHTS GO, DECIDED ONCE.**
///
/// `[model] weight_cache_dir` when the operator wrote one, and
/// `$PIE_HOME/cache/weights` when they did not. `$PIE_HOME` is this layer's to
/// know — the engine has never been told it, which is why the old env-var form
/// fell back to XDG — so the derivation is here and not in the shell.
///
/// Under `cache/`, not `models/`. `models/` is the artifact store, and these
/// are the opposite kind of thing: device bytes for one engine, one TP layout
/// and one ABI version, rebuilt by a single cold load. Sharing a directory
/// left `.weights` files sitting in a store that scans for `.zt`.
///
/// **TWO CALLERS, ONE ANSWER** (§M wave M-1). The serving boot resolves it
/// once and threads it into every `DeviceBoot`, and `pie model import` reads
/// it to decide whether there is anywhere to prepare INTO. A second
/// derivation would be a second chance for the two to disagree about where a
/// hundred gigabytes live.
#[must_use]
pub fn resolved_weight_cache_dir(cfg: &crate::config::Config) -> String {
    if cfg.model.weight_cache_dir.is_empty() {
        crate::state::weight_cache_dir()
            .to_string_lossy()
            .into_owned()
    } else {
        cfg.model.weight_cache_dir.clone()
    }
}

// THE CACHE-ROOT `OnceLock`, THE NINE TOML HELPERS AND THE LAUNCH-STATE
// DIRECTORY STOOD HERE. `$PIE_HOME/cache` — the root every engine-side disk
// cache derives from, convention rather than configuration — crosses as a
// `DeviceBoot` field now instead of through a process global, and with no
// per-launch TOML to write there is no `$PIE_HOME/standalone/<pid>` tree to
// create, sweep at boot, or remove at shutdown. `state::engine_cache_dir`
// is still the one spelling of the root.

/// What a load answered about itself.
///
/// A rename, not an alias with a new spelling behind it: the 30-field
/// `EngineCapabilities` mixed three subjects (the device, the load, and the
/// MODEL) and the contract's [`Capabilities`](engine::Capabilities)
/// separates them — `device`, `pools`, `limits`, and a
/// `ModelProfile` carried whole rather than rebuilt from eight booleans
/// (`engine::caps`'s header). Three of the old fields have no successor
/// because they were never the engine's to answer: `snapshot_dir`,
/// `model_id` and `arch_name` say where the CALLER's checkpoint came from,
/// and the caller is this crate. They are on
/// [`GroupEngine`](crate::translate::GroupEngine) now.
pub use engine::Capabilities as EngineCapabilities;

/// The ceilings a load is baked against, out of what the operator stated.
///
/// **THIS IS WHERE `ModelLoadDesc` WENT.** Its four fields said nothing this
/// does not: `snapshot_dir` is [`Checkpoint::Path`](engine::Checkpoint),
/// `component` is which `Trace` you hand over, and `runtime_quant`/`mxfp4_moe`
/// were a quantization word and a MoE lowering name that a backend
/// string-matched — the plan's params carry their own dtypes, and which
/// kernel answers an op is the dispatch arm's decision (design §6). What is
/// left is arithmetic about the pools, and it is the operator's.
///
/// `slots` is derived rather than stated because the two knobs an operator
/// has are a page count and a context length, and the shell's paging hands
/// each seated sequence one block of `max_context / page_size` pages: how
/// many sequences fit is that division, not a third knob to keep in step.
#[cfg(any(feature = "_engine-cuda", test))]
fn cuda_budgets(
    opts: &CudaNativeEngineOptions,
    adapter_seats: u32,
    patch_ceilings: (Option<u32>, Option<u32>),
) -> engine::Budgets {
    let page_size = opts.kv_page_size.unwrap_or(16).max(1);
    // No CUDA knob states a context ceiling — `max_model_len` is the Metal
    // options' — so this is the contract's own default, stated once here
    // rather than guessed twice.
    let max_context = engine::Budgets::default().max_context;
    let pages_per_slot = max_context.div_ceil(page_size).max(1);
    engine::Budgets {
        max_lanes: opts.max_forward_requests.unwrap_or(256).max(1),
        max_tokens: opts.max_forward_tokens.unwrap_or(8192).max(1),
        // Empty is a DEFERRAL, not "no buckets": `engine_cuda::api::lattice`
        // fills a stated-nothing budget with `default_lattice` (8, 16, …,
        // max_tokens), and under `[engine] graphs = "on"` — the default —
        // fires pad to those rungs and replay one captured exec per bucket.
        // This line's old claim ("v1 pads nothing: a fire's shape IS its
        // graph key") described `Graphs::Off`, which is a diagnostic mode
        // now, not the served one.
        buckets: Vec::new(),
        // **THE BUDGET IS AN INTENT, AND NOW AN OPERATOR NAMES IT** (palo C2,
        // alto D1). Capacity is a SHAPE the model text declares — every bank
        // a plan carries is reserved at load whatever this number is, and
        // `Engine::register_adapter` is checked against that shape — so what
        // `max_adapters` states is how many the DEPLOYMENT intends to
        // register, and `model_compiler::compile` refuses a load whose intent
        // is bigger than what the text seats.
        //
        // It was hard-coded `0` under a comment saying "the knob, and the
        // request-side id that would make it worth having, arrive with the
        // client-facing half". The knob is `[model.adapters]` and it arrives
        // here as one number (article 8): `AdapterConfig::seats`, which is
        // what the operator stated or what their roster needs. Zero is still
        // the answer for a deployment that states nothing, and zero is still
        // the load where the correction op never launches.
        max_adapters: adapter_seats,
        page_size,
        max_context,
        slots: opts
            .max_total_pages
            .map_or(256, |pages| (pages / pages_per_slot).max(1)),
        // **THE SECOND ROW AXIS, AND `None` IS THE ANSWER FOR ALMOST EVERY
        // DEPLOYMENT** (alto multimodal §5.5). Both absent means the shell
        // derives a ladder from the loaded TEXT — rungs at whole images when
        // the plan states a patch axis, and no ladder at all when it does not
        // — so a vision SKU serves with zero configuration and a text-only
        // one is byte-for-byte the load it always was.
        //
        // `[model]` rather than `[model.engine.options]` for `max_adapters`'
        // reason: a ceiling on a row axis is not a fact about a backend.
        max_patches: patch_ceilings.0,
        max_images: patch_ceilings.1,
    }
}

/// The cuda boot, as the shell's own type — `write_cuda_startup_toml`'s
/// successor, and the spec of what actually crosses: the shell reads the
/// device, the two cache directories and the `[engine]` knobs, and nothing
/// else the old TOML carried was read by anything.
///
/// **ONLY WHAT THE OPERATOR STATED DEVIATES FROM THE SHELL'S OWN DEFAULTS.**
/// The knob fields on [`CudaNativeEngineOptions`] are `Option`s for exactly
/// this: an absent knob takes `Knobs::default()` — the ENGINE's answer —
/// rather than a default this layer invented, so the two sides cannot hold
/// different beliefs about what "unstated" means. `gpu_mem_utilization` is
/// the one knob with no absence to express: the config type has already
/// turned its absence into `0.90`, so it always crosses.
///
/// # Errors
///
/// A `[engine] graphs` spelling the shell does not speak — the config layer
/// does not police that word, so this is the first and only refusal.
#[cfg(feature = "_engine-cuda")]
fn device_boot(
    opts: &CudaNativeEngineOptions,
    weight_cache_dir: &Path,
    cache_dir: &Path,
    adapter_dir: Option<&Path>,
) -> Result<DeviceBoot> {
    let graphs = match opts.graphs.as_deref() {
        None => Graphs::default(),
        Some(word) => word
            .parse::<Graphs>()
            .map_err(|error| anyhow!("[engine] graphs: {error}"))?,
    };
    let mut knobs = Knobs {
        gpu_mem_utilization: opts.gpu_mem_utilization,
        ..Knobs::default()
    };
    if let Some(pad) = opts.pad {
        knobs.pad = pad;
    }
    if let Some(bodies) = opts.bodies {
        knobs.bodies = bodies;
    }
    if let Some(megabytes) = opts.bodies_mem {
        knobs.bodies_mem = megabytes;
    }
    if let Some(copies) = opts.fallback_copy {
        knobs.copies = copies;
    }
    if let Some(grouped) = opts.grouped {
        knobs.grouped = grouped;
    }
    if let Some(streams) = opts.side_streams {
        knobs.side_streams = Some(streams);
    }
    Ok(DeviceBoot {
        ordinal: ordinal_of(&opts.device),
        graphs,
        knobs,
        weight_cache_dir: Some(weight_cache_dir.to_path_buf()),
        cache_dir: Some(cache_dir.to_path_buf()),
        adapter_dir: adapter_dir.map(Path::to_path_buf),
    })
}

/// A one-way, human-readable record of what a boot actually asked for, under
/// `$PIE_HOME/logs/` — the successor to the operator-readable `engine.toml`.
///
/// **NOTHING MAY EVER READ THIS BACK.** It is a dump, not a wire: the struct
/// already crossed, and a reader here would be a second boot format growing in
/// a log directory. Best-effort for the same reason the launch-state sweep
/// was: a dump must never be why a boot fails.
#[cfg(feature = "_engine-cuda")]
fn dump_device_boot(boot: &DeviceBoot, group_id: usize, rank: Option<usize>) {
    let dir = crate::paths::pie_home().join("logs");
    let name = match rank {
        Some(rank) => format!("engine-boot-g{group_id}-r{rank}.txt"),
        None => format!("engine-boot-g{group_id}.txt"),
    };
    if let Err(error) = std::fs::create_dir_all(&dir)
        .and_then(|()| std::fs::write(dir.join(&name), format!("{boot:#?}\n")))
    {
        tracing::warn!(%error, name, "could not write the engine boot dump");
    }
}

/// **PREPARE ONE FRESHLY IMPORTED ARTIFACT'S WEIGHT TIERS** (§M wave M-1:
/// `.wiki/alto/zt-as-serving-artifact.md`).
///
/// `pie model import` calls this once, after it has written a servable `.zt`,
/// so that the tier artifact §K/§L reads on a warm boot is written HERE
/// instead of by the first serve. Nothing is served and no engine survives the
/// call: the whole product is the file left in the weight cache directory,
/// which is returned so the command can name it.
///
/// # What it reads, and why none of it is new
///
/// Every number comes off the deployment's own `[model]` table, through the
/// same doors the serving boot uses and in the same order:
///
/// ```text
///   flavor + options   preflight::{resolve_flavor, build_embedded_options}
///   the boot struct    device_boot — dumped one-way under logs/ when verbose
///   the token budget   cuda_budgets, off the same `[engine] options`
///   the two weight     ModelConfig::residency — `[model] device_weight_budget`
///     budgets          and `[model] host_weight_budget`
///   the cache dir      resolved_weight_cache_dir
///   the run-ahead      `[runtime] frame_dispatch_depth` (article 8)
/// ```
///
/// There is no second reader anywhere in this function. That matters more
/// here than anywhere else in the tree: the tier artifact's KEY is a function
/// of the residency plan the two weight budgets decide, so an import that read
/// a budget differently from the serve would write a file under a name the
/// serve never looks up — a hundred gigabytes of disk and no warm boot.
///
/// # What it is pointed at
///
/// `artifact`, the file the import just wrote — NOT `[model] model`. An
/// operator importing a model the config does not name yet is the ordinary
/// case, and preparing whatever the config happens to point at would be a
/// command doing something nobody asked for.
///
/// `[model] sku` is honoured only when the config's model resolves to THIS
/// file, for the same reason: the key names a row, and a row stated about
/// another checkpoint is not a statement about this one. Otherwise the SKU is
/// identified from the checkpoint, which is what an unconfigured serve does.
///
/// # Errors
///
/// A config whose engine is not the CUDA shell, an artifact that is neither a
/// `.zt` nor a snapshot directory, a checkpoint no SKU claims, or whatever
/// the bake and the landing refused. **Every one of them is survivable**: the
/// import's own product is already on disk and is a good checkpoint.
///
/// **WHAT "SURVIVABLE" MEANS HAS NARROWED, THOUGH** (§M-3). It used to mean a
/// boot with no tier artifact ran the cold path it had always run; there is no
/// cold path, so a streamed SKU whose prepare failed will not boot at all until
/// one succeeds. The call still returns an error rather than panicking and the
/// ordinary post-import caller still exits zero — the artifact it was asked for
/// exists — but `pie model import --prepare-only` propagates it, because
/// preparing is the whole of what that invocation was asked to do.
#[cfg(feature = "_engine-cuda")]
pub fn prepare_weight_artifact(cfg: &crate::config::Config, artifact: &Path) -> Result<PathBuf> {
    let m = &cfg.model;
    validate_snapshot_dir(artifact)?;
    let flavor = crate::preflight::resolve_flavor(m.engine.kind, &m.name)?;
    let options = crate::preflight::build_embedded_options(m, flavor)?;
    // THE `else` IS UNREACHABLE IN A CUDA-ONLY BUILD AND LOAD-BEARING IN ANY
    // OTHER, exactly as it is in `create_engine_backend_group`.
    #[allow(
        irrefutable_let_patterns,
        reason = "`EngineOptions` has one variant in a CUDA-only build"
    )]
    let EngineOptions::CudaNative(opts) = &options else {
        return Err(anyhow!(
            "this build prepares weight artifacts for the cuda shell only; \
             `[model.engine] type` names {:?}",
            m.engine.kind
        ));
    };

    // `$PIE_HOME` is this layer's to know; both directories cross as
    // `DeviceBoot` fields, resolved here exactly as the serving boot resolves
    // them.
    let engine_cache_dir = crate::state::engine_cache_dir();
    let cache_dir = PathBuf::from(resolved_weight_cache_dir(cfg));
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("create the weight cache directory {cache_dir:?}"))?;

    // The same open the serving path makes, kept for its refusal: an artifact
    // that cannot answer its metadata fails here rather than at the first
    // serve. The config it lifts used to travel beside the boot TOML; nothing
    // read it, and nothing crosses here now.
    crate::weights::resolve(&artifact.to_string_lossy())
        .with_context(|| format!("resolving the artifact {artifact:?}"))?
        .metadata()
        .with_context(|| format!("reading the model metadata for {artifact:?}"))?;

    let boot = device_boot(
        opts,
        &cache_dir,
        &engine_cache_dir,
        m.adapter_mount().as_deref(),
    )?;
    if cfg.server.verbose {
        dump_device_boot(&boot, 0, None);
    }

    let request = ::runtime::engine::load::request_of(
        stated_sku(m, artifact),
        artifact,
        model_ir::Platform::Cuda,
        cuda_budgets(opts, m.adapters.seats(), m.patch_ceilings()),
        m.residency(),
        -1,
        u8::try_from(cfg.runtime.frame_dispatch_depth).unwrap_or(u8::MAX),
    )?;
    ::runtime::engine::backend::open::prepare_cuda(boot, request)?;
    Ok(cache_dir)
}

/// `[model] sku`, but only when the config is talking about THIS checkpoint.
///
/// A row named in the config is a statement about `[model] model`. Applied to
/// a different file it is not a hint, it is a wrong answer — and one that
/// would be paid for in a full cold load. `None` identifies instead, which is
/// what a serve with no `[model] sku` does.
#[cfg(feature = "_engine-cuda")]
fn stated_sku<'a>(m: &'a crate::config::ModelConfig, artifact: &Path) -> Option<&'a str> {
    let sku = m.sku.as_deref()?;
    let resolved = crate::weights::resolve(&m.model).ok()?;
    let same =
        std::fs::canonicalize(resolved.path()).ok()? == std::fs::canonicalize(artifact).ok()?;
    same.then_some(sku)
}

/// Hand an engine its model: trace the plan runtime-side, state the ceilings,
/// and land the checkpoint.
///
/// The tracing is the RUNTIME's (design §7, decision 18) and reaching it
/// through `::runtime::engine::load` is what keeps `model` out of this crate's
/// dependency graph — the note on the manifest's deleted `model` edge is the
/// same ruling from the other side.
fn land(
    backend: &mut ::runtime::engine::EngineBox,
    snapshot_dir: &Path,
    budgets: engine::Budgets,
    residency: engine::Residency,
    platform: model_ir::Platform,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    // **`[model] sku`, OR `None` TO IDENTIFY ONE** (media-door §6). A
    // checkpoint that fits two rows — a vision artifact fits its text trunk's
    // row and its own — is identified as the first that fits, and the
    // operator names the other one when they want it.
    sku: Option<&str>,
) -> Result<engine::Loaded> {
    if component != crate::executor::ModelComponent::Full {
        // palo B-component: an encoder is a traced plan like any other, and
        // the catalog ships no encoder trace. Refused by name rather than
        // loaded as the full model, which is what the old
        // `ModelComponent::Encode` did — it staged the whole 48 GiB
        // checkpoint and died in `cudaMalloc`.
        return Err(anyhow!(
            "this build loads only the full model; {component:?} needs a traced plan the catalog \
             does not ship"
        ));
    }
    // **THE RUN-AHEAD DEPTH, CROSSING ONCE** (article 8). `[runtime]
    // frame_dispatch_depth` is the deployment's statement of how many frames
    // it will keep posted to the engine; the engine's staging ring and
    // settlement event pool are sized from it and from nothing else.
    // **THE OPERATOR'S ROW WINS, AND ONLY WHEN THEY STATED ONE.** `request`
    // identifies; `request_for` takes a name the caller already knows and
    // asks the catalog for its trace. A sku that does not fit this checkpoint
    // is refused where every mismatched load is — `checkpoint::plan::compile`,
    // inside the weight residency, naming the param and both shapes — which is
    // the same sentence an operator gets today for a checkpoint that fits no
    // row at all.
    let request = ::runtime::engine::load::request_of(
        sku,
        snapshot_dir,
        platform,
        budgets,
        // **THE DAY ARRIVED: `[model]`'S TWO WEIGHT BUDGETS LAND HERE**
        // (alto design §7, W-3). The residency policy is two budgets and
        // `None` is uncapped, so a config that states neither reaches the
        // engine as `Residency::uncapped()` and this deployment behaves
        // exactly as it did before the keys existed -- the whole weight table
        // on the device, never moving. A budget a shell cannot meet by
        // holding LESS refuses the load by name, with both numbers, rather
        // than being rounded to the nearest thing that is built.
        //
        // `[model] device_weight_budget` is not `[engine]
        // gpu_mem_utilization`: this one bounds the model's weight table,
        // that one bounds the elastic physical pool (KV pages and scratch).
        // `crate::config::ModelConfig::residency` carries the whole argument.
        residency,
        -1,
        frames_in_flight,
    )?;
    backend.load(request).map_err(anyhow::Error::from)
}

/// **Write the operator's adapters into the banks the load just reserved**
/// (alto design §8, decision 17; survey §2 debt 6's "doors on both sides with
/// nothing between").
///
/// `runtime::engine::verbs::register_adapter` has existed since the palo
/// rewrite with no caller anywhere: the bytes had a door, a lane had a field,
/// and nothing in the tree walked through either. This is the first caller,
/// and boot is the right instant for it — registration is a control-plane
/// RESIDENCY verb, once per adapter and never again
/// (`engine::adapter`'s own argument for why it is not a `transfer`), so
/// it belongs beside the load rather than on any request path.
///
/// It runs HERE, between `land` and the engine's handover to the scheduler,
/// because that is the last instant the worker holds the `EngineBox` itself.
///
/// A plane file is one bank's slot, verbatim: the padding is the caller's
/// (`engine::adapter` argues why a shell cannot do it), and a file whose
/// length is not the slot's is refused by the engine with both numbers in it.
///
/// # Errors
///
/// A plane file that will not read, or whatever the engine refused — an
/// unknown bank, an id past the capacity the plan seats, a plane that is not
/// one slot's bytes, or `Unsupported` from a shell whose loads seat no bank.
fn register_operator_adapters(
    backend: &mut ::runtime::engine::EngineBox,
    adapters: &crate::config::AdapterConfig,
) -> Result<()> {
    for adapter in &adapters.registered {
        let mut planes = Vec::with_capacity(adapter.planes.len());
        for (bank, path) in &adapter.planes {
            let bytes = std::fs::read(path).with_context(|| {
                format!(
                    "read the plane for bank {bank:?} of adapter {} from {path:?}",
                    adapter.id
                )
            })?;
            planes.push(engine::AdapterPlane {
                bank: bank.clone(),
                bytes,
            });
        }
        let registration = engine::AdapterRegistration {
            id: adapter.id,
            planes,
        };
        ::runtime::engine::verbs::register_adapter(backend, &registration).with_context(|| {
            format!("registering adapter {} into this model's banks", adapter.id)
        })?;
        tracing::info!(
            id = adapter.id,
            planes = adapter.planes.len(),
            "registered an operator-declared adapter into this model's banks"
        );
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Native engine creation helpers.
// -----------------------------------------------------------------------------
//
// `write_cuda_startup_toml`, `write_metal_startup_toml` and the per-launch
// state directory they wrote into STOOD HERE, ~350 lines between them. Of the
// ~30 keys the cuda writer emitted, the shell read four subjects — the
// device, the two cache directories, and the `[engine]` knobs — and they are
// `DeviceBoot` fields now; the metal shell read exactly one key, and it
// crosses as in-memory bytes at the call site. Everything else on that wire
// was written and read by nothing.

/// What an engine may be pointed at: a `.zt` artifact, or a snapshot directory.
///
/// The GGUF refusal that used to live here is gone with the reason for it.
/// It existed because the LoadPlan executors could not decode GGUF's blocked
/// schemes at load time — but `pie model import` decodes them now,
/// so what reaches an engine is a `.zt` either way and there is no format left
/// to refuse. A `.gguf` handed straight to `serve` still fails, one step later
/// and with a better message: convert it first.
fn validate_snapshot_dir(snapshot_dir: &Path) -> Result<()> {
    if snapshot_dir.is_dir()
        || (snapshot_dir.is_file() && crate::weights::is_artifact_path(snapshot_dir))
    {
        return Ok(());
    }
    Err(anyhow!(
        "model {snapshot_dir:?} is neither a .zt artifact nor a snapshot directory; \
         `pie model import` writes the former"
    ))
}

#[cfg(feature = "_engine-cuda")]
pub(crate) fn create_engine_backend_group(
    rank_options: &[EngineOptions],
    snapshot_dir: &Path,
    weight_cache_dir: &Path,
    cache_dir: &Path,
    // The shared-adapter mount, or `None` for the feature off (alto adapter §3.3).
    adapter_dir: Option<&Path>,
    group_id: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    adapters: &crate::config::AdapterConfig,
    residency: engine::Residency,
    // **THE SECOND ROW AXIS'S CEILINGS, OR BOTH `None` TO DERIVE THEM** (alto
    // multimodal §5.5) — `[model] max_patches` / `[model] max_images`, threaded
    // as VALUES beside `residency` for W-3's reason: this layer holds a boot
    // struct and an adapter roster, not the worker's `Config`.
    patch_ceilings: (Option<u32>, Option<u32>),
    // `[model] sku`, or `None` to identify one — see `land`.
    sku: Option<&str>,
) -> Result<crate::translate::GroupEngine> {
    validate_snapshot_dir(snapshot_dir)?;
    if rank_options.is_empty() {
        return Err(anyhow!("cuda group requires at least one rank"));
    }

    let mut boots = Vec::with_capacity(rank_options.len());
    for (rank, rank_options) in rank_options.iter().enumerate() {
        // THE `else` IS UNREACHABLE IN ONE BUILD AND LOAD-BEARING IN EVERY
        // OTHER. `EngineOptions`' variants are feature-gated, so a binary
        // built with CUDA and nothing else has a one-variant enum and the
        // pattern is irrefutable; add `engine-metal` or `engine-vulkan` and
        // the refusal below is the only thing standing between a metal
        // option set and `device_boot`. Allowed rather than rewritten,
        // because the rewrite is to delete a check that a different feature
        // list needs.
        #[allow(
            irrefutable_let_patterns,
            reason = "`EngineOptions` has one variant in a CUDA-only build"
        )]
        let EngineOptions::CudaNative(opts) = rank_options else {
            return Err(anyhow!(
                "cuda group creation requires cuda-native rank options"
            ));
        };
        if opts.mtp_assistant_snapshot_dir.is_some() {
            return Err(anyhow!(
                "mtp_assistant_snapshot_dir is not supported by the single-model \
                 LoadPlan boot contract"
            ));
        }
        // One boot per rank, each naming its own device; the struct crosses
        // directly, so a field cannot silently fail to arrive.
        let boot = device_boot(opts, weight_cache_dir, cache_dir, adapter_dir)?;
        if opts.verbose {
            dump_device_boot(&boot, group_id, Some(rank));
        }
        boots.push(boot);
    }

    let ranks = boots.len();
    let (mut backend, opened) = ::runtime::engine::backend::open::cuda_group(boots)?;
    if opened != ranks {
        return Err(anyhow!(
            "cuda group opened {opened} ranks for {ranks} rank configs"
        ));
    }
    // ONE LOAD, NOT ONE PER RANK. `load_model` took a `Vec<ModelLoadDesc>`,
    // one descriptor per rank, and cross-checked that they agreed about the
    // model; a rank is not a load (`LoadRequest` is one plan, `Shard::Cut` is
    // in the plan), and `open::cuda_group` refuses a multi-rank launch by
    // name until `palo B-tp` builds one.
    #[allow(
        irrefutable_let_patterns,
        reason = "`EngineOptions` has one variant in a CUDA-only build"
    )]
    let EngineOptions::CudaNative(opts) = &rank_options[0] else {
        unreachable!("validated cuda options above");
    };
    let loaded = land(
        &mut backend,
        snapshot_dir,
        cuda_budgets(opts, adapters.seats(), patch_ceilings),
        residency,
        model_ir::Platform::Cuda,
        component,
        frames_in_flight,
        sku,
    )?;
    // THE BANKS ARE RESERVED BY THE LOAD; THIS IS THE WRITE. Between the
    // load and the handover is the last instant this layer holds the engine.
    register_operator_adapters(&mut backend, adapters)?;

    Ok(crate::translate::GroupEngine {
        caps: loaded.caps,
        facts: loaded.facts,
        snapshot_dir: snapshot_dir.to_path_buf(),
        backend,
    })
}

#[cfg_attr(
    not(feature = "_engine-cuda"),
    allow(
        unused_variables,
        unreachable_code,
        reason = "with no `engine-*` feature `EngineOptions` is uninhabited, so \
                  every path that takes one diverges"
    )
)]
pub(crate) fn create_engine_backend(
    options: &EngineOptions,
    snapshot_dir: &Path,
    weight_cache_dir: &Path,
    cache_dir: &Path,
    // The shared-adapter mount, or `None` for the feature off (alto adapter §3.3).
    adapter_dir: Option<&Path>,
    group_id: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    adapters: &crate::config::AdapterConfig,
    residency: engine::Residency,
    // **THE SECOND ROW AXIS'S CEILINGS, OR BOTH `None` TO DERIVE THEM** (alto
    // multimodal §5.5) — `[model] max_patches` / `[model] max_images`, threaded
    // as VALUES beside `residency` for W-3's reason: this layer holds a boot
    // struct and an adapter roster, not the worker's `Config`.
    patch_ceilings: (Option<u32>, Option<u32>),
    // `[model] sku`, or `None` to identify one — see `land`.
    sku: Option<&str>,
) -> Result<crate::translate::GroupEngine> {
    // Each is used only inside a `#[cfg(feature = "engine-…")]` arm below.
    let _ = (group_id, weight_cache_dir, cache_dir, adapter_dir);
    validate_snapshot_dir(snapshot_dir)?;

    // TYPED, because with no `engine-*` feature `EngineOptions` has no
    // variants at all and this `match` diverges — inference has nothing to
    // work from. That build reaches no device, which is the truth since the
    // interpreter backend was deleted: there is no ungated flavor left to
    // fall back to.
    let (mut backend, budgets, platform): (
        ::runtime::engine::EngineBox,
        engine::Budgets,
        model_ir::Platform,
    ) = match options {
        #[cfg(not(any(
            feature = "_engine-cuda",
            all(feature = "engine-metal", target_vendor = "apple")
        )))]
        _ => unreachable!("`EngineOptions` has no variants in this build"),
        #[cfg(feature = "_engine-cuda")]
        EngineOptions::CudaNative(opts) => {
            if opts.mtp_assistant_snapshot_dir.is_some() {
                return Err(anyhow!(
                    "mtp_assistant_snapshot_dir is not supported by the single-model \
                     LoadPlan boot contract"
                ));
            }
            // THE STRUCT, not a document describing it. The TOML this
            // replaced was written, re-read and parsed on the far side,
            // and a key nobody read fell back to a default IN SILENCE;
            // a typed field cannot fail to arrive.
            let boot = device_boot(opts, weight_cache_dir, cache_dir, adapter_dir)?;
            if opts.verbose {
                dump_device_boot(&boot, group_id, None);
            }
            let backend = ::runtime::engine::backend::open::cuda(boot)?;
            (
                backend,
                cuda_budgets(opts, adapters.seats(), patch_ceilings),
                model_ir::Platform::Cuda,
            )
        }
        // METAL, BACK AT P5. The document it hands over is built in
        // memory: of the ~14 keys the old bootstrap file carried, the
        // shell reads exactly ONE — `[metal] gpu_mem_utilization` — and
        // `[metal.tuning]` is the shell's own file, never this layer's to
        // write. No file, no launch-state directory, one parser.
        //
        // THE VULKAN AND WGPU ARMS STOOD HERE TOO and are still out:
        // neither engine has the baker executor R3 named as the condition
        // of its return.
        #[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
        EngineOptions::Metal(opts) => {
            // `{:?}` so the float keeps its decimal point: `1.0` written
            // `{}` is `1`, which TOML reads as an integer and the shell's
            // float reader would pass over.
            let mut boot_doc = format!(
                "[metal]\ngpu_mem_utilization = {:?}\n",
                opts.gpu_mem_utilization
            );
            // The one `[model]` key the metal door reads (alto adapter
            // §3.3); quoted through `toml::Value` so an unusual path
            // cannot break the document.
            if let Some(mount) = adapter_dir {
                boot_doc.push_str(&format!(
                    "\n[model]\nadapter_dir = {}\n",
                    toml::Value::String(mount.display().to_string())
                ));
            }
            let backend = ::runtime::engine::backend::open::metal(boot_doc.as_bytes())?;
            let page_size = opts.kv_page_size.max(1);
            let max_context = opts
                .max_model_len
                .unwrap_or_else(|| engine::Budgets::default().max_context);
            // **`total_pages` IS A PAGE COUNT AND `slots` IS A SEAT
            // COUNT**, and the budget below handed the first over as the
            // second until this line existed. `Paging::of` gives every
            // seat a block of `max_context / page_size` pages and reserves
            // the product (`model_exec::store::kv::Paging::pages`), so
            // 1024 PAGES read as 1024 SEATS reserved 131072 pages — a
            // hundred and twenty-eight times the pool the operator asked
            // for. On qwen35-d0.8b that is 8 GiB of KV per attention
            // layer, and what it buys is a first command buffer that
            // either fails with `kIOGPUCommandBufferCallbackError
            // OutOfMemory` or hangs the device outright.
            //
            // The division is `cuda_budgets`' own, and that function's
            // header already says why it is the only honest one: "the
            // shell's paging hands each seated sequence one block of
            // `max_context / page_size` pages: how many sequences fit is
            // that division, not a third knob to keep in step". Both arms
            // read the operator's page cap as a page cap now.
            let pages_per_slot = max_context.div_ceil(page_size).max(1);
            (
                backend,
                engine::Budgets {
                    max_lanes: opts.max_forward_requests.max(1),
                    max_tokens: opts.max_forward_tokens.max(1),
                    buckets: Vec::new(),
                    // The same one number the CUDA arm takes. `[model]`
                    // rather than `[model.engine.options]` is what makes
                    // it portable: debt 6's complaint was that Metal had
                    // no configuration at all, and an adapter seat count
                    // is not a fact about a backend.
                    max_adapters: adapters.seats(),
                    page_size,
                    max_context,
                    slots: (opts.total_pages / pages_per_slot).max(1),
                    // The metal mirror binds no patch seat and refuses
                    // every patch-axis input by name, so a ladder here
                    // would be a ceiling on a rectangle this plane cannot
                    // resolve. `None` is the honest answer, not a default.
                    max_patches: None,
                    max_images: None,
                },
                model_ir::Platform::Metal,
            )
        }
    };
    // Uniform across backends now that the load is a request rather than a
    // compiled plan (§10.3). Unreachable in a build with no `engine-*`
    // feature, where the match above diverges on an empty enum.
    #[cfg_attr(
        not(feature = "_engine-cuda"),
        allow(
            unreachable_code,
            reason = "`EngineOptions` has no variants in this build"
        )
    )]
    let loaded = land(
        &mut backend,
        snapshot_dir,
        budgets,
        residency,
        platform,
        component,
        frames_in_flight,
        sku,
    )?;

    // THE BANKS ARE RESERVED BY THE LOAD; THIS IS THE WRITE. See
    // `register_operator_adapters` — this is `verbs::register_adapter`'s
    // first caller in the tree.
    register_operator_adapters(&mut backend, adapters)?;

    Ok(crate::translate::GroupEngine {
        caps: loaded.caps,
        facts: loaded.facts,
        snapshot_dir: snapshot_dir.to_path_buf(),
        backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool arithmetic an operator's two knobs come out as.
    ///
    /// `slots` is the one number nobody states: a page cap and a context
    /// length determine it, and stating it separately would be a third knob
    /// to keep in step with the other two.
    #[test]
    fn the_pool_budget_derives_its_seat_count_from_the_page_cap() {
        let mut opts = CudaNativeEngineOptions {
            kv_page_size: Some(16),
            max_total_pages: Some(1024),
            ..Default::default()
        };
        let budgets = cuda_budgets(&opts, 0, (None, None));
        assert_eq!(budgets.page_size, 16);
        // 4096 tokens of context is 256 pages a slot; 1024 pages seats four.
        assert_eq!(budgets.max_context, 4096);
        assert_eq!(budgets.slots, 4);

        // No cap stated: the contract's own default seat count, not a
        // division by nothing.
        opts.max_total_pages = None;
        assert_eq!(cuda_budgets(&opts, 0, (None, None)).slots, 256);

        // A cap smaller than one slot's block still seats one — a pool that
        // seats nothing is a load that cannot fire.
        opts.max_total_pages = Some(1);
        assert_eq!(cuda_budgets(&opts, 0, (None, None)).slots, 1);
    }

    /// **THE OPERATOR'S SEAT COUNT REACHES THE BUDGET**, which is the half of
    /// design §8 that was a hard-coded zero (survey §2 debt 6: doors on both
    /// sides with nothing between).
    ///
    /// `max_adapters` is an INTENT, not a pool size: the banks are reserved
    /// at load whatever this says, and `model_compiler::compile` refuses a
    /// load whose intent is bigger than what the model text seats. So the
    /// only thing this layer owes is that the number an operator wrote is
    /// the number that crosses.
    #[test]
    fn the_adapter_seat_count_crosses_into_the_budget() {
        let opts = CudaNativeEngineOptions::default();
        assert_eq!(
            cuda_budgets(&opts, 0, (None, None)).max_adapters,
            0,
            "a deployment that states no adapters still registers none"
        );
        assert_eq!(cuda_budgets(&opts, 8, (None, None)).max_adapters, 8);
    }

    // `caps_json_round_trips` STOOD HERE. It deserialized a
    // `EngineCapabilities` from a JSON document with an `abi_version` in it
    // and asserted the round trip — a test about a 30-field flat struct with
    // `#[serde(default)]` on two thirds of it, four of whose fields
    // (`abi_version`, `arch_name`, `snapshot_dir`, and the flat
    // `max_forward_*`) have no successor at all. `Capabilities` is four typed
    // records and `serde`'s derive is what round-trips it; there is nothing
    // left here that a hand-written document would check.

    // `gemma4_encode_component_loads_and_encodes` STOOD HERE, `#[ignore]`d
    // and by its own header never once run. It was written against four
    // things the palo contract rewrite deleted outright — `ModelComponent`
    // on a load request, `MediaEncodePlan` with a completion to await, the
    // executor server it stood one up through, and `KvDtype` — and the thing
    // it was waiting for (component-scoped loading) is now a different
    // question entirely: an encoder is a traced `Trace`, and the catalog ships
    // none. `embedded_engine::land` refuses a non-`Full` component by name,
    // which is the statement this test was keeping alive.

    /// What an engine may be handed: an artifact, or a snapshot directory.
    ///
    /// This used to pin a GGUF-specific refusal, which existed because the
    /// LoadPlan executors could not decode GGUF's blocked schemes at load
    /// time. `pie model import` decodes them now, so a served model
    /// is a `.zt` whatever it started as, and the refusal has nothing left to
    /// name. A `.gguf` handed straight to `serve` is still rejected — as one
    /// of the things that is not an artifact, with the fix in the message.
    #[test]
    fn an_engine_takes_an_artifact_or_a_snapshot_and_nothing_else() {
        let tmp = tempfile::tempdir().unwrap();

        let artifact = tmp.path().join("model.zt");
        std::fs::write(&artifact, b"stand-in").unwrap();
        validate_snapshot_dir(&artifact).unwrap();

        let snapshot = tmp.path().join("snap");
        std::fs::create_dir(&snapshot).unwrap();
        validate_snapshot_dir(&snapshot).unwrap();

        let gguf = tmp.path().join("model.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();
        let error = validate_snapshot_dir(&gguf).unwrap_err().to_string();
        assert!(error.contains("pie model import"), "{error}");

        let error = validate_snapshot_dir(&tmp.path().join("nope"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("neither a .zt artifact"), "{error}");
    }

    // THE TEN TOML ROUND-TRIP TESTS STOOD HERE — cache root, knob omission,
    // schema match, the pid sweep, calibration, the `[distributed]` block and
    // the carried config. They pinned a wire that no longer exists; the facts
    // that survive the wire are about `device_boot` and are pinned below.
    //
    // The tests below need `DeviceBoot`, which exists only under
    // `_engine-cuda`, so they run only in a cuda-featured test build
    // (`cargo test -p worker --features engine-cuda-13`).

    /// **ONLY OPERATOR-STATED KNOBS DEVIATE FROM THE ENGINE'S DEFAULTS** —
    /// what `the_startup_toml_writes_only_the_engine_knobs_an_operator_stated`
    /// pinned about the old wire, restated about the mapping: an absent knob
    /// takes `Knobs::default()`, the shell's own answer, never one this layer
    /// invented. The fraction is the exception with no absence to express —
    /// the config type has already turned its absence into `0.90`.
    #[cfg(feature = "_engine-cuda")]
    #[test]
    fn only_stated_knobs_deviate_from_the_engines_defaults() {
        let boot = |opts: &CudaNativeEngineOptions| {
            device_boot(opts, Path::new("/w"), Path::new("/c"), None).unwrap()
        };

        let quiet = boot(&CudaNativeEngineOptions::default());
        let stock = Knobs::default();
        assert_eq!(quiet.knobs.gpu_mem_utilization, 0.90);
        assert_eq!(quiet.knobs.pad, stock.pad);
        assert_eq!(quiet.knobs.bodies, stock.bodies);
        assert_eq!(quiet.knobs.bodies_mem, stock.bodies_mem);
        assert_eq!(quiet.knobs.copies, stock.copies);
        assert_eq!(quiet.knobs.grouped, stock.grouped);
        assert_eq!(quiet.knobs.side_streams, stock.side_streams);

        let stated = boot(&CudaNativeEngineOptions {
            pad: Some(false),
            bodies: Some(false),
            bodies_mem: Some(256),
            fallback_copy: Some(false),
            grouped: Some(false),
            side_streams: Some(0),
            ..Default::default()
        });
        assert!(!stated.knobs.pad);
        assert!(!stated.knobs.bodies);
        assert_eq!(stated.knobs.bodies_mem, 256);
        assert!(!stated.knobs.copies);
        assert!(!stated.knobs.grouped);
        assert_eq!(stated.knobs.side_streams, Some(0));
    }

    /// The graphs word is the shell's to parse, and a spelling it does not
    /// speak refuses by the key's name — the config layer never polices it,
    /// so this is the one refusal on that word.
    #[cfg(feature = "_engine-cuda")]
    #[test]
    fn the_graphs_word_parses_or_refuses_by_the_keys_name() {
        let boot = |graphs: Option<&str>| {
            device_boot(
                &CudaNativeEngineOptions {
                    graphs: graphs.map(String::from),
                    ..Default::default()
                },
                Path::new("/w"),
                Path::new("/c"),
                None,
            )
        };
        assert!(
            matches!(boot(None).unwrap().graphs, Graphs::On),
            "absent takes the shell's own default, which is the served path"
        );
        assert!(matches!(boot(Some("off")).unwrap().graphs, Graphs::Off));
        assert!(matches!(
            boot(Some("shaped")).unwrap().graphs,
            Graphs::Shaped
        ));
        assert!(matches!(boot(Some("on")).unwrap().graphs, Graphs::On));
        let error = boot(Some("sideways")).unwrap_err().to_string();
        assert!(error.contains("graphs"), "{error}");
    }

    /// The device string, both cache directories and the shared-adapter
    /// mount arrive — the four deployment facts (with the knobs) that were
    /// ever actually read off the old wire. The mount half is what
    /// `[model] adapter_dir`'s writer commit exists for: a key nobody
    /// emitted was a key nobody could set, and a field cannot go unemitted.
    #[cfg(feature = "_engine-cuda")]
    #[test]
    fn the_device_the_dirs_and_the_mount_cross_into_the_boot() {
        let boot = device_boot(
            &CudaNativeEngineOptions {
                device: "cuda:1".to_string(),
                ..Default::default()
            },
            Path::new("/pie/cache/weights"),
            Path::new("/pie/cache"),
            Some(Path::new("/srv/adapters")),
        )
        .unwrap();
        assert_eq!(boot.ordinal, 1);
        assert_eq!(
            boot.weight_cache_dir.as_deref(),
            Some(Path::new("/pie/cache/weights"))
        );
        assert_eq!(boot.cache_dir.as_deref(), Some(Path::new("/pie/cache")));
        assert_eq!(
            boot.adapter_dir.as_deref(),
            Some(Path::new("/srv/adapters"))
        );

        let unstated = device_boot(
            &CudaNativeEngineOptions::default(),
            Path::new("/w"),
            Path::new("/c"),
            None,
        )
        .unwrap();
        assert_eq!(
            unstated.adapter_dir, None,
            "no mount stated is the feature off"
        );
    }
}
