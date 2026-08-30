//! Engine-backend bootstrap helpers for pie-worker.
//!
//! This module exposes:
//!   * [`EngineCapabilities`] — typed engine capability payloads.
//!   * [`write_cuda_startup_toml`] — emits the per-launch TOML the one hosted
//!     engine reads at creation. Its Metal, Vulkan and wgpu counterparts stood
//!     beside it and left with the engines that read them.
//!   * [`create_engine_backend`] — build a runtime-owned [`::runtime::engine::EngineBox`]
//!     plus its caps before `::runtime::bootstrap`.

#[cfg(feature = "_engine-cuda")]
use std::ffi::CStr;
#[cfg(feature = "_engine-cuda")]
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
// Every `with_context` in this file is inside cuda-gated code: the one
// bootstrap-TOML writer left, and the two seams that read back what it wrote.
#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
use anyhow::Context;

#[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
use crate::config::MetalEngineOptions;
#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
#[cfg(any(feature = "_engine-cuda", test))]
use crate::config::{CudaMemoryProfile, CudaNativeEngineOptions};
use crate::engine_ffi::Flavor;

// THE TWO LINK ANCHORS ARE GONE WITH THE C++ THEY SERVED.
//
// `PIE_LOADER_ENTRY_ANCHOR` and `PIE_FORWARD_ENTRY_ANCHOR` existed because a
// linker never pulls an rlib member in on behalf of a C++ reference: the only
// callers of `pie_loader_compile_model` and `pie_forward_trace_llama_like`
// were the C++ engines, which link after Rust, so without a reference from
// reachable Rust the entry points were simply absent at final link.
//
// Both engines are Rust now. `model-loader` and `model` are called directly,
// through their own types, and there is nothing on the far side of an FFI
// boundary to keep alive.

#[cfg(feature = "_engine-cuda")]
#[repr(C)]
struct NcclUniqueId {
    internal: [u8; 128],
}

#[cfg(feature = "_engine-cuda")]
unsafe extern "C" {
    fn ncclGetUniqueId(unique_id: *mut NcclUniqueId) -> c_int;
    fn ncclGetErrorString(result: c_int) -> *const c_char;
}

#[cfg(feature = "_engine-cuda")]
fn nccl_unique_id_hex() -> Result<String> {
    let mut id = NcclUniqueId { internal: [0; 128] };
    let rc = unsafe { ncclGetUniqueId(&mut id as *mut NcclUniqueId) };
    if rc != 0 {
        let msg = unsafe { CStr::from_ptr(ncclGetErrorString(rc)) }
            .to_string_lossy()
            .into_owned();
        return Err(anyhow!("ncclGetUniqueId: {msg}"));
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(id.internal.len() * 2);
    for b in id.internal {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    Ok(out)
}

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

/// Read only by the startup-TOML writers and the state-dir path, which are
/// linked only when a real engine is. With no engine feature the descriptor is
/// still THREADED (`create_engine_backend` takes `Option<&TpLaunch>` in every
/// build) but never inspected -- so the allow is scoped to exactly that build,
/// and a field that dies under an engine build is still caught.
///
#[cfg_attr(
    not(any(feature = "_engine-cuda", test)),
    allow(dead_code, reason = "read by the cfg-gated TOML writers")
)]
#[derive(Clone)]
pub(crate) struct TpLaunch {
    size: usize,
    rank: usize,
    nccl_unique_id_hex: String,
}

#[cfg(feature = "_engine-cuda")]
pub(crate) fn tp_launches(size: usize) -> Result<Vec<TpLaunch>> {
    let nccl_unique_id_hex = nccl_unique_id_hex()?;
    Ok((0..size)
        .map(|rank| TpLaunch {
            size,
            rank,
            nccl_unique_id_hex: nccl_unique_id_hex.clone(),
        })
        .collect())
}

/// This model's materialized-weight artifact directory, installed once before
/// any engine is created and written into every bootstrap TOML from there.
///
/// Install-at-bootstrap rather than a parameter because the TOML writers sit
/// five call layers below the only place holding a parsed `Config`. First
/// writer wins, so a directory a live engine is already using cannot move.
static WEIGHT_CACHE_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Install the resolved weight-artifact directory. The caller resolves the
/// `$PIE_HOME/models` default, because `$PIE_HOME` is the bin/worker layer's
/// to know and the engine has never been told it.
pub fn set_weight_cache_dir(dir: String) {
    let _ = WEIGHT_CACHE_DIR.set(dir);
}

/// Read back by `write_cuda_startup_toml`, which is the only thing that
/// puts this on the wire -- so the reader is gated exactly as the writer is.
#[cfg(any(feature = "_engine-cuda", test))]
fn weight_cache_dir() -> String {
    WEIGHT_CACHE_DIR.get().cloned().unwrap_or_default()
}

/// The root every engine-side disk cache derives from: `$PIE_HOME/cache`.
///
/// Location is convention, not configuration -- there is no config field for
/// it, and `$PIE_HOME` is the one lever that moves it. Before this the engine
/// caches derived from `$XDG_CACHE_HOME`/`$HOME/.cache` instead, not as a
/// choice but because the engine had never been told `$PIE_HOME`. That split
/// pie's state across two roots: `pie serve` wrote programs, logs and
/// optimized checkpoints under one and compiled PTIR, GEMM tuning and planner
/// profiles under another.
static CACHE_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Install the resolved cache root, before any engine is created. First writer
/// wins, so a cache a live engine is already using cannot move.
pub fn set_cache_dir(dir: String) {
    let _ = CACHE_DIR.set(dir);
}

/// Gated as `write_cuda_startup_toml` is, like `weight_cache_dir` above and
/// for the same reason: emitting a bootstrap TOML is all this and the nine
/// helpers under it are for, and the one seam still emitting one is cuda's.
#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn cache_dir() -> String {
    CACHE_DIR.get().cloned().unwrap_or_default()
}

/// Emit `[cache] dir` into an engine's bootstrap TOML.
///
/// Omitted when unset so an engine launched with a hand-written TOML (its own
/// `dev.toml`, say) keeps the XDG derivation rather than losing its cache to
/// an empty path.
#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn insert_cache_table(doc: &mut toml::Table) {
    let dir = cache_dir();
    if dir.is_empty() {
        return;
    }
    let mut table = toml::Table::new();
    insert_str(&mut table, "dir", dir);
    insert_table(doc, "cache", table);
}

#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn insert_int(table: &mut toml::Table, key: &str, value: impl Into<i64>) {
    table.insert(key.into(), toml::Value::Integer(value.into()));
}

#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn insert_str(table: &mut toml::Table, key: &str, value: impl Into<String>) {
    table.insert(key.into(), toml::Value::String(value.into()));
}

#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn insert_bool(table: &mut toml::Table, key: &str, value: bool) {
    table.insert(key.into(), toml::Value::Boolean(value));
}

#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn insert_table(doc: &mut toml::Table, key: &str, table: toml::Table) {
    doc.insert(key.into(), toml::Value::Table(table));
}

#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn path_string(path: &Path) -> String {
    path.display().to_string()
}

/// Writes the checkpoint's config beside the bootstrap TOML and names it in
/// `[model]`.
///
/// Beside rather than inlined: the engine already takes a path, and opening a
/// second one is less machinery than teaching TOML to carry a JSON document.
///
/// Unconditional. It was optional while a snapshot reached the engine without
/// one and each engine parsed `config.json` itself; `weights.rs` lifts that
/// case now, so there is one lifter and every boot writes this file. The type
/// says so, which is what keeps the deleted branch from growing back.
///
/// Named `config` rather than `descriptor` because that is what it is. The
/// old name meant a `pie.model/1` document — ~40 resolved fields, a schema, a
/// reader in each engine — and that document is deleted. What travels here is
/// the checkpoint's own `config.json`, verbatim, read for exactly one field.
#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn write_config_beside(out_path: &Path, config: &[u8], model: &mut toml::Table) -> Result<()> {
    let beside = out_path.with_file_name("model.config.json");
    std::fs::write(&beside, config).with_context(|| format!("write model config {beside:?}"))?;
    insert_str(model, "config", path_string(&beside));
    Ok(())
}

/// Name the model in `[model] id`, when the operator named one.
///
/// # What crosses the boundary
///
/// A string, and a config path read for one field. The `pie.model/1`
/// document that used to travel here is gone: the worker wrote a JSON
/// blob of ~40 resolved fields, named its path here, and each engine
/// parsed it back — `engine-cuda` through `model::descriptor` into an
/// `HfConfig`, `engine-metal` through its OWN reader into its OWN
/// `ModelFacts`, with its own defaulting rules. Two readers of one
/// document, under two failure policies: the facts reader swallowed a
/// missing field with a default, the descriptor reader refused. So the
/// two sides could hold different beliefs about one checkpoint and
/// neither would say anything.
///
/// An id cannot do that, because both engines link the same `const`
/// table. A wrong id fails to resolve — at the door, with the nearest
/// ids named — and a right one reaches a row that answers every question
/// the same way on both sides, because it is the same row.
///
/// What still travels beside it is the checkpoint's own `config.json`,
/// verbatim and unresolved, and an engine reads ONE field out of it —
/// the declared quantization, which is the thing no row can state
/// because the same model is published at four bits and at eight. It is
/// not a second answer to "what is this model"; it is the answer to
/// "how was this copy of it encoded", and the two cannot be confused
/// because only one of them is a row.
///
/// # Why it is optional
///
/// Because the checkpoint can answer for itself. Absent an id, an engine
/// matches the TENSORS against the catalog, which is the answer that
/// does not depend on anyone having written anything down. The id is an
/// OVERRIDE, for the case where a checkpoint is genuinely a known model
/// under an unknown name — a fine-tune, a re-upload, a mirror that
/// renamed the directory — and it does not skip the manifest check.
#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn insert_model_id(model: &mut toml::Table, id: Option<&str>) {
    if let Some(id) = id.filter(|s| !s.is_empty()) {
        insert_str(model, "id", id);
    }
}

#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple"),
    test
))]
fn write_toml_table(out_path: &Path, doc: toml::Table) -> Result<()> {
    let serialized = toml::to_string(&doc).map_err(|e| anyhow!("serialize bootstrap TOML: {e}"))?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create bootstrap toml dir {parent:?}: {e}"))?;
    }
    std::fs::write(out_path, serialized)
        .map_err(|e| anyhow!("write bootstrap toml {out_path:?}: {e}"))?;
    Ok(())
}

/// Default per-launch state directory: `$PIE_HOME/standalone/<pid>/`.
/// We use a per-pid subdir so concurrent invocations of `pie` (rare
/// but legal — different ports) don't clobber each other's TOML or
/// aux sockets.
pub fn launch_state_dir() -> PathBuf {
    launch_state_root().join(std::process::id().to_string())
}

/// Root of the per-launch state directories. Public so `state::entries` names
/// the same path the sweep walks -- a listing that pointed elsewhere would
/// report nothing and reclaim nothing.
pub fn launch_state_root() -> PathBuf {
    crate::paths::pie_home().join("standalone")
}

/// Whether a process id is still running.
///
/// `kill(pid, 0)` delivers no signal and only reports reachability: `Ok` means
/// alive, `EPERM` means alive but not ours, `ESRCH` means gone. Anything other
/// than a definite `ESRCH` is treated as alive, because the cost of the two
/// mistakes is not symmetric — a stale directory is a few bytes, deleting a
/// live launch's bootstrap TOML is an engine that cannot boot.
#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn pid_is_alive(_pid: u32) -> bool {
    true
}

/// Remove `$PIE_HOME/standalone/<pid>` directories whose process is gone.
///
/// Each launch writes an engine bootstrap TOML under its own pid and nothing ever
/// removed it, so every `pie serve` left a directory behind for the life of
/// the machine. Sweeping at boot rather than only at shutdown is what makes it
/// bounded: the leak's whole population is launches that did NOT exit cleanly.
///
/// Best-effort throughout. A directory that cannot be read or removed is left
/// alone: this runs on the boot path and must never be the reason a start
/// fails.
pub fn sweep_stale_launch_state() {
    let root = launch_state_root();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return;
    };
    let self_pid = std::process::id();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|n| n.parse::<u32>().ok()) else {
            // Not a pid directory — not ours to reason about.
            continue;
        };
        if pid == self_pid || pid_is_alive(pid) {
            continue;
        }
        let path = entry.path();
        if let Err(error) = std::fs::remove_dir_all(&path) {
            tracing::debug!(?path, %error, "could not sweep stale launch state");
        }
    }
}

/// Remove this process's launch state directory. Called on clean shutdown; the
/// boot sweep is what covers the unclean ones.
pub fn remove_launch_state() {
    let dir = launch_state_dir();
    if let Err(error) = std::fs::remove_dir_all(&dir)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::debug!(?dir, %error, "could not remove launch state");
    }
}

/// What a load answered about itself.
///
/// A rename, not an alias with a new spelling behind it: the 30-field
/// `EngineCapabilities` mixed three subjects (the device, the load, and the
/// MODEL) and the contract's [`Capabilities`](engine_api::Capabilities)
/// separates them — `device`, `pools`, `limits`, and a
/// `ModelProfile` carried whole rather than rebuilt from eight booleans
/// (`engine-api::caps`'s header). Three of the old fields have no successor
/// because they were never the engine's to answer: `snapshot_dir`,
/// `model_id` and `arch_name` say where the CALLER's checkpoint came from,
/// and the caller is this crate. They are on
/// [`GroupEngine`](crate::translate::GroupEngine) now.
pub use engine_api::Capabilities as EngineCapabilities;

/// The ceilings a load is baked against, out of what the operator stated.
///
/// **THIS IS WHERE `ModelLoadDesc` WENT.** Its four fields said nothing this
/// does not: `snapshot_dir` is [`Checkpoint::Path`](engine_api::Checkpoint),
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
fn cuda_budgets(opts: &CudaNativeEngineOptions, adapter_seats: u32) -> engine_api::Budgets {
    let page_size = opts.kv_page_size.unwrap_or(16).max(1);
    // No CUDA knob states a context ceiling — `max_model_len` is the Metal
    // options' — so this is the contract's own default, stated once here
    // rather than guessed twice.
    let max_context = engine_api::Budgets::default().max_context;
    let pages_per_slot = max_context.div_ceil(page_size).max(1);
    engine_api::Budgets {
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
    }
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
    budgets: engine_api::Budgets,
    platform: engine_api::model_ir::Platform,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
) -> Result<engine_api::Loaded> {
    if component != crate::executor::ModelComponent::Full {
        // palo B-component: an encoder is a traced plan like any other, and
        // the catalog ships no encoder trace. Refused by name rather than
        // loaded as the full model, which is what the old
        // `ModelComponent::Encode` did — it staged the whole 48 GiB
        // checkpoint and died in `cudaMalloc`.
        return Err(anyhow!(
            "this build loads only the full model; {component:?} needs a traced              plan the catalog does not ship"
        ));
    }
    // **THE RUN-AHEAD DEPTH, CROSSING ONCE** (article 8). `[runtime]
    // frame_dispatch_depth` is the deployment's statement of how many frames
    // it will keep posted to the engine; the engine's staging ring and
    // settlement event pool are sized from it and from nothing else.
    let request = ::runtime::engine::load::request(
        snapshot_dir,
        platform,
        budgets,
        // **UNCAPPED, WHICH IS WHAT THIS DEPLOYMENT HAS ALWAYS ASKED FOR**
        // (alto design §7). The residency policy is two budgets and `None`
        // is uncapped; no shell in this workspace streams weights, so the
        // only budget a worker could state that would change anything is one
        // that refuses the load. Stated by name rather than left to
        // `Default` so that the day `[model]` grows a weight budget, this is
        // the line it arrives at.
        engine_api::Residency::uncapped(),
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
/// (`engine_api::adapter`'s own argument for why it is not a `transfer`), so
/// it belongs beside the load rather than on any request path.
///
/// It runs HERE, between `land` and the engine's handover to the scheduler,
/// because that is the last instant the worker holds the `EngineBox` itself.
///
/// A plane file is one bank's slot, verbatim: the padding is the caller's
/// (`engine_api::adapter` argues why a shell cannot do it), and a file whose
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
            planes.push(::runtime::engine::AdapterPlane {
                bank: bank.clone(),
                bytes,
            });
        }
        let registration = ::runtime::engine::AdapterRegistration {
            id: adapter.id,
            planes,
        };
        ::runtime::engine::verbs::register_adapter(backend, &registration).with_context(|| {
            format!(
                "registering adapter {} into this model's banks",
                adapter.id
            )
        })?;
        tracing::info!(
            id = adapter.id,
            planes = adapter.planes.len(),
            "registered an operator-declared adapter into this model's banks"
        );
    }
    Ok(())
}

/// Write the cuda engine's bootstrap TOML. Schema mirrors
/// `crates/engine-cuda/csrc/src/config.hpp`: `[model]` with
/// `snapshot_dir`/`device`/`dtype` plus model-execution knobs,
/// `[batching]` with KV-page geometry plus `swap_pool_size`, and `[runtime]`
/// with the server verbosity flag.
///
/// `[distributed]` is emitted only for TP launches; single-rank uses the
/// cuda engine's default (`tp_size=1, tp_rank=0`).
// Gated with `test` as well as the feature ON PURPOSE. Emitting the startup
// TOML is pure string work -- it needs no CUDA, no nvcc and no GPU -- so its
// tests run on every host, which is the only reason they run at all here.
#[cfg(any(feature = "_engine-cuda", test))]
pub(crate) fn write_cuda_startup_toml(
    out_path: &Path,
    opts: &CudaNativeEngineOptions,
    weight_dtype: &str,
    snapshot_dir: &Path,
    _group_id: usize,
    tp: Option<&TpLaunch>,
    config: &[u8],
) -> Result<()> {
    let mut doc = toml::Table::new();

    let mut model = toml::Table::new();
    insert_str(&mut model, "snapshot_dir", path_string(snapshot_dir));
    insert_str(&mut model, "weight_cache_dir", weight_cache_dir());
    write_config_beside(out_path, config, &mut model)?;
    insert_model_id(&mut model, opts.model_id.as_deref());
    insert_str(&mut model, "device", &opts.device);
    // A `[model]` key, so it arrives beside the options rather than inside
    // them: what the checkpoint holds is a fact about the weights, and the
    // engine's own `[model]` table is where it lands.
    insert_str(&mut model, "dtype", weight_dtype);
    insert_int(&mut model, "mtp_num_drafts", opts.mtp_num_drafts);
    insert_bool(
        &mut model,
        "stream_routed_experts",
        opts.stream_routed_experts,
    );
    // Omitted when absent rather than written as a sentinel: the engine's
    // own default IS the derivation, so an absent key and a "0 means derive"
    // key would be two spellings of one thing.
    // The engine still speaks GiB floats; the unit lives in the config type,
    // not on the wire.
    if let Some(size) = opts.expert_cache {
        model.insert(
            "expert_cache_gb".into(),
            toml::Value::Float(size.as_gib_f64()),
        );
    }
    if let Some(size) = opts.expert_host_cache {
        model.insert(
            "expert_host_cache_gb".into(),
            toml::Value::Float(size.as_gib_f64()),
        );
    }
    insert_bool(
        &mut model,
        "enable_system_speculation",
        opts.enable_system_speculation,
    );
    insert_table(&mut doc, "model", model);

    let mut batching = toml::Table::new();
    batching.insert(
        "gpu_mem_utilization".into(),
        toml::Value::Float(opts.gpu_mem_utilization),
    );
    insert_str(
        &mut batching,
        "memory_profile",
        match opts.memory_profile {
            CudaMemoryProfile::Auto => "auto",
            CudaMemoryProfile::Latency => "latency",
            CudaMemoryProfile::Throughput => "throughput",
        },
    );
    if let Some(size) = opts.kv_page_size {
        insert_int(&mut batching, "kv_page_size", size);
    }
    insert_int(&mut batching, "swap_pool_size", opts.swap_pool_size);
    if let Some(pages) = opts.max_total_pages {
        insert_int(&mut batching, "total_pages", pages);
    }
    // Omitted when absent, like the other derived keys: the engine defaults
    // them to "let the planner choose", so writing a sentinel would be a
    // second spelling of an absent key.
    if let Some(tokens) = opts.max_forward_tokens {
        insert_int(&mut batching, "max_forward_tokens", tokens);
    }
    if let Some(requests) = opts.max_forward_requests {
        insert_int(&mut batching, "max_forward_requests", requests);
    }
    insert_str(&mut batching, "kv_cache_dtype", opts.kv_cache_dtype.clone());
    // Written only when asked for, like the derived keys above: the engine
    // defaults it to false too, so emitting `false` would be a second spelling
    // of an absent key. It also keeps the bootstrap TOML saying nothing about
    // calibration on every ordinary boot.
    if opts.calibrate_planner {
        insert_bool(&mut batching, "calibrate_planner", true);
    }
    insert_table(&mut doc, "batching", batching);

    let mut runtime = toml::Table::new();
    insert_bool(&mut runtime, "verbose", opts.verbose);
    insert_table(&mut doc, "runtime", runtime);

    // **`[engine]`: THE SHELL'S OWN WORDS** (alto article 9 — shells read no
    // environment). Nine `PIE_CUDA_*` variables were read inside the shell at
    // load and are typed `Boot` fields now; this is the table they arrive
    // through, and `runtime::engine::backend::cuda::knobs` is the one reader.
    //
    // EVERY KEY IS OMITTED WHEN THE OPERATOR STATED NOTHING, like the derived
    // keys above and for the same reason: the shell's own default IS the
    // answer for an absent key, so writing one would be a second spelling of
    // absence — and, here, a second place the default is written down.
    let mut engine = toml::Table::new();
    if let Some(graphs) = opts.graphs.as_deref() {
        insert_str(&mut engine, "graphs", graphs);
    }
    if let Some(pad) = opts.pad {
        insert_bool(&mut engine, "pad", pad);
    }
    if let Some(fold) = opts.fold {
        insert_bool(&mut engine, "fold", fold);
    }
    if let Some(pipeline) = opts.pipeline {
        insert_bool(&mut engine, "pipeline", pipeline);
    }
    if let Some(policy) = opts.fold_disable.as_deref() {
        insert_str(&mut engine, "fold_disable", policy);
    }
    if let Some(copies) = opts.fallback_copy {
        insert_bool(&mut engine, "fallback_copy", copies);
    }
    if let Some(grouped) = opts.grouped {
        insert_bool(&mut engine, "grouped", grouped);
    }
    if let Some(streams) = opts.side_streams {
        insert_int(&mut engine, "side_streams", i64::from(streams));
    }
    if !engine.is_empty() {
        insert_table(&mut doc, "engine", engine);
    }

    insert_cache_table(&mut doc);

    if let Some(tp) = tp {
        let mut distributed = toml::Table::new();
        insert_int(&mut distributed, "tp_size", tp.size as i64);
        insert_int(&mut distributed, "tp_rank", tp.rank as i64);
        insert_str(
            &mut distributed,
            "nccl_unique_id_hex",
            tp.nccl_unique_id_hex.clone(),
        );
        insert_table(&mut doc, "distributed", distributed);
    }

    write_toml_table(out_path, doc)
}

#[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
/// Emit the metal engine's bootstrap TOML — same `[model]` + `[batching]` +
/// `[runtime]` layout consumed by `crates/engine-metal/csrc/src/config.hpp`. The metal
/// launch state is identical apart from the `metal:N` backend selector.
///
/// Not gated on `engine-metal`: what it produces is a TOML file, and whether
/// the operator's settings survive into that file is a question a machine
/// without a Metal device can still answer. Gating it would put the test out
/// of reach of every machine that is not a Mac.
pub(crate) fn write_metal_startup_toml(
    out_path: &Path,
    options: &MetalEngineOptions,
    snapshot_dir: &Path,
    _group_id: usize,
    config: &[u8],
) -> Result<()> {
    let mut doc = toml::Table::new();

    let mut model = toml::Table::new();
    insert_str(&mut model, "hf_path", path_string(snapshot_dir));
    // Same arrangement as the CUDA engine.
    write_config_beside(out_path, config, &mut model)?;
    insert_model_id(&mut model, options.model_id.as_deref());
    insert_str(&mut model, "backend", &options.device);
    insert_bool(
        &mut model,
        "stream_routed_experts",
        options.stream_routed_experts,
    );
    // Omitted when unset rather than written as 0: the engine reads an absent
    // key as "the whole bank stays resident", which is the same statement.
    if let Some(bytes) = options.expert_slab_bytes {
        model.insert(
            "expert_slab_bytes".into(),
            toml::Value::Integer(bytes as i64),
        );
    }
    insert_table(&mut doc, "model", model);

    let mut batching = toml::Table::new();
    insert_int(&mut batching, "kv_page_size", options.kv_page_size);
    insert_int(&mut batching, "total_pages", options.total_pages);
    insert_int(
        &mut batching,
        "max_forward_tokens",
        options.max_forward_tokens,
    );
    insert_int(
        &mut batching,
        "max_forward_requests",
        options.max_forward_requests,
    );
    insert_int(&mut batching, "cpu_pages", options.cpu_pages);
    insert_str(
        &mut batching,
        "kv_cache_dtype",
        options.kv_cache_dtype.clone(),
    );
    // Omitted when unset rather than written as 0: the engine reads absent and
    // zero the same way, and a config that does not mention the knob is the
    // honest record of a run that did not use it.
    if let Some(len) = options.max_model_len {
        insert_int(&mut batching, "max_model_len", len);
    }
    insert_table(&mut doc, "batching", batching);

    let mut runtime = toml::Table::new();
    insert_bool(&mut runtime, "verbose", options.verbose);
    insert_table(&mut doc, "runtime", runtime);
    insert_cache_table(&mut doc);

    write_toml_table(out_path, doc)
}

// -----------------------------------------------------------------------------
// Native engine creation helpers.
// -----------------------------------------------------------------------------

#[cfg(any(
    feature = "_engine-cuda",
    all(feature = "engine-metal", target_vendor = "apple")
))]
fn local_engine_state_dir(group_id: usize, tp: Option<&TpLaunch>) -> Result<PathBuf> {
    let rank_suffix = tp
        .as_ref()
        .map(|tp| format!("-r{}", tp.rank))
        .unwrap_or_default();
    let state_dir = launch_state_dir().join(format!("g{group_id}{rank_suffix}"));
    std::fs::create_dir_all(&state_dir)
        .map_err(|e| anyhow!("create state dir {state_dir:?}: {e}"))?;
    Ok(state_dir)
}

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
    weight_dtype: &str,
    snapshot_dir: &Path,
    config: &[u8],
    group_id: usize,
    tp_launches: &[TpLaunch],
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    adapters: &crate::config::AdapterConfig,
) -> Result<crate::translate::GroupEngine> {
    validate_snapshot_dir(snapshot_dir)?;
    if rank_options.is_empty() {
        return Err(anyhow!("cuda group requires at least one rank"));
    }
    if rank_options.len() != tp_launches.len() {
        return Err(anyhow!(
            "cuda group rank options ({}) and tp launches ({}) length mismatch",
            rank_options.len(),
            tp_launches.len()
        ));
    }

    let mut config_blobs = Vec::with_capacity(rank_options.len());
    for (rank_options, tp) in rank_options.iter().zip(tp_launches.iter()) {
        // THE `else` IS UNREACHABLE IN ONE BUILD AND LOAD-BEARING IN EVERY
        // OTHER. `EngineOptions`' variants are feature-gated, so a binary
        // built with CUDA and nothing else has a one-variant enum and the
        // pattern is irrefutable; add `engine-metal` or `engine-vulkan` and
        // the refusal below is the only thing standing between a metal
        // option set and `write_cuda_startup_toml`. Allowed rather than
        // rewritten, because the rewrite is to delete a check that a
        // different feature list needs.
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
        let state_dir = local_engine_state_dir(group_id, Some(tp))?;
        let toml_path = state_dir.join("engine.toml");
        write_cuda_startup_toml(
            &toml_path,
            opts,
            weight_dtype,
            snapshot_dir,
            group_id,
            Some(tp),
            config,
        )?;
        // THE DOCUMENT, not the path to it. See the single-rank arm in
        // `create_engine_backend` for what handing over the path cost.
        config_blobs.push(std::fs::read(&toml_path).with_context(|| {
            format!("read the engine boot config just written to {toml_path:?}")
        })?);
    }

    let ranks = rank_options.len();
    let (mut backend, opened) = ::runtime::engine::backend::open::cuda_group(config_blobs)?;
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
        cuda_budgets(opts, adapters.seats()),
        engine_api::model_ir::Platform::Cuda,
        component,
        frames_in_flight,
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
    weight_dtype: &str,
    snapshot_dir: &Path,
    config: &[u8],
    group_id: usize,
    tp: Option<&TpLaunch>,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    adapters: &crate::config::AdapterConfig,
) -> Result<crate::translate::GroupEngine> {
    // Each is used only inside a `#[cfg(feature = "engine-…")]` arm below.
    let _ = (group_id, tp, config, weight_dtype);
    validate_snapshot_dir(snapshot_dir)?;

    // TYPED, because with no `engine-*` feature `EngineOptions` has no
    // variants at all and this `match` diverges — inference has nothing to
    // work from. That build reaches no device, which is the truth since the
    // interpreter backend was deleted: there is no ungated flavor left to
    // fall back to.
    let (mut backend, budgets, platform): (
        ::runtime::engine::EngineBox,
        engine_api::Budgets,
        engine_api::model_ir::Platform,
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
                let state_dir = local_engine_state_dir(group_id, tp)?;
                let toml_path = state_dir.join("engine.toml");
                write_cuda_startup_toml(
                    &toml_path,
                    opts,
                    weight_dtype,
                    snapshot_dir,
                    group_id,
                    tp,
                    config,
                )?;
                // THE DOCUMENT, not the path to it.
                //
                // This handed over `toml_path.to_string_lossy()`, and the
                // engine parses what it is given as TOML: `Shell::open` ->
                // `load::create_impl` does `from_utf8(bytes).parse::<toml::
                // Table>().ok().unwrap_or_default()`. A PATH is valid UTF-8
                // and is not TOML, so it parsed to nothing and every boot key
                // fell back to a default IN SILENCE -- `[model] config`,
                // `[model] id` and `[engine] runahead`, all written into the
                // file and none of them read.
                //
                // The tolerance that hid it was the runtime's own boot
                // reader, which took bytes "that are a PATH rather than a
                // document" for "the operator stated nothing". That reader
                // had no callers left and is gone; this seam hands over the
                // document, so there is nothing left to tolerate.
                let boot_doc = std::fs::read(&toml_path).with_context(|| {
                    format!("read the engine boot config just written to {toml_path:?}")
                })?;
                let backend = ::runtime::engine::backend::open::cuda(&boot_doc)?;
                (
                    backend,
                    cuda_budgets(opts, adapters.seats()),
                    engine_api::model_ir::Platform::Cuda,
                )
            }
            // METAL, BACK AT P5, AND IT HANDS OVER THE DOCUMENT — the same
            // shape as the CUDA arm above, which it did NOT have before R3.
            // `open::metal` used to be given the PATH, on the reading that the
            // engine opened the file itself; the engine does not, and says so:
            // `Shell::open` takes `[model] id` already parsed, because "a boot
            // TOML is the runtime's format, and an engine that read one would be
            // the second thing entitled to an opinion about its shape." The
            // file is still written — it is what an operator reads to see what
            // the launch actually asked for — and then read back, so exactly
            // one thing parses it.
            //
            // THE VULKAN AND WGPU ARMS STOOD HERE TOO and are still out:
            // neither engine has the baker executor R3 named as the condition
            // of its return.
            #[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
            EngineOptions::Metal(opts) => {
                let state_dir = local_engine_state_dir(group_id, tp)?;
                let toml_path = state_dir.join("engine.toml");
                write_metal_startup_toml(&toml_path, opts, snapshot_dir, group_id, config)?;
                let boot_doc = std::fs::read(&toml_path).with_context(|| {
                    format!("read the engine boot config just written to {toml_path:?}")
                })?;
                let backend = ::runtime::engine::backend::open::metal(&boot_doc)?;
                (
                    backend,
                    engine_api::Budgets {
                        max_lanes: opts.max_forward_requests.max(1),
                        max_tokens: opts.max_forward_tokens.max(1),
                        buckets: Vec::new(),
                        // The same one number the CUDA arm takes. `[model]`
                        // rather than `[model.engine.options]` is what makes
                        // it portable: debt 6's complaint was that Metal had
                        // no configuration at all, and an adapter seat count
                        // is not a fact about a backend.
                        max_adapters: adapters.seats(),
                        page_size: opts.kv_page_size.max(1),
                        max_context: opts
                            .max_model_len
                            .unwrap_or_else(|| engine_api::Budgets::default().max_context),
                        slots: opts.total_pages.max(1),
                    },
                    engine_api::model_ir::Platform::Metal,
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
        platform,
        component,
        frames_in_flight,
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
        let budgets = cuda_budgets(&opts, 0);
        assert_eq!(budgets.page_size, 16);
        // 4096 tokens of context is 256 pages a slot; 1024 pages seats four.
        assert_eq!(budgets.max_context, 4096);
        assert_eq!(budgets.slots, 4);

        // No cap stated: the contract's own default seat count, not a
        // division by nothing.
        opts.max_total_pages = None;
        assert_eq!(cuda_budgets(&opts, 0).slots, 256);

        // A cap smaller than one slot's block still seats one — a pool that
        // seats nothing is a load that cannot fire.
        opts.max_total_pages = Some(1);
        assert_eq!(cuda_budgets(&opts, 0).slots, 1);
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
            cuda_budgets(&opts, 0).max_adapters,
            0,
            "a deployment that states no adapters still registers none"
        );
        assert_eq!(cuda_budgets(&opts, 8).max_adapters, 8);
    }

    /// A stand-in checkpoint config for the tests that are about something
    /// else. The writers move the bytes without reading them, so the smallest
    /// valid document is the honest fixture: anything richer would suggest
    /// these tests check the config's content, and none of them do
    /// (`the_startup_toml_always_carries_the_config` is the one that checks
    /// it arrives).
    const CONFIG: &[u8] = br#"{}"#;

    /// What `[model] weight_dtype` defaults to. It is a model key rather than
    /// an engine option, so these tests state it the way the caller does
    /// instead of reading it off `CudaNativeEngineOptions::default()`.
    const DTYPE: &str = "bfloat16";

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

    #[cfg(feature = "_engine-cuda")]
    #[test]
    fn tp_launches_share_nccl_id_and_assign_all_ranks() {
        let launches = tp_launches(3).unwrap();
        assert_eq!(launches.len(), 3);
        assert!(!launches[0].nccl_unique_id_hex.is_empty());
        assert!(
            launches
                .iter()
                .all(|launch| launch.nccl_unique_id_hex == launches[0].nccl_unique_id_hex)
        );
        assert_eq!(
            launches
                .iter()
                .map(|launch| launch.rank)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(launches.iter().all(|launch| launch.size == 3));
    }

    #[test]
    fn the_startup_toml_carries_the_cache_root() {
        // The engine derives every disk cache from this. Without it the caches
        // fall back to XDG, which is what split pie's state across two roots.
        //
        // NOTE: this installs a process-global OnceLock that outlives the test,
        // so every later test in this binary sees `[cache]` emitted. Nothing
        // asserts its absence today; a test that needs it unset cannot share a
        // process with this one.
        set_cache_dir("/pie-home/cache".to_string());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("engine.toml");
        let snap = dir.path().join("snapshot");
        write_cuda_startup_toml(
            &out,
            &CudaNativeEngineOptions::default(),
            DTYPE,
            &snap,
            0,
            None,
            CONFIG,
        )
        .unwrap();
        let val: toml::Value = toml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(val["cache"]["dir"].as_str().unwrap(), "/pie-home/cache");
    }

    /// **THE `[engine]` TABLE, WRITTEN** (alto wave P, article 9). The nine
    /// `PIE_CUDA_*` words the shell used to read are `[engine]` keys now, and a
    /// key nobody writes is a key nobody can set — the same failure
    /// `[model] weight_cache_dir` had for a whole rewrite.
    ///
    /// A default boot writes NO `[engine]` table at all, because the shell's
    /// own default is the answer for an absent key and a second spelling of
    /// absence is a second place the default is written down.
    #[test]
    fn the_startup_toml_writes_only_the_engine_knobs_an_operator_stated() {
        let dir = tempfile::tempdir().unwrap();
        let snap = dir.path().join("snapshot");

        let write = |opts: &CudaNativeEngineOptions, name: &str| -> toml::Value {
            let out = dir.path().join(name);
            write_cuda_startup_toml(&out, opts, DTYPE, &snap, 0, None, CONFIG).unwrap();
            toml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap()
        };

        let quiet = write(&CudaNativeEngineOptions::default(), "quiet.toml");
        assert!(
            quiet.get("engine").is_none(),
            "a boot that stated no knob must write no `[engine]` table: {quiet:?}"
        );

        let stated = write(
            &CudaNativeEngineOptions {
                graphs: Some("on".into()),
                fold: Some(true),
                pad: Some(false),
                fold_disable: Some("library".into()),
                side_streams: Some(0),
                ..CudaNativeEngineOptions::default()
            },
            "stated.toml",
        );
        let engine = &stated["engine"];
        assert_eq!(engine["graphs"].as_str().unwrap(), "on");
        assert_eq!(engine["fold"].as_bool().unwrap(), true);
        assert_eq!(engine["pad"].as_bool().unwrap(), false);
        assert_eq!(engine["fold_disable"].as_str().unwrap(), "library");
        assert_eq!(engine["side_streams"].as_integer().unwrap(), 0);
        // And nothing the operator did not state.
        assert!(engine.get("pipeline").is_none());
        assert!(engine.get("grouped").is_none());
        assert!(engine.get("fallback_copy").is_none());
    }

    #[test]
    fn the_sweep_reclaims_dead_pids_and_spares_live_ones() {
        let home = tempfile::tempdir().unwrap();
        // SAFETY: single-threaded test; PIE_HOME is read, never written, by
        // the code under test.
        unsafe { std::env::set_var("PIE_HOME", home.path()) };

        let root = home.path().join("standalone");
        let self_pid = std::process::id();
        // A pid that cannot be running: pid 0 is the kernel's, never a
        // reachable user process, so `kill(0, 0)` reports it as not ours.
        let dead = root.join("999999999");
        let live = root.join(self_pid.to_string());
        let foreign = root.join("not-a-pid");
        for d in [&dead, &live, &foreign] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("engine.toml"), "x").unwrap();
        }

        sweep_stale_launch_state();

        assert!(!dead.exists(), "a dead pid's state must be reclaimed");
        assert!(
            live.exists(),
            "the running process's own state must survive"
        );
        assert!(
            foreign.exists(),
            "a directory that is not a pid is not ours to remove"
        );
    }

    /// A calibration request reaches the engine, and only ever from memory.
    ///
    /// This is the whole route that replaced `[engine] calibrate_planner`:
    /// `pie config tune` sets `server.calibrate_planner` on a config it
    /// derived, `::runtime::apply_embedded_calibration` puts it on the engine
    /// options, and this is where it becomes something the C++ side reads. The
    /// per-launch bootstrap TOML is the only file it ever appears in, and that
    /// file is regenerated every boot -- so the request cannot outlive the boot
    /// that made it.
    #[test]
    fn a_calibration_request_reaches_the_engine_and_stops_there() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let opts = CudaNativeEngineOptions {
            device: "cuda:0".to_string(),
            calibrate_planner: true,
            ..Default::default()
        };

        write_cuda_startup_toml(&out, &opts, DTYPE, &snap, 0, None, CONFIG).unwrap();
        let val: toml::Value = toml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(
            val["batching"]["calibrate_planner"].as_bool(),
            Some(true),
            "the engine never hears the request"
        );

        // And the field is not part of the file format: a user config that
        // spells it is refused, so this value can only have come from memory.
        let asked = "\
[model]
name = \"m\"
model = \"x\"
[engine]
type = \"cuda_native\"
device = [\"cuda:0\"]
calibrate_planner = true
";
        let err = crate::config::Config::parse(asked)
            .expect_err("a measurement is not a setting")
            .to_string();
        assert!(err.contains("calibrate_planner"), "got: {err}");
    }

    #[test]
    fn cuda_startup_toml_matches_engine_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let opts = CudaNativeEngineOptions {
            device: "cuda:0".to_string(),
            ..Default::default()
        };

        write_cuda_startup_toml(&out, &opts, DTYPE, &snap, 0, None, CONFIG).unwrap();

        // Re-parse the emitted TOML to confirm the schema the cuda
        // engine expects matches what we wrote (engine-side parsing
        // in crates/engine-cuda/csrc/src/config.hpp).
        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert!(
            val["model"].get("model").is_none(),
            "cuda derives from snapshot_dir"
        );
        assert_eq!(
            val["model"]["snapshot_dir"].as_str().unwrap(),
            snap.to_str().unwrap()
        );
        assert_eq!(val["model"]["device"].as_str().unwrap(), "cuda:0");
        assert_eq!(val["model"]["dtype"].as_str().unwrap(), "bfloat16");
        assert!(val["model"].get("runtime_quant").is_none()); // omitted when empty
        // Derived values are OMITTED, not written as a sentinel. The engine's
        // own default is the derivation, so emitting `0 = derive` would be a
        // second spelling of an absent key.
        assert!(val["batching"].get("kv_page_size").is_none());
        assert_eq!(val["batching"]["kv_cache_dtype"].as_str().unwrap(), "auto");
        assert_eq!(
            val["batching"]["gpu_mem_utilization"].as_float().unwrap(),
            0.90
        );
        assert_eq!(val["batching"]["memory_profile"].as_str().unwrap(), "auto");
        assert!(val["batching"].get("total_pages").is_none());
        // An ordinary boot says nothing about calibration: it is one run of a
        // measurement, not a setting every bootstrap file restates. The only
        // thing that ever turns it on is `pie config tune`, on a config it
        // derived in memory -- see `CudaNativeEngineOptions::calibrate_planner`
        // for why it cannot come from a file.
        assert!(val["batching"].get("calibrate_planner").is_none());
        assert_eq!(val["batching"].as_table().unwrap().len(), 4);
        assert_eq!(val["batching"]["swap_pool_size"].as_integer().unwrap(), 0);
        // Expert streaming is off unless an operator asks for it: for a model
        // that fits it is strictly slower, and it costs graph capture besides.
        assert!(!val["model"]["stream_routed_experts"].as_bool().unwrap());
        assert!(val["model"].get("expert_cache_gb").is_none());
        assert!(val["model"].get("expert_host_cache_gb").is_none());
        assert!(!val["runtime"]["verbose"].as_bool().unwrap());
    }

    #[test]
    fn cuda_startup_toml_emits_runtime_verbose_when_set() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let opts = CudaNativeEngineOptions {
            device: "cuda:0".to_string(),
            verbose: true,
            ..Default::default()
        };

        write_cuda_startup_toml(&out, &opts, DTYPE, &snap, 0, None, CONFIG).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert!(val["runtime"]["verbose"].as_bool().unwrap());
    }

    #[test]
    fn cuda_startup_toml_keeps_runtime_quant_out_of_engine_config() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let opts = CudaNativeEngineOptions {
            device: "cuda:1".to_string(),
            runtime_quant: "fp8".to_string(),
            ..Default::default()
        };

        write_cuda_startup_toml(&out, &opts, DTYPE, &snap, 3, None, CONFIG).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert!(val["model"].get("runtime_quant").is_none());
        assert_eq!(val["model"]["device"].as_str().unwrap(), "cuda:1");
    }

    #[test]
    fn cuda_startup_toml_keeps_mxfp4_policy_out_of_engine_config() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let opts = CudaNativeEngineOptions {
            device: "cuda:0".to_string(),
            mxfp4_moe: "bf16".to_string(),
            ..Default::default()
        };

        write_cuda_startup_toml(&out, &opts, DTYPE, &snap, 0, None, CONFIG).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert!(val["model"].get("mxfp4_moe").is_none());
    }

    #[test]
    fn cuda_startup_toml_emits_distributed_block_for_tp() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let opts = CudaNativeEngineOptions {
            device: "cuda:1".to_string(),
            ..Default::default()
        };
        let tp = TpLaunch {
            size: 2,
            rank: 1,
            nccl_unique_id_hex: "abcd".to_string(),
        };

        write_cuda_startup_toml(&out, &opts, DTYPE, &snap, 4, Some(&tp), CONFIG).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(val["distributed"]["tp_size"].as_integer().unwrap(), 2);
        assert_eq!(val["distributed"]["tp_rank"].as_integer().unwrap(), 1);
        assert_eq!(
            val["distributed"]["nccl_unique_id_hex"].as_str().unwrap(),
            "abcd",
        );
        assert!(
            val["distributed"].get("startup_barrier_path").is_none(),
            "startup_barrier_path no longer emitted (replaced by in-process std::barrier)"
        );
    }

    /// The checkpoint's config travels beside the bootstrap TOML — always.
    ///
    /// This used to assert the other half too: that the key is *absent* for a
    /// snapshot, which is what let each engine keep a `config.json` parser for
    /// the absent case. `weights.rs` lifts a snapshot's config now, so there
    /// is no absent case to pin and the parsers are gone. The writers still
    /// take it as an argument rather than deriving it from the path — lifting
    /// it is the resolver's job, done once — so this is about the *contract*,
    /// not about where the bytes came from.
    ///
    /// The Metal and Vulkan writers were pinned here beside it, taking the
    /// same argument; both left with the engines that read them.
    #[test]
    fn the_startup_toml_always_carries_the_config() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir.path().join("snap");
        std::fs::create_dir(&snapshot).unwrap();
        let body = br#"{"version":"pie.model/1","hidden_size":64}"#;

        let out = dir.path().join("cuda").join("engine.toml");
        std::fs::create_dir_all(out.parent().unwrap()).unwrap();
        let cuda = CudaNativeEngineOptions::default();
        write_cuda_startup_toml(&out, &cuda, DTYPE, &snapshot, 0, None, body).unwrap();

        let doc: toml::Value = toml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        let carried = doc["model"]
            .get("config")
            .and_then(|v| v.as_str())
            .map(|path| std::fs::read(path).unwrap());
        assert_eq!(carried.as_deref(), Some(body.as_slice()));
    }
}
