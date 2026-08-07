//! Driver-backend bootstrap helpers for pie-worker.
//!
//! This module exposes:
//!   * [`DriverCapabilities`] — typed driver capability payloads.
//!   * [`write_cuda_startup_toml`] / [`write_metal_startup_toml`] — emit the
//!     per-launch TOML each native driver reads at creation.
//!   * [`create_driver_backend`] — build a runtime-owned [`DriverBackend`]
//!     plus its caps before `pie_engine::bootstrap`.

#[cfg(feature = "driver-cuda")]
use std::ffi::CStr;
#[cfg(feature = "driver-cuda")]
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

use crate::config::{
    CudaMemoryProfile, CudaNativeDriverOptions, DummyDriverOptions, MetalDriverOptions,
};
use crate::driver_ffi::Flavor;

/// Anchors the loader ABI's C entry points into the final binary.
///
/// `pie-loader-capi` is an rlib and the only callers of
/// `pie_loader_compile_model` and friends are the C++ drivers, which link
/// *after* Rust. A linker never pulls an rlib member in on behalf of a C++
/// reference, so without a reference from reachable Rust the entry points are
/// simply absent and the failure surfaces as an undefined symbol at final
/// link (`loader/architecture.md` §3.4). The `#[used]` table inside
/// `pie_loader_capi::entry` keeps all six alive once the object is pulled in;
/// this static is what pulls it in.
///
/// **This is load-bearing.** No Rust in this process calls the loader any more —
/// §12 step 2 moved plan compilation behind the FFI, and the boot's request
/// entry is `pie_loader_compile_model` (`plan/model-in-rust.md` §6) — so this
/// reference is the only thing keeping the object file in the link.
#[used]
static PIE_LOADER_ENTRY_ANCHOR: unsafe extern "C" fn(
    *const pie_loader_capi::model::PieLoaderModelRequest,
    *mut *mut pie_loader_capi::PieLoaderPlan,
    *mut u32,
    *mut *mut pie_loader_capi::PieLoaderDiagnostics,
) -> pie_loader_capi::PieLoaderStatus = pie_loader_capi::model::pie_loader_compile_model;

#[cfg(feature = "driver-cuda")]
#[repr(C)]
struct NcclUniqueId {
    internal: [u8; 128],
}

#[cfg(feature = "driver-cuda")]
unsafe extern "C" {
    fn ncclGetUniqueId(unique_id: *mut NcclUniqueId) -> c_int;
    fn ncclGetErrorString(result: c_int) -> *const c_char;
}

#[cfg(feature = "driver-cuda")]
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

/// Per-flavor driver options, passed to native-driver creation helpers so the
/// caller doesn't have to discriminate on `DriverKind` in two places.
///
/// The `Dummy` variant carries `random_seed` and `activation_dtype`
/// alongside `DummyDriverOptions` because those are universal
/// `[model.driver]` fields.
///
/// `Clone` exists so `serve.rs` can rebuild a per-group variant
/// (different `device`) from a model-level template without
/// re-deserializing TOML.
#[derive(Clone)]
pub enum DriverOptions {
    #[cfg(feature = "driver-cuda")]
    CudaNative(CudaNativeDriverOptions),
    #[cfg(feature = "driver-metal")]
    Metal(MetalDriverOptions),
    Dummy {
        opts: DummyDriverOptions,
        random_seed: u64,
        activation_dtype: String,
    },
}

impl DriverOptions {
    /// Which compiled flavor this options bundle targets.
    pub fn flavor(&self) -> Flavor {
        match self {
            #[cfg(feature = "driver-cuda")]
            DriverOptions::CudaNative(_) => Flavor::Cuda,
            #[cfg(feature = "driver-metal")]
            DriverOptions::Metal(_) => Flavor::Metal,
            DriverOptions::Dummy { .. } => Flavor::Dummy,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TpLaunch {
    size: usize,
    rank: usize,
    nccl_unique_id_hex: String,
}

#[cfg(feature = "driver-cuda")]
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

fn insert_int(table: &mut toml::Table, key: &str, value: impl Into<i64>) {
    table.insert(key.into(), toml::Value::Integer(value.into()));
}

/// This model's materialized-weight artifact directory, installed once before
/// any driver is created and written into every startup TOML from there.
///
/// Install-at-bootstrap rather than a parameter because the TOML writers sit
/// five call layers below the only place holding a parsed `Config`. First
/// writer wins, so a directory a live driver is already using cannot move.
static WEIGHT_CACHE_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Install the resolved weight-artifact directory. The caller resolves the
/// `$PIE_HOME/models` default, because `$PIE_HOME` is the bin/worker layer's
/// to know and the driver has never been told it.
pub fn set_weight_cache_dir(dir: String) {
    let _ = WEIGHT_CACHE_DIR.set(dir);
}

fn weight_cache_dir() -> String {
    WEIGHT_CACHE_DIR.get().cloned().unwrap_or_default()
}

/// The root every driver-side disk cache derives from: `$PIE_HOME/cache`.
///
/// Location is convention, not configuration -- there is no config field for
/// it, and `$PIE_HOME` is the one lever that moves it. Before this the driver
/// caches derived from `$XDG_CACHE_HOME`/`$HOME/.cache` instead, not as a
/// choice but because the driver had never been told `$PIE_HOME`. That split
/// pie's state across two roots: `pie serve` wrote programs, logs and
/// optimized checkpoints under one and compiled PTIR, GEMM tuning and planner
/// profiles under another.
static CACHE_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Install the resolved cache root, before any driver is created. First writer
/// wins, so a cache a live driver is already using cannot move.
pub fn set_cache_dir(dir: String) {
    let _ = CACHE_DIR.set(dir);
}

fn cache_dir() -> String {
    CACHE_DIR.get().cloned().unwrap_or_default()
}

/// Emit `[cache] dir` into a driver's startup TOML.
///
/// Omitted when unset so a driver launched with a hand-written TOML (its own
/// `dev.toml`, say) keeps the XDG derivation rather than losing its cache to
/// an empty path.
fn insert_cache_table(doc: &mut toml::Table) {
    let dir = cache_dir();
    if dir.is_empty() {
        return;
    }
    let mut table = toml::Table::new();
    insert_str(&mut table, "dir", dir);
    insert_table(doc, "cache", table);
}

fn insert_str(table: &mut toml::Table, key: &str, value: impl Into<String>) {
    table.insert(key.into(), toml::Value::String(value.into()));
}

fn insert_bool(table: &mut toml::Table, key: &str, value: bool) {
    table.insert(key.into(), toml::Value::Boolean(value));
}

fn insert_table(doc: &mut toml::Table, key: &str, table: toml::Table) {
    doc.insert(key.into(), toml::Value::Table(table));
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}

/// Writes the compiled model config beside the startup TOML and names it in
/// `[model]`.
///
/// Beside rather than inlined: the driver already takes a path, and opening a
/// second one is less machinery than teaching TOML to carry a JSON document.
///
/// Unconditional. It was optional while a snapshot reached the driver without
/// a descriptor and each driver parsed `config.json` itself; `weights.rs`
/// normalizes that case now, so there is one normalizer and every boot writes
/// this file. The type says so, which is what keeps the deleted branch from
/// growing back.
fn write_descriptor_beside(
    out_path: &Path,
    descriptor: &[u8],
    model: &mut toml::Table,
) -> Result<()> {
    let beside = out_path.with_file_name("model.descriptor.json");
    std::fs::write(&beside, descriptor)
        .with_context(|| format!("write model descriptor {beside:?}"))?;
    insert_str(model, "descriptor", path_string(&beside));
    Ok(())
}

fn write_toml_table(out_path: &Path, doc: toml::Table) -> Result<()> {
    let serialized = toml::to_string(&doc).map_err(|e| anyhow!("serialize startup TOML: {e}"))?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create startup toml dir {parent:?}: {e}"))?;
    }
    std::fs::write(out_path, serialized)
        .map_err(|e| anyhow!("write startup toml {out_path:?}: {e}"))?;
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
/// live launch's startup TOML is a driver that cannot boot.
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
/// Each launch writes a driver startup TOML under its own pid and nothing ever
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

// `DriverCapabilities` is owned by `pie-driver-abi` (single source of truth
// for the driver ↔ runtime interface). Re-exported here so existing call
// sites in pie-worker keep working through the
// `embedded_driver::DriverCapabilities` path.
pub use pie_driver_abi::DriverCapabilities;

/// Parse a capability JSON blob into the typed driver-capability struct.
/// Lives in pie-worker (not bridge) so bridge can stay free of a
/// serde_json dependency.
fn parse_caps_json(json: &str) -> Result<DriverCapabilities> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| anyhow::anyhow!("driver caps JSON parse: {e}"))?;
    serde_json::from_value(value).map_err(|e| anyhow::anyhow!("driver caps schema mismatch: {e}"))
}

/// Read model facts out of `<snapshot>/config.json`.
/// Used by [`write_dummy_startup_toml`] when the user didn't explicitly
/// specify them in `[model.driver.options]`. Mirrors the legacy Python
/// dummy driver's `hf_utils.load_hf_config()`-based discovery.
fn read_hf_config_defaults(snapshot_dir: &Path) -> Result<(u32, String, u32)> {
    let path = snapshot_dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| anyhow!("read {path:?}: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| anyhow!("parse {path:?}: {e}"))?;

    let vocab_size = v
        .get("vocab_size")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| anyhow!("`vocab_size` missing from {path:?}"))? as u32;

    let raw_arch = v
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|a| a.as_str())
        .ok_or_else(|| anyhow!("`architectures[0]` missing from {path:?}"))?;
    // "Qwen3ForCausalLM" → "qwen3" — same heuristic the Python wrapper used,
    // and the same one `driver/metal`'s `arch_stem` applies. The task suffix
    // is what comes off: a multimodal release is named
    // `<Stem>ForConditionalGeneration`, and leaving that whole misses every
    // registry keyed on the stem.
    //
    // The list is explicit rather than "cut at the first `for`" because
    // `ReformerForCausalLM` has one inside its own stem.
    let raw_arch_lower = raw_arch.to_lowercase();
    let arch_name = raw_arch_lower
        .strip_suffix("forconditionalgeneration")
        .or_else(|| raw_arch_lower.strip_suffix("forcausallm"))
        .unwrap_or(&raw_arch_lower)
        .to_string();

    let max_model_len = v
        .get("max_position_embeddings")
        .or_else(|| v.get("max_sequence_length"))
        .or_else(|| v.get("model_max_length"))
        .or_else(|| v.get("context_length"))
        .or_else(|| v.get("n_positions"))
        .and_then(|x| x.as_u64())
        .unwrap_or(4096) as u32;

    Ok((vocab_size, arch_name, max_model_len))
}

/// Emit the metal driver's startup TOML — same `[model]` + `[batching]` +
/// `[runtime]` layout consumed by `driver/metal/src/config.hpp`. The metal
/// launch state is identical apart from the `metal:N` backend selector.
///
/// Not gated on `driver-metal`: what it produces is a TOML file, and whether
/// the operator's settings survive into that file is a question a machine
/// without a Metal device can still answer. Gating it would put the test out
/// of reach of every machine that is not a Mac.
pub fn write_metal_startup_toml(
    out_path: &Path,
    options: &MetalDriverOptions,
    snapshot_dir: &Path,
    _group_id: usize,
    descriptor: &[u8],
) -> Result<()> {
    let mut doc = toml::Table::new();

    let mut model = toml::Table::new();
    insert_str(&mut model, "hf_path", path_string(snapshot_dir));
    // Same arrangement as the CUDA driver.
    write_descriptor_beside(out_path, descriptor, &mut model)?;
    insert_str(&mut model, "backend", &options.device);
    insert_bool(
        &mut model,
        "stream_routed_experts",
        options.stream_routed_experts,
    );
    // Omitted when unset rather than written as 0: the driver reads an absent
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
    // Omitted when unset rather than written as 0: the driver reads absent and
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

/// Build the model-load request the driver will compile from.
///
/// The runtime states *what* it wants loaded — the checkpoint, the quantization
/// it would prefer, the MoE lowering, the component scope — and nothing about
/// the device. The driver measures the device and calls the loader itself
/// (`loader/architecture.md` §3).
fn model_load_desc(
    snapshot_dir: &Path,
    runtime_quant: &str,
    mxfp4_moe: &str,
    component: pie_driver_abi::ModelComponent,
) -> Result<pie_driver_abi::ModelLoadDesc> {
    let mxfp4_moe = pie_driver_abi::Mxfp4MoeRequest::parse(mxfp4_moe)
        .ok_or_else(|| anyhow!("unknown mxfp4_moe policy '{mxfp4_moe}'"))?;
    Ok(pie_driver_abi::ModelLoadDesc {
        snapshot_dir: snapshot_dir.to_path_buf(),
        runtime_quant: runtime_quant.to_string(),
        mxfp4_moe,
        component,
    })
}

/// Write the cuda driver's startup TOML. Schema mirrors
/// `driver/cuda/src/config.hpp`: `[model]` with
/// `snapshot_dir`/`device`/`dtype` plus model-execution knobs,
/// `[batching]` with KV-page geometry plus `swap_pool_size`, and `[runtime]`
/// with the server verbosity flag.
///
/// `[distributed]` is emitted only for TP launches; single-rank uses the
/// cuda driver's default (`tp_size=1, tp_rank=0`).
pub(crate) fn write_cuda_startup_toml(
    out_path: &Path,
    opts: &CudaNativeDriverOptions,
    snapshot_dir: &Path,
    _group_id: usize,
    tp: Option<&TpLaunch>,
    descriptor: &[u8],
) -> Result<()> {
    let mut doc = toml::Table::new();

    let mut model = toml::Table::new();
    insert_str(&mut model, "snapshot_dir", path_string(snapshot_dir));
    insert_str(&mut model, "weight_cache_dir", weight_cache_dir());
    write_descriptor_beside(out_path, descriptor, &mut model)?;
    insert_str(&mut model, "device", &opts.device);
    insert_str(&mut model, "dtype", opts.weight_dtype.clone());
    insert_int(&mut model, "mtp_num_drafts", opts.mtp_num_drafts);
    insert_bool(
        &mut model,
        "stream_routed_experts",
        opts.stream_routed_experts,
    );
    // Omitted when absent rather than written as a sentinel: the driver's
    // own default IS the derivation, so an absent key and a "0 means derive"
    // key would be two spellings of one thing.
    // The driver still speaks GiB floats; the unit lives in the config type,
    // not on the wire.
    if let Some(size) = opts.expert_cache {
        model.insert("expert_cache_gb".into(), toml::Value::Float(size.as_gib_f64()));
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
    // Omitted when absent, like the other derived keys: the driver defaults
    // them to "let the planner choose", so writing a sentinel would be a
    // second spelling of an absent key.
    if let Some(tokens) = opts.max_forward_tokens {
        insert_int(&mut batching, "max_forward_tokens", tokens);
    }
    if let Some(requests) = opts.max_forward_requests {
        insert_int(&mut batching, "max_forward_requests", requests);
    }
    insert_str(&mut batching, "kv_cache_dtype", opts.kv_cache_dtype.clone());
    // Written only when asked for, like the derived keys above: the driver
    // defaults it to false too, so emitting `false` would be a second spelling
    // of an absent key. It also keeps the startup TOML saying nothing about
    // calibration on every ordinary boot.
    if opts.calibrate_planner {
        insert_bool(&mut batching, "calibrate_planner", true);
    }
    insert_table(&mut doc, "batching", batching);

    let mut runtime = toml::Table::new();
    insert_bool(&mut runtime, "verbose", opts.verbose);
    insert_table(&mut doc, "runtime", runtime);
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

// -----------------------------------------------------------------------------
// Native driver creation helpers.
// -----------------------------------------------------------------------------

fn local_driver_state_dir(group_id: usize, tp: Option<&TpLaunch>) -> Result<PathBuf> {
    let rank_suffix = tp
        .as_ref()
        .map(|tp| format!("-r{}", tp.rank))
        .unwrap_or_default();
    let state_dir = launch_state_dir().join(format!("g{group_id}{rank_suffix}"));
    std::fs::create_dir_all(&state_dir)
        .map_err(|e| anyhow!("create state dir {state_dir:?}: {e}"))?;
    Ok(state_dir)
}

fn dummy_native_options(
    opts: &DummyDriverOptions,
    snapshot_dir: &Path,
    _random_seed: u64,
    activation_dtype: &str,
) -> Result<pie_driver_dummy_lib::DummyDriverOptions> {
    let (vocab_size, arch_name, max_model_len) = match (opts.vocab_size, opts.arch_name.as_deref())
    {
        (Some(v), Some(a)) => {
            let (_, _, auto_len) =
                read_hf_config_defaults(snapshot_dir).unwrap_or_else(|_| (v, a.to_string(), 4096));
            (v, a.to_string(), auto_len)
        }
        (v_opt, a_opt) => {
            let (auto_v, auto_a, auto_len) = read_hf_config_defaults(snapshot_dir)
                .with_context(|| "auto-discovering vocab_size + arch_name for dummy driver")?;
            (
                v_opt.unwrap_or(auto_v),
                a_opt.map(str::to_string).unwrap_or(auto_a),
                auto_len,
            )
        }
    };

    let max_forward_tokens = 4096u32;
    let max_forward_requests = 128u32;
    let total_pages = 256u32
        .max(max_forward_tokens.div_ceil(16))
        .max(max_model_len.div_ceil(16))
        .max(max_forward_requests.saturating_mul(2));

    Ok(pie_driver_dummy_lib::DummyDriverOptions {
        total_pages,
        kv_page_size: 16,
        swap_pool_size: 0,
        vocab_size,
        max_model_len,
        arch_name,
        activation_dtype: activation_dtype.to_string(),
        snapshot_dir: path_string(snapshot_dir),
        max_forward_tokens,
        max_forward_requests,
        max_page_refs: total_pages,
        has_mtp_logits: true,
        has_mtp_drafts: true,
        has_value_head: true,
        has_attn_score: true,
        callback_delay_ms: 0,
        reject_launches: false,
        reject_launches_remaining: 0,
        fail_launches_after_accept: false,
        retry_launches_remaining: 0,
        elastic_admission: false,
        prepare_exhaustions_remaining: 0,
        prepare_impossible_above_kv_pages: 0,
        operation_log: None,
        launch_observer: None,
    })
}

/// What a driver may be pointed at: a `.zt` artifact, or a snapshot directory.
///
/// The GGUF refusal that used to live here is gone with the reason for it.
/// It existed because the LoadPlan executors could not decode GGUF's blocked
/// schemes at load time — but `pie model import` decodes them now,
/// so what reaches a driver is a `.zt` either way and there is no format left
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

#[cfg(feature = "driver-cuda")]
pub(crate) fn create_driver_backend_group(
    rank_options: &[DriverOptions],
    snapshot_dir: &Path,
    descriptor: &[u8],
    group_id: usize,
    tp_launches: &[TpLaunch],
    component: pie_driver_abi::ModelComponent,
) -> Result<crate::translate::GroupDriver> {
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
        let DriverOptions::CudaNative(opts) = rank_options else {
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
        let state_dir = local_driver_state_dir(group_id, Some(tp))?;
        let toml_path = state_dir.join("driver.toml");
        write_cuda_startup_toml(
            &toml_path,
            opts,
            snapshot_dir,
            group_id,
            Some(tp),
            descriptor,
        )?;
        config_blobs.push(toml_path.to_string_lossy().into_owned().into_bytes());
    }

    let (mut backend, facts) = pie_engine::driver::DriverBackend::cuda_group_create(config_blobs)?;
    if facts.len() != rank_options.len() {
        return Err(anyhow!(
            "cuda group returned {} device-facts payloads for {} ranks",
            facts.len(),
            rank_options.len()
        ));
    }
    // Each rank's descriptor is identical: the per-rank facts (rank index, TP
    // width, device capability) reach the loader through the driver's own
    // startup TOML, not through the request.
    let descs = rank_options
        .iter()
        .map(|options| {
            let DriverOptions::CudaNative(opts) = options else {
                unreachable!("validated cuda options above");
            };
            model_load_desc(
                snapshot_dir,
                &opts.runtime_quant,
                &opts.mxfp4_moe,
                component,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let caps = backend.load_model(descs)?;
    Ok(crate::translate::GroupDriver { caps, backend })
}

pub(crate) fn create_driver_backend(
    options: &DriverOptions,
    snapshot_dir: &Path,
    descriptor: &[u8],
    group_id: usize,
    tp: Option<&TpLaunch>,
    component: pie_driver_abi::ModelComponent,
) -> Result<crate::translate::GroupDriver> {
    // Each is used only inside a `#[cfg(feature = "driver-…")]` arm below.
    let _ = (group_id, tp, descriptor);
    validate_snapshot_dir(snapshot_dir)?;

    let (mut backend, runtime_quant, mxfp4_moe) = match options {
        #[cfg(feature = "driver-cuda")]
        DriverOptions::CudaNative(opts) => {
            if opts.mtp_assistant_snapshot_dir.is_some() {
                return Err(anyhow!(
                    "mtp_assistant_snapshot_dir is not supported by the single-model \
                     LoadPlan boot contract"
                ));
            }
            let state_dir = local_driver_state_dir(group_id, tp)?;
            let toml_path = state_dir.join("driver.toml");
            write_cuda_startup_toml(&toml_path, opts, snapshot_dir, group_id, tp, descriptor)?;
            let config_path = toml_path.to_string_lossy();
            let (backend, _facts) =
                pie_engine::driver::DriverBackend::cuda_create(config_path.as_bytes())?;
            (
                backend,
                opts.runtime_quant.as_str(),
                opts.mxfp4_moe.as_str(),
            )
        }
        #[cfg(feature = "driver-metal")]
        DriverOptions::Metal(opts) => {
            let state_dir = local_driver_state_dir(group_id, tp)?;
            let toml_path = state_dir.join("driver.toml");
            write_metal_startup_toml(&toml_path, opts, snapshot_dir, group_id, descriptor)?;
            let config_path = toml_path.to_string_lossy();
            let (backend, _facts) =
                pie_engine::driver::DriverBackend::metal_create(config_path.as_bytes())?;
            (backend, "", "auto")
        }
        DriverOptions::Dummy {
            opts,
            random_seed,
            activation_dtype,
        } => {
            let options = dummy_native_options(opts, snapshot_dir, *random_seed, activation_dtype)?;
            let (backend, _facts) = pie_engine::driver::DriverBackend::dummy(options)?;
            (backend, "", "auto")
        }
    };
    // Uniform across backends now that the descriptor is a request rather than a
    // compiled plan: the dummy driver simply ignores everything but the
    // component scope (§10.3).
    let desc = model_load_desc(snapshot_dir, runtime_quant, mxfp4_moe, component)?;
    let caps = backend.load_model(vec![desc])?;

    Ok(crate::translate::GroupDriver { caps, backend })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in `pie.model/1` document for the tests that are about
    /// something else. The writers move the bytes without reading them, so
    /// the smallest valid document is the honest fixture: anything richer
    /// would suggest these tests check the descriptor's content, and none of
    /// them do (`the_startup_toml_always_carries_a_descriptor` is the one
    /// that checks it arrives).
    const DESCRIPTOR: &[u8] = br#"{"version":"pie.model/1"}"#;

    #[test]
    fn caps_json_round_trips() {
        let json = format!(
            r#"{{
            "abi_version": {},
            "total_pages": 1024,
            "kv_page_size": 32,
            "swap_pool_size": 0,
            "max_forward_tokens": 4096,
            "max_forward_requests": 512,
            "max_page_refs": 262144,
            "arch_name": "qwen3",
            "vocab_size": 151936,
            "max_model_len": 4096,
            "activation_dtype": "bfloat16",
            "snapshot_dir": "/tmp/snap"
        }}"#,
            pie_driver_abi::PIE_DRIVER_ABI_VERSION
        );
        let caps = parse_caps_json(&json).unwrap();
        assert_eq!(caps.abi_version, pie_driver_abi::PIE_DRIVER_ABI_VERSION);
        assert_eq!(caps.total_pages, 1024);
        assert_eq!(caps.arch_name, "qwen3");
        assert_eq!(caps.snapshot_dir, "/tmp/snap");
        assert_eq!(caps.max_forward_tokens, 4096);
        assert_eq!(caps.max_page_refs, 262144);
    }

    #[test]
    fn dummy_boot_uses_create_compile_load_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join("snapshot");
        std::fs::create_dir(&snapshot).unwrap();
        std::fs::write(
            snapshot.join("config.json"),
            r#"{
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "max_position_embeddings": 128
            }"#,
        )
        .unwrap();
        let header =
            r#"{"model.embed_tokens.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut checkpoint = (header.len() as u64).to_le_bytes().to_vec();
        checkpoint.extend_from_slice(header.as_bytes());
        checkpoint.extend_from_slice(&[1, 2, 3, 4]);
        std::fs::write(snapshot.join("model.safetensors"), checkpoint).unwrap();

        let group = create_driver_backend(
            &DriverOptions::Dummy {
                opts: DummyDriverOptions {
                    vocab_size: None,
                    arch_name: None,
                    ready_timeout: crate::config::Duration::from_secs(5),
                },
                random_seed: 7,
                activation_dtype: "f32".to_string(),
            },
            &snapshot,
            DESCRIPTOR,
            0,
            None,
            pie_driver_abi::ModelComponent::Full,
        )
        .unwrap();
        assert_eq!(group.caps.arch_name, "qwen3");
        assert_eq!(group.caps.vocab_size, 32);
        assert_eq!(group.caps.snapshot_dir, snapshot.display().to_string());
    }

    #[cfg(feature = "driver-cuda")]
    #[tokio::test]
    #[ignore = "requires PIE_TEST_GEMMA4_SNAPSHOT and a CUDA GPU"]
    async fn gemma4_encode_component_loads_and_encodes() {
        let snapshot = std::env::var_os("PIE_TEST_GEMMA4_SNAPSHOT")
            .map(PathBuf::from)
            .expect("set PIE_TEST_GEMMA4_SNAPSHOT");
        let options = DriverOptions::CudaNative(CudaNativeDriverOptions {
            device: "cuda:0".to_string(),
            gpu_mem_utilization: 0.5,
            verbose: true,
            ..Default::default()
        });
        let mut group = create_driver_backend(
            &options,
            &snapshot,
            // A test snapshot, not an artifact: no compiled descriptor.
            None,
            0,
            None,
            pie_driver_abi::ModelComponent::Encode,
        )
        .unwrap();
        assert!(group.caps.supports_media_encode);
        assert_eq!(group.caps.total_pages, 0);
        assert!(group.backend.export_kv_handle().is_none());

        let patch_count = 9usize;
        let pixel_bytes = patch_count * 3 * 16 * 16 * std::mem::size_of::<f32>();
        let hidden_size = group.caps.hidden_size;
        let make_encode = || {
            let mut patch_positions = Vec::with_capacity(patch_count * 2);
            for y in 0..3 {
                for x in 0..3 {
                    patch_positions.extend([x, y]);
                }
            }
            pie_driver_abi::MediaEncodePlan {
                image_grids: vec![1, 3, 3],
                image_pixels: vec![0; pixel_bytes],
                image_pixel_indptr: vec![0, pixel_bytes as u32],
                image_patch_positions: patch_positions,
                image_anchor_rows: vec![0],
                audio_features: Vec::new(),
                audio_feature_indptr: Vec::new(),
                audio_anchor_rows: Vec::new(),
                output_rows: vec![0; hidden_size as usize * 2],
                output_row_indptr: vec![0; 2],
            }
        };
        let audio_frames = 16usize;
        let audio_rows = 4usize;
        let make_audio = || pie_driver_abi::MediaEncodePlan {
            image_grids: Vec::new(),
            image_pixels: Vec::new(),
            image_pixel_indptr: Vec::new(),
            image_patch_positions: Vec::new(),
            image_anchor_rows: Vec::new(),
            audio_features: vec![0; audio_frames * 128 * std::mem::size_of::<f32>()],
            audio_feature_indptr: vec![0, (audio_frames * 128 * std::mem::size_of::<f32>()) as u32],
            audio_anchor_rows: vec![0],
            output_rows: vec![0; audio_rows * hidden_size as usize * 2],
            output_row_indptr: vec![0; 2],
        };
        let mut encode = make_encode();
        let completion = group.backend.encode(&mut encode).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(300), completion)
            .await
            .expect("encode completion timed out")
            .unwrap();
        assert_eq!(encode.output_row_indptr, vec![0, 1]);
        assert!(encode.output_rows.iter().any(|byte| *byte != 0));
        let tower_rows = encode.output_rows;
        let mut audio = make_audio();
        let completion = group.backend.encode(&mut audio).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(300), completion)
            .await
            .expect("audio encode completion timed out")
            .unwrap();
        assert_eq!(audio.output_row_indptr, vec![0, audio_rows as u32]);
        assert!(audio.output_rows.iter().any(|byte| *byte != 0));
        let tower_audio_rows = audio.output_rows;

        let model = pie_driver_abi::ModelIdentity {
            hash: [9; 32],
            component: pie_driver_abi::ModelComponent::Encode,
        };
        let layout = pie_driver_abi::KvLayout {
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            page_size: 0,
            dtype: pie_driver_abi::KvDtype::Bf16,
            kind: pie_driver_abi::KvLayoutKind::KvSeparate,
            storage_format: String::new(),
            region_page_bytes: Vec::new(),
        };
        let server = crate::executor::ExecutorServer::bind(
            "127.0.0.1:0",
            crate::translate::ModelDrivers {
                groups: vec![group],
            },
            model.clone(),
            1,
        )
        .await
        .unwrap();
        let client = crate::executor::connect(server.endpoint()).await.unwrap();
        let hello = client
            .execute(
                tarpc::context::current(),
                pie_driver_abi::ExecutorRequest::Hello(pie_driver_abi::HelloRequest {
                    wire_version: pie_driver_abi::REMOTE_WIRE_VERSION,
                    client_nonce: 1,
                    model,
                    kv_layout: layout,
                    peer_conn: None,
                }),
            )
            .await
            .unwrap()
            .unwrap();
        let pie_driver_abi::ExecutorResponse::Hello(hello) = hello else {
            panic!("executor Hello response");
        };
        assert_eq!(hello.grant.num_pages, 0);

        let image = make_encode();
        let response = client
            .execute(
                tarpc::context::current(),
                pie_driver_abi::ExecutorRequest::Encode(pie_driver_abi::RemoteEncode {
                    plan: pie_driver_abi::LaunchPlan {
                        token_ids: vec![0],
                        qo_indptr: vec![0, 1],
                        image_grids: image.image_grids,
                        image_pixels: image.image_pixels,
                        image_pixel_indptr: image.image_pixel_indptr,
                        image_patch_positions: image.image_patch_positions,
                        image_anchor_rows: image.image_anchor_rows,
                        ..Default::default()
                    },
                    blobs: Vec::new(),
                }),
            )
            .await
            .unwrap()
            .unwrap();
        let pie_driver_abi::ExecutorResponse::Embeddings(image) = response else {
            panic!("executor image response");
        };
        assert_eq!(image.rows, tower_rows);

        let audio = make_audio();
        let response = client
            .execute(
                tarpc::context::current(),
                pie_driver_abi::ExecutorRequest::Encode(pie_driver_abi::RemoteEncode {
                    plan: pie_driver_abi::LaunchPlan {
                        token_ids: vec![0; audio_rows],
                        qo_indptr: vec![0, audio_rows as u32],
                        audio_features: audio.audio_features,
                        audio_feature_indptr: audio.audio_feature_indptr,
                        audio_anchor_rows: audio.audio_anchor_rows,
                        ..Default::default()
                    },
                    blobs: Vec::new(),
                }),
            )
            .await
            .unwrap()
            .unwrap();
        let pie_driver_abi::ExecutorResponse::Embeddings(audio) = response else {
            panic!("executor audio response");
        };
        assert_eq!(audio.rows, tower_audio_rows);
        server.shutdown().await;

        let full_options = DriverOptions::CudaNative(CudaNativeDriverOptions {
            device: "cuda:0".to_string(),
            gpu_mem_utilization: 1.0,
            memory_profile: CudaMemoryProfile::Latency,
            total_pages: 1,
            ..Default::default()
        });
        let mut full = create_driver_backend(
            &full_options,
            &snapshot,
            None,
            1,
            None,
            pie_driver_abi::ModelComponent::Full,
        )
        .unwrap();
        assert!(full.caps.supports_media_encode);
        let mut inline = make_encode();
        let completion = full.backend.encode(&mut inline).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(300), completion)
            .await
            .expect("full-model encode completion timed out")
            .unwrap();
        assert_eq!(inline.output_row_indptr, vec![0, 1]);
        assert_eq!(inline.output_rows, tower_rows);
        let mut inline_audio = make_audio();
        let completion = full.backend.encode(&mut inline_audio).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(300), completion)
            .await
            .expect("full-model audio encode completion timed out")
            .unwrap();
        assert_eq!(inline_audio.output_row_indptr, vec![0, audio_rows as u32]);
        assert_eq!(inline_audio.output_rows, tower_audio_rows);
    }

    /// What a driver may be handed: an artifact, or a snapshot directory.
    ///
    /// This used to pin a GGUF-specific refusal, which existed because the
    /// LoadPlan executors could not decode GGUF's blocked schemes at load
    /// time. `pie model import` decodes them now, so a served model
    /// is a `.zt` whatever it started as, and the refusal has nothing left to
    /// name. A `.gguf` handed straight to `serve` is still rejected — as one
    /// of the things that is not an artifact, with the fix in the message.
    #[test]
    fn a_driver_takes_an_artifact_or_a_snapshot_and_nothing_else() {
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

    #[cfg(feature = "driver-cuda")]
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
        // The driver derives every disk cache from this. Without it the caches
        // fall back to XDG, which is what split pie's state across two roots.
        //
        // NOTE: this installs a process-global OnceLock that outlives the test,
        // so every later test in this binary sees `[cache]` emitted. Nothing
        // asserts its absence today; a test that needs it unset cannot share a
        // process with this one.
        set_cache_dir("/pie-home/cache".to_string());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("driver.toml");
        let snap = dir.path().join("snapshot");
        write_cuda_startup_toml(&out, &CudaNativeDriverOptions::default(), &snap, 0, None, DESCRIPTOR)
            .unwrap();
        let val: toml::Value = toml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(val["cache"]["dir"].as_str().unwrap(), "/pie-home/cache");
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
            std::fs::write(d.join("driver.toml"), "x").unwrap();
        }

        sweep_stale_launch_state();

        assert!(!dead.exists(), "a dead pid's state must be reclaimed");
        assert!(live.exists(), "the running process's own state must survive");
        assert!(
            foreign.exists(),
            "a directory that is not a pid is not ours to remove"
        );
    }

    /// A calibration request reaches the driver, and only ever from memory.
    ///
    /// This is the whole route that replaced `[driver] calibrate_planner`:
    /// `pie config tune` sets `server.calibrate_planner` on a config it
    /// derived, `engine::apply_embedded_calibration` puts it on the driver
    /// options, and this is where it becomes something the C++ side reads. The
    /// per-launch startup TOML is the only file it ever appears in, and that
    /// file is regenerated every boot -- so the request cannot outlive the boot
    /// that made it.
    #[test]
    fn a_calibration_request_reaches_the_driver_and_stops_there() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:0".to_string();
        opts.calibrate_planner = true;

        write_cuda_startup_toml(&out, &opts, &snap, 0, None, DESCRIPTOR).unwrap();
        let val: toml::Value = toml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(
            val["batching"]["calibrate_planner"].as_bool(),
            Some(true),
            "the driver never hears the request"
        );

        // And the field is not part of the file format: a user config that
        // spells it is refused, so this value can only have come from memory.
        let asked = "\
[model]
name = \"m\"
hf_repo = \"x\"
[driver]
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
    fn cuda_startup_toml_matches_driver_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:0".to_string();

        write_cuda_startup_toml(&out, &opts, &snap, 0, None, DESCRIPTOR).unwrap();

        // Re-parse the emitted TOML to confirm the schema the cuda
        // driver expects matches what we wrote (driver-side parsing
        // in driver/cuda/src/config.hpp).
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
        // Derived values are OMITTED, not written as a sentinel. The driver's
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
        // measurement, not a setting every startup file restates. The only
        // thing that ever turns it on is `pie config tune`, on a config it
        // derived in memory -- see `CudaNativeDriverOptions::calibrate_planner`
        // for why it cannot come from a file.
        assert!(val["batching"].get("calibrate_planner").is_none());
        assert_eq!(val["batching"].as_table().unwrap().len(), 4);
        assert_eq!(val["batching"]["swap_pool_size"].as_integer().unwrap(), 0);
        // Expert streaming is off unless an operator asks for it: for a model
        // that fits it is strictly slower, and it costs graph capture besides.
        assert_eq!(
            val["model"]["stream_routed_experts"].as_bool().unwrap(),
            false
        );
        assert!(val["model"].get("expert_cache_gb").is_none());
        assert!(val["model"].get("expert_host_cache_gb").is_none());
        assert_eq!(val["runtime"]["verbose"].as_bool().unwrap(), false);
    }

    /// Expert streaming is one decision, so it is one setting, and it has to
    /// reach both drivers by the same name.
    ///
    /// It did not. `[model].stream_routed_experts` was emitted for cuda only,
    /// and metal read `PIE_METAL_STREAM_EXPERTS` from the environment -- so
    /// setting the documented option on a Metal backend did nothing, and said
    /// nothing about doing nothing. The failure mode of a switch nobody wired
    /// up is silence, which is why it needs a test rather than a reading.
    #[test]
    fn metal_startup_toml_carries_expert_streaming() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = tmp.path().join("snap");

        let mut off = MetalDriverOptions::default();
        off.device = "metal:0".to_string();
        let out_off = tmp.path().join("off.toml");
        write_metal_startup_toml(&out_off, &off, &snap, 0, DESCRIPTOR).unwrap();
        let val: toml::Value = toml::from_str(&std::fs::read_to_string(&out_off).unwrap()).unwrap();
        assert_eq!(val["model"]["backend"].as_str().unwrap(), "metal:0");
        assert_eq!(
            val["model"]["stream_routed_experts"].as_bool().unwrap(),
            false,
            "streaming is off unless asked for: it trades resident memory for \
             page faults, which only pays when the weights do not fit"
        );

        let mut on = MetalDriverOptions::default();
        on.device = "metal:0".to_string();
        on.stream_routed_experts = true;
        let out_on = tmp.path().join("on.toml");
        write_metal_startup_toml(&out_on, &on, &snap, 0, DESCRIPTOR).unwrap();
        let val: toml::Value = toml::from_str(&std::fs::read_to_string(&out_on).unwrap()).unwrap();
        assert_eq!(
            val["model"]["stream_routed_experts"].as_bool().unwrap(),
            true,
            "the operator asked for streaming and the driver never heard about it"
        );
    }

    /// The bounded form of the same trade has to reach the driver too, and it
    /// is the one whose absence is hardest to notice: `expert_slab_bytes` is
    /// the only setting under which a checkpoint larger than the machine can
    /// be admitted, the C++ has read it since the slab landed, and no operator
    /// could say it -- it was reachable only from a test binary's environment.
    /// A model that does not fit then refuses to load with a message about
    /// arithmetic rather than about the switch that would have helped.
    ///
    /// Omitted and not zeroed when unset, because the driver already reads an
    /// absent key as "keep the whole bank resident".
    #[test]
    fn metal_startup_toml_carries_expert_slab_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = tmp.path().join("snap");

        let mut off = MetalDriverOptions::default();
        off.device = "metal:0".to_string();
        let out_off = tmp.path().join("off.toml");
        write_metal_startup_toml(&out_off, &off, &snap, 0, DESCRIPTOR).unwrap();
        let val: toml::Value = toml::from_str(&std::fs::read_to_string(&out_off).unwrap()).unwrap();
        assert!(
            val["model"].get("expert_slab_bytes").is_none(),
            "an unset budget is an absent key, not a zero: the driver's own \
             default is already the derivation"
        );

        let mut on = MetalDriverOptions::default();
        on.device = "metal:0".to_string();
        on.expert_slab_bytes = Some(2048 * 1024 * 1024);
        let out_on = tmp.path().join("on.toml");
        write_metal_startup_toml(&out_on, &on, &snap, 0, DESCRIPTOR).unwrap();
        let val: toml::Value = toml::from_str(&std::fs::read_to_string(&out_on).unwrap()).unwrap();
        assert_eq!(
            val["model"]["expert_slab_bytes"].as_integer().unwrap(),
            2048 * 1024 * 1024,
            "the operator capped the expert bank and the driver never heard about it"
        );
    }

    /// The option is spelled the same in the config an operator writes, not
    /// just in the file we generate. A metal `[model.driver.options]` block
    /// carrying it must parse -- `deny_unknown_fields` means a name that only
    /// cuda knows is a hard error, which is the good failure but not this one.
    #[test]
    fn metal_driver_options_accept_expert_streaming_by_the_cuda_name() {
        let parsed: MetalDriverOptions = toml::from_str("stream_routed_experts = true").unwrap();
        assert!(parsed.stream_routed_experts);

        let cuda: CudaNativeDriverOptions = toml::from_str("stream_routed_experts = true").unwrap();
        assert_eq!(
            parsed.stream_routed_experts, cuda.stream_routed_experts,
            "the two backends must answer to one name"
        );
    }

    #[test]
    fn cuda_startup_toml_emits_runtime_verbose_when_set() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:0".to_string();
        opts.verbose = true;

        write_cuda_startup_toml(&out, &opts, &snap, 0, None, DESCRIPTOR).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(val["runtime"]["verbose"].as_bool().unwrap(), true);
    }

    #[test]
    fn cuda_startup_toml_keeps_runtime_quant_out_of_driver_config() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:1".to_string();
        opts.runtime_quant = "fp8".to_string();

        write_cuda_startup_toml(&out, &opts, &snap, 3, None, DESCRIPTOR).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert!(val["model"].get("runtime_quant").is_none());
        assert_eq!(val["model"]["device"].as_str().unwrap(), "cuda:1");
    }

    #[test]
    fn cuda_startup_toml_keeps_mxfp4_policy_out_of_driver_config() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:0".to_string();
        opts.mxfp4_moe = "bf16".to_string();

        write_cuda_startup_toml(&out, &opts, &snap, 0, None, DESCRIPTOR).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert!(val["model"].get("mxfp4_moe").is_none());
    }

    #[test]
    fn cuda_startup_toml_emits_distributed_block_for_tp() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:1".to_string();
        let tp = TpLaunch {
            size: 2,
            rank: 1,
            nccl_unique_id_hex: "abcd".to_string(),
        };

        write_cuda_startup_toml(&out, &opts, &snap, 4, Some(&tp), DESCRIPTOR).unwrap();

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

    /// The compiled model config travels beside the startup TOML — always.
    ///
    /// This used to assert the other half too: that the key is *absent* for a
    /// snapshot, which is what let each driver keep a `config.json` parser for
    /// the absent case. `weights.rs` normalizes a snapshot into a descriptor
    /// now, so there is no absent case to pin and the parsers are gone. The
    /// writers still take the descriptor as an argument rather than deriving
    /// it from the path — lifting it is the resolver's job, done once — so
    /// this is about the *contract*, not about where the bytes came from.
    ///
    /// Both drivers take the same arrangement, so both are pinned here.
    #[test]
    fn the_startup_toml_always_carries_a_descriptor() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir.path().join("snap");
        std::fs::create_dir(&snapshot).unwrap();
        let body = br#"{"version":"pie.model/1","hidden_size":64}"#;

        let cuda = CudaNativeDriverOptions::default();
        let metal = MetalDriverOptions::default();

        let carried = |name: &str, write: &dyn Fn(&Path)| -> Option<Vec<u8>> {
            let out = dir.path().join(name).join("driver.toml");
            std::fs::create_dir_all(out.parent().unwrap()).unwrap();
            write(&out);
            let doc: toml::Value = toml::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
            doc["model"]
                .get("descriptor")
                .and_then(|v| v.as_str())
                .map(|path| std::fs::read(path).unwrap())
        };

        assert_eq!(
            carried("cuda", &|out| {
                write_cuda_startup_toml(out, &cuda, &snapshot, 0, None, body).unwrap()
            })
            .as_deref(),
            Some(body.as_slice())
        );
        assert_eq!(
            carried("metal", &|out| {
                write_metal_startup_toml(out, &metal, &snapshot, 0, body).unwrap()
            })
            .as_deref(),
            Some(body.as_slice())
        );
    }
}
