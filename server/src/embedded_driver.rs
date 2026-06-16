//! Embedded native driver: in-process supervisor for the C++ driver lib.
//!
//! Replaces the Python `pie_driver_portable.worker` for the standalone
//! path. The driver is no longer a subprocess — it runs as a thread
//! linked into our binary. We still preserve the shmem + control-socket
//! protocol so the runtime side (`pie::driver::*`) doesn't know the
//! difference between subprocess and embedded mode.
//!
//! This module exposes:
//!   * [`DriverCapabilities`] — typed view over the caps JSON the C
//!     entry hands back via the `ready_cb` callback.
//!   * [`write_startup_toml`] / [`write_cuda_startup_toml`] /
//!     [`write_dummy_startup_toml`] — emit the per-launch TOML each
//!     driver flavor reads on startup. Mirror
//!     `pie_driver_*.worker._write_startup_toml`.
//!   * [`EmbeddedDriver`] — owns the driver thread + its on-disk
//!     launch state, and bridges the C `ready_cb` to a Rust channel.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};

use crate::config::{
    CudaMemoryProfile, CudaNativeDriverOptions, DummyDriverOptions, PortableDriverOptions,
};
use crate::driver_ffi::{self, Flavor};

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

/// Per-flavor driver options, passed to [`EmbeddedDriver::start`] so the
/// caller doesn't have to discriminate on `DriverKind` in two places.
///
/// The `Dummy` variant carries `random_seed` and `activation_dtype`
/// alongside `DummyDriverOptions` because those are universal
/// `[model.driver]` fields, not options — and `write_dummy_startup_toml`
/// needs both to construct the caps payload.
///
/// `Clone` exists so `serve.rs` can rebuild a per-group variant
/// (different `device`) from a model-level template without
/// re-deserializing TOML.
#[derive(Clone)]
pub enum DriverOptions {
    #[cfg(feature = "driver-portable")]
    Portable(PortableDriverOptions),
    #[cfg(feature = "driver-cuda")]
    CudaNative(CudaNativeDriverOptions),
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
            #[cfg(feature = "driver-portable")]
            DriverOptions::Portable(_) => Flavor::Portable,
            #[cfg(feature = "driver-cuda")]
            DriverOptions::CudaNative(_) => Flavor::Cuda,
            DriverOptions::Dummy { .. } => Flavor::Dummy,
        }
    }
}

fn ready_timeout(options: &DriverOptions) -> Duration {
    let ready_timeout_s = match options {
        #[cfg(feature = "driver-portable")]
        DriverOptions::Portable(p) => p.ready_timeout_s,
        #[cfg(feature = "driver-cuda")]
        DriverOptions::CudaNative(opts) => opts.ready_timeout_s,
        DriverOptions::Dummy { opts, .. } => opts.ready_timeout_s,
    };
    Duration::from_secs_f64(ready_timeout_s.max(1.0))
}

#[derive(Clone)]
pub(crate) struct TpLaunch {
    size: usize,
    rank: usize,
    nccl_unique_id_hex: String,
}

#[derive(Clone, Copy)]
enum InProcCtxKind {
    Blocking,
    Polling,
}

unsafe fn release_inproc_ctx(kind: Option<InProcCtxKind>, ctx: *mut c_void) {
    if ctx.is_null() {
        return;
    }
    match kind {
        Some(InProcCtxKind::Blocking) => unsafe { pie::driver::InProcChannel::release(ctx) },
        Some(InProcCtxKind::Polling) => unsafe { pie::driver::InProcPollingChannel::release(ctx) },
        None => {}
    }
}

struct PendingEmbeddedDriver {
    flavor: Flavor,
    /// `/pie_shmem_g{idx}` for shmem-based drivers (dummy + subprocess
    /// Python); `None` for the inproc embedded drivers (cuda/portable)
    /// since they don't open a shmem region.
    shmem_name: Option<String>,
    thread: Option<JoinHandle<i32>>,
    caps_ctx: *mut CapsCtx,
    caps_rx: Option<mpsc::Receiver<String>>,
    state_dir: PathBuf,
    /// FFI ctx pointer from an in-process channel's `ffi_vtable`,
    /// or null for non-inproc paths. Released on `join`.
    inproc_ctx: *mut c_void,
    inproc_ctx_kind: Option<InProcCtxKind>,
}

impl PendingEmbeddedDriver {
    fn wait_for_caps(&mut self, timeout: Duration) -> Result<DriverCapabilities> {
        let caps_rx = self
            .caps_rx
            .take()
            .ok_or_else(|| anyhow!("internal error: caps receiver missing"))?;
        let json = match caps_rx.recv_timeout(timeout) {
            Ok(j) => j,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let thread = self.thread.take();
                let rc = thread.map(|h| h.join().unwrap_or(-1)).unwrap_or(-1);
                if !self.caps_ctx.is_null() {
                    unsafe { drop(Box::from_raw(self.caps_ctx)) };
                    self.caps_ctx = std::ptr::null_mut();
                }
                if !self.inproc_ctx.is_null() {
                    unsafe { release_inproc_ctx(self.inproc_ctx_kind, self.inproc_ctx) };
                    self.inproc_ctx = std::ptr::null_mut();
                    self.inproc_ctx_kind = None;
                }
                return Err(anyhow!(
                    "driver thread exited (rc={rc}) before emitting capabilities; \
                     check stderr for the [pie-driver-{}] error message",
                    self.flavor.as_str(),
                ));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(anyhow!(
                    "driver did not emit capabilities within {:.1}s",
                    timeout.as_secs_f64()
                ));
            }
        };
        parse_caps_json(&json)
    }

    fn into_driver(mut self, caps: DriverCapabilities) -> EmbeddedDriver {
        let inproc_ctx = std::mem::replace(&mut self.inproc_ctx, std::ptr::null_mut());
        let inproc_ctx_kind = self.inproc_ctx_kind.take();
        EmbeddedDriver {
            shmem_name: self.shmem_name.clone(),
            flavor: self.flavor,
            caps,
            thread: self.thread.take(),
            caps_ctx: self.caps_ctx,
            _state_dir: self.state_dir.clone(),
            inproc_ctx,
            inproc_ctx_kind,
        }
    }
}

impl Drop for PendingEmbeddedDriver {
    fn drop(&mut self) {
        if self.thread.is_none() {
            return;
        }
        let _ = self.thread.take();
        // Same safety tradeoff as EmbeddedDriver::drop: the detached
        // C thread may still call ready_cb, so leave caps_ctx alive.
    }
}

/// Per-DP-replica shmem name. Mirrors `runtime/src/device.rs::shmem_name`
/// (Python wrapper too — `_write_startup_toml(group_id=...)`).
pub fn shmem_name(group_id: usize) -> String {
    format!("/pie_shmem_g{group_id}")
}

/// `[shmem]` TOML block. `req_buf` is fixed at 4 MiB across drivers;
/// `resp_buf` varies — portable/dummy fit response payloads in 4 MiB,
/// but the cuda driver must hold a full-vocab `Sampler::Dist` (≈2.6
/// MiB on 150K-vocab models) plus per-request overhead and the
/// spec-mode multi-slot tail, so it uses 8 MiB. The runtime reads the
/// buffer sizes from the shmem header at attach time, so each driver
/// can pick its own size.
fn shmem_table(group_id: usize, resp_buf: usize, spin_budget_us: u64) -> toml::Table {
    let mut table = toml::Table::new();
    insert_str(&mut table, "name", shmem_name(group_id));
    insert_int(&mut table, "num_slots", 8);
    insert_int(&mut table, "req_buf", 4 * 1024 * 1024);
    insert_int(&mut table, "resp_buf", resp_buf as i64);
    insert_u64(&mut table, "spin_budget_us", spin_budget_us);
    table
}

fn insert_int(table: &mut toml::Table, key: &str, value: impl Into<i64>) {
    table.insert(key.into(), toml::Value::Integer(value.into()));
}

fn insert_str(table: &mut toml::Table, key: &str, value: impl Into<String>) {
    table.insert(key.into(), toml::Value::String(value.into()));
}

fn insert_u64(table: &mut toml::Table, key: &str, value: u64) {
    if let Ok(value) = i64::try_from(value) {
        insert_int(table, key, value);
    } else {
        insert_str(table, key, value.to_string());
    }
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

/// Default response buffer size for the dummy driver's shmem ring
/// (4 MiB). Embedded C++ drivers no longer use a shmem ring on the
/// forward path; only the dummy driver still does.
const SHMEM_RESP_BUF_DEFAULT: usize = 4 * 1024 * 1024;

/// Default per-launch state directory: `$PIE_HOME/standalone/<pid>/`.
/// We use a per-pid subdir so concurrent invocations of `pie` (rare
/// but legal — different ports) don't clobber each other's TOML or
/// aux sockets.
pub fn launch_state_dir() -> PathBuf {
    pie::path::get_pie_home()
        .join("standalone")
        .join(std::process::id().to_string())
}

// `DriverCapabilities` is owned by `pie-bridge` (single source of truth
// for the driver ↔ runtime interface). Re-exported here so existing call
// sites in pie-server keep working through the
// `embedded_driver::DriverCapabilities` path.
pub use pie_bridge::DriverCapabilities;

/// Parse a caps blob the C entry emits via `ready_cb` into the bridge's
/// struct. Lives in pie-server (not bridge) so bridge can stay free of a
/// serde_json dependency.
fn parse_caps_json(json: &str) -> Result<DriverCapabilities> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| anyhow::anyhow!("driver caps JSON parse: {e}"))?;
    serde_json::from_value(value).map_err(|e| anyhow::anyhow!("driver caps schema mismatch: {e}"))
}

/// Write the portable driver's startup TOML, returning the path the
/// driver should be invoked with. Mirrors the layout consumed by
/// `driver/portable/src/config.hpp`.
///
/// The driver consumes:
///   - `[model]` — local snapshot dir + GGML offload knobs.
///   - `[runtime]` — logging / diagnostic flags.
///
/// Cold-path RPC (page copies, adapter loads) is no longer wired
/// through this TOML — it now travels through direct `extern "C"`
/// calls into the portable lib, keyed by the `--driver-id` the host
/// passes on the command line.
pub fn write_startup_toml(
    out_path: &Path,
    options: &PortableDriverOptions,
    snapshot_dir: &Path,
    _group_id: usize,
) -> Result<()> {
    let mut doc = toml::Table::new();

    let mut model = toml::Table::new();
    insert_str(&mut model, "hf_path", path_string(snapshot_dir));
    insert_str(&mut model, "backend", &options.device);
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
    insert_table(&mut doc, "batching", batching);

    let mut runtime = toml::Table::new();
    insert_bool(&mut runtime, "verbose", options.verbose);
    insert_table(&mut doc, "runtime", runtime);

    write_toml_table(out_path, doc)
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
    // "Qwen3ForCausalLM" → "qwen3" — same heuristic the Python wrapper used.
    let raw_arch_lower = raw_arch.to_lowercase();
    let arch_name = raw_arch_lower
        .strip_suffix("forcausallm")
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

/// Write the dummy driver's startup TOML. Shape mirrors `driver/dummy/src/config.rs`:
/// `[shmem]` (same as portable) + `[dummy]` (knobs the dummy fabricates in
/// lieu of model introspection). The dummy ignores `[model]` / `[batching]`
/// so we omit them.
///
/// `vocab_size` and `arch_name` fall back to `<snapshot>/config.json`
/// when not explicitly set by the user, so a minimal `type = "dummy"`
/// config with just `device = ["cpu"]` works against any HF model.
pub fn write_dummy_startup_toml(
    out_path: &Path,
    opts: &DummyDriverOptions,
    snapshot_dir: &Path,
    random_seed: u64,
    activation_dtype: &str,
    group_id: usize,
    spin_budget_us: u64,
) -> Result<()> {
    let (vocab_size, arch_name, max_model_len) = match (opts.vocab_size, opts.arch_name.as_deref())
    {
        (Some(v), Some(a)) => {
            let (_, _, auto_len) =
                read_hf_config_defaults(snapshot_dir).unwrap_or_else(|_| (v, a.to_string(), 4096));
            (v, a.to_string(), auto_len)
        }
        (v_opt, a_opt) => {
            let (auto_v, auto_a, auto_len) =
                read_hf_config_defaults(snapshot_dir).with_context(|| {
                    format!(
                        "auto-discovering vocab_size + arch_name for dummy driver. \
                     Set them explicitly in [model.driver.options] to skip this lookup."
                    )
                })?;
            (
                v_opt.unwrap_or(auto_v),
                a_opt.map(str::to_string).unwrap_or(auto_a),
                auto_len,
            )
        }
    };

    let mut doc = toml::Table::new();
    insert_table(
        &mut doc,
        "shmem",
        shmem_table(group_id, SHMEM_RESP_BUF_DEFAULT, spin_budget_us),
    );

    let mut dummy = toml::Table::new();
    insert_int(&mut dummy, "vocab_size", vocab_size);
    insert_str(&mut dummy, "arch_name", arch_name);
    insert_int(&mut dummy, "max_model_len", max_model_len);
    insert_str(&mut dummy, "activation_dtype", activation_dtype);
    insert_int(&mut dummy, "random_seed", random_seed as i64);
    insert_str(&mut dummy, "snapshot_dir", path_string(snapshot_dir));
    insert_table(&mut doc, "dummy", dummy);

    write_toml_table(out_path, doc)
}

/// Write the cuda driver's startup TOML. Schema mirrors
/// `driver/cuda/src/config.hpp`: `[model]` with
/// `hf_repo`/`snapshot_dir`/`device`/`dtype`/optional load policy knobs,
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
) -> Result<()> {
    let mut doc = toml::Table::new();

    let mut model = toml::Table::new();
    insert_str(&mut model, "snapshot_dir", path_string(snapshot_dir));
    insert_str(&mut model, "device", &opts.device);
    insert_str(&mut model, "dtype", opts.weight_dtype.clone());
    if !opts.runtime_quant.is_empty() {
        insert_str(&mut model, "runtime_quant", opts.runtime_quant.clone());
    }
    if !opts.mxfp4_moe.is_empty() && opts.mxfp4_moe != "auto" {
        insert_str(&mut model, "mxfp4_moe", opts.mxfp4_moe.clone());
    }
    if !opts.mtp_assistant_snapshot_dir.is_empty() {
        insert_str(
            &mut model,
            "mtp_assistant_snapshot_dir",
            opts.mtp_assistant_snapshot_dir.clone(),
        );
    }
    insert_int(&mut model, "mtp_num_drafts", opts.mtp_num_drafts);
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
            CudaMemoryProfile::Balanced => "balanced",
            CudaMemoryProfile::Throughput => "throughput",
            CudaMemoryProfile::Capacity => "capacity",
        },
    );
    insert_int(&mut batching, "kv_page_size", opts.kv_page_size);
    insert_int(&mut batching, "swap_pool_size", opts.swap_pool_size);
    insert_str(&mut batching, "kv_cache_dtype", opts.kv_cache_dtype.clone());
    insert_table(&mut doc, "batching", batching);

    let mut runtime = toml::Table::new();
    insert_bool(&mut runtime, "verbose", opts.verbose);
    insert_table(&mut doc, "runtime", runtime);

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
// EmbeddedDriver — thread that runs the in-process C++ driver entry.
// -----------------------------------------------------------------------------
//
// The C++ entry hands caps back via a `ready_cb(*const c_char, *mut c_void)`
// callback. We pass each launch its own `Box<CapsCtx>` as `ready_ctx`,
// reaching it back via raw pointer in the callback — that lets multiple
// driver threads run concurrently (one per DP replica) without colliding
// on a process-global slot.
//
// Lifetime: the box is leaked into C land for the driver thread's full
// duration, and reclaimed via `Box::from_raw` in [`EmbeddedDriver::join`].

/// Per-launch callback target. Held alive by the parent `EmbeddedDriver`
/// for the whole lifetime of the driver thread; the thread reads from
/// it through a raw pointer.
struct CapsCtx {
    tx: Mutex<Option<mpsc::SyncSender<String>>>,
}

/// Dispatch into the right driver entry: in-process if a vtable was
/// supplied (cuda_native leader), shmem otherwise. Runs on the spawned
/// driver thread; returns the driver's process-style rc.
fn run_driver(
    flavor: Flavor,
    argc: c_int,
    argv: *mut *mut c_char,
    cb: driver_ffi::ReadyCb,
    ctx: *mut c_void,
    inproc_vtable: Option<pie::driver::InProcVTable>,
) -> i32 {
    if let Some(vtable) = inproc_vtable {
        return unsafe { driver_ffi::run_inproc(flavor, argc, argv, 0, cb, ctx, vtable) }
            .unwrap_or(-1);
    }
    unsafe {
        driver_ffi::run(
            flavor, argc, argv, /*install_signal_handlers=*/ 0, cb, ctx,
        )
    }
}

unsafe extern "C" fn embedded_caps_cb(caps_json: *const c_char, ctx: *mut c_void) {
    if ctx.is_null() {
        return;
    }
    let json = unsafe { CStr::from_ptr(caps_json) }
        .to_string_lossy()
        .into_owned();
    let ctx_ref = unsafe { &*(ctx as *const CapsCtx) };
    if let Some(tx) = ctx_ref.tx.lock().unwrap().take() {
        let _ = tx.send(json);
    }
}

/// Owns the embedded driver thread + its on-disk launch state. The
/// driver runs its `pie_driver_<flavor>_run*` entry to completion on
/// its thread; [`Self::request_stop`] signals the C++ `DriverServer`
/// to exit and [`Self::join`] waits for the thread to land.
pub struct EmbeddedDriver {
    pub caps: DriverCapabilities,
    /// Which compiled driver flavor backs this instance. `request_stop`
    /// dispatches via this so heterogeneous configs (e.g. one cuda +
    /// one portable model) signal the right entry.
    pub flavor: Flavor,
    /// Shared-memory region the driver owns (e.g. `/pie_shmem_g0`).
    /// Unix builds `shm_unlink` this on shutdown to clean up after a
    /// hard kill that bypassed the driver's own teardown. `None` when
    /// the driver runs on the in-process channel (cuda / portable).
    pub shmem_name: Option<String>,
    thread: Option<JoinHandle<i32>>,
    /// Raw pointer to the leaked `Box<CapsCtx>`. The C driver thread
    /// dereferences this via `embedded_caps_cb`, so it must outlive the
    /// thread; reclaimed in [`Self::join`].
    caps_ctx: *mut CapsCtx,
    _state_dir: PathBuf,
    /// FFI ctx pointer from an in-process channel's `ffi_vtable`,
    /// or null for non-inproc paths. Released on `join`.
    inproc_ctx: *mut c_void,
    inproc_ctx_kind: Option<InProcCtxKind>,
}

// `caps_ctx: *mut CapsCtx` is the only non-Send field. It points to a
// heap-allocated `CapsCtx { tx: Mutex<...> }` we own exclusively; the
// only other reader is the C driver thread, which is joined before
// `caps_ctx` is freed. So crossing thread boundaries is sound.
unsafe impl Send for EmbeddedDriver {}

impl EmbeddedDriver {
    /// Spawn the driver thread for the given model snapshot. Blocks
    /// until the driver emits caps via its `ready_cb` (or the
    /// `ready_timeout_s` from `options` elapses).
    ///
    /// `snapshot_dir` must be a local directory containing an HF-style
    /// model checkpoint (config.json, *.safetensors, tokenizer.json).
    pub fn start(
        options: &DriverOptions,
        snapshot_dir: &Path,
        group_id: usize,
        use_inproc_polling_channel: bool,
        spin_budget_us: u64,
    ) -> Result<Self> {
        let mut pending = Self::spawn_rank(
            options,
            snapshot_dir,
            group_id,
            None,
            use_inproc_polling_channel,
            spin_budget_us,
        )?;
        let timeout = ready_timeout(options);
        let caps = pending.wait_for_caps(timeout)?;
        Ok(pending.into_driver(caps))
    }

    /// Spawn one embedded cuda TP group. Rank 0 owns the shmem server and
    /// emits capabilities; followers only participate in NCCL collectives.
    #[cfg(feature = "driver-cuda")]
    pub fn start_cuda_tp_group(
        options_by_rank: &[DriverOptions],
        snapshot_dir: &Path,
        group_id: usize,
        use_inproc_polling_channel: bool,
        spin_budget_us: u64,
    ) -> Result<Vec<Self>> {
        if options_by_rank.is_empty() {
            return Err(anyhow!("cuda TP group must contain at least one rank"));
        }
        if options_by_rank.len() == 1 {
            return Ok(vec![Self::start(
                &options_by_rank[0],
                snapshot_dir,
                group_id,
                use_inproc_polling_channel,
                spin_budget_us,
            )?]);
        }
        for opt in options_by_rank {
            if !matches!(opt, DriverOptions::CudaNative(_)) {
                return Err(anyhow!(
                    "start_cuda_tp_group only accepts cuda_native options"
                ));
            }
        }

        let uid = nccl_unique_id_hex()?;
        let tp_size = options_by_rank.len();
        let mut pending = Vec::with_capacity(tp_size);
        for (rank, opt) in options_by_rank.iter().enumerate() {
            pending.push(Self::spawn_rank(
                opt,
                snapshot_dir,
                group_id,
                Some(TpLaunch {
                    size: tp_size,
                    rank,
                    nccl_unique_id_hex: uid.clone(),
                }),
                use_inproc_polling_channel,
                spin_budget_us,
            )?);
        }

        let timeout = ready_timeout(&options_by_rank[0]);
        let leader_caps = pending[0].wait_for_caps(timeout)?;
        let mut drivers = Vec::with_capacity(tp_size);
        for p in pending.into_iter() {
            // Every rank in this DP-group sees the same caps. Embedded
            // drivers (cuda / portable) leave shmem_name = None — there
            // is no shmem region for either leader or followers.
            drivers.push(p.into_driver(leader_caps.clone()));
        }
        Ok(drivers)
    }

    fn spawn_rank(
        options: &DriverOptions,
        snapshot_dir: &Path,
        group_id: usize,
        tp: Option<TpLaunch>,
        use_inproc_polling_channel: bool,
        spin_budget_us: u64,
    ) -> Result<PendingEmbeddedDriver> {
        // Portable allows either a safetensors dir or a single .gguf file
        // (model.cpp branches on is_regular_file). CUDA + dummy still need
        // a directory — those drivers fail later with their own message
        // when handed a file, so we don't gate that here.
        let is_gguf_file = snapshot_dir.is_file()
            && snapshot_dir.extension().is_some_and(|e| e == "gguf");
        if !snapshot_dir.is_dir() && !is_gguf_file {
            return Err(anyhow!(
                "snapshot_dir {snapshot_dir:?} does not exist, or is not a directory or .gguf file"
            ));
        }

        let rank_suffix = tp
            .as_ref()
            .map(|tp| format!("-r{}", tp.rank))
            .unwrap_or_default();
        let state_dir = launch_state_dir().join(format!("g{group_id}{rank_suffix}"));
        std::fs::create_dir_all(&state_dir)
            .map_err(|e| anyhow!("create state dir {state_dir:?}: {e}"))?;

        let toml_path = state_dir.join("driver.toml");
        let flavor = options.flavor();
        match options {
            #[cfg(feature = "driver-portable")]
            DriverOptions::Portable(p) => {
                write_startup_toml(&toml_path, p, snapshot_dir, group_id)?;
            }
            #[cfg(feature = "driver-cuda")]
            DriverOptions::CudaNative(opts) => {
                write_cuda_startup_toml(&toml_path, opts, snapshot_dir, group_id, tp.as_ref())?;
            }
            DriverOptions::Dummy {
                opts,
                random_seed,
                activation_dtype,
            } => {
                write_dummy_startup_toml(
                    &toml_path,
                    opts,
                    snapshot_dir,
                    *random_seed,
                    activation_dtype,
                    group_id,
                    spin_budget_us,
                )?;
            }
        }

        // Per-launch caps box, leaked into C as ready_ctx and reclaimed
        // in `join()`. Each driver thread carries its own pointer, so
        // multiple `start()` calls can run concurrently — which DP > 1
        // exercises (one driver thread per replica).
        let (caps_tx, caps_rx) = mpsc::sync_channel::<String>(1);
        let caps_ctx = Box::into_raw(Box::new(CapsCtx {
            tx: Mutex::new(Some(caps_tx)),
        }));

        let toml_path_str = toml_path
            .to_str()
            .ok_or_else(|| {
                // SAFETY: we just allocated; nothing else points to it.
                unsafe { drop(Box::from_raw(caps_ctx)) };
                anyhow!("toml path is not valid UTF-8: {toml_path:?}")
            })?
            .to_owned();

        // Both cuda_native and portable drive their C++ engines via the
        // in-process channel: the runtime owns the channel, the C++
        // driver thread consumes via FFI callbacks. TP followers
        // (cuda rank > 0) still receive a vtable but never use it — the
        // C++ side detects follower rank and goes into tp_follower_serve
        // (NCCL broadcasts) instead of serve_forever, so the channel
        // for a follower stays untouched.
        let tp_is_leader = tp.as_ref().map(|t| t.rank == 0).unwrap_or(true);
        let mut inproc_vtable: Option<pie::driver::InProcVTable> = None;
        let mut inproc_ctx: *mut c_void = std::ptr::null_mut();
        let mut inproc_ctx_kind: Option<InProcCtxKind> = None;
        let uses_inproc = match flavor {
            #[cfg(feature = "driver-cuda")]
            Flavor::Cuda => true,
            #[cfg(feature = "driver-portable")]
            Flavor::Portable => true,
            _ => false,
        };
        if uses_inproc {
            let vtable = if use_inproc_polling_channel {
                let channel = pie::driver::InProcPollingChannel::with_spin_budget(spin_budget_us)?;
                let vtable = channel.ffi_vtable();
                inproc_ctx_kind = Some(InProcCtxKind::Polling);
                if tp_is_leader {
                    pie::driver::install_channel(group_id, Arc::new(channel));
                }
                vtable
            } else {
                let channel = pie::driver::InProcChannel::with_spin_budget(spin_budget_us);
                let vtable = channel.ffi_vtable();
                inproc_ctx_kind = Some(InProcCtxKind::Blocking);
                if tp_is_leader {
                    pie::driver::install_channel(group_id, Arc::new(channel));
                }
                vtable
            };
            inproc_ctx = vtable.ctx;
            inproc_vtable = Some(vtable);
        }

        // Cast through usize so the raw pointer crosses the thread
        // boundary — `*mut c_void` isn't `Send`, but the underlying
        // address is just a number we promise (above, in `unsafe impl
        // Send for EmbeddedDriver`) that nobody else writes through
        // until the thread is joined.
        let caps_ctx_addr = caps_ctx as usize;
        let thread_name = tp
            .as_ref()
            .map(|tp| format!("pie-driver-{}-g{group_id}-r{}", flavor.as_str(), tp.rank))
            .unwrap_or_else(|| format!("pie-driver-{}-g{group_id}", flavor.as_str()));
        let thread = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || -> i32 {
                // Argv storage lives on the thread's stack for the
                // entire run — safe to hand its raw pointers to the
                // FFI entry. The group_id is already plumbed through
                // the in-process vtable (`install_channel(group_id, …)`
                // above) so the C++ backends don't need it on argv;
                // their CLI parsers reject any flag they don't define.
                let argv_owned: Vec<CString> = vec![
                    CString::new(flavor.argv0()).unwrap(),
                    CString::new("--config").unwrap(),
                    CString::new(toml_path_str).unwrap(),
                ];
                let mut argv_ptrs: Vec<*mut c_char> = argv_owned
                    .iter()
                    .map(|s| s.as_ptr() as *mut c_char)
                    .collect();
                run_driver(
                    flavor,
                    argv_ptrs.len() as c_int,
                    argv_ptrs.as_mut_ptr(),
                    embedded_caps_cb,
                    caps_ctx_addr as *mut c_void,
                    inproc_vtable,
                )
            })
            .map_err(|e| {
                // SAFETY: thread didn't start; we still hold the only
                // reference to caps_ctx, and the inproc channel (if any)
                // remains installed in CHANNELS — drop our local handle.
                unsafe { drop(Box::from_raw(caps_ctx)) };
                if !inproc_ctx.is_null() {
                    unsafe { release_inproc_ctx(inproc_ctx_kind, inproc_ctx) };
                }
                anyhow!("spawn pie-driver thread: {e}")
            })?;

        // Only shmem-based flavors (dummy) own a runtime-visible region;
        // embedded cuda/portable drivers leave the name as None.
        let shmem_name = if matches!(flavor, Flavor::Dummy) {
            Some(shmem_name(group_id))
        } else {
            None
        };

        Ok(PendingEmbeddedDriver {
            shmem_name,
            flavor,
            thread: Some(thread),
            caps_ctx,
            caps_rx: Some(caps_rx),
            state_dir,
            inproc_ctx,
            inproc_ctx_kind,
        })
    }

    /// Signal the driver thread's serve loop to exit. Idempotent; safe
    /// to call from any thread. After this returns the thread will
    /// finish whatever request it's mid-flight on, then exit naturally.
    /// Pair with [`Self::join`] to block until exit.
    pub fn request_stop(&self) {
        driver_ffi::request_stop(self.flavor);
    }

    /// Non-consuming liveness check — true if the driver thread has
    /// exited (cleanly or otherwise). Used by [`serve::run_async`]'s
    /// watchdog to detect a driver dying mid-serve and trigger orderly
    /// shutdown.
    pub fn is_finished(&self) -> bool {
        self.thread
            .as_ref()
            .map(|h| h.is_finished())
            .unwrap_or(true)
    }

    /// Wait for the driver thread to exit, returning its rc. Caller
    /// should normally call [`Self::request_stop`] first; otherwise
    /// the driver will block forever in its serve loop.
    ///
    /// On return the per-launch caps box is freed — safe because the
    /// thread is no longer running.
    pub fn join(mut self) -> i32 {
        let rc = match self.thread.take().map(|h| h.join()) {
            Some(Ok(rc)) => rc,
            Some(Err(_)) => -1,
            None => 0,
        };
        // SAFETY: thread is joined; nobody else holds caps_ctx or inproc_ctx.
        if !self.caps_ctx.is_null() {
            unsafe { drop(Box::from_raw(self.caps_ctx)) };
            self.caps_ctx = std::ptr::null_mut();
        }
        if !self.inproc_ctx.is_null() {
            unsafe { release_inproc_ctx(self.inproc_ctx_kind, self.inproc_ctx) };
            self.inproc_ctx = std::ptr::null_mut();
            self.inproc_ctx_kind = None;
        }
        rc
    }
}

impl Drop for EmbeddedDriver {
    /// Safety net for the "caller forgot to call `join`" path. We can't
    /// reclaim `caps_ctx` here — the C driver thread might still be
    /// running and reading through that pointer (callback can fire
    /// arbitrarily late if the driver re-emits readiness on reload).
    /// Leak intentionally; the OS reclaims on process exit. The
    /// production path in `serve.rs` always pairs `request_stop` +
    /// `join`, which takes the safe branch above.
    fn drop(&mut self) {
        if self.thread.is_none() {
            // join() already ran; nothing to do.
            return;
        }
        // Detach the thread (drop JoinHandle without joining); the
        // thread runs until the process exits. Don't touch caps_ctx —
        // it stays leaked but reachable.
        let _ = self.thread.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shmem_name_format() {
        assert_eq!(shmem_name(0), "/pie_shmem_g0");
        assert_eq!(shmem_name(7), "/pie_shmem_g7");
    }

    #[test]
    fn caps_json_round_trips() {
        let json = r#"{
            "total_pages": 1024,
            "kv_page_size": 32,
            "swap_pool_size": 0,
            "max_forward_tokens": 4096,
            "max_forward_requests": 512,
            "max_page_refs": 262144,
            "max_logit_rows": 4096,
            "max_prob_rows": 4096,
            "max_custom_mask_bytes": 8388608,
            "max_sampler_rows": 4096,
            "max_logprob_labels": 4096,
            "arch_name": "qwen3",
            "vocab_size": 151936,
            "max_model_len": 4096,
            "activation_dtype": "bfloat16",
            "snapshot_dir": "/tmp/snap",
            "shmem_name": "/pie_shmem_g0"
        }"#;
        let caps = parse_caps_json(json).unwrap();
        assert_eq!(caps.total_pages, 1024);
        assert_eq!(caps.arch_name, "qwen3");
        assert_eq!(caps.shmem_name.as_deref(), Some("/pie_shmem_g0"));
        assert_eq!(caps.max_forward_tokens, 4096);
        assert_eq!(caps.max_page_refs, 262144);
    }

    #[test]
    fn dummy_startup_toml_writes_spin_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("dummy.toml");
        let snap = tmp.path().join("snap");
        let opts = DummyDriverOptions {
            vocab_size: Some(32000),
            arch_name: Some("qwen3".to_string()),
            ready_timeout_s: 5.0,
        };

        write_dummy_startup_toml(&out, &opts, &snap, 42, "bfloat16", 0, u64::MAX).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(
            val["shmem"]["spin_budget_us"].as_str(),
            Some("18446744073709551615"),
        );
        assert!(val["dummy"].get("max_num_kv_pages").is_none());
        assert!(val["dummy"].get("kv_page_size").is_none());
        assert!(val["dummy"].get("max_forward_tokens").is_none());
        assert!(val["dummy"].get("max_forward_requests").is_none());
    }

    #[test]
    fn startup_toml_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("startup.toml");
        let snap = tmp.path().join("snap");
        let opts = PortableDriverOptions::default();

        write_startup_toml(&out, &opts, &snap, 0).unwrap();

        // Re-parse the emitted TOML to confirm the schema the driver
        // expects matches what we wrote (driver-side parsing in
        // driver/portable/src/config.hpp uses the same structure).
        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert!(
            val.get("shmem").is_none(),
            "portable no longer emits [shmem]"
        );
        assert!(
            val.get("aux_ipc").is_none(),
            "portable no longer emits [aux_ipc]"
        );
        assert_eq!(val["batching"]["kv_page_size"].as_integer().unwrap(), 32);
        assert_eq!(val["batching"]["total_pages"].as_integer().unwrap(), 1024);
        assert!(val["batching"].get("max_num_kv_pages").is_none());
        assert_eq!(val["batching"]["kv_cache_dtype"].as_str().unwrap(), "auto");
        assert_eq!(
            val["model"]["hf_path"].as_str().unwrap(),
            snap.to_str().unwrap()
        );
        assert_eq!(val["model"]["backend"].as_str().unwrap(), "auto");
    }

    #[test]
    fn cuda_startup_toml_matches_driver_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:0".to_string();

        write_cuda_startup_toml(&out, &opts, &snap, 0, None).unwrap();

        // Re-parse the emitted TOML to confirm the schema the cuda
        // driver expects matches what we wrote (driver-side parsing
        // in driver/cuda/src/config.hpp).
        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert!(val.get("shmem").is_none(), "cuda no longer emits [shmem]");
        assert!(
            val["model"].get("hf_repo").is_none(),
            "cuda derives from snapshot_dir"
        );
        assert_eq!(
            val["model"]["snapshot_dir"].as_str().unwrap(),
            snap.to_str().unwrap()
        );
        assert_eq!(val["model"]["device"].as_str().unwrap(), "cuda:0");
        assert_eq!(val["model"]["dtype"].as_str().unwrap(), "bfloat16");
        assert!(val["model"].get("runtime_quant").is_none()); // omitted when empty
        assert_eq!(val["batching"]["kv_page_size"].as_integer().unwrap(), 32);
        assert_eq!(val["batching"]["kv_cache_dtype"].as_str().unwrap(), "auto");
        assert_eq!(
            val["batching"]["gpu_mem_utilization"].as_float().unwrap(),
            0.90
        );
        assert_eq!(val["batching"]["memory_profile"].as_str().unwrap(), "auto");
        assert_eq!(val["batching"].as_table().unwrap().len(), 5);
        assert_eq!(val["batching"]["swap_pool_size"].as_integer().unwrap(), 0);
        assert_eq!(val["runtime"]["verbose"].as_bool().unwrap(), false);
    }

    #[test]
    fn cuda_startup_toml_emits_runtime_verbose_when_set() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:0".to_string();
        opts.verbose = true;

        write_cuda_startup_toml(&out, &opts, &snap, 0, None).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(val["runtime"]["verbose"].as_bool().unwrap(), true);
    }

    #[test]
    fn cuda_startup_toml_emits_runtime_quant_when_set() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:1".to_string();
        opts.runtime_quant = "fp8".to_string();

        write_cuda_startup_toml(&out, &opts, &snap, 3, None).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(val["model"]["runtime_quant"].as_str().unwrap(), "fp8");
        assert_eq!(val["model"]["device"].as_str().unwrap(), "cuda:1");
    }

    #[test]
    fn cuda_startup_toml_emits_mxfp4_policy_when_non_default() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("cuda.toml");
        let snap = tmp.path().join("snap");
        let mut opts = CudaNativeDriverOptions::default();
        opts.device = "cuda:0".to_string();
        opts.mxfp4_moe = "bf16".to_string();

        write_cuda_startup_toml(&out, &opts, &snap, 0, None).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(val["model"]["mxfp4_moe"].as_str().unwrap(), "bf16");
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

        write_cuda_startup_toml(&out, &opts, &snap, 4, Some(&tp)).unwrap();

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
}
