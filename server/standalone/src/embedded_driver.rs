//! Embedded native driver: in-process supervisor for the C++ driver lib.
//!
//! Replaces the Python `pie_driver_portable.worker` for the standalone
//! path. The driver is no longer a subprocess — it runs as a thread
//! linked into our binary. We still preserve the shmem + control-socket
//! protocol so the runtime side (`pie::device::*`) doesn't know the
//! difference between subprocess and embedded mode.
//!
//! At M2.3 this module exposes:
//!   * [`DriverCapabilities`] — typed view over the caps JSON the C
//!     entry hands back via the `ready_cb` callback.
//!   * [`write_startup_toml`] — emits the per-launch TOML the driver
//!     reads on startup. Mirrors
//!     `pie_driver_portable.worker._write_startup_toml`.
//!
//! The thread-spawn + caps-channel plumbing lands in M2.4 alongside the
//! `bootstrap::Config` translation.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Result, anyhow};
use serde::Deserialize;

use crate::config::{DummyDriverOptions, PortableDriverOptions};
use crate::driver_ffi;

/// Per-flavor driver options, passed to [`EmbeddedDriver::start`] so the
/// caller doesn't have to discriminate on `DriverKind` in two places.
///
/// The `Dummy` variant carries `random_seed` and `activation_dtype`
/// alongside `DummyDriverOptions` because those are universal
/// `[model.driver]` fields, not options — and `write_dummy_startup_toml`
/// needs both to construct the caps payload.
pub enum DriverOptions {
    Portable(PortableDriverOptions),
    Dummy {
        opts: DummyDriverOptions,
        random_seed: u64,
        activation_dtype: String,
    },
}

/// Per-DP-replica shmem name. Mirrors `runtime/src/device.rs::shmem_name`
/// (Python wrapper too — `_write_startup_toml(group_id=...)`).
pub fn shmem_name(group_id: usize) -> String {
    format!("/pie_shmem_g{group_id}")
}

/// `[shmem]` TOML block — req/resp sizes must match the runtime's
/// `SHMEM_*` constants in `runtime/src/device.rs` exactly. Both
/// startup-TOML writers below splice this in.
fn render_shmem_block(group_id: usize) -> String {
    format!(
        "[shmem]\n\
name = \"{}\"\n\
num_slots = 8\n\
req_buf = 4194304\n\
resp_buf = 4194304\n\
spin_us = 0\n",
        shmem_name(group_id),
    )
}

/// Default per-launch state directory: `$PIE_HOME/standalone/<pid>/`.
/// We use a per-pid subdir so concurrent invocations of `pie` (rare
/// but legal — different ports) don't clobber each other's TOML or
/// aux sockets.
pub fn launch_state_dir() -> PathBuf {
    pie::path::get_pie_home()
        .join("standalone")
        .join(std::process::id().to_string())
}

/// Caps JSON the C entry emits via `ready_cb`. Schema mirrors
/// `driver/portable/src/entry.cpp` (search for `nlohmann::json caps`)
/// and the Python `DriverCapabilities` dataclass.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // RPC schema fields read by clients, not the host process.
pub struct DriverCapabilities {
    pub total_pages: u32,
    pub kv_page_size: u32,
    pub swap_pool_size: u32,
    pub max_batch_tokens: u32,
    pub max_batch_size: u32,
    pub arch_name: String,
    pub vocab_size: u32,
    pub max_model_len: u32,
    pub activation_dtype: String,
    pub snapshot_dir: String,
    pub shmem_name: String,
}

impl DriverCapabilities {
    pub fn from_json(json: &str) -> Result<Self> {
        // toml crate also handles JSON-shaped values via serde, but using
        // serde_json here would pull a new dep. The caps payload is
        // small and JSON-safe, so we route through `toml::Value` after
        // a crude JSON-to-TOML adapter via... actually `serde_json` is
        // in the workspace already (runtime depends on it transitively
        // is irrelevant here). For now, use the runtime's existing
        // `serde_json` re-export, falling back to a tiny dependency
        // bump if needed.
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("driver caps JSON parse: {e}"))?;
        serde_json::from_value(value)
            .map_err(|e| anyhow::anyhow!("driver caps schema mismatch: {e}"))
    }
}

/// Write the driver's startup TOML, returning the path the driver
/// should be invoked with. Mirrors
/// `pie_driver_portable.worker._write_startup_toml`.
///
/// The driver consumes:
///   - `[shmem]` — channel name + buffer sizes (must match
///     `runtime/src/device.rs` SHMEM_* constants exactly; req/resp
///     size mismatches silently break the channel).
///   - `[model]` — local snapshot dir + GGML offload knobs.
///   - `[batching]` — KV page geometry + per-batch budgets.
///   - `[aux_ipc]` — unix socket the driver listens on for
///     control-plane page-copy commands. Empty path disables.
pub fn write_startup_toml(
    out_path: &Path,
    options: &PortableDriverOptions,
    snapshot_dir: &Path,
    aux_socket_path: &Path,
    group_id: usize,
) -> Result<()> {
    let toml = format!(
        "# Auto-generated by pie-standalone; do not edit.\n\
\n\
{shmem_block}\n\
[model]\n\
hf_path = \"{snapshot}\"\n\
n_gpu_layers = {n_gpu_layers}\n\
n_ctx = {n_ctx}\n\
\n\
[batching]\n\
kv_page_size = {kv_page_size}\n\
max_num_kv_pages = {max_num_kv_pages}\n\
max_batch_tokens = {max_batch_tokens}\n\
max_batch_size = {max_batch_size}\n\
cpu_pages = {cpu_pages}\n\
\n\
[aux_ipc]\n\
socket_path = \"{aux_socket}\"\n",
        shmem_block = render_shmem_block(group_id),
        snapshot = snapshot_dir.display(),
        n_gpu_layers = options.n_gpu_layers,
        n_ctx = options.n_ctx,
        kv_page_size = options.kv_page_size,
        max_num_kv_pages = options.max_num_kv_pages,
        max_batch_tokens = options.max_batch_tokens,
        max_batch_size = options.max_batch_size,
        cpu_pages = options.cpu_pages,
        aux_socket = aux_socket_path.display(),
    );

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create startup toml dir {parent:?}: {e}"))?;
    }
    std::fs::write(out_path, toml)
        .map_err(|e| anyhow!("write startup toml {out_path:?}: {e}"))?;
    Ok(())
}

/// Write the dummy driver's startup TOML. Shape mirrors `driver/dummy/src/config.rs`:
/// `[shmem]` (same as portable) + `[dummy]` (knobs the dummy fabricates in
/// lieu of model introspection). The dummy ignores `[model]`, `[batching]`,
/// and `[aux_ipc]` so we omit them.
pub fn write_dummy_startup_toml(
    out_path: &Path,
    opts: &DummyDriverOptions,
    snapshot_dir: &Path,
    random_seed: u64,
    activation_dtype: &str,
    group_id: usize,
) -> Result<()> {
    let toml = format!(
        "# Auto-generated by pie-standalone; do not edit.\n\
\n\
{shmem_block}\n\
[dummy]\n\
kv_page_size = {kv_page_size}\n\
max_num_kv_pages = {max_num_kv_pages}\n\
max_batch_tokens = {max_batch_tokens}\n\
max_batch_size = {max_batch_size}\n\
vocab_size = {vocab_size}\n\
arch_name = \"{arch_name}\"\n\
max_model_len = {max_model_len}\n\
activation_dtype = \"{activation_dtype}\"\n\
random_seed = {random_seed}\n\
snapshot_dir = \"{snapshot}\"\n",
        shmem_block = render_shmem_block(group_id),
        kv_page_size = opts.kv_page_size,
        max_num_kv_pages = opts.max_num_kv_pages,
        max_batch_tokens = opts.max_batch_tokens,
        max_batch_size = opts.max_batch_size,
        vocab_size = opts.vocab_size,
        arch_name = opts.arch_name,
        max_model_len = opts.max_model_len,
        snapshot = snapshot_dir.display(),
    );

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create startup toml dir {parent:?}: {e}"))?;
    }
    std::fs::write(out_path, toml)
        .map_err(|e| anyhow!("write startup toml {out_path:?}: {e}"))?;
    Ok(())
}

// -----------------------------------------------------------------------------
// EmbeddedDriver — thread that runs the in-process C++ driver entry.
// -----------------------------------------------------------------------------
//
// The C++ entry hands caps back via a callback. Bridging that to a
// Rust channel needs an `extern "C" fn`, which can't capture state, so
// we route through a process-global slot that holds a one-shot
// `SyncSender<String>` for the current launch. v0 supports a single
// embedded driver instance per process — multi-replica is post-v1.

fn caps_slot() -> &'static Mutex<Option<mpsc::SyncSender<String>>> {
    static SLOT: OnceLock<Mutex<Option<mpsc::SyncSender<String>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

unsafe extern "C" fn embedded_caps_cb(caps_json: *const c_char, _ctx: *mut c_void) {
    let json = unsafe { CStr::from_ptr(caps_json) }
        .to_string_lossy()
        .into_owned();
    if let Some(tx) = caps_slot().lock().unwrap().take() {
        let _ = tx.send(json);
    }
}

/// Owns the embedded driver thread + its on-disk launch state.
///
/// The driver runs `pie_driver_portable_run` to completion on its
/// thread. To shut it down cleanly the host needs to (eventually)
/// signal the C++ ShmemServer to stop — that path is wired in M2.4b.
/// For now `EmbeddedDriver` owns a `JoinHandle` that completes when
/// the driver returns naturally (e.g. on Ctrl-C in the host's signal
/// handler, which we'll add).
pub struct EmbeddedDriver {
    pub caps: DriverCapabilities,
    #[allow(dead_code)] // exposed for future RPC handshake; see M2.4b.
    pub shmem_name: String,
    #[allow(dead_code)] // exposed for future RPC handshake; see M2.4b.
    pub aux_socket_path: PathBuf,
    thread: Option<JoinHandle<i32>>,
    _state_dir: PathBuf,
}

impl EmbeddedDriver {
    /// Spawn the driver thread for the given model snapshot. Blocks
    /// until the driver emits caps via its `ready_cb` (or the
    /// `ready_timeout_s` from `options` elapses).
    ///
    /// `snapshot_dir` must be a local directory containing an HF-style
    /// model checkpoint (config.json, *.safetensors, tokenizer.json).
    /// HF download support comes later.
    pub fn start(
        options: &DriverOptions,
        snapshot_dir: &Path,
        group_id: usize,
    ) -> Result<Self> {
        if !snapshot_dir.is_dir() {
            return Err(anyhow!(
                "snapshot_dir {snapshot_dir:?} does not exist or is not a directory"
            ));
        }

        let state_dir = launch_state_dir().join(format!("g{group_id}"));
        std::fs::create_dir_all(&state_dir)
            .map_err(|e| anyhow!("create state dir {state_dir:?}: {e}"))?;

        let toml_path = state_dir.join("driver.toml");
        let aux_socket_path = state_dir.join("aux.sock");
        match options {
            DriverOptions::Portable(p) => {
                write_startup_toml(&toml_path, p, snapshot_dir, &aux_socket_path, group_id)?;
            }
            DriverOptions::Dummy { opts, random_seed, activation_dtype } => {
                write_dummy_startup_toml(
                    &toml_path,
                    opts,
                    snapshot_dir,
                    *random_seed,
                    activation_dtype,
                    group_id,
                )?;
            }
        }

        // Install the one-shot caps slot. Refuse if a previous launch
        // didn't drain the slot — indicates concurrent EmbeddedDriver,
        // which we don't support in v0.
        let (caps_tx, caps_rx) = mpsc::sync_channel::<String>(1);
        {
            let mut slot = caps_slot().lock().unwrap();
            if slot.is_some() {
                return Err(anyhow!(
                    "another EmbeddedDriver launch is already in progress"
                ));
            }
            *slot = Some(caps_tx);
        }

        let toml_path_str = toml_path
            .to_str()
            .ok_or_else(|| anyhow!("toml path is not valid UTF-8: {toml_path:?}"))?
            .to_owned();

        let thread = std::thread::Builder::new()
            .name(format!("pie-driver-{}-g{group_id}", driver_ffi::FLAVOR))
            .spawn(move || -> i32 {
                // Argv storage lives on the thread's stack for the
                // entire run — safe to hand its raw pointers to the
                // FFI entry.
                let argv_owned: Vec<CString> = vec![
                    CString::new(driver_ffi::ARGV0).unwrap(),
                    CString::new("--config").unwrap(),
                    CString::new(toml_path_str).unwrap(),
                ];
                let mut argv_ptrs: Vec<*mut c_char> = argv_owned
                    .iter()
                    .map(|s| s.as_ptr() as *mut c_char)
                    .collect();
                unsafe {
                    driver_ffi::run(
                        argv_ptrs.len() as c_int,
                        argv_ptrs.as_mut_ptr(),
                        /*install_signal_handlers=*/ 0,
                        embedded_caps_cb,
                        std::ptr::null_mut(),
                    )
                }
            })
            .map_err(|e| anyhow!("spawn pie-driver thread: {e}"))?;

        let ready_timeout_s = match options {
            DriverOptions::Portable(p) => p.ready_timeout_s,
            DriverOptions::Dummy { opts, .. } => opts.ready_timeout_s,
        };
        let timeout = Duration::from_secs_f64(ready_timeout_s.max(1.0));
        let json = match caps_rx.recv_timeout(timeout) {
            Ok(j) => j,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let rc = thread.join().unwrap_or(-1);
                // Drop the slot in case the cb never fired but a
                // sender is still installed.
                caps_slot().lock().unwrap().take();
                return Err(anyhow!(
                    "driver thread exited (rc={rc}) before emitting capabilities; \
                     check stderr for the [pie-driver-portable] error message"
                ));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                caps_slot().lock().unwrap().take();
                return Err(anyhow!(
                    "driver did not emit capabilities within {:.1}s",
                    timeout.as_secs_f64()
                ));
            }
        };

        let caps = DriverCapabilities::from_json(&json)?;
        Ok(EmbeddedDriver {
            shmem_name: caps.shmem_name.clone(),
            aux_socket_path,
            caps,
            thread: Some(thread),
            _state_dir: state_dir,
        })
    }

    /// Signal the driver thread's serve loop to exit. Idempotent; safe
    /// to call from any thread. After this returns the thread will
    /// finish whatever request it's mid-flight on, then exit naturally.
    /// Pair with [`Self::join`] to block until exit.
    pub fn request_stop(&self) {
        unsafe { driver_ffi::request_stop() };
    }

    /// Wait for the driver thread to exit, returning its rc. Caller
    /// should normally call [`Self::request_stop`] first; otherwise
    /// the driver will block forever in its serve loop.
    pub fn join(mut self) -> i32 {
        match self.thread.take().map(|h| h.join()) {
            Some(Ok(rc)) => rc,
            Some(Err(_)) => -1,
            None => 0,
        }
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
            "max_batch_tokens": 10240,
            "max_batch_size": 512,
            "arch_name": "qwen3",
            "vocab_size": 151936,
            "max_model_len": 4096,
            "activation_dtype": "bfloat16",
            "snapshot_dir": "/tmp/snap",
            "shmem_name": "/pie_shmem_g0"
        }"#;
        let caps = DriverCapabilities::from_json(json).unwrap();
        assert_eq!(caps.total_pages, 1024);
        assert_eq!(caps.arch_name, "qwen3");
        assert_eq!(caps.shmem_name, "/pie_shmem_g0");
    }

    #[test]
    fn startup_toml_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("startup.toml");
        let snap = tmp.path().join("snap");
        let aux = tmp.path().join("aux.sock");
        let opts = PortableDriverOptions::default();

        write_startup_toml(&out, &opts, &snap, &aux, 0).unwrap();

        // Re-parse the emitted TOML to confirm the schema the driver
        // expects matches what we wrote (driver-side parsing in
        // driver/portable/src/config.hpp uses the same structure).
        let text = std::fs::read_to_string(&out).unwrap();
        let val: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(val["shmem"]["name"].as_str().unwrap(), "/pie_shmem_g0");
        assert_eq!(val["batching"]["kv_page_size"].as_integer().unwrap(), 32);
        assert_eq!(val["model"]["hf_path"].as_str().unwrap(), snap.to_str().unwrap());
        assert_eq!(val["aux_ipc"]["socket_path"].as_str().unwrap(), aux.to_str().unwrap());
    }
}
