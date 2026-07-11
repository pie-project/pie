//! Native-driver bootstrap helpers for pie-worker.
//!
//! This module exposes:
//!   * [`DriverCapabilities`] — typed driver capability payloads.
//!   * [`write_cuda_startup_toml`] / [`write_metal_startup_toml`] — emit the
//!     per-launch TOML each native driver reads at creation.
//!   * [`create_native_driver`] — build a runtime-owned [`NativeDriver`]
//!     plus its caps before `pie_engine::bootstrap`.

#[cfg(feature = "driver-cuda")]
use std::ffi::CStr;
#[cfg(feature = "driver-cuda")]
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

#[cfg(feature = "driver-metal")]
use crate::config::MetalDriverOptions;
use crate::config::{CudaMemoryProfile, CudaNativeDriverOptions, DummyDriverOptions};
use crate::driver_ffi::Flavor;

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
    pie_engine::util::get_pie_home()
        .join("standalone")
        .join(std::process::id().to_string())
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

/// Emit the metal driver's startup TOML — same `[model]` + `[batching]` +
/// `[runtime]` layout consumed by `driver/metal/src/config.hpp`. The metal
/// launch state is identical apart from the `metal:N` backend selector.
#[cfg(feature = "driver-metal")]
pub fn write_metal_startup_toml(
    out_path: &Path,
    options: &MetalDriverOptions,
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

/// Compile the checkpoint's StorageProgram in-process (weight-loader Variant A)
/// and write it beside the startup TOML, returning the file path to record in
/// `[model].storage_program_path`. The embedded cuda driver deserializes that
/// file instead of running its own C++ checkpoint parse + compile (the
/// *locality switch* = path-present). The **bulk weight bytes never cross**:
/// the serialized program records only where each tensor lives; the driver
/// reads the payloads from its own copy of the checkpoint at execution.
///
/// Returns `None` (and logs) when the in-process compile can't produce a
/// provably-correct program for this checkpoint — e.g. the config can't be
/// parsed, or the request needs a device capability the runtime can't query
/// before load (`runtime_quant="fp8"` → `fp8_native`; `mxfp4_moe="native"` →
/// Blackwell). The embedded driver then transparently falls back to its own
/// C++ compile, which runs after the device query. See plan.md
/// "device-cap-derived gaps".
#[cfg(feature = "driver-cuda")]
fn compile_cuda_storage_program(
    out_path: &Path,
    snapshot_dir: &Path,
    opts: &CudaNativeDriverOptions,
    tp: Option<&TpLaunch>,
) -> Option<PathBuf> {
    use pie_weight_loader::inproc::{compile_snapshot_to_bytes, parse_model_config};
    use pie_weight_loader::storage::StorageTarget;
    use pie_weight_loader::types::{BackendKind, Mxfp4MoePolicy};

    // Static CUDA storage-target constants (mirror driver/cuda/src/model/
    // loaded_model.cpp `kStorageMaxTileBytes` / `kStoragePreferredAlignment`).
    const CUDA_MAX_TILE_BYTES: u64 = 64 * 1024 * 1024;
    const CUDA_PREFERRED_ALIGNMENT: u32 = 256;

    // `runtime_quant="fp8"` is cleared by the driver on GPUs without native FP8
    // (`fp8_native`), a capability the runtime can't query before load; defer to
    // the C++ compile so we never emit a program the device would reject.
    if opts.runtime_quant == "fp8" {
        tracing::info!(
            "weight-loader: runtime_quant='fp8' is device-conditional (fp8_native); \
             deferring storage compile to the embedded cuda driver"
        );
        return None;
    }

    let model = match parse_model_config(snapshot_dir, opts.runtime_quant.clone()) {
        Ok(model) => model,
        Err(err) => {
            tracing::warn!(
                "weight-loader: in-process storage compile skipped ({err}); the \
                 embedded cuda driver will compile the checkpoint itself"
            );
            return None;
        }
    };

    // Resolve the MXFP4 MoE policy the way the C++ driver does
    // (select_mxfp4_moe_lowering) for a non-Blackwell target. `native_mxfp4_moe`
    // is GPU-derived (sm_100+) and can't be device-queried before load, so it
    // defaults to false; it only affects the compiled program when the model
    // requests `mxfp4_moe="native"` (handled below) — routed/eager are
    // device-independent.
    let native_mxfp4_moe = false;
    let mxfp4_moe = match opts.mxfp4_moe.as_str() {
        "" | "auto" | "routed_dequant" | "packed" => Mxfp4MoePolicy::RoutedDecode,
        "bf16" | "dequant" | "eager_bf16" => Mxfp4MoePolicy::EagerBf16,
        "native" => {
            tracing::info!(
                "weight-loader: mxfp4_moe='native' needs a Blackwell device query \
                 unavailable pre-load; deferring storage compile to the C++ path"
            );
            return None;
        }
        other => {
            tracing::warn!(
                "weight-loader: unknown mxfp4_moe='{other}'; deferring storage \
                 compile to the embedded cuda driver"
            );
            return None;
        }
    };

    let (tp_rank, tp_size) = tp.map(|t| (t.rank as u32, t.size as u32)).unwrap_or((0, 1));
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank,
        tp_size,
        max_tile_bytes: CUDA_MAX_TILE_BYTES,
        preferred_alignment: CUDA_PREFERRED_ALIGNMENT,
        mxfp4_moe,
        native_mxfp4_moe,
    };

    let bytes = match compile_snapshot_to_bytes(snapshot_dir, &model, target) {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                "weight-loader: in-process storage compile failed ({err}); the \
                 embedded cuda driver will compile the checkpoint itself"
            );
            return None;
        }
    };

    // Write beside the (already per-rank) startup TOML so concurrent TP ranks
    // don't collide.
    let bin_path = out_path.with_extension("storage_program.bin");
    if let Err(err) = std::fs::write(&bin_path, &bytes) {
        tracing::warn!(
            "weight-loader: failed to write storage program {bin_path:?} ({err}); \
             the embedded cuda driver will compile the checkpoint itself"
        );
        return None;
    }
    tracing::info!(
        "weight-loader: compiled StorageProgram in-process → {} ({} bytes); the \
         embedded cuda driver will deserialize it (bulk weights never cross)",
        bin_path.display(),
        bytes.len()
    );
    Some(bin_path)
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
    // Weight-loader Variant A: compile the StorageProgram in-process and hand it
    // to the embedded cuda driver as a file path (bulk weights never cross).
    #[cfg(feature = "driver-cuda")]
    if let Some(bin_path) = compile_cuda_storage_program(out_path, snapshot_dir, opts, tp) {
        insert_str(&mut model, "storage_program_path", path_string(&bin_path));
    }
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
    insert_int(&mut batching, "total_pages", opts.total_pages);
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
        callback_delay_ms: 0,
        reject_launches: false,
        reject_launches_remaining: 0,
        fail_launches_after_accept: false,
        operation_log: None,
        launch_observer: None,
    })
}

fn validate_snapshot_dir(snapshot_dir: &Path) -> Result<()> {
    let is_gguf_file =
        snapshot_dir.is_file() && snapshot_dir.extension().is_some_and(|e| e == "gguf");
    if !snapshot_dir.is_dir() && !is_gguf_file {
        return Err(anyhow!(
            "snapshot_dir {snapshot_dir:?} does not exist, or is not a directory or .gguf file"
        ));
    }
    Ok(())
}

#[cfg(feature = "driver-cuda")]
pub(crate) fn create_native_driver_group(
    rank_options: &[DriverOptions],
    snapshot_dir: &Path,
    group_id: usize,
    tp_launches: &[TpLaunch],
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
        let state_dir = local_driver_state_dir(group_id, Some(tp))?;
        let toml_path = state_dir.join("driver.toml");
        write_cuda_startup_toml(&toml_path, opts, snapshot_dir, group_id, Some(tp))?;
        config_blobs.push(toml_path.to_string_lossy().into_owned().into_bytes());
    }

    let (native, caps) = pie_engine::driver::NativeDriver::cuda_group_create(config_blobs)?;
    Ok(crate::translate::GroupDriver { caps, native })
}

pub(crate) fn create_native_driver(
    options: &DriverOptions,
    snapshot_dir: &Path,
    group_id: usize,
    tp: Option<&TpLaunch>,
) -> Result<crate::translate::GroupDriver> {
    let _ = (group_id, tp);
    validate_snapshot_dir(snapshot_dir)?;

    let (native, caps) = match options {
        #[cfg(feature = "driver-cuda")]
        DriverOptions::CudaNative(opts) => {
            let state_dir = local_driver_state_dir(group_id, tp)?;
            let toml_path = state_dir.join("driver.toml");
            write_cuda_startup_toml(&toml_path, opts, snapshot_dir, group_id, tp)?;
            let config_path = toml_path.to_string_lossy();
            pie_engine::driver::NativeDriver::cuda_create(config_path.as_bytes())?
        }
        #[cfg(feature = "driver-metal")]
        DriverOptions::Metal(opts) => {
            let state_dir = local_driver_state_dir(group_id, tp)?;
            let toml_path = state_dir.join("driver.toml");
            write_metal_startup_toml(&toml_path, opts, snapshot_dir, group_id)?;
            let config_path = toml_path.to_string_lossy();
            pie_engine::driver::NativeDriver::metal_create(config_path.as_bytes())?
        }
        DriverOptions::Dummy {
            opts,
            random_seed,
            activation_dtype,
        } => {
            let options = dummy_native_options(opts, snapshot_dir, *random_seed, activation_dtype)?;
            pie_engine::driver::NativeDriver::dummy(options)?
        }
    };

    Ok(crate::translate::GroupDriver { caps, native })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_json_round_trips() {
        let json = r#"{
            "abi_version": 1,
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
        }"#;
        let caps = parse_caps_json(json).unwrap();
        assert_eq!(caps.abi_version, 1);
        assert_eq!(caps.total_pages, 1024);
        assert_eq!(caps.arch_name, "qwen3");
        assert_eq!(caps.snapshot_dir, "/tmp/snap");
        assert_eq!(caps.max_forward_tokens, 4096);
        assert_eq!(caps.max_page_refs, 262144);
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
        assert_eq!(val["batching"]["total_pages"].as_integer().unwrap(), 0);
        assert_eq!(val["batching"].as_table().unwrap().len(), 6);
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
