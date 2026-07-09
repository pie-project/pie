//! In-process compile entry (weight-loader Variant A, step 2).
//!
//! This is the Rust-native replacement for the C++ loader's
//! parse-checkpoint → build-FFI-input → invoke-compile pipeline
//! (`driver/cuda/src/loader/rust_loader_bridge.hpp`,
//! `compile_rust_loader_plan_from_metadata`). The runtime links this crate as
//! an `rlib` and calls [`compile_snapshot_to_bytes`] directly — no FFI — to
//! turn a checkpoint directory into a serialized [`StorageProgram`] IR that is
//! shipped to the driver. The **bulk weight bytes never cross this boundary**:
//! the program only records where each tensor lives (file + offset + span) and
//! how to place it; the driver reads the payloads from its own copy of the
//! checkpoint during execution.
//!
//! Parity with the C++ path is intentional and load-bearing:
//! * file discovery mirrors `discover_safetensors_manifest`
//!   (`SingleFile` preference: prefer `model.safetensors`, else the sharded
//!   `model.safetensors.index.json` `weight_map`, else the single file);
//! * every checkpoint tensor is emitted as [`Encoding::Raw`] with the storage
//!   dtype (MXFP4 recognition is by *name* inside the compiler);
//! * the RuntimeABI is left implicit so the compiler uses
//!   [`RuntimeAbi::default_for_target`] — the C++ side only sets the ABI
//!   *name*/*version* (`"pie-cuda"`, `1`), never the tensor contracts, so the
//!   two paths compile byte-identical programs.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::abi::RuntimeAbi;
use crate::checkpoint_header::parse_safetensors_checkpoint;
use crate::config::ModelConfig;
use crate::error::CompileError;
use crate::gguf::parse_gguf_checkpoint;
use crate::source::CheckpointMetadata;
use crate::storage::{StorageProgram, StorageTarget};
use crate::storage_compiler::compile_storage_program;

/// Discover the safetensors shard files for a snapshot directory, matching the
/// C++ `discover_safetensors_manifest` with a `SingleFile` layout preference.
///
/// Returns the shard paths in the order the C++ loader assigns file ids:
/// a lone `model.safetensors`, otherwise the sorted unique shard names from
/// `model.safetensors.index.json`'s `weight_map`.
pub fn discover_safetensors_files(snapshot_dir: &Path) -> Result<Vec<PathBuf>, CompileError> {
    let single = snapshot_dir.join("model.safetensors");
    let index = snapshot_dir.join("model.safetensors.index.json");

    // SingleFile preference: a lone `model.safetensors` wins even when an index
    // is also present.
    if single.is_file() {
        return Ok(vec![single]);
    }

    if index.is_file() {
        let text = std::fs::read_to_string(&index).map_err(|err| {
            CompileError::InvalidInput(format!("cannot read {}: {err}", index.display()))
        })?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
            CompileError::InvalidInput(format!("{} is not valid JSON: {err}", index.display()))
        })?;
        let weight_map = value.get("weight_map").and_then(serde_json::Value::as_object).ok_or_else(
            || CompileError::InvalidInput(format!("{} missing 'weight_map'", index.display())),
        )?;
        // Unique shard names, sorted — a BTreeSet reproduces the C++ dedup+sort.
        let mut shard_names = BTreeSet::new();
        for shard in weight_map.values() {
            let shard = shard.as_str().ok_or_else(|| {
                CompileError::InvalidInput(format!(
                    "{} weight_map has a non-string shard",
                    index.display()
                ))
            })?;
            shard_names.insert(shard.to_string());
        }
        return Ok(shard_names.into_iter().map(|s| snapshot_dir.join(s)).collect());
    }

    if single.is_file() {
        return Ok(vec![single]);
    }

    Err(CompileError::InvalidInput(format!(
        "no model.safetensors[.index.json] in {}",
        snapshot_dir.display()
    )))
}

/// The single GGUF checkpoint file for a snapshot directory, if present. GGUF
/// checkpoints are a single-file format (`model.gguf` or a lone `*.gguf`).
fn discover_gguf_file(snapshot_dir: &Path) -> Option<PathBuf> {
    let named = snapshot_dir.join("model.gguf");
    if named.is_file() {
        return Some(named);
    }
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(snapshot_dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")))
        .collect();
    ggufs.sort();
    ggufs.into_iter().next()
}

/// Parse a checkpoint directory's headers into a [`CheckpointMetadata`],
/// picking the on-disk format (safetensors, else GGUF). Only headers are read;
/// bulk tensor bytes are never mapped.
pub fn parse_checkpoint_metadata(snapshot_dir: &Path) -> Result<CheckpointMetadata, CompileError> {
    // Safetensors takes precedence — it is the canonical HF snapshot format and
    // the C++ loader opens it first.
    match discover_safetensors_files(snapshot_dir) {
        Ok(files) => parse_safetensors_checkpoint(&files),
        Err(safetensors_err) => {
            if let Some(gguf) = discover_gguf_file(snapshot_dir) {
                parse_gguf_checkpoint(&gguf)
            } else {
                Err(safetensors_err)
            }
        }
    }
}

/// Compile a checkpoint directory into a [`StorageProgram`] entirely in-process.
///
/// `model` and `target` carry the same coarse configuration the C++ bridge
/// builds from the HF config + backend caps. The RuntimeABI is derived with
/// [`RuntimeAbi::default_for_target`], exactly as the FFI compile does when the
/// caller supplies no explicit contracts.
pub fn compile_snapshot(
    snapshot_dir: &Path,
    model: &ModelConfig,
    target: StorageTarget,
) -> Result<StorageProgram, CompileError> {
    let metadata = parse_checkpoint_metadata(snapshot_dir)?;
    let abi = RuntimeAbi::default_for_target(&metadata, model, &target)?;
    compile_storage_program(&metadata, model, &abi, target)
}

/// Compile a checkpoint directory and serialize the resulting program with the
/// same `bincode` wire format the C++ deserialize+execute path consumes
/// (`pie_loader_program_deserialize`). This is the payload the runtime ships to
/// the driver over the new storage-program boundary op.
pub fn compile_snapshot_to_bytes(
    snapshot_dir: &Path,
    model: &ModelConfig,
    target: StorageTarget,
) -> Result<Vec<u8>, CompileError> {
    let program = compile_snapshot(snapshot_dir, model, target)?;
    bincode::serialize(&program)
        .map_err(|err| CompileError::Internal(format!("storage program serialize failed: {err}")))
}

/// Parse the coarse [`ModelConfig`] fields the storage compiler needs from a
/// snapshot's `config.json`, mirroring the HF-key precedence in the C++
/// `parse_hf_config` (`driver/cuda/src/loader/hf_config.cpp`):
///
/// * `model_type` — top-level `model_type` (nested multimodal towers keep the
///   outer type here, which is all the storage compile needs);
/// * `num_hidden_layers` — required;
/// * `num_experts` — first of `num_local_experts`, `num_experts`,
///   `n_routed_experts`, else 0 (dense);
/// * `num_experts_per_tok` — `num_experts_per_tok`, else 0;
/// * `quant_method` — `quantization_config.quant_method` (checked on the text
///   config then the root), else empty.
///
/// `runtime_quant` is the runtime's own quantization request (e.g. `"fp8"`),
/// not a checkpoint field; the caller passes it through from boot config.
pub fn parse_model_config(
    snapshot_dir: &Path,
    runtime_quant: impl Into<String>,
) -> Result<ModelConfig, CompileError> {
    let path = snapshot_dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|err| {
        CompileError::InvalidInput(format!("cannot read {}: {err}", path.display()))
    })?;
    let root: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        CompileError::InvalidInput(format!("{} is not valid JSON: {err}", path.display()))
    })?;

    // Some multimodal checkpoints nest the language-model config under
    // `text_config`; the storage compile keys off the text tower's structure.
    let text_cfg = root.get("text_config").filter(|v| v.is_object());
    let cfg = text_cfg.unwrap_or(&root);

    let model_type = cfg
        .get("model_type")
        .or_else(|| root.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    let num_hidden_layers = cfg
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            CompileError::InvalidInput(format!("{} lacks num_hidden_layers", path.display()))
        })?;

    let first_i64 = |keys: &[&str]| -> i64 {
        for k in keys {
            if let Some(v) = cfg.get(*k).and_then(serde_json::Value::as_i64) {
                return v;
            }
        }
        0
    };
    let num_experts = first_i64(&["num_local_experts", "num_experts", "n_routed_experts"]);
    let num_experts_per_tok = first_i64(&["num_experts_per_tok"]);

    let quant_method = cfg
        .get("quantization_config")
        .filter(|v| v.is_object())
        .or_else(|| root.get("quantization_config").filter(|v| v.is_object()))
        .and_then(|q| q.get("quant_method"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    Ok(ModelConfig {
        model_type,
        quant_method,
        runtime_quant: runtime_quant.into(),
        num_hidden_layers: num_hidden_layers.max(0) as u32,
        num_experts: num_experts.max(0) as u32,
        num_experts_per_tok: num_experts_per_tok.max(0) as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("wl_inproc_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn discovers_single_file_over_index() {
        let dir = tmpdir("single");
        write(&dir, "model.safetensors", "x");
        write(&dir, "model.safetensors.index.json", "{}");
        let files = discover_safetensors_files(&dir).unwrap();
        assert_eq!(files, vec![dir.join("model.safetensors")]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discovers_sorted_unique_shards_from_index() {
        let dir = tmpdir("sharded");
        write(
            &dir,
            "model.safetensors.index.json",
            r#"{"weight_map":{"a":"model-00002.safetensors","b":"model-00001.safetensors","c":"model-00001.safetensors"}}"#,
        );
        let files = discover_safetensors_files(&dir).unwrap();
        assert_eq!(
            files,
            vec![
                dir.join("model-00001.safetensors"),
                dir.join("model-00002.safetensors"),
            ]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parses_dense_model_config() {
        let dir = tmpdir("cfg_dense");
        write(
            &dir,
            "config.json",
            r#"{"model_type":"qwen3","num_hidden_layers":28}"#,
        );
        let cfg = parse_model_config(&dir, "").unwrap();
        assert_eq!(cfg.model_type, "qwen3");
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_experts, 0);
        assert_eq!(cfg.quant_method, "");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parses_moe_and_quant_config() {
        let dir = tmpdir("cfg_moe");
        write(
            &dir,
            "config.json",
            r#"{"model_type":"gpt_oss","num_hidden_layers":24,"num_local_experts":32,"num_experts_per_tok":4,"quantization_config":{"quant_method":"mxfp4"}}"#,
        );
        let cfg = parse_model_config(&dir, "fp8").unwrap();
        assert_eq!(cfg.model_type, "gpt_oss");
        assert_eq!(cfg.num_experts, 32);
        assert_eq!(cfg.num_experts_per_tok, 4);
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.runtime_quant, "fp8");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_config_num_experts_fallbacks() {
        let dir = tmpdir("cfg_fallback");
        write(
            &dir,
            "config.json",
            r#"{"model_type":"deepseek_v3","num_hidden_layers":4,"n_routed_experts":8}"#,
        );
        let cfg = parse_model_config(&dir, "").unwrap();
        assert_eq!(cfg.num_experts, 8);
        std::fs::remove_dir_all(&dir).ok();
    }
}
