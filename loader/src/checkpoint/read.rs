//! The one place in the loader that opens a checkpoint.
//!
//! `compile` is a function of three values — what is in the file, what the
//! caller wants out, and what the device can do (architecture.md §12 row 12) —
//! and the first of those is a [`CheckpointMetadata`], not a directory. This
//! module is what turns the directory into the value: file discovery, header
//! parsing, and the `config.json` read. Everything below it is pure, and
//! `standalone.rs` is the test that keeps it that way.
//!
//! Parity with the C++ path it replaced is intentional and load-bearing: file
//! discovery mirrors `discover_safetensors_manifest` (`SingleFile` preference:
//! prefer `model.safetensors`, else the sharded `model.safetensors.index.json`
//! `weight_map`, else the single file), and every checkpoint tensor is emitted
//! as [`crate::types::Encoding::Raw`] with the storage dtype — MXFP4
//! recognition is by *name*, inside the compiler.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::checkpoint::CheckpointMetadata;
use crate::checkpoint::gguf::parse_gguf_checkpoint;
use crate::checkpoint::safetensors::parse_safetensors_checkpoint;
use crate::error::Error;

/// Discover the safetensors shard files for a snapshot directory, matching the
/// C++ `discover_safetensors_manifest` with a `SingleFile` layout preference.
///
/// Returns the shard paths in the order the C++ loader assigns file ids:
/// a lone `model.safetensors`, otherwise the sorted unique shard names from
/// `model.safetensors.index.json`'s `weight_map`.
pub fn discover_safetensors_files(snapshot_dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let single = snapshot_dir.join("model.safetensors");
    let index = snapshot_dir.join("model.safetensors.index.json");

    // SingleFile preference: a lone `model.safetensors` wins even when an index
    // is also present.
    if single.is_file() {
        return Ok(vec![single]);
    }

    if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", index.display())))?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
            Error::Checkpoint(format!("{} is not valid JSON: {err}", index.display()))
        })?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                Error::Checkpoint(format!("{} missing 'weight_map'", index.display()))
            })?;
        // Unique shard names, sorted — a BTreeSet reproduces the C++ dedup+sort.
        let mut shard_names = BTreeSet::new();
        for shard in weight_map.values() {
            let shard = shard.as_str().ok_or_else(|| {
                Error::Checkpoint(format!(
                    "{} weight_map has a non-string shard",
                    index.display()
                ))
            })?;
            shard_names.insert(shard.to_string());
        }
        return Ok(shard_names
            .into_iter()
            .map(|s| snapshot_dir.join(s))
            .collect());
    }

    Err(Error::Checkpoint(format!(
        "no model.safetensors[.index.json] in {}",
        snapshot_dir.display()
    )))
}

/// The single GGUF checkpoint file for a snapshot directory, if present. GGUF
/// checkpoints are a single-file format (`model.gguf` or a lone `*.gguf`).
fn discover_gguf_file(snapshot_dir: &Path) -> Option<PathBuf> {
    if snapshot_dir.is_file()
        && snapshot_dir
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        return Some(snapshot_dir.to_path_buf());
    }
    let named = snapshot_dir.join("model.gguf");
    if named.is_file() {
        return Some(named);
    }
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(snapshot_dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    ggufs.sort();
    ggufs.into_iter().next()
}

/// The single `.zt` checkpoint for a snapshot directory, if present.
fn discover_zt_file(snapshot_dir: &Path) -> Option<PathBuf> {
    if snapshot_dir.is_file()
        && snapshot_dir
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("zt"))
    {
        return Some(snapshot_dir.to_path_buf());
    }
    let named = snapshot_dir.join("model.zt");
    named.is_file().then_some(named)
}

/// Parse a checkpoint's headers into a [`CheckpointMetadata`]. Only headers
/// are read; bulk tensor bytes are never mapped.
///
/// The reader is chosen by what is on disk, in the order a snapshot is likely
/// to hold it: a `.zt` artifact (what `pie model optimize` writes), else the
/// canonical HF safetensors layout, else GGUF. Formats beyond those three —
/// `.pt`, `.npz`, `.h5`, `.onnx` — are read through [`zt::parse_checkpoint`]
/// when named directly, since a snapshot directory has no convention for
/// them.
pub fn parse_checkpoint_metadata(snapshot_dir: &Path) -> Result<CheckpointMetadata, Error> {
    if let Some(zt) = discover_zt_file(snapshot_dir) {
        return crate::checkpoint::zt::parse_checkpoint(&zt);
    }
    if snapshot_dir.is_file() {
        if let Some(gguf) = discover_gguf_file(snapshot_dir) {
            return parse_gguf_checkpoint(&gguf);
        }
        // A file that is not GGUF or `.zt`: let the projections identify it.
        return crate::checkpoint::zt::parse_checkpoint(snapshot_dir);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("load_planner_inproc_{tag}_{}", std::process::id()));
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
    fn direct_gguf_paths_are_discovered() {
        let dir = tmpdir("direct_gguf");
        let path = dir.join("model.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        assert_eq!(discover_gguf_file(&path), Some(path.clone()));
        std::fs::remove_dir_all(&dir).ok();
    }
}
