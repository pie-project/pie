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

use crate::checkpoint::zt;
use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use crate::error::Error;
use crate::types::{CheckpointFormat, DType, Encoding, FileId, TensorId};

/// A safetensors shard whose leading header bytes have already been read.
///
/// `header` includes the eight-byte little-endian JSON length and the padded
/// JSON body, exactly as stored at offset zero. Keeping I/O outside this type
/// lets local files and remote ranges share the loader's format authority.
pub struct SafetensorsShardHeader {
    pub path: String,
    pub size_bytes: u64,
    pub header: Vec<u8>,
}

#[derive(serde::Deserialize)]
struct SafetensorsTensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Project already-fetched safetensors headers into loader metadata.
///
/// This is the metadata-first counterpart of [`parse_checkpoint_metadata`].
/// Fetching the bytes is a provisioning concern; validating the tensor names,
/// encodings, shapes, and addresses remains a checkpoint concern.
pub fn parse_safetensors_headers(
    shards: &[SafetensorsShardHeader],
) -> Result<CheckpointMetadata, Error> {
    let mut files = Vec::with_capacity(shards.len());
    let mut tensors = Vec::new();
    let mut names = BTreeSet::new();

    for (file_index, shard) in shards.iter().enumerate() {
        let file_id = FileId(u32::try_from(file_index).map_err(|_| {
            Error::Checkpoint("checkpoint has more files than a file id holds".into())
        })?);
        if shard.header.len() < 8 {
            return Err(Error::Checkpoint(format!(
                "{}: safetensors header is shorter than its 8-byte length",
                shard.path
            )));
        }
        let json_len = u64::from_le_bytes(shard.header[..8].try_into().unwrap());
        if json_len > 100 << 20 {
            return Err(Error::Checkpoint(format!(
                "{}: safetensors header is {json_len} bytes; refusing more than 100 MiB",
                shard.path
            )));
        }
        let data_start = 8u64.checked_add(json_len).ok_or_else(|| {
            Error::Checkpoint(format!(
                "{}: safetensors header length overflows",
                shard.path
            ))
        })?;
        if data_start != shard.header.len() as u64 {
            return Err(Error::Checkpoint(format!(
                "{}: declared safetensors header is {json_len} bytes but {} bytes were supplied",
                shard.path,
                shard.header.len().saturating_sub(8)
            )));
        }
        if data_start > shard.size_bytes {
            return Err(Error::Checkpoint(format!(
                "{}: safetensors header ends at {data_start}, past the {}-byte file",
                shard.path, shard.size_bytes
            )));
        }

        let root: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&shard.header[8..]).map_err(|err| {
                Error::Checkpoint(format!("{}: invalid safetensors header: {err}", shard.path))
            })?;
        let mut occupied = Vec::new();
        let mut shard_tensors = Vec::new();
        for (name, value) in root {
            if name == "__metadata__" {
                continue;
            }
            if !names.insert(name.clone()) {
                return Err(Error::Checkpoint(format!(
                    "tensor {name:?} appears in more than one safetensors shard"
                )));
            }
            let entry: SafetensorsTensorHeader = serde_json::from_value(value).map_err(|err| {
                Error::Checkpoint(format!("{} tensor {name:?}: {err}", shard.path))
            })?;
            let [start, end] = entry.data_offsets;
            if start > end {
                return Err(Error::Checkpoint(format!(
                    "{} tensor {name:?}: data offsets go backwards ({start}..{end})",
                    shard.path
                )));
            }
            let span_bytes = end - start;
            let dtype = safetensors_dtype(&entry.dtype).ok_or_else(|| {
                Error::Checkpoint(format!(
                    "{} tensor {name:?}: unsupported safetensors dtype {:?}",
                    shard.path, entry.dtype
                ))
            })?;
            let shape: Vec<i64> = entry
                .shape
                .iter()
                .map(|&dimension| {
                    i64::try_from(dimension).map_err(|_| {
                        Error::Checkpoint(format!(
                            "{} tensor {name:?}: dimension {dimension} does not fit i64",
                            shard.path
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;
            let expected = crate::types::tensor_nbytes(&shape, dtype.bytes()).ok_or_else(|| {
                Error::Checkpoint(format!(
                    "{} tensor {name:?}: shape byte count overflows",
                    shard.path
                ))
            })?;
            if expected != span_bytes {
                return Err(Error::Checkpoint(format!(
                    "{} tensor {name:?}: shape and dtype require {expected} bytes, header declares {span_bytes}",
                    shard.path
                )));
            }
            let file_offset = data_start.checked_add(start).ok_or_else(|| {
                Error::Checkpoint(format!(
                    "{} tensor {name:?}: file offset overflows",
                    shard.path
                ))
            })?;
            let file_end = data_start.checked_add(end).ok_or_else(|| {
                Error::Checkpoint(format!(
                    "{} tensor {name:?}: file end overflows",
                    shard.path
                ))
            })?;
            if file_end > shard.size_bytes {
                return Err(Error::Checkpoint(format!(
                    "{} tensor {name:?}: range ends at {file_end}, past the {}-byte file",
                    shard.path, shard.size_bytes
                )));
            }
            occupied.push((start, end, name.clone()));
            shard_tensors.push((name, file_offset, span_bytes, shape, Encoding::Raw(dtype)));
        }

        occupied.sort_unstable_by_key(|(start, _, _)| *start);
        let mut next = 0u64;
        for (start, end, name) in &occupied {
            if *start != next {
                return Err(Error::Checkpoint(format!(
                    "{}: tensor {name:?} starts at {start}, leaving a gap after {next}",
                    shard.path
                )));
            }
            next = *end;
        }
        for pair in occupied.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(Error::Checkpoint(format!(
                    "{}: tensor ranges {:?} and {:?} overlap",
                    shard.path, pair[0].2, pair[1].2
                )));
            }
        }
        let data_bytes = shard.size_bytes - data_start;
        if next != data_bytes {
            return Err(Error::Checkpoint(format!(
                "{}: tensors cover {next} data bytes but the file carries {data_bytes}",
                shard.path
            )));
        }
        shard_tensors.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (name, file_offset, span_bytes, shape, encoding) in shard_tensors {
            let id = TensorId(u32::try_from(tensors.len()).map_err(|_| {
                Error::Checkpoint("checkpoint has more tensors than a tensor id holds".into())
            })?);
            tensors.push(RawTensor {
                id,
                name,
                file_id,
                file_offset,
                span_bytes,
                shape,
                encoding,
            });
        }
        files.push(CheckpointFile {
            id: file_id,
            path: shard.path.clone(),
            size_bytes: shard.size_bytes,
            format: CheckpointFormat::Safetensors,
        });
    }
    tensors.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    for (index, tensor) in tensors.iter_mut().enumerate() {
        tensor.id = TensorId(index as u32);
    }
    Ok(CheckpointMetadata { files, tensors })
}

fn safetensors_dtype(dtype: &str) -> Option<DType> {
    Some(match dtype {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F8_E4M3" | "F8_E4M3FN" => DType::F8E4M3,
        "F8_E5M2" => DType::F8E5M2,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "I8" => DType::I8,
        "U64" => DType::U64,
        "U32" => DType::U32,
        "U16" => DType::U16,
        "U8" => DType::U8,
        "BOOL" => DType::Bool,
        _ => return None,
    })
}

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
/// Every format is read the same way, through [`zt`]: `ztensor-compat`
/// projects safetensors, GGUF, `.npz`, `.pt`, `.h5` and `.onnx` into one
/// object model, and `zt` translates that model into the loader's. This module
/// is what remains of format knowledge here, and it is not about *formats* --
/// it is about **layout**: which files on disk make up one checkpoint. That is
/// a question no format answers about itself. A safetensors snapshot states it
/// in `model.safetensors.index.json`, a convention beside the format; GGUF and
/// `.zt` are single-file and state it by being one file.
///
/// The order below is the order a snapshot is likely to hold: a `.zt`
/// artifact (what `pie model convert` writes), else the canonical HF
/// safetensors layout, else GGUF.
/// The bytes of the metadata object named `path`, or `None` if the checkpoint
/// has no such object.
///
/// Lives here rather than on [`CheckpointMetadata`] because this module is
/// where the filesystem is allowed to exist: everything below the reader takes
/// values, and `standalone.rs` enforces it. Consumers that instead resolved
/// the object to a file, seeked to its offset and read its span were
/// reimplementing addressing the loader already does — and would keep working
/// while silently disagreeing with it if, say, a sharded artifact put the
/// object somewhere other than the root.
pub fn read_meta(metadata: &CheckpointMetadata, path: &str) -> Result<Option<Vec<u8>>, Error> {
    use std::io::{Read, Seek, SeekFrom};

    let Some(object) = metadata.meta_object(path) else {
        return Ok(None);
    };
    let file = metadata
        .files
        .iter()
        .find(|file| file.id == object.file_id)
        .ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} points at a file the checkpoint lacks",
                object.name
            ))
        })?;
    let mut handle = std::fs::File::open(&file.path)
        .map_err(|err| Error::Checkpoint(format!("cannot open {}: {err}", file.path)))?;
    handle
        .seek(SeekFrom::Start(object.file_offset))
        .map_err(|err| Error::Checkpoint(format!("cannot seek in {}: {err}", file.path)))?;
    let mut bytes = vec![0u8; object.span_bytes as usize];
    handle.read_exact(&mut bytes).map_err(|err| {
        Error::Checkpoint(format!(
            "cannot read {} from {}: {err}",
            object.name, file.path
        ))
    })?;
    Ok(Some(bytes))
}

pub fn parse_checkpoint_metadata(snapshot_dir: &Path) -> Result<CheckpointMetadata, Error> {
    if let Some(zt) = discover_zt_file(snapshot_dir) {
        return zt::parse_checkpoint(&zt);
    }
    if snapshot_dir.is_file() {
        // A file names itself; detection is the projections' job, not a
        // suffix's.
        return zt::parse_checkpoint(snapshot_dir);
    }
    // Safetensors takes precedence — it is the canonical HF snapshot format and
    // the C++ loader opens it first.
    match discover_safetensors_files(snapshot_dir) {
        Ok(files) => zt::parse_checkpoint_files(&files),
        Err(safetensors_err) => match discover_gguf_file(snapshot_dir) {
            Some(gguf) => zt::parse_checkpoint(&gguf),
            None => Err(safetensors_err),
        },
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

    #[test]
    fn metadata_first_headers_match_the_filesystem_reader() {
        let dir = tmpdir("header_parity");
        let path = dir.join("model.safetensors");
        let json = r#"{"z":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&(json.len() as u64).to_le_bytes());
        file.extend_from_slice(json.as_bytes());
        file.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, &file).unwrap();

        let filesystem = parse_checkpoint_metadata(&dir).unwrap();
        let projected = parse_safetensors_headers(&[SafetensorsShardHeader {
            path: path.display().to_string(),
            size_bytes: file.len() as u64,
            header: file[..8 + json.len()].to_vec(),
        }])
        .unwrap();
        assert_eq!(projected, filesystem);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn metadata_first_headers_reject_unclaimed_payload_bytes() {
        let json = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        let mut header = Vec::new();
        header.extend_from_slice(&(json.len() as u64).to_le_bytes());
        header.extend_from_slice(json.as_bytes());
        let error = parse_safetensors_headers(&[SafetensorsShardHeader {
            path: "model.safetensors".into(),
            size_bytes: header.len() as u64 + 2,
            header,
        }])
        .unwrap_err()
        .to_string();
        assert!(error.contains("leaving a gap"), "got: {error}");
    }
}
