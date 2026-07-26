//! Native Rust safetensors header parser for load planning.
//!
//! This module reads *only* the safetensors framing — the leading 8-byte
//! little-endian header length followed by the JSON tensor index — and never
//! touches the bulk tensor bytes that follow. It produces a
//! [`CheckpointMetadata`] describing where each tensor lives so the storage
//! compiler can plan device residency without the runtime ever mapping weight
//! payloads.
//!
//! Every checkpoint tensor is emitted as [`Encoding::Raw`] with the storage
//! dtype the safetensors header declares. Scheme recognition that turns MXFP4
//! `*_blocks`/`*_scales` pairs (or block-FP8 weights) into quantized runtime
//! tensors happens later, by *name*, inside the storage compiler
//! (`abi.rs::default_for_target`). Real MXFP4 checkpoints (gpt-oss) already
//! ship their blocks/scales as `U8`; the theoretical `F4_E2M1`/`F8_E8M0` dtype
//! tags are mapped onto `U8` storage for parity with the C++
//! `dtype_from_safetensors` (`driver/cuda/src/tensor.cpp`).
//!
//! The 64-bit float dtype (`F64`) has no device representation here and is
//! rejected; `I64`/`U64` are carried through untouched for integer index
//! tables such as DeepSeek-V4's `ffn.gate.tid2eid`.
//!
//! The offset convention is the safetensors format's own:
//! `data_section_offset = 8 + header_size` and a tensor's absolute
//! `file_offset = data_section_offset + data_offsets[0]`.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use crate::error::{Error, OrOverflow};
use crate::types::{CheckpointFormat, DType, Encoding, FileId, TensorId, tensor_elements};

/// Width of the safetensors little-endian header-length prefix.
pub(crate) const SAFETENSORS_LEN_PREFIX: usize = 8;

/// One tensor entry parsed from a safetensors JSON index, before it is lowered
/// into a [`RawTensor`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    /// `data_offsets[0]` — start of the tensor within the data section.
    pub begin: u64,
    /// `data_offsets[1]` — one-past-end of the tensor within the data section.
    pub end: u64,
}

/// Map a safetensors dtype string onto a loader [`DType`], matching the C++
/// `dtype_from_safetensors` (`driver/cuda/src/tensor.cpp`).
///
/// MXFP4 storage tags (`F4_E2M1` blocks, `F8_E8M0` scales) map onto `U8` — the
/// scheme is recognised later by *name* in the storage compiler. `F64` has no
/// device representation here and is rejected; `I64`/`U64` pass through.
pub fn dtype_from_safetensors(s: &str) -> Result<DType, Error> {
    Ok(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F8_E4M3" => DType::F8E4M3,
        "F8_E5M2" => DType::F8E5M2,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "I8" => DType::I8,
        "U32" => DType::U32,
        "U16" => DType::U16,
        "U8" => DType::U8,
        "BOOL" => DType::Bool,
        // MXFP4 rides on U8 storage: `*_blocks` (`F4_E2M1`, two nibbles/byte)
        // + `*_scales` (`F8_E8M0`) tensor pairs recognised by schema/name in
        // the storage compiler. Real gpt-oss checkpoints already declare these
        // as `U8`; the packed tags are accepted for C++ parity. `F4_E2M1`'s
        // packing is what [`expected_span_bytes`] exists to name.
        //
        // `F8_E8M0` deliberately does NOT map to `DType::E8M0`, even though
        // that dtype now exists: the same tag names MXFP4's scales, which the
        // repack path reads as bytes. A contract that wants the exponent
        // decoded says so with `Transmute(.., E8M0)` -- naming the reading is
        // that node's whole job, and it leaves the file's tag meaning what
        // gpt-oss means by it.
        "F4_E2M1" | "F8_E8M0" => DType::U8,
        // Integer index tables ride through untouched: DeepSeek-V4's hash
        // router ships `ffn.gate.tid2eid` as `[vocab, K]` I64 and the kernel
        // reads it as `const int64_t*`. The loader never interprets these
        // values, it only has to account for their width.
        "I64" => DType::I64,
        "U64" => DType::U64,
        // No 64-bit float runtime dtype or executor contract. Reject
        // explicitly rather than narrowing checkpoint metadata.
        "F64" => {
            return Err(Error::Checkpoint(format!(
                "safetensors 64-bit dtype {s} is unsupported by the header parser"
            )));
        }
        other => {
            return Err(Error::Checkpoint(format!(
                "unsupported safetensors dtype {other}"
            )));
        }
    })
}

/// Storage bytes that `shape` implies for a safetensors dtype `tag`.
///
/// safetensors counts `shape` in *logical elements* for every dtype, including
/// sub-byte ones, so `F4_E2M1` (4-bit) stores two elements per byte while every
/// other tag we accept stores one element in `dtype.bytes()`.
///
/// Naming that relation for all tags is what bounds `shape`. The file's length
/// bounds the declared span, and the span bounds `shape` only through this — and
/// `shape`, not the span, is what the compiler sizes device buffers and reads
/// from. Exempting a tag from the check therefore does not merely skip a size
/// assertion; it lets that tag's `shape` be any number the header cares to
/// write. It did: `F4_E2M1` was exempt, and eight bytes of file bought a 1 GiB
/// allocation.
fn expected_span_bytes(tag: &str, shape: &[i64], dtype: DType, name: &str) -> Result<u64, Error> {
    let elements = tensor_elements(shape).or_overflow(format!(
        "safetensors tensor {name} has a shape whose element count overflows"
    ))?;
    if tag != "F4_E2M1" {
        return elements.checked_mul(dtype.bytes()).or_overflow(format!(
            "safetensors tensor {name} has a shape whose byte size overflows"
        ));
    }
    if elements % 2 != 0 {
        return Err(Error::Checkpoint(format!(
            "safetensors tensor {name} is F4_E2M1 with an odd element count {elements}, \
             which no whole number of storage bytes holds"
        )));
    }
    Ok(elements / 2)
}

/// Read the safetensors header length + JSON index from `prefix`, returning the
/// header size and the parsed tensor entries. `prefix` must contain at least
/// the leading 8-byte length plus the whole JSON header; any trailing bulk
/// bytes are ignored.
pub fn parse_safetensors_index(prefix: &[u8]) -> Result<(u64, Vec<SafetensorsEntry>), Error> {
    if prefix.len() < SAFETENSORS_LEN_PREFIX {
        return Err(Error::Checkpoint(format!(
            "safetensors prefix is {} bytes, need at least {} for the length header",
            prefix.len(),
            SAFETENSORS_LEN_PREFIX
        )));
    }
    let header_size = u64::from_le_bytes(prefix[..SAFETENSORS_LEN_PREFIX].try_into().unwrap());
    let json_end = SAFETENSORS_LEN_PREFIX
        .checked_add(
            usize::try_from(header_size)
                .or_overflow(format!("safetensors header size {header_size} too large"))?,
        )
        .or_overflow("safetensors header size overflows usize")?;
    if prefix.len() < json_end {
        return Err(Error::Checkpoint(format!(
            "safetensors header truncated: need {json_end} bytes, have {}",
            prefix.len()
        )));
    }
    let json_bytes = &prefix[SAFETENSORS_LEN_PREFIX..json_end];
    let value: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|err| Error::Checkpoint(format!("safetensors header is not valid JSON: {err}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| Error::Checkpoint("safetensors header JSON is not an object".to_string()))?;

    let mut entries = Vec::with_capacity(object.len());
    for (name, spec) in object {
        // The optional `__metadata__` key carries free-form string metadata,
        // never a tensor.
        if name == "__metadata__" {
            continue;
        }
        let spec = spec.as_object().ok_or_else(|| {
            Error::Checkpoint(format!("safetensors entry {name} is not an object"))
        })?;
        let dtype = spec
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::Checkpoint(format!("safetensors entry {name} lacks a string dtype"))
            })?
            .to_string();
        let shape = spec
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                Error::Checkpoint(format!("safetensors entry {name} lacks a shape array"))
            })?
            .iter()
            .map(|dim| {
                dim.as_i64().filter(|d| *d >= 0).ok_or_else(|| {
                    Error::Checkpoint(format!(
                        "safetensors entry {name} has a non-negative-integer shape dim"
                    ))
                })
            })
            .collect::<Result<Vec<i64>, _>>()?;
        let offsets = spec
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                Error::Checkpoint(format!(
                    "safetensors entry {name} lacks a data_offsets array"
                ))
            })?;
        if offsets.len() != 2 {
            return Err(Error::Checkpoint(format!(
                "safetensors entry {name} data_offsets must have exactly 2 elements"
            )));
        }
        let begin = offsets[0].as_u64().ok_or_else(|| {
            Error::Checkpoint(format!("safetensors entry {name} has a bad begin offset"))
        })?;
        let end = offsets[1].as_u64().ok_or_else(|| {
            Error::Checkpoint(format!("safetensors entry {name} has a bad end offset"))
        })?;
        if end < begin {
            return Err(Error::Checkpoint(format!(
                "safetensors entry {name} has end offset {end} < begin offset {begin}"
            )));
        }
        entries.push(SafetensorsEntry {
            dtype,
            name: name.clone(),
            shape,
            begin,
            end,
        });
    }
    // Deterministic ordering by name mirrors the C++ loader's name-keyed map.
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((header_size, entries))
}

/// Lower parsed [`SafetensorsEntry`]s into [`RawTensor`]s for one file.
///
/// `data_section_offset` is `8 + header_size`; tensor `file_offset`s are made
/// absolute against it. `id_base` is the [`TensorId`] to assign to the first
/// entry (subsequent entries increment from there).
///
/// `file_bytes` is what makes the result *true of a file* rather than merely
/// self-consistent. The size check below only relates `shape` to the declared
/// span, so without the file's own length nothing stops a header from placing a
/// tensor past the end — and every offset here is handed to an executor that
/// reads it.
pub fn tensors_from_safetensors_entries(
    entries: &[SafetensorsEntry],
    file_id: FileId,
    data_section_offset: u64,
    file_bytes: u64,
    id_base: u32,
) -> Result<Vec<RawTensor>, Error> {
    let mut out = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let dtype = dtype_from_safetensors(&entry.dtype)?;
        let span_bytes = entry.end.checked_sub(entry.begin).ok_or_else(|| {
            Error::Checkpoint(format!(
                "safetensors tensor {} has data_offsets [{}, {}), which end before they begin",
                entry.name, entry.begin, entry.end
            ))
        })?;
        let expected = expected_span_bytes(&entry.dtype, &entry.shape, dtype, &entry.name)?;
        if expected != span_bytes {
            return Err(Error::Checkpoint(format!(
                "safetensors tensor {} byte-size mismatch: data_offsets span {span_bytes}, expected {expected} for shape {:?} × {:?}",
                entry.name, entry.shape, dtype
            )));
        }
        let file_offset = data_section_offset
            .checked_add(entry.begin)
            .or_overflow(format!(
                "safetensors tensor {} file offset overflows",
                entry.name
            ))?;
        let file_end = file_offset.checked_add(span_bytes).or_overflow(format!(
            "safetensors tensor {} extent overflows",
            entry.name
        ))?;
        if file_end > file_bytes {
            return Err(Error::Checkpoint(format!(
                "safetensors tensor {} spans [{file_offset}, {file_end}) but the file is {file_bytes} bytes",
                entry.name
            )));
        }
        let id = id_base
            .checked_add(index as u32)
            .or_overflow("safetensors tensor id space overflows u32")?;
        out.push(RawTensor {
            id: TensorId(id),
            name: entry.name.clone(),
            file_id,
            file_offset,
            span_bytes,
            shape: entry.shape.clone(),
            encoding: Encoding::Raw(dtype),
        });
    }
    Ok(out)
}

/// Read only the safetensors framing (8-byte length + JSON header) from a file
/// on disk. The bulk tensor bytes are never read.
///
/// Returns the framing and the file's size, which the caller needs anyway and
/// which was already read here to bound the header.
pub fn read_safetensors_header_prefix(path: &Path) -> Result<(Vec<u8>, u64), Error> {
    let mut file = File::open(path)
        .map_err(|err| Error::Checkpoint(format!("cannot open {}: {err}", path.display())))?;
    let file_bytes = file
        .metadata()
        .map(|meta| meta.len())
        .map_err(|err| Error::Checkpoint(format!("cannot stat {}: {err}", path.display())))?;
    let mut len_buf = [0u8; SAFETENSORS_LEN_PREFIX];
    file.read_exact(&mut len_buf).map_err(|err| {
        Error::Checkpoint(format!(
            "cannot read safetensors length header from {}: {err}",
            path.display()
        ))
    })?;
    let header_size = u64::from_le_bytes(len_buf);
    // The declared size is the file's claim, not a fact about it, and it is
    // read before anything can contradict it: an eight-byte file may say its
    // header is 2^60 bytes. Sizing the buffer on that claim aborts the process
    // on a malformed checkpoint, so the claim is bounded by the one number the
    // file cannot lie about first.
    let available = file_bytes.saturating_sub(SAFETENSORS_LEN_PREFIX as u64);
    if header_size > available {
        return Err(Error::Checkpoint(format!(
            "safetensors header of {} declares {header_size} bytes but only {available} follow the length prefix",
            path.display()
        )));
    }
    let header_bytes = usize::try_from(header_size).or_overflow(format!(
        "safetensors header size {header_size} too large in {}",
        path.display()
    ))?;
    let framing = SAFETENSORS_LEN_PREFIX
        .checked_add(header_bytes)
        .or_overflow("safetensors framing size overflows usize")?;
    let mut prefix = len_buf.to_vec();
    prefix.resize(framing, 0);
    file.read_exact(&mut prefix[SAFETENSORS_LEN_PREFIX..])
        .map_err(|err| {
            Error::Checkpoint(format!(
                "cannot read safetensors JSON header from {}: {err}",
                path.display()
            ))
        })?;
    Ok((prefix, file_bytes))
}

/// Parse a dense safetensors checkpoint spanning one or more shard files into a
/// [`CheckpointMetadata`]. Only headers are read; bulk bytes are never mapped.
///
/// `files` is an ordered list of shard paths; [`FileId`]s are assigned by
/// position. [`TensorId`]s are assigned by a **global** ascending name sort
/// across every shard — matching the C++ loader, which sorts the full
/// `loader.tensor_names()` set before numbering
/// (`add_checkpoint_metadata_to_rust_input`). This global ordering is
/// load-bearing: the driver's executor indexes its `source_tensor_names` list
/// by the plan's `tensor_id`, so a per-shard ordering would mis-resolve
/// strided source reads on multi-shard checkpoints.
pub fn parse_safetensors_checkpoint(files: &[PathBuf]) -> Result<CheckpointMetadata, Error> {
    let mut checkpoint_files = Vec::with_capacity(files.len());
    let mut tensors = Vec::new();
    for (index, path) in files.iter().enumerate() {
        let file_id = FileId(index as u32);
        let (prefix, size_bytes) = read_safetensors_header_prefix(path)?;
        let (header_size, entries) = parse_safetensors_index(&prefix)?;
        let data_section_offset = SAFETENSORS_LEN_PREFIX as u64 + header_size;
        // Per-file ids are provisional — reassigned below by global name sort.
        let file_tensors = tensors_from_safetensors_entries(
            &entries,
            file_id,
            data_section_offset,
            size_bytes,
            0,
        )?;
        checkpoint_files.push(CheckpointFile {
            id: file_id,
            path: path.to_string_lossy().into_owned(),
            size_bytes,
            format: CheckpointFormat::Safetensors,
        });
        tensors.extend(file_tensors);
    }
    // Global ascending name sort + dense renumber, so `tensor_id` matches the
    // driver's C++ `source_tensor_names` ordering across all shards.
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    for (index, tensor) in tensors.iter_mut().enumerate() {
        tensor.id =
            TensorId(u32::try_from(index).or_overflow("checkpoint tensor id space overflows u32")?);
    }
    Ok(CheckpointMetadata {
        files: checkpoint_files,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a safetensors `prefix` (8-byte length + JSON) from a raw JSON body.
    fn framed(json: &str) -> Vec<u8> {
        let body = json.as_bytes();
        let mut out = (body.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(body);
        out
    }

    /// Write `bytes` to a uniquely named temp file and return its path.
    fn temp_checkpoint(tag: &str, bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "pie_loader_{tag}_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn parses_single_dense_tensor() {
        // 2x3 F16 = 12 bytes.
        let prefix = framed(r#"{"w":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#);
        let (header_size, entries) = parse_safetensors_index(&prefix).unwrap();
        assert_eq!(header_size as usize, prefix.len() - SAFETENSORS_LEN_PREFIX);
        assert_eq!(entries.len(), 1);
        let dso = SAFETENSORS_LEN_PREFIX as u64 + header_size;
        let tensors =
            tensors_from_safetensors_entries(&entries, FileId(0), dso, dso + 12, 0).unwrap();
        let t = &tensors[0];
        assert_eq!(t.name, "w");
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.encoding, Encoding::Raw(DType::F16));
        assert_eq!(t.span_bytes, 12);
        assert_eq!(t.file_offset, dso);
    }

    #[test]
    fn skips_metadata_key_and_sorts_by_name() {
        let prefix = framed(
            r#"{"__metadata__":{"format":"pt"},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        );
        let (_, entries) = parse_safetensors_index(&prefix).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "a");
        assert_eq!(entries[1].name, "b");
    }

    #[test]
    fn assigns_absolute_file_offsets() {
        let prefix = framed(
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[8],"data_offsets":[4,12]}}"#,
        );
        let (header_size, entries) = parse_safetensors_index(&prefix).unwrap();
        let dso = SAFETENSORS_LEN_PREFIX as u64 + header_size;
        let tensors =
            tensors_from_safetensors_entries(&entries, FileId(0), dso, dso + 12, 0).unwrap();
        assert_eq!(tensors[0].file_offset, dso);
        assert_eq!(tensors[1].file_offset, dso + 4);
        assert_eq!(tensors[1].span_bytes, 8);
    }

    #[test]
    fn maps_mxfp4_block_dtype_to_u8_and_halves_the_span() {
        // `F4_E2M1` packs two nibbles per byte: shape [16] logical elements are
        // stored in 8 bytes, so the span is validated against shape÷2, not
        // shape×1.
        let prefix =
            framed(r#"{"w_blocks":{"dtype":"F4_E2M1","shape":[16],"data_offsets":[0,8]}}"#);
        let (header_size, entries) = parse_safetensors_index(&prefix).unwrap();
        let dso = SAFETENSORS_LEN_PREFIX as u64 + header_size;
        let tensors =
            tensors_from_safetensors_entries(&entries, FileId(0), dso, dso + 8, 0).unwrap();
        assert_eq!(tensors[0].encoding, Encoding::Raw(DType::U8));
        assert_eq!(tensors[0].span_bytes, 8);
    }

    #[test]
    fn maps_mxfp4_scale_dtype_to_u8() {
        assert_eq!(dtype_from_safetensors("F8_E8M0").unwrap(), DType::U8);
        assert_eq!(dtype_from_safetensors("F4_E2M1").unwrap(), DType::U8);
    }

    #[test]
    fn rejects_64bit_float_dtype() {
        assert!(
            dtype_from_safetensors("F64").is_err(),
            "F64 should be rejected"
        );
    }

    #[test]
    fn accepts_64bit_integer_index_tables() {
        assert_eq!(dtype_from_safetensors("I64").unwrap(), DType::I64);
        assert_eq!(dtype_from_safetensors("U64").unwrap(), DType::U64);
        assert_eq!(DType::I64.bytes(), 8);
        assert_eq!(DType::U64.bytes(), 8);
    }

    #[test]
    fn accepts_fp8_dense_dtypes() {
        assert_eq!(dtype_from_safetensors("F8_E4M3").unwrap(), DType::F8E4M3);
        assert_eq!(dtype_from_safetensors("F8_E5M2").unwrap(), DType::F8E5M2);
    }

    #[test]
    fn rejects_byte_size_mismatch() {
        // 2x3 F16 should be 12 bytes; declare 8.
        let prefix = framed(r#"{"w":{"dtype":"F16","shape":[2,3],"data_offsets":[0,8]}}"#);
        let (_, entries) = parse_safetensors_index(&prefix).unwrap();
        let err = tensors_from_safetensors_entries(&entries, FileId(0), 8, 20, 0).unwrap_err();
        assert!(format!("{err}").contains("mismatch"), "got: {err}");
    }

    #[test]
    fn rejects_short_prefix_and_bad_offsets() {
        assert!(parse_safetensors_index(&[0u8; 4]).is_err());
        let prefix = framed(r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#);
        assert!(parse_safetensors_index(&prefix).is_err());
    }

    #[test]
    fn rejects_a_tensor_reaching_past_the_end_of_the_file() {
        // The header describes eight bytes of tensor; the file carries four.
        let prefix = framed(r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#);
        let mut bytes = prefix.clone();
        bytes.extend_from_slice(&[0xEEu8; 4]);
        let path = temp_checkpoint("truncated", &bytes);

        let err = parse_safetensors_checkpoint(std::slice::from_ref(&path)).unwrap_err();
        std::fs::remove_file(&path).ok();

        assert!(format!("{err}").contains("but the file is"), "got: {err}");
    }

    #[test]
    fn rejects_an_mxfp4_tensor_reaching_past_the_end_of_the_file() {
        // shape [16] nibbles = 8 storage bytes, so the entry is self-consistent
        // and only the file's own length can contradict it.
        let prefix =
            framed(r#"{"w_blocks":{"dtype":"F4_E2M1","shape":[16],"data_offsets":[0,8]}}"#);
        let (_, entries) = parse_safetensors_index(&prefix).unwrap();
        let dso = prefix.len() as u64;
        let err =
            tensors_from_safetensors_entries(&entries, FileId(0), dso, dso + 7, 0).unwrap_err();
        assert!(format!("{err}").contains("but the file is"), "got: {err}");
    }

    #[test]
    fn rejects_an_mxfp4_shape_that_the_declared_span_cannot_hold() {
        // The hole this closes: `F4_E2M1` used to skip the size check entirely,
        // and `shape` is what the compiler sizes device buffers from. Eight
        // bytes of file bought a 1 GiB allocation and reads to match.
        let prefix =
            framed(r#"{"w_blocks":{"dtype":"F4_E2M1","shape":[1073741824],"data_offsets":[0,8]}}"#);
        let (_, entries) = parse_safetensors_index(&prefix).unwrap();
        let dso = prefix.len() as u64;
        let err =
            tensors_from_safetensors_entries(&entries, FileId(0), dso, dso + 8, 0).unwrap_err();
        assert!(
            format!("{err}").contains("byte-size mismatch"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_an_mxfp4_shape_that_no_whole_byte_count_holds() {
        let prefix =
            framed(r#"{"w_blocks":{"dtype":"F4_E2M1","shape":[15],"data_offsets":[0,8]}}"#);
        let (_, entries) = parse_safetensors_index(&prefix).unwrap();
        let dso = prefix.len() as u64;
        let err =
            tensors_from_safetensors_entries(&entries, FileId(0), dso, dso + 8, 0).unwrap_err();
        assert!(format!("{err}").contains("odd element count"), "got: {err}");
    }

    #[test]
    fn rejects_a_header_larger_than_the_file() {
        // Eight bytes claiming a 2^64-byte header. Believing it sizes a buffer
        // no allocator can serve, and the abort is not catchable.
        let path = temp_checkpoint("huge_header", &u64::MAX.to_le_bytes());

        let err = parse_safetensors_checkpoint(std::slice::from_ref(&path)).unwrap_err();
        std::fs::remove_file(&path).ok();

        assert!(format!("{err}").contains("declares"), "got: {err}");
    }

    #[test]
    fn reads_header_prefix_from_file_without_bulk() {
        let prefix = framed(r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#);
        // Append fake bulk bytes the parser must never need.
        let mut file_bytes = prefix.clone();
        file_bytes.extend_from_slice(&[0xEEu8; 8]);
        let path = temp_checkpoint("header", &file_bytes);

        let meta = parse_safetensors_checkpoint(std::slice::from_ref(&path)).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(meta.files.len(), 1);
        assert_eq!(meta.tensors.len(), 1);
        assert_eq!(meta.tensors[0].name, "w");
        assert_eq!(meta.tensors[0].span_bytes, 8);
        assert_eq!(meta.files[0].format, CheckpointFormat::Safetensors);
        assert_eq!(meta.files[0].size_bytes, file_bytes.len() as u64);
    }
}
