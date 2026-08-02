//! Reading a checkpoint through zTensor.
//!
//! One reader for every format the loader accepts. `ztensor-compat` projects
//! `.safetensors`, `.gguf`, `.npz`, `.pt`, `.h5` and `.onnx` into one object
//! model — named objects, each a shape and a set of parts, each part a byte
//! range in some file — and this module is the translation of that model into
//! the loader's [`CheckpointMetadata`]. The two are close enough that most of
//! it is renaming: an object's part is a [`RawTensor`], its blob reference is
//! `(file_id, file_offset, span_bytes)`.
//!
//! What is *not* renaming is the encoding. The loader's [`Encoding`] names a
//! quantization scheme; zTensor names a layout profile and a logical type. The
//! table in [`encoding_of`] is the whole of that correspondence, and it is the
//! part to read carefully — everything else here is plumbing.
//!
//! # Parts and names
//!
//! A zTensor object may carry several parts (a quantized weight is payload
//! plus scales). The loader's tensor space is flat and name-addressed, so a
//! multi-part object becomes one [`RawTensor`] per part: the `"data"` part
//! keeps the object's name, and any other part is suffixed `.<part>`. That
//! matches how the same tensors are named in the checkpoints these files come
//! from (`*_blocks` / `*_scales` in safetensors MXFP4), so a contract written
//! against a converted checkpoint reads the same either way.

use std::path::Path;

use ztensor::{DType as ZDType, Manifest, Object, Part};

use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use crate::error::Error;
use crate::types::{
    Axis, CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};

/// Opens a checkpoint of any supported format and describes it.
///
/// Only metadata is read: the projections map the file and parse its header,
/// and bulk tensor bytes are never touched here.
pub fn parse_checkpoint(path: &Path) -> Result<CheckpointMetadata, Error> {
    let format = ztensor_compat::detect(path).map_err(zt_err)?;
    let source = ztensor_compat::open_any(path).map_err(zt_err)?;
    metadata_of(source.manifest(), path, format)
}

fn zt_err(err: ztensor::Error) -> Error {
    Error::Checkpoint(err.to_string())
}

fn checkpoint_format(label: &str) -> CheckpointFormat {
    match label {
        "safetensors" => CheckpointFormat::Safetensors,
        "gguf" => CheckpointFormat::Gguf,
        _ => CheckpointFormat::Unknown,
    }
}

/// Translates a zTensor manifest into the loader's checkpoint description.
///
/// `path` is the file the manifest came from; a sharded model's shards are
/// resolved by the same positional convention `Model::open` uses, so their
/// paths follow from it.
pub fn metadata_of(
    manifest: &Manifest,
    path: &Path,
    format_label: &str,
) -> Result<CheckpointMetadata, Error> {
    let format = checkpoint_format(format_label);
    let mut files = vec![CheckpointFile {
        id: FileId(0),
        path: path.to_string_lossy().into_owned(),
        size_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        format,
    }];
    // Shard index k is the k-th entry of the table, and the loader's file ids
    // are dense, so the two agree by construction.
    for (&index, shard) in &manifest.shards {
        let shard_path = shard_path(path, index);
        files.push(CheckpointFile {
            id: FileId(u32::try_from(index).map_err(|_| {
                Error::Checkpoint(format!("shard index {index} does not fit a file id"))
            })?),
            path: shard_path.to_string_lossy().into_owned(),
            size_bytes: shard.size,
            format,
        });
    }

    let mut tensors = Vec::new();
    for (name, object) in &manifest.objects {
        for (part_name, part) in &object.parts {
            let tensor_name = if part_name == "data" {
                name.clone()
            } else {
                format!("{name}.{part_name}")
            };
            if part.encoding.is_some() {
                return Err(Error::Checkpoint(format!(
                    "{tensor_name}: compressed parts cannot be planned — the loader \
                     addresses checkpoint bytes where they lie; convert the file to a \
                     raw one first"
                )));
            }
            let id = TensorId(u32::try_from(tensors.len()).map_err(|_| {
                Error::Checkpoint("checkpoint has more tensors than a tensor id holds".into())
            })?);
            tensors.push(RawTensor {
                id,
                name: tensor_name,
                file_id: FileId(u32::try_from(part.blob.shard).map_err(|_| {
                    Error::Checkpoint("shard index does not fit a file id".into())
                })?),
                file_offset: part.blob.offset,
                span_bytes: part.blob.length,
                shape: shape_of(object, part_name, part)?,
                encoding: encoding_of(object, part)?,
            });
        }
    }

    Ok(CheckpointMetadata { files, tensors })
}

/// `<stem>-<index:05>.zt` beside the root — the positional shard convention.
fn shard_path(root: &Path, index: u64) -> std::path::PathBuf {
    let dir = root.parent().unwrap_or(Path::new("."));
    let stem = root
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    dir.join(format!("{stem}-{index:05}.zt"))
}

/// The shape a part presents to the planner.
///
/// The object's shape describes the *object*, and for a single-part object
/// that is also the part's shape. A secondary part (scales beside a payload)
/// has its own extent, which the loader needs as a shape of its own; it is
/// derived from the bytes, since the object shape does not describe it.
fn shape_of(object: &Object, part_name: &str, part: &Part) -> Result<Vec<i64>, Error> {
    if part_name == "data" {
        return object
            .shape
            .iter()
            .map(|&d| {
                i64::try_from(d)
                    .map_err(|_| Error::Checkpoint(format!("dimension {d} does not fit an i64")))
            })
            .collect();
    }
    let width = part.dtype.width().max(1);
    let elements = part.decoded_size() / width;
    Ok(vec![i64::try_from(elements).map_err(|_| {
        Error::Checkpoint("part element count does not fit an i64".into())
    })?])
}

/// zTensor storage type to loader dtype.
fn dtype_of(dtype: ZDType, ltype: Option<&str>) -> Result<DType, Error> {
    // A registered logical type names what the bytes *mean*; where the loader
    // has a dtype for it, that is the dtype to use.
    if let Some(ltype) = ltype {
        return Ok(match ltype {
            "f8_e4m3fn" | "f8_e4m3fnuz" => DType::F8E4M3,
            "f8_e5m2" | "f8_e5m2fnuz" => DType::F8E5M2,
            "f8_e8m0" => DType::E8M0,
            "bool" => DType::Bool,
            // f4_e2m1 has no loader dtype of its own: MXFP4 payloads ride on
            // U8 storage there, and the scheme names the packing.
            "f4_e2m1" => DType::U8,
            other => {
                return Err(Error::Checkpoint(format!(
                    "logical type {other:?} has no loader dtype"
                )));
            }
        });
    }
    Ok(match dtype {
        ZDType::F32 => DType::F32,
        ZDType::F16 => DType::F16,
        ZDType::BF16 => DType::BF16,
        ZDType::I64 => DType::I64,
        ZDType::I32 => DType::I32,
        ZDType::I16 => DType::I16,
        ZDType::I8 => DType::I8,
        ZDType::U64 => DType::U64,
        ZDType::U32 => DType::U32,
        ZDType::U16 => DType::U16,
        ZDType::U8 => DType::U8,
        ZDType::F64 => {
            return Err(Error::Checkpoint(
                "f64 tensors have no device representation".into(),
            ));
        }
    })
}

/// The encoding a part carries: its layout profile decides whether it is a
/// plain dtype or a quantized payload, and the object's attributes carry the
/// scheme's parameters.
///
/// Layouts the loader has no scheme for are an error, not a guess — reading a
/// quantized payload as raw bytes of its storage type is exactly the silent
/// misinterpretation the object model exists to prevent.
fn encoding_of(object: &Object, part: &Part) -> Result<Encoding, Error> {
    let layout = object.layout.as_str();
    let dtype = dtype_of(part.dtype, part.ltype.as_deref())?;

    // `dense` is the only layout whose parts are plain values.
    if layout == "dense" {
        return Ok(Encoding::Raw(dtype));
    }

    let scheme = scheme_of(layout, object)?;
    let group_size = attr_u64(object, "group_size")
        .or_else(|| attr_u64(object, "block_size"))
        .or_else(|| attr_u64(object, "elems_per_block"))
        .unwrap_or_else(|| u64::from(scheme.default_group_size()));
    let bits = attr_u64(object, "bits").unwrap_or_else(|| u64::from(scheme.default_bits()));
    let axis = attr_u64(object, "axis").and_then(|a| u8::try_from(a).ok()).map(Axis);

    Ok(Encoding::Quant(
        QuantSpec {
            scheme,
            // What the payload decodes to. The checkpoint does not say, and
            // every device path the loader targets decodes to BF16.
            logical_dtype: DType::BF16,
            bits_per_element: u8::try_from(bits).unwrap_or(0),
            group_size: u32::try_from(group_size).unwrap_or(0),
            channel_axis: axis,
        }
        .normalized(),
    ))
}

/// Layout profile id to quantization scheme.
///
/// The `gguf.*` family is the projection of GGUF's block structs, kept
/// verbatim; `zt.mx/1` is OCP Microscaling. `zt.quant_group/1` names a
/// *family* — affine integers with a group scale — whose members differ in
/// packing order and zero-point convention, which the profile does not
/// distinguish; a file this loader wrote records which member it is in the
/// object's `scheme` attribute, and that is what is read back here. A
/// `zt.quant_group/1` object from elsewhere carries no such attribute and is
/// refused rather than guessed at.
///
/// A profile this table does not name is refused.
fn scheme_of(layout: &str, object: &Object) -> Result<QuantScheme, Error> {
    if layout == "zt.quant_group/1" {
        let named = attr_text(object, "scheme").ok_or_else(|| {
            Error::Checkpoint(
                "zt.quant_group/1 without a 'scheme' attribute: the profile names a \
                 family, and which member decides how the payload is packed"
                    .to_string(),
            )
        })?;
        return scheme_from_name(named);
    }
    Ok(match layout {
        "zt.mx/1" => QuantScheme::Mxfp4E2M1E8M0,
        "gguf.q4_0/1" => QuantScheme::GgufQ4_0,
        "gguf.q4_k/1" => QuantScheme::GgufQ4K,
        "gguf.q5_0/1" => QuantScheme::GgufQ5_0,
        "gguf.q5_k/1" => QuantScheme::GgufQ5K,
        "gguf.q8_0/1" => QuantScheme::GgufQ8_0,
        "gguf.mxfp4/1" => QuantScheme::Mxfp4E2M1E8M0,
        other => {
            return Err(Error::Checkpoint(format!(
                "layout {other:?} has no loader quantization scheme; a plan cannot \
                 address bytes it cannot describe"
            )));
        }
    })
}

/// The affine-group members, by the name `write_zt` records.
fn scheme_from_name(name: &str) -> Result<QuantScheme, Error> {
    Ok(match name {
        "AwqInt4" => QuantScheme::AwqInt4,
        "GptqInt4" => QuantScheme::GptqInt4,
        "MlxAffineU4" => QuantScheme::MlxAffineU4,
        "Int4B8" => QuantScheme::Int4B8,
        "Int8Symmetric" => QuantScheme::Int8Symmetric,
        "Int8Asymmetric" => QuantScheme::Int8Asymmetric,
        "Fp8E4M3" => QuantScheme::Fp8E4M3,
        "Fp8E5M2" => QuantScheme::Fp8E5M2,
        other => {
            return Err(Error::Checkpoint(format!(
                "scheme {other:?} is not one this loader knows"
            )));
        }
    })
}

fn attr_text<'a>(object: &'a Object, key: &str) -> Option<&'a str> {
    object
        .attributes
        .as_ref()?
        .as_map()?
        .iter()
        .find(|(k, _)| k.as_text() == Some(key))
        .and_then(|(_, v)| v.as_text())
}

fn attr_u64(object: &Object, key: &str) -> Option<u64> {
    let attrs = object.attributes.as_ref()?;
    attrs
        .as_map()?
        .iter()
        .find(|(k, _)| k.as_text() == Some(key))
        .and_then(|(_, v)| v.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use ztensor::{BlobRef, Layout};

    fn part(dtype: ZDType, ltype: Option<&str>, offset: u64, length: u64) -> Part {
        Part {
            dtype,
            ltype: ltype.map(str::to_string),
            blob: BlobRef {
                shard: 0,
                offset,
                length,
            },
            encoding: None,
            decoded_length: None,
            digest: None,
        }
    }

    fn manifest_with(name: &str, object: Object) -> Manifest {
        let mut objects = BTreeMap::new();
        objects.insert(name.to_string(), object);
        Manifest {
            attributes: None,
            shards: BTreeMap::new(),
            objects,
        }
    }

    #[test]
    fn dense_object_becomes_one_raw_tensor() {
        let mut parts = BTreeMap::new();
        parts.insert("data".to_string(), part(ZDType::BF16, None, 65536, 32));
        let object = Object {
            shape: vec![4, 4],
            layout: Layout::Dense,
            attributes: None,
            parts,
        };
        let metadata =
            metadata_of(&manifest_with("w", object), Path::new("m.zt"), "zt").unwrap();
        assert_eq!(metadata.tensors.len(), 1);
        let tensor = &metadata.tensors[0];
        assert_eq!(tensor.name, "w");
        assert_eq!(tensor.shape, vec![4, 4]);
        assert_eq!(tensor.file_offset, 65536);
        assert_eq!(tensor.span_bytes, 32);
        assert_eq!(tensor.encoding, Encoding::Raw(DType::BF16));
    }

    #[test]
    fn secondary_parts_take_a_suffixed_name() {
        let mut parts = BTreeMap::new();
        parts.insert("data".to_string(), part(ZDType::U8, Some("f4_e2m1"), 65536, 512));
        parts.insert(
            "scales".to_string(),
            part(ZDType::U8, Some("f8_e8m0"), 131072, 32),
        );
        let object = Object {
            shape: vec![32, 32],
            layout: Layout::from_name("zt.mx/1"),
            attributes: Some(ztensor::cbor::Value::Map(vec![(
                ztensor::cbor::Value::Text("block_size".into()),
                ztensor::cbor::Value::Uint(32),
            )])),
            parts,
        };
        let metadata =
            metadata_of(&manifest_with("w", object), Path::new("m.zt"), "zt").unwrap();
        let names: Vec<&str> = metadata.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["w", "w.scales"]);

        let payload = metadata.tensor_by_name("w").unwrap();
        match &payload.encoding {
            Encoding::Quant(spec) => {
                assert_eq!(spec.scheme, QuantScheme::Mxfp4E2M1E8M0);
                assert_eq!(spec.group_size, 32);
            }
            other => panic!("expected a quantized payload, got {other:?}"),
        }
        // The scales part keeps its own dtype and its element count as shape.
        let scales = metadata.tensor_by_name("w.scales").unwrap();
        assert_eq!(scales.shape, vec![32]);
    }

    #[test]
    fn unknown_layout_is_refused() {
        let mut parts = BTreeMap::new();
        parts.insert("data".to_string(), part(ZDType::U8, None, 65536, 32));
        let object = Object {
            shape: vec![32],
            layout: Layout::from_name("vendor.mystery/1"),
            attributes: None,
            parts,
        };
        let err = metadata_of(&manifest_with("w", object), Path::new("m.zt"), "zt").unwrap_err();
        assert!(format!("{err}").contains("no loader quantization scheme"));
    }

    #[test]
    fn compressed_parts_are_refused() {
        let mut p = part(ZDType::U8, None, 65536, 16);
        p.encoding = Some("zt.zstd-seekable/1".into());
        p.decoded_length = Some(32);
        let mut parts = BTreeMap::new();
        parts.insert("data".to_string(), p);
        let object = Object {
            shape: vec![32],
            layout: Layout::Dense,
            attributes: None,
            parts,
        };
        let err = metadata_of(&manifest_with("w", object), Path::new("m.zt"), "zt").unwrap_err();
        assert!(format!("{err}").contains("compressed parts"));
    }
}
