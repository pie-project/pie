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

use std::path::{Path, PathBuf};

use ztensor::{DType as ZDType, Object, Part};

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
    let format = checkpoint_format(ztensor_compat::detect(path).map_err(zt_err)?);
    let source = ztensor_compat::open_any(path).map_err(zt_err)?;
    let manifest = source.manifest();

    // A `.zt` model may name shards, and the manifest gives each part the
    // shard index its bytes live in. The loader's file ids are dense and the
    // shard table is keyed from 1, so index and id agree by construction.
    let mut files = vec![CheckpointFile {
        id: FileId(0),
        path: path.to_string_lossy().into_owned(),
        size_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        format,
    }];
    for (&index, shard) in &manifest.shards {
        files.push(CheckpointFile {
            id: FileId(u32::try_from(index).map_err(|_| {
                Error::Checkpoint(format!("shard index {index} does not fit a file id"))
            })?),
            path: shard_path(path, index).to_string_lossy().into_owned(),
            size_bytes: shard.size,
            format,
        });
    }

    let mut tensors = Vec::new();
    for (name, object) in &manifest.objects {
        collect(&mut tensors, name, object, |part| {
            u32::try_from(part.blob.shard)
                .map(FileId)
                .map_err(|_| Error::Checkpoint("shard index does not fit a file id".into()))
        })?;
    }
    Ok(CheckpointMetadata { files, tensors })
}

/// Opens a set of files that together hold one checkpoint.
///
/// What a sharded snapshot is. Each file describes itself completely and none
/// of them names the others, so the set is the caller's claim — HF states it
/// in `model.safetensors.index.json`, which is a convention beside the format
/// rather than anything inside it. [`ztensor::Composite`] is that shape: one
/// name space over N sources, with the file each name came from recorded, and
/// a name in two files refused rather than resolved by precedence.
///
/// The single-file case is not routed here. It would work, but it would also
/// answer a question nobody asked — with one file there is no set to state.
pub fn parse_checkpoint_files(paths: &[PathBuf]) -> Result<CheckpointMetadata, Error> {
    let composite = ztensor_compat::open_all(paths).map_err(zt_err)?;

    let mut files = Vec::with_capacity(paths.len());
    for (index, path) in paths.iter().enumerate() {
        files.push(CheckpointFile {
            id: FileId(u32::try_from(index).map_err(|_| {
                Error::Checkpoint("checkpoint has more files than a file id holds".into())
            })?),
            path: path.to_string_lossy().into_owned(),
            size_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            format: checkpoint_format(ztensor_compat::detect(path).map_err(zt_err)?),
        });
    }

    let mut tensors = Vec::new();
    for (index, name, object) in composite.objects() {
        // Every part of a composite addresses its own file, so the source
        // index is the whole of the file identity -- there is no shard table
        // to consult, and a blob offset means what it meant in the file it
        // came from.
        let file = FileId(u32::try_from(index).map_err(|_| {
            Error::Checkpoint("checkpoint has more files than a file id holds".into())
        })?);
        collect(&mut tensors, name, object, |_| Ok(file))?;
    }
    Ok(CheckpointMetadata { files, tensors })
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

/// Appends one [`RawTensor`] per part of `object`.
///
/// A zTensor object may carry several parts and the loader's tensor space is
/// flat, so the `"data"` part keeps the object's name and any other part is
/// suffixed `.<part>` -- which is how the same tensors are named in the
/// checkpoints they come from (`*_blocks` / `*_scales` in safetensors MXFP4).
fn collect(
    tensors: &mut Vec<RawTensor>,
    name: &str,
    object: &Object,
    file_of: impl Fn(&Part) -> Result<FileId, Error>,
) -> Result<(), Error> {
    for (part_name, part) in &object.parts {
        let tensor_name = if part_name == "data" {
            name.to_string()
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
            file_id: file_of(part)?,
            file_offset: part.blob.offset,
            span_bytes: part.blob.length,
            shape: shape_of(object, part_name, part)?,
            encoding: encoding_of(object, part)?,
        });
    }
    Ok(())
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

/// Layout profile to quantization scheme.
///
/// `zt.quant_group/1` is parametric (core spec §5.2): the profile names a
/// space, and the attributes say which point. So the scheme is *derived*
/// from the packing order, the scale form and the zero-point form rather
/// than read out of a name the file was asked to carry — a file written by
/// something that never heard of this enum still lands on the right one.
///
/// The `gguf.*` family is opaque: the block struct is preserved verbatim and
/// the profile identifies the layout. `zt.mx/1` is OCP Microscaling.
///
/// A profile this function cannot resolve is refused. A plan cannot address
/// bytes it cannot describe, and guessing is what the object model exists to
/// prevent.
fn scheme_of(layout: &str, object: &Object) -> Result<QuantScheme, Error> {
    if layout == "zt.quant_group/1" {
        return affine_group_scheme(object);
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

/// Which point of the affine-group space an object names.
///
/// The parameters that separate the schemes this loader knows are the bit
/// width, the packing order, and the form of the zero point. Anything the
/// combination does not name is refused rather than rounded to the nearest
/// scheme: reading GPTQ codes as AWQ would decode every weight backwards
/// within its word.
fn affine_group_scheme(object: &Object) -> Result<QuantScheme, Error> {
    let missing = |what: &str| {
        Error::Checkpoint(format!(
            "zt.quant_group/1 is parametric and its {what} attribute is required; \
             a decoder cannot be chosen without it"
        ))
    };
    let bits = attr_u64(object, "bits").ok_or_else(|| missing("bits"))?;
    let packing = attr_map(object, "packing").ok_or_else(|| missing("packing"))?;
    let order = map_text(packing, "order").ok_or_else(|| missing("packing.order"))?;
    let zero = attr_map(object, "zero_point").ok_or_else(|| missing("zero_point"))?;
    let form = map_text(zero, "form").ok_or_else(|| missing("zero_point.form"))?;
    let zero_packing = map_text(zero, "packing");

    Ok(match (bits, order, form, zero_packing) {
        (4, "lsb_first", "tensor", Some("same_as_data")) => QuantScheme::AwqInt4,
        (4, "msb_first", "tensor", Some("same_as_data")) => QuantScheme::GptqInt4,
        (4, "lsb_first", "tensor", Some("plain")) => QuantScheme::MlxAffineU4,
        (4, "lsb_first", "implied", _) => QuantScheme::Int4B8,
        (8, _, "none", _) => QuantScheme::Int8Symmetric,
        (8, _, "tensor", _) => QuantScheme::Int8Asymmetric,
        _ => {
            return Err(Error::Checkpoint(format!(
                "zt.quant_group/1 with bits {bits}, packing order {order:?}, zero point \
                 {form:?}{} names no scheme this loader implements",
                zero_packing
                    .map(|p| format!(" packed {p:?}"))
                    .unwrap_or_default()
            )));
        }
    })
}

fn attr_map<'a>(object: &'a Object, key: &str) -> Option<&'a [(ztensor::cbor::Value, ztensor::cbor::Value)]> {
    object
        .attributes
        .as_ref()?
        .as_map()?
        .iter()
        .find(|(k, _)| k.as_text() == Some(key))
        .and_then(|(_, v)| v.as_map())
}

fn map_text<'a>(
    entries: &'a [(ztensor::cbor::Value, ztensor::cbor::Value)],
    key: &str,
) -> Option<&'a str> {
    entries
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

    /// The tensors one object lowers to, which is what every test here is
    /// about: the translation from a zTensor object to the loader's flat,
    /// name-addressed tensor space.
    fn lower(name: &str, object: Object) -> Result<Vec<RawTensor>, Error> {
        let mut tensors = Vec::new();
        collect(&mut tensors, name, &object, |_| Ok(FileId(0)))?;
        Ok(tensors)
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
        let tensors = lower("w", object).unwrap();
        assert_eq!(tensors.len(), 1);
        let tensor = &tensors[0];
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
        let tensors = lower("w", object).unwrap();
        let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["w", "w.scales"]);

        let payload = tensors.iter().find(|t| t.name == "w").unwrap();
        match &payload.encoding {
            Encoding::Quant(spec) => {
                assert_eq!(spec.scheme, QuantScheme::Mxfp4E2M1E8M0);
                assert_eq!(spec.group_size, 32);
            }
            other => panic!("expected a quantized payload, got {other:?}"),
        }
        // The scales part keeps its own dtype and its element count as shape.
        let scales = tensors.iter().find(|t| t.name == "w.scales").unwrap();
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
        let err = lower("w", object).unwrap_err();
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
        let err = lower("w", object).unwrap_err();
        assert!(format!("{err}").contains("compressed parts"));
    }
}
