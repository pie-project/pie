//! Reading a checkpoint through zTensor.
//!
//! One reader for every format the loader accepts. `ztensor-compat` projects
//! `.safetensors`, `.gguf`, `.npz`, `.pt`, `.h5` and `.onnx` into one object
//! model — named tensors, each a shape and a set of parts, each part a byte
//! range in some file — and this module is the translation of that model into
//! the loader's [`CheckpointMetadata`]. The two are close enough that most of
//! it is renaming: a tensor's part is a [`RawTensor`], its address is
//! `(file_id, file_offset, span_bytes)`.
//!
//! Nothing here reads tensor bytes, so nothing here maps a file: the source is
//! *indexed*, which answers where every tensor lives for the cost of a header
//! read rather than a mapping of the whole checkpoint.
//!
//! What is *not* renaming is the encoding. The loader's [`Encoding`] names a
//! quantization scheme; zTensor names a layout profile and a logical type. The
//! table in [`encoding_of`] is the whole of that correspondence, and it is the
//! part to read carefully — everything else here is plumbing.
//!
//! # Parts and names
//!
//! A zTensor tensor may carry several parts (a quantized weight is payload
//! plus scales). The loader's tensor space is flat and name-addressed, so a
//! multi-part tensor becomes one [`RawTensor`] per part: the `"data"` part
//! keeps the tensor's name, and any other part is suffixed `.<part>`. That
//! matches how the same tensors are named in the checkpoints these files come
//! from (`*_blocks` / `*_scales` in safetensors MXFP4), so a contract written
//! against a converted checkpoint reads the same either way.

use std::path::{Path, PathBuf};

use ztensor::cbor::Value;
use ztensor::{DType as ZDType, Source};

use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use crate::error::Error;
use crate::types::{
    Axis, CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};

/// Opens a checkpoint of any supported format and describes it.
///
/// A `.zt` root that names shards brings them along; every other format is one
/// file that describes itself.
pub fn parse_checkpoint(path: &Path) -> Result<CheckpointMetadata, Error> {
    describe(&ztensor_compat::index(path).map_err(zt_err)?)
}

/// Opens a set of files that together hold one checkpoint.
///
/// What a sharded snapshot is. Each file describes itself completely and none
/// of them names the others, so the set is the caller's claim — HF states it
/// in `model.safetensors.index.json`, which is a convention beside the format
/// rather than anything inside it. So this takes a list, and a name in two
/// files is refused rather than resolved by precedence.
pub fn parse_checkpoint_files(paths: &[PathBuf]) -> Result<CheckpointMetadata, Error> {
    describe(&ztensor_compat::index_all(paths).map_err(zt_err)?)
}

/// Turns an opened source into the loader's metadata.
///
/// The single-file and multi-file cases are one function because the source
/// already resolved that difference: `stores()` is the files this checkpoint
/// is made of, in the order that fixes their ids, and every tensor's address
/// names the file it lives in.
fn describe(source: &Source) -> Result<CheckpointMetadata, Error> {
    let mut files = Vec::with_capacity(source.stores().len());
    for (index, store) in source.stores().iter().enumerate() {
        files.push(CheckpointFile {
            id: FileId(u32::try_from(index).map_err(|_| {
                Error::Checkpoint("checkpoint has more files than a file id holds".into())
            })?),
            path: store.path().display().to_string(),
            size_bytes: store.len(),
            format: checkpoint_format(store.format()),
        });
    }

    let mut tensors = Vec::new();
    for tensor in source.tensors() {
        for part_name in tensor.parts() {
            let part = tensor.part(part_name).map_err(zt_err)?;
            let name = if part_name == "data" {
                tensor.name().to_string()
            } else {
                format!("{}.{part_name}", tensor.name())
            };
            // The loader addresses checkpoint bytes where they lie, so a part
            // with no address cannot be planned at all — a compressed `.zt`
            // part, a deflated zip entry, a chunked HDF5 dataset.
            // zTensor says *why* a part has no address — stored under an
            // encoding, produced by a foreign reader — and that is the half
            // that tells someone what to do about it, so it is kept.
            let at = part.locate().map_err(|why| {
                Error::Checkpoint(format!(
                    "{name}: this part has no address ({why}) — the loader addresses \
                     checkpoint bytes where they lie; convert the file to a raw \
                     one first"
                ))
            })?;
            let id = TensorId(u32::try_from(tensors.len()).map_err(|_| {
                Error::Checkpoint("checkpoint has more tensors than a tensor id holds".into())
            })?);
            tensors.push(RawTensor {
                id,
                name,
                file_id: FileId(at.store.0),
                file_offset: at.offset,
                span_bytes: at.len,
                shape: shape_of(&tensor, part_name, &part)?,
                encoding: encoding_of(&tensor, &part)?,
            });
        }
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

/// The shape a part presents to the planner.
///
/// The object's shape describes the *object*, and for a single-part object
/// that is also the part's shape. A secondary part (scales beside a payload)
/// has its own extent, which the loader needs as a shape of its own; it is
/// derived from the bytes, since the object shape does not describe it.
fn shape_of(
    tensor: &ztensor::Tensor<'_>,
    part_name: &str,
    part: &ztensor::Part<'_>,
) -> Result<Vec<i64>, Error> {
    if part_name == "data" {
        return tensor
            .shape()
            .iter()
            .map(|&d| {
                i64::try_from(d)
                    .map_err(|_| Error::Checkpoint(format!("dimension {d} does not fit an i64")))
            })
            .collect();
    }
    let width = part.dtype().width().max(1);
    let elements = part.nbytes() / width;
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
fn encoding_of(tensor: &ztensor::Tensor<'_>, part: &ztensor::Part<'_>) -> Result<Encoding, Error> {
    let layout = tensor.layout();
    let attrs = tensor.attributes();
    let dtype = dtype_of(part.dtype(), part.logical())?;

    // `dense` is the only layout whose parts are plain values.
    if layout == "dense" {
        return Ok(Encoding::Raw(dtype));
    }

    let scheme = scheme_of(layout, attrs)?;
    let group_size = attr_u64(attrs, "group_size")
        .or_else(|| attr_u64(attrs, "block_size"))
        .or_else(|| attr_u64(attrs, "elems_per_block"))
        .unwrap_or_else(|| u64::from(scheme.default_group_size()));
    let bits = attr_u64(attrs, "bits").unwrap_or_else(|| u64::from(scheme.default_bits()));
    let axis = attr_u64(attrs, "axis").and_then(|a| u8::try_from(a).ok()).map(Axis);

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
fn scheme_of(layout: &str, attrs: Option<&Value>) -> Result<QuantScheme, Error> {
    if layout == "zt.quant_group/1" {
        return affine_group_scheme(attrs);
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
fn affine_group_scheme(attrs: Option<&Value>) -> Result<QuantScheme, Error> {
    let missing = |what: &str| {
        Error::Checkpoint(format!(
            "zt.quant_group/1 is parametric and its {what} attribute is required; \
             a decoder cannot be chosen without it"
        ))
    };
    let bits = attr_u64(attrs, "bits").ok_or_else(|| missing("bits"))?;
    let packing = attr_map(attrs, "packing").ok_or_else(|| missing("packing"))?;
    let order = map_text(packing, "order").ok_or_else(|| missing("packing.order"))?;
    let zero = attr_map(attrs, "zero_point").ok_or_else(|| missing("zero_point"))?;
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

fn attr_map<'a>(attrs: Option<&'a Value>, key: &str) -> Option<&'a [(Value, Value)]> {
    attrs?
        .as_map()?
        .iter()
        .find(|(k, _)| k.as_text() == Some(key))
        .and_then(|(_, v)| v.as_map())
}

fn map_text<'a>(entries: &'a [(Value, Value)], key: &str) -> Option<&'a str> {
    entries
        .iter()
        .find(|(k, _)| k.as_text() == Some(key))
        .and_then(|(_, v)| v.as_text())
}

fn attr_u64(attrs: Option<&Value>, key: &str) -> Option<u64> {
    attrs?
        .as_map()?
        .iter()
        .find(|(k, _)| k.as_text() == Some(key))
        .and_then(|(_, v)| v.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ztensor::{cbor, DType as ZDType, Writer};

    /// Writes a file and describes it, which is the whole path this module is:
    /// zTensor's object model in, the loader's flat tensor space out.
    fn lower(name: &str, write: impl FnOnce(&mut Writer)) -> Result<Vec<RawTensor>, Error> {
        let path = std::env::temp_dir().join(format!(
            "pie-zt-{}-{name}-{}.zt",
            std::process::id(),
            name.len()
        ));
        let mut writer = Writer::options()
            .canonical(false)
            .align(65536)
            .create(&path)
            .unwrap();
        write(&mut writer);
        writer.finish().unwrap();
        let described = parse_checkpoint(&path);
        let _ = std::fs::remove_file(&path);
        described.map(|m| m.tensors)
    }

    #[test]
    fn a_dense_tensor_becomes_one_raw_tensor() {
        let tensors = lower("dense", |w| {
            w.add("w", [4u64, 4], ZDType::BF16, &[0u8; 32]).unwrap();
        })
        .unwrap();
        assert_eq!(tensors.len(), 1);
        let tensor = &tensors[0];
        assert_eq!(tensor.name, "w");
        assert_eq!(tensor.shape, vec![4, 4]);
        assert_eq!(tensor.span_bytes, 32);
        assert_eq!(tensor.file_offset % 65536, 0);
        assert_eq!(tensor.encoding, Encoding::Raw(DType::BF16));
    }

    #[test]
    fn secondary_parts_take_a_suffixed_name() {
        let tensors = lower("mx", |w| {
            w.object("w")
                .shape([32u64, 32])
                .layout("zt.mx/1")
                .attr("block_size", 32u64)
                .part("data")
                .dtype(ZDType::U8)
                .logical("f4_e2m1")
                .bytes(&[0u8; 512])
                .part("scales")
                .dtype(ZDType::U8)
                .logical("f8_e8m0")
                .bytes(&[0u8; 32])
                .add()
                .unwrap();
        })
        .unwrap();
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
    fn an_unknown_layout_is_refused() {
        let err = lower("mystery", |w| {
            w.object("w")
                .shape([32u64])
                .layout("vendor.mystery/1")
                .part("data")
                .dtype(ZDType::U8)
                .bytes(&[0u8; 32])
                .add()
                .unwrap();
        })
        .unwrap_err();
        assert!(format!("{err}").contains("no loader quantization scheme"));
    }

    #[test]
    fn a_part_with_no_address_is_refused() {
        // A compressed part is stored bytes, not tensor bytes: there is no
        // range of the file the planner could point a device at.
        let err = lower("zstd", |w| {
            w.object("w")
                .shape([32u64])
                .part("data")
                .dtype(ZDType::U8)
                .encoding("zt.zstd-seekable/1")
                .bytes(&[0u8; 32])
                .add()
                .unwrap();
        })
        .unwrap_err();
        assert!(format!("{err}").contains("no address"), "{err}");
    }

    #[test]
    fn attributes_choose_the_quantization_scheme() {
        // `zt.quant_group/1` is parametric: the same layout id names different
        // schemes depending on the packing and zero-point attributes.
        let tensors = lower("awq", |w| {
            w.object("w")
                .shape([32u64, 32])
                .layout("zt.quant_group/1")
                .attr("bits", 4u64)
                .attr(
                    "packing",
                    cbor::map([("order", "lsb_first")]),
                )
                .attr(
                    "zero_point",
                    cbor::map([("form", "tensor"), ("packing", "same_as_data")]),
                )
                .attr("group_size", 128u64)
                .part("data")
                .dtype(ZDType::U8)
                .bytes(&[0u8; 512])
                .add()
                .unwrap();
        })
        .unwrap();
        match &tensors[0].encoding {
            Encoding::Quant(spec) => {
                assert_eq!(spec.scheme, QuantScheme::AwqInt4);
                assert_eq!(spec.group_size, 128);
            }
            other => panic!("expected a quantized payload, got {other:?}"),
        }
    }
}
