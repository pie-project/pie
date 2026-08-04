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

use ztensor::format::cbor::Value;
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
    describe(&ztensor_compat::index(path).map_err(Error::from)?)
}

/// Opens a set of files that together hold one checkpoint.
///
/// What a sharded snapshot is. Each file describes itself completely and none
/// of them names the others, so the set is the caller's claim — HF states it
/// in `model.safetensors.index.json`, which is a convention beside the format
/// rather than anything inside it. So this takes a list, and a name in two
/// files is refused rather than resolved by precedence.
pub fn parse_checkpoint_files(paths: &[PathBuf]) -> Result<CheckpointMetadata, Error> {
    describe(&ztensor_compat::index_all(paths).map_err(Error::from)?)
}

/// Verifies every tensor digest of a `.zt` artifact; returns the tensor count.
///
/// The gate a destructive caller (`pie model convert --delete-source`) runs
/// before destroying what the artifact was computed from. A part *without* a
/// digest fails rather than passes: "nothing was checked" cannot justify a
/// delete.
pub fn verify_checkpoint(path: &Path) -> Result<usize, Error> {
    let source = Source::open(path).map_err(Error::from)?;
    let mut count = 0usize;
    for tensor in source.tensors() {
        if tensor.verify().map_err(Error::from)? == ztensor::Verified::NoDigest {
            return Err(Error::Checkpoint(format!(
                "tensor '{}' carries no digest to verify",
                tensor.name()
            )));
        }
        count += 1;
    }
    Ok(count)
}

/// A `.zt` artifact's identity: the digest of every tensor it names, folded
/// into one value. `None` for anything that is not a `.zt` with a manifest.
///
/// The manifest holds each object's own digest, and for a sharded artifact it
/// holds the shard table — whose entries are whole-file digests. So hashing
/// the manifest is a claim about the whole artifact, single-file or not, for
/// the cost of reading a header. It also survives the file being moved, which
/// an identity derived from a path does not.
///
/// This lives here rather than in the worker because "what identifies a
/// checkpoint" is a question about the format, and because the worker has no
/// business depending on zTensor directly.
pub fn artifact_identity(path: &Path) -> Result<Option<Vec<u8>>, Error> {
    if !path.is_file()
        || !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zt"))
    {
        return Ok(None);
    }
    let Some(manifest) = ztensor::read::manifest_of(path).map_err(Error::from)? else {
        // A data shard carries no manifest, so it identifies nothing on its
        // own — the root that names it does.
        return Ok(None);
    };
    let mut identity = Vec::new();
    for (name, object) in &manifest.objects {
        identity.extend_from_slice(name.as_bytes());
        for (part_name, part) in &object.parts {
            identity.extend_from_slice(part_name.as_bytes());
            if let Some(digest) = &part.digest {
                identity.extend_from_slice(digest.as_bytes());
            } else {
                // Canonical form gives every part a digest; a part without one
                // still has to contribute something, or two artifacts that
                // differ only there would collide.
                identity.extend_from_slice(&part.blob.length.to_le_bytes());
            }
        }
    }
    for (name, shard) in &manifest.shards {
        identity.extend_from_slice(name.as_bytes());
        identity.extend_from_slice(shard.digest.as_bytes());
        identity.extend_from_slice(&shard.size.to_le_bytes());
    }
    Ok(Some(identity))
}

/// Reads a checkpoint's file-level attributes as a flat text map.
///
/// The read side of what [`CheckpointWriter`](super::write::CheckpointWriter)
/// writes as provenance. It is a function rather than a field on
/// [`CheckpointMetadata`] because the two answer different questions and have
/// different readers: metadata says which tensors exist and where, which every
/// planner needs, while attributes say where the artifact *came from*, which
/// only `pie model list` and the re-convert skip check ask about. Charging
/// every reader — including the FFI marshaller and sixty-odd tests that build
/// metadata by hand — for a map they never read would be the wrong trade.
///
/// Attributes whose value is not text are skipped: the format allows arbitrary
/// CBOR there and the GGUF projection uses it for whole tokenizer tables, none
/// of which is provenance.
pub fn read_attributes(path: &Path) -> Result<std::collections::BTreeMap<String, String>, Error> {
    let source = Source::open(path).map_err(Error::from)?;
    let Some(Value::Map(entries)) = source.attributes() else {
        return Ok(std::collections::BTreeMap::new());
    };
    Ok(entries
        .iter()
        .filter_map(|(key, value)| match (key, value) {
            (Value::Text(key), Value::Text(value)) => Some((key.clone(), value.clone())),
            _ => None,
        })
        .collect())
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
            let part = tensor.part(part_name).map_err(Error::from)?;
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

fn checkpoint_format(label: &str) -> CheckpointFormat {
    match label {
        "zt" => CheckpointFormat::Zt,
        "safetensors" => CheckpointFormat::Safetensors,
        "gguf" => CheckpointFormat::Gguf,
        "npz" => CheckpointFormat::Npz,
        "pt" => CheckpointFormat::Pt,
        "hdf5" => CheckpointFormat::Hdf5,
        "onnx" => CheckpointFormat::Onnx,
        // Only reachable if zTensor learns a format this build has no name
        // for; `every_format_zt_can_report_has_a_name` fails when that day
        // comes rather than letting a file be labelled as nothing.
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
    let name = tensor.name();
    // An attribute the checkpoint did not state is the scheme's default, which
    // is a fact about the scheme. An attribute it *did* state that will not fit
    // the loader's representation is a different thing entirely, and falling
    // back to the default there would silently plan for a layout the file does
    // not have — `group_size` and `bits` decide byte spans, so the planner
    // would go on to address the wrong bytes and no one would be told.
    let fits = |value: Option<u64>, field: &str, max: u64| -> Result<Option<u64>, Error> {
        match value {
            Some(v) if v > max => Err(Error::Checkpoint(format!(
                "{name}: {field} is {v}, which the loader cannot represent (max {max})"
            ))),
            other => Ok(other),
        }
    };

    let group_size = fits(
        attr_u64(attrs, "group_size")
            .or_else(|| attr_u64(attrs, "block_size"))
            .or_else(|| attr_u64(attrs, "elems_per_block")),
        "group_size",
        u64::from(u32::MAX),
    )?
    .unwrap_or_else(|| u64::from(scheme.default_group_size()));
    let bits = fits(attr_u64(attrs, "bits"), "bits", u64::from(u8::MAX))?
        .unwrap_or_else(|| u64::from(scheme.default_bits()));
    let axis = fits(attr_u64(attrs, "axis"), "axis", u64::from(u8::MAX))?.map(|a| Axis(a as u8));

    Ok(Encoding::Quant(
        QuantSpec {
            scheme,
            // What the payload decodes to. The checkpoint does not say, and
            // every device path the loader targets decodes to BF16.
            logical_dtype: DType::BF16,
            bits_per_element: bits as u8,
            group_size: group_size as u32,
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

fn attr_map<'a>(attrs: Option<&'a Value>, key: &str) -> Option<&'a Value> {
    attrs?.get(key).filter(|v| v.as_map().is_some())
}

fn map_text<'a>(entries: &'a Value, key: &str) -> Option<&'a str> {
    entries.get(key)?.as_text()
}

fn attr_u64(attrs: Option<&Value>, key: &str) -> Option<u64> {
    attrs?.get(key)?.as_u64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ztensor::format::cbor;
    use ztensor::{DType as ZDType, Writer};

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

    /// Every format zTensor can report has a name here.
    ///
    /// The loader compiles in every `ztensor-compat` projection, so it reads
    /// all of them — and a file it read perfectly well used to be handed to
    /// the driver labelled `Unknown`, which reads as "could not identify" when
    /// the truth was "did not bother to say".
    ///
    /// `Unknown` stays for the case this cannot cover: a newer zTensor
    /// reporting a format this build predates. That is exactly when this test
    /// fails, which is the point of it.
    #[test]
    fn every_format_zt_can_report_has_a_name() {
        for label in ztensor_compat::FORMATS {
            assert_ne!(
                checkpoint_format(label),
                CheckpointFormat::Unknown,
                "zTensor reports {label:?}, which CheckpointFormat does not name"
            );
        }
    }

    /// A quant attribute the loader cannot represent is refused, not rounded.
    ///
    /// This is the module's own rule — a layout it has no scheme for is an
    /// error rather than a guess — applied to the layout's *parameters*.
    /// `group_size` and `bits` decide byte spans, so substituting a default
    /// for a value the checkpoint actually stated would have the planner
    /// address the wrong bytes while every test still passed.
    #[test]
    fn a_quant_attribute_that_does_not_fit_is_refused() {
        for (attr, value) in [("bits", 999u64), ("group_size", 1u64 << 33), ("axis", 300)] {
            let result = lower(attr, |w| {
                w.object("w", |o| {
                    o.shape([32u64, 32])
                        .layout("zt.mx/1")
                        .attr(attr, value)
                        .part("data", |p| {
                            p.dtype(ZDType::U8).logical("f4_e2m1").bytes(&[0u8; 512])
                        })
                        .part("scales", |p| {
                            p.dtype(ZDType::U8).logical("f8_e8m0").bytes(&[0u8; 32])
                        })
                })
                .unwrap();
            });
            let err = result.expect_err(&format!("{attr}={value} was accepted"));
            let text = err.to_string();
            assert!(
                text.contains(attr),
                "{attr}: message does not name it: {text}"
            );
            assert!(
                text.contains(&value.to_string()),
                "{attr}: message does not say what was stated: {text}"
            );
        }
    }

    #[test]
    fn secondary_parts_take_a_suffixed_name() {
        let tensors = lower("mx", |w| {
            w.object("w", |o| {
                o.shape([32u64, 32])
                    .layout("zt.mx/1")
                    .attr("block_size", 32u64)
                    .part("data", |p| {
                        p.dtype(ZDType::U8).logical("f4_e2m1").bytes(&[0u8; 512])
                    })
                    .part("scales", |p| {
                        p.dtype(ZDType::U8).logical("f8_e8m0").bytes(&[0u8; 32])
                    })
            })
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
            w.object("w", |o| {
                o.shape([32u64])
                    .layout("vendor.mystery/1")
                    .part("data", |p| p.dtype(ZDType::U8).bytes(&[0u8; 32]))
            })
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
            w.object("w", |o| {
                o.shape([32u64]).part("data", |p| {
                    p.dtype(ZDType::U8)
                        .encoding("zt.zstd-seekable/1")
                        .bytes(&[0u8; 32])
                })
            })
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
            w.object("w", |o| {
                o.shape([32u64, 32])
                    .layout("zt.quant_group/1")
                    .attr("bits", 4u64)
                    .attr("packing", cbor::map([("order", "lsb_first")]))
                    .attr(
                        "zero_point",
                        cbor::map([("form", "tensor"), ("packing", "same_as_data")]),
                    )
                    .attr("group_size", 128u64)
                    .part("data", |p| p.dtype(ZDType::U8).bytes(&[0u8; 512]))
            })
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
