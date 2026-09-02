//! Reads a checkpoint (via `ztensor-compat`) into the loader's [`Metadata`]; a multi-part tensor becomes one [`RawTensor`] per part, suffixed `.<part>` except `"data"`.

use std::path::{Path, PathBuf};

use ztensor::format::cbor::Value;
use ztensor::{DType as ZDType, Source};

use crate::file::{Attribute, Attributes, File, Metadata, RawTensor, TokenizerTables};
use crate::error::Error;
use crate::types::{
    Axis, CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};

// A `.zt` root that names shards brings them along; every other format is
// one file that describes itself.
pub fn parse(path: &Path) -> Result<Metadata, Error> {
    describe(&ztensor_compat::index(path).map_err(Error::from)?)
}

// A tensor name that appears in two files is refused, not resolved by precedence.
pub fn parse_files(paths: &[PathBuf]) -> Result<Metadata, Error> {
    describe(&ztensor_compat::index_all(paths).map_err(Error::from)?)
}

// The file-level key-values, for a caller asking what the checkpoint says about itself.
pub fn parse_attributes(path: &Path) -> Result<Attributes, Error> {
    Ok(attributes_of(
        &ztensor_compat::index(path).map_err(Error::from)?,
    ))
}

// Sorts before reading: merge keeps only the first source with file-level
// key-values, so the answer depends on shard order.
pub fn parse_attributes_files(paths: &[PathBuf]) -> Result<Attributes, Error> {
    let mut paths = paths.to_vec();
    paths.sort();
    Ok(attributes_of(
        &ztensor_compat::index_all(&paths).map_err(Error::from)?,
    ))
}

// Separate from `parse_attributes` since a full vocabulary/merge list is
// expensive. Returns empty tables for a checkpoint that carries none.
pub fn parse_tokenizer_tables(path: &Path) -> Result<TokenizerTables, Error> {
    Ok(tokenizer_tables_of(
        &ztensor_compat::index(path).map_err(Error::from)?,
    ))
}

fn tokenizer_tables_of(source: &Source) -> TokenizerTables {
    let Some(Value::Map(entries)) = source.attributes() else {
        return TokenizerTables::default();
    };
    let find = |name: &str| {
        entries
            .iter()
            .find(|(key, _)| matches!(key, Value::Text(text) if text == name))
            .map(|(_, value)| value)
    };
    let text = |name: &str| match find(name) {
        Some(Value::Text(value)) => Some(value.clone()),
        _ => None,
    };
    // A non-text entry is dropped, not defaulted to an empty string.
    let texts = |name: &str| match find(name) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::Text(value) => Some(value.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let ints = |name: &str| match find(name) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::Uint(value) => i64::try_from(*value).ok(),
                Value::Nint(value) => Some(-1 - i64::try_from(*value).unwrap_or(i64::MAX)),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    TokenizerTables {
        model: text("tokenizer.ggml.model").unwrap_or_default(),
        pre: text("tokenizer.ggml.pre"),
        tokens: texts("tokenizer.ggml.tokens"),
        token_types: ints("tokenizer.ggml.token_type"),
        merges: texts("tokenizer.ggml.merges"),
    }
}

// Arrays and nested maps become `Attribute::Aggregate`; a present key stays
// present either way, so `get` returning None still means the file didn't say.
fn attributes_of(source: &Source) -> Attributes {
    let Some(Value::Map(entries)) = source.attributes() else {
        return Attributes::default();
    };
    Attributes::from_pairs(entries.iter().filter_map(|(key, value)| {
        let Value::Text(key) = key else { return None };
        let value = match value {
            Value::Uint(v) => Attribute::Uint(*v),
            // CBOR encodes a negative as `-1 - n`.
            Value::Nint(v) => Attribute::Int(-1 - i64::try_from(*v).unwrap_or(i64::MAX)),
            Value::Float(v) => Attribute::Float(*v),
            Value::Bool(v) => Attribute::Bool(*v),
            Value::Text(v) => Attribute::Text(v.clone()),
            _ => Attribute::Aggregate,
        };
        Some((key.clone(), value))
    }))
}

// Verifies every tensor digest; a part without a digest fails rather than passes.
pub fn verify(path: &Path) -> Result<usize, Error> {
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

// Digest of every named tensor folded into one value; None if not a `.zt`
// with a manifest. Derived from the manifest, not the path, so it survives a move.
pub fn artifact_identity(path: &Path) -> Result<Option<Vec<u8>>, Error> {
    if !path.is_file()
        || !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zt"))
    {
        return Ok(None);
    }
    let Some(manifest) = ztensor::read::manifest_of(path).map_err(Error::from)? else {
        // A data shard carries no manifest; the root that names it does.
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
                // A part without a digest still must contribute something,
                // or two artifacts differing only there would collide.
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

// Flat text map of file-level attributes; entries whose value isn't text are skipped.
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

// Single- and multi-file cases share this path; `stores()` already resolves
// that difference, and each file's id is fixed by its order.
fn describe(source: &Source) -> Result<Metadata, Error> {
    let mut files = Vec::with_capacity(source.stores().len());
    for (index, store) in source.stores().iter().enumerate() {
        files.push(File {
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
            // The loader addresses bytes where they lie, so a part with no
            // address (compressed, chunked, deflated) cannot be planned.
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
    Ok(Metadata { files, tensors })
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
        // Reachable only if zTensor learns a format this build has no name for.
        _ => CheckpointFormat::Unknown,
    }
}

// A secondary part (e.g. scales) has its own extent, derived from its byte length.
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
    // A logical type, if present, takes precedence over the storage dtype.
    if let Some(ltype) = ltype {
        return Ok(match ltype {
            "f8_e4m3fn" | "f8_e4m3fnuz" => DType::E4m3,
            "f8_e5m2" | "f8_e5m2fnuz" => DType::E5m2,
            "f8_e8m0" => DType::E8m0,
            "bool" => DType::Bool,
            // MXFP4 payloads ride on U8 storage; the scheme names the packing.
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
        ZDType::BF16 => DType::Bf16,
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

// Layout decides plain dtype vs. quantized payload; a layout with no known
// scheme is an error, not a guess.
pub fn encoding_of(
    tensor: &ztensor::Tensor<'_>,
    part: &ztensor::Part<'_>,
) -> Result<Encoding, Error> {
    let layout = tensor.layout();
    let attrs = tensor.attributes();
    let dtype = dtype_of(part.dtype(), part.logical())?;

    // `dense` is the only layout whose parts are plain values.
    if layout == "dense" {
        return Ok(Encoding::Raw(dtype));
    }

    let scheme = scheme_of(layout, attrs)?;
    let name = tensor.name();
    // Unstated falls back to the scheme default; stated but too large to
    // represent is refused, not silently truncated.
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
            // Not stated by the checkpoint; every device path decodes to BF16.
            logical_dtype: DType::Bf16,
            bits_per_element: bits as u8,
            group_size: group_size as u32,
            channel_axis: axis,
        }
        .normalized(),
    ))
}

// `zt.quant_group/1` is parametric (scheme derived from packing/scale/zero
// form); `gguf.*` maps directly by profile. An unresolvable profile is refused.
fn scheme_of(layout: &str, attrs: Option<&Value>) -> Result<QuantScheme, Error> {
    if layout == "zt.quant_group/1" {
        return affine_group_scheme(attrs);
    }
    Ok(match layout {
        "zt.mx/1" => QuantScheme::Mxfp4E2M1E8M0,
        "gguf.q4_0/1" => QuantScheme::GgufQ4_0,
        "gguf.q4_1/1" => QuantScheme::GgufQ4_1,
        "gguf.q2_k/1" => QuantScheme::GgufQ2K,
        "gguf.q3_k/1" => QuantScheme::GgufQ3K,
        "gguf.q4_k/1" => QuantScheme::GgufQ4K,
        "gguf.q5_0/1" => QuantScheme::GgufQ5_0,
        "gguf.q5_1/1" => QuantScheme::GgufQ5_1,
        "gguf.q5_k/1" => QuantScheme::GgufQ5K,
        "gguf.q6_k/1" => QuantScheme::GgufQ6K,
        "gguf.iq4_nl/1" => QuantScheme::GgufIq4Nl,
        "gguf.iq4_xs/1" => QuantScheme::GgufIq4Xs,
        "gguf.q8_0/1" => QuantScheme::GgufQ8_0,
        "gguf.mxfp4/1" => QuantScheme::GgufMxfp4,
        "gguf.iq2_xxs/1" => QuantScheme::GgufIq2Xxs,
        "gguf.iq2_xs/1" => QuantScheme::GgufIq2Xs,
        "gguf.iq2_s/1" => QuantScheme::GgufIq2S,
        "gguf.iq3_xxs/1" => QuantScheme::GgufIq3Xxs,
        "gguf.iq3_s/1" => QuantScheme::GgufIq3S,
        other => {
            return Err(Error::Checkpoint(format!(
                "layout {other:?} has no loader quantization scheme; a plan cannot \
                 address bytes it cannot describe"
            )));
        }
    })
}

// Separated by packing order, zero-point form, scale form, and (for
// AwqInt4/GptqInt4/Int4B8) bit width. An unmatched combination is refused.
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
    // Required: two schemes below share packing order and zero point and
    // are told apart only by this field.
    let scale = map_scale_form(attrs).ok_or_else(|| missing("scale_form"))?;

    Ok(match (bits, order, form, zero_packing, scale) {
        (4, "lsb_first", "tensor", Some("same_as_data"), _) => QuantScheme::AwqInt4,
        (4, "msb_first", "tensor", Some("same_as_data"), _) => QuantScheme::GptqInt4,
        // MlxAffineU4 spans bit widths 2, 4, and 8.
        (2 | 4 | 8, "lsb_first", "tensor", Some("plain"), "f16_factors") => {
            QuantScheme::MlxAffineU4
        }
        (4, "lsb_first", "implied", _, _) => QuantScheme::Int4B8,
        (8, _, "none", _, _) => QuantScheme::Int8Symmetric,
        // Same fields as the 8-bit MLX row above except scale_form
        // (f32 factors vs f16 factors).
        (8, "lsb_first", "tensor", Some("plain"), "f32_factors") => {
            QuantScheme::Int8Asymmetric
        }
        _ => {
            return Err(Error::Checkpoint(format!(
                "zt.quant_group/1 with bits {bits}, packing order {order:?}, zero point \
                 {form:?}{}, scales {scale:?} names no scheme this loader implements",
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

// A plain text key at the object's own level, not inside `packing`/`zero_point`.
fn map_scale_form(attrs: Option<&Value>) -> Option<&str> {
    attrs?.get("scale_form")?.as_text()
}

