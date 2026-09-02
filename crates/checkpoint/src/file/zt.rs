//! Reads a checkpoint (via `ztensor-compat`) into the loader's [`Metadata`].
//! An object is one blob; under the canonical layout each of its planes
//! becomes one [`RawTensor`], named by [`plane_name`], so a quantized weight
//! reads as its codes, its `.scales` and its `.biases` exactly as a trace
//! declares them.

use std::path::{Path, PathBuf};

use ztensor::format::cbor::Value;
use ztensor::{Plane, Source, Tensor, Term};

use crate::error::Error;
use crate::file::{Attribute, Attributes, File, Metadata, RawTensor, TokenizerTables};
use crate::term::{
    dtype_of_leaf, gguf_scheme, gguf_type_of, plane_name, spec_of_canonical, MMA_TILED,
};
use crate::types::{Axis, CheckpointFormat, DType, Encoding, FileId, QuantSpec, TensorId};

// A `.zt` root that names shards brings them along; every other format is
// one file that describes itself.
pub fn parse(path: &Path) -> Result<Metadata, Error> {
    describe(&ztensor_compat::index(path).map_err(Error::from)?)
}

// A tensor name that appears in two files is refused, not resolved by precedence.
pub fn parse_files(paths: &[PathBuf]) -> Result<Metadata, Error> {
    describe(&ztensor_compat::index_all(paths).map_err(Error::from)?)
}

/// The objects [`parse`] splits into planes, as `(object, plane names in
/// canonical order)`. A leaf, a named layout and a gguf block array are one
/// plane and are not listed.
pub fn parse_groups(path: &Path) -> Result<Vec<(String, Vec<String>)>, Error> {
    let source = ztensor_compat::index(path).map_err(Error::from)?;
    let mut groups = Vec::new();
    for tensor in source.tensors() {
        let planes = planes_of(&tensor)?;
        if planes.len() > 1 {
            let names = planes.into_iter().map(|plane| plane.name).collect();
            groups.push((tensor.name().to_string(), names));
        }
    }
    Ok(groups)
}

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

fn attributes_of(source: &Source) -> Attributes {
    let Some(Value::Map(entries)) = source.attributes() else {
        return Attributes::default();
    };
    Attributes::from_pairs(entries.iter().filter_map(|(key, value)| {
        let Value::Text(key) = key else { return None };
        let value = match value {
            Value::Uint(v) => Attribute::Uint(*v),
            Value::Nint(v) => Attribute::Int(-1 - i64::try_from(*v).unwrap_or(i64::MAX)),
            Value::Float(v) => Attribute::Float(*v),
            Value::Bool(v) => Attribute::Bool(*v),
            Value::Text(v) => Attribute::Text(v.clone()),
            _ => Attribute::Aggregate,
        };
        Some((key.clone(), value))
    }))
}

// Verifies every object's digest; an object without one fails rather than passes.
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

// Digest of every named object folded into one value; None if not a `.zt`
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
        return Ok(None);
    };
    let mut identity = Vec::new();
    for (name, object) in &manifest.objects {
        let Some(digest) = &object.blob.digest else {
            return Err(Error::Checkpoint(format!(
                "{}: object {name:?} carries no digest, so the file has no identity",
                path.display()
            )));
        };
        identity.extend_from_slice(name.as_bytes());
        identity.extend_from_slice(digest.to_string().as_bytes());
    }
    for (name, shard) in &manifest.shards {
        identity.extend_from_slice(name.as_bytes());
        identity.extend_from_slice(shard.digest.to_string().as_bytes());
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
        let at = tensor.locate().map_err(|why| {
            Error::Unsupported(format!(
                "{}: this blob is encoded and not addressable ({why}); the loader \
                 addresses checkpoint bytes where they lie",
                tensor.name()
            ))
        })?;
        for plane in planes_of(&tensor)? {
            let id = TensorId(u32::try_from(tensors.len()).map_err(|_| {
                Error::Checkpoint("checkpoint has more tensors than a tensor id holds".into())
            })?);
            tensors.push(RawTensor {
                id,
                name: plane.name,
                file_id: FileId(at.store.0),
                file_offset: at.offset + plane.offset,
                span_bytes: plane.len,
                shape: plane.shape,
                encoding: plane.encoding,
            });
        }
    }
    Ok(Metadata { files, tensors })
}

/// One plane of an object as the loader reads it.
struct PlaneRead {
    name: String,
    offset: u64,
    len: u64,
    shape: Vec<i64>,
    encoding: Encoding,
}

/// The planes an object holds, in blob order. One for a leaf, a named
/// layout, or a gguf block array; one per plane for a group term.
fn planes_of(tensor: &Tensor<'_>) -> Result<Vec<PlaneRead>, Error> {
    let name = tensor.name();
    let signed = |shape: &[u64]| -> Result<Vec<i64>, Error> {
        shape
            .iter()
            .map(|&d| {
                i64::try_from(d)
                    .map_err(|_| Error::Checkpoint(format!("{name}: dimension {d} does not fit an i64")))
            })
            .collect()
    };
    match (tensor.layout(), tensor.term()) {
        (None | Some(MMA_TILED), Some(term)) => {
            let planes = term.planes(tensor.shape()).map_err(Error::from)?;
            canonical(name, term, &planes, signed)
        }
        (Some(layout), _) => {
            let Some(kind) = gguf_type_of(layout) else {
                return Err(Error::Unsupported(format!(
                    "{name}: layout {layout:?} is one this loader cannot address bytes under"
                )));
            };
            let scheme = gguf_scheme(kind).ok_or_else(|| {
                Error::Unsupported(format!("{name}: ggml type {kind:?} has no loader scheme"))
            })?;
            let row = ztensor::vocab::gguf::row_of(kind).ok_or_else(|| {
                Error::Internal(format!("gguf type {kind:?} has no registry row"))
            })?;
            let spec = QuantSpec {
                scheme,
                logical_dtype: DType::Bf16,
                bits_per_element: scheme.default_bits(),
                group_size: u32::try_from(row.elems_per_block).map_err(|_| {
                    Error::Internal(format!("gguf type {kind:?} has a block no u32 counts"))
                })?,
                channel_axis: None,
            };
            Ok(vec![PlaneRead {
                name: name.to_string(),
                offset: 0,
                len: tensor.nbytes(),
                shape: signed(tensor.shape())?,
                encoding: Encoding::Quant(spec),
            }])
        }
        (None, None) => Err(Error::Checkpoint(format!("{name}: no type and no layout"))),
    }
}

fn canonical(
    name: &str,
    term: &Term,
    planes: &[Plane],
    signed: impl Fn(&[u64]) -> Result<Vec<i64>, Error>,
) -> Result<Vec<PlaneRead>, Error> {
    let mut out = Vec::with_capacity(planes.len());
    for plane in planes {
        let encoding = match plane.path.as_str() {
            "code" => {
                let mut spec = spec_of_canonical(term).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "{name}: type `{term}` names no quantization this loader decodes \
                         out of separate planes"
                    ))
                })?;
                spec.channel_axis = plane
                    .shape
                    .len()
                    .checked_sub(1)
                    .and_then(|last| u8::try_from(last).ok())
                    .map(Axis);
                Encoding::Quant(spec)
            }
            _ => Encoding::Raw(dtype_of_leaf(plane.leaf).ok_or_else(|| {
                Error::Unsupported(format!(
                    "{name}: plane {:?} holds `{}`, which has no device representation",
                    plane.path, plane.leaf
                ))
            })?),
        };
        out.push(PlaneRead {
            name: plane_name(name, &plane.path),
            offset: plane.offset,
            len: plane.len,
            shape: signed(&plane.shape)?,
            encoding,
        });
    }
    Ok(out)
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
        _ => CheckpointFormat::Unknown,
    }
}

/// How the object's own bytes are encoded: the code plane's quantization
/// for a group term or a gguf block array, the leaf otherwise.
pub fn encoding_of(tensor: &Tensor<'_>) -> Result<Encoding, Error> {
    planes_of(tensor)?
        .into_iter()
        .next()
        .map(|plane| plane.encoding)
        .ok_or_else(|| Error::Checkpoint(format!("{}: an object with no planes", tensor.name())))
}

