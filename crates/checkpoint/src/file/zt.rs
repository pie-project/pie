//! Reading a checkpoint through zTensor.
//!
//! One reader for every format the loader accepts. `ztensor-compat` projects
//! `.safetensors`, `.gguf`, `.npz`, `.pt`, `.h5` and `.onnx` into one object
//! model — named tensors, each a shape and a set of parts, each part a byte
//! range in some file — and this module is the translation of that model into
//! the loader's [`Metadata`]. The two are close enough that most of
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

use crate::file::{Attribute, Attributes, File, Metadata, RawTensor, TokenizerTables};
use crate::error::Error;
use crate::types::{
    Axis, CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};

/// Opens a checkpoint of any supported format and describes it.
///
/// A `.zt` root that names shards brings them along; every other format is one
/// file that describes itself.
pub fn parse(path: &Path) -> Result<Metadata, Error> {
    describe(&ztensor_compat::index(path).map_err(Error::from)?)
}

/// Opens a set of files that together hold one checkpoint.
///
/// What a sharded snapshot is. Each file describes itself completely and none
/// of them names the others, so the set is the caller's claim — HF states it
/// in `model.safetensors.index.json`, which is a convention beside the format
/// rather than anything inside it. So this takes a list, and a name in two
/// files is refused rather than resolved by precedence.
pub fn parse_files(paths: &[PathBuf]) -> Result<Metadata, Error> {
    describe(&ztensor_compat::index_all(paths).map_err(Error::from)?)
}

/// The file-level key-values, for a caller asking what the checkpoint says
/// about itself. See `read::parse_attributes`.
pub fn parse_attributes(path: &Path) -> Result<Attributes, Error> {
    Ok(attributes_of(
        &ztensor_compat::index(path).map_err(Error::from)?,
    ))
}

/// [`parse_attributes`], for a checkpoint the caller claims is one set.
///
/// Sorts before reading, and the sort is load-bearing. `Source::merge` does
/// NOT merge the file-level key-values: it keeps the first source that has
/// any and drops the rest. That is the right rule for the case this exists to
/// serve -- only shard one of a split GGUF carries a key-value block -- but it
/// makes the answer a function of the order the caller passed.
///
/// Measured on `qwen2.5-7b-instruct-q4_0-*-of-00002`, before the sort:
///
/// | given | tensors | attribute keys |
/// |-------|---------|----------------|
/// | `[1, 2]` | 339 | 29, with `general.architecture` and the vocabulary |
/// | `[2, 1]` | 339 | 3: `split.no`, `split.count`, `split.tensors.count` |
///
/// Note the tensor count. Reversed, nothing is missing and nothing errors --
/// the caller gets a whole checkpoint that does not say what architecture it
/// is or carry a tokenizer, which downstream reads as a file that declined to
/// answer. A later shard has three keys rather than none, so it is not empty
/// and `merge` keeps it.
///
/// Every discovery path in this module already produces shard order --
/// `discover_gguf_files` sorts and `gguf_shard_set` builds `1..=count`,
/// safetensors comes out of a `BTreeSet` -- so today this sort changes
/// nothing. It is here so that the property lives where it is DEPENDED on
/// rather than only where it happens to be produced; a shard name's index is
/// fixed-width within one set, so lexicographic order is index order.
/// [`parse_tokenizer_tables`]'s caller already reads the set this way.
pub fn parse_attributes_files(paths: &[PathBuf]) -> Result<Attributes, Error> {
    let mut paths = paths.to_vec();
    paths.sort();
    Ok(attributes_of(
        &ztensor_compat::index_all(&paths).map_err(Error::from)?,
    ))
}

/// The `tokenizer.ggml.*` tables, whole.
///
/// Separate from [`parse_attributes`] because the cost is: this reads a
/// 150,000-entry vocabulary and its merge list, which the description of a
/// model has no use for. A caller pays for it by asking.
///
/// Empty tables rather than an error for a file that carries none — a
/// safetensors checkpoint has no `tokenizer.ggml.*` keys and is not wrong for
/// it, and a GGUF converted for its weights alone is a legitimate thing.
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
    // A non-text entry inside a table is dropped rather than defaulted. An
    // empty string at token 40,000 would shift nothing and read as a real
    // token; a short table trips the length check the loader already makes.
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

/// zTensor's CBOR attribute map, projected onto the loader's flat one.
///
/// Arrays and nested maps land on [`Attribute::Aggregate`]: the only ones any
/// real file carries are GGUF's tokenizer tables, and copying a 150,000-entry
/// vocabulary into a description of the model would cost every reader for a
/// reader that does not exist. A key that is present stays present, so
/// `get` returning `None` still means the file did not say.
fn attributes_of(source: &Source) -> Attributes {
    let Some(Value::Map(entries)) = source.attributes() else {
        return Attributes::default();
    };
    Attributes::from_pairs(entries.iter().filter_map(|(key, value)| {
        let Value::Text(key) = key else { return None };
        let value = match value {
            Value::Uint(v) => Attribute::Uint(*v),
            // CBOR spells a negative as `-1 - n`, which is the one shape a
            // cast cannot round-trip on its own.
            Value::Nint(v) => Attribute::Int(-1 - i64::try_from(*v).unwrap_or(i64::MAX)),
            Value::Float(v) => Attribute::Float(*v),
            Value::Bool(v) => Attribute::Bool(*v),
            Value::Text(v) => Attribute::Text(v.clone()),
            _ => Attribute::Aggregate,
        };
        Some((key.clone(), value))
    }))
}

/// Verifies every tensor digest of a `.zt` artifact; returns the tensor count.
///
/// The gate a destructive caller (`pie model import --delete-source`) runs
/// before destroying what the artifact was computed from. A part *without* a
/// digest fails rather than passes: "nothing was checked" cannot justify a
/// delete.
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
/// The read side of what [`Writer`](super::write::Writer) writes as
/// provenance. It is a function rather than a field on [`Metadata`] because
/// the two answer different questions and have different readers: metadata
/// says which tensors exist and where, which every planner needs, while
/// attributes say where the artifact *came from*, which only `pie model list`
/// and the re-convert skip check ask about. Charging every reader — including
/// the FFI marshaller and sixty-odd tests that build metadata by hand — for a
/// map they never read would be the wrong trade.
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
            "f8_e4m3fn" | "f8_e4m3fnuz" => DType::E4m3,
            "f8_e5m2" | "f8_e5m2fnuz" => DType::E5m2,
            "f8_e8m0" => DType::E8m0,
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

/// The encoding a part carries: its layout profile decides whether it is a
/// plain dtype or a quantized payload, and the object's attributes carry the
/// scheme's parameters.
///
/// Layouts the loader has no scheme for are an error, not a guess — reading a
/// quantized payload as raw bytes of its storage type is exactly the silent
/// misinterpretation the object model exists to prevent.
///
/// Public because it is the ONE translation from a part's stored `(dtype,
/// logical, layout)` facts to an [`Encoding`], and a model deciding whether the
/// bytes it was handed are the ones it wants asks here rather than growing a
/// second copy of the scheme table beside this one.
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
            logical_dtype: DType::Bf16,
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

/// Which point of the affine-group space an object names.
///
/// The parameters that separate the schemes this loader knows are the packing
/// order, the form of the zero point, the form of the SCALES, and — for the
/// rows this tree reads at one width only — the bit width. Anything the
/// combination does not name is refused rather than rounded to the nearest
/// scheme: reading GPTQ codes as AWQ would decode every weight backwards
/// within its word.
///
/// # The width is a point, and only two rows are entitled to name one
///
/// `zt.quant_group/1` is parametric, and the core spec is explicit about what
/// that costs a reader: *"a parametric profile MUST make every parameter its
/// decoder needs a required attribute; anything left unstated will be inferred
/// from something incidental, which is how a file that happens to use a
/// 32-element group comes to be read as the one scheme that used to have
/// 32-element groups."* This function was doing that with `bits`. The MLX row
/// sat at four because MLX was four when the row was written, so the two
/// widths that joined it later — eight for the MoE router gates, two for the
/// DQ expert banks — fell off the table: two came back a REFUSAL of a file
/// this tree had just written, and eight came back `Int8Asymmetric`, which is
/// the same tuple in every field a reader was looking at.
///
/// So the width is matched only where this tree genuinely reads one width:
/// `AwqInt4`, `GptqInt4` and `Int4B8` are four-bit rows here and
/// [`QuantSpec::term`](crate::term) says so in its own words, and a wider
/// declaration under those tuples is a scheme this crate cannot read back
/// rather than a row to invent. The MLX row spans its three, and the tie it
/// then makes with `Int8Asymmetric` at eight is broken by `scale_form`, which
/// is the field whose whole job is to say what a factors plane holds.
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
    // **THE FIFTH PARAMETER, AND IT IS THE ONE THAT WAS NOT BEING READ.** The
    // core spec names five things a `zt.quant_group/1` object states — bit
    // width, group size, packing order, scale form, zero-point form — and the
    // doc above this function names three of them plus this one. It was never
    // fetched. Two schemes in the table below write the SAME packing order and
    // the SAME zero point (`tensor`, packed `plain`), differ in nothing else a
    // reader can see, and are told apart here and nowhere else.
    let scale = map_scale_form(attrs).ok_or_else(|| missing("scale_form"))?;

    Ok(match (bits, order, form, zero_packing, scale) {
        (4, "lsb_first", "tensor", Some("same_as_data"), _) => QuantScheme::AwqInt4,
        (4, "msb_first", "tensor", Some("same_as_data"), _) => QuantScheme::GptqInt4,
        // **THREE WIDTHS, ONE SCHEME.** `MlxAffineU4` names the arithmetic and
        // `bits` says how wide a code is — the argument `Dtype::U8g64` and the
        // `U2g*` rows are each written on. Pinning this row at four was the
        // reader disagreeing with its own writer: `file/write.rs` emits this
        // exact tuple at two and at eight, so a 2-bit expert bank this tree
        // WROTE came back "names no scheme this loader implements", and an
        // 8-bit router gate came back as the row below.
        (2 | 4 | 8, "lsb_first", "tensor", Some("plain"), "f16_factors") => {
            QuantScheme::MlxAffineU4
        }
        (4, "lsb_first", "implied", _, _) => QuantScheme::Int4B8,
        (8, _, "none", _, _) => QuantScheme::Int8Symmetric,
        // The contested cell's other half. Eight bits, codes packed
        // `lsb_first`, a zero-point tensor packed `plain` — every field an
        // 8-bit MLX bank writes — and `f32_factors` where MLX writes
        // `f16_factors`. That one word is the whole difference, which is why
        // the row that matched on `(8, _, "tensor", _)` swallowed the MLX
        // bank silently and handed a kernel an int8 decoder for bf16 factors.
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

/// The `scale_form` attribute, which is a plain text key at the object's own
/// level rather than inside `packing` or `zero_point`.
fn map_scale_form(attrs: Option<&Value>) -> Option<&str> {
    attrs?.get("scale_form")?.as_text()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ztensor::format::cbor;
    use ztensor::{DType as ZDType, Writer};

    /// Writes a file and describes it, which is the whole path this module is:
    /// zTensor's object model in, the loader's flat tensor space out.
    fn lower(name: &str, write: impl FnOnce(&mut Writer)) -> Result<Vec<RawTensor>, Error> {
        let path = lower_file(name, write);
        let described = parse(&path);
        let _ = std::fs::remove_file(&path);
        described.map(|m| m.tensors)
    }

    /// [`lower`], stopping at the file, for a test that reads it another way.
    fn lower_file(name: &str, write: impl FnOnce(&mut Writer)) -> std::path::PathBuf {
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
        path
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
        assert_eq!(tensor.encoding, Encoding::Raw(DType::Bf16));
    }

    /// A checkpoint's key-values survive the trip, and silence stays silent.
    ///
    /// The projection is lossy on purpose — an array becomes `Aggregate` —
    /// so the property worth pinning is that a key which is present stays
    /// present. A caller reads `get(...).is_none()` as "the file did not
    /// say", and that reading is only true if nothing is dropped on the way.
    #[test]
    fn attributes_round_trip_and_an_array_keeps_only_its_key() {
        let path = std::env::temp_dir().join(format!("pie-zt-attrs-{}.zt", std::process::id()));
        let mut writer = Writer::options().create(&path).unwrap();
        writer.set_attributes(cbor::Value::Map(vec![
            (
                cbor::Value::Text("general.architecture".into()),
                cbor::Value::Text("qwen2".into()),
            ),
            (
                cbor::Value::Text("qwen2.block_count".into()),
                cbor::Value::Uint(24),
            ),
            (cbor::Value::Text("rope.scale".into()), cbor::Value::Nint(2)),
            (
                cbor::Value::Text("tokenizer.ggml.tokens".into()),
                cbor::Value::Array(vec![cbor::Value::Text("a".into())]),
            ),
        ]));
        writer.add("w", [2u64, 2], ZDType::BF16, &[0u8; 8]).unwrap();
        writer.finish().unwrap();
        let attributes = parse_attributes(&path).unwrap();

        assert_eq!(attributes.architecture(), Some("qwen2"));
        assert_eq!(
            attributes.get("qwen2.block_count"),
            Some(&Attribute::Uint(24))
        );
        // CBOR spells -3 as Nint(2).
        assert_eq!(attributes.get("rope.scale"), Some(&Attribute::Int(-3)));
        assert_eq!(
            attributes.get("tokenizer.ggml.tokens"),
            Some(&Attribute::Aggregate),
            "the vocabulary is not carried, but the file did say it has one"
        );
        assert_eq!(attributes.get("absent"), None);

        // A checkpoint that says nothing about itself is not an error.
        let bare = lower_file("bare-attrs", |w| {
            w.add("w", [2u64, 2], ZDType::BF16, &[0u8; 8]).unwrap();
        });
        assert!(parse_attributes(&bare).unwrap().is_empty());
        let _ = std::fs::remove_file(&bare);
        let _ = std::fs::remove_file(&path);
    }

    /// A set is read in shard order, whatever order it was handed in.
    ///
    /// `Source::merge` keeps the first source that HAS file-level key-values
    /// and drops every later one, so which file answers is decided by the
    /// caller's order. This reproduces the split-GGUF shape that makes that
    /// matter: one member carrying the real block, another carrying only the
    /// three `split.*` keys llama.cpp writes into shards two and up.
    ///
    /// Written with `.zt` files because the rule under test is `merge`'s and
    /// is format-blind; the names carry the fixed-width index a split GGUF
    /// uses, since that is what makes sorting the same thing as ordering.
    ///
    /// The tensor assertion is the point. Reversed and unsorted, this used to
    /// come back with BOTH tensors and three keys -- a complete checkpoint
    /// that cannot say what it is. Nothing is missing and nothing errors, so
    /// only asking for the architecture catches it.
    #[test]
    fn a_shard_set_is_read_in_shard_order_however_it_is_handed_over() {
        let first = lower_file("m-00001-of-00002", |w| {
            w.set_attributes(cbor::Value::Map(vec![
                (
                    cbor::Value::Text("general.architecture".into()),
                    cbor::Value::Text("llama".into()),
                ),
                (cbor::Value::Text("split.no".into()), cbor::Value::Uint(0)),
            ]));
            w.add("one", [2u64, 2], ZDType::BF16, &[0u8; 8]).unwrap();
        });
        let second = lower_file("m-00002-of-00002", |w| {
            w.set_attributes(cbor::Value::Map(vec![
                (cbor::Value::Text("split.no".into()), cbor::Value::Uint(1)),
                (
                    cbor::Value::Text("split.tensors.count".into()),
                    cbor::Value::Uint(1),
                ),
            ]));
            w.add("two", [2u64, 2], ZDType::BF16, &[0u8; 8]).unwrap();
        });

        for order in [
            vec![first.clone(), second.clone()],
            vec![second.clone(), first.clone()],
        ] {
            let whole = parse_files(&order).unwrap();
            assert_eq!(
                whole.tensors.len(),
                2,
                "both members are present either way, which is why this is silent"
            );
            let attributes = parse_attributes_files(&order).unwrap();
            assert_eq!(
                attributes.architecture(),
                Some("llama"),
                "the member carrying the key-value block has to be the one that answers"
            );
        }
        let _ = std::fs::remove_file(&first);
        let _ = std::fs::remove_file(&second);
    }

    /// Every format zTensor can report has a name here.
    ///
    /// The loader compiles in every `ztensor-compat` projection, so it reads
    /// all of them — and a file it read perfectly well used to be handed to
    /// the engine labelled `Unknown`, which reads as "could not identify" when
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

    /// One `zt.quant_group/1` object, built from parts, for the tests that
    /// ask which scheme a given point of the space names.
    fn affine(
        name: &'static str,
        bits: u64,
        order: &'static str,
        zero: cbor::Value,
        scale_form: &'static str,
    ) -> Result<Vec<RawTensor>, Error> {
        lower(name, move |w| {
            w.object("w", |o| {
                o.shape([32u64, 32])
                    .layout("zt.quant_group/1")
                    .attr("bits", bits)
                    .attr("packing", cbor::map([("order", order)]))
                    .attr("zero_point", zero)
                    .attr("scale_form", scale_form)
                    .attr("group_size", 128u64)
                    .part("data", |p| p.dtype(ZDType::U8).bytes(&[0u8; 512]))
            })
            .unwrap();
        })
    }

    fn scheme_read_back(tensors: &[RawTensor]) -> QuantScheme {
        match &tensors[0].encoding {
            Encoding::Quant(spec) => spec.scheme,
            other => panic!("expected a quantized payload, got {other:?}"),
        }
    }

    #[test]
    fn attributes_choose_the_quantization_scheme() {
        // `zt.quant_group/1` is parametric: the same layout id names different
        // schemes depending on the packing and zero-point attributes.
        let tensors = affine(
            "awq",
            4,
            "lsb_first",
            cbor::map([("form", "tensor"), ("packing", "same_as_data")]),
            "f16_factors",
        )
        .unwrap();
        assert_eq!(scheme_read_back(&tensors), QuantScheme::AwqInt4);
        match &tensors[0].encoding {
            Encoding::Quant(spec) => assert_eq!(spec.group_size, 128),
            other => panic!("expected a quantized payload, got {other:?}"),
        }
    }

    /// The one cell of the space two schemes both write, and the field that
    /// tells them apart.
    ///
    /// Eight bits, codes `lsb_first`, a zero-point TENSOR packed `plain`:
    /// that is every visible field of an 8-bit MLX router gate, and it is
    /// also every visible field of an `Int8Asymmetric` tensor. They differ in
    /// `scale_form` and in nothing else, so a reader that does not fetch it
    /// answers one of them with the other — which is what happened, in the
    /// direction that stays quiet: MLX's bf16 factors read back as a scheme
    /// whose factors are f32.
    #[test]
    fn the_scale_form_separates_the_two_schemes_that_share_a_cell() {
        let plain = || cbor::map([("form", "tensor"), ("packing", "plain")]);
        assert_eq!(
            scheme_read_back(&affine("mlx8", 8, "lsb_first", plain(), "f16_factors").unwrap()),
            QuantScheme::MlxAffineU4,
            "an 8-bit MLX bank is the row whose factors are bf16"
        );
        assert_eq!(
            scheme_read_back(&affine("i8a", 8, "lsb_first", plain(), "f32_factors").unwrap()),
            QuantScheme::Int8Asymmetric,
        );
    }

    /// MLX spans three widths, and the reader spans the same three.
    ///
    /// The 2-bit row is the one that was missing outright: a DQ expert bank
    /// this tree WROTE came back "names no scheme this loader implements".
    #[test]
    fn the_mlx_row_is_read_at_each_of_its_three_widths() {
        for bits in [2u64, 4, 8] {
            let tensors = affine(
                "mlx",
                bits,
                "lsb_first",
                cbor::map([("form", "tensor"), ("packing", "plain")]),
                "f16_factors",
            )
            .unwrap_or_else(|err| panic!("MLX at {bits} bits: {err}"));
            assert_eq!(scheme_read_back(&tensors), QuantScheme::MlxAffineU4);
        }
    }

    /// The fifth parameter is REQUIRED, and an object that omits it is
    /// refused rather than read at whichever row happens to match without it.
    ///
    /// The core spec's own words for why: *"a parametric profile MUST make
    /// every parameter its decoder needs a required attribute; anything left
    /// unstated will be inferred from something incidental."*
    #[test]
    fn an_affine_object_that_states_no_scale_form_is_refused() {
        let err = lower("no-scale-form", |w| {
            w.object("w", |o| {
                o.shape([32u64, 32])
                    .layout("zt.quant_group/1")
                    .attr("bits", 4u64)
                    .attr("packing", cbor::map([("order", "lsb_first")]))
                    .attr(
                        "zero_point",
                        cbor::map([("form", "tensor"), ("packing", "plain")]),
                    )
                    .part("data", |p| p.dtype(ZDType::U8).bytes(&[0u8; 512]))
            })
            .unwrap();
        })
        .unwrap_err();
        assert!(format!("{err}").contains("scale_form"), "{err}");
    }
}
