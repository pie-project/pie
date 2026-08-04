//! The materialized-weight artifact: a snapshot of finished device weights.
//!
//! Materialized weights are a deterministic function of the compile cache key,
//! yet they are otherwise rebuilt every boot — for an FP8→MXFP4 model that is a
//! large transcode. The driver snapshots them after the first load and reloads
//! them straight into device memory on the next one. This module owns what that
//! snapshot *is*; the driver owns when to take one and where to keep it.
//!
//! # Why the loader and not the driver
//!
//! The artifact used to be PIEWSTOR, a hand-rolled binary layout in the CUDA
//! driver: length-prefixed strings read straight into allocations, offsets
//! trusted as written, and a bespoke FNV variant for integrity. That is a file
//! format, and pie already has one — with a validating reader, digests over
//! every blob, and aligned placement it does not have to arrange by hand. The
//! artifact is now a `.zt` file, so the driver keeps the two things only it can
//! do: the device copies, and the policy around them.
//!
//! # What a snapshot has to record
//!
//! A weight store is a set of *owned* buffers plus *views* — tensors that point
//! into another tensor's bytes, which is how a fused QKV projection or a
//! stacked expert weight is addressed without copying. Only the owned buffers
//! have bytes, and they are the objects of the file. A view is not storage, so
//! it is not an object; it is a window, recorded in the file's attributes as
//! the root it lies in and the offset it starts at. Restoring a view as a copy
//! would double the memory the views exist to save.
//!
//! Two things cross this boundary as strings the loader stores and never
//! interprets: a tensor's runtime dtype and a quantization entry's scale kind.
//! Both are the *driver's* vocabulary — its dtype set includes tile-packed
//! types the loader has no name for, and inventing loader enums for them would
//! be a second copy to keep in step. What the loader does interpret is the
//! physical type the bytes are stored as, which is what makes the object
//! well-formed to any reader.

use std::path::Path;

use ztensor::format::cbor::Value;
use ztensor::{DType as ZDType, Sink, Source, Writer};

use crate::error::Error;
use crate::types::DType;

/// The pie-level schema above the container. An artifact written under a
/// different one is a miss, not an error: the driver recomputes.
pub const SCHEMA_VERSION: u64 = 1;

/// The attribute the whole record lives under.
const RECORD: &str = "pie.weight_store";
/// The per-object attribute carrying the driver's own name for its runtime type.
const RUNTIME_DTYPE: &str = "pie.dtype";

/// Blob placement. Larger than the 16 KiB the PIEWSTOR codec used, and for the
/// same reason: a blob that begins on a page can be handed to the device as a
/// mapping instead of a copy, and it has to be a page of *its own* or faulting
/// one weight drags in its neighbour. 64 KiB is the largest page pie targets
/// (ARM), and every smaller one divides it.
const ALIGN: u64 = 64 * 1024;

/// One tensor with bytes of its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    /// How the bytes are stored, in the loader's vocabulary. This is what the
    /// container records, and what makes `product(shape) × width` the blob's
    /// length.
    pub dtype: DType,
    /// The driver's own name for the type its kernels will read. Opaque here.
    pub runtime_dtype: String,
    pub shape: Vec<i64>,
}

impl Tensor {
    /// The byte length the payload must have.
    pub fn nbytes(&self) -> Result<u64, Error> {
        let elements = self.shape.iter().try_fold(1u64, |acc, &d| {
            let d = u64::try_from(d).map_err(|_| {
                Error::Checkpoint(format!("tensor {} has negative extent {d}", self.name))
            })?;
            acc.checked_mul(d).ok_or_else(|| {
                Error::Checkpoint(format!("tensor {}: shape product overflows", self.name))
            })
        })?;
        elements.checked_mul(self.dtype.bytes()).ok_or_else(|| {
            Error::Checkpoint(format!("tensor {}: byte length overflows", self.name))
        })
    }
}

/// A tensor that is a window into another tensor's bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    pub name: String,
    /// The object this window lies in. Empty when the view has no bytes at all,
    /// which a store is allowed to hold.
    pub root: String,
    pub byte_offset: u64,
    pub dtype: DType,
    pub runtime_dtype: String,
    pub shape: Vec<i64>,
    /// What the store itself records this view as cut from. Not always the
    /// root: a view of a view names the intermediate, and the store's own
    /// bookkeeping is about that name, not about who owns the allocation.
    pub backing: String,
}

/// A quantized weight's companion metadata: which tensors hold its scales, and
/// how the kernels apply them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quant {
    pub name: String,
    /// The driver's own name for the granularity its kernels expect. Opaque here.
    pub kind: String,
    pub scale: String,
    pub zero_point: String,
    pub group_size: i32,
    pub channel_axis: i32,
}

/// Writes an artifact, one tensor's bytes at a time.
///
/// Payloads arrive in chunks because they are copied off a device: a store is
/// tens of gigabytes and the host must never hold more than a chunk of it.
/// Views and quantization entries may be added at any point; they are metadata
/// and land in the manifest when the file is closed.
///
/// The file is written beside its destination and renamed into place, so a run
/// that dies mid-write leaves nothing that parses.
pub struct ArtifactWriter {
    /// `None` only after [`finish`](ArtifactWriter::finish) has taken it.
    writer: Option<Writer>,
    open: Option<Sink>,
    key: String,
    tensors: Vec<Tensor>,
    views: Vec<View>,
    quants: Vec<Quant>,
}

impl ArtifactWriter {
    /// Opens an artifact for `key` at `path`.
    ///
    /// Not canonical form: canonical placement requires objects in ascending
    /// name order, and the tensors arrive in whatever order the store iterates.
    /// A device snapshot has no reproducibility claim to trade for that — it is
    /// addressed by the cache key, never by its hash.
    pub fn create(path: &Path, key: &str) -> Result<Self, Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                Error::Checkpoint(format!("cannot create {}: {err}", parent.display()))
            })?;
        }
        // `publish` writes beside the destination and renames at the end, and
        // removes the partial if this writer is dropped without finishing.
        let writer = Writer::options()
            .canonical(false)
            .align(ALIGN)
            .publish(path)
            .map_err(Error::from)?;
        Ok(Self {
            writer: Some(writer),
            open: None,
            key: key.to_string(),
            tensors: Vec::new(),
            views: Vec::new(),
            quants: Vec::new(),
        })
    }

    /// Declares a tensor and opens it for writing. Its payload is exactly
    /// [`Tensor::nbytes`] bytes, delivered by [`write`](Self::write).
    pub fn begin_tensor(&mut self, tensor: Tensor) -> Result<(), Error> {
        if self.open.is_some() {
            return Err(Error::Checkpoint(format!(
                "tensor {} was begun while another is still open",
                tensor.name
            )));
        }
        let shape: Vec<u64> = tensor
            .shape
            .iter()
            .map(|&d| {
                u64::try_from(d).map_err(|_| {
                    Error::Checkpoint(format!("tensor {} has negative extent {d}", tensor.name))
                })
            })
            .collect::<Result<_, _>>()?;
        let (dtype, ltype) = storage_of(tensor.dtype);
        let attributes = Value::Map(vec![(
            Value::Text(RUNTIME_DTYPE.into()),
            Value::Text(tensor.runtime_dtype.clone()),
        )]);
        let nbytes = tensor.nbytes()?;
        let sink = self
            .writer()
            .stream(&tensor.name, |o| {
                o.shape(shape).attributes(attributes).part("data", |mut p| {
                    p = p.dtype(dtype);
                    if let Some(ltype) = ltype {
                        p = p.logical(ltype);
                    }
                    p.length(nbytes)
                })
            })
            .map_err(Error::from)?;
        self.open = Some(sink);
        self.tensors.push(tensor);
        Ok(())
    }

    /// Appends bytes to the open tensor.
    pub fn write(&mut self, chunk: &[u8]) -> Result<(), Error> {
        let sink = self
            .open
            .as_mut()
            .ok_or_else(|| Error::Checkpoint("no tensor is open".into()))?;
        let writer = self.writer.as_mut().expect("writer present");
        sink.write(writer, chunk).map_err(Error::from)
    }

    /// Closes the open tensor, which must have received its whole payload.
    pub fn end_tensor(&mut self) -> Result<(), Error> {
        let sink = self
            .open
            .take()
            .ok_or_else(|| Error::Checkpoint("no tensor is open".into()))?;
        let writer = self.writer.as_mut().expect("writer present");
        sink.close(writer).map_err(Error::from)
    }

    fn writer(&mut self) -> &mut Writer {
        self.writer.as_mut().expect("writer present")
    }

    pub fn add_view(&mut self, view: View) {
        self.views.push(view);
    }

    pub fn add_quant(&mut self, quant: Quant) {
        self.quants.push(quant);
    }

    /// Writes the record, closes the file and publishes it.
    pub fn finish(mut self) -> Result<(), Error> {
        if self.open.is_some() {
            return Err(Error::Checkpoint(
                "a tensor is still open; its payload is incomplete".into(),
            ));
        }
        let record = Value::Map(vec![
            (Value::Text("key".into()), Value::Text(self.key.clone())),
            (Value::Text("version".into()), Value::Uint(SCHEMA_VERSION)),
            (
                Value::Text("views".into()),
                Value::Array(self.views.iter().map(view_value).collect()),
            ),
            (
                Value::Text("quant".into()),
                Value::Array(self.quants.iter().map(quant_value).collect()),
            ),
        ]);
        let mut writer = self.writer.take().expect("writer present");
        writer.set_attributes(Value::Map(vec![(Value::Text(RECORD.into()), record)]));
        writer.finish().map_err(Error::from)?;
        Ok(())
    }
}

/// An opened artifact.
///
/// The file is mapped, so a tensor's bytes are a borrow of the mapping rather
/// than a copy: the driver streams them to the device straight from here.
pub struct Artifact {
    source: Source,
    tensors: Vec<Tensor>,
    views: Vec<View>,
    quants: Vec<Quant>,
}

impl Artifact {
    /// Opens the artifact at `path`, refusing one written for another key or
    /// under another schema.
    ///
    /// Both refusals are misses rather than errors in the caller's eyes — the
    /// distinction the caller needs is only whether it has to recompute.
    pub fn open(path: &Path, expected_key: &str) -> Result<Self, Error> {
        let source = Source::open(path).map_err(Error::from)?;
        let attributes = source
            .attributes()
            .ok_or_else(|| Error::Checkpoint("not a weight-store artifact".into()))?;
        let record = attributes
            .get(RECORD)
            .and_then(Value::as_map)
            .ok_or_else(|| Error::Checkpoint("not a weight-store artifact".into()))?;
        let record = Value::Map(record.to_vec());

        let version = record
            .get("version")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Checkpoint("artifact states no schema version".into()))?;
        if version != SCHEMA_VERSION {
            return Err(Error::Checkpoint(format!(
                "artifact schema {version}, expected {SCHEMA_VERSION}"
            )));
        }
        let key = record
            .get("key")
            .and_then(Value::as_text)
            .ok_or_else(|| Error::Checkpoint("artifact states no key".into()))?;
        if key != expected_key {
            return Err(Error::Checkpoint(format!(
                "artifact was written for {key:?}, not {expected_key:?}"
            )));
        }

        let mut tensors = Vec::new();
        for tensor in source.tensors() {
            let name = tensor.name();
            let part = tensor
                .data()
                .map_err(|_| Error::Checkpoint(format!("object {name:?} has no data part")))?;
            let runtime_dtype = tensor
                .attributes()
                .and_then(|a| a.get(RUNTIME_DTYPE))
                .and_then(Value::as_text)
                .ok_or_else(|| {
                    Error::Checkpoint(format!("object {name:?} names no runtime dtype"))
                })?;
            tensors.push(Tensor {
                name: name.to_string(),
                dtype: dtype_of(part.dtype(), part.logical()).ok_or_else(|| {
                    Error::Checkpoint(format!("object {name:?} has an unreadable element type"))
                })?,
                runtime_dtype: runtime_dtype.to_string(),
                shape: tensor
                    .shape()
                    .iter()
                    .map(|&d| {
                        i64::try_from(d).map_err(|_| {
                            Error::Checkpoint(format!("object {name:?} has an extent past i64"))
                        })
                    })
                    .collect::<Result<_, _>>()?,
            });
        }

        let views = record
            .get("views")
            .and_then(Value::as_array)
            .map(|entries| entries.iter().map(read_view).collect::<Result<Vec<_>, _>>())
            .transpose()?
            .unwrap_or_default();
        let quants = record
            .get("quant")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .map(read_quant)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();

        Ok(Self {
            source,
            tensors,
            views,
            quants,
        })
    }

    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    pub fn views(&self) -> &[View] {
        &self.views
    }

    pub fn quants(&self) -> &[Quant] {
        &self.quants
    }

    /// A tensor's bytes, borrowed from the mapping.
    pub fn bytes(&self, index: usize) -> Result<&[u8], Error> {
        self.source
            .tensor(&self.tensor(index)?.name)
            .and_then(|t| t.data()?.map())
            .map_err(Error::from)
    }

    /// Where a tensor's bytes begin in the file. What a caller checks when it
    /// wants to know the artifact is mappable, which is not a question the
    /// bytes themselves can answer.
    pub fn file_offset(&self, index: usize) -> Result<u64, Error> {
        let name = &self.tensor(index)?.name;
        self.source
            .tensor(name)
            .and_then(|t| t.data()?.locate())
            .map(|at| at.offset)
            .map_err(Error::from)
    }

    /// Whether a tensor's bytes still match the digest written with them.
    ///
    /// A mismatch is an answer rather than a failure: what the caller does with
    /// it is recompute, the same as for an artifact that was never there. A
    /// part carrying no digest at all is `false` too — this writer puts one on
    /// every part, so its absence is not something to vouch for either.
    pub fn verify(&self, index: usize) -> Result<bool, Error> {
        let name = &self.tensor(index)?.name;
        match self.source.tensor(name).and_then(|t| t.verify()) {
            Ok(verified) => Ok(verified.is_checked()),
            // A mismatch is a rejected file to zTensor and a cache miss here;
            // both mean the same thing to the caller, which is recompute.
            Err(err) if err.rule() == Some(ztensor::Rule::Digest) => Ok(false),
            Err(err) => Err(err.into()),
        }
    }

    fn tensor(&self, index: usize) -> Result<&Tensor, Error> {
        self.tensors
            .get(index)
            .ok_or_else(|| Error::Checkpoint(format!("no tensor at index {index}")))
    }
}

fn text(map: &Value, key: &str) -> Result<String, Error> {
    map.get(key)
        .and_then(Value::as_text)
        .map(str::to_string)
        .ok_or_else(|| Error::Checkpoint(format!("artifact record has no {key:?}")))
}

/// A signed integer as CBOR writes it: either major type 0 or major type 1.
fn int(map: &Value, key: &str) -> Result<i32, Error> {
    let bad = || Error::Checkpoint(format!("artifact record has no integer {key:?}"));
    match map.get(key).ok_or_else(bad)? {
        Value::Uint(n) => i32::try_from(*n).map_err(|_| bad()),
        Value::Nint(n) => i64::try_from(*n)
            .ok()
            .and_then(|n| i32::try_from(-1 - n).ok())
            .ok_or_else(bad),
        _ => Err(bad()),
    }
}

fn shape_of(map: &Value, key: &str) -> Result<Vec<i64>, Error> {
    map.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Checkpoint(format!("artifact record has no {key:?}")))?
        .iter()
        .map(|d| {
            d.as_u64()
                .and_then(|d| i64::try_from(d).ok())
                .ok_or_else(|| Error::Checkpoint(format!("{key:?} holds a non-extent")))
        })
        .collect()
}

fn value_of(shape: &[i64]) -> Value {
    Value::Array(
        shape
            .iter()
            .map(|&d| Value::Uint(d.max(0) as u64))
            .collect(),
    )
}

fn signed(v: i32) -> Value {
    if v < 0 {
        Value::Nint((-1 - i64::from(v)) as u64)
    } else {
        Value::Uint(v as u64)
    }
}

fn view_value(view: &View) -> Value {
    Value::Map(vec![
        (Value::Text("name".into()), Value::Text(view.name.clone())),
        (Value::Text("root".into()), Value::Text(view.root.clone())),
        (Value::Text("offset".into()), Value::Uint(view.byte_offset)),
        (
            Value::Text("dtype".into()),
            Value::Text(dtype_name(view.dtype).into()),
        ),
        (
            Value::Text("runtime_dtype".into()),
            Value::Text(view.runtime_dtype.clone()),
        ),
        (Value::Text("shape".into()), value_of(&view.shape)),
        (
            Value::Text("backing".into()),
            Value::Text(view.backing.clone()),
        ),
    ])
}

fn read_view(entry: &Value) -> Result<View, Error> {
    Ok(View {
        name: text(entry, "name")?,
        root: text(entry, "root")?,
        byte_offset: entry
            .get("offset")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Checkpoint("view has no offset".into()))?,
        dtype: dtype_named(&text(entry, "dtype")?)?,
        runtime_dtype: text(entry, "runtime_dtype")?,
        shape: shape_of(entry, "shape")?,
        backing: text(entry, "backing")?,
    })
}

fn quant_value(quant: &Quant) -> Value {
    Value::Map(vec![
        (Value::Text("name".into()), Value::Text(quant.name.clone())),
        (Value::Text("kind".into()), Value::Text(quant.kind.clone())),
        (
            Value::Text("scale".into()),
            Value::Text(quant.scale.clone()),
        ),
        (
            Value::Text("zero_point".into()),
            Value::Text(quant.zero_point.clone()),
        ),
        (Value::Text("group_size".into()), signed(quant.group_size)),
        (
            Value::Text("channel_axis".into()),
            signed(quant.channel_axis),
        ),
    ])
}

fn read_quant(entry: &Value) -> Result<Quant, Error> {
    Ok(Quant {
        name: text(entry, "name")?,
        kind: text(entry, "kind")?,
        scale: text(entry, "scale")?,
        zero_point: text(entry, "zero_point")?,
        group_size: int(entry, "group_size")?,
        channel_axis: int(entry, "channel_axis")?,
    })
}

/// How a dtype's elements are stored: a container type, and a logical type when
/// the container has no type of its own for them.
fn storage_of(dtype: DType) -> (ZDType, Option<&'static str>) {
    match dtype {
        DType::F32 => (ZDType::F32, None),
        DType::F16 => (ZDType::F16, None),
        DType::BF16 => (ZDType::BF16, None),
        DType::F8E4M3 => (ZDType::U8, Some("f8_e4m3fn")),
        DType::F8E5M2 => (ZDType::U8, Some("f8_e5m2")),
        DType::E8M0 => (ZDType::U8, Some("f8_e8m0")),
        DType::I64 => (ZDType::I64, None),
        DType::I32 => (ZDType::I32, None),
        DType::I16 => (ZDType::I16, None),
        DType::I8 => (ZDType::I8, None),
        DType::U64 => (ZDType::U64, None),
        DType::U32 => (ZDType::U32, None),
        DType::U16 => (ZDType::U16, None),
        DType::U8 => (ZDType::U8, None),
        DType::Bool => (ZDType::U8, Some("bool")),
    }
}

/// The inverse of [`storage_of`]. `None` for a type this artifact never writes.
fn dtype_of(dtype: ZDType, ltype: Option<&str>) -> Option<DType> {
    Some(match (dtype, ltype) {
        (ZDType::U8, Some("f8_e4m3fn")) => DType::F8E4M3,
        (ZDType::U8, Some("f8_e5m2")) => DType::F8E5M2,
        (ZDType::U8, Some("f8_e8m0")) => DType::E8M0,
        (ZDType::U8, Some("bool")) => DType::Bool,
        (_, Some(_)) => return None,
        (ZDType::F32, None) => DType::F32,
        (ZDType::F16, None) => DType::F16,
        (ZDType::BF16, None) => DType::BF16,
        (ZDType::I64, None) => DType::I64,
        (ZDType::I32, None) => DType::I32,
        (ZDType::I16, None) => DType::I16,
        (ZDType::I8, None) => DType::I8,
        (ZDType::U64, None) => DType::U64,
        (ZDType::U32, None) => DType::U32,
        (ZDType::U16, None) => DType::U16,
        (ZDType::U8, None) => DType::U8,
        (ZDType::F64, None) => return None,
    })
}

/// A view has no object to carry its element type, so the record spells it.
fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F8E4M3 => "f8e4m3",
        DType::F8E5M2 => "f8e5m2",
        DType::E8M0 => "e8m0",
        DType::I64 => "i64",
        DType::I32 => "i32",
        DType::I16 => "i16",
        DType::I8 => "i8",
        DType::U64 => "u64",
        DType::U32 => "u32",
        DType::U16 => "u16",
        DType::U8 => "u8",
        DType::Bool => "bool",
    }
}

fn dtype_named(name: &str) -> Result<DType, Error> {
    Ok(match name {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f8e4m3" => DType::F8E4M3,
        "f8e5m2" => DType::F8E5M2,
        "e8m0" => DType::E8M0,
        "i64" => DType::I64,
        "i32" => DType::I32,
        "i16" => DType::I16,
        "i8" => DType::I8,
        "u64" => DType::U64,
        "u32" => DType::U32,
        "u16" => DType::U16,
        "u8" => DType::U8,
        "bool" => DType::Bool,
        other => return Err(Error::Checkpoint(format!("unknown dtype {other:?}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("zt_wstore_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tensor(name: &str, dtype: DType, shape: Vec<i64>, runtime: &str) -> Tensor {
        Tensor {
            name: name.to_string(),
            dtype,
            runtime_dtype: runtime.to_string(),
            shape,
        }
    }

    /// The whole snapshot, out and back: bytes, element types, the driver's own
    /// type names, the windows into each root, and the quantization companions.
    #[test]
    fn a_snapshot_round_trips() {
        let dir = tmpdir("round");
        let path = dir.join("weights.zt");

        let a = vec![0xa1u8; 606];
        let b: Vec<u8> = (0..512u32).map(|i| (i % 251) as u8).collect();

        let mut writer = ArtifactWriter::create(&path, "key-1").unwrap();
        writer
            .begin_tensor(tensor("a", DType::BF16, vec![3, 101], "bf16"))
            .unwrap();
        // Uneven chunks: how the producer sliced its copies is not the file's
        // business.
        for chunk in a.chunks(97) {
            writer.write(chunk).unwrap();
        }
        writer.end_tensor().unwrap();

        writer
            .begin_tensor(tensor("arena", DType::U8, vec![512], "mxfp4-packed"))
            .unwrap();
        writer.write(&b).unwrap();
        writer.end_tensor().unwrap();

        writer.add_view(View {
            name: "expert.0".into(),
            root: "arena".into(),
            byte_offset: 256,
            dtype: DType::U8,
            runtime_dtype: "mxfp4-packed".into(),
            shape: vec![256],
            backing: "arena".into(),
        });
        writer.add_quant(Quant {
            name: "expert.0".into(),
            kind: "per_group".into(),
            scale: "expert.0.scale".into(),
            zero_point: String::new(),
            group_size: 32,
            channel_axis: -1,
        });
        writer.finish().unwrap();

        let artifact = Artifact::open(&path, "key-1").unwrap();
        assert_eq!(artifact.tensors().len(), 2);
        assert_eq!(
            artifact.tensors()[0],
            tensor("a", DType::BF16, vec![3, 101], "bf16")
        );
        assert_eq!(artifact.bytes(0).unwrap(), a);
        assert_eq!(artifact.bytes(1).unwrap(), b);
        assert!(artifact.verify(0).unwrap());
        assert!(artifact.verify(1).unwrap());

        assert_eq!(artifact.views().len(), 1);
        assert_eq!(artifact.views()[0].root, "arena");
        assert_eq!(artifact.views()[0].byte_offset, 256);
        assert_eq!(artifact.views()[0].runtime_dtype, "mxfp4-packed");
        assert_eq!(artifact.quants()[0].group_size, 32);
        assert_eq!(artifact.quants()[0].channel_axis, -1);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every payload begins on a page of its own. This is what lets a weight be
    /// mapped rather than copied, and nothing a caller can observe depends on
    /// it — which is the kind of arithmetic that rots quietly.
    #[test]
    fn every_payload_begins_on_a_page() {
        let dir = tmpdir("pages");
        let path = dir.join("weights.zt");
        let mut writer = ArtifactWriter::create(&path, "k").unwrap();
        // Sizes that are deliberately not multiples of the alignment: tensors
        // that happened to be page-sized would pass without any padding done.
        for (name, len) in [("a", 606u64), ("b", 182), ("c", 20)] {
            writer
                .begin_tensor(tensor(name, DType::U8, vec![len as i64], "u8"))
                .unwrap();
            writer.write(&vec![7u8; len as usize]).unwrap();
            writer.end_tensor().unwrap();
        }
        writer.finish().unwrap();

        let artifact = Artifact::open(&path, "k").unwrap();
        let mut last_end = 0u64;
        for index in 0..artifact.tensors().len() {
            let offset = artifact.file_offset(index).unwrap();
            assert_eq!(
                offset % 16384,
                0,
                "payload {index} does not begin on a page"
            );
            assert!(
                offset >= last_end,
                "payload {index} runs into its neighbour"
            );
            last_end = offset + artifact.bytes(index).unwrap().len() as u64;
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An artifact for another plan, or from another schema, is refused rather
    /// than half-read.
    #[test]
    fn a_foreign_artifact_is_refused() {
        let dir = tmpdir("foreign");
        let path = dir.join("weights.zt");
        let mut writer = ArtifactWriter::create(&path, "mine").unwrap();
        writer
            .begin_tensor(tensor("a", DType::U8, vec![8], "u8"))
            .unwrap();
        writer.write(&[0u8; 8]).unwrap();
        writer.end_tensor().unwrap();
        writer.finish().unwrap();

        assert!(Artifact::open(&path, "yours").is_err());
        assert!(Artifact::open(&path, "mine").is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Corruption is an error, not a wrong answer: the property the hand-rolled
    /// checksum was there for, now the container's.
    #[test]
    fn a_corrupt_payload_fails_its_digest() {
        let dir = tmpdir("corrupt");
        let path = dir.join("weights.zt");
        let mut writer = ArtifactWriter::create(&path, "k").unwrap();
        writer
            .begin_tensor(tensor("a", DType::U8, vec![4096], "u8"))
            .unwrap();
        writer.write(&vec![3u8; 4096]).unwrap();
        writer.end_tensor().unwrap();
        writer.finish().unwrap();

        let offset = Artifact::open(&path, "k").unwrap().file_offset(0).unwrap();
        let mut raw = std::fs::read(&path).unwrap();
        raw[offset as usize] ^= 0xff;
        std::fs::write(&path, &raw).unwrap();

        let artifact = Artifact::open(&path, "k").expect("the manifest is intact");
        assert!(!artifact.verify(0).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A payload short of what it declared is refused at the end of the file
    /// rather than published with a hole in it.
    #[test]
    fn a_short_payload_is_refused() {
        let dir = tmpdir("short");
        let path = dir.join("weights.zt");
        let mut writer = ArtifactWriter::create(&path, "k").unwrap();
        writer
            .begin_tensor(tensor("a", DType::U8, vec![64], "u8"))
            .unwrap();
        writer.write(&[0u8; 32]).unwrap();
        assert!(writer.end_tensor().is_err());
        assert!(writer.finish().is_err());
        assert!(!path.exists(), "an unfinished artifact was published");
        std::fs::remove_dir_all(&dir).ok();
    }
}
